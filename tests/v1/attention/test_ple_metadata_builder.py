# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE cache-group ownership and fixed-address mixed graph inputs."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen3_8_flash_next.ple_attn import (
    PLEAttentionBackend,
    PLEAttentionMetadataBuilder,
    PLEGraphInputs,
)
from vllm.models.qwen3_8_flash_next.ple_layer import Qwen3_8FlashNextPLELayer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec


def _builder(num_spec):
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=16, max_num_batched_tokens=256
    )
    config.scheduler_config.max_num_seqs = 4
    config.cache_config.mamba_cache_mode = "align"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    if num_spec:
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_spec
        )
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 12),),
        dtypes=(torch.bfloat16,),
        num_speculative_blocks=num_spec,
    )
    return PLEAttentionMetadataBuilder(
        spec, ["model.layers.0.ple"], config, torch.device("cpu")
    )


def test_ple_backend_does_not_change_short_conv_capability():
    assert Qwen3_8FlashNextPLELayer.get_attn_backend(None) is PLEAttentionBackend
    assert (
        PLEAttentionMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.ALWAYS
    )
    assert (
        ShortConvAttentionMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_ple_capture_buffers_follow_mixtures_and_cache_groups(num_spec):
    builders = [_builder(num_spec) for _ in range(3)]
    owned = [b.graph_inputs for b in builders]
    addresses = [
        {n: t.data_ptr() for n, t in vars(x).items() if isinstance(t, torch.Tensor)}
        for x in owned
    ]
    captured = None
    for lengths, queries in (
        ([80, 33, 1], [1, 33, 1]),
        ([80, 66, 17], [1 + num_spec, 1, 17]),
        ([90, 70, 30], [1, 1, 1]),
    ):
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=lengths, query_lens=queries),
            16,
            torch.device("cpu"),
            arange_block_indices=True,
        ).replace(
            is_prefilling=torch.tensor([q == s for q, s in zip(queries, lengths)])
        )
        width = common.block_table_tensor.shape[1] + num_spec
        common = common.replace(
            block_table_tensor=torch.arange(3 * width, dtype=torch.int32).view(3, width)
        )
        if captured is None:
            captured = [
                builder.build_for_cudagraph_capture(common) for builder in builders
            ]
        tables = [common.block_table_tensor + 100 * (group + 1) for group in range(3)]
        accepted = torch.tensor([2, 1, 1], dtype=torch.int32) if num_spec else None
        first = builders[0].build(
            0,
            common.replace(block_table_tensor=tables[0]),
            num_accepted_tokens=accepted,
        )
        original = owned[0].state_slot_ids.clone()
        results = [first]
        for builder, table in zip(builders[1:], tables[1:]):
            results.append(builder.update_block_table(first, table, None))
        torch.testing.assert_close(owned[0].state_slot_ids, original, rtol=0, atol=0)
        for group, (result, inputs) in enumerate(zip(results, owned)):
            assert result.graph_inputs is captured[group].graph_inputs is inputs
            assert addresses[group] == {
                n: t.data_ptr()
                for n, t in vars(inputs).items()
                if isinstance(t, torch.Tensor)
            }
            assert inputs.num_seqs.item() == 3
            assert inputs.num_tokens.item() == sum(queries)
            assert inputs.query_start_loc[:4].tolist() == [
                0,
                queries[0],
                sum(queries[:2]),
                sum(queries),
            ]
            assert inputs.state_slot_ids[3].item() == -1
            assert inputs.state_is_fresh[3].item()
            assert not inputs.request_is_prefill[3].item()
            assert inputs.num_accepted_tokens[3].item() == 1
            slots = inputs.state_slot_ids[:3]
            assert torch.all((slots >= (group + 1) * 100) & (slots < (group + 2) * 100))
            expected_modes = [False] * result.num_decodes + [True] * result.num_prefills
            assert inputs.request_is_prefill[:3].tolist() == expected_modes
            if num_spec:
                assert inputs.num_accepted_tokens[0].item() == 2


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_ple_layer_graph_staging_uses_runtime_contents(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = (
        torch.device(device_type, torch.accelerator.current_device_index())
        if device_type == "cuda"
        else torch.device("cpu")
    )
    inputs = PLEGraphInputs(4, 256, device)
    metadata = SimpleNamespace(
        graph_inputs=inputs,
        num_reqs=3,
        num_decodes=1,
        num_prefills=2,
        state_indices_tensor_d=torch.tensor([[3, 4, 5, 6]], device=device),
        state_indices_tensor_p=torch.tensor([7, 8], device=device),
        has_initial_states_p=torch.tensor([False, True], device=device),
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32, device=device),
    )
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    torch.nn.Module.__init__(layer)
    layer.max_seqs, layer.max_tokens = 4, 256
    for name, tensor in vars(inputs).items():
        if isinstance(tensor, torch.Tensor):
            setattr(layer, "_" + name, torch.empty_like(tensor))
    offsets = torch.tensor([0, 4, 20, 33], dtype=torch.int32, device=device)
    inputs.stage(metadata, offsets)

    def stage():
        layer._prepare_metadata(metadata, offsets, token_count=256)

    stage()
    graph = None
    if device_type == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage()
    addresses = {
        n: t.data_ptr() for n, t in vars(inputs).items() if isinstance(t, torch.Tensor)
    }
    for decode_count in (3, 0, 1):
        metadata.num_decodes = decode_count
        metadata.num_prefills = 3 - decode_count
        metadata.state_indices_tensor_d = torch.tensor([12, 15, 20], device=device)
        metadata.state_indices_tensor_p = torch.tensor([21, 22, 23], device=device)
        metadata.has_initial_states_p = torch.tensor(
            [True, False, False], device=device
        )
        metadata.num_accepted_tokens = torch.tensor([3, 2, 1], device=device)
        inputs.stage(metadata, offsets)
        allocated = torch.accelerator.memory_allocated(device) if graph else 0
        if graph:
            graph.replay()
            torch.accelerator.synchronize(device)
            assert torch.accelerator.memory_allocated(device) == allocated
        else:
            stage()
        for name, pointer in addresses.items():
            expected = getattr(inputs, name)
            assert expected.data_ptr() == pointer
            torch.testing.assert_close(
                getattr(layer, "_" + name), expected, rtol=0, atol=0
            )


@pytest.mark.parametrize("num_decodes", [0, 1, 4])
def test_ple_padding_translates_the_vllm_null_block(num_decodes):
    inputs = PLEGraphInputs(4, 256, torch.device("cpu"))
    slots = torch.tensor([7, NULL_BLOCK_ID, NULL_BLOCK_ID, NULL_BLOCK_ID])
    metadata = SimpleNamespace(
        num_reqs=4,
        num_decodes=num_decodes,
        num_prefills=4 - num_decodes,
        state_indices_tensor_d=slots[:num_decodes],
        state_indices_tensor_p=slots[num_decodes:],
        has_initial_states_p=torch.zeros(4 - num_decodes, dtype=torch.bool),
        num_accepted_tokens=torch.ones(num_decodes, dtype=torch.int32),
    )
    inputs.stage(metadata, torch.tensor([0, 4, 4, 4, 4], dtype=torch.int32))
    assert inputs.state_slot_ids.tolist() == [7, -1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_ple_padded_graph_initializes_poisoned_state_before_decode(monkeypatch):
    from b12x.sequence import ple
    from b12x.sequence.ple.reference import ple_projected_sequence_reference
    from triton.runtime import JITFunction

    device = torch.device("cuda", torch.accelerator.current_device_index())
    tokens, streams, hidden, history = 256, 2, 128, 9
    inputs = PLEGraphInputs(4, tokens, device)
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    torch.nn.Module.__init__(layer)
    layer.max_seqs, layer.max_tokens = 4, tokens
    for name, tensor in vars(inputs).items():
        if isinstance(tensor, torch.Tensor):
            setattr(layer, "_" + name, torch.empty_like(tensor))

    def random(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.2

    residual, key = random(tokens, streams, hidden), random(tokens, streams, hidden)
    value = random(tokens, hidden)
    weights = [random(streams * hidden) for _ in range(3)]
    conv_weight = random(streams * hidden, 4)
    state = random(4, streams * hidden, history + 3)
    plan = ple.plan(
        ple.Caps(
            device=device,
            mode="mixed",
            max_tokens=tokens,
            max_seqs=4,
            max_state_slots=4,
            max_speculative_tokens=3,
            streams=streams,
            hidden_size=hidden,
            kernel_size=4,
            dilation=3,
        )
    )
    scratch = torch.empty(
        plan.scratch_specs()[0].shape, dtype=torch.uint8, device=device
    )
    output = torch.empty_like(residual)
    binding = ple.bind(
        plan,
        scratch=scratch,
        residual=residual,
        key=key,
        value=value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        conv_state=state,
        out=output,
        **{
            name: getattr(layer, "_" + name)
            for name in (
                "query_start_loc",
                "state_slot_ids",
                "state_is_fresh",
                "num_accepted_tokens",
                "request_is_prefill",
                "num_seqs",
                "num_tokens",
            )
        },
    )
    metadata = SimpleNamespace(
        graph_inputs=inputs,
        num_reqs=4,
        num_decodes=1,
        num_prefills=3,
        state_indices_tensor_d=torch.tensor([1], device=device),
        state_indices_tensor_p=torch.tensor([2, 0, 0], device=device),
        has_initial_states_p=torch.zeros(3, dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([4], dtype=torch.int32, device=device),
    )
    offsets = torch.tensor([0, 4, 243, 243, 243], dtype=torch.int32, device=device)
    inputs.stage(metadata, offsets)

    def invoke(bound):
        layer._prepare_metadata(metadata, offsets, token_count=bound)
        ple.run_mixed(binding, eps=1e-6, token_count=bound)

    invoke(tokens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(tokens)

    def reject_compilation(*args, **kwargs):
        pytest.fail("PLE replay compiled a kernel")

    monkeypatch.setattr(JITFunction, "_do_compile", reject_compilation)
    state[0].fill_(float("nan"))
    state[2].fill_(float("nan"))
    prior = state.clone()
    expected, expected_state = ple_projected_sequence_reference(
        residual[4:243],
        key[4:243],
        value[4:243],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=3,
    )
    allocated = torch.accelerator.memory_allocated(device)
    addresses = tuple(t.data_ptr() for t in (output, scratch, state))
    scratch.fill_(0xFF)
    graph.replay()
    torch.accelerator.synchronize(device)
    assert torch.accelerator.memory_allocated(device) == allocated
    assert binding.error_code.item() == 0
    assert torch.isfinite(output[:243]).all()
    assert output[4:243].count_nonzero() > 0
    torch.testing.assert_close(output[4:243], expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        state[2, :, :history], expected_state, rtol=0.02, atol=0.0078125
    )
    assert state[2, :, history:].count_nonzero() == 0
    torch.testing.assert_close(state[0], prior[0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(state[3], prior[3], rtol=0, atol=0)

    metadata.num_decodes, metadata.num_prefills = 4, 0
    metadata.state_indices_tensor_d = torch.tensor([1, 2, 0, 0], device=device)
    metadata.num_accepted_tokens = torch.tensor(
        [4, 1, 1, 1], dtype=torch.int32, device=device
    )
    offsets.copy_(torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32, device=device))
    inputs.stage(metadata, offsets)
    expected, _ = ple_projected_sequence_reference(
        residual[4:8],
        key[4:8],
        value[4:8],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=3,
        prior_state=state[2, :, :history].clone(),
    )
    allocated = torch.accelerator.memory_allocated(device)
    scratch.fill_(0xFF)
    graph.replay()
    torch.accelerator.synchronize(device)
    assert binding.error_code.item() == 0
    assert torch.accelerator.memory_allocated(device) == allocated
    assert tuple(t.data_ptr() for t in (output, scratch, state)) == addresses
    assert torch.isfinite(output[:8]).all()
    torch.testing.assert_close(output[4:8], expected, rtol=0.02, atol=0.0078125)
