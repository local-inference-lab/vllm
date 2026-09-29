# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.parametrize("replicate_w1", [False, True])
def test_dspark_sharded_markov_weights_and_local_sampling(
    monkeypatch, default_vllm_config, replicate_w1
):
    from vllm.model_executor.layers import vocab_parallel_embedding as embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", str(int(replicate_w1)))
    monkeypatch.setattr(embedding, "get_tensor_model_parallel_rank", lambda: 11)
    monkeypatch.setattr(embedding, "get_tensor_model_parallel_world_size", lambda: 12)
    head = DSparkMarkovHead(190, 190, 8, prefix="markov_head")
    assert head.markov_w2.weight.shape == (16, 8)
    assert head.markov_w1.weight.shape == (190 if replicate_w1 else 16, 8)
    processor = LogitsProcessor(190)
    assert head.local_bias(torch.ones(2, 8), processor).shape == (2, 16)

    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.markov_head = head
    model.lm_head = embedding.ParallelLMHead(190, 8, prefix="lm_head")
    model.logits_processor = processor
    model._argmax_enabled = True
    model._argmax_capacity = 8
    model._argmax_plan = object()
    model._argmax_output = torch.empty(8, dtype=torch.int64)
    calls = []

    def sample(base, bias, out, *, plan):
        assert plan is model._argmax_plan
        calls.append((base.clone(), bias.clone()))
        return out.fill_(7)

    model._argmax_runtime = SimpleNamespace(fused_add_argmax=sample)
    assert model.supports_local_draft_argmax()
    base = torch.zeros(2, 16, dtype=torch.bfloat16)
    bias = torch.zeros_like(base)
    base[:, -2:] = 1000
    assert model.sample_local_draft_logits(base, bias).tolist() == [7, 7]
    assert torch.isneginf(calls[0][0][:, -2:]).all()
    assert torch.isneginf(calls[0][1][:, -2:]).all()

    processor.scale = 0.5
    assert not model.supports_local_draft_argmax()
    processor.scale = 1
    head.markov_w2.num_embeddings_padded += 64
    assert not model.supports_local_draft_argmax()


def test_dspark_sharded_head_rejects_gathered_topk(monkeypatch, default_vllm_config):
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    with pytest.raises(ValueError, match="top-k requires replicated"):
        DSparkMarkovHead(128, 128, 8, "head", retain_weight_for_gather=True)


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_v41_dspark_retains_each_markov_embedding_during_graph_replay(
    default_vllm_config,
    monkeypatch,
    id_dtype,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x Markov embedding requires SM12x")
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "0")
    from b12x._lib.runtime_control import kernel_resolution_guard

    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkMarkovHead as V41DSparkMarkovHead,
    )
    from vllm.v1.worker import workspace

    device = torch.device("cuda")
    vocab, rank = 131, 128
    with torch.device(device), torch.no_grad():
        head = V41DSparkMarkovHead(vocab, vocab, rank, prefix="markov_head").bfloat16()
        checkpoint = (
            torch.arange(vocab, device=device)[:, None] / 16
            + torch.arange(rank, device=device)[None, :] / 128
        ).bfloat16()
        default_weight_loader(head.markov_w1.weight, checkpoint)
        token_ids = [
            torch.tensor(values, device=device, dtype=id_dtype)
            for values in ([0, 1, 130], [64, 95, 96], [127, 17, 33])
        ]
        eager = [head.embed(ids) for ids in token_ids]
        # A later Markov step must not overwrite an earlier retained result.
        head.embed(token_ids[-1].roll(1))
        for rows, ids in zip(eager, token_ids):
            torch.testing.assert_close(rows, checkpoint[ids.long()], rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard("V4.1 retained DSpark Markov rows"):
            with (
                workspace.collect_cuda_graph_capture_resources() as retained,
                torch.cuda.graph(graph),
            ):
                captured = [head.embed(ids) for ids in token_ids]
            for offset in (1, 19):
                for step, ids in enumerate(token_ids):
                    ids.add_(offset + step).remainder_(vocab)
                graph.replay()
                torch.accelerator.synchronize()
                for rows, ids in zip(captured, token_ids):
                    torch.testing.assert_close(
                        rows,
                        checkpoint[ids.long()],
                        rtol=0,
                        atol=0,
                    )
            graph.reset()
            del retained


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(K3DSparkModel, "decoder_layer_cls", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("tp_size", "expect_sharded"),
    [(1, False), (8, True), (12, False), (16, True)],
)
def test_k3_dspark_context_projection_uses_divisible_tp_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    expect_sharded: bool,
) -> None:
    context_projection_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    def make_sharded_projection(*args, **kwargs):
        context_projection_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", make_sharded_projection)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(K3DSparkModel, "decoder_layer_cls", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", DummyModule)

    config = SimpleNamespace(
        target_hidden_size=7168,
        num_target_layers=5,
        hidden_size=7168,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=163840,
        draft_vocab_size=163840,
        markov_rank=256,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    model = K3DSparkModel(
        vllm_config=vllm_config,
        start_layer_id=0,
        prefix="model",
    )

    assert model.context_proj_sharded is expect_sharded
    assert model._streamed_context_states.dtype == model.context_proj.weight.dtype
    assert model._streamed_context_states.device == model.context_proj.weight.device
    assert bool(context_projection_calls) is expect_sharded
    if expect_sharded:
        args, kwargs = context_projection_calls[0]
        assert args == (7168 * 5, 7168)
        assert kwargs["gather_output"] is True
        assert kwargs["quant_config"] is None


@pytest.mark.cpu_test
def test_k3_dspark_streams_auxiliary_projection_into_tp_local_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        target_hidden_size=4,
        hidden_size=2,
        target_layer_ids=(0, 1),
    )
    model.context_proj_sharded = True
    model.context_proj = nn.Linear(8, 2, bias=False, dtype=torch.bfloat16)
    model.context_norm = SimpleNamespace(
        weight=torch.tensor([0.75, 1.25], dtype=torch.bfloat16),
        variance_epsilon=1e-5,
    )
    model.context_kv_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model._max_num_context_tokens = 1024
    model._streamed_aux_layer_ids = (1, 2)
    model._streamed_aux_scratch = None
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    model._context_local_width = 2
    model._context_local_start = 0
    model.register_buffer(
        "_streamed_context_states",
        torch.empty(1024, 2, dtype=torch.bfloat16),
        persistent=False,
    )
    scratch = torch.empty(1024, 4, dtype=torch.bfloat16)
    model.bind_auxiliary_stream_scratch(scratch)
    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        lambda tensor: tensor,
    )

    first = torch.randn(1024, 4, dtype=torch.bfloat16)
    second = torch.randn(1024, 4, dtype=torch.bfloat16)
    residual = torch.randn(1024, 4, dtype=torch.bfloat16)
    combined = torch.cat((first, second + residual), dim=-1)
    projected = torch.nn.functional.linear(combined, model.context_proj.weight)
    expected = projected * torch.rsqrt(
        projected.float().square().mean(dim=-1, keepdim=True) + 1e-5
    ).to(projected.dtype)
    expected *= model.context_norm.weight

    assert model.can_stream_auxiliary_states((1, 2), first)
    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, residual)
        output = model.finish_auxiliary_stream()

    # Streaming replaces one BF16 GEMM with an ordered sum of BF16 GEMMs, so
    # cancellation can move individual low-magnitude elements by several
    # ulps. Preserve the projected vector rather than requiring elementwise
    # equality between the two legal BF16 accumulation orders.
    mean_absolute_error = (output.float() - expected.float()).abs().mean()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.float().flatten(), dim=0
    )
    assert mean_absolute_error.item() < 2e-3
    assert cosine_similarity.item() > 0.9999
    assert model.is_streamed_context_states([output])
    assert not model.is_streamed_context_states([output])
    assert output.data_ptr() == model._streamed_context_states.data_ptr()

    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, residual)
        next_output = model.finish_auxiliary_stream()

    assert not model.is_streamed_context_states([output])
    assert model.is_streamed_context_states([next_output])


@pytest.mark.cpu_test
def test_k3_dspark_streamed_auxiliary_normalization_uses_global_hidden_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        target_hidden_size=4,
        hidden_size=4,
        target_layer_ids=(0, 1),
    )
    model.context_proj_sharded = True
    model.context_proj = nn.Linear(8, 2, bias=False, dtype=torch.bfloat16)
    model.context_norm = SimpleNamespace(
        weight=torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16),
        variance_epsilon=1e-5,
    )
    model.context_kv_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model._max_num_context_tokens = 1024
    model._streamed_aux_layer_ids = (1, 2)
    model._streamed_aux_scratch = None
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    model._context_local_width = 2
    model._context_local_start = 2
    model.register_buffer(
        "_streamed_context_states",
        torch.empty(1024, 2, dtype=torch.bfloat16),
        persistent=False,
    )
    model.bind_auxiliary_stream_scratch(torch.empty(1024, 4, dtype=torch.bfloat16))

    first = torch.randn(1024, 4, dtype=torch.bfloat16)
    second = torch.randn(1024, 4, dtype=torch.bfloat16)
    combined = torch.cat((first, second), dim=-1)
    local_projection = torch.nn.functional.linear(combined, model.context_proj.weight)
    other_rank_projection = torch.randn_like(local_projection)

    def add_other_rank_squared_norm(tensor: torch.Tensor) -> torch.Tensor:
        tensor.add_(other_rank_projection.float().square().sum(dim=-1, keepdim=True))
        return tensor

    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        add_other_rank_squared_norm,
    )
    expected_scale = torch.rsqrt(
        (
            local_projection.float().square().sum(dim=-1, keepdim=True)
            + other_rank_projection.float().square().sum(dim=-1, keepdim=True)
        )
        / model.config.hidden_size
        + model.context_norm.variance_epsilon
    ).to(local_projection.dtype)
    expected = local_projection * expected_scale * model.context_norm.weight[2:4]

    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, None)
        output = model.finish_auxiliary_stream()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cpu_test
def test_k3_dspark_disables_streaming_for_incompatible_scratch_width() -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(target_hidden_size=8)
    model._max_num_context_tokens = 16
    model._streamed_aux_scratch = torch.empty(16, 8)

    assert not model.bind_auxiliary_stream_scratch(torch.empty(16, 4))
    assert model._streamed_aux_scratch is None


@pytest.mark.cpu_test
def test_k3_dspark_preserves_fallback_when_target_has_no_stream_hook() -> None:
    wrapper = object.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(bind_auxiliary_stream_scratch=Mock())
    target_model = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(model=object())
    )

    wrapper.bind_target_auxiliary_stream(
        target_model,
        torch.empty(16, 8),
    )

    wrapper.model.bind_auxiliary_stream_scratch.assert_not_called()


@pytest.mark.cpu_test
def test_k3_dspark_projects_streamed_context_one_layer_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model._context_local_start = 2
    model._context_local_width = 2
    model._context_kv_width = 3
    model._context_kv_lora_rank = 2
    model._context_rope_dim = 1
    model._context_rms_norm_eps = 1e-5
    model._context_kv_norm_weights = torch.ones(2, 2)
    model.context_kv_proj = nn.Linear(4, 6, bias=False)
    model.context_kv_proj.weight.data.copy_(torch.arange(24).view(6, 4))
    cache_updates = []

    class Attention:
        def __init__(self):
            self.rotary_emb = SimpleNamespace(head_size=1, is_neox_style=True)
            self.impl = SimpleNamespace(
                do_kv_cache_update=lambda *args: cache_updates.append(args)
            )
            self.kv_cache = object()
            self.kv_cache_dtype = "bf16"
            self._k_scale = 1.0

    model.layers = [
        SimpleNamespace(self_attn=Attention()),
        SimpleNamespace(self_attn=Attention()),
    ]
    model._get_rope_inputs = lambda positions: (positions, torch.empty(0))
    reduced = []

    def record_all_reduce(tensor):
        reduced.append(tensor.clone())
        return tensor

    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        record_all_reduce,
    )
    rms_norm_io = []

    def record_rms_norm(output, input_, weight, eps):
        rms_norm_io.append((output, input_))
        output.copy_(input_)

    monkeypatch.setattr(dspark_mla.ops, "rms_norm", record_rms_norm)
    monkeypatch.setattr(dspark_mla.ops, "rotary_embedding", lambda *args: None)
    context = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([5, 6])
    first_slots = torch.tensor([0, 1])

    model._precompute_streamed_context_kv(
        context,
        positions,
        [first_slots, None],
    )

    expected = []
    for layer_index in range(2):
        weight = model.context_kv_proj.weight[
            layer_index * 3 : (layer_index + 1) * 3, 2:4
        ]
        expected.append(torch.nn.functional.linear(context, weight))
    torch.testing.assert_close(reduced[0], expected[0])
    torch.testing.assert_close(reduced[1], expected[1])
    assert all(output.data_ptr() != input_.data_ptr() for output, input_ in rms_norm_io)
    assert len(cache_updates) == 1
    torch.testing.assert_close(cache_updates[0][0], expected[0][:, :2])
    torch.testing.assert_close(cache_updates[0][1], expected[0][:, 2:].view(2, 1, 1))
    assert cache_updates[0][3] is first_slots


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.float8_e8m0fnu, torch.float32]
)
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_module", "shard_id"),
    [
        ("mtp.0.attn.wq_a.scale", "model.layers.0.attn.fused_wqa_wkv", 0),
        ("mtp.0.attn.wkv.scale", "model.layers.0.attn.fused_wqa_wkv", 1),
        ("mtp.0.main_proj.scale", "model.main_proj", None),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            "model.layers.0.ffn.shared_experts.gate_up_proj",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            "model.layers.0.ffn.shared_experts.down_proj",
            None,
        ),
    ],
)
def test_v41_dspark_loads_linear_scales(
    monkeypatch, scale_dtype, checkpoint_name, runtime_module, shard_id
):
    """Checkpoint ``.scale`` maps to the quant method's scale parameter and
    loads untouched. MXFP8 block-scale expansion lives in the KMxfp8Static
    loader (see tests/quantization/test_modelopt.py), not in load_weights."""
    from vllm.models.deepseek_v41.nvidia import dspark

    mxfp8 = scale_dtype != torch.float32
    scale_name = "weight_scale" if mxfp8 else "weight_scale_inv"
    runtime_name = f"{runtime_module}.{scale_name}"
    raw = torch.tensor([[120, 127], [128, 130]], dtype=torch.uint8)
    checkpoint_scale = raw.view(scale_dtype) if mxfp8 else raw.float()
    param = nn.Parameter(torch.empty_like(checkpoint_scale), requires_grad=False)
    shards = []

    def load_scale(param, weight, *args):
        shards.append(args)
        assert weight.dtype == checkpoint_scale.dtype
        param.copy_(weight)

    param.weight_loader = load_scale
    draft = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, n_routed_experts=1),
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32] if mxfp8 else [128, 128]
        ),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))],
            confidence_head=None,
        ),
        named_parameters=lambda: [(runtime_name, param)],
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )

    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(
        draft, [(checkpoint_name, checkpoint_scale)]
    )

    assert loaded == {runtime_name}
    assert shards == [() if shard_id is None else (shard_id,)]
    torch.testing.assert_close(param, checkpoint_scale)


@pytest.mark.parametrize("layer_groups", [None, [0, 1, 0]])
@torch.inference_mode()
def test_v41_context_graph_replay_matches_checkpoint_projection(
    default_vllm_config, workspace_init, monkeypatch, layer_groups
):
    """Real native projections, rotary and cache writes across shrinking batches."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native V4.1 context preparation requires SM12x")
    from contextlib import nullcontext

    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention import compressed_sparse_mla
    from b12x.attention.compressed_sparse_mla import rotary
    from b12x.preparation import PreparationSession, PreparedCall

    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4_1.b12x_layers import (
        B12xFP8LinearMethod,
        B12xRMSNorm,
    )
    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkContextCudaGraphs,
        DSparkDeepseekV4Model,
        _ContextKVProjection,
    )
    from vllm.utils.b12x import B12xWorkload, b12x_unit_providers
    from vllm.v1.worker import workspace
    from vllm.v1.worker.gpu import cudagraph_utils

    # The exercised context path is replicated and has no collectives. Supply
    # only the distributed capture envelope, not mock model/compute operations.
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cudagraph_utils, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(cudagraph_utils, "graph_capture", lambda device: nullcontext())
    config = default_vllm_config
    config.scheduler_config.max_num_batched_tokens = 8
    config.scheduler_config.max_num_seqs = 8
    config.compilation_config.max_cudagraph_capture_size = 8
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator(device=device).manual_seed(1831)

    class NativeLinear(nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(n, k, device=device, generator=generator).to(
                    torch.float8_e4m3fn
                ),
                requires_grad=False,
            )
            self.weight_scale_inv = torch.randint(
                122,
                128,
                (n // 32, k // 32),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ).view(torch.float8_e8m0fnu)
            self.method = B12xFP8LinearMethod(
                SimpleNamespace(weight_block_size=[32, 32])
            )
            self.method.process_weights_after_loading(self)

        def forward(self, x):
            return self.method.apply(self, x)

    with torch.device(device):
        model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=128, dspark_target_layer_ids=(0, 1))
        model.main_proj = NativeLinear(128, 256)
        model.main_norm = B12xRMSNorm(128)
        model.layers = nn.ModuleList()
        model._context_kv_projections = []
        page_bytes = compressed_sparse_mla.page_nbytes(
            32, cache_format="deepseek_v41", cache_kind="swa"
        )
        phases = (
            torch.arange(128, dtype=torch.float32)[:, None]
            * (torch.arange(32, dtype=torch.float32)[None, :] + 1)
            / 128
        )
        for _ in range(3):
            attn = DeepseekV4Attention.__new__(DeepseekV4Attention)
            nn.Module.__init__(attn)
            # Context insertion needs no query-attention plan.
            attn._ready = True
            attn.q_lora_rank = 256
            attn.fused_wqa_wkv = NativeLinear(768, 128)
            attn.kv_norm = B12xRMSNorm(512)
            attn.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.cat((phases.cos(), phases.sin()), dim=-1)
            )
            attn.swa_cache_layer = SimpleNamespace(
                kv_cache=torch.full((4, page_bytes), 91, dtype=torch.uint8),
                block_size=32,
            )
            attn._helper_plans = {
                "kv": rotary.plan(
                    rotary.Query(max_rows=8, heads=1, dim=512, cos_sin_dtype="float32"),
                    device=device,
                ),
                "swa_cache_write": compressed_sparse_mla.plan_cache_writer(
                    compressed_sparse_mla.CacheWriterQuery(
                        max_rows=8, page_size=32, cache_kind="swa", slot_dtype="int64"
                    ),
                    device=device,
                ),
            }
            layer = nn.Module()
            layer.attn = attn
            model.layers.append(layer)
            model._context_kv_projections.append(_ContextKVProjection(attn, 8))

        hidden = torch.zeros(8, 128, dtype=torch.bfloat16)
        positions = torch.zeros(8, dtype=torch.int64)
        slots = torch.full((2, 8), -1, dtype=torch.int64)
        context = DSparkContextCudaGraphs(
            model, config, hidden, positions, slots, layer_groups, 8
        )
        session = PreparationSession(device=device, autotune=False)
        workload = B12xWorkload(
            stage="weights",
            token_counts=tuple(
                context.manager.compilation_config.cudagraph_capture_sizes
            ),
            fixed_token_counts=(),
            output_dtype=torch.bfloat16,
            max_tokens=8,
            max_seqs=8,
            max_model_len=8,
        )
        # _ContextKVProjection is a non-module owner: it self-registers via
        # register_b12x_unit_provider, so its units are collected the same way
        # collect_b12x_units gathers PCIe/RoCE communicator units in production.
        units = [
            unit
            for provider in b12x_unit_providers()
            for unit in provider.get_b12x_preparation_units(provider, workload)
        ]
        units.extend(_units_from_modules(model, workload))
        prep_requests = [request for unit in units for request in unit.requests]
        for index, layer in enumerate(model.layers):

            def prepare_rotary(state, table=layer.attn.rotary_emb.cos_sin_cache):
                source = torch.ones((1, 1, 512), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                positions = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(
                    run=lambda: state.run(source, positions, table, out=output)
                )

            def prepare_writer(state):
                source = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
                cache = torch.empty((1, page_bytes), dtype=torch.uint8, device=device)
                slots = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, cache, slots))

            for role, prepare in (
                ("kv", prepare_rotary),
                ("swa_cache_write", prepare_writer),
            ):
                prep_requests.append(
                    layer.attn._helper_plans[role].request(
                        name=f"context.{index}.{role}", prepare_call=prepare
                    )
                )
        result = session.prepare(prep_requests, autotune=False)
        assert result is not None
        # Compare KV-only checkpoint packing against the original fused Q|KV
        # projection (nonuniform UE8M0 scales distinguish every block).
        x = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator)
        for layer, projection in zip(model.layers, model._context_kv_projections):
            full = layer.attn.fused_wqa_wkv(x)
            torch.testing.assert_close(projection(x), full[:, 256:], rtol=0, atol=0)
        for capacity in context.manager.compilation_config.cudagraph_capture_sizes:
            context._forward(capacity)
        workspace.lock_workspace()
        with kernel_resolution_guard("DSpark context capacities are prepared"):
            try:
                with session.capture():
                    context.capture()
                for rows in (7, 3, 6, 1):
                    # New allocations on every call ensure graphs never bind the
                    # target's transient auxiliary outputs.
                    aux = [
                        torch.randn(
                            rows, 128, dtype=torch.bfloat16, generator=generator
                        )
                        for _ in range(2)
                    ]
                    # Stale padding must not read this RoPE row.
                    positions.fill_(1000000)
                    positions[:rows].copy_(torch.arange(rows, dtype=torch.int64) + rows)
                    slots[0].copy_(torch.arange(8, dtype=torch.int64) + 32)
                    slots[1].copy_(torch.arange(8, dtype=torch.int64) + 64)
                    slots[:, rows - 1] = -1  # rejected suffix
                    if rows > 1:
                        slots[1, 0] = -1  # group-specific nonresident/PAD row
                    main_x = model.combine_hidden_states(torch.cat(aux, dim=-1))
                    expected = []
                    for i, layer in enumerate(model.layers):
                        attn = layer.attn
                        attn.swa_cache_layer.kv_cache.fill_(91)
                        # Oracle uses the original fused checkpoint projection.
                        kv = attn.kv_norm(attn.fused_wqa_wkv(main_x)[:, 256:])
                        group = 0 if layer_groups is None else layer_groups[i]
                        attn.insert_context_kv(
                            kv, positions[:rows], slots[group, :rows]
                        )
                        expected.append(attn.swa_cache_layer.kv_cache.clone())
                        attn.swa_cache_layer.kv_cache.fill_(91)
                    context.run(aux, rows)
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(hidden[:rows], main_x, rtol=0, atol=0)
                    for layer, cache in zip(model.layers, expected):
                        torch.testing.assert_close(
                            layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0
                        )
                    # A restored cache must not consume changed sources or metadata.
                    for source in aux:
                        source.fill_(float("nan"))
                    slots.fill_(0)
                    context.run(aux, rows, context_kv_is_restored=True)
                    for layer, cache in zip(model.layers, expected):
                        torch.testing.assert_close(
                            layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0
                        )
                assert not context.can_run(9)
                with pytest.raises(ValueError):
                    context.run(aux, 9)
            finally:
                context.close()
                workspace.unlock_workspace()
        session.close()
