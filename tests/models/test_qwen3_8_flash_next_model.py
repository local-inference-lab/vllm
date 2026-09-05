# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction and execution tests for Qwen3.8-Flash-Next."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
import vllm.distributed.utils as distributed_utils
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module
import vllm.model_executor.offloader as offloader
import vllm.models.qwen3_8_flash_next.hyperconnection as hyperconnection_module
import vllm.models.qwen3_8_flash_next.model as model_module
import vllm.models.qwen3_8_flash_next.nvidia.qsa as qsa_module
import vllm.models.qwen3_8_flash_next.ple_layer as ple_layer_module
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    _resolve_gdn_decode_kernel,
)
from vllm.models.qwen3_8_flash_next.ple_layer import Qwen3_8FlashNextPLELayer
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

_ALIGNED_PAGE_SIZE_BYTES = 818_176
_ALIGNED_BLOCK_SIZE = 752


def test_mtp_compaction_selector_reuses_aot_callable_across_draft_phases():
    from vllm.models.qwen3_8_flash_next.mtp import (
        Qwen3_8FlashNextMultiTokenPredictor,
    )

    predictor = SimpleNamespace(
        _decode_output_indices=torch.zeros(1, dtype=torch.int64)
    )
    select = Qwen3_8FlashNextMultiTokenPredictor.set_prefill_output_indices
    source = torch.arange(32).reshape(4, 8)
    tail = torch.tensor([3])
    select(predictor, tail)

    def compact(x):
        return x[predictor._prefill_output_indices]

    compiled = torch.compile(compact, fullgraph=True).aot_compile(((source,), {}))
    for index in (3, 1, 2):
        tail.fill_(index)
        select(predictor, tail)
        torch.testing.assert_close(compiled(source), source[index : index + 1])
        select(predictor, None)
        torch.testing.assert_close(compiled(source), source[:1])


@pytest.mark.parametrize("indices", [[0], [3], [0, 3]])
def test_mtp_compaction_preserves_attention_rows_and_selected_outputs(indices):
    """Cache-producing attention sees every row; tokenwise MLP sees only tails."""
    layer = model_module.Qwen3_8FlashNextDecoderLayer.__new__(
        model_module.Qwen3_8FlashNextDecoderLayer
    )
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = "full_attention"
    rows = {}

    def attention(*, hidden_states, positions):
        rows["attention"] = hidden_states.clone()
        return hidden_states + positions[:, None]

    def mlp(x):
        rows["mlp"] = x.shape[0]
        return x.square()

    layer.attn_hyper_connection = SimpleNamespace(mix=lambda x: (x, x * 2, x / 2))
    layer.mlp_hyper_connection = SimpleNamespace(
        combine_and_mix=lambda x, attn, injection: (x + attn, attn + injection, x)
    )
    layer.self_attn = attention
    layer.mlp = mlp
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    kwargs = dict(
        hidden_states=hidden,
        prev_block_output=None,
        prev_injection=None,
        positions=torch.arange(4),
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )
    expected = layer(**kwargs)
    selection = torch.tensor(indices)
    actual = layer(**kwargs, output_indices=selection)
    torch.testing.assert_close(rows["attention"], hidden * 2)
    assert rows["mlp"] == len(indices)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference[selection], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
@pytest.mark.parametrize("kind", ["gdn", "qsa"])
def test_attention_projection_overlap_replays_with_changed_inputs(
    monkeypatch, num_tokens: int, kind: str
) -> None:
    """The fork/join must precede consumers on every graph replay."""
    torch.manual_seed(42)
    qkvz = nn.Linear(2560, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    ba = nn.Linear(2560, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        in_proj_qkvz=lambda x: (qkvz(x), None),
        in_proj_ba=lambda x: (ba(x), None),
        qkv_proj=lambda x: (qkvz(x), None),
        indexer=SimpleNamespace(index_qk_proj=lambda x: (ba(x), None)),
    )
    module = gdn_module if kind == "gdn" else qsa_module
    op = (
        torch.ops.vllm.qwen_gdn_input_projections
        if kind == "gdn"
        else torch.ops.vllm.qwen3_8_flash_next_qsa_input_projections
    )
    stream = torch.cuda.Stream()
    monkeypatch.setattr(module, "aux_stream", lambda: stream)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test.gdn": layer}),
    )

    @torch.compile(backend="eager", fullgraph=True)
    def project(x):
        qkvz_out, ba_out = op(x, 512, 64, "test.gdn")
        return qkvz_out, ba_out.sigmoid()

    x = torch.randn(num_tokens, 2560, device="cuda", dtype=torch.bfloat16)
    main_stream = torch.cuda.Stream()
    main_stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(main_stream):
        for _ in range(3):
            project(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main_stream):
            actual = project(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            expected = (qkvz(x), ba(x).sigmoid())
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [4, 32])
def test_ple_prefetch_joins_before_embedding_consumers(monkeypatch, num_tokens) -> None:
    embedding = ple_layer_module.Qwen3_8FlashNextNGramEmbedding.__new__(
        ple_layer_module.Qwen3_8FlashNextNGramEmbedding
    )
    nn.Module.__init__(embedding)
    embedding.owner_prefix = "test.ple"
    embedding._embedding_out = torch.empty(32, 32, device="cuda")
    table = torch.randn(512, 32, device="cuda")

    def lookup(ids, query_start_loc, history):
        rows = (ids + history[0, 0]) % table.shape[0]
        embedding._embedding_out[: ids.numel()].copy_(table[rows])

    embedding._run_embedding = lookup
    stream = torch.cuda.Stream()
    monkeypatch.setattr(ple_layer_module, "_get_prefetch_stream", lambda: stream)
    monkeypatch.setattr(
        ple_layer_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"test.ple": SimpleNamespace(ple_embedding=embedding)}
        ),
    )

    @torch.compile(fullgraph=True)
    def forward(ids, query_start_loc, history, hidden):
        embedding.prefetch(ids, query_start_loc, history)
        hidden = hidden * 2
        return embedding(ids, query_start_loc, history, wait_for=hidden) + hidden

    ids = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")
    history = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    hidden = torch.randn(num_tokens, 32, device="cuda")
    main_stream = torch.cuda.Stream()
    main_stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(main_stream):
        for _ in range(3):
            forward(ids, query_start_loc, history, hidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main_stream):
            actual = forward(ids, query_start_loc, history, hidden)
        for _ in range(3):
            ids.random_(0, 512)
            history.random_(0, 512)
            hidden.normal_()
            graph.replay()
            expected = table[(ids + history[0, 0]) % 512] + hidden * 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _RecordingPlan:
    def __init__(self) -> None:
        self.bind_kwargs: dict[str, Any] | None = None
        self.binding = object()

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.binding

    def scratch_specs(self):
        return (
            SimpleNamespace(
                shape=(1,),
                dtype=torch.uint8,
                device=torch.device("cpu"),
            ),
        )


def _allocate_aligned_mamba_cache(
    *,
    layer_name: str,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[torch.dtype, ...],
    mamba_type: MambaAttentionBackendEnum,
    num_blocks: int = 2,
) -> torch.Tensor:
    spec = MambaSpec(
        shapes=shapes,
        dtypes=dtypes,
        block_size=_ALIGNED_BLOCK_SIZE,
        page_size_padded=_ALIGNED_PAGE_SIZE_BYTES,
        mamba_type=mamba_type,
        num_speculative_blocks=2,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                layers=[layer_name],
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], spec)],
    )
    return allocate_kv_cache(
        config,
        torch.device("cpu"),
        KVCacheLayout.BLHNC,
    )[layer_name]


def _set_tensor_attributes(module: nn.Module, *names: str) -> None:
    for name in names:
        setattr(module, name, torch.empty(0))


def test_ple_cpu_offload_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    assert ple_layer_module._resolve_ple_table_memory(None) == "mapped_host"


def test_qwen3_8_prefers_b12x_gdn_unless_explicitly_overridden(monkeypatch) -> None:
    config = SimpleNamespace(additional_config={})
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=True) == ("b12x", False)
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=False) == ("cuda", False)

    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=True) == ("triton", True)

    config.additional_config["gdn_decode_kernel"] = "b12x"
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=False) == ("b12x", True)


def test_explicit_ple_table_memory_overrides_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    assert (
        ple_layer_module._resolve_ple_table_memory({"ple_table_memory": "device"})
        == "device"
    )


def test_ple_registers_request_dependent_piecewise_splitting_ops_once() -> None:
    compilation_config = SimpleNamespace(
        static_forward_context={},
        splitting_ops=[],
    )

    ple_layer_module._register_ple_compilation_context(
        compilation_config,
        "model.layers.1.ple",
        nn.Identity(),
    )
    ple_layer_module._register_ple_compilation_context(
        compilation_config,
        "model.layers.5.ple",
        nn.Identity(),
    )

    assert compilation_config.splitting_ops == list(ple_layer_module._PLE_SPLITTING_OPS)
    assert set(compilation_config.static_forward_context) == {
        "model.layers.1.ple",
        "model.layers.5.ple",
    }


def test_gated_residual_uses_canonical_combine_ops(monkeypatch) -> None:
    combined = torch.full((2, 8), 3.0, dtype=torch.bfloat16)
    normalized = torch.full((2, 8), 5.0, dtype=torch.bfloat16)
    plan = object()
    binding = object()
    workspace = SimpleNamespace(plan=plan, bind=Mock(return_value=binding))
    api = SimpleNamespace(
        run_combine_norm=Mock(return_value=(combined, normalized)),
        run_combine=Mock(return_value=combined),
    )

    mixer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(rms_norm_eps=1e-6)
    mixer.hc_norm = SimpleNamespace(weight=torch.empty(8, dtype=torch.bfloat16))
    object.__setattr__(mixer, "_workspace", workspace)
    monkeypatch.setattr(hyperconnection_module, "_hyperconnection_api", lambda: api)
    monkeypatch.setattr(
        hyperconnection_module.GatedResidual,
        "_mix_normalized",
        lambda _self, value, _binding: (value[:, :2], None),
    )

    hidden_states = torch.empty((2, 8), dtype=torch.bfloat16)
    block_output = torch.empty((2, 2), dtype=torch.bfloat16)
    injection = torch.empty((2, 4), dtype=torch.bfloat16)
    actual_combined, block_input, actual_injection = mixer.combine_and_mix(
        hidden_states,
        block_output,
        injection,
    )
    final_combined = mixer.combine(hidden_states, block_output, injection)

    assert actual_combined is combined
    assert block_input.data_ptr() == normalized.data_ptr()
    assert actual_injection is None
    assert final_combined is combined
    api.run_combine_norm.assert_called_once_with(
        hidden_states,
        block_output,
        injection,
        mixer.hc_norm.weight,
        eps=1e-6,
        plan=plan,
    )
    workspace.bind.assert_called_once_with(2)
    api.run_combine.assert_called_once_with(
        hidden_states,
        block_output,
        injection,
        plan=plan,
    )


def test_decoder_layer_factory_accepts_make_layers_prefix(monkeypatch) -> None:
    created_layers: list[tuple[str, str]] = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeWorkspace(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeDecoderLayer(nn.Module):
        def __init__(
            self,
            _vllm_config,
            layer_type: str,
            _workspace,
            *,
            prefix: str,
        ) -> None:
            super().__init__()
            created_layers.append((prefix, layer_type))

    class FakePPGroup:
        rank_in_group = 0
        world_size = 1
        is_last_rank = False

    class FakeOffloader:
        @staticmethod
        def wrap_modules(modules):
            return list(modules)

    pp_group = FakePPGroup()
    monkeypatch.setattr(model_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(model_module, "HyperConnectionWorkspace", FakeWorkspace)
    monkeypatch.setattr(model_module, "Qwen3_8FlashNextDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        distributed_utils,
        "get_pp_indices",
        lambda num_layers, _rank, _world_size: (0, num_layers),
    )
    monkeypatch.setattr(offloader, "get_offloader", lambda: FakeOffloader())

    text_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
        indexer_n_heads=None,
        hc_count=4,
        hc_lowrank=2,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=None,
    )

    model = model_module.Qwen3_8FlashNextModel(
        vllm_config=vllm_config,
        prefix="model",
    )

    assert len(model.layers) == 2
    assert created_layers == [
        ("model.layers.0", "full_attention"),
        ("model.layers.1", "linear_attention"),
    ]


def test_ple_bind_preserves_exact_aligned_page_stride(monkeypatch) -> None:
    shapes = ((10_240, 11),)
    dtypes = (torch.bfloat16,)
    raw_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.1.ple",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    plan = _RecordingPlan()
    planned_slots: list[int] = []

    monkeypatch.setattr(
        Qwen3_8FlashNextPLELayer, "get_state_shape", lambda _self: shapes
    )
    monkeypatch.setattr(
        Qwen3_8FlashNextPLELayer, "get_state_dtype", lambda _self: dtypes
    )

    def make_plan(_self, max_state_slots: int):
        planned_slots.append(max_state_slots)
        return plan

    monkeypatch.setattr(Qwen3_8FlashNextPLELayer, "_make_plan", make_plan)
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer.conv_state_len = 9
    layer.num_spec_tokens = 2
    layer.hc_hidden_size = 10_240
    _set_tensor_attributes(
        layer,
        "_scratch",
        "_residual",
        "_key",
        "_value",
        "_query_start_loc",
        "_state_slot_ids",
        "_state_is_fresh",
        "_num_accepted_tokens",
        "_num_seqs",
        "_num_tokens",
        "_out",
        "_request_is_prefill",
    )
    layer.norm_key = SimpleNamespace(weight=torch.empty(0))
    layer.norm_query = SimpleNamespace(weight=torch.empty(0))
    layer.norm_conv = SimpleNamespace(weight=torch.empty(0))
    layer.conv1d = SimpleNamespace(weight=torch.empty(1, 1, 1))

    layer.bind_kv_cache(raw_cache)

    (conv_state,) = layer.kv_cache
    assert raw_cache.shape == (2, 1, 1, 225_280)
    assert raw_cache.stride(0) == _ALIGNED_PAGE_SIZE_BYTES
    assert conv_state.stride() == (409_088, 11, 1)
    assert not conv_state.is_contiguous()
    assert conv_state[0].is_contiguous()
    assert conv_state.data_ptr() == raw_cache.data_ptr()
    assert (
        conv_state.untyped_storage().data_ptr()
        == raw_cache.untyped_storage().data_ptr()
    )
    assert planned_slots == [2]
    assert plan.bind_kwargs is None
    assert layer._bind_ple() is plan.binding
    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["conv_state"] is conv_state
    assert not hasattr(layer, "_binding")

    layer.unbind_kv_cache()

    assert layer.kv_cache == ()
    assert layer._plan is None
    with pytest.raises(RuntimeError, match="was not bound"):
        layer._bind_ple()

    replacement_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.1.ple",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    layer.bind_kv_cache(replacement_cache)

    assert planned_slots == [2, 2]
    assert layer.kv_cache[0] is not conv_state
    assert layer._bind_ple() is plan.binding
    assert plan.bind_kwargs["conv_state"] is layer.kv_cache[0]


def test_b12x_gdn_bind_preserves_exact_aligned_page_stride(monkeypatch) -> None:
    shapes = ((2_560, 5), (12, 128, 128))
    dtypes = (torch.bfloat16, torch.float32)
    raw_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.0.linear_attn",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    plan = _RecordingPlan()
    bind_mock = Mock(wraps=plan.bind)
    monkeypatch.setattr(plan, "bind", bind_mock)
    planned_slots: list[int] = []

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "get_state_shape", lambda _: shapes)
    monkeypatch.setattr(QwenGatedDeltaNetAttention, "get_state_dtype", lambda _: dtypes)

    def make_plan(_self, max_state_slots: int):
        planned_slots.append(max_state_slots)
        return plan

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "_make_b12x_gdn_plan", make_plan)
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.gdn_decode_kernel = "b12x"
    layer._b12x_plan = None
    _set_tensor_attributes(
        layer,
        "_b12x_scratch",
        "_b12x_mixed_qkv",
        "_b12x_a",
        "_b12x_b",
        "_b12x_z",
        "_b12x_query_start_loc",
        "_b12x_num_accepted_tokens",
        "_b12x_state_indices",
        "_b12x_num_seqs",
        "_b12x_num_tokens",
        "_b12x_output",
    )
    layer.A_log = nn.Parameter(torch.empty(0))
    layer.dt_bias = nn.Parameter(torch.empty(0))
    layer.norm = SimpleNamespace(weight=torch.empty(0))

    layer.bind_kv_cache(raw_cache)
    bind_mock.assert_not_called()
    binding = layer._bind_b12x_gdn_decode()
    assert binding is plan.binding
    layer._bind_b12x_gdn_decode()
    assert bind_mock.call_count == 2

    conv_state, recurrent_state = layer.kv_cache
    assert conv_state.stride() == (409_088, 5, 1)
    assert recurrent_state.stride() == (204_544, 16_384, 128, 1)
    assert recurrent_state.storage_offset() == 6_400
    assert recurrent_state.data_ptr() == raw_cache.data_ptr() + 25_600
    assert not recurrent_state.is_contiguous()
    assert recurrent_state[0].is_contiguous()
    assert planned_slots == [2]
    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["recurrent_state"] is recurrent_state
    assert not hasattr(layer, "_b12x_binding")

    layer.unbind_kv_cache()

    assert layer.kv_cache == ()
    assert layer._b12x_plan is None
    with pytest.raises(RuntimeError, match="KV cache was not bound"):
        layer._bind_b12x_gdn_decode()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_b12x_gdn_binds_live_projections_and_stages_rollback_metadata(
    monkeypatch,
) -> None:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._b12x_max_tokens = 6
    layer._b12x_max_seqs = 2
    layer._b12x_state_index_columns = 3
    plan = _RecordingPlan()
    bind_mock = Mock(wraps=plan.bind)
    monkeypatch.setattr(plan, "bind", bind_mock)
    layer._b12x_plan = plan
    layer._b12x_scratch = torch.empty(1, device="cuda")
    layer.A_log = torch.empty(2, device="cuda")
    layer.dt_bias = torch.empty(2, device="cuda")
    layer.norm = SimpleNamespace(weight=torch.empty(4, device="cuda"))
    layer.kv_cache = (torch.empty(0), torch.empty(10, 2, 4, 4, device="cuda"))
    layer.layer_norm_epsilon = 1e-6
    layer.head_k_dim = 128
    layer._b12x_mixed_qkv = torch.empty(6, 8, device="cuda")
    layer._b12x_a = torch.empty(6, 2, device="cuda")
    layer._b12x_b = torch.empty(6, 2, device="cuda")
    layer._b12x_z = torch.empty(6, 2, 4, device="cuda")
    layer._b12x_output = torch.full((6, 2, 4), 17.0, device="cuda")
    layer._b12x_query_start_loc = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    layer._b12x_num_accepted_tokens = torch.full(
        (2,), -1, dtype=torch.int32, device="cuda"
    )
    layer._b12x_state_indices = torch.full((2, 3), -1, dtype=torch.int32, device="cuda")
    layer._b12x_num_seqs = torch.zeros(1, dtype=torch.int32, device="cuda")
    layer._b12x_num_tokens = torch.zeros(1, dtype=torch.int32, device="cuda")
    calls: list[tuple[object, float, float]] = []

    def run(binding, *, eps: float, scale: float) -> None:
        calls.append((binding, eps, scale))
        assert plan.bind_kwargs is not None
        plan.bind_kwargs["output"].fill_(17.0)

    layer._b12x_gdn_api = SimpleNamespace(run=run)
    mixed_qkv = torch.arange(40, dtype=torch.float32, device="cuda").reshape(5, 8)
    a = torch.arange(10, dtype=torch.float32, device="cuda").reshape(5, 2)
    b = a + 20
    output_gate = torch.arange(40, dtype=torch.float32, device="cuda").reshape(5, 2, 4)
    state_indices = torch.tensor(
        [[7, 8, 9], [4, 5, 6]], dtype=torch.int32, device="cuda"
    )
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    core_attn_out = torch.zeros(5, 2, 4, device="cuda")

    layer._run_b12x_gdn_decode_post_conv(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        output_gate=output_gate,
        core_attn_out=core_attn_out,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        num_requests=2,
    )

    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["mixed_qkv"] is mixed_qkv
    assert plan.bind_kwargs["a"] is a
    assert plan.bind_kwargs["b"] is b
    assert plan.bind_kwargs["z"] is output_gate
    assert plan.bind_kwargs["output"].data_ptr() == core_attn_out.data_ptr()
    torch.testing.assert_close(layer._b12x_query_start_loc, query_start_loc)
    torch.testing.assert_close(layer._b12x_num_accepted_tokens, accepted)
    torch.testing.assert_close(layer._b12x_state_indices, state_indices)
    torch.testing.assert_close(
        layer._b12x_num_seqs, torch.tensor([2], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        layer._b12x_num_tokens, torch.tensor([5], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(core_attn_out, torch.full_like(core_attn_out, 17.0))
    assert calls == [(plan.binding, 1e-6, 128**-0.5)]
    bind_mock.assert_called_once()
    assert plan.bind_kwargs["recurrent_state"] is layer.kv_cache[1]
    assert plan.bind_kwargs["scratch"] is layer._b12x_scratch


@pytest.mark.parametrize("head_dim, expected", [(128, "flashinfer"), (64, "triton")])
def test_sm120_gdn_prefill_selects_supported_flashinfer_geometry(
    monkeypatch, head_dim, expected
):
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability=lambda _cap: False,
        is_device_capability_family=lambda cap: cap == 120,
        get_cuda_runtime_major=lambda: 13,
    )
    monkeypatch.setattr(gdn_module, "current_platform", platform)
    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_dim)
        ),
    )
    assert gdn_module._resolve_gdn_prefill_backend(config) == ("auto", expected)


def test_sm120_flashinfer_sequence_offsets_preserve_values(monkeypatch):
    monkeypatch.setattr(
        gdn_module,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda cap: cap == 120),
    )
    offsets = torch.tensor([0, 1, 129, 6019], dtype=torch.int32)
    converted = gdn_module._prepare_flashinfer_cu_seqlens(offsets)
    assert converted.dtype == torch.int64
    torch.testing.assert_close(converted, offsets.to(torch.int64))
    assert gdn_module._prepare_flashinfer_cu_seqlens(converted) is converted
    assert gdn_module._prepare_flashinfer_cu_seqlens(None) is None


@pytest.mark.skipif(
    not gdn_module.current_platform.is_device_capability_family(120),
    reason="requires SM12x FlashInfer GDN",
)
@pytest.mark.parametrize("boundaries", [[0, 129], [0, 1, 130, 259]])
def test_sm120_flashinfer_gdn_int32_offsets_match_int64(boundaries):
    """The scheduler's int32 offsets preserve the supported kernel result."""
    torch.manual_seed(312)
    rows = boundaries[-1]
    shape = (1, rows, 4, 128)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    gate_shape = (1, rows, 4)
    g = -torch.rand(gate_shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(gate_shape, device="cuda", dtype=torch.float32)
    state = torch.randn(len(boundaries) - 1, 4, 128, 128, device="cuda")
    offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int64)
    expected = gdn_module.fi_chunk_gated_delta_rule(
        q, k, v, g, beta, state.clone(), True, offsets
    )
    actual = gdn_module.fi_chunk_gated_delta_rule(
        q, k, v, g, beta, state.clone(), True, offsets.to(torch.int32)
    )
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all() and torch.count_nonzero(result)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
