# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    BATCH_SPECS,
    _test_backend_correctness,
)
from tests.v1.attention.utils import BatchSpec
from vllm.config import ModelConfig, set_current_vllm_config
from vllm.model_executor.kernels.attention import b12x_mla_query
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_paged_attention
from vllm.utils.b12x import PreparationResourceUnavailableError
from vllm.v1.attention.backends import b12x
from vllm.v1.attention.backends.b12x import (
    B12xPagedAttentionBackend,
    B12xPagedAttentionImpl,
    _kv_page_size,
    _max_page_table_width,
)
from vllm.v1.attention.backends.mla import b12x_indexer, b12x_mla_sparse
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout


def _require_b12x_paged_attention() -> None:
    capability = current_platform.get_device_capability()
    if (
        not current_platform.is_cuda()
        or capability is None
        or not B12xPagedAttentionBackend.supports_compute_capability(capability)
    ):
        pytest.skip("b12x paged attention requires SM120 or SM121.")

    paged_attention = get_b12x_paged_attention()
    if paged_attention is None or not paged_attention.is_supported():
        pytest.skip("b12x paged attention is not available.")


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_bf16_mla_query_uses_public_run_api(monkeypatch) -> None:
    from vllm.utils.b12x import register_b12x_layer

    calls = []
    module = SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(b12x_mla_query, "get_b12x_mla_query_projection", lambda: module)
    q_nope = torch.empty((8, 2, 192), dtype=torch.bfloat16)
    weight = torch.empty((8, 192, 512), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 8, 64), dtype=torch.bfloat16)
    output = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    plan = object()
    plan_calls = []

    class _FakeQueryLayer:
        def b12x_query_plan(self, tokens):
            plan_calls.append(tokens)
            return plan

    layer = _FakeQueryLayer()
    register_b12x_layer("test.mla_query", layer)

    b12x_mla_query._b12x_bf16_mla_query_impl(
        q_nope, weight, q_pe, output, "test.mla_query"
    )

    assert plan_calls == [q_nope.shape[1]]
    assert calls == [((q_nope, weight, q_pe, output), {"plan": plan})]


def test_b12x_bf16_mla_query_uses_backend_workspace(monkeypatch) -> None:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.q_pad_num_heads = None
    layer.kv_lora_rank = 512
    layer.qk_rope_head_dim = 64
    layer.W_UK_T = torch.nn.Parameter(
        torch.empty((8, 192, 512), dtype=torch.bfloat16), requires_grad=False
    )
    layer._b12x_query_layer_name = "test.mla_query_backend_workspace"
    workspace = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    layer.impl = SimpleNamespace(get_fused_mla_query_output=lambda *args: workspace)
    calls = []
    monkeypatch.setattr(
        mla_attention, "can_implement_bf16_mla_query", lambda **kwargs: True
    )

    def run_query(*args, **kwargs):
        calls.append(args)
        return args[-1]

    monkeypatch.setattr(mla_attention, "run_bf16_mla_query", run_query)
    q_nope = torch.empty((8, 2, 192), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 8, 64), dtype=torch.bfloat16)

    result = layer._try_fused_mla_query(q_nope, q_pe)

    assert result is workspace
    assert calls == [(q_nope, layer.W_UK_T, q_pe, workspace)]


def _mla_query_layer() -> MLAAttention:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.W_UK_T = torch.nn.Parameter(
        torch.empty((8, 192, 512), dtype=torch.bfloat16), requires_grad=False
    )
    layer._b12x_query_prefix = "test.mla_query_plans"
    layer._b12x_query_plans = {}
    return layer


def test_mla_query_plan_declares_an_unplanned_row_count_without_preparing(
    monkeypatch,
) -> None:
    import b12x.preparation as preparation

    layer = _mla_query_layer()
    planned = object()
    layer._b12x_query_plans[4] = planned
    prepared = []
    monkeypatch.setattr(
        preparation, "prepare_default", lambda request: prepared.append(request)
    )

    assert layer.b12x_query_plan(4) is planned
    assert prepared == []

    declared = layer.b12x_query_plan(11)
    assert layer.b12x_query_plan(11) is declared
    assert layer._b12x_query_plans == {4: planned, 11: declared}
    assert declared.query.max_rows == 11
    # Serving never prepares: the plan materializes its default on first use.
    assert prepared == []

    # The startup unit's prepare call for the same count binds the layer's weight.
    runs = []
    state = SimpleNamespace(run=lambda *args: runs.append(args))
    layer._b12x_query_call(11)(state).run()
    (q_nope, weight, q_pe, output), = runs
    assert weight is layer.W_UK_T
    assert q_nope.shape == (8, 11, 192)
    assert q_pe.shape == (11, 8, 64)
    assert output.shape == (11, 8, 576)


def test_mla_query_preparation_units_declare_the_planned_row_counts(
    monkeypatch,
) -> None:
    import b12x.preparation as preparation

    from vllm.utils.b12x import B12xWorkload

    layer = _mla_query_layer()
    monkeypatch.setattr(
        mla_attention,
        "can_implement_bf16_mla_query",
        lambda **kwargs: 1 <= kwargs["max_m"] <= 32,
    )
    prepared = []
    monkeypatch.setattr(
        preparation, "prepare_default", lambda request: prepared.append(request)
    )
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8, 128),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=128,
        max_seqs=8,
        max_model_len=1024,
    )

    (unit,) = layer.get_b12x_preparation_units(layer, workload)

    assert unit.name == "MLA_QUERY"
    assert unit.key == ("test.mla_query_plans", (1, 8))
    assert [request.name for request in unit.requests] == [
        "test.mla_query_plans.m1",
        "test.mla_query_plans.m8",
    ]
    assert [request.plan.query.max_rows for request in unit.requests] == [1, 8]
    assert layer.b12x_query_plan(8) is unit.requests[1].plan
    assert prepared == []


def test_mla_preallocates_absorbed_weights_before_dequantization(
    monkeypatch,
) -> None:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(
        process_weights_after_loading=lambda dtype: None,
    )
    layer.is_amx_bmm_enabled = False
    layer.kv_lora_rank = 2
    layer.num_heads = 2
    layer.qk_nope_head_dim = 3
    layer.v_head_dim = 4
    layer.kv_b_proj = torch.nn.Module()
    layer.kv_b_proj.qweight = torch.nn.Parameter(
        torch.ones((14, 2), dtype=torch.int8), requires_grad=False
    )
    layer.kv_b_proj.quant_method = object()
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.dcp_q_replicate = False
    layer.quant_config = None
    layer.layer_name = "test"
    dequantized = torch.arange(28.0, dtype=torch.float32).reshape(14, 2)
    events = []
    preallocated = []
    preallocate = mla_attention._preallocate_absorbed_mla_weights

    def track_preallocation(*args, **kwargs):
        events.append("preallocate")
        weights = preallocate(*args, **kwargs)
        preallocated.extend(weights)
        return weights

    def fake_dequant(*args, **kwargs):
        events.append("dequantize")
        return dequantized

    monkeypatch.setattr(
        mla_attention,
        "_preallocate_absorbed_mla_weights",
        track_preallocation,
    )
    monkeypatch.setattr(mla_attention, "get_and_maybe_dequant_weights", fake_dequant)
    monkeypatch.setattr(mla_attention, "set_default_quant_scales", lambda *a, **k: None)

    with torch.no_grad():
        layer.process_weights_after_loading(torch.float32)

    assert events == ["preallocate", "dequantize"]
    assert layer.W_UV.data_ptr() == preallocated[0].data_ptr()
    assert layer.W_UK_T.data_ptr() == preallocated[1].data_ptr()


@pytest.mark.parametrize(
    ("max_model_len", "page_table_width"),
    [
        (64, 2),
        (129, 4),
        (4096, 64),
        (999936, 15624),
        (999937, 15626),
        (1000000, 15626),
        (1000001, 15626),
    ],
)
def test_b12x_dsa_indexer_owns_prefill_width_cap(
    monkeypatch, max_model_len: int, page_table_width: int
) -> None:
    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        plan=lambda caps: SimpleNamespace(caps=caps),
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=8),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    with set_current_vllm_config(config):
        indexer = b12x_indexer.B12xSparseIndexer(
            k_cache=object(),
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=4,
            head_dim=128,
            max_model_len=max_model_len,
            max_total_seq_len=max_model_len,
            topk_indices_buffer=torch.empty((8, 4), dtype=torch.int32),
            skip_k_cache_insert=True,
            num_q_heads=16,
            output_physical_slots=True,
        )

    torch.testing.assert_close(
        indexer.active_width_cap,
        torch.tensor([max_model_len], dtype=torch.int32),
    )
    assert indexer._max_page_table_width == page_table_width
    assert indexer.output_physical_slots
    assert indexer._prepared_plans == {}


def test_b12x_dsa_indexer_reuses_capacity_and_declares_overflow(
    monkeypatch,
) -> None:
    import b12x.preparation as preparation

    class _Plan:
        def __init__(self, caps, invocation):
            self.caps, self.invocation = caps, invocation
            self.request_kwargs = None

        def request(self, **kwargs):
            self.request_kwargs = kwargs
            return ("request", kwargs["name"])

    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        plan=lambda caps, *, invocation: _Plan(caps, invocation),
        invocation_from_descriptors=lambda caps, *, operands: (
            "invocation", caps.max_q_rows, tuple(operands),
        ),
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=8),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )
    k_cache = SimpleNamespace(
        prefix="layer", kv_cache=torch.empty((2, 64, 132), dtype=torch.uint8)
    )
    with set_current_vllm_config(config):
        indexer = b12x_indexer.B12xSparseIndexer(
            k_cache=k_cache,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=4,
            head_dim=128,
            max_model_len=4096,
            max_total_seq_len=4096,
            topk_indices_buffer=torch.empty((8, 4), dtype=torch.int32),
            skip_k_cache_insert=True,
            num_q_heads=16,
            output_physical_slots=True,
        )
    planned = object()
    indexer._prepared_plans = {("decode", 4): planned}
    prepared = []
    monkeypatch.setattr(
        indexer, "_prepare_call", lambda mode, caps: f"call:{mode}:{caps.max_q_rows}"
    )
    monkeypatch.setattr(
        preparation, "prepare_default", lambda request: prepared.append(request)
    )

    assert indexer._plan("decode", 4) is planned
    assert indexer._plan("decode", 3) is planned
    assert indexer._prepared_plans == {("decode", 4): planned}
    assert prepared == []

    plan = indexer._plan("prefill", 11)
    assert isinstance(plan, _Plan)
    assert (plan.caps.mode, plan.caps.max_q_rows, plan.caps.max_batch) == ("prefill", 11, 8)
    assert plan.caps.max_page_table_width == indexer._max_page_table_width == 64
    assert plan.caps.num_q_heads == 16 and plan.caps.topk == 4
    assert plan.caps.output_index_space == "physical"
    assert plan.invocation[:2] == ("invocation", 11)
    assert plan.request_kwargs is None
    assert indexer._plan("prefill", 11) is plan
    assert indexer._prepared_plans[("prefill", 11)] is plan
    assert indexer._plan("prefill", 7) is plan
    assert ("prefill", 7) not in indexer._prepared_plans

    decode = indexer._plan("decode", 11)
    assert (decode.caps.mode, decode.caps.max_q_rows, decode.caps.max_batch) == ("decode", 11, 11)
    assert indexer._plan("decode", 11) is decode
    # Serving never prepares: the plans materialize their defaults on first use.
    assert prepared == []


def test_b12x_sparse_mla_prefill_binds_request_sequence_lengths(
    monkeypatch,
) -> None:
    calls = {}

    def bind(plan, **kwargs):
        calls["plan"] = plan
        calls["bind"] = kwargs
        return object()

    plan = SimpleNamespace()
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._ckv_gather_enabled = False
    # The route and the declared row count key each plan.
    impl._plans = {("decode", 1): object(), ("extend", 8): plan}
    impl._scratch_nbytes = 64
    impl._cache_record_bytes = 656
    impl._max_tokens = 8
    impl._input_num_heads = 2
    impl._q_head_dim = 576
    impl._topk_tokens = 4
    impl._ckv_local_capacity = 0
    impl.topk_indices_buffer = torch.zeros((8, 4), dtype=torch.int32)
    impl.dcp_world_size = 1
    impl.kv_lora_rank = 512
    impl.num_heads = 2
    impl.need_to_return_lse_for_decode = False
    impl._physical_selection_provider = None
    impl._bind = bind
    impl._run = lambda binding: torch.empty((8, 2, 512))

    monkeypatch.setattr(
        b12x_mla_sparse, "current_workspace_manager", lambda: _Workspace()
    )
    metadata = SimpleNamespace(
        max_query_len=8,
        is_spec_decode=False,
        num_reqs=1,
        num_prefills=1,
        num_decode_tokens=0,
        num_actual_tokens=8,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        req_id_per_token=torch.zeros(8, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        cache_seq_lens_per_token=torch.arange(1, 9, dtype=torch.int32),
    )
    q = torch.empty((8, 2, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 656), dtype=torch.uint8)

    impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

    assert calls["plan"] is plan
    torch.testing.assert_close(calls["bind"]["cache_lengths"], metadata.seq_lens)
    assert calls["bind"]["cache_lengths"].shape == (1,)
    assert (
        calls["bind"]["selected_indices"].data_ptr()
        == impl.topk_indices_buffer.data_ptr()
    )
    assert (
        calls["bind"]["selected_lengths"].data_ptr()
        == metadata.cache_seq_lens_per_token.data_ptr()
    )


def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
):
    return q_idx + context_len >= kv_idx


def _causal_sliding_window_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
    sliding_window: int,
):
    causal_mask = q_idx + context_len >= kv_idx
    window_mask = q_idx + context_len - kv_idx < sliding_window
    return causal_mask & window_mask


@pytest.mark.parametrize(
    ("dtype", "kv_cache_dtype", "paged_attention", "expected_reason"),
    [
        pytest.param(
            torch.float16,
            "fp8_e4m3",
            None,
            "b12x currently requires bfloat16 queries",
            id="query-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "float16",
            None,
            "b12x does not support float16 KV cache",
            id="kv-cache-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            None,
            "Install the b12x backend with `pip install vllm[b12x]`",
            id="package-not-installed",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: False),
            "b12x paged attention is not supported on the current device",
            id="device-api",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: True),
            None,
            id="supported",
        ),
    ],
)
def test_b12x_attention_config_support(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    paged_attention,
    expected_reason: str | None,
) -> None:
    monkeypatch.setattr(
        b12x,
        "get_b12x_paged_attention",
        lambda: paged_attention,
    )

    reason = B12xPagedAttentionBackend.supports_combination(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=128,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )

    assert reason == expected_reason


def test_b12x_attention_uses_two_plane_nhd_cache() -> None:
    spec = B12xPagedAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
    )

    assert spec.num_head_slots == 2
    assert spec.state_content_bytes == 4 * 128 * 2
    assert B12xPagedAttentionBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.LBHNC,
        KVCacheLayout.BLHNC,
    )
    assert B12xPagedAttentionBackend.supports_block_size(128)
    assert not B12xPagedAttentionBackend.supports_block_size(32)


def test_b12x_attention_hybrid_cache_capacity_includes_expansion() -> None:
    assert _max_page_table_width(4096, 128, 4096, False) == 32
    assert _max_page_table_width(4096, 128, 4096, True) == 64


def test_b12x_attention_runtime_page_size_comes_from_cache() -> None:
    key_cache = torch.empty((3, 64, 4, 128), device="meta")
    value_cache = torch.empty_like(key_cache)

    assert _kv_page_size(key_cache, value_cache) == 64
    with pytest.raises(ValueError, match="matching K/V page sizes"):
        _kv_page_size(key_cache, torch.empty((3, 128, 4, 128), device="meta"))


def test_b12x_attention_requires_prepared_decode_plan() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    impl._plans = {}
    impl._verify_q_per_req = 0
    impl._extend_q_capacities = (16,)
    metadata = SimpleNamespace(max_query_len=1)

    with pytest.raises(PreparationResourceUnavailableError, match="not prepared"):
        impl._select_plan(metadata, 7, 7, 7, 64)


def test_b12x_attention_fp8_descales_follow_request_batch() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    impl.kv_cache_dtype = "fp8_e4m3"
    layer = SimpleNamespace(
        _k_scale=torch.tensor(2.0),
        _v_scale=torch.tensor([3.0, 4.0, 5.0]),
    )

    k_descale, v_descale = impl._prepare_fp8_descales(
        layer, num_reqs=2, device=torch.device("cpu")
    )

    torch.testing.assert_close(k_descale, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(v_descale, torch.tensor([3.0, 4.0]))
    assert k_descale.stride() == (0,)
    assert v_descale.stride() == (1,)


def test_b12x_attention_sinks_refresh_in_place_after_reload() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    impl._sinks_source = source
    impl.sinks = None

    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.sinks is not None
    sinks_ptr = impl.sinks.data_ptr()
    source.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
    impl.process_weights_after_loading(torch.bfloat16)

    assert impl.sinks.data_ptr() == sinks_ptr
    torch.testing.assert_close(impl.sinks, source.float())


@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_decode", "small_prefill", "mixed_small", "medium_decode"],
)
@pytest.mark.parametrize(
    ("kv_cache_dtype", "model_dtype"),
    [
        ("auto", None),
        ("bfloat16", torch.bfloat16),
        ("fp8_e4m3", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("block_size", [64, 128])
def test_b12x_causal_backend_correctness(
    default_vllm_config,
    workspace_init,
    batch_spec_name: str,
    kv_cache_dtype: str,
    model_dtype: torch.dtype | None,
    block_size: int,
) -> None:
    """b12x causal paged attention matches the shared SDPA reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        model_dtype=model_dtype,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


@pytest.mark.parametrize(
    "batch_spec",
    [
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[1, 1]), id="decode"),
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[8, 8]), id="prefill"),
    ],
)
def test_b12x_causal_sliding_window(
    default_vllm_config,
    workspace_init,
    batch_spec: BatchSpec,
) -> None:
    """b12x causal sliding-window attention matches the shared reference."""
    _require_b12x_paged_attention()

    model = "microsoft/Phi-tiny-MoE-instruct"
    sliding_window = ModelConfig(
        model=model, max_model_len=max(batch_spec.seq_lens)
    ).get_sliding_window()
    assert sliding_window is not None
    mask = partial(_causal_sliding_window_mask, sliding_window=sliding_window)

    _test_backend_correctness(
        batch_spec,
        model,
        [AttentionBackendEnum.B12X],
        mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_attention_sinks(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x attention sinks match the explicit sink reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_prefill"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        use_sinks=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_decode_cuda_graph_replay(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x decode output remains correct after CUDA graph replay."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_decode"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_speculative_verification_uses_cuda_graph_plan(
    default_vllm_config,
    workspace_init,
) -> None:
    """Exercise the verifier plan rather than the general extend plan."""
    _require_b12x_paged_attention()
    batch_spec = BatchSpec(seq_lens=[32, 40], query_lens=[4, 4])

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=128,
        num_speculative_tokens=3,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("compressed", [False, True])
@torch.inference_mode()
def test_b12x_dsa_indexer_live_rows_reuse_capacity_with_high_page_ids(mode, compressed, monkeypatch):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer.reference import (
        pack_index_k_cache_reference, unpack_index_k_cache_reference,
    )
    from b12x.preparation import PreparationSession
    from vllm.utils.b12x import B12xWorkload
    import vllm.v1.worker.workspace as workspace

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(71)
    heads, topk, max_rows, width = 16, 512, 128, 16
    packed = pack_index_k_cache_reference(torch.randn(1024, 128, device=device))
    decoded = unpack_index_k_cache_reference(packed, num_tokens=1024)
    high_page = (1 << 31) // (64 * 132) + 3
    cache = torch.empty((high_page + width, 64, 132), dtype=torch.uint8, device=device)
    cache[:width].copy_(packed.reshape(width, 64, 132))
    cache[high_page:].copy_(packed.reshape(width, 64, 132))
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_rows, max_num_seqs=4),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1,
        ),
    )
    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4_indexer
    with set_current_vllm_config(config):
        cls = c4_indexer.B12xC4SparseIndexer if compressed else b12x_indexer.B12xSparseIndexer
        options = {"compress_ratio": 4} if compressed else {"num_q_heads": heads, "output_physical_slots": True}
        indexer = cls(
            k_cache=SimpleNamespace(prefix="indexer", kv_cache=cache),
            quant_block_size=128, scale_fmt="ue8m0", topk_tokens=topk, head_dim=128,
            max_model_len=1024, max_total_seq_len=1024,
            topk_indices_buffer=torch.empty((max_rows, topk), dtype=torch.int32, device=device),
            skip_k_cache_insert=True, **options,
        )
        if compressed:
            indexer.set_b12x_index_cache(cache, num_q_heads=heads)
    workload = B12xWorkload(
        stage="state", token_counts=(4, 125, max_rows), fixed_token_counts=(4,),
        output_dtype=torch.bfloat16, max_tokens=max_rows, max_seqs=4, max_model_len=1024,
    )
    units = indexer.get_b12x_preparation_units(indexer, workload)
    plans = indexer._plans if compressed else indexer._prepared_plans
    lookup = indexer._plan_for if compressed else indexer._plan
    assert set(plans) == {("decode", 4), ("prefill", max_rows)}
    capacity = 4 if mode == "decode" else max_rows
    q = torch.randn((capacity, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.rand((capacity, heads), device=device)
    lengths = torch.full((capacity,), 1024, dtype=torch.int32, device=device)
    pages = torch.arange(high_page, high_page + width, dtype=torch.int32, device=device)[None]
    pages = pages.expand(capacity, width)
    if mode == "decode":
        pages = pages.contiguous()
    output = torch.empty((capacity, topk), dtype=torch.int32, device=device)
    plan = lookup(mode, capacity)

    def run(rows):
        assert lookup(mode, rows) is plan
        if compressed:
            return indexer.run_paged_topk(
                q=q[:rows], weights=weights[:rows], kv_cache=cache,
                seq_lens=lengths[:rows], block_table=pages[:rows], output=output[:rows],
                shared_page_table=mode == "prefill",
            )
        b12x_indexer._run_paged_topk(
            module=dsa_indexer, plan=plan, q=q[:rows], weights=weights[:rows],
            kv_cache=cache, seq_lens=lengths[:rows], block_table=pages[:rows],
            active_width=indexer.active_width_cap, output=output[:rows], scores=None,
        )

    def expected(rows):
        logits = torch.einsum("rhd,kd->rhk", q[:rows].float(), decoded.float())
        scores = (logits.relu_() * weights[:rows, :, None]).sum(dim=1)
        return scores.topk(topk, dim=1).indices.add(0 if compressed else high_page * 64).to(torch.int32).sort(dim=1).values

    with PreparationSession(device=device, autotune=False) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        run(capacity)
        manager.lock()
        session.freeze()
        live_rows = 3 if mode == "decode" else 11
        with kernel_resolution_guard("DSA live rows within prepared capacity"):
            for rows in (1, live_rows, capacity):
                run(rows)
                torch.testing.assert_close(output[:rows].sort(dim=1).values, expected(rows), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    run(live_rows)
                q.copy_((-q.float()).to(q.dtype))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output[:live_rows].sort(dim=1).values, expected(live_rows), rtol=0, atol=0)
            finally:
                graph.reset()
