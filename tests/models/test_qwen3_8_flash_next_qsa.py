# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.models.qwen3_8_flash_next.nvidia.qsa as qsa_module
from vllm.models.qwen3_8_flash_next.common import qsa_cache as qsa_cache_module
from vllm.models.qwen3_8_flash_next.common.qsa_cache import (
    qsa_compressed_cache_view,
    qsa_compressed_slot_mapping,
    qsa_logical_positions,
    qsa_raw_slot_mapping,
)
from vllm.models.qwen3_8_flash_next.model_state import Qwen3_8FlashNextModelState
from vllm.models.qwen3_8_flash_next.nvidia.qsa import (
    Qwen3_8FlashNextQSAAttention,
    Qwen3_8FlashNextQSABackend,
    Qwen3_8FlashNextQSAImpl,
    Qwen3_8FlashNextQSAMetadata,
    Qwen3_8FlashNextQSAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import select_common_block_size


def test_qsa_backend_platform_probe_uses_b12x_selector_geometry(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def cache_requirements(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(compressed_page_nbytes=320)

    monkeypatch.setattr(
        qsa_cache_module,
        "get_b12x_qsa",
        lambda: SimpleNamespace(cache_requirements=cache_requirements),
    )
    probe = FullAttentionSpec(
        block_size=1,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )

    packed = Qwen3_8FlashNextQSABackend.customize_spec(probe)
    fp8_probe = FullAttentionSpec(
        block_size=1,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
    )
    fp8_packed = Qwen3_8FlashNextQSABackend.customize_spec(fp8_probe)

    assert packed.unpadded_page_size_bytes == 2048
    assert packed.page_size_padded == 2128
    assert packed.page_size_bytes == 2128
    assert fp8_packed.unpadded_page_size_bytes == 1024
    assert fp8_packed.page_size_padded == 1104
    assert fp8_packed.page_size_bytes == 1104
    assert calls == [
        {
            "main_page_size": 4,
            "kv_heads": 2,
            "head_dim": 256,
            "compress_ratio": 4,
            "index_head_dim": 128,
            "dtype": torch.bfloat16,
            "kv_dtype": torch.bfloat16,
        },
        {
            "main_page_size": 4,
            "kv_heads": 2,
            "head_dim": 256,
            "compress_ratio": 4,
            "index_head_dim": 128,
            "dtype": torch.bfloat16,
            "kv_dtype": torch.float8_e4m3fn,
        },
    ]


def test_qsa_backend_selects_the_manager_block_without_dense_page_limits() -> None:
    assert select_common_block_size(384, [Qwen3_8FlashNextQSABackend]) == 384
    assert select_common_block_size(512, [Qwen3_8FlashNextQSABackend]) == 512
    assert Qwen3_8FlashNextQSABackend.supports_block_size(384)
    assert Qwen3_8FlashNextQSABackend.supports_block_size(512)
    assert not Qwen3_8FlashNextQSABackend.supports_block_size(12)
    assert Qwen3_8FlashNextQSABackend.get_preferred_block_size(70) == 72


def test_qsa_selector_tail_is_zero_copy_in_block_outer_layer_pages() -> None:
    num_pages = 3
    num_layers = 2
    page_size = 8
    packed_kv_width = 512
    main_page_elements = 2 * page_size * packed_kv_width
    tail_elements = page_size // 4 * 128
    padded_page_elements = main_page_elements + tail_elements
    backing = torch.zeros(
        num_pages * num_layers * padded_page_elements,
        dtype=torch.bfloat16,
    )

    layer_views = [
        backing.as_strided(
            (num_pages, 2, page_size, packed_kv_width),
            (
                num_layers * padded_page_elements,
                page_size * packed_kv_width,
                packed_kv_width,
                1,
            ),
            storage_offset=layer * padded_page_elements,
        )
        for layer in range(num_layers)
    ]
    tails = [
        qsa_compressed_cache_view(
            view,
            compress_ratio=4,
            index_head_dim=128,
        )
        for view in layer_views
    ]
    tails[0].fill_(11)
    tails[1].fill_(22)

    assert torch.count_nonzero(layer_views[0]) == 0
    assert torch.count_nonzero(layer_views[1]) == 0
    assert torch.all(tails[0] == 11)
    assert torch.all(tails[1] == 22)
    assert tails[0].data_ptr() != tails[1].data_ptr()


def test_qsa_selector_tail_remains_bf16_with_fp8_main_cache() -> None:
    num_pages = 3
    page_size = 8
    packed_kv_width = 512
    main_page_nbytes = 2 * page_size * packed_kv_width
    tail_elements = page_size // 4 * 128
    tail_nbytes = tail_elements * torch.bfloat16.itemsize
    padded_page_nbytes = main_page_nbytes + tail_nbytes
    backing = torch.zeros(num_pages * padded_page_nbytes, dtype=torch.uint8)
    main_cache = backing.view(torch.float8_e4m3fn).as_strided(
        (num_pages, 2, page_size, packed_kv_width),
        (padded_page_nbytes, page_size * packed_kv_width, packed_kv_width, 1),
    )

    tail = qsa_compressed_cache_view(
        main_cache,
        compress_ratio=4,
        index_head_dim=128,
    )
    tail.fill_(7)

    assert tail.dtype == torch.bfloat16
    assert tail.stride(0) == padded_page_nbytes // torch.bfloat16.itemsize
    for page in range(num_pages):
        start = page * padded_page_nbytes
        assert torch.count_nonzero(backing[start : start + main_page_nbytes]) == 0
    assert torch.all(tail == 7)


def test_qsa_main_cache_views_reinterpret_fp8_storage() -> None:
    impl = Qwen3_8FlashNextQSAImpl.__new__(Qwen3_8FlashNextQSAImpl)
    impl.num_kv_heads = 1
    impl.head_size = 256
    impl.kv_cache_dtype = "fp8"
    storage = torch.empty(2, 2, 16, 256, dtype=torch.uint8)

    key_cache, value_cache = impl._kv_cache_views(storage)

    assert key_cache.dtype == current_platform.fp8_dtype()
    assert value_cache.dtype == current_platform.fp8_dtype()
    assert key_cache.shape == value_cache.shape == (2, 16, 1, 256)
    assert (
        key_cache.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    )


def test_qsa_bind_uses_shared_workspace_with_smaller_profile_cache(
    monkeypatch,
) -> None:
    actual_pages = 2
    page_size = 8
    max_seq_len = 40
    planned_pages = max_seq_len // page_size
    main_k_cache = torch.empty(
        actual_pages,
        page_size,
        1,
        256,
        dtype=torch.bfloat16,
    )
    main_v_cache = torch.empty_like(main_k_cache)
    compressed_cache = torch.empty(
        actual_pages,
        page_size // 4,
        128,
        dtype=torch.bfloat16,
    )
    bind_kwargs: dict[str, Any] = {}
    planned_caps: list[SimpleNamespace] = []
    shared_scratch = torch.empty(32, dtype=torch.uint8)
    binding = object()

    class FakePlan:
        def __init__(self, caps):
            self.caps = caps
            self.caps.main_table_width = math.ceil(
                caps.max_seq_len / caps.main_page_size
            )
            self.caps.compressed_table_width = math.ceil(
                (caps.max_seq_len // caps.compress_ratio) / caps.compressed_page_size
            )

        def scratch_specs(self):
            return (SimpleNamespace(shape=(32,), dtype=torch.uint8),)

        def bind(self, **kwargs):
            bind_kwargs.update(kwargs)
            return binding

    def plan(caps):
        planned_caps.append(caps)
        return FakePlan(caps)

    fake_qsa = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        is_supported=lambda: True,
        plan=plan,
    )
    monkeypatch.setattr(qsa_module, "get_b12x_qsa", lambda: fake_qsa)
    monkeypatch.setattr(
        qsa_module,
        "get_b12x_scratch_buffers",
        lambda _plan: [shared_scratch],
    )
    monkeypatch.setattr(
        qsa_module,
        "qsa_compressed_cache_view",
        lambda *_args, **_kwargs: compressed_cache,
    )

    owner = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.impl = SimpleNamespace(
        _kv_cache_views=lambda _cache: (main_k_cache, main_v_cache)
    )
    owner.max_tokens = 8
    owner.max_seqs = 2
    owner.max_seq_len = max_seq_len
    owner.max_speculative_tokens = 2
    owner.max_decode_rows = 6
    owner.compress_ratio = 4
    owner.raw_ring_capacity = 8
    owner.budget = 2048
    owner.num_heads = 6
    owner.num_kv_heads = 1
    owner.head_dim = 256
    owner.index_heads = 4
    owner.index_head_dim = 128
    owner.position_axes = 1
    owner.rotary_emb = SimpleNamespace(
        rotary_dim=64,
        cos_sin_cache=torch.empty(64, 64),
    )
    owner.indexer = SimpleNamespace(
        q_layernorm=SimpleNamespace(weight=torch.empty(128), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.empty(128)),
    )
    owner._raw_k_ring = torch.empty(2, 8, 128, dtype=torch.bfloat16)
    owner._raw_logical_positions = torch.empty(2, 8, dtype=torch.int64)
    owner._raw_rope_positions = torch.empty(2, 8, 1, dtype=torch.int64)
    owner._raw_interval_start_positions = torch.empty(2, dtype=torch.int64)
    owner._raw_state_slot_ids = torch.empty(2, dtype=torch.int32)
    owner._qsa_output = torch.empty(8, 6, 256, dtype=torch.bfloat16)
    owner._selected_positions = torch.empty(8, 2051, dtype=torch.int32)
    owner._k_scale = torch.ones(1, dtype=torch.float32)
    owner._v_scale = torch.ones(1, dtype=torch.float32)
    owner.kv_cache_torch_dtype = torch.bfloat16
    owner.kv_cache_kernel_dtype = torch.bfloat16

    kv_cache = torch.empty(
        actual_pages,
        2,
        page_size,
        256,
        dtype=torch.bfloat16,
    )
    owner.bind_kv_cache(kv_cache)

    assert len(planned_caps) == 2
    assert not bind_kwargs
    context = owner._qsa_decode_context
    assert context.plan.caps.max_q_rows == owner.max_decode_rows
    assert context.plan.caps.max_seq_len == max_seq_len
    assert owner._bind_qsa_context(context) is binding
    caps = planned_caps[0]
    assert caps.max_seq_len == max_seq_len
    assert caps.max_q_rows == owner.max_tokens
    assert caps.num_main_cache_pages == planned_pages
    assert caps.num_compressed_cache_pages == planned_pages
    assert owner._main_block_table.shape == (owner.max_seqs, planned_pages)
    assert bind_kwargs["main_k_cache"] is main_k_cache
    assert bind_kwargs["main_v_cache"] is main_v_cache
    assert bind_kwargs["k_descale"] is owner._k_scale
    assert bind_kwargs["v_descale"] is owner._v_scale
    assert bind_kwargs["compressed_k_cache"] is compressed_cache
    assert main_k_cache.shape[0] < caps.num_main_cache_pages
    assert compressed_cache.shape[0] < caps.num_compressed_cache_pages
    assert not hasattr(owner, "_qsa_binding")
    assert bind_kwargs["scratch"].data_ptr() == shared_scratch.data_ptr()
    assert bind_kwargs["output"].shape[0] == owner.max_decode_rows
    assert bind_kwargs["selected_positions"].shape[0] == owner.max_decode_rows
    bind_kwargs.clear()
    assert owner._bind_qsa_context(context) is binding
    assert bind_kwargs["main_k_cache"] is main_k_cache
    caller_output = torch.empty(1, 6, 256, dtype=torch.bfloat16)
    assert owner._bind_qsa_context(context, output=caller_output) is binding
    assert bind_kwargs["output"] is caller_output

    owner.unbind_kv_cache()

    assert owner.kv_cache.numel() == 0
    assert owner._main_block_table is None
    assert owner._compressed_cache is None
    assert owner._qsa_plan is None
    assert owner._qsa_decode_context is None
    assert owner._qsa_prefill_bindings == ()
    assert owner._qsa_scratch is None

    replacement_cache = torch.empty_like(kv_cache)
    owner.bind_kv_cache(replacement_cache)

    assert owner.kv_cache is replacement_cache
    assert len(planned_caps) == 4
    assert owner._qsa_plan is not None
    assert owner._qsa_decode_context is not context
    assert owner._bind_qsa_context(owner._qsa_decode_context) is binding
    assert owner._qsa_scratch is not None


def test_qsa_registers_piecewise_splitting_op_once() -> None:
    compilation_config = SimpleNamespace(
        static_forward_context={},
        splitting_ops=[],
    )

    qsa_module._register_qsa_compilation_context(
        compilation_config,
        "model.layers.3.attn",
        object(),
    )
    qsa_module._register_qsa_compilation_context(
        compilation_config,
        "model.layers.7.attn",
        object(),
    )

    assert compilation_config.splitting_ops == [
        qsa_module._QSA_SPLITTING_OP,
        qsa_module._QSA_PROJECTED_READ_OP,
    ]
    assert set(compilation_config.static_forward_context) == {
        "model.layers.3.attn",
        "model.layers.7.attn",
    }


@pytest.mark.parametrize(
    "rows,query_len,mode,capturing,eligible",
    [
        (4, 4, "FULL", True, True),
        (4, 4, "NONE", True, True),
        (4, 4, "PIECEWISE", True, False),
        (4, 4, "FULL", False, False),
        (32, 4, "FULL", True, False),
        (4, 5, "FULL", True, False),
    ],
)
def test_qsa_selector_fork_requires_one_full_graph(
    monkeypatch,
    rows,
    query_len,
    mode,
    capturing,
    eligible,
) -> None:
    """A selector fork must not escape a piecewise graph before its join."""
    layer = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.overlap_input_projections = True
    layer.max_decode_rows = 16
    layer.max_speculative_tokens = 3
    layer.layer_name = "qsa"
    metadata = Qwen3_8FlashNextQSAMetadata.__new__(Qwen3_8FlashNextQSAMetadata)
    metadata.num_actual_tokens = 4
    metadata.max_query_len = query_len
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        qsa_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=getattr(qsa_module.CUDAGraphMode, mode),
            attn_metadata={"qsa": metadata},
        ),
    )
    assert (layer._parallel_selector_metadata(rows) is metadata) == eligible


def test_qsa_prefill_context_capacities_cover_the_configured_limit() -> None:
    assert qsa_module._qsa_prefill_context_capacities(262144, 4096) == (
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
    )
    assert qsa_module._qsa_prefill_context_capacities(40000, 32768) == (
        32768,
        40000,
    )
    assert qsa_module._qsa_prefill_context_capacities(262144, 6019) == (
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
    )


@pytest.mark.parametrize("max_tokens", [4, 10])
def test_qsa_warmup_runs_every_context_with_only_padded_requests(
    monkeypatch, max_tokens
) -> None:
    caps = SimpleNamespace(
        device=torch.device("cpu"),
        q_heads=2,
        kv_heads=1,
        head_dim=16,
        index_heads=2,
        index_head_dim=8,
        position_axes=3,
        max_batch=2,
        main_page_size=16,
        selection_width=8,
        kv_dtype=torch.bfloat16,
    )
    contexts = tuple(
        SimpleNamespace(
            max_seq_len=capacity,
            plan=SimpleNamespace(caps=caps),
        )
        for capacity in (32, 64)
    )
    layer = SimpleNamespace(
        _qsa_prefill_bindings=contexts,
        _qsa_decode_context=SimpleNamespace(
            plan=SimpleNamespace(caps=SimpleNamespace(max_q_rows=6))
        ),
        _bind_qsa_context=lambda context: context,
        max_tokens=max_tokens,
        max_speculative_tokens=2,
        max_decode_rows=6,
    )
    calls = []

    def run(binding, **inputs):
        rows = inputs["query"].shape[0]
        calls.append((binding, rows))
        assert inputs["query"].shape == (rows, 2, 16)
        assert inputs["index_query"].shape == (rows, 2, 8)
        assert inputs["raw_index_key"].shape == (rows, 8)
        assert inputs["rope_positions"].shape == (rows, 3)
        assert (inputs["request_ids"] == -1).all()
        assert (inputs["query_positions"] == -1).all()
        assert not inputs["sequence_lengths"].any()
        assert not inputs["query_start_loc"].any()

    monkeypatch.setattr(qsa_module, "get_b12x_qsa", lambda: SimpleNamespace(run=run))
    unit = qsa_module._B12xQSAWarmup().get_b12x_warmup_unit(
        layer, (1, 4, 8), torch.bfloat16
    )
    unit.compile()
    assert calls == [
        *((context, max_tokens - 2) for context in contexts),
        (layer._qsa_decode_context, 1),
        (layer._qsa_decode_context, 4),
    ]


def test_qsa_run_consumes_projection_views_and_writes_live_output(monkeypatch) -> None:
    rows = 2
    query = torch.randn(rows, 6, 256, dtype=torch.bfloat16)
    projection = torch.randn(rows, 640, dtype=torch.bfloat16)
    index_query = projection[:, :512].view(rows, 4, 128)
    raw_index_key = projection[:, 512:]
    output = torch.full((4, 6, 256), 23, dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)
    staged = SimpleNamespace(
        request_ids=torch.zeros(rows, dtype=torch.int32),
        logical_positions=positions,
        sequence_lengths=torch.tensor([rows]),
        query_start_loc=torch.tensor([0, rows]),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        is_prefilling=torch.ones(1, dtype=torch.bool),
    )
    context = SimpleNamespace(main_block_table=None, compressed_block_table=None)
    calls = []

    def run(binding, **inputs):
        calls.append(inputs)
        for name, original in (
            ("query", query),
            ("index_query", index_query),
            ("raw_index_key", raw_index_key),
        ):
            assert inputs[name].data_ptr() == original.data_ptr()
            assert inputs[name].stride() == original.stride()
        assert binding.output.data_ptr() == output.data_ptr()
        assert binding.output.shape[0] == rows
        binding.output.fill_(7)

    owner = SimpleNamespace(
        _qsa_binding_for_workload=lambda **_: context,
        _prepare_qsa_metadata=lambda *_: staged,
        _shared_qsa_rope_positions=lambda *_: positions,
        _bind_qsa_context=lambda _context, _staged, out: SimpleNamespace(output=out),
        _record_mtp_selection_metadata=lambda *_: None,
        impl=SimpleNamespace(do_kv_cache_update=lambda *_: None),
        kv_cache=None,
    )
    monkeypatch.setattr(qsa_module, "get_b12x_qsa", lambda: SimpleNamespace(run=run))
    Qwen3_8FlashNextQSAAttention._run_b12x_qsa(
        owner,
        metadata=SimpleNamespace(max_seq_len=rows, slot_mapping=positions),
        positions=positions,
        query=query,
        key=query,
        value=query,
        index_query=index_query,
        raw_index_key=raw_index_key,
        output=output,
        rows=rows,
    )
    assert len(calls) == 1
    assert torch.all(output[:rows] == 7)
    assert torch.count_nonzero(output[rows:]) == 0


def test_qsa_selects_the_smallest_sufficient_prefill_context_plan() -> None:
    owner = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    owner.max_decode_rows = 6
    owner.max_seq_len = 63
    binding_32 = object()
    binding_64 = object()
    decode_plan = object()
    scratch = torch.empty(0)
    table_32 = torch.empty(2, 4, dtype=torch.int32)
    table_64 = torch.empty(2, 8, dtype=torch.int32)
    owner._qsa_prefill_bindings = (
        qsa_module._QSAContextPlan(32, binding_32, scratch, table_32, table_32),
        qsa_module._QSAContextPlan(64, binding_64, scratch, table_64, table_64),
    )
    owner._qsa_decode_context = qsa_module._QSAContextPlan(
        64, decode_plan, scratch, table_64, table_64
    )

    assert owner._qsa_binding_for_workload(rows=1, max_seq_len=20).plan is decode_plan
    assert owner._qsa_binding_for_workload(rows=7, max_seq_len=20).plan is binding_32
    assert owner._qsa_binding_for_workload(rows=7, max_seq_len=40).plan is binding_64
    with pytest.raises(ValueError, match="exceeds the configured limit"):
        owner._qsa_binding_for_workload(rows=7, max_seq_len=64)
    with pytest.raises(ValueError, match="exceeds the configured limit"):
        owner._qsa_binding_for_workload(rows=1, max_seq_len=64)


def test_qsa_prefill_dispatches_through_the_b12x_transaction(monkeypatch) -> None:
    rows = 8
    metadata = Qwen3_8FlashNextQSAMetadata(
        num_actual_tokens=rows,
        max_query_len=rows,
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        max_seq_len=rows,
        seq_lens=torch.tensor([rows], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(rows, dtype=torch.int64),
        is_prefilling=torch.ones(1, dtype=torch.bool),
    )
    owner = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.layer_name = "model.layers.0.attn"
    owner.kv_cache = torch.ones(1)
    calls: list[dict[str, Any]] = []
    owner._run_b12x_qsa = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setattr(
        qsa_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={owner.layer_name: metadata}),
    )
    positions = torch.arange(rows, dtype=torch.int64)
    query = torch.empty(rows, 12, 256)
    key = torch.empty(rows, 1, 256)
    value = torch.empty_like(key)
    index_query = torch.empty(rows, 4, 128)
    raw_index_key = torch.empty(rows, 128)
    output = torch.empty_like(query)

    owner._run_qsa(
        positions,
        query,
        key,
        value,
        index_query,
        raw_index_key,
        output,
    )

    assert len(calls) == 1
    assert calls[0]["metadata"] is metadata
    assert calls[0]["rows"] == rows


def test_qsa_logical_positions_mark_graph_padding_invalid() -> None:
    positions = qsa_logical_positions(
        sequence_lengths=torch.tensor([5, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32),
        request_ids=torch.tensor([0, 0, -1, -1], dtype=torch.int32),
    )

    assert torch.equal(positions, torch.tensor([3, 4, -1, -1]))


def test_qsa_raw_slot_mapping_marks_graph_padding_invalid() -> None:
    slots = qsa_raw_slot_mapping(
        state_slot_ids=torch.tensor([7, -1], dtype=torch.int32),
        request_ids=torch.tensor([0, -1, 1], dtype=torch.int32),
        logical_positions=torch.tensor([9, -1, 3], dtype=torch.int64),
        raw_ring_capacity=8,
    )

    assert slots.dtype == torch.int64
    assert torch.equal(slots, torch.tensor([57, -1, -1]))


def test_qsa_compressed_slot_mapping_keeps_pool_offsets_in_int64() -> None:
    high_page = 134_217_729
    slots = qsa_compressed_slot_mapping(
        block_table=torch.tensor([[high_page]], dtype=torch.int32),
        request_ids=torch.tensor([0], dtype=torch.int32),
        logical_positions=torch.tensor([3], dtype=torch.int64),
        main_page_size=64,
        compress_ratio=4,
    )

    assert slots.dtype == torch.int64
    assert int(slots[0]) == high_page * 16
    assert int(slots[0]) > torch.iinfo(torch.int32).max


def _bare_qwen_model_state_for_draft_metadata() -> Qwen3_8FlashNextModelState:
    state = Qwen3_8FlashNextModelState.__new__(Qwen3_8FlashNextModelState)
    state.max_num_reqs = 8
    state.uses_qsa = True
    state.qsa_state_slot_ids = torch.arange(8, dtype=torch.int32)
    state._qsa_default_slot_ids = state.qsa_state_slot_ids.clone()
    state.qsa_state_is_fresh = torch.ones(8, dtype=torch.bool)
    state.qsa_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.mamba_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.num_accepted_tokens_gpu = torch.ones(8, dtype=torch.int32)
    state.qsa_committed_num_accepted_tokens_gpu = torch.ones(8, dtype=torch.int32)
    state._qsa_draft_is_prefilling = torch.zeros(8, dtype=torch.bool)
    state._qsa_draft_is_prefilling_gpu = torch.zeros(8, dtype=torch.bool)
    return state


def test_qsa_acceptance_survives_mamba_state_page_alignment_reset() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_state_is_fresh_gpu = torch.zeros(8, dtype=torch.bool)
    state.qsa_committed_num_accepted_tokens_gpu[7] = 3
    state.num_accepted_tokens_gpu[7] = 1

    _, _, accepted = state._prepare_qsa_state(
        SimpleNamespace(
            num_reqs=1,
            idx_mapping=torch.tensor([7], dtype=torch.int32),
        ),
        num_reqs=1,
    )

    assert torch.equal(accepted, torch.tensor([3], dtype=torch.int32))


def test_qsa_and_mamba_use_independent_acceptance_after_page_alignment() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_state_is_fresh_gpu = torch.zeros(8, dtype=torch.bool)
    state.qsa_committed_num_accepted_tokens_gpu[7] = 4
    state.num_accepted_tokens_gpu[7] = 1
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([7], dtype=torch.int32),
    )

    _, _, qsa_accepted = state._prepare_qsa_state(input_batch, num_reqs=1)
    mamba_accepted = state._prepare_mamba_acceptance(input_batch, num_reqs=1)

    assert torch.equal(qsa_accepted, torch.tensor([4], dtype=torch.int32))
    assert torch.equal(mamba_accepted, torch.tensor([1], dtype=torch.int32))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_qsa_postprocess_commits_acceptance_before_mamba_alignment_reset() -> None:
    class ResetAcceptedTokens:
        def run_fused_postprocess_align(
            self,
            num_reqs: int,
            num_accepted_tokens_gpu: torch.Tensor,
            state_idx_gpu: torch.Tensor,
            num_computed_tokens: torch.Tensor,
            idx_mapping: torch.Tensor,
        ) -> None:
            num_accepted_tokens_gpu.fill_(1)

    state = Qwen3_8FlashNextModelState.__new__(Qwen3_8FlashNextModelState)
    state.uses_qsa = True
    state.qsa_committed_num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state.qsa_state_is_fresh_gpu = torch.ones(5, dtype=torch.bool, device="cuda")
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = True
    state._mamba_ctx = ResetAcceptedTokens()
    state._mamba_state_idx_gpu = torch.zeros(5, dtype=torch.int32, device="cuda")
    state.recoverssm = None

    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int32, device="cuda")
    num_sampled = torch.tensor([4, 2, 3], dtype=torch.int32, device="cuda")
    num_computed_tokens = torch.zeros(5, dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    assert state.num_accepted_tokens_gpu.tolist() == [1, 1, 1, 1, 1]
    assert state.qsa_committed_num_accepted_tokens_gpu.tolist() == [9, 3, 9, 4, 9]
    assert state.qsa_state_is_fresh_gpu.tolist() == [True, False, True, False, True]


def test_qsa_draft_metadata_uses_persistent_slot_and_safe_padding() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_committed_num_accepted_tokens_gpu[7] = 3

    metadata = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([7, -1, -1, -1], dtype=torch.int32),
        num_reqs=1,
        num_reqs_padded=4,
        draft_index=1,
    )

    assert metadata is not None
    assert torch.equal(
        metadata.qsa_state_slot_ids,
        torch.tensor([7, 1, 2, 3], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.qsa_state_is_fresh,
        torch.tensor([False, True, True, True]),
    )
    assert torch.equal(
        metadata.qsa_num_accepted_tokens,
        torch.tensor([3, 1, 1, 1], dtype=torch.int32),
    )
    assert not torch.any(metadata.is_prefilling)
    assert (
        metadata.qsa_is_prefilling.data_ptr()
        == state._qsa_draft_is_prefilling_gpu.data_ptr()
    )


def test_qsa_draft_metadata_tracks_batch_reordering_by_persistent_slot() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_committed_num_accepted_tokens_gpu[7] = 3

    first = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([7, 3], dtype=torch.int32),
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=1,
    )
    assert first is not None
    slot_ptr = first.qsa_state_slot_ids.data_ptr()
    fresh_ptr = first.qsa_state_is_fresh.data_ptr()
    accepted_ptr = first.qsa_num_accepted_tokens.data_ptr()
    assert torch.equal(
        first.qsa_state_slot_ids,
        torch.tensor([7, 3, 2, 3], dtype=torch.int32),
    )
    assert torch.equal(
        first.qsa_num_accepted_tokens,
        torch.tensor([3, 1, 1, 1], dtype=torch.int32),
    )

    reordered = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([3, 7], dtype=torch.int32),
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=1,
    )
    assert reordered is not None
    assert reordered.qsa_state_slot_ids.data_ptr() == slot_ptr
    assert reordered.qsa_state_is_fresh.data_ptr() == fresh_ptr
    assert reordered.qsa_num_accepted_tokens.data_ptr() == accepted_ptr
    assert torch.equal(
        reordered.qsa_state_slot_ids,
        torch.tensor([3, 7, 2, 3], dtype=torch.int32),
    )
    assert torch.equal(
        reordered.qsa_state_is_fresh,
        torch.tensor([False, False, True, True]),
    )
    assert torch.equal(
        reordered.qsa_num_accepted_tokens,
        torch.tensor([1, 3, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        state._qsa_default_slot_ids,
        torch.arange(8, dtype=torch.int32),
    )


def test_qsa_draft_metadata_uses_one_accepted_token_after_first_lookahead() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_committed_num_accepted_tokens_gpu[7] = 3

    metadata = state.prepare_draft_attn_metadata(
        idx_mapping=torch.tensor([7, 3], dtype=torch.int32),
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=2,
    )

    assert metadata is not None
    assert torch.equal(
        metadata.qsa_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )


def test_qsa_draft_metadata_rejects_non_lookahead_step() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()

    with pytest.raises(RuntimeError, match="draft_index >= 1"):
        state.prepare_draft_attn_metadata(
            idx_mapping=torch.tensor([7], dtype=torch.int32),
            num_reqs=1,
            num_reqs_padded=1,
            draft_index=0,
        )


def test_non_qsa_draft_metadata_is_a_noop_at_step_zero() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.uses_qsa = False

    assert (
        state.prepare_draft_attn_metadata(
            idx_mapping=torch.tensor([7], dtype=torch.int32),
            num_reqs=1,
            num_reqs_padded=1,
            draft_index=0,
        )
        is None
    )


def test_qsa_mtp_metadata_preserves_previous_acceptance_until_first_lookahead() -> None:
    state = _bare_qwen_model_state_for_draft_metadata()
    state.qsa_state_is_fresh_gpu = torch.zeros(8, dtype=torch.bool)
    state.qsa_committed_num_accepted_tokens_gpu[3] = 2
    state.qsa_committed_num_accepted_tokens_gpu[7] = 4
    idx_mapping = torch.tensor([7, 3, -1, -1], dtype=torch.int32)

    def make_builder() -> Qwen3_8FlashNextQSAMetadataBuilder:
        builder = Qwen3_8FlashNextQSAMetadataBuilder.__new__(
            Qwen3_8FlashNextQSAMetadataBuilder
        )
        builder._request_ids = torch.empty(4, dtype=torch.int32)
        builder.max_speculative_tokens = 4
        builder._capture_state_slot_ids = torch.arange(4, dtype=torch.int32)
        builder._capture_state_is_fresh = torch.ones(4, dtype=torch.bool)
        builder._capture_num_accepted_tokens = torch.ones(4, dtype=torch.int32)
        builder._capture_is_prefilling = torch.zeros(4, dtype=torch.bool)
        return builder

    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=4,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=torch.zeros((4, 1), dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        is_prefilling=torch.zeros(4, dtype=torch.bool),
    )
    target_builder = make_builder()
    draft_builder = make_builder()

    target_slots, target_fresh, target_accepted = state._prepare_qsa_state(
        SimpleNamespace(num_reqs=2, idx_mapping=idx_mapping),
        num_reqs=4,
    )
    reused_target_metadata = target_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=target_slots,
        qsa_state_is_fresh=target_fresh,
        qsa_num_accepted_tokens=target_accepted,
    )
    previous_accepted = torch.tensor([4, 2, 1, 1], dtype=torch.int32)
    assert torch.equal(
        reused_target_metadata.qsa_num_accepted_tokens,
        previous_accepted,
    )

    state.qsa_committed_num_accepted_tokens_gpu[3] = 3
    state.qsa_committed_num_accepted_tokens_gpu[7] = 2
    assert torch.equal(
        reused_target_metadata.qsa_num_accepted_tokens,
        previous_accepted,
    )

    first_lookahead = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=1,
    )
    assert first_lookahead is not None
    first_lookahead_metadata = draft_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=first_lookahead.qsa_state_slot_ids,
        qsa_state_is_fresh=first_lookahead.qsa_state_is_fresh,
        qsa_num_accepted_tokens=first_lookahead.qsa_num_accepted_tokens,
    )
    current_accepted = torch.tensor([2, 3, 1, 1], dtype=torch.int32)
    assert torch.equal(
        first_lookahead_metadata.qsa_num_accepted_tokens,
        current_accepted,
    )
    assert torch.equal(
        reused_target_metadata.qsa_num_accepted_tokens,
        previous_accepted,
    )

    later_lookahead = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=2,
    )
    assert later_lookahead is not None
    later_lookahead_metadata = draft_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=later_lookahead.qsa_state_slot_ids,
        qsa_state_is_fresh=later_lookahead.qsa_state_is_fresh,
        qsa_num_accepted_tokens=later_lookahead.qsa_num_accepted_tokens,
    )
    assert torch.equal(
        later_lookahead_metadata.qsa_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )
    assert torch.equal(
        reused_target_metadata.qsa_num_accepted_tokens,
        previous_accepted,
    )


def test_qsa_builder_stages_runtime_state_in_capture_buffers() -> None:
    builder = Qwen3_8FlashNextQSAMetadataBuilder.__new__(
        Qwen3_8FlashNextQSAMetadataBuilder
    )
    builder._request_ids = torch.empty(4, dtype=torch.int32)
    builder.max_speculative_tokens = 2
    builder._capture_state_slot_ids = torch.arange(4, dtype=torch.int32)
    builder._capture_state_is_fresh = torch.ones(4, dtype=torch.bool)
    builder._capture_num_accepted_tokens = torch.ones(4, dtype=torch.int32)
    builder._capture_is_prefilling = torch.zeros(4, dtype=torch.bool)
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1, 0, 0, 0], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=4,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=torch.zeros((4, 1), dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        is_prefilling=torch.zeros(4, dtype=torch.bool),
    )
    captured = builder.build_for_cudagraph_capture(common)
    slot_ptr = captured.qsa_state_slot_ids.data_ptr()
    fresh_ptr = captured.qsa_state_is_fresh.data_ptr()
    accepted_ptr = captured.qsa_num_accepted_tokens.data_ptr()
    prefill_ptr = captured.is_prefilling.data_ptr()

    runtime_slots = torch.tensor([7, 1, 2, 3], dtype=torch.int32)
    runtime_fresh = torch.tensor([False, True, True, True])
    runtime_accepted = torch.ones(4, dtype=torch.int32)
    runtime_is_prefilling = torch.tensor([False, True, False, False])
    runtime = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=runtime_slots,
        qsa_state_is_fresh=runtime_fresh,
        qsa_num_accepted_tokens=runtime_accepted,
        qsa_is_prefilling=runtime_is_prefilling,
    )

    assert captured.qsa_state_slot_ids.data_ptr() == slot_ptr
    assert captured.qsa_state_is_fresh.data_ptr() == fresh_ptr
    assert captured.qsa_num_accepted_tokens.data_ptr() == accepted_ptr
    assert runtime.qsa_state_slot_ids.data_ptr() == slot_ptr
    assert runtime.qsa_state_is_fresh.data_ptr() == fresh_ptr
    assert runtime.qsa_num_accepted_tokens.data_ptr() == accepted_ptr
    assert runtime.is_prefilling.data_ptr() == prefill_ptr
    assert torch.equal(captured.qsa_state_slot_ids, runtime_slots)
    assert torch.equal(captured.qsa_state_is_fresh, runtime_fresh)
    assert torch.equal(captured.qsa_num_accepted_tokens, runtime_accepted)
    assert torch.equal(captured.is_prefilling, runtime_is_prefilling)

    runtime_slots.fill_(99)
    runtime_fresh.fill_(False)
    runtime_accepted.fill_(99)
    assert torch.equal(
        captured.qsa_state_slot_ids,
        torch.tensor([7, 1, 2, 3], dtype=torch.int32),
    )
    assert torch.equal(
        captured.qsa_state_is_fresh,
        torch.tensor([False, True, True, True]),
    )
    assert torch.equal(
        captured.qsa_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )

    reordered_slots = torch.tensor([3, 7, 2, 3], dtype=torch.int32)
    reordered_fresh = torch.tensor([False, False, True, True])
    reordered_accepted = torch.tensor([1, 3, 1, 1], dtype=torch.int32)
    reordered_is_prefilling = torch.tensor([True, False, False, False])
    reordered = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=reordered_slots,
        qsa_state_is_fresh=reordered_fresh,
        qsa_num_accepted_tokens=reordered_accepted,
        qsa_is_prefilling=reordered_is_prefilling,
    )

    assert reordered.qsa_state_slot_ids.data_ptr() == slot_ptr
    assert reordered.qsa_state_is_fresh.data_ptr() == fresh_ptr
    assert reordered.qsa_num_accepted_tokens.data_ptr() == accepted_ptr
    assert reordered.is_prefilling.data_ptr() == prefill_ptr
    assert torch.equal(captured.qsa_state_slot_ids, reordered_slots)
    assert torch.equal(captured.qsa_state_is_fresh, reordered_fresh)
    assert torch.equal(captured.qsa_num_accepted_tokens, reordered_accepted)
    assert torch.equal(
        captured.is_prefilling,
        torch.tensor([True, False, False, False]),
    )


def test_qsa_cache_group_rebinding_preserves_live_metadata_owner() -> None:
    def make_builder() -> Qwen3_8FlashNextQSAMetadataBuilder:
        builder = Qwen3_8FlashNextQSAMetadataBuilder.__new__(
            Qwen3_8FlashNextQSAMetadataBuilder
        )
        builder._request_ids = torch.empty(4, dtype=torch.int32)
        builder.max_speculative_tokens = 2
        builder._capture_state_slot_ids = torch.arange(4, dtype=torch.int32)
        builder._capture_state_is_fresh = torch.ones(4, dtype=torch.bool)
        builder._capture_num_accepted_tokens = torch.ones(4, dtype=torch.int32)
        builder._capture_is_prefilling = torch.zeros(4, dtype=torch.bool)
        return builder

    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9, 0, 0], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=4,
        max_query_len=1,
        max_seq_len=9,
        block_table_tensor=torch.zeros((4, 1), dtype=torch.int32),
        slot_mapping=torch.full((4,), -1, dtype=torch.int64),
        is_prefilling=torch.tensor([False, True, False, False]),
    )
    builder_a = make_builder()
    builder_b = make_builder()
    metadata_a = builder_a.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=torch.tensor([7, 3, 2, 1], dtype=torch.int32),
        qsa_state_is_fresh=torch.tensor([False, True, True, True]),
        qsa_num_accepted_tokens=torch.tensor([2, 1, 1, 1], dtype=torch.int32),
    )
    block_table_b = torch.ones((4, 1), dtype=torch.int32)
    slot_mapping_b = torch.arange(4, dtype=torch.int64)

    metadata_b = builder_b.update_block_table(
        metadata_a,
        block_table_b,
        slot_mapping_b,
    )

    assert metadata_b.block_table is block_table_b
    assert metadata_b.slot_mapping is slot_mapping_b
    assert metadata_b.request_ids.data_ptr() == builder_a._request_ids.data_ptr()
    assert (
        metadata_b.qsa_state_slot_ids.data_ptr()
        == builder_a._capture_state_slot_ids.data_ptr()
    )
    assert (
        metadata_b.qsa_state_is_fresh.data_ptr()
        == builder_a._capture_state_is_fresh.data_ptr()
    )
    assert (
        metadata_b.qsa_num_accepted_tokens.data_ptr()
        == builder_a._capture_num_accepted_tokens.data_ptr()
    )
    assert (
        metadata_b.is_prefilling.data_ptr()
        == builder_a._capture_is_prefilling.data_ptr()
    )
    torch.testing.assert_close(
        metadata_b.qsa_state_slot_ids,
        metadata_a.qsa_state_slot_ids,
    )
    torch.testing.assert_close(
        metadata_b.qsa_state_is_fresh,
        metadata_a.qsa_state_is_fresh,
    )
    torch.testing.assert_close(
        metadata_b.qsa_num_accepted_tokens,
        metadata_a.qsa_num_accepted_tokens,
    )
    updated = builder_a.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        qsa_state_slot_ids=torch.tensor([3, 7, 2, 1], dtype=torch.int32),
        qsa_state_is_fresh=torch.tensor([False, False, True, True]),
        qsa_num_accepted_tokens=torch.tensor([1, 3, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata_b.qsa_state_slot_ids, updated.qsa_state_slot_ids
    )
    torch.testing.assert_close(
        metadata_b.qsa_num_accepted_tokens, updated.qsa_num_accepted_tokens
    )


def test_qsa_builder_updates_fused_draft_acceptance_in_place() -> None:
    builder = Qwen3_8FlashNextQSAMetadataBuilder.__new__(
        Qwen3_8FlashNextQSAMetadataBuilder
    )
    accepted = torch.tensor([4, 2, 1, 1], dtype=torch.int32)
    accepted_ptr = accepted.data_ptr()
    metadata = SimpleNamespace(qsa_num_accepted_tokens=accepted)

    assert builder.supports_draft_decode_metadata_update

    builder.update_draft_decode_metadata(metadata)

    assert metadata.qsa_num_accepted_tokens.data_ptr() == accepted_ptr
    assert torch.equal(
        metadata.qsa_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )


def test_qsa_speculative_anchor_snapshot_restores_all_persistent_slots() -> None:
    owner = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.max_seqs = 8
    owner._raw_interval_start_positions = torch.tensor(
        [-1, 7, 19, 31, 43, 55, 67, 79], dtype=torch.int64
    )
    owner._raw_interval_start_snapshot = torch.empty_like(
        owner._raw_interval_start_positions
    )
    anchor_ptr = owner._raw_interval_start_positions.data_ptr()
    snapshot_ptr = owner._raw_interval_start_snapshot.data_ptr()

    owner.snapshot_speculative_interval_starts()
    owner._raw_interval_start_positions.add_(100)
    owner.restore_speculative_interval_starts()

    assert owner._raw_interval_start_positions.data_ptr() == anchor_ptr
    assert owner._raw_interval_start_snapshot.data_ptr() == snapshot_ptr
    assert torch.equal(
        owner._raw_interval_start_positions,
        torch.tensor([-1, 7, 19, 31, 43, 55, 67, 79], dtype=torch.int64),
    )


def _require_qsa_gpu() -> torch.device:
    if os.environ.get("B12X_QSA_GPU_TEST") != "1":
        pytest.skip("set B12X_QSA_GPU_TEST=1 to run QSA GPU tests")
    if not torch.accelerator.is_available():
        pytest.skip("QSA GPU tests require CUDA")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    if current_platform.get_device_capability(device.index or 0) not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("QSA GPU tests require SM120 or SM121")
    return device


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64])
def test_mtp_selection_capture_reorders_errors_and_replaces_causal_tail_on_replay(
    source_dtype: torch.dtype,
) -> None:
    """A draft round owns its anchor selection; padding cannot inherit errors."""
    device = _require_qsa_gpu()
    layer = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
    torch.nn.Module.__init__(layer)
    layer._share_mtp_indices = True
    layer.max_tokens, layer.max_seqs, layer.max_speculative_tokens = 16, 4, 3
    layer._native_selection_width = 2051
    layer._selected_positions = torch.arange(
        16 * 2051, device=device, dtype=torch.int32
    ).view(16, 2051)
    layer._mtp_source_positions = (
        torch.arange(16, device=device, dtype=torch.int64) + 63
    )
    layer._mtp_source_errors = torch.zeros(16, device=device, dtype=torch.int32)
    layer._mtp_source_errors[7] = 512
    layer._mtp_shared_selected_positions = torch.full(
        (4, 2054), -1, device=device, dtype=torch.int32
    )
    layer._mtp_captured_lengths = torch.empty(4, device=device, dtype=torch.int64)
    layer._mtp_captured_errors = torch.empty(4, device=device, dtype=torch.int32)
    layer._mtp_read_errors = torch.empty(4, device=device, dtype=torch.int32)
    source_rows = torch.tensor([7, 3, 0, -1], device=device, dtype=source_dtype)
    request_ids = torch.tensor([0, 1, -1, -1], device=device, dtype=torch.int32)
    positions = torch.tensor([72, 67, -1, -1], device=device, dtype=torch.int64)

    def capture_and_read():
        layer.compact_topk_indices(source_rows)
        layer._append_mtp_selection_tail(positions, request_ids)

    capture_and_read()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_and_read()
    for shift in (0, 100):
        layer._selected_positions.add_(shift)
        graph.replay()
        torch.testing.assert_close(
            layer._mtp_shared_selected_positions[:3, :2051],
            layer._selected_positions[source_rows[:3].long()],
            rtol=0,
            atol=0,
        )
        assert layer._mtp_shared_selected_positions[:, 2051:].tolist() == [
            [71, 72, -1],
            [67, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
        assert layer._mtp_read_errors.tolist() == [512, 0, 0, 0]
        # Rejection can shorten a speculative tail; no future column may survive.
        immutable = layer._mtp_shared_selected_positions[:, :2051].clone()
        positions[0] = 71
        layer._append_mtp_selection_tail(positions, request_ids)
        assert layer._mtp_shared_selected_positions[0, 2051:].tolist() == [71, -1, -1]
        assert torch.equal(layer._mtp_shared_selected_positions[:, :2051], immutable)
        positions[0] = 75
        layer._append_mtp_selection_tail(positions, request_ids)
        assert layer._mtp_read_errors[0].item() != 0
        positions[0] = 72


def test_qsa_rope_staging_masks_graph_padding() -> None:
    device = _require_qsa_gpu()
    rows = 12
    source = torch.arange(3 * rows, dtype=torch.int64, device=device).view(3, rows).t()
    assert not source.is_contiguous()
    request_ids = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, -1, -1, -1, -1],
        dtype=torch.int32,
        device=device,
    )
    output = torch.full((rows, 3), 12345, dtype=torch.int64, device=device)
    output_ptr = output.data_ptr()

    qsa_module._stage_qsa_rope_positions_kernel[(rows,)](
        source,
        request_ids,
        output,
        source.stride(0),
        source.stride(1),
        output.stride(0),
        rows,
        POSITION_AXES=3,
        num_warps=1,
    )

    assert output.data_ptr() == output_ptr
    assert torch.equal(output[:8], source[:8])
    assert torch.equal(output[8:], torch.full_like(output[8:], -1))


@pytest.mark.parametrize("max_seqs", [2, 16])
def test_qsa_staging_shares_live_request_state_but_not_group_page_tables(
    monkeypatch,
    max_seqs: int,
) -> None:
    """Shared staging must refresh under replay and preserve group-specific pages."""
    device = _require_qsa_gpu()
    context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(qsa_module, "get_forward_context", lambda: context)

    def owner():
        layer = Qwen3_8FlashNextQSAAttention.__new__(Qwen3_8FlashNextQSAAttention)
        torch.nn.Module.__init__(layer)
        layer.max_seqs = max_seqs
        layer.max_speculative_tokens, layer.position_axes = 3, 1
        for name, shape, dtype in (
            ("_sequence_lengths", (max_seqs,), torch.int32),
            ("_query_start_loc", (max_seqs + 1,), torch.int32),
            ("_raw_state_slot_ids", (max_seqs,), torch.int32),
            ("_state_is_fresh", (max_seqs,), torch.bool),
            ("_num_accepted_tokens", (max_seqs,), torch.int32),
            ("_is_prefilling", (max_seqs,), torch.bool),
            ("_rope_position_input", (4, 1), torch.int64),
        ):
            setattr(layer, name, torch.full(shape, 1, dtype=dtype, device=device))
        return layer

    a, b = owner(), owner()
    metadata = Qwen3_8FlashNextQSAMetadata(
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=8,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32, device=device),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device),
        slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
        request_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
        is_prefilling=torch.zeros(2, dtype=torch.bool, device=device),
        qsa_state_slot_ids=torch.tensor([1, 0], dtype=torch.int32, device=device),
        qsa_state_is_fresh=torch.zeros(2, dtype=torch.bool, device=device),
        qsa_num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32, device=device),
    )
    metadata_b = qsa_module.replace(metadata, block_table=metadata.block_table + 4)
    table_a, table_b = (
        torch.empty((max_seqs, 2), dtype=torch.int32, device=device),
        torch.empty((max_seqs, 2), dtype=torch.int32, device=device),
    )
    positions = torch.arange(4, dtype=torch.int64, device=device)

    def stage():
        context.additional_kwargs.clear()
        first = a._stage_runtime_metadata(
            metadata,
            4,
            row_capacity=4,
            main_block_table=table_a,
            compressed_block_table=table_a,
        )
        second = b._stage_runtime_metadata(
            metadata_b,
            4,
            row_capacity=4,
            main_block_table=table_b,
            compressed_block_table=table_b,
        )
        assert first is second
        rope_a = a._shared_qsa_rope_positions(positions, first.request_ids, 4)
        rope_b = b._shared_qsa_rope_positions(positions, second.request_ids, 4)
        assert rope_a is rope_b
        return first, rope_a

    stage()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staged, rope = stage()
    metadata.seq_lens.add_(3)
    metadata.qsa_num_accepted_tokens.copy_(
        torch.tensor([3, 1], dtype=torch.int32, device=device)
    )
    metadata.qsa_state_slot_ids.copy_(
        torch.tensor([0, 1], dtype=torch.int32, device=device)
    )
    positions.add_(7)
    graph.replay()
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(staged.sequence_lengths[:2], metadata.seq_lens)
    torch.testing.assert_close(
        staged.num_accepted_tokens[:2], metadata.qsa_num_accepted_tokens
    )
    torch.testing.assert_close(staged.state_slot_ids[:2], metadata.qsa_state_slot_ids)
    torch.testing.assert_close(rope[:, 0], positions)
    torch.testing.assert_close(table_a[:2], metadata.block_table)
    torch.testing.assert_close(table_b[:2], metadata_b.block_table)
    assert torch.all(table_a[2:] == -1)
    assert torch.all(table_b[2:] == -1)
    assert torch.all(b._num_accepted_tokens == 1)
    query_ptr = staged.query_start_loc.data_ptr()
    for terminal in (3, 2, 4):
        # Replay keeps four token rows but shrinks the packed live prefix.
        # Unused request capacity must not turn padded token rows into a request.
        metadata.query_start_loc[-1:].fill_(terminal)
        metadata.request_ids.copy_(
            torch.tensor(
                [0, 0] + [1] * (terminal - 2) + [-1] * (4 - terminal),
                dtype=torch.int32,
                device=device,
            )
        )
        graph.replay()
        expected = torch.full_like(staged.query_start_loc, terminal)
        expected[:2].copy_(metadata.query_start_loc[:2])
        torch.testing.assert_close(staged.query_start_loc, expected)
        assert staged.query_start_loc.data_ptr() == query_ptr
        assert torch.all(staged.logical_positions[terminal:] == -1)
    distinct = qsa_module.replace(
        metadata_b, qsa_num_accepted_tokens=metadata.qsa_num_accepted_tokens.clone()
    )
    separate = b._stage_runtime_metadata(
        distinct,
        4,
        row_capacity=4,
        main_block_table=table_b,
        compressed_block_table=table_b,
    )
    assert separate is not staged
