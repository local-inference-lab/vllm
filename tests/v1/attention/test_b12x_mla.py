# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import mla_attention
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import b12x_mla
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAImpl,
    B12xMLAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_b12x_mla_is_registered_with_k3_envelope() -> None:
    assert AttentionBackendEnum.B12X_MLA.get_class() is B12xMLABackend
    assert B12xMLABackend.get_name() == "B12X_MLA"
    assert B12xMLABackend.get_supported_head_sizes() == [576]
    assert B12xMLABackend.supports_block_size(944)
    assert not B12xMLABackend.supports_block_size(936)
    assert B12xMLABackend.supports_compute_capability(DeviceCapability(12, 0))
    assert not B12xMLABackend.supports_compute_capability(DeviceCapability(10, 0))
    assert B12xMLABackend.supports_non_causal()
    assert (
        B12xMLAMetadataBuilder._cudagraph_support
        is b12x_mla.AttentionCGSupport.UNIFORM_BATCH
    )
    assert B12xMLAMetadataBuilder.query_len_support is b12x_mla.QueryLenSupport.UNIFORM


@pytest.mark.parametrize(
    ("logical_heads", "kernel_heads"),
    ((6, 8), (8, 8), (12, 16), (16, 16)),
)
def test_b12x_mla_pads_query_heads_to_kernel_tile(
    logical_heads: int, kernel_heads: int
) -> None:
    assert b12x_mla._kernel_query_heads(logical_heads) == kernel_heads


def test_b12x_mla_uses_gathered_dcp_head_geometry() -> None:
    assert b12x_mla._kernel_query_heads(6, 8) == 48
    assert b12x_mla._kernel_query_heads(12, 8) == 96
    assert b12x_mla._kernel_query_heads(8, 12) == 96
    assert b12x_mla._kernel_query_heads(6, 16) == 96
    with pytest.raises(ValueError, match="multiple of eight"):
        b12x_mla._kernel_query_heads(6, 2)


@pytest.mark.parametrize(
    ("max_seq_len", "expected"),
    ((None, 8), (0, 1), (64, 1), (65, 1), (256, 1), (257, 2), (4096, 8)),
)
def test_b12x_mla_limits_active_cache_splits(
    max_seq_len: int | None, expected: int
) -> None:
    plan = SimpleNamespace(num_splits=8, chunks_per_split=4)
    assert b12x_mla._active_dense_mla_splits(plan, max_seq_len) == expected


def test_b12x_mla_row_caps_cover_full_graph_shapes() -> None:
    assert b12x_mla._dense_mla_plan_row_caps(28) == (1, 2, 4, 8, 16, 28)


def test_b12x_mla_selects_smallest_covering_plan() -> None:
    plans = {1: "b1", 2: "b2", 4: "b4", 8: "b8"}

    assert b12x_mla._select_dense_mla_plan(plans, 1) == "b1"
    assert b12x_mla._select_dense_mla_plan(plans, 3) == "b4"
    assert b12x_mla._select_dense_mla_plan(plans, 8) == "b8"
    with pytest.raises(ValueError, match="exceed the planned capacities"):
        b12x_mla._select_dense_mla_plan(plans, 9)


def test_b12x_mla_plans_local_interleaved_dcp_cache() -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=64,
        ),
        model_config=SimpleNamespace(max_model_len=1_048_576),
    )

    assert b12x_mla._max_dcp_local_cache_tokens(config) == 131_072


def test_mla_uses_one_kv_shard_for_replicated_dcp_cache() -> None:
    replicated = SimpleNamespace(get_num_dcp_kv_shards=lambda _: 1)
    sharded = SimpleNamespace(get_num_dcp_kv_shards=lambda dcp_size: dcp_size)

    assert mla_attention._get_mla_kv_dcp_world_size(replicated, 16) == 1
    assert mla_attention._get_mla_kv_dcp_world_size(sharded, 16) == 16


def test_mla_rejects_partial_dcp_cache_without_matching_subgroup() -> None:
    partial = SimpleNamespace(get_num_dcp_kv_shards=lambda _: 4)

    with pytest.raises(NotImplementedError, match="partial DCP KV sharding"):
        mla_attention._get_mla_kv_dcp_world_size(partial, 16)


def _support_reason(
    monkeypatch, *, dcp_size: int, local_heads: int = 6, pcp_size: int = 1
) -> str | None:
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        prefill_context_parallel_size=pcp_size,
        cp_kv_cache_interleave_size=64,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            model_type="kimi_linear",
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        max_model_len=1_048_576,
        get_num_attention_heads=lambda _: local_heads,
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    monkeypatch.setattr(b12x_mla, "_load_dense_mla", lambda: object())
    monkeypatch.setattr(b12x_mla, "get_current_vllm_config", lambda: config)
    return B12xMLABackend.supports_combination(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8",
        block_size=944,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )


@pytest.mark.parametrize(
    ("dcp_size", "local_heads"),
    ((8, 12), (12, 8), (16, 6)),
)
def test_b12x_mla_selects_supported_native_dcp_geometry(
    monkeypatch, dcp_size: int, local_heads: int
) -> None:
    assert (
        _support_reason(
            monkeypatch,
            dcp_size=dcp_size,
            local_heads=local_heads,
        )
        is None
    )


def test_b12x_mla_rejects_unsupported_parallel_geometry(monkeypatch) -> None:
    dcp_reason = _support_reason(monkeypatch, dcp_size=2)
    pcp_reason = _support_reason(monkeypatch, dcp_size=8, pcp_size=2)

    assert dcp_reason is not None
    assert "multiple of eight" in dcp_reason
    assert pcp_reason is not None
    assert "prefill context parallelism" in pcp_reason


class _FakePlan:
    caps = SimpleNamespace(max_page_table_width=4)
    num_splits = 1
    chunks_per_split = 1

    def shapes_and_dtypes(self):
        return (((256,), torch.uint8),)


class _FakeDenseMLA:
    def __init__(self) -> None:
        self.bindings: list[SimpleNamespace] = []
        self.compile_count = 0

    def bind(self, plan, **kwargs):
        binding = SimpleNamespace(plan=plan, **kwargs)
        self.bindings.append(binding)
        return binding

    def compile(self, *, binding) -> None:
        self.compile_count += 1

    def run(self, *, binding):
        lse = torch.zeros(
            binding.output.shape[:2],
            dtype=torch.float32,
            device=binding.output.device,
        )
        return binding.output, lse


def _fake_impl(monkeypatch, *, num_heads: int = 8) -> tuple[B12xMLAImpl, _FakeDenseMLA]:
    impl = object.__new__(B12xMLAImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.scale = 192**-0.5
    impl.dcp_world_size = 1
    impl._dcp_comm_backend = "a2a"
    impl._dcp_max_batch_size = 16
    impl.dcp_q_replicate = False
    impl._compiled_bindings = set()
    dense_mla = _FakeDenseMLA()
    impl._dense_mla = dense_mla
    return impl, dense_mla


def test_b12x_mla_adapter_binds_common_decode_metadata(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    q_nope = torch.randn(batch, 8, 512, dtype=torch.bfloat16)
    q_rope = torch.randn(batch, 8, 64, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)
    output_2, _ = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)

    assert output.shape == (batch, 8, 512)
    assert output.dtype == torch.bfloat16
    assert lse is not None and lse.dtype == torch.float32
    assert output_2.shape == output.shape
    assert dense_mla.compile_count == 1
    binding = dense_mla.bindings[0]
    assert binding.q.shape == (batch, 8, 576)
    assert binding.q.is_contiguous()
    assert binding.kv_cache is cache
    assert binding.page_table is metadata.decode.block_table
    assert binding.cache_seqlens is metadata.decode.seq_lens
    assert binding.q_scale is None
    assert binding.kv_scale is None
    assert binding.sm_scale == impl.scale
    assert binding.active_splits == 1
    assert binding.scratch is metadata.dense_mla_scratch


def test_b12x_mla_adapter_accepts_non_multiple_of_eight_heads(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    q = torch.randn(1, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_padded_q=torch.empty(1, 8, 576, dtype=torch.bfloat16),
        dense_mla_padded_output=torch.empty(1, 8, 512, dtype=torch.bfloat16),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(_q_scale=torch.tensor(0.25), _k_scale=torch.tensor(0.5))

    output, _ = impl.forward_mqa(q, cache, metadata, layer)

    assert output.shape == (1, 6, 512)
    binding = dense_mla.bindings[0]
    assert binding.q.shape == (1, 8, 576)
    torch.testing.assert_close(binding.q[:, :6], q)
    torch.testing.assert_close(binding.q[:, 6:], torch.zeros_like(binding.q[:, 6:]))


def test_b12x_mla_adapter_binds_causal_multiquery_blocks(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    query_len = 8
    total_q = batch * query_len
    q = torch.randn(total_q, 8, 576, dtype=torch.bfloat16)
    cache = torch.randn(8, 16, 576, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)
    source_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dtype=torch.int32,
    )
    flat_table = source_table[:, None, :].expand(-1, query_len, -1).reshape(total_q, -1)
    flat_lens = torch.cat(
        (
            torch.arange(25, 33, dtype=torch.int32),
            torch.arange(41, 49, dtype=torch.int32),
        )
    )
    flat_query_start = torch.arange(total_q + 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_flat_block_table=flat_table,
        dense_mla_flat_seq_lens=flat_lens,
        dense_mla_flat_query_start_loc=flat_query_start,
        query_start_loc=query_start_loc,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([32, 48], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert output.shape == (total_q, 8, 512)
    assert lse is not None and lse.shape == (total_q, 8)
    assert binding.q.shape[0] == total_q
    assert binding.cache_seqlens.data_ptr() == flat_lens.data_ptr()
    assert binding.cu_seqlens_q.data_ptr() == flat_query_start.data_ptr()


def test_b12x_mla_builder_flattens_non_causal_draft_block(monkeypatch) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._dense_mla_scratch = torch.empty(256, dtype=torch.uint8)
    builder._dense_mla_padded_q = None
    builder._dense_mla_padded_output = None
    builder._max_dense_mla_rows = 16
    builder._dense_mla_flat_block_table = torch.zeros(16, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(16, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(17, dtype=torch.int32)
    builder.dcp_world_size = 1

    source_table = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=False,
        num_decodes=1,
        num_decode_tokens=8,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([32], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    assert result.dense_mla_scratch is builder._dense_mla_scratch
    torch.testing.assert_close(
        result.dense_mla_flat_block_table,
        source_table.expand(8, -1),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_seq_lens,
        torch.full((8,), 32, dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_query_start_loc,
        torch.arange(9, dtype=torch.int32),
    )


def test_b12x_mla_builder_flattens_causal_verification_block(monkeypatch) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._dense_mla_scratch = torch.empty(256, dtype=torch.uint8)
    builder._dense_mla_padded_q = None
    builder._dense_mla_padded_output = None
    builder._max_dense_mla_rows = 16
    builder._dense_mla_flat_block_table = torch.zeros(16, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(16, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(17, dtype=torch.int32)
    builder._dense_mla_causal_offsets = torch.arange(-7, 1, dtype=torch.int32)
    builder._dense_mla_flat_global_seq_lens = None
    builder._dense_mla_flat_dcp_remainder = None
    builder.dcp_world_size = 1

    source_table = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_decodes=1,
        num_decode_tokens=8,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([32], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    torch.testing.assert_close(
        result.dense_mla_flat_block_table,
        source_table.expand(8, -1),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_seq_lens,
        torch.arange(25, 33, dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_query_start_loc,
        torch.arange(9, dtype=torch.int32),
    )


def test_b12x_mla_builder_maps_causal_verification_lengths_to_dcp_rank(
    monkeypatch,
) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._dense_mla_scratch = torch.empty(256, dtype=torch.uint8)
    builder._dense_mla_padded_q = None
    builder._dense_mla_padded_output = None
    builder._max_dense_mla_rows = 8
    builder._dense_mla_flat_block_table = torch.zeros(8, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(8, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(9, dtype=torch.int32)
    builder._dense_mla_causal_offsets = torch.arange(-3, 1, dtype=torch.int32)
    builder._dense_mla_flat_global_seq_lens = torch.empty(8, dtype=torch.int32)
    builder._dense_mla_flat_dcp_remainder = torch.empty(8, dtype=torch.int32)
    builder.dcp_world_size = 4
    builder._dcp_rank = 3
    builder.cp_kv_cache_interleave_size = 2

    metadata = SimpleNamespace(
        causal=True,
        num_decodes=1,
        num_decode_tokens=4,
        decode=SimpleNamespace(
            block_table=torch.tensor([[3, 4, 5, 6]], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            dcp_tot_seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    torch.testing.assert_close(
        result.dense_mla_flat_seq_lens,
        torch.tensor([2, 3, 4, 4], dtype=torch.int32),
    )


def test_b12x_mla_builder_preserves_tiled_q4_dcp_verification(
    monkeypatch,
) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._dense_mla_plans = {4: _FakePlan()}
    verify_plan = SimpleNamespace(caps=SimpleNamespace(max_page_table_width=4))
    builder._dense_mla_verify_plans = {1: verify_plan}
    builder._dense_mla_scratch = torch.empty(256, dtype=torch.uint8)
    builder._dense_mla_padded_q = None
    builder._dense_mla_padded_output = None
    builder._max_dense_mla_rows = 8
    builder._dense_mla_flat_block_table = torch.zeros(8, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(8, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(9, dtype=torch.int32)
    builder._dense_mla_causal_offsets = torch.arange(-3, 1, dtype=torch.int32)
    builder._dense_mla_flat_global_seq_lens = torch.empty(8, dtype=torch.int32)
    builder._dense_mla_flat_dcp_remainder = torch.empty(8, dtype=torch.int32)
    builder.dcp_world_size = 4
    builder._dcp_rank = 3
    builder.cp_kv_cache_interleave_size = 2

    source_table = torch.tensor([[3, 4, 5, 6, 90]], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_decodes=1,
        num_decode_tokens=4,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            dcp_tot_seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    assert result.dense_mla_plan is verify_plan
    torch.testing.assert_close(
        result.dense_mla_verify_block_table,
        source_table[:, :4],
    )
    torch.testing.assert_close(
        result.dense_mla_query_cache_seq_lens,
        torch.tensor([2, 3, 4, 4], dtype=torch.int32),
    )
    assert getattr(result, "dense_mla_flat_block_table", None) is None


def test_b12x_mla_builder_bounds_single_token_draft_table(monkeypatch) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._dense_mla_scratch = torch.empty(256, dtype=torch.uint8)
    builder._dense_mla_padded_q = None
    builder._dense_mla_padded_output = None
    builder._max_dense_mla_rows = 2
    builder._dense_mla_flat_block_table = torch.zeros(2, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(2, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(3, dtype=torch.int32)
    builder._dense_mla_causal_offsets = torch.zeros(1, dtype=torch.int32)
    builder._dense_mla_flat_global_seq_lens = None
    builder._dense_mla_flat_dcp_remainder = None
    builder.dcp_world_size = 1

    source_table = torch.tensor(
        [[0, 1, 2, 3, 90, 91], [4, 5, 6, 7, 92, 93]],
        dtype=torch.int32,
    )
    source_lens = torch.tensor([32, 48], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=False,
        num_decodes=2,
        num_decode_tokens=2,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=source_lens,
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    torch.testing.assert_close(
        result.dense_mla_flat_block_table,
        source_table[:, :4],
    )
    torch.testing.assert_close(result.dense_mla_flat_seq_lens, source_lens)
    torch.testing.assert_close(
        result.dense_mla_flat_query_start_loc,
        torch.arange(3, dtype=torch.int32),
    )


def test_b12x_mla_adapter_uses_flattened_non_causal_rows(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    query_rows = 8
    q = torch.randn(query_rows, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    source_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    flat_table = source_table.expand(query_rows, -1).contiguous()
    flat_lens = torch.full((query_rows,), 49, dtype=torch.int32)
    flat_query_start = torch.arange(query_rows + 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_padded_q=torch.empty(query_rows, 8, 576, dtype=torch.bfloat16),
        dense_mla_padded_output=torch.empty(query_rows, 8, 512, dtype=torch.bfloat16),
        dense_mla_flat_block_table=flat_table,
        dense_mla_flat_seq_lens=flat_lens,
        dense_mla_flat_query_start_loc=flat_query_start,
        query_start_loc=torch.tensor([0, query_rows], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([49], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(_q_scale=torch.tensor(0.25), _k_scale=torch.tensor(0.5))

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.page_table is flat_table
    assert binding.cache_seqlens is flat_lens
    assert binding.cu_seqlens_q.data_ptr() == flat_query_start.data_ptr()
    assert output.shape == (query_rows, 6, 512)
    assert lse is not None and lse.shape == (query_rows, 6)


def test_b12x_mla_adapter_uses_tiled_query_visibility(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    query_rows = 4
    q = torch.randn(query_rows, 8, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    source_table = torch.tensor([[0, 1, 2, 3, 90]], dtype=torch.int32)
    verify_table = source_table[:, :4].contiguous()
    query_cache_seq_lens = torch.tensor([29, 30, 31, 32], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_verify_block_table=verify_table,
        dense_mla_query_cache_seq_lens=query_cache_seq_lens,
        query_start_loc=query_start_loc,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([32], dtype=torch.int32),
        ),
    )

    output, lse = impl.forward_mqa(
        q,
        cache,
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    binding = dense_mla.bindings[0]
    assert binding.page_table is verify_table
    assert binding.cu_seqlens_q.data_ptr() == query_start_loc.data_ptr()
    assert binding.query_cache_seqlens is query_cache_seq_lens
    assert output.shape == (query_rows, 8, 512)
    assert lse is not None and lse.shape == (query_rows, 8)


def test_b12x_mla_adapter_passes_fp8_scales(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    q = torch.empty(1, 8, 576, dtype=torch.float8_e4m3fn)
    cache = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.q_scale is layer._q_scale
    assert binding.kv_scale is layer._k_scale


def test_b12x_mla_adapter_gathers_and_combines_dcp(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    impl.dcp_world_size = 8
    batch = 2
    q = torch.randn(batch, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    group = SimpleNamespace(world_size=8)
    calls: list[str] = []

    monkeypatch.setattr(b12x_mla, "get_dcp_group", lambda: group)

    def gather(local_q, actual_group, *, max_batch_size, output_head_dim, out):
        assert actual_group is group
        assert max_batch_size == 16
        assert output_head_dim == 512
        assert out.shape == (batch, 48, 576)
        calls.append("gather")
        out.copy_(local_q.repeat(1, 8, 1))
        return out

    def reduce(output, lse, actual_group, **kwargs):
        assert actual_group is group
        assert output.shape == (batch, 48, 512)
        assert lse.shape == (batch, 48)
        assert "seq_lens" not in kwargs
        assert "query_start_loc" not in kwargs
        assert kwargs["b12x_query_head_dim"] == 576
        calls.append("reduce")
        return output[:, :6].add(1)

    monkeypatch.setattr(b12x_mla, "dcp_b12x_all_gather_heads", gather)
    monkeypatch.setattr(b12x_mla, "dcp_a2a_lse_reduce", reduce)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_padded_q=torch.empty(batch, 48, 576, dtype=torch.bfloat16),
        dense_mla_padded_output=torch.zeros(batch, 48, 512, dtype=torch.bfloat16),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([17, 31], dtype=torch.int32),
        ),
    )

    output, lse = impl.forward_mqa(
        q,
        cache,
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    assert calls == ["gather", "reduce"]
    assert dense_mla.bindings[0].q.shape == (batch, 48, 576)
    assert dense_mla.bindings[0].q.data_ptr() == metadata.dense_mla_padded_q.data_ptr()
    torch.testing.assert_close(output, torch.ones_like(output))
    assert lse is None


def test_b12x_mla_adapter_skips_query_gather_for_qrep(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    impl.dcp_world_size = 8
    impl.dcp_q_replicate = True
    batch = 2
    q = torch.randn(batch, 48, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    group = SimpleNamespace(world_size=8)
    calls: list[str] = []

    monkeypatch.setattr(b12x_mla, "get_dcp_group", lambda: group)

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("qrep must skip the query all-gather")

    def reduce(output, lse, actual_group, **kwargs):
        assert actual_group is group
        calls.append("reduce")
        return output[:, :6]

    monkeypatch.setattr(b12x_mla, "dcp_b12x_all_gather_heads", unexpected_gather)
    monkeypatch.setattr(b12x_mla, "dcp_a2a_lse_reduce", reduce)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_padded_q=torch.empty(batch, 48, 576, dtype=torch.bfloat16),
        dense_mla_padded_output=torch.zeros(batch, 48, 512, dtype=torch.bfloat16),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([17, 31], dtype=torch.int32),
        ),
    )

    output, lse = impl.forward_mqa(
        q,
        cache,
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    assert calls == ["reduce"]
    assert dense_mla.bindings[0].q.data_ptr() == q.data_ptr()
    assert output.shape == (batch, 6, 512)
    assert lse is None


def test_b12x_mla_adapter_skips_dcp_for_replicated_cache(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    impl.dcp_world_size = 8
    batch = 2
    q = torch.randn(batch, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("replicated KV attention must not use DCP collectives")

    monkeypatch.setattr(b12x_mla, "get_dcp_group", unexpected_collective)
    monkeypatch.setattr(b12x_mla, "dcp_b12x_all_gather_heads", unexpected_collective)
    monkeypatch.setattr(b12x_mla, "dcp_a2a_lse_reduce", unexpected_collective)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_padded_q=torch.empty(batch, 8, 576, dtype=torch.bfloat16),
        dense_mla_padded_output=torch.empty(batch, 8, 512, dtype=torch.bfloat16),
        dense_mla_dcp_world_size=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([17, 31], dtype=torch.int32),
        ),
    )

    output, lse = impl.forward_mqa(
        q,
        cache,
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    assert output.shape == (batch, 6, 512)
    assert lse is not None and lse.shape == (batch, 6)
    assert dense_mla.bindings[0].q.shape == (batch, 8, 576)


def test_b12x_mla_adapter_requires_caller_owned_scratch(monkeypatch) -> None:
    impl, _ = _fake_impl(monkeypatch)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
        ),
    )

    with pytest.raises(RuntimeError, match="caller-owned dense MLA scratch"):
        impl.forward_mqa(
            torch.randn(1, 8, 576, dtype=torch.bfloat16),
            torch.randn(1, 16, 576, dtype=torch.bfloat16),
            metadata,
            SimpleNamespace(_q_scale=None, _k_scale=None),
        )
