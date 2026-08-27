# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import b12x_mla
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAImpl,
    B12xMLAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_b12x_mla_is_registered_with_k3_envelope() -> None:
    assert AttentionBackendEnum.B12X.get_class() is B12xMLABackend
    assert B12xMLABackend.get_name() == "B12X"
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


def test_b12x_mla_plans_local_interleaved_dcp_cache() -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=64,
        ),
        model_config=SimpleNamespace(max_model_len=1_048_576),
    )

    assert b12x_mla._max_dcp_local_cache_tokens(config) == 131_072


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


def _fake_impl(
    *, num_heads: int = 8, dcp_world_size: int = 1
) -> tuple[B12xMLAImpl, _FakeDenseMLA]:
    impl = object.__new__(B12xMLAImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.scale = 192**-0.5
    impl.dcp_world_size = dcp_world_size
    impl._compiled_bindings = set()
    dense_mla = _FakeDenseMLA()
    impl._dense_mla = dense_mla
    return impl, dense_mla


def test_b12x_mla_adapter_binds_common_decode_metadata() -> None:
    impl, dense_mla = _fake_impl()
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


def test_b12x_mla_adapter_consumes_generic_dcp_query_gather() -> None:
    impl, dense_mla = _fake_impl(num_heads=6, dcp_world_size=16)
    batch = 2
    q = torch.randn(batch, 96, 576, dtype=torch.bfloat16)
    cache = torch.randn(8, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_scratch=torch.empty(256, dtype=torch.uint8),
        dense_mla_dcp_world_size=16,
        max_seq_len=64,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    assert output.shape == (batch, 96, 512)
    assert lse is not None and lse.shape == (batch, 96)
    assert dense_mla.compile_count == 1
    assert dense_mla.bindings[0].q is q
