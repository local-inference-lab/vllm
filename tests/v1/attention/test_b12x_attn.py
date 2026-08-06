# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends import b12x_attn
from vllm.v1.attention.backends.b12x_attn import (
    B12XPagedAttentionBackend,
    B12XPagedAttentionImpl,
    B12XPagedMetadataBuilder,
    _kv_page_size,
    _max_page_table_width,
    _uses_b12x_dflash_attention,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.utils import select_common_block_size


class _Page128OnlyBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PAGE128_ONLY"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]


@pytest.mark.parametrize(
    ("method", "backend", "expected"),
    [
        ("dflash", AttentionBackendEnum.B12X_ATTN, True),
        ("dflash", AttentionBackendEnum.TRITON_ATTN, False),
        ("dflash", None, False),
        ("draft_model", AttentionBackendEnum.B12X_ATTN, False),
    ],
)
def test_b12x_contiguous_dflash_requires_draft_backend(
    method: str,
    backend: AttentionBackendEnum | None,
    expected: bool,
) -> None:
    spec_config = SimpleNamespace(method=method, attention_backend=backend)

    assert _uses_b12x_dflash_attention(spec_config) is expected


def test_b12x_dense_backend_advertises_page128() -> None:
    assert B12XPagedAttentionBackend.get_supported_kernel_block_sizes() == [64, 128]
    assert B12XPagedAttentionBackend.supports_block_size(64)
    assert B12XPagedAttentionBackend.supports_block_size(128)
    assert not B12XPagedAttentionBackend.supports_block_size(32)
    assert B12XPagedAttentionBackend.get_preferred_block_size(16) == 128
    assert B12XPagedAttentionBackend.get_preferred_block_size(64) == 64


def test_b12x_dense_backend_advertises_sliding_window() -> None:
    assert B12XPagedAttentionBackend.supports_sliding_window()


def test_b12x_dense_backend_supports_uniform_verifier_graphs() -> None:
    assert (
        B12XPagedMetadataBuilder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    )


def test_b12x_dense_kv_cache_shape_accepts_page128() -> None:
    assert B12XPagedAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        cache_dtype_str="bfloat16",
    ) == (3, 2, 128, 4, 128)


def test_b12x_dense_can_share_page128_group() -> None:
    assert (
        select_common_block_size(
            128,
            [B12XPagedAttentionBackend, _Page128OnlyBackend],
        )
        == 128
    )


def test_b12x_hybrid_align_reserves_expanded_page_table() -> None:
    assert _max_page_table_width(4096, 128, 4096, "none") == 32
    assert _max_page_table_width(4096, 128, 4096, "align") == 64

    storage_block_size = 3200
    expanded_width = (
        (4096 + storage_block_size - 1)
        // storage_block_size
        * (storage_block_size // 128)
    )
    assert expanded_width == 50
    assert expanded_width <= _max_page_table_width(4096, 128, 4096, "align")


def test_b12x_runtime_page_size_comes_from_static_cache_shape() -> None:
    key_cache = torch.empty((3, 64, 4, 128), device="meta")
    value_cache = torch.empty_like(key_cache)

    assert _kv_page_size(key_cache, value_cache) == 64

    with pytest.raises(ValueError, match="matching K/V page sizes"):
        _kv_page_size(
            key_cache,
            torch.empty((3, 128, 4, 128), device="meta"),
        )


def test_b12x_lazily_prepares_missing_decode_capture_bucket(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    # A lazy mid-batch plan may be larger than every materialized capture plan;
    # initialization reserves Sparkinfer's all-batch envelope for this case.
    plan = SimpleNamespace(layout=SimpleNamespace(nbytes=96))
    created: list[tuple[int, int]] = []

    def create_plan(page_size: int, size: int) -> SimpleNamespace:
        created.append((page_size, size))
        return plan

    impl._decode_plans = {}
    impl._create_decode_plan = create_plan
    impl._scratch_nbytes = 128
    impl._extend_plans = {64: object()}
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: False)

    selected = impl._select_plan(metadata, 7, 7, 7, 64)

    assert selected is plan
    assert impl._decode_plans == {(64, 7): plan}
    assert created == [(64, 7)]

    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert created == [(64, 7)]


def test_b12x_selects_preplanned_uniform_verifier_bucket() -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    plan = object()
    impl._verify_q_per_req = 8
    impl._verify_plans = {(64, 1): plan}
    impl._decode_plans = {}
    impl._extend_plans = {64: object()}
    metadata = SimpleNamespace(max_query_len=8)

    assert impl._select_plan(metadata, 8, 8, 1, 64) is plan


def test_b12x_partial_storage_limit_accepts_zero(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_B12X_PAGED_DECODE_MAX_PARTIAL_ROWS", "0")

    assert (
        b12x_attn._env_optional_storage_limit(
            "VLLM_B12X_PAGED_DECODE_MAX_PARTIAL_ROWS",
            allow_zero=True,
        )
        == 0
    )


def test_b12x_lazy_decode_bucket_exceeding_envelope_fails_closed(
    monkeypatch,
) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl._decode_plans = {}
    impl._create_decode_plan = lambda _page_size, _size: SimpleNamespace(
        layout=SimpleNamespace(nbytes=65)
    )
    impl._scratch_nbytes = 64
    impl._extend_plans = {64: object()}
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: False)

    with pytest.raises(RuntimeError, match="exceeds reserved scratch"):
        impl._select_plan(metadata, 7, 7, 7, 64)


def test_b12x_missing_decode_bucket_fails_closed_during_capture(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl._decode_plans = {}
    impl._create_decode_plan = lambda page_size, size: pytest.fail(
        f"created plan for page {page_size}, batch {size} during capture"
    )
    impl._scratch_nbytes = 64
    impl._extend_plans = {64: object()}
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: True)

    with pytest.raises(RuntimeError, match="batch size 7"):
        impl._select_plan(metadata, 7, 7, 7, 64)


def test_b12x_decode_forward_leaves_split_policy_to_plan(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl.output_head_size = 4
    impl.dtype = torch.float32
    impl.window_left = -1
    impl.sinks = None
    impl._scratch_nbytes = 32
    cache = torch.empty((1, 64, 1, 4))
    impl._kv_cache_views = lambda kv_cache: (cache, cache)
    impl._prepare_sinks = lambda sinks, device: None
    impl._prepare_fp8_descales = lambda layer, num_reqs, device: (None, None)

    bind_kwargs: dict[str, object] = {}

    def bind(**kwargs):
        bind_kwargs.update(kwargs)
        return object()

    plan = SimpleNamespace(bind=bind, caps=SimpleNamespace(mode="decode"))
    impl._select_plan = lambda metadata, total_q, q_capacity, num_reqs, page_size: plan
    impl._paged_attention_forward = lambda *, binding: binding

    workspace = SimpleNamespace(
        get_simultaneous=lambda specs: (torch.empty(32, dtype=torch.uint8),)
    )
    monkeypatch.setattr(b12x_attn, "current_workspace_manager", lambda: workspace)

    metadata = SimpleNamespace(
        num_actual_tokens=2,
        max_query_len=1,
        causal=True,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        query_start_loc=torch.arange(3, dtype=torch.int32),
    )
    query = torch.zeros((2, 1, 4))
    output = torch.empty_like(query)
    kv_cache = torch.ones(1)

    result = impl.forward(
        SimpleNamespace(),
        query,
        torch.empty(0),
        torch.empty(0),
        kv_cache,
        metadata,
        output,
    )

    assert result is output
    assert "fixed_split_size" not in bind_kwargs
    assert "disable_split_kv" not in bind_kwargs
    assert bind_kwargs["active_total_q"] == 2
