# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache, build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_flash_attention_kv_cache_with_virtual_blocks():
    num_storage_blocks = 2
    kernel_block_size = 128
    spec = FullAttentionSpec(
        block_size=1024,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.int8,
        page_size_padded=589824,
    )
    virtual_blocks_per_storage_block = spec.block_size // kernel_block_size

    raw_tensors = {
        "layer": torch.zeros(
            spec.page_size_bytes * num_storage_blocks, dtype=torch.int8
        )
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [kernel_block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (16, 2, 128, 2, 64)
    assert kv_cache.stride(0) == (
        spec.page_size_bytes // virtual_blocks_per_storage_block
    )
    assert kv_cache[-1].storage_offset() < raw_tensors["layer"].numel()


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8


class _FakeMetadataBuilder:
    supports_exact_metadata_reuse = False

    def __init__(self):
        self.num_computed_tokens: torch.Tensor | None = None

    def build(self, common_prefix_len, common_attn_metadata):
        self.num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        return object()

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return object()


class _FakeAttentionGroup:
    def __init__(self, builder: _FakeMetadataBuilder, group_id: int):
        self.builder = builder
        self.kv_cache_spec = SimpleNamespace(group_id=group_id)
        self.layer_names = [f"layer.{group_id}"]

    def get_metadata_builder(self, builder_index: int):
        assert builder_index == 0
        return self.builder


def _build_fake_group_metadata(
    builders: list[_FakeMetadataBuilder],
    *,
    for_cudagraph_capture: bool = False,
):
    num_groups = len(builders)
    groups = [[_FakeAttentionGroup(builder, i)] for i, builder in enumerate(builders)]
    kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(dcp_replicated=False))
        for _ in builders
    ]
    return build_attn_metadata(
        attn_groups=groups,
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([17], dtype=torch.int32),
        max_seq_len=17,
        block_tables=tuple(
            torch.tensor([[i]], dtype=torch.int32) for i in range(num_groups)
        ),
        slot_mappings=torch.zeros((num_groups, 1), dtype=torch.int64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=kv_cache_groups),
        for_cudagraph_capture=for_cudagraph_capture,
    )


def test_build_attn_metadata_skips_exact_key_for_unsupported_builders():
    builders = [_FakeMetadataBuilder(), _FakeMetadataBuilder()]

    with patch(
        "vllm.v1.worker.gpu.attn_utils.exact_attention_metadata_cache_key"
    ) as cache_key:
        _build_fake_group_metadata(builders)

    cache_key.assert_not_called()
    assert builders[0].num_computed_tokens is builders[1].num_computed_tokens


def test_build_attn_metadata_skips_exact_key_during_cudagraph_capture():
    builder = _FakeMetadataBuilder()
    builder.supports_exact_metadata_reuse = True

    with patch(
        "vllm.v1.worker.gpu.attn_utils.exact_attention_metadata_cache_key"
    ) as cache_key:
        _build_fake_group_metadata([builder], for_cudagraph_capture=True)

    cache_key.assert_not_called()
