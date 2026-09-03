# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_resolve_native_nvfp4_dsv4_kv_cache_dtype() -> None:
    cache_config = SimpleNamespace(cache_dtype="nvfp4")

    cache_dtype, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        True, cache_config.cache_dtype, cache_config
    )

    assert cache_dtype == "nvfp4_ds_mla"
    assert torch_dtype == torch.uint8
    assert cache_config.cache_dtype == "nvfp4_ds_mla"
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch_dtype,
        cache_dtype_str=cache_dtype,
        compress_ratio=4,
        alignment=512,
        model_version="deepseek_v4",
    )
    assert spec.real_page_size_bytes == (64 // 4) * 432


def test_reject_native_nvfp4_without_sparse_mla_layout() -> None:
    with pytest.raises(ValueError, match="requires the sparse MLA layout"):
        _resolve_dsv4_kv_cache_dtype(False, "nvfp4", None)
