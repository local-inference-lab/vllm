# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v4.common.ops import nvfp4_staging
from vllm.models.deepseek_v4.common.ops.nvfp4_staging import (
    get_deepseek_v4_nvfp4_staging,
)
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


def test_native_nvfp4_staging_replaces_small_construction_workspace() -> None:
    nvfp4_staging._STAGING.clear()
    try:
        small = get_deepseek_v4_nvfp4_staging(
            device="cpu", max_num_tokens=2, producer="swa"
        )
        large = get_deepseek_v4_nvfp4_staging(
            device="cpu", max_num_tokens=4, producer="swa"
        )

        assert small.cache.shape[1] == 2
        assert large.cache.shape[1] == 4
        assert (
            get_deepseek_v4_nvfp4_staging(
                device="cpu", max_num_tokens=2, producer="swa"
            )
            is large
        )
    finally:
        nvfp4_staging._STAGING.clear()


@pytest.mark.parametrize(
    ("is_cuda", "dcp_size", "error"),
    [
        (False, 1, "only supported by the CUDA"),
        (True, 2, "does not support decode context parallelism"),
    ],
)
def test_native_nvfp4_config_rejects_unsupported_backend(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    dcp_size: int,
    error: str,
) -> None:
    from vllm import platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: is_cuda),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(cache_dtype="nvfp4_ds_mla"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
    )

    with pytest.raises(ValueError, match=error):
        VllmConfig.validate_nvfp4_kv_cache_with_mla(config)
