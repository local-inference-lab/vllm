# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import VllmConfig
from vllm.platforms.cuda import CudaPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _kvarn_config(*, async_scheduling: bool = True) -> VllmConfig:
    config = VllmConfig()
    config.model_config = SimpleNamespace(  # type: ignore[assignment]
        is_mm_prefix_lm=False, use_mla=True
    )
    config.cache_config.cache_dtype = "kvarn_mla_k5_g64"
    config.cache_config.block_size = 64
    config.cache_config.enable_prefix_caching = False
    config.cache_config.kv_offloading_size = None
    config.cache_config.kv_cache_dtype_skip_layers = []
    config.attention_config.backend = AttentionBackendEnum.B12X_MLA_SPARSE
    config.parallel_config.enable_dbo = False
    config.scheduler_config.async_scheduling = async_scheduling
    config.kv_transfer_config = None
    return config


def _mtp_config(**overrides) -> SimpleNamespace:
    values = {
        "method": "mtp",
        "num_speculative_tokens": 4,
        "parallel_drafting": False,
        "attention_backend": AttentionBackendEnum.B12X_MLA_SPARSE,
        "kv_cache_dtype": "kvarn_mla_k5_g64",
        "moe_backend": "b12x",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_mla_kvarn_admits_sync_and_async_native_draft(
    async_scheduling: bool,
) -> None:
    config = _kvarn_config(async_scheduling=async_scheduling)
    config.speculative_config = _mtp_config()  # type: ignore[assignment]

    CudaPlatform.check_and_update_config(config)


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_mla_kvarn_admits_ngram_speculation_without_a_draft_cache(
    method: str,
) -> None:
    config = _kvarn_config()
    config.speculative_config = SimpleNamespace(  # type: ignore[assignment]
        method=method,
        num_speculative_tokens=4,
    )

    CudaPlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("prefix", True, "prefix caching"),
        ("offload", 1.0, "offloading"),
        ("transfer", SimpleNamespace(), "KV transfer"),
        ("dbo", True, "dual-batch overlap"),
        ("mixed_dtype", ["0"], "mixed KV-cache dtypes"),
        ("backend", None, "B12X_MLA_SPARSE"),
        ("block", 32, "block-size 64"),
        ("non_mla", False, "MLA model"),
    ],
)
def test_mla_kvarn_rejects_unsupported_features(
    feature: str,
    value: Any,
    message: str,
) -> None:
    config = _kvarn_config()
    if feature == "prefix":
        config.cache_config.enable_prefix_caching = value
    elif feature == "offload":
        config.cache_config.kv_offloading_size = value
    elif feature == "transfer":
        config.kv_transfer_config = value
    elif feature == "dbo":
        config.parallel_config.enable_dbo = value
    elif feature == "backend":
        config.attention_config.backend = value
    elif feature == "block":
        config.cache_config.block_size = value
    elif feature == "non_mla":
        config.model_config.use_mla = value  # type: ignore[misc]
    else:
        config.cache_config.kv_cache_dtype_skip_layers = value

    with pytest.raises(ValueError, match=message):
        CudaPlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attention_backend": None}, "speculative.*B12X_MLA_SPARSE"),
        ({"kv_cache_dtype": None}, "speculative.*kvarn_mla_k5_g64"),
    ],
)
def test_mla_kvarn_requires_explicit_native_draft_config(
    override: dict[str, object],
    message: str,
) -> None:
    config = _kvarn_config()
    config.speculative_config = _mtp_config(**override)  # type: ignore[assignment]

    with pytest.raises(ValueError, match=message):
        CudaPlatform.check_and_update_config(config)


def test_mla_kvarn_explicit_draft_config_is_native() -> None:
    config = _kvarn_config()
    config.speculative_config = _mtp_config()  # type: ignore[assignment]

    CudaPlatform.check_and_update_config(config)

    spec = config.speculative_config
    assert spec is not None
    assert spec.attention_backend is AttentionBackendEnum.B12X_MLA_SPARSE
    assert spec.kv_cache_dtype == "kvarn_mla_k5_g64"
    assert spec.moe_backend == "b12x"
