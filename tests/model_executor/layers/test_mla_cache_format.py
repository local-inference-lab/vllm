# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

import pytest

from vllm.model_executor.layers.mla_cache_format import (
    KV_FP8_ROPE_ENV,
    NVFP4_MLA_DYNAMIC_SCALE_ENV,
    NVFP4_MLA_SCALES_ENV,
    Nvfp4MlaCacheFormat,
)


def test_from_env_captures_one_server_static_mode(monkeypatch):
    monkeypatch.setenv(NVFP4_MLA_DYNAMIC_SCALE_ENV, "1")
    monkeypatch.setenv(KV_FP8_ROPE_ENV, "1")
    monkeypatch.delenv(NVFP4_MLA_SCALES_ENV, raising=False)

    cache_format = Nvfp4MlaCacheFormat.from_env()
    monkeypatch.setenv(NVFP4_MLA_DYNAMIC_SCALE_ENV, "0")

    assert cache_format.dynamic_scale
    assert cache_format.fp8_rope
    assert cache_format.scales_file == ""
    assert (
        cache_format.record_abi("nvfp4_ds_mla")
        == "nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1"
    )


@pytest.mark.parametrize(
    "cache_format",
    [
        Nvfp4MlaCacheFormat(
            dynamic_scale=True,
            fp8_rope=True,
            scales_file="/tmp/static-scales.json",
        ),
        Nvfp4MlaCacheFormat(
            dynamic_scale=True,
            fp8_rope=False,
            scales_file="",
        ),
    ],
)
def test_invalid_dynamic_combinations_fail_closed(cache_format):
    with pytest.raises(ValueError):
        cache_format.validate()


def test_static_scale_contents_participate_in_record_abi(tmp_path):
    scales = tmp_path / "scales.json"
    payload = b'{"format":"example","scales":[1.0]}'
    scales.write_bytes(payload)
    cache_format = Nvfp4MlaCacheFormat(
        dynamic_scale=False,
        fp8_rope=True,
        scales_file=str(scales),
    )

    expected_digest = hashlib.sha256(payload).hexdigest()
    assert cache_format.record_abi("nvfp4_ds_mla") == (
        f"nvfp4_ds_mla:fp8-rope-368:static-calibrated-v1:{expected_digest}"
    )

    scales.write_bytes(b'{"format":"example","scales":[2.0]}')
    assert cache_format.record_abi("nvfp4_ds_mla") != (
        f"nvfp4_ds_mla:fp8-rope-368:static-calibrated-v1:{expected_digest}"
    )


def test_missing_static_scale_file_cannot_form_persistent_abi(tmp_path):
    cache_format = Nvfp4MlaCacheFormat(
        dynamic_scale=False,
        fp8_rope=True,
        scales_file=str(tmp_path / "missing.json"),
    )
    with pytest.raises(ValueError, match="Cannot fingerprint"):
        cache_format.record_abi("nvfp4_ds_mla")


def test_non_nvfp4_cache_abi_is_unaffected_by_nvfp4_mode():
    cache_format = Nvfp4MlaCacheFormat(
        dynamic_scale=True,
        fp8_rope=False,
        scales_file="/does/not/matter",
    )
    assert cache_format.record_abi("bfloat16") == "bfloat16:default-v1"
