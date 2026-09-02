# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persisting-L2 set-aside sizing for the GLM-5.3 L2 weight prefetcher."""

import pytest
import torch

from vllm.models.glm5next.nvidia import l2_prefetch as l2pf

MAX = 84_000_000


@pytest.mark.parametrize(
    ("raw", "max_bytes", "expected"),
    [
        ("max", MAX, MAX),
        (" MAX ", MAX, MAX),
        ("all", MAX, MAX),
        ("40", MAX, 40_000_000),
        ("40.5", MAX, 40_500_000),
        ("200", MAX, MAX),  # clamped to the device maximum
        ("0", MAX, 0),
        ("off", MAX, 0),
        ("", MAX, 0),
        (None, MAX, 0),
        ("-5", MAX, 0),
        ("bogus", MAX, 0),
        ("max", 0, 0),  # device without a persisting L2 set-aside
    ],
)
def test_persisting_l2_request(raw, max_bytes, expected):
    assert l2pf.persisting_l2_request(raw, max_bytes) == expected


def test_default_request_leaves_the_driver_limit_unchanged():
    assert l2pf.PERSIST_L2 == "0"


def test_invalid_numeric_environment_values_use_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_GLM53_L2_PREFETCH_MAX_TOKENS", "invalid")
    monkeypatch.setenv("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB", "invalid")
    assert l2pf._int_env("VLLM_GLM53_L2_PREFETCH_MAX_TOKENS", 256) == 256
    assert l2pf._mb("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB", "20") == 20_000_000


def _driver():
    from cuda.bindings import driver as cu

    return cu


def _read_limit(cu, device: torch.device) -> int:
    with torch.accelerator.device_index(device.index):
        torch.empty(1, device=device)  # make the primary context current
        err, value = cu.cuCtxGetLimit(cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE)
    assert err == cu.CUresult.CUDA_SUCCESS
    return int(value)


def _set_limit(cu, device: torch.device, value: int) -> None:
    with torch.accelerator.device_index(device.index):
        torch.empty(1, device=device)
        (err,) = cu.cuCtxSetLimit(cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE, value)
    assert err == cu.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_configure_persisting_l2_applies_to_the_primary_context():
    cu = _driver()
    device = torch.device("cuda", 0)
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, max_bytes = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if max_bytes <= 0:
        pytest.skip("device has no persisting L2 set-aside")
    before = _read_limit(cu, device)
    try:
        assert l2pf.configure_persisting_l2(device, request="max") == max_bytes
        assert _read_limit(cu, device) == max_bytes

        # A zero request never touches the driver state.
        assert l2pf.configure_persisting_l2(device, request="0") == 0
        assert _read_limit(cu, device) == max_bytes

        # Over-sized requests are clamped to the device maximum.
        assert l2pf.configure_persisting_l2(device, request="100000") == max_bytes
    finally:
        _set_limit(cu, device, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_prefetcher_applies_the_env_request_once(monkeypatch):
    cu = _driver()
    device = torch.device("cuda", 0)
    before = _read_limit(cu, device)
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, max_bytes = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
        dev,
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if max_bytes <= 0:
        pytest.skip("device has no persisting L2 set-aside")
    calls: list[str | None] = []
    real = l2pf.configure_persisting_l2

    def spy(dev, request=None):
        calls.append(request)
        return real(dev, request=request)

    monkeypatch.setattr(l2pf, "configure_persisting_l2", spy)
    monkeypatch.setattr(l2pf.L2Prefetcher, "_instances", {})
    monkeypatch.setattr(l2pf, "_persisting_l2_applied", {})
    monkeypatch.setattr(l2pf, "PERSIST_L2", "1")
    try:
        first = l2pf.L2Prefetcher.get(device)
        second = l2pf.L2Prefetcher.get(device)
        assert first is second
        assert calls == [None]
        applied = _read_limit(cu, device)
        assert first.persisting_l2_bytes == applied
        # CUDA rounds the request up to the device's allocation granularity.
        assert 1_000_000 <= applied < int(max_bytes)
    finally:
        _set_limit(cu, device, before)
