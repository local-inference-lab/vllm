# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear router GEMMs.

These assert the ``allow_cublas_router_gemm`` dispatch flag directly, so they
run device-free by mocking the platform predicates. The flag decides whether
the bf16xbf16->fp32 router GEMM uses ``torch.mm``'s fused out_dtype epilogue
(one kernel) or falls back to a bf16 matmul plus a standalone bf16->fp32 copy.

ROCm's fused ``torch.mm`` branch and SM120's CuTe branch are both guarded on
``not bias`` so a biased gate cannot silently drop its bias term.
"""

import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def _make_gate(
    monkeypatch,
    *,
    is_rocm: bool,
    is_cuda: bool = False,
    device_capability: tuple[int, int] | None = None,
    bias: bool = False,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    force_fp32_compute: bool = False,
) -> GateLinear:
    """Build a GateLinear with platform predicates mocked, no GPU needed."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: 1,
        )

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(
        platform,
        "is_device_capability",
        lambda capability: capability == device_capability,
    )
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)
    if is_cuda and device_capability == (12, 0):
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.is_available",
            lambda: True,
        )

    return GateLinear(
        input_size=2048,
        output_size=64,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
        force_fp32_compute=force_fp32_compute,
    )


def test_rocm_no_bias_bf16_fp32_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False)
    assert not gate.allow_specialized_router_gemm
    assert gate.allow_cublas_router_gemm


def test_rocm_bias_disables_fused_gemm(monkeypatch):
    # torch.mm cannot add a bias, so a biased gate must not take the fused path.
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True)
    assert not gate.allow_cublas_router_gemm


def test_rocm_fp32_weight_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, params_dtype=torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_rocm_non_fp32_out_dtype_disables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, out_dtype=torch.bfloat16)
    assert not gate.allow_cublas_router_gemm


def test_non_rocm_non_cuda_disables_fused_gemm(monkeypatch):
    # Neither the CUDA specialized path nor the ROCm branch applies.
    gate = _make_gate(monkeypatch, is_rocm=False, is_cuda=False)
    assert not gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_enables_fused_gemm(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=False, out_dtype=None)
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_cublas_router_gemm


def test_rocm_set_out_dtype_respects_bias_guard(monkeypatch):
    gate = _make_gate(monkeypatch, is_rocm=True, bias=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert not gate.allow_cublas_router_gemm


def test_sm120_enables_only_ll_bf16_router(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
    )

    assert gate.allow_ll_bf16_gemm
    assert not gate.allow_specialized_router_gemm
    assert not gate.allow_dsv3_router_gemm
    assert not gate.allow_cublas_router_gemm


def test_sm120_ll_bf16_respects_bias(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        bias=True,
    )

    assert not gate.allow_ll_bf16_gemm


def test_sm120_force_fp32_compute_preserves_fp32_weight_contract(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        force_fp32_compute=True,
    )

    assert gate.weight.dtype == torch.float32
    assert not gate.allow_ll_bf16_gemm


def test_sm120_set_out_dtype_enables_ll_bf16(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        out_dtype=None,
    )

    assert not gate.allow_ll_bf16_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm
