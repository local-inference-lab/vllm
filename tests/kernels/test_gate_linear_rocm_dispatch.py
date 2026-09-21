# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic eligibility tests for GateLinear router GEMMs.

These assert the dispatch flags directly, so they run device-free by mocking
the platform predicates. ``allow_cublas_router_gemm`` selects the
bf16xbf16->fp32 ``torch.mm`` epilogue, while ``allow_fp32_router_gemm`` selects
the gfx950 low-M kernel with fp32 weights and output.

ROCm's fused ``torch.mm`` branch and SM120's router branches are both guarded on
``not bias`` so a biased gate cannot silently drop its bias term.
"""

from types import SimpleNamespace

import pytest
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
    input_size: int = 2048,
    output_size: int = 64,
    on_gfx950: bool = False,
    parallel_world_size: int = 1,
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
            lambda: parallel_world_size,
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
    if is_rocm:
        # The module resolves its architecture at import time. Mock that
        # hardware probe too, so eligibility tests remain device-free.
        with monkeypatch.context() as probe:
            probe.setattr(
                torch.cuda,
                "get_device_properties",
                lambda *args, **kwargs: SimpleNamespace(gcnArchName="gfx950"),
            )
            import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: on_gfx950)

    return GateLinear(
        input_size=input_size,
        output_size=output_size,
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


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [(3072, 256), (4096, 8), (4096, 192), (6144, 128), (6144, 256)],
)
def test_rocm_gfx950_enables_fp32_router_gemm(
    monkeypatch, input_size: int, output_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=True,
    )
    assert gate.allow_fp32_router_gemm
    assert not gate.allow_cublas_router_gemm


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_rocm_fp32_router_gemm_is_replicated_across_ep_sizes(
    monkeypatch, ep_size: int
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=torch.float32,
        input_size=6144,
        output_size=128,
        on_gfx950=True,
        parallel_world_size=ep_size,
    )
    assert gate.weight.shape == (128, 6144)
    assert gate.allow_fp32_router_gemm


@pytest.mark.parametrize(
    ("input_size", "output_size", "params_dtype", "bias", "on_gfx950"),
    [
        pytest.param(6144, 128, torch.float32, False, False, id="gfx942"),
        pytest.param(2048, 64, torch.float32, False, True, id="shape"),
        pytest.param(6144, 128, torch.bfloat16, False, True, id="bf16-weight"),
        pytest.param(6144, 128, torch.float32, True, True, id="bias"),
    ],
)
def test_rocm_fp32_router_gemm_rejects_unsupported_configs(
    monkeypatch,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    bias: bool,
    on_gfx950: bool,
) -> None:
    gate = _make_gate(
        monkeypatch,
        is_rocm=True,
        params_dtype=params_dtype,
        bias=bias,
        input_size=input_size,
        output_size=output_size,
        on_gfx950=on_gfx950,
    )
    assert not gate.allow_fp32_router_gemm


def test_sm120_enables_bf16_fp32_paths_without_datacenter_kernels(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
    )

    assert gate.allow_ll_bf16_gemm
    assert not gate.allow_specialized_router_gemm
    assert not gate.allow_fp32_router_gemm
    assert gate.allow_cublas_router_gemm


def test_sm120_ll_bf16_respects_bias(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        bias=True,
    )

    assert not gate.allow_ll_bf16_gemm
    assert not gate.allow_cublas_router_gemm


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
    assert not gate.allow_cublas_router_gemm


def test_sm120_set_out_dtype_enables_ll_bf16(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        out_dtype=None,
    )

    assert not gate.allow_ll_bf16_gemm
    assert not gate.allow_cublas_router_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm
    assert gate.allow_cublas_router_gemm


def test_sm120_bf16_output_does_not_enable_fp32_gemm(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
        out_dtype=torch.bfloat16,
    )
    assert not gate.allow_ll_bf16_gemm
    assert not gate.allow_cublas_router_gemm


def test_sm120_capture_preserves_router_graph_pool_layout(monkeypatch):
    gate = _make_gate(
        monkeypatch,
        is_rocm=False,
        is_cuda=True,
        device_capability=(12, 0),
    )

    class FakeInput:
        dtype = torch.bfloat16
        shape = (32, 2048)

        def new_empty(self, shape):
            captured_allocations.append(shape)
            return self

    x = FakeInput()
    expected = torch.empty((32, 64), dtype=torch.float32)
    captured_allocations: list[tuple[int, ...]] = []

    monkeypatch.setattr(
        "vllm.compilation.breakable_cudagraph.BreakableCUDAGraphCapture.current",
        lambda: object(),
    )
    monkeypatch.setattr(torch, "mm", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        torch.ops.vllm,
        "sm120_cublas_router_gemm",
        gate_linear_mod.sm120_cublas_router_gemm_impl,
    )

    output, output_bias = gate(x)

    assert output is expected
    assert output_bias is None
    assert captured_allocations == [(32, 64)]
