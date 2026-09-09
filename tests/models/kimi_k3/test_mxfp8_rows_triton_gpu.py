# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identity of the Triton MXFP8 row quantizer against the b12x kernel.

One CUDA device. The b12x quantizer is the reference the served dense GEMMs
consume; the Triton replica must produce the same E4M3 bytes and the same
UE8M0 scale bytes in both layouts for every row, including all-zero groups,
groups whose maximum lands on a power of two, saturating magnitudes and
subnormal-scale magnitudes.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.accelerator.device_count() < 1, reason="CUDA is required."
)


def _reference(source: torch.Tensor):
    from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute

    from vllm.models.kimi_k3.nvidia.ops.mxfp8_rows_triton import (
        mxfp8_scale_mma_numel,
    )

    rows, k = source.shape
    device = source.device
    values = torch.full((rows, k), 0x7F, dtype=torch.uint8, device=device)
    scale_rows = torch.full((rows, k // 32), 0xAA, dtype=torch.uint8, device=device)
    mma_numel = mxfp8_scale_mma_numel(rows, k)
    scale_mma = torch.full((mma_numel,), 0xAA, dtype=torch.uint8, device=device)
    quantize_mxfp8_rows_cute(source, values, scale_rows, scale_mma)
    return values, scale_rows, scale_mma


def _candidate(source: torch.Tensor):
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_rows_triton import (
        mxfp8_scale_mma_numel,
        quantize_mxfp8_rows_triton,
    )

    rows, k = source.shape
    device = source.device
    values = torch.full((rows, k), 0x7F, dtype=torch.uint8, device=device)
    scale_rows = torch.full((rows, k // 32), 0xAA, dtype=torch.uint8, device=device)
    mma_numel = mxfp8_scale_mma_numel(rows, k)
    scale_mma = torch.full((mma_numel,), 0xAA, dtype=torch.uint8, device=device)
    quantize_mxfp8_rows_triton(source, values, scale_rows, scale_mma)
    return values, scale_rows, scale_mma


def _cases(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(20260908)
    for rows, k in (
        (1, 7168),
        (4, 7168),
        (8, 7168),
        (4, 1408),
        (9, 1408),
        (16, 3584),
        (200, 256),
    ):
        x = torch.randn(rows, k, generator=generator) * 3.0
        yield f"normal {rows}x{k}", x
    x = torch.randn(4, 7168, generator=generator)
    x[:, :64] = 0.0  # two all-zero groups
    x[1, 64:96] = 2.0 ** torch.arange(32).float().sub(16)  # powers of two only
    x[2, 96:128] = 1.0e-30  # tiny magnitudes (subnormal scales)
    x[3, 128:160] = 3.0e38  # saturating magnitudes
    x[0, 160:192] = -0.0
    yield "edge groups 4x7168", x
    x = torch.randn(8, 1408, generator=generator) * 1.0e-4
    yield "small 8x1408", x
    x = torch.randn(8, 1408, generator=generator) * 1.0e4
    yield "large 8x1408", x


def test_triton_mxfp8_rows_match_b12x_bitwise() -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    checked = 0
    for name, host in _cases(device):
        source = host.to(device=device, dtype=torch.bfloat16).contiguous()
        ref_values, ref_rows, ref_mma = _reference(source)
        got_values, got_rows, got_mma = _candidate(source)
        torch.accelerator.synchronize(device)
        assert torch.equal(got_values, ref_values), f"{name}: E4M3 values differ"
        assert torch.equal(got_rows, ref_rows), f"{name}: row scales differ"
        assert torch.equal(got_mma, ref_mma), f"{name}: MMA-order scales differ"
        checked += 1
    assert checked >= 10
