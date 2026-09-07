# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton replica of the b12x MXFP8 row quantizer (Kimi-K3 decode glue).

The b12x dense GEMM consumes activations as E4M3 values with one UE8M0 scale
per 32-element group, the scales stored twice: row-major (`scale_rows`,
``[rows, K // 32]``) and in the GEMM's MMA order (`scale_mma`, flat, offset
``row32 * 16 + row4 * 4 + tile_m * ceil(groups / 4) * 512 + k4 + tile_k *
512`` with ``row32 = row % 32``, ``row4 = row // 32 % 4``, ``tile_m = row //
128``, ``k4 = group % 4``, ``tile_k = group // 4``). The served decode step
runs that quantizer as its own kernel after every norm that feeds a wide
projection (KDA in-projection K = 7,168 and out-projection K = 1,408).

This module reproduces the quantizer's arithmetic bit for bit so a norm
kernel can emit the quantized rows in its epilogue:

* group scale: ``max_abs`` of the 32 bf16 values (as fp32), ``x = max_abs *
  fp32(1 / 448)``, UE8M0 byte = the exponent of ``x`` rounded up to the next
  power of two when the mantissa is non-zero (``pow2_ceil_ue8m0``), byte 127
  when ``max_abs == 0``;
* inverse scale: ``2 ** (127 - byte)`` built from the exponent field
  (``(254 - byte) << 23``), 0 for byte 0;
* values: ``value * inverse_scale`` converted to E4M3 with round-to-nearest
  and saturation (``cvt.rn.satfinite.e4m3x2.f32``).

``quantize_mxfp8_rows_triton`` is the standalone form (one launch); the GPU
test compares it against ``b12x._lib.quant.mxfp8_rows.quantize_mxfp8_rows_cute``
on random, zero, subnormal-scale and saturating rows.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

GROUP = 32


@triton.jit
def _mxfp8_rows_kernel(
    source_ptr,
    values_ptr,
    scale_rows_ptr,
    scale_mma_ptr,
    stride_source_m,
    stride_values_m,
    stride_scale_rows_m,
    groups_k: tl.constexpr,
    mma_tile_groups: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group_block = tl.program_id(1)
    group_ids = group_block * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    group_mask = group_ids < groups_k
    lane = tl.arange(0, GROUP)
    columns = group_ids[:, None] * GROUP + lane[None, :]
    mask = group_mask[:, None] & (lane[None, :] < GROUP)
    values = tl.load(
        source_ptr + row * stride_source_m + columns, mask=mask, other=0.0
    ).to(tl.float32)

    # UE8M0 scale per group: pow2_ceil of max_abs / 448 (fp32 multiply by the
    # rounded reciprocal, as the b12x kernel does), byte 127 for an all-zero
    # group.
    max_abs = tl.max(tl.abs(values), axis=1)
    scaled = max_abs * 0.002232142857142857  # fp32(1.0 / 448.0)
    bits = scaled.to(tl.int32, bitcast=True)
    mantissa = bits & 0x007FFFFF
    bumped = tl.where(mantissa != 0, (bits + 0x00800000) & 0x7F800000, bits)
    scale_byte = (bumped >> 23) & 0xFF
    scale_byte = tl.where(max_abs == 0.0, 127, scale_byte)
    inverse_bits = tl.maximum(254 - scale_byte, 0) << 23
    inverse = inverse_bits.to(tl.float32, bitcast=True)
    inverse = tl.where(scale_byte == 0, 0.0, inverse)

    quantized = (values * inverse[:, None]).to(tl.float8e4nv)
    tl.store(
        values_ptr + row * stride_values_m + columns,
        quantized,
        mask=mask,
    )
    scale_u8 = scale_byte.to(tl.uint8)
    tl.store(
        scale_rows_ptr + row * stride_scale_rows_m + group_ids,
        scale_u8,
        mask=group_mask,
    )
    row32 = row % 32
    row4 = (row // 32) % 4
    tile_m = row // 128
    k4 = group_ids % 4
    tile_k = group_ids // 4
    mma_offsets = (
        row32 * 16 + row4 * 4 + tile_m * (mma_tile_groups * 512) + k4 + tile_k * 512
    )
    tl.store(scale_mma_ptr + mma_offsets, scale_u8, mask=group_mask)


def mxfp8_scale_mma_numel(rows: int, k: int) -> int:
    """Size of the MMA-order scale buffer for ``rows`` rows of width ``k``."""
    groups = k // GROUP
    return ((rows + 127) // 128) * ((groups + 3) // 4) * 512


def quantize_mxfp8_rows_triton(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
) -> None:
    """Quantize contiguous bf16/fp16 rows into the b12x dense-GEMM layouts.

    ``values`` is ``[rows, K]`` float8_e4m3fn (or uint8 of the same bytes),
    ``scale_rows`` ``[rows, K // 32]`` uint8 and ``scale_mma`` a flat uint8
    buffer of at least ``mxfp8_scale_mma_numel(rows, K)`` bytes.
    """
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("source must be a contiguous [rows, K] tensor")
    if source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"source must be bf16 or fp16, got {source.dtype}")
    rows, k = (int(v) for v in source.shape)
    if k % GROUP:
        raise ValueError(f"K must be a multiple of {GROUP}, got {k}")
    groups = k // GROUP
    if values.shape != (rows, k) or values.element_size() != 1:
        raise ValueError("values must be [rows, K] of one-byte elements")
    if scale_rows.shape != (rows, groups) or scale_rows.dtype != torch.uint8:
        raise ValueError("scale_rows must be [rows, K // 32] uint8")
    if scale_mma.dtype != torch.uint8 or scale_mma.numel() < mxfp8_scale_mma_numel(
        rows, k
    ):
        raise ValueError("scale_mma is too small or not uint8")
    if rows == 0:
        return
    groups_per_program = 8 if groups >= 8 else triton.next_power_of_2(groups)
    grid = (rows, triton.cdiv(groups, groups_per_program))
    values_view = (
        values.view(torch.float8_e4m3fn)
        if values.dtype != torch.float8_e4m3fn
        else values
    )
    _mxfp8_rows_kernel[grid](
        source,
        values_view,
        scale_rows,
        scale_mma,
        source.stride(0),
        values_view.stride(0),
        scale_rows.stride(0),
        groups_k=groups,
        mma_tile_groups=(groups + 3) // 4,
        GROUPS_PER_PROGRAM=groups_per_program,
        num_warps=4,
    )
