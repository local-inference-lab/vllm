# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack one padded row-parallel projection input without padding other ranks."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_projection_shard(
    x_ptr,
    out_ptr,
    rows: tl.constexpr,
    input_width: tl.constexpr,
    stride_row: tl.constexpr,
    start: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // width
    col = offsets % width + start
    value = tl.load(
        x_ptr + row * stride_row + col,
        mask=(row < rows) & (col < input_width),
        other=0,
    )
    tl.store(out_ptr + offsets, value, mask=row < rows)


def pack_projection_shard(x: torch.Tensor, start: int, width: int) -> torch.Tensor:
    """Copy a column shard, filling columns beyond the logical input with zero.

    The result equals padding the full input, selecting a rank's shard and
    making it contiguous. Retaining the padded K width preserves the GEMM
    shape, weight layout and reduction order. The input is never modified.
    """
    if x.ndim != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise ValueError("Projection shard requires CUDA rows with unit column stride")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Projection shard requires bf16, fp16 or fp32 input")
    if start < 0 or width <= 0:
        raise ValueError(
            "Projection shard needs a nonnegative start and positive width"
        )
    if start + width <= x.shape[1]:
        return x.narrow(1, start, width).contiguous()
    out = x.new_empty((x.shape[0], width))
    if out.numel():
        _pack_projection_shard[(triton.cdiv(out.numel(), 512),)](
            x,
            out,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            start,
            width,
            BLOCK=512,
        )
    return out
