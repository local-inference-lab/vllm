# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-count-invariant GEMMs for the small-N projections of a prefill chunk.

A prefill chunk that runs as two row halves reproduces the bits of the whole
chunk only if every row-local kernel gives a row the same result whether the
call carries all rows or a subset. cuBLAS chooses its kernel and its split-K
factor from the problem shape, so for the three smallest-N GEMMs of a Kimi-K3
MoE layer (the router gate, K 7168 N 104 with fp32 output; the routed latent
projection ``routed_expert_down_proj``, K 7168 N 400 per rank; the tier-2
up-projection tail, K 3584 N 796 accumulated into a column window) a
4,608-row call and two 2,304-row calls accumulate in different orders and
differ in some rows. The shared experts' down-projection takes the kernel as
well when its weight is an unquantized bf16 tensor.

This module computes those GEMMs with one Triton kernel whose reduction order
is fixed by its tile constants alone: every output tile reduces the full K
extent inside one program, in ``BLOCK_K`` slabs, in fp32, with no split-K.
A row's result depends only on its own values, the weight and the constants,
not on the row count, the row's index or the tile it lands in, so a row
computed alone reproduces the bits of the same row inside a 4,608-row call.
The result is of the same precision class as cuBLAS (bf16 operands, fp32
accumulation, one rounding into the output) but not bit-identical to it.

Opt-in: ``VLLM_K3_INVARIANT_SMALL_N_GEMM=1``. Once on, calls of at least
``VLLM_K3_INVARIANT_GEMM_MIN_ROWS`` rows (default 32) take the fixed-order
kernel and smaller (decode-size) calls keep cuBLAS; the threshold only
separates decode from prefill, every prefill-size call of a shape takes the
same kernel. ``VLLM_K3_INVARIANT_GEMM_TILE=BLOCK_M,BLOCK_N,BLOCK_K,warps,
stages`` overrides the tile constants for measurement; a different tile is a
different (still row-count-invariant) summation order.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.triton_utils import tl, triton

_BLOCK_K = 64


@triton.jit(do_not_specialize=["M"])
def _fixed_order_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """``C[m, n] = sum_k A[m, k] B[k, n]`` (plus ``C`` when ``ACCUMULATE``)
    for one ``BLOCK_M x BLOCK_N`` output tile, reducing K in ``BLOCK_K``
    slabs in ascending order into an fp32 accumulator. Rows and columns past
    ``M`` and ``N`` are masked, which does not change the arithmetic of the
    rows inside."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m < M
    col_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k < K - k0
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = row_mask[:, None] & col_mask[None, :]
    if ACCUMULATE:
        acc += tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@functools.lru_cache(maxsize=1)
def settings() -> tuple[bool, int]:
    """``(enabled, min_rows)`` from the environment, read once."""
    enabled = os.getenv("VLLM_K3_INVARIANT_SMALL_N_GEMM", "0") == "1"
    min_rows = max(1, int(os.getenv("VLLM_K3_INVARIANT_GEMM_MIN_ROWS", "32")))
    return enabled, min_rows


@functools.lru_cache(maxsize=1)
def _tile_override() -> tuple[int, int, int, int, int] | None:
    raw = os.getenv("VLLM_K3_INVARIANT_GEMM_TILE", "")
    if not raw:
        return None
    parts = tuple(int(part) for part in raw.split(","))
    if len(parts) != 5 or any(part <= 0 for part in parts):
        raise ValueError(
            "VLLM_K3_INVARIANT_GEMM_TILE must be BLOCK_M,BLOCK_N,BLOCK_K,warps,stages"
        )
    return parts


def tiles_for(n: int) -> tuple[int, int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`` for an output
    width ``n``. Narrow outputs (one column tile) take shorter row tiles so a
    2,304-row call still fills the machine."""
    override = _tile_override()
    if override is not None:
        return override
    block_n = 64 if n <= 64 else 128
    block_m = 32 if n <= 128 else 64
    return block_m, block_n, _BLOCK_K, 4, 3


def applies_to(*tensors: torch.Tensor) -> bool:
    """Whether the fixed-order kernel handles a call whose row count is the
    first tensor's: the feature is on, the call is prefill-size and every
    operand is a 2-D CUDA bf16 tensor."""
    enabled, min_rows = settings()
    if not enabled or not tensors or tensors[0].shape[0] < min_rows:
        return False
    return all(
        t.ndim == 2 and t.dtype == torch.bfloat16 and t.device.type == "cuda"
        for t in tensors
    )


def _check(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("fixed-order GEMM takes 2-D operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions differ: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if tuple(c.shape) != (a.shape[0], b.shape[1]):
        raise ValueError(
            f"output shape {tuple(c.shape)} does not match [{a.shape[0]}, {b.shape[1]}]"
        )
    if a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("operands must share a bf16 or fp16 dtype")
    if c.dtype not in (a.dtype, torch.float32):
        raise ValueError("output must be the operand dtype or fp32")
    if not (a.device == b.device == c.device):
        raise ValueError("operands and output must share a device")


def _launch(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, *, accumulate: bool
) -> None:
    _check(a, b, c)
    m, k = a.shape
    n = b.shape[1]
    block_m, block_n, block_k, warps, stages = tiles_for(n)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fixed_order_gemm_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ACCUMULATE=accumulate,
        num_warps=warps,
        num_stages=stages,
    )


def mm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``a @ b`` with the fixed reduction order; ``out`` may be any 2-D
    strided tensor of the operand dtype or fp32, ``out_dtype`` selects the
    dtype of a fresh output."""
    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]), dtype=out_dtype or a.dtype, device=a.device
        )
    elif out_dtype is not None and out.dtype != out_dtype:
        raise ValueError("out and out_dtype disagree")
    _launch(a, b, out, accumulate=False)
    return out


def addmm_(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``c += a @ b`` in place, the sum formed in fp32 and rounded once into
    ``c`` (the beta-add epilogue of ``torch.Tensor.addmm_``)."""
    _launch(a, b, c, accumulate=True)
    return c
