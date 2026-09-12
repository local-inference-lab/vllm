# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free greedy reduction of base logits and sequential Markov bias."""

import torch

from vllm.triton_utils import tl, triton

_BLOCK = 2048


@triton.jit
def _first_argmax(av, ai, bv, bi):
    a_nan = av != av
    b_nan = bv != bv
    take_a = (a_nan & (~b_nan | (ai < bi))) | (
        ~a_nan & ~b_nan & ((av > bv) | ((av == bv) & (ai < bi)))
    )
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


@triton.jit
def _partial_argmax(
    base,
    bias,
    values,
    indices,
    base_stride,
    bias_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(base + row * base_stride + col, col < VOCAB, other=0)
    b = tl.load(bias + row * bias_stride + col, col < VOCAB, other=0)
    # Preserve the materialized torch.add rounding before comparing logits.
    summed = (
        (a.to(tl.float32) + b.to(tl.float32)).to(base.dtype.element_ty).to(tl.float32)
    )
    summed = tl.where(col < VOCAB, summed, -float("inf"))
    idx = tl.where(col < VOCAB, col, 2147483647)
    value, index = tl.reduce((summed, idx), 0, _first_argmax)
    tl.store(values + row * PARTS + part, value)
    tl.store(indices + row * PARTS + part, index)


@triton.jit
def _finish_argmax(
    values,
    indices,
    output,
    output_stride,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    _, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(output + row * output_stride, result)


def scratch_shape(max_requests: int, vocabulary_size: int) -> tuple[int, int]:
    return max_requests, triton.cdiv(vocabulary_size, _BLOCK)


def sample_greedy_markov(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    output: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """Write exactly ``(base_logits + markov_bias).argmax(-1)`` into output.

    Both inputs share FP32, FP16, or BF16 dtype and have contiguous vocabulary
    columns. Request strides may differ. The caller owns disjoint FP32/int32
    scratch of ``scratch_shape(capacity, vocabulary_size)`` and int32/int64
    output, which may be a strided speculative-token column. Ties, NaNs, and
    addition rounding follow PyTorch argmax. No proposal truncation is used.
    """
    rows, vocab = base_logits.shape
    parts = triton.cdiv(vocab, _BLOCK)
    _partial_argmax[(rows, parts)](
        base_logits,
        markov_bias,
        partial_values,
        partial_indices,
        base_logits.stride(0),
        markov_bias.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax[(rows,)](
        partial_values,
        partial_indices,
        output,
        output.stride(0),
        parts,
        triton.next_power_of_2(parts),
    )
