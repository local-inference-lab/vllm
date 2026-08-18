# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN tile-level dequant reference (pure PyTorch).

Inverse of ``kvarn_store_tile_{k,v}``. Produces the dequantized tile in the
**rotated** frame; the caller is responsible for the inverse Hadamard
(matmul with H, which is its own inverse) and any subsequent attention math.

The Triton port (Stage 4) lives in ``triton_kvarn_decode.py`` and must
produce numerically equivalent outputs (cosine ≥ 0.999 vs this reference).
"""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops


def kvarn_hadamard(
    value: torch.Tensor,
    fallback_matrix: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the normalized Sylvester Hadamard transform.

    CUDA uses vLLM's in-place Hadacore kernel. The matrix path keeps CPU
    references and unsupported builds functional.
    """
    if value.is_cuda and hasattr(torch.ops._C, "hadacore_transform"):
        if out is None:
            out = value.contiguous().clone()
        else:
            out.copy_(value)
        transformed = ops.hadacore_transform(
            out.reshape(-1, value.shape[-1]), inplace=True
        )
        if transformed.data_ptr() != out.data_ptr():
            out.copy_(transformed.reshape_as(out))
        return out
    if out is None:
        return torch.matmul(value, fallback_matrix)
    torch.matmul(value, fallback_matrix, out=out)
    return out


def _unpack_4bit(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """Inverse of ``kvarn_store::_pack_4bit``.

    ``packed`` has shape ``[..., original_last_dim // 2]`` uint8; returns
    shape ``[..., original_last_dim]`` uint8 with values in [0, 15].
    """
    assert original_last_dim % 2 == 0
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.empty(
        *packed.shape[:-1],
        original_last_dim,
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _unpack_lowbit(
    packed: torch.Tensor, original_last_dim: int, bits: int
) -> torch.Tensor:
    """Unpack the dense little-endian bit stream produced by ``_pack_lowbit``."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    value_indices = torch.arange(
        original_last_dim, dtype=torch.int64, device=packed.device
    )
    bit_offsets = value_indices * bits
    bit_numbers = torch.arange(bits, dtype=torch.int64, device=packed.device)
    source_bits = bit_offsets[:, None] + bit_numbers[None, :]
    source_bytes = torch.div(source_bits, 8, rounding_mode="floor")
    shifts = source_bits.remainder(8)
    loaded = packed[..., source_bytes]
    bits_unpacked = (loaded >> shifts) & 1
    weights = 1 << bit_numbers
    return (bits_unpacked * weights).sum(dim=-1).to(torch.uint8)


def kvarn_dequant_tile_k(
    q_packed_uint8: torch.Tensor,
    s_col_K: torch.Tensor,
    zp_K: torch.Tensor,
    s_row_K: torch.Tensor,
    group: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one K tile back to the rotated ``[D, group]`` frame.

    Args:
        q_packed_uint8 : ``[D, group // (8//bits)]`` uint8.
        s_col_K        : ``[D]`` fp16  — absorbed per-channel scale.
        zp_K           : ``[D]`` fp16  — absorbed per-channel zero.
        s_row_K        : ``[group]`` fp16 — per-token-in-tile sinkhorn scale.
        group          : tile width in tokens.
        bits           : quant bit-width of K (default 4).

    Returns:
        ``[D, group]`` fp32 dequantized tile in the rotated frame.
        Identity: ``out[r,c] = (q[r,c] * s_col_K[r] + zp_K[r]) * s_row_K[c]``.
    """
    q = _unpack_lowbit(q_packed_uint8, group, bits).float()  # [D, group]
    s_col = s_col_K.float().unsqueeze(-1)  # [D, 1]
    zp = zp_K.float().unsqueeze(-1)  # [D, 1]
    s_row = s_row_K.float().unsqueeze(0)  # [1, group]
    return (q * s_col + zp) * s_row


def kvarn_dequant_tile_v(
    q_packed_uint8: torch.Tensor,
    s_col_V: torch.Tensor,
    s_row_V: torch.Tensor,
    zp_V: torch.Tensor,
    head_dim: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one V tile back to the rotated ``[group, D]`` frame.

    Args:
        q_packed_uint8 : ``[group, D // (8//bits)]`` uint8.
        s_col_V        : ``[D]`` fp16  — per-channel sinkhorn scale (untouched).
        s_row_V        : ``[group]`` fp16 — absorbed per-token-in-tile scale.
        zp_V           : ``[group]`` fp16 — absorbed per-token-in-tile zero.
        head_dim       : tile width in channels.
        bits           : quant bit-width of V (default 4; k4v2 uses 2).

    Returns:
        ``[group, head_dim]`` fp32 dequantized tile in the rotated frame.
        Identity: ``out[t,c] = (q[t,c] * s_row_V[t] + zp_V[t]) * s_col_V[c]``.
    """
    q = _unpack_lowbit(q_packed_uint8, head_dim, bits).float()  # [group, D]
    s_row = s_row_V.float().unsqueeze(-1)  # [group, 1]
    zp = zp_V.float().unsqueeze(-1)  # [group, 1]
    s_col = s_col_V.float().unsqueeze(0)  # [1, D]
    return (q * s_row + zp) * s_col
