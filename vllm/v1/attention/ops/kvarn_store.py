# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN tile-level store reference (pure PyTorch).

Quantizes one tile of K (or V) per call. The Triton port (Stage 4) lives in
``triton_kvarn_store.py`` and must produce byte-identical outputs.

Inputs to the K path are tile-shaped ``[D, group]`` (channels × tokens — the
KIVI K-axis orientation) **after** Hadamard rotation. Inputs to the V path
are tile-shaped ``[group, D]`` (tokens × channels — the KIVI V-axis
orientation) also after Hadamard rotation. The Hadamard rotation is applied
externally via a cuBLAS GEMM, identically to TurboQuant's MSE path.

The output is a packed record matching the cache layout from
``KVarNConfig`` — see that file for byte offsets.
"""

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
)


def _rtn_range(t: torch.Tensor, dim: int):
    """Per-row range. With KVARN_RTN_QUANTILE=q > 0 (e.g. 0.005), uses
    percentiles [q, 1-q] instead of min/max — values outside get clamped at
    quantize time, sacrificing outliers for finer bulk resolution. Critical
    for k2v2 on models like Qwen3-30B-A3B-Thinking where K outliers
    (max/std ≈ 6) waste 2-bit resolution.
    """
    q_str = os.environ.get("KVARN_RTN_QUANTILE", "")
    if q_str and float(q_str) > 0:
        q = float(q_str)
        lo = torch.quantile(t, q, dim=dim, keepdim=True)
        hi = torch.quantile(t, 1.0 - q, dim=dim, keepdim=True)
        return lo, hi
    return t.amin(dim=dim, keepdim=True), t.amax(dim=dim, keepdim=True)


# ──────────────────────────────────────────────────────────────────────────────
# Asymmetric per-row RTN
# ──────────────────────────────────────────────────────────────────────────────


def _asymmetric_rtn_per_row(
    tile: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row asymmetric RTN over the full row (no sub-grouping).

    Args:
        tile: [R, C] fp32.
        bits: Integer width in [1, 8].

    Returns:
        q     [R, C] int32 in [0, 2^bits - 1]
        scale [R, 1] fp32
        zp    [R, 1] fp32  (= row minimum)
    """
    qmax = (1 << bits) - 1
    lo = tile.amin(dim=1, keepdim=True)
    hi = tile.amax(dim=1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((tile - zp) / scale), 0, qmax).to(torch.int32)
    return q, scale, zp


def _weighted_affine_refit(
    tile: torch.Tensor,
    q: torch.Tensor,
    other_scale: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit fixed integer codes for original-space squared error."""
    weights = other_scale.float().square().unsqueeze(-2)
    q_float = q.float()
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    q_mean = (weights * q_float).sum(dim=-1, keepdim=True) / weight_sum
    tile_mean = (weights * tile).sum(dim=-1, keepdim=True) / weight_sum
    q_centered = q_float - q_mean
    denominator = (weights * q_centered.square()).sum(dim=-1, keepdim=True)
    numerator = (weights * q_centered * (tile - tile_mean)).sum(dim=-1, keepdim=True)
    fitted_scale = (numerator / denominator.clamp_min(1e-20)).clamp_min(1e-10)
    fitted_zp = tile_mean - fitted_scale * q_mean
    usable = denominator > 1e-20
    return (
        torch.where(usable, fitted_scale, scale),
        torch.where(usable, fitted_zp, zp),
    )


def _quantize_rows(
    tile: torch.Tensor,
    bits: int,
    other_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qmax = (1 << bits) - 1
    lo, hi = _rtn_range(tile, dim=-1)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((tile - zp) / scale), 0, qmax).to(torch.int32)
    default_refit = "1" if bits == 5 else "0"
    if os.environ.get("KVARN_AFFINE_REFIT", default_refit) == "1":
        scale, zp = _weighted_affine_refit(tile, q, other_scale, scale, zp)
    return q, scale, zp


def _pack_4bit(q: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit ints (last dim even) into uint8 pairs, two-per-byte.

    Layout: low nibble = even-indexed, high nibble = odd-indexed.
    """
    assert q.shape[-1] % 2 == 0, "last dim must be even for 4-bit pairing"
    q = q.to(torch.uint8) & 0xF
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _pack_2bit(q: torch.Tensor) -> torch.Tensor:
    """Pack four 2-bit integers per byte in little-endian order."""
    assert q.shape[-1] % 4 == 0, "last dim must be divisible by four"
    values = (q.to(torch.uint8) & 0x3).view(*q.shape[:-1], -1, 4)
    return (
        values[..., 0]
        | (values[..., 1] << 2)
        | (values[..., 2] << 4)
        | (values[..., 3] << 6)
    )


def _pack_lowbit(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes into a dense little-endian bit stream.

    The code at index ``i`` begins at bit ``i * bits``. This is the BeeLlama
    KVarN layout and supports widths that do not divide a byte, notably K5/V5.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    q = q.to(torch.uint8) & ((1 << bits) - 1)
    last_dim = q.shape[-1]
    packed_bytes = (last_dim * bits + 7) // 8
    if bits == 2 and last_dim % 4 == 0:
        return _pack_2bit(q)
    if bits == 4 and last_dim % 2 == 0:
        return _pack_4bit(q)
    bit_positions = torch.arange(packed_bytes * 8, dtype=torch.int64, device=q.device)
    source_indices = torch.div(bit_positions, bits, rounding_mode="floor")
    valid = source_indices < last_dim
    source_indices = source_indices.clamp_max(max(last_dim - 1, 0))
    source_bits = bit_positions.remainder(bits)
    source = q[..., source_indices]
    bit_values = ((source >> source_bits) & 1) * valid
    bit_values = bit_values.reshape(*q.shape[:-1], packed_bytes, 8)
    byte_weights = 1 << torch.arange(8, dtype=torch.uint8, device=q.device)
    return (bit_values * byte_weights).sum(dim=-1).to(torch.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# K tile store: per-channel RTN, [D, group] orientation
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_store_tile_k(
    k_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated K tile.

    Args:
        k_tile_rotated: ``[D, group]`` fp32 / fp16 — channels × tokens, *after*
            Hadamard rotation along head_dim. Caller is responsible for the
            external ``(K @ H).T`` GEMM.
        bits: key bit-width (typically 4).
        sinkhorn_iters: log-domain iterations.

    Returns dict with packed cache record:
        q_packed_uint8 : ``[D, group/2]`` uint8 — 4-bit pairs (low=even, high=odd)
        s_col_K        : ``[D]``        fp16   — absorbed per-channel scale
        zp_K           : ``[D]``        fp16   — absorbed per-channel zero
        s_row_K        : ``[group]``    fp16   — per-token-in-tile sinkhorn scale
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    tile = k_tile_rotated.float()
    D, G = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    # In [D, group] orientation:
    #   s_col_sinkhorn is [1, G] = per-token-in-tile
    #   s_row_sinkhorn is [D, 1] = per-channel
    s_chan = s_row_sinkhorn  # [D, 1]
    s_tok = s_col_sinkhorn  # [1, G]

    q, rtn_scale, rtn_zp = _quantize_rows(balanced, bits, s_tok.squeeze(0))
    s_col_K = (s_chan * rtn_scale).squeeze(-1)
    zp_K = (s_chan * rtn_zp).squeeze(-1)
    s_row_K = s_tok.squeeze(0)

    q_packed = _pack_lowbit(q, bits)

    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K.to(torch.float16),
        "zp_K": zp_K.to(torch.float16),
        "s_row_K": s_row_K.to(torch.float16),
    }


# ──────────────────────────────────────────────────────────────────────────────
# V tile store: per-token RTN, [group, D] orientation
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_store_tile_k_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched K-path RTN + scale absorption + 4-bit packing.

    Assumes the sinkhorn step already ran (e.g. via the Triton kernel).

    Args:
        balanced : ``[N, D, group]`` fp32 — sinkhorn-balanced K tiles.
        s_col    : ``[N, group]`` fp32 — per-token sinkhorn scale (axis-1).
        s_row    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-0).
        bits     : key bit-width (4).

    Returns dict of per-tile (N-batched) tensors:
        q_packed_uint8 : ``[N, D, group/2]`` uint8
        s_col_K        : ``[N, D]``         fp16 — absorbed per-channel scale
        zp_K           : ``[N, D]``         fp16 — absorbed per-channel zero
        s_row_K        : ``[N, group]``     fp16 — per-token sinkhorn scale
    """
    q, scale, zp = _quantize_rows(balanced, bits, s_col)
    s_col_K = s_row * scale.squeeze(-1)
    zp_K = s_row * zp.squeeze(-1)
    s_row_K = s_col
    s_col_K = s_col_K.to(torch.float16)
    zp_K = zp_K.to(torch.float16)
    s_row_K = s_row_K.to(torch.float16)
    q_packed = _pack_lowbit(q, bits)
    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K,
        "zp_K": zp_K,
        "s_row_K": s_row_K,
    }


def kvarn_store_tile_v_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched V-path RTN + scale absorption + 4-bit packing.

    Args:
        balanced : ``[N, group, D]`` fp32 — sinkhorn-balanced V tiles.
        s_col    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-1).
        s_row    : ``[N, group]`` fp32 — per-token-in-tile sinkhorn scale (axis-0).
        bits     : value bit-width (4).

    Returns dict of per-tile (N-batched) tensors mirroring `kvarn_store_tile_v`.
    """
    q, scale, zp = _quantize_rows(balanced, bits, s_col)
    s_row_V = s_row * scale.squeeze(-1)
    zp_V = s_row * zp.squeeze(-1)
    s_col_V = s_col
    s_row_V = s_row_V.to(torch.float16)
    zp_V = zp_V.to(torch.float16)
    s_col_V = s_col_V.to(torch.float16)
    q_packed = _pack_lowbit(q, bits)
    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V,
        "s_row_V": s_row_V,
        "zp_V": zp_V,
    }


def kvarn_store_tile_v(
    v_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated V tile.

    Args:
        v_tile_rotated: ``[group, D]`` fp32 / fp16 — tokens × channels, *after*
            Hadamard rotation along head_dim. Caller is responsible for the
            external ``V @ H`` GEMM.
        bits: value bit-width (typically 4).
        sinkhorn_iters: log-domain iterations.

    Returns dict with packed cache record:
        q_packed_uint8 : ``[group, D/2]`` uint8 — 4-bit pairs
        s_col_V        : ``[D]``          fp16   — per-channel scale (untouched)
        s_row_V        : ``[group]``      fp16   — absorbed per-token-in-tile scale
        zp_V           : ``[group]``      fp16   — absorbed per-token-in-tile zero
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    tile = v_tile_rotated.float()
    G, D = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    # In [group, D] orientation:
    #   s_col_sinkhorn is [1, D] = per-channel
    #   s_row_sinkhorn is [G, 1] = per-token-in-tile
    s_chan = s_col_sinkhorn  # [1, D]
    s_tok = s_row_sinkhorn  # [G, 1]

    q, rtn_scale, rtn_zp = _quantize_rows(balanced, bits, s_chan.squeeze(0))
    s_row_V = (s_tok * rtn_scale).squeeze(-1)
    zp_V = (s_tok * rtn_zp).squeeze(-1)
    s_col_V = s_chan.squeeze(0)

    q_packed = _pack_lowbit(q, bits)

    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V.to(torch.float16),
        "s_row_V": s_row_V.to(torch.float16),
        "zp_V": zp_V.to(torch.float16),
    }
