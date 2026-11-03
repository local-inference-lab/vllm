# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN MLA (arXiv:2606.03458) packed-record layout and reference codec.

KVarN = Hadamard rotation + round-to-nearest quantization of the glm5_next
MLA compressed latent, with per-token asymmetric RTN on both planes (the
ninfer record scheme). The latent serves both the score (K) and
value-aggregation (V) attention roles from one cache record, so each record
carries TWO code planes at independent bit widths (``k_bits`` score path,
``v_bits`` value path), each with its own inline fp16 scale + zero-point.

Dtype grammar: ``kvarn_mla_k<k>[v<v>]_g<group>`` where ``group`` equals the
vLLM cache block size (one tile == one page). Launched variants:
``kvarn_mla_k4v4_g256``, ``kvarn_mla_k5v5_g256``, ``kvarn_mla_k4v2_g256``,
and the SINGLE-PLANE ``kvarn_mla_k4_g256``.

Single-plane (``v_bits == 0``): ONE code plane over the rotated latent --
one RTN grid, one inline fp16 scale + zero-point -- consumed by BOTH the
score (K) and value (V) paths through their folded projections
(``k_hat == v_hat``). The dual-plane variants code the same latent twice at
independent bit widths (8.25 bits/dim for k4v4); the single-plane variant is
the shared-code point at ~4.3 bits/dim (record 264B; pool 297 B/token with
the 33 B C4 index tail — records end exactly on the C4 boundary). The
record keeps the dual-plane field layout with the ``v_codes`` plane empty
(0 bytes) and ``v_scale``/``v_zp`` stored as zero, so all record offsets and
the mod-33 geometry rule are variant-independent.

Record layout (per token, latent_dim=512), little-endian:

======================  ==========================================
field                   size
======================  ==========================================
k_codes                 ceil(latent_dim * k_bits / 8)   (per-token)
v_codes                 ceil(latent_dim * v_bits / 8)   (per-token)
k_scale   (fp16)        2
k_zp      (fp16)        2
v_scale   (fp16)        2
v_zp      (fp16)        2
======================  ==========================================

Record width: 8-byte aligned content, padded by the mod-33 rule
(``0 < r % 33 <= 21``) so the GLM C4-page alignment gap (records end ->
8448B indexer tail) fits the reserved per-tile tail region and the
parent-page stride stays an exact multiple of the 8448-byte C4 page
(33 * 256 = 8448). The per-tile tail region (6400B for group=256) remains
reserved in the page geometry but carries no scales: the per-token record
scheme keeps every scale inline, so the tail is slack.

Dequantization identities (rotated frame; the Hadamard inverse lives in
the folded W_uv / absorbed Q projections):

    k_hat[r, c] = k_codes[r, c] * k_scale[r] + k_zp[r]
    v_hat[r, c] = v_codes[r, c] * v_scale[r] + v_zp[r]
"""

from __future__ import annotations

import dataclasses
import math

import torch

from vllm.v1.kv_cache_interface import parse_kvarn_mla_dtype

# Reserved per-tile tail region (mirrors b12x _kvarn_writer._TILE_TAIL_BYTES):
# sized so tile_stride = group*record_bytes + 6400 is an exact multiple of
# the 8448B C4 page (256*536 + 6400 = 143616 = 17 * 8448). Carries no data
# under the per-token record scheme; kept for C4 page alignment.
_KVARN_TILE_TAIL_BYTES = 6400

# GLM C4 index page: 64 pools x 132 bytes = 8448 bytes (33 * 256).
_KVARN_INDEX_PAGE_BYTES = 8448

__all__ = [
    "KvarnMlaLayout",
    "kvarn_mla_layout_from_dtype",
    "hadamard_matrix",
    "pack_kvarn_mla_records",
    "unpack_kvarn_mla_records",
    "pack_bits",
    "unpack_bits",
]


@dataclasses.dataclass(frozen=True)
class KvarnMlaLayout:
    """Byte geometry of the KVarN MLA record.

    ``v_bits == 0`` selects the single-plane variant (one shared code plane,
    ``v_code_bytes == 0``); otherwise the dual-rate record with independent
    ``k_bits``/``v_bits`` planes.
    """

    k_bits: int
    v_bits: int
    group: int
    latent_dim: int = 512

    @property
    def k_code_bytes(self) -> int:
        return math.ceil(self.latent_dim * self.k_bits / 8)

    @property
    def v_code_bytes(self) -> int:
        return math.ceil(self.latent_dim * self.v_bits / 8)

    @property
    def content_bytes(self) -> int:
        return self.k_code_bytes + self.v_code_bytes + 8

    @property
    def record_bytes(self) -> int:
        """Padded record width.

        Dual-plane variants: 8-byte aligned content PLUS the mod-33 rule
        (``0 < r % 33 <= 21``) keeps the GLM C4-page alignment gap (records
        end -> 8448B indexer tail) at >= 3072 bytes so the reserved per-tile
        tail region fits inside it and the parent-page stride stays an exact
        multiple of the 8448-byte C4 page (33 * 256 = 8448).

        Single-plane variants (``v_bits == 0``) carry no tile-tail data, so
        the mod-33 bump is skipped and ``record % 33 == 0`` is allowed:
        records end exactly on a C4 boundary and the FP8 index tail starts
        flush (264 = 8 C4 pages for k4), removing the 25 B/token per-tile
        slack the dual rule pads in.
        """
        content = self.k_code_bytes + self.v_code_bytes + 8
        r = (content + 7) // 8 * 8
        if self.v_bits == 0:
            return r
        # m == 0 puts the records end exactly on the C4 boundary (no gap);
        # require 0 < m <= 21 so the gap fits the reserved tile tail.
        while r % 33 == 0 or r % 33 > 21:
            r += 8
        return r

    @property
    def _tile_tail_bytes(self) -> int:
        """Reserved per-tile tail region: dual variants reserve the legacy
        VarN/tile-scale region; single-plane records carry every scale
        inline and records end on the C4 boundary, so the region is zero."""
        return 0 if self.v_bits == 0 else _KVARN_TILE_TAIL_BYTES

    @property
    def tile_stride_bytes(self) -> int:
        """Kernel tile stride: one tile of ``group`` records plus the reserved
        per-tile tail region, rounded up to a whole C4 index page so the FP8
        index tail placement (which aligns to C4 pages) sees the same stride
        as the kernels."""
        raw = self.record_bytes * self.group + self._tile_tail_bytes
        return (
            (raw + _KVARN_INDEX_PAGE_BYTES - 1) // _KVARN_INDEX_PAGE_BYTES
        ) * _KVARN_INDEX_PAGE_BYTES

    @property
    def page_tail_bytes_per_token(self) -> int:
        """Per-token page tail: just enough so one page of ``group`` tokens
        holds the rounded tile span plus the FP8 index tail (one C4 page).
        The result keeps ``page_size_bytes`` an exact C4-page multiple."""
        needed = self.tile_stride_bytes + _KVARN_INDEX_PAGE_BYTES
        return (needed - self.record_bytes * self.group) // self.group

    @property
    def page_size_bytes(self) -> int:
        return (
            self.record_bytes * self.group + self.page_tail_bytes_per_token * self.group
        )

    @property
    def k_code_offset(self) -> int:
        return 0

    @property
    def v_code_offset(self) -> int:
        return self.k_code_bytes

    @property
    def k_scale_offset(self) -> int:
        return self.v_code_offset + self.v_code_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_scale_offset + 2

    @property
    def v_scale_offset(self) -> int:
        return self.k_zp_offset + 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_scale_offset + 2


def kvarn_mla_layout_from_dtype(kv_cache_dtype: str) -> KvarnMlaLayout:
    parsed = parse_kvarn_mla_dtype(kv_cache_dtype)
    if parsed is None:
        raise ValueError(f"Not a kvarn MLA dtype: {kv_cache_dtype!r}")
    k_bits, v_bits, group = parsed
    return KvarnMlaLayout(k_bits=k_bits, v_bits=v_bits, group=group)


@torch.no_grad()
def hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix (dim must be a power of two)."""
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == dim
    return h / math.sqrt(dim)


def kvarn_cache_dtype_enabled(cache_dtype: object) -> bool:
    """True when a cache config selects a kvarn MLA dtype."""
    return isinstance(cache_dtype, str) and cache_dtype.startswith("kvarn_mla_")


def rotate_latent(x: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """Rotate the latent dimension of ``x`` with ``h_t = H^T`` (H orthonormal).

    ``x``: ``[..., latent_dim]`` or ``[..., heads, latent_dim]``; the rotation
    applies to the last dimension only, broadcasting over all leading dims.
    """
    return torch.matmul(x, h_t)


@torch.no_grad()
def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer codes ([..., C], 0 <= code < 2**bits) little-endian.

    Returns ``[..., ceil(C * bits / 8)]`` uint8. Bit i of the logical stream
    occupies byte ``i // 8``, bit position ``i % 8`` (LSB-first, dense).
    """
    orig_shape = values.shape
    flat = values.reshape(-1, orig_shape[-1]).to(torch.int32)
    cols = flat.shape[1]
    out_bytes = math.ceil(cols * bits / 8)
    acc = torch.zeros(flat.shape[0], out_bytes, dtype=torch.int32, device=values.device)
    bit_index = torch.arange(cols, device=values.device) * bits
    rows = flat.shape[0]
    for b in range(bits):
        bit_val = (flat >> b) & 1
        # position of this code-bit in the dense LSB-first stream
        pos = bit_index + b
        dst_byte = (pos // 8).long().unsqueeze(0).expand(rows, cols)
        dst_shift = (pos % 8).to(torch.int32)
        acc.scatter_add_(1, dst_byte, bit_val << dst_shift)
    return acc.to(torch.uint8).reshape(*orig_shape[:-1], out_bytes)


@torch.no_grad()
def unpack_bits(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns int32 codes [..., cols]."""
    orig_shape = packed.shape
    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    rows = flat.shape[0]
    bit_index = torch.arange(cols, device=packed.device) * bits
    codes = torch.zeros(rows, cols, dtype=torch.int32, device=packed.device)
    for b in range(bits):
        pos = bit_index + b
        src_byte = flat[:, (pos // 8).long()]
        codes |= ((src_byte >> (pos % 8).to(torch.int32)) & 1) << b
    return codes.reshape(*orig_shape[:-1], cols)


def _rtn_codes(
    x: torch.Tensor, bits: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-slice RTN along ``dim``.

    Returns ``(codes, scale, zp)`` with ``x_hat = codes * scale + zp``.
    """
    levels = 2**bits - 1
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / levels
    codes = torch.round((x - mn) / scale).clamp_(0, levels).to(torch.int32)
    # zp folded: x_hat = (codes + mn/scale_rounded...) -> store scale & zp
    zp = mn
    return codes, scale.squeeze(dim), zp.squeeze(dim)


@torch.no_grad()
def pack_kvarn_mla_records(
    latent: torch.Tensor,
    layout: KvarnMlaLayout,
) -> torch.Tensor:
    """Pack rotated latents into per-token KVarN records.

    Args:
        latent: ``[T, latent_dim]`` float tensor, already Hadamard-rotated.
        layout: record geometry.

    Returns:
        ``records [T, record_bytes]`` uint8. Each record carries the code
        plane(s) and their inline fp16 scale/zero-point (asymmetric RTN
        per token, ninfer scheme — no tile-coupled scales, no VarN).
        Single-plane layouts (``v_bits == 0``) carry one plane; the ``v_*``
        fields stay zero.
    """
    tokens, latent_dim = latent.shape
    assert latent_dim == layout.latent_dim
    x = latent.to(torch.float32)
    # Score path: per-token asymmetric RTN on the rotated latent.
    k_codes, k_scale, k_zp = _rtn_codes(x, layout.k_bits, dim=1)
    single_plane = layout.v_bits == 0
    if single_plane:
        # Shared-code point: no second plane; v fields stay zero and both
        # attention paths consume k_hat through their folded projections.
        v_codes, v_scale, v_zp = None, None, None
    else:
        # Value path: per-token asymmetric RTN at the v_bits width.
        v_codes, v_scale, v_zp = _rtn_codes(x, layout.v_bits, dim=1)
    # k_hat = k_codes * k_scale + k_zp ; v_hat = v_codes * v_scale + v_zp

    records = torch.zeros(
        tokens, layout.record_bytes, dtype=torch.uint8, device=latent.device
    )
    rec_f16 = records.view(torch.float16)
    records[:, layout.k_code_offset : layout.k_code_offset + layout.k_code_bytes] = (
        pack_bits(k_codes, layout.k_bits).to(torch.uint8)
    )
    if not single_plane:
        records[
            :, layout.v_code_offset : layout.v_code_offset + layout.v_code_bytes
        ] = pack_bits(v_codes, layout.v_bits).to(torch.uint8)
        rec_f16[:, layout.v_scale_offset // 2] = v_scale.to(torch.float16)
        rec_f16[:, layout.v_zp_offset // 2] = v_zp.to(torch.float16)
    rec_f16[:, layout.k_scale_offset // 2] = k_scale.to(torch.float16)
    rec_f16[:, layout.k_zp_offset // 2] = k_zp.to(torch.float16)
    return records


@torch.no_grad()
def unpack_kvarn_mla_records(
    records: torch.Tensor,
    layout: KvarnMlaLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize records back to ``(k_hat, v_hat)`` [T, latent_dim] f32.

    Reference (non-fused) inverse of :func:`pack_kvarn_mla_records`.
    """
    latent_dim = layout.latent_dim
    rec_f16 = records.view(torch.float16)

    k_codes = unpack_bits(
        records[:, layout.k_code_offset : layout.k_code_offset + layout.k_code_bytes],
        layout.k_bits,
        latent_dim,
    ).to(torch.float32)
    k_scale = rec_f16[:, layout.k_scale_offset // 2].to(torch.float32)
    k_zp = rec_f16[:, layout.k_zp_offset // 2].to(torch.float32)

    # Score path: k_hat = codes * k_scale + k_zp (rotated frame).
    k_hat = k_codes * k_scale.unsqueeze(1) + k_zp.unsqueeze(1)
    if layout.v_bits == 0:
        # Single-plane: the value path reads the shared k_hat codes.
        return k_hat, k_hat
    v_codes = unpack_bits(
        records[:, layout.v_code_offset : layout.v_code_offset + layout.v_code_bytes],
        layout.v_bits,
        latent_dim,
    ).to(torch.float32)
    v_scale = rec_f16[:, layout.v_scale_offset // 2].to(torch.float32)
    v_zp = rec_f16[:, layout.v_zp_offset // 2].to(torch.float32)

    # Value path: v_hat = codes * v_scale + v_zp (rotated frame).
    v_hat = v_codes * v_scale.unsqueeze(1) + v_zp.unsqueeze(1)

    return k_hat, v_hat
