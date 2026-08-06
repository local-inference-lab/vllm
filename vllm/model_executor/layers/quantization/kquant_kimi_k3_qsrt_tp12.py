# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict rank-local reader for kquant's fixed Kimi-K3 QSRT TP12 slab.

The serving path intentionally does not import kquant. This module owns the
small frozen disk ABI needed to turn one rank section into B12X's public
P24/P33 pair inputs and the production X4T kept-tier tensors.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.kquant_x4t import (
        X4TScaleComponents,
    )


TP_SIZE = 12
EXPERTS = 896
HIDDEN = 3584
INTERMEDIATE = 3072
LOCAL_INTERMEDIATE = INTERMEDIATE // TP_SIZE
ALIGNMENT = 4096
HEADER_BYTES = ALIGNMENT
FORMAT_BYTES = ALIGNMENT
FORMAT_TABLE_BYTES = EXPERTS
SHARED_SCALE_BYTES = 6 * ALIGNMENT
FORMAT_MXFP4 = 0xFF
PAIR_BYTES = 256 * HIDDEN * 3 // 8
PAIR_WORDS = PAIR_BYTES // torch.int16.itemsize
EXPERT_RANK_TRELLIS_BYTES = 3 * PAIR_BYTES
LOCAL_SCALE_BYTES = LOCAL_INTERMEDIATE * torch.float16.itemsize
EXPERT_RANK_SCALE_BYTES = 3 * LOCAL_SCALE_BYTES
MATRIX_WEIGHTS = LOCAL_INTERMEDIATE * HIDDEN
MATRIX_PACKED_BYTES = MATRIX_WEIGHTS // 2
MATRIX_SCALE_BYTES = MATRIX_WEIGHTS // 32
MATRIX_MXFP4_BYTES = MATRIX_PACKED_BYTES + MATRIX_SCALE_BYTES
EXPERT_RANK_MXFP4_BYTES = 3 * MATRIX_MXFP4_BYTES
PAIR_ROTATION_EXPERT_MULTIPLIER = 5
LEGACY_MAGIC = b"KQTP12V3"
LEGACY_HEADER_VERSION = 3
MAGIC = b"KQTP12V4"
HEADER_VERSION = 4
QSRT_MAGIC = b"KQTP12V5"
QSRT_HEADER_VERSION = 5
MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
CODEBOOK_MCG = "mcg"
CODEBOOK_MUL1_E4M3 = "mul1-e4m3"
CODEBOOK_SQG_NORMAL_E4M3 = "sqg-normal-e4m3"
CODEBOOK_SQG_CHEB_NORMAL_E4M3 = "sqg-cheb-normal-e4m3"
CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3 = "sqg-cheb-normal-k2-q8h4-w2-e4m3"
CODEBOOK_IDS = {
    CODEBOOK_MCG: 1,
    CODEBOOK_MUL1_E4M3: 2,
    CODEBOOK_SQG_NORMAL_E4M3: 3,
    CODEBOOK_SQG_CHEB_NORMAL_E4M3: 4,
    CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3: 5,
}
CODEBOOKS_BY_ID = {value: key for key, value in CODEBOOK_IDS.items()}
CODEBOOK_MULTIPLIERS = {
    CODEBOOK_MCG: MCG_MULT,
    CODEBOOK_MUL1_E4M3: MUL1_MULT,
    CODEBOOK_SQG_NORMAL_E4M3: 0,
    CODEBOOK_SQG_CHEB_NORMAL_E4M3: 0,
    CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3: 0,
}
KEEP_STORAGE_INLINE_MXFP4 = "inline-mxfp4"
KEEP_STORAGE_EXTERNAL_X4T = "external-x4t"
KEEP_STORAGE_IDS = {
    KEEP_STORAGE_INLINE_MXFP4: 0,
    KEEP_STORAGE_EXTERNAL_X4T: 1,
}
KEEP_STORAGES_BY_ID = {value: key for key, value in KEEP_STORAGE_IDS.items()}
_LEGACY_HEADER = struct.Struct("<8s8H2I5Q")
_HEADER = struct.Struct("<8s9H2I5Q")
_QSRT_HEADER = struct.Struct("<8s10H2I5Q")


def _align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def _logical_pair_index(layer: int, expert: int, rank: int) -> int:
    rotation = (PAIR_ROTATION_EXPERT_MULTIPLIER * expert + layer) % TP_SIZE
    return (rank - rotation) % TP_SIZE


def _pread_exact(descriptor: int, count: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        piece = os.pread(descriptor, count - len(result), offset + len(result))
        if not piece:
            raise ValueError("mixed-EXL slab ended before the expected offset")
        result.extend(piece)
    return bytes(result)


def _tensor_from_bytes(
    payload: bytes | bytearray,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    count = 1
    for extent in shape:
        count *= extent
    expected_bytes = count * torch.empty((), dtype=dtype).element_size()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"mixed-EXL tensor payload has {len(payload)} bytes, "
            f"expected {expected_bytes} for {dtype} {shape}"
        )
    if count == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).clone()


def _pread_tensor_exact(
    descriptor: int,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    offset: int,
) -> torch.Tensor:
    """Read one contiguous tensor without a bytes/bytearray staging copy.

    Production TP12 rank sections are hundreds of MiB.  Reading them as one
    tensor is both cheaper than constructing one Python object per expert and,
    more importantly, preserves a single sequential I/O stream per TP rank.
    ``preadv`` writes directly into the tensor-owned CPU allocation.
    """

    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    if tensor.numel() == 0:
        return tensor
    target = memoryview(tensor.numpy()).cast("B")
    completed = 0
    while completed < len(target):
        count = os.preadv(descriptor, [target[completed:]], offset + completed)
        if count <= 0:
            raise ValueError("mixed-EXL tensor ended before the expected offset")
        completed += count
    return tensor


@dataclass(frozen=True)
class TP12SlabLayout:
    compressed_experts: int
    kept_experts: int
    keep_storage: str = KEEP_STORAGE_INLINE_MXFP4

    def __post_init__(self) -> None:
        if self.compressed_experts < 0 or self.kept_experts < 0:
            raise ValueError("mixed-EXL tier counts must be non-negative")
        if self.compressed_experts + self.kept_experts != EXPERTS:
            raise ValueError(f"mixed-EXL slab must account for {EXPERTS} experts")
        if self.keep_storage not in KEEP_STORAGE_IDS:
            raise ValueError(
                f"mixed-EXL high-tier storage is unsupported: {self.keep_storage!r}"
            )

    @property
    def rank_trellis_bytes(self) -> int:
        return self.compressed_experts * EXPERT_RANK_TRELLIS_BYTES

    @property
    def rank_scale_payload_bytes(self) -> int:
        return self.compressed_experts * EXPERT_RANK_SCALE_BYTES

    @property
    def rank_scale_bytes(self) -> int:
        return _align_up(self.rank_scale_payload_bytes)

    @property
    def rank_keep_bytes(self) -> int:
        if self.keep_storage == KEEP_STORAGE_EXTERNAL_X4T:
            return 0
        return self.kept_experts * EXPERT_RANK_MXFP4_BYTES

    @property
    def rank_stride(self) -> int:
        return self.rank_trellis_bytes + self.rank_scale_bytes + self.rank_keep_bytes

    @property
    def rank_sections_offset(self) -> int:
        return HEADER_BYTES + FORMAT_BYTES + SHARED_SCALE_BYTES

    @property
    def disk_bytes(self) -> int:
        return self.rank_sections_offset + TP_SIZE * self.rank_stride

    def rank_offset(self, rank: int) -> int:
        if not 0 <= rank < TP_SIZE:
            raise ValueError(f"TP12 rank must be in 0..{TP_SIZE - 1}")
        return self.rank_sections_offset + rank * self.rank_stride


@dataclass(frozen=True)
class TP12SlabHeader:
    layer: int
    layout: TP12SlabLayout
    codebook: str

    @property
    def keep_storage(self) -> str:
        return self.layout.keep_storage


@dataclass(frozen=True)
class TP12RankPayload:
    """CPU rank payload in B12X and stock MXFP4 loader orientations."""

    layer: int
    rank: int
    compressed_expert_ids: torch.Tensor
    kept_expert_ids: torch.Tensor
    w13_trellis: torch.Tensor
    w2_trellis: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    intermediate_rotations: torch.Tensor
    down_svh: torch.Tensor
    fc1_pair_modes: torch.Tensor
    fc2_pair_modes: torch.Tensor
    w13_mxfp4: torch.Tensor
    w13_mxfp4_scale: torch.Tensor
    w2_mxfp4: torch.Tensor
    w2_mxfp4_scale: torch.Tensor
    w13_x4t_scale_components: tuple[X4TScaleComponents, ...] | None
    w2_x4t_scale_components: tuple[X4TScaleComponents, ...] | None

    def b12x_prepare_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "w1_fp4": self.w13_trellis,
            "w2_fp4": self.w2_trellis,
            "gate_suh": self.gate_suh,
            "up_suh": self.up_suh,
            "intermediate_rotations": self.intermediate_rotations,
            "down_svh": self.down_svh,
            "trellis_fc1_pair_modes": self.fc1_pair_modes,
            "trellis_fc2_pair_modes": self.fc2_pair_modes,
        }


def parse_tp12_slab_header(payload: bytes) -> TP12SlabHeader:
    if len(payload) != HEADER_BYTES:
        raise ValueError(f"mixed-EXL header must contain {HEADER_BYTES} bytes")
    magic = payload[:8]
    if magic == QSRT_MAGIC:
        (
            magic,
            version,
            header_bytes,
            tp_size,
            layer,
            experts,
            compressed,
            kept,
            format_bits,
            codebook_id,
            keep_storage_id,
            multiplier,
            alignment,
            format_offset,
            shared_scale_offset,
            rank_sections_offset,
            rank_stride,
            disk_bytes,
        ) = _QSRT_HEADER.unpack_from(payload)
        try:
            codebook = CODEBOOKS_BY_ID[codebook_id]
        except KeyError as exc:
            raise ValueError(
                f"mixed-EXL header codebook ID is unsupported: {codebook_id}"
            ) from exc
        try:
            keep_storage = KEEP_STORAGES_BY_ID[keep_storage_id]
        except KeyError as exc:
            raise ValueError(
                "mixed-EXL header high-tier storage ID is unsupported: "
                f"{keep_storage_id}"
            ) from exc
        if keep_storage != KEEP_STORAGE_EXTERNAL_X4T:
            raise ValueError("mixed-EXL V5 header must use external X4T storage")
        expected_version = QSRT_HEADER_VERSION
        expected_multiplier = CODEBOOK_MULTIPLIERS[codebook]
        padding_begin = _QSRT_HEADER.size
    elif magic == MAGIC:
        (
            magic,
            version,
            header_bytes,
            tp_size,
            layer,
            experts,
            compressed,
            kept,
            format_bits,
            codebook_id,
            multiplier,
            alignment,
            format_offset,
            shared_scale_offset,
            rank_sections_offset,
            rank_stride,
            disk_bytes,
        ) = _HEADER.unpack_from(payload)
        try:
            codebook = CODEBOOKS_BY_ID[codebook_id]
        except KeyError as exc:
            raise ValueError(
                f"mixed-EXL header codebook ID is unsupported: {codebook_id}"
            ) from exc
        expected_version = HEADER_VERSION
        expected_multiplier = CODEBOOK_MULTIPLIERS[codebook]
        keep_storage = KEEP_STORAGE_INLINE_MXFP4
        padding_begin = _HEADER.size
    elif magic == LEGACY_MAGIC:
        (
            magic,
            version,
            header_bytes,
            tp_size,
            layer,
            experts,
            compressed,
            kept,
            format_bits,
            multiplier,
            alignment,
            format_offset,
            shared_scale_offset,
            rank_sections_offset,
            rank_stride,
            disk_bytes,
        ) = _LEGACY_HEADER.unpack_from(payload)
        codebook = CODEBOOK_MCG
        expected_version = LEGACY_HEADER_VERSION
        expected_multiplier = MCG_MULT
        keep_storage = KEEP_STORAGE_INLINE_MXFP4
        padding_begin = _LEGACY_HEADER.size
    else:
        raise ValueError(f"mixed-EXL header magic mismatch: {magic!r} != {MAGIC!r}")
    expected = {
        "version": (version, expected_version),
        "header bytes": (header_bytes, HEADER_BYTES),
        "TP size": (tp_size, TP_SIZE),
        "expert count": (experts, EXPERTS),
        "format bits": (format_bits, 8),
        "codebook multiplier": (multiplier, expected_multiplier),
        "alignment": (alignment, ALIGNMENT),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(
                f"mixed-EXL header {name} mismatch: {actual!r} != {wanted!r}"
            )
    if not 1 <= layer <= 92:
        raise ValueError(f"mixed-EXL slab layer must be in 1..92, got {layer}")
    layout = TP12SlabLayout(compressed, kept, keep_storage=keep_storage)
    offsets = {
        "format offset": (format_offset, HEADER_BYTES),
        "shared-scale offset": (shared_scale_offset, HEADER_BYTES + FORMAT_BYTES),
        "rank-sections offset": (rank_sections_offset, layout.rank_sections_offset),
        "rank stride": (rank_stride, layout.rank_stride),
        "disk bytes": (disk_bytes, layout.disk_bytes),
    }
    for name, (actual, wanted) in offsets.items():
        if actual != wanted:
            raise ValueError(f"mixed-EXL header {name} mismatch: {actual} != {wanted}")
    if any(payload[padding_begin:]):
        raise ValueError("mixed-EXL header contains nonzero reserved bytes")
    return TP12SlabHeader(layer=layer, layout=layout, codebook=codebook)


def _read_format_table(payload: bytes, layout: TP12SlabLayout) -> tuple[int, ...]:
    if len(payload) != FORMAT_BYTES or any(payload[FORMAT_TABLE_BYTES:]):
        raise ValueError("mixed-EXL format section is malformed")
    formats = tuple(payload[:FORMAT_TABLE_BYTES])
    compressed = 0
    kept = 0
    for code in formats:
        if code == FORMAT_MXFP4:
            kept += 1
            continue
        if code >> 4 > 12 or code & 0x0F > 12:
            raise ValueError(f"mixed-EXL format table has invalid code 0x{code:02x}")
        compressed += 1
    if (compressed, kept) != (layout.compressed_experts, layout.kept_experts):
        raise ValueError("mixed-EXL format table disagrees with header tier counts")
    return formats


def _select_tier_experts(
    formats: tuple[int, ...], selected_experts: Sequence[int] | None
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, int], dict[int, int]]:
    compressed_all = tuple(i for i, code in enumerate(formats) if code != FORMAT_MXFP4)
    kept_all = tuple(i for i, code in enumerate(formats) if code == FORMAT_MXFP4)
    compressed_slots = {expert: slot for slot, expert in enumerate(compressed_all)}
    kept_slots = {expert: slot for slot, expert in enumerate(kept_all)}
    if selected_experts is None:
        return compressed_all, kept_all, compressed_slots, kept_slots
    selected = tuple(int(expert) for expert in selected_experts)
    if selected != tuple(sorted(set(selected))) or any(
        expert < 0 or expert >= EXPERTS for expert in selected
    ):
        raise ValueError("selected experts must be sorted unique IDs in 0..895")
    selected_set = set(selected)
    return (
        tuple(expert for expert in compressed_all if expert in selected_set),
        tuple(expert for expert in kept_all if expert in selected_set),
        compressed_slots,
        kept_slots,
    )


def _validate_bits(
    formats: tuple[int, ...], expected_bits: Sequence[int] | None
) -> None:
    if expected_bits is None:
        return
    if len(expected_bits) != EXPERTS:
        raise ValueError(f"hybrid bit map must contain {EXPERTS} entries")
    actual = tuple(4 if code == FORMAT_MXFP4 else 3 for code in formats)
    if tuple(int(value) for value in expected_bits) != actual:
        raise ValueError("hybrid bit map disagrees with mixed-EXL slab format table")


def read_tp12_rank_payload(
    path: str | Path,
    *,
    layer: int,
    rank: int,
    x4t_path: str | Path | None = None,
    x4t_tp12_path: str | Path | None = None,
    expected_bits: Sequence[int] | None = None,
    expected_codebook: str | None = None,
    selected_experts: Sequence[int] | None = None,
) -> TP12RankPayload:
    """Load one complete serving rank, or a selected subset for diagnostics."""

    if x4t_tp12_path is not None and selected_experts is not None:
        raise ValueError(
            "persistent X4T TP12 rank shards require the complete serving tier"
        )
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        header = parse_tp12_slab_header(_pread_exact(descriptor, HEADER_BYTES, 0))
        if header.layer != layer:
            raise ValueError(
                f"mixed-EXL slab has layer {header.layer}, expected {layer}"
            )
        if expected_codebook is not None and header.codebook != expected_codebook:
            raise ValueError(
                "mixed-EXL slab codebook disagrees with the quantization config: "
                f"{header.codebook!r} != {expected_codebook!r}"
            )
        if path.stat().st_size != header.layout.disk_bytes:
            raise ValueError("mixed-EXL slab file size disagrees with its header")
        formats = _read_format_table(
            _pread_exact(descriptor, FORMAT_BYTES, HEADER_BYTES), header.layout
        )
        _validate_bits(formats, expected_bits)
        compressed, kept, compressed_slots, kept_slots = _select_tier_experts(
            formats, selected_experts
        )

        shared_raw = _pread_exact(
            descriptor, SHARED_SCALE_BYTES, HEADER_BYTES + FORMAT_BYTES
        )
        shared_payload_bytes = 3 * HIDDEN * torch.float16.itemsize
        if any(shared_raw[shared_payload_bytes:]):
            raise ValueError("mixed-EXL shared-scale section has nonzero padding")
        shared = _tensor_from_bytes(
            shared_raw[:shared_payload_bytes],
            dtype=torch.float16,
            shape=(3, HIDDEN),
        )
        if not bool(torch.all(torch.isfinite(shared))):
            raise ValueError("mixed-EXL shared scales contain non-finite values")

        layout = header.layout
        rank_offset = layout.rank_offset(rank)
        scale_offset = rank_offset + layout.rank_trellis_bytes
        scale_raw = _pread_exact(descriptor, layout.rank_scale_bytes, scale_offset)
        if any(scale_raw[layout.rank_scale_payload_bytes :]):
            raise ValueError("mixed-EXL rank-scale section has nonzero padding")
        local_all = _tensor_from_bytes(
            scale_raw[: layout.rank_scale_payload_bytes],
            dtype=torch.float16,
            shape=(layout.compressed_experts, 3, LOCAL_INTERMEDIATE),
        )
        if not bool(torch.all(torch.isfinite(local_all))):
            raise ValueError("mixed-EXL local scales contain non-finite values")

        compressed_count = len(compressed)
        complete_compressed_tier = compressed_count == layout.compressed_experts
        if complete_compressed_tier:
            # The normal serving path consumes the complete compressed tier in
            # its canonical slot order.  Preserve one sequential rank-local
            # read instead of issuing one read/allocation/copy per expert.
            raw_trellis = _pread_tensor_exact(
                descriptor,
                dtype=torch.int16,
                shape=(compressed_count, 3, PAIR_WORDS),
                offset=rank_offset,
            )
            w13_trellis = raw_trellis[:, :2].permute(1, 0, 2).contiguous()
            w2_trellis = raw_trellis[:, 2].contiguous()
            local = local_all
        else:
            # Diagnostics can request an arbitrary sparse expert subset.  Keep
            # random access for that uncommon path rather than reading a full
            # rank section merely to inspect a handful of experts.
            w13_trellis = torch.empty(
                (2, compressed_count, PAIR_WORDS), dtype=torch.int16
            )
            w2_trellis = torch.empty((compressed_count, PAIR_WORDS), dtype=torch.int16)
            local = torch.empty(
                (compressed_count, 3, LOCAL_INTERMEDIATE), dtype=torch.float16
            )
            for target_slot, expert in enumerate(compressed):
                source_slot = compressed_slots[expert]
                raw = _tensor_from_bytes(
                    _pread_exact(
                        descriptor,
                        EXPERT_RANK_TRELLIS_BYTES,
                        rank_offset + source_slot * EXPERT_RANK_TRELLIS_BYTES,
                    ),
                    dtype=torch.int16,
                    shape=(3, PAIR_WORDS),
                )
                w13_trellis[:, target_slot].copy_(raw[:2])
                w2_trellis[target_slot].copy_(raw[2])
                local[target_slot].copy_(local_all[source_slot])

        kept_count = len(kept)
        use_x4t_tp12_cache = (
            layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4T
            and x4t_tp12_path is not None
            and selected_experts is None
        )
        direct_x4t_components = (
            layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4T
            and x4t_tp12_path is not None
            and selected_experts is None
        )
        w13_mxfp4 = (
            torch.empty((0, 0, 0), dtype=torch.uint8)
            if use_x4t_tp12_cache
            else torch.empty(
                (kept_count, 2 * LOCAL_INTERMEDIATE, HIDDEN // 2),
                dtype=torch.uint8,
            )
        )
        w13_scale = (
            torch.empty((0, 0, 0), dtype=torch.uint8)
            if direct_x4t_components
            else torch.empty(
                (kept_count, 2 * LOCAL_INTERMEDIATE, HIDDEN // 32),
                dtype=torch.uint8,
            )
        )
        w2_mxfp4 = (
            torch.empty((0, 0, 0), dtype=torch.uint8)
            if use_x4t_tp12_cache
            else torch.empty(
                (kept_count, HIDDEN, LOCAL_INTERMEDIATE // 2), dtype=torch.uint8
            )
        )
        w2_scale = (
            torch.empty((0, 0, 0), dtype=torch.uint8)
            if direct_x4t_components
            else torch.empty(
                (kept_count, HIDDEN, LOCAL_INTERMEDIATE // 32), dtype=torch.uint8
            )
        )
        w13_x4t_components: list[X4TScaleComponents] | None = (
            [] if direct_x4t_components else None
        )
        w2_x4t_components: list[X4TScaleComponents] | None = (
            [] if direct_x4t_components else None
        )
        if layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4T:
            if use_x4t_tp12_cache:
                from vllm.model_executor.layers.quantization.kquant_x4t import (
                    read_x4t_tp12_rank_shard,
                )

                assert x4t_tp12_path is not None
                cache = read_x4t_tp12_rank_shard(
                    x4t_tp12_path,
                    layer=layer,
                    rank=rank,
                    expected_expert_ids=kept,
                )
                w13_mxfp4 = cache.w13_packed
                w2_mxfp4 = cache.w2_packed
                w13_x4t_components = list(cache.w13_scales)
                w2_x4t_components = list(cache.w2_scales)
            else:
                if x4t_path is None:
                    raise ValueError(
                        "mixed-EXL V5 diagnostics require an X4T sidecar when "
                        "no persistent TP12 rank shard is supplied"
                    )
                from vllm.model_executor.layers.quantization.kquant_x4t import (
                    X4TLayerReader,
                )

                x4t = X4TLayerReader(x4t_path)
                if x4t.layer != layer:
                    raise ValueError(
                        f"X4T sidecar has layer {x4t.layer}, expected {layer}"
                    )
                kept_set = set(kept_slots)
                for expert in range(EXPERTS):
                    expected_present = expert in kept_set
                    for matrix in ("w1", "w3", "w2"):
                        if x4t.has(expert, matrix) != expected_present:
                            raise ValueError(
                                "X4T sidecar inventory disagrees with the "
                                "mixed-EXL format table at expert "
                                f"{expert} {matrix}"
                            )
                for target_slot, expert in enumerate(kept):
                    if direct_x4t_components:
                        w1_rank, w3_rank, w2_rank = x4t.read_rank_triplet(expert, rank)
                        w1_packed = w1_rank.packed
                        w3_packed = w3_rank.packed
                        w2_packed = w2_rank.packed
                    else:
                        w1_matrix = x4t.read_rank(expert, "w1", rank)
                        w3_matrix = x4t.read_rank(expert, "w3", rank)
                        w2_matrix = x4t.read_rank(expert, "w2", rank)
                        w1_packed = w1_matrix.packed
                        w3_packed = w3_matrix.packed
                        w2_packed = w2_matrix.packed
                    w13_mxfp4[target_slot, :LOCAL_INTERMEDIATE].copy_(w1_packed)
                    w13_mxfp4[target_slot, LOCAL_INTERMEDIATE:].copy_(w3_packed)
                    w2_mxfp4[target_slot].copy_(w2_packed)
                    if direct_x4t_components:
                        assert w13_x4t_components is not None
                        assert w2_x4t_components is not None
                        w13_x4t_components.append(
                            w1_rank.scale.concatenate_rows(w3_rank.scale)
                        )
                        w2_x4t_components.append(w2_rank.scale)
                    else:
                        w13_scale[target_slot, :LOCAL_INTERMEDIATE].copy_(
                            w1_matrix.scale
                        )
                        w13_scale[target_slot, LOCAL_INTERMEDIATE:].copy_(
                            w3_matrix.scale
                        )
                        w2_scale[target_slot].copy_(w2_matrix.scale)
        else:
            if x4t_path is not None or x4t_tp12_path is not None:
                raise ValueError(
                    "inline mixed-EXL slab must not name an X4T sidecar/cache"
                )
            keep_offset = scale_offset + layout.rank_scale_bytes
            for target_slot, expert in enumerate(kept):
                source_slot = kept_slots[expert]
                raw = _pread_exact(
                    descriptor,
                    EXPERT_RANK_MXFP4_BYTES,
                    keep_offset + source_slot * EXPERT_RANK_MXFP4_BYTES,
                )
                cursor = 0
                for projection in range(2):
                    packed = _tensor_from_bytes(
                        raw[cursor : cursor + MATRIX_PACKED_BYTES],
                        dtype=torch.uint8,
                        shape=(LOCAL_INTERMEDIATE, HIDDEN // 2),
                    )
                    cursor += MATRIX_PACKED_BYTES
                    scale = _tensor_from_bytes(
                        raw[cursor : cursor + MATRIX_SCALE_BYTES],
                        dtype=torch.uint8,
                        shape=(LOCAL_INTERMEDIATE, HIDDEN // 32),
                    )
                    cursor += MATRIX_SCALE_BYTES
                    begin = projection * LOCAL_INTERMEDIATE
                    end = begin + LOCAL_INTERMEDIATE
                    w13_mxfp4[target_slot, begin:end].copy_(packed)
                    w13_scale[target_slot, begin:end].copy_(scale)
                w2_mxfp4[target_slot].copy_(
                    _tensor_from_bytes(
                        raw[cursor : cursor + MATRIX_PACKED_BYTES],
                        dtype=torch.uint8,
                        shape=(HIDDEN, LOCAL_INTERMEDIATE // 2),
                    )
                )
                cursor += MATRIX_PACKED_BYTES
                w2_scale[target_slot].copy_(
                    _tensor_from_bytes(
                        raw[cursor : cursor + MATRIX_SCALE_BYTES],
                        dtype=torch.uint8,
                        shape=(HIDDEN, LOCAL_INTERMEDIATE // 32),
                    )
                )
                cursor += MATRIX_SCALE_BYTES
                if cursor != EXPERT_RANK_MXFP4_BYTES:
                    raise AssertionError("mixed-EXL MXFP4 byte accounting drifted")
    finally:
        os.close(descriptor)

    r13 = torch.tensor(
        [formats[expert] >> 4 for expert in compressed], dtype=torch.int32
    )
    r2 = torch.tensor(
        [formats[expert] & 0x0F for expert in compressed], dtype=torch.int32
    )
    logical_pairs = torch.tensor(
        [_logical_pair_index(layer, expert, rank) for expert in compressed],
        dtype=torch.int32,
    )
    return TP12RankPayload(
        layer=layer,
        rank=rank,
        compressed_expert_ids=torch.tensor(compressed, dtype=torch.int32),
        kept_expert_ids=torch.tensor(kept, dtype=torch.int32),
        w13_trellis=w13_trellis,
        w2_trellis=w2_trellis,
        gate_suh=shared[0].reshape(1, -1).contiguous(),
        up_suh=shared[1].reshape(1, -1).contiguous(),
        intermediate_rotations=local.reshape(
            len(compressed), 3 * LOCAL_INTERMEDIATE
        ).contiguous(),
        down_svh=shared[2].reshape(1, -1).contiguous(),
        fc1_pair_modes=(r13 > logical_pairs).to(torch.int32).contiguous(),
        fc2_pair_modes=(r2 > logical_pairs).to(torch.int32).contiguous(),
        w13_mxfp4=w13_mxfp4,
        w13_mxfp4_scale=w13_scale,
        w2_mxfp4=w2_mxfp4,
        w2_mxfp4_scale=w2_scale,
        w13_x4t_scale_components=(
            tuple(w13_x4t_components) if w13_x4t_components is not None else None
        ),
        w2_x4t_scale_components=(
            tuple(w2_x4t_components) if w2_x4t_components is not None else None
        ),
    )


__all__ = [
    "ALIGNMENT",
    "CODEBOOK_IDS",
    "CODEBOOK_MCG",
    "CODEBOOK_MUL1_E4M3",
    "CODEBOOK_SQG_NORMAL_E4M3",
    "CODEBOOK_SQG_CHEB_NORMAL_E4M3",
    "CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3",
    "CODEBOOK_MULTIPLIERS",
    "EXPERTS",
    "EXPERT_RANK_TRELLIS_BYTES",
    "FORMAT_BYTES",
    "FORMAT_MXFP4",
    "FORMAT_TABLE_BYTES",
    "HEADER_BYTES",
    "HIDDEN",
    "LOCAL_INTERMEDIATE",
    "KEEP_STORAGE_EXTERNAL_X4T",
    "KEEP_STORAGE_INLINE_MXFP4",
    "LEGACY_HEADER_VERSION",
    "LEGACY_MAGIC",
    "MAGIC",
    "MCG_MULT",
    "MUL1_MULT",
    "PAIR_WORDS",
    "SHARED_SCALE_BYTES",
    "QSRT_HEADER_VERSION",
    "QSRT_MAGIC",
    "TP_SIZE",
    "TP12RankPayload",
    "TP12SlabHeader",
    "TP12SlabLayout",
    "parse_tp12_slab_header",
    "read_tp12_rank_payload",
]
