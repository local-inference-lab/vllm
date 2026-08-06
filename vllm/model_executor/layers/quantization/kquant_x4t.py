# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict reader for kquant's tile-oriented X4T MXFP4 sidecars."""

from __future__ import annotations

import math
import mmap
import os
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

X4T_MAGIC = b"KQX4T\0\0\0"
X4T_LAYER_MAGIC = b"KQX4TLY\0"
X4T_RECORD_MAGIC = b"KQ4T"
X4T_VERSION = 1
X4T_TILE_ROWS = 16
X4T_POSITION_BITS = 24
X4T_POSITION_MASK = (1 << X4T_POSITION_BITS) - 1
X4T_LAYER_HEADER_BYTES = 4096
X4T_DIRECTORY_BYTES = 65536
X4T_DIRECTORY_ENTRY_BYTES = 24
X4T_DATA_OFFSET = X4T_LAYER_HEADER_BYTES + X4T_DIRECTORY_BYTES
X4T_RECORD_ALIGNMENT = 4096
X4T_EXPERTS_PER_LAYER = 896
X4T_MATRIX_ORDER = ("w1", "w3", "w2")
MXFP4_BLOCK = 32
X4T_TP12_CACHE_FORMAT = "kquant-x4t-tp12"
X4T_TP12_CACHE_VERSION = 1
X4T_TP12_CACHE_KEYS = frozenset(
    {
        "expert_ids",
        "w13_packed",
        "w2_packed",
        "w13_fixed",
        "w13_exception_offsets",
        "w13_exceptions",
        "w2_fixed",
        "w2_exception_offsets",
        "w2_exceptions",
    }
)

_SCALE_HEADER = struct.Struct("<8sBBH I H H I I Q Q 20s")
_LAYER_HEADER = struct.Struct("<8sHHIHHHHQQQQIIII")
_DIRECTORY_ENTRY = struct.Struct("<QQII")
_RECORD_HEADER = struct.Struct("<4sHHBBHIIQQ28s")

_MATRIX_SHAPES = {
    "w1": (3072, 3584),
    "w3": (3072, 3584),
    "w2": (3584, 3072),
}


def _align_up(value: int) -> int:
    return (value + X4T_RECORD_ALIGNMENT - 1) & -X4T_RECORD_ALIGNMENT


def _matrix_id(matrix: str) -> int:
    try:
        return X4T_MATRIX_ORDER.index(matrix)
    except ValueError as exc:
        raise ValueError(f"unsupported X4T matrix {matrix!r}") from exc


def _entry_index(expert: int, matrix: str) -> int:
    if isinstance(expert, bool) or not isinstance(expert, int):
        raise TypeError("X4T expert ID must be an integer")
    if not 0 <= expert < X4T_EXPERTS_PER_LAYER:
        raise ValueError("X4T expert ID must lie in 0..895")
    return expert * len(X4T_MATRIX_ORDER) + _matrix_id(matrix)


def _pread_exact(descriptor: int, count: int, offset: int) -> bytes:
    payload = bytearray()
    while len(payload) < count:
        piece = os.pread(descriptor, count - len(payload), offset + len(payload))
        if not piece:
            raise ValueError("X4T sidecar ended before the expected offset")
        payload.extend(piece)
    return bytes(payload)


def _pread_tensor_exact(
    descriptor: int,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    offset: int,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    if tensor.numel() == 0:
        return tensor
    target = memoryview(tensor.numpy()).cast("B")
    completed = 0
    while completed < len(target):
        count = os.preadv(descriptor, [target[completed:]], offset + completed)
        if count <= 0:
            raise ValueError("X4T sidecar ended before the expected offset")
        completed += count
    return tensor


def _validate_scale(scale: torch.Tensor) -> tuple[int, int]:
    if (
        scale.dtype != torch.uint8
        or scale.ndim != 2
        or scale.device.type != "cpu"
        or not scale.is_contiguous()
    ):
        raise ValueError("X4T scale must be contiguous two-dimensional CPU uint8")
    rows, columns = map(int, scale.shape)
    if rows <= 0 or rows % X4T_TILE_ROWS:
        raise ValueError("X4T scale rows must be a positive multiple of 16")
    if not 1 <= columns <= 255:
        raise ValueError("X4T scale columns must lie in 1..255")
    if rows * columns > X4T_POSITION_MASK:
        raise ValueError("X4T scale exceeds its 24-bit position field")
    return rows, columns


def pack_x4t_scale_components(
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the canonical fixed stream and exception words for one plane."""

    rows, columns = _validate_scale(scale)
    selector_bytes = math.ceil(columns / 8)
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)
    source = scale.numpy()
    indexed = (np.arange(rows, dtype=np.int64)[:, None] << 8) + source
    histogram = np.bincount(indexed.ravel(), minlength=rows * 256).reshape(rows, 256)
    bases = np.argmax(histogram[:, :-1] + histogram[:, 1:], axis=1).astype(np.uint8)
    source_i16 = source.astype(np.int16)
    base_i16 = bases.astype(np.int16)
    low = source_i16 == base_i16[:, None]
    high = source_i16 == (base_i16[:, None] + 1)
    selectors = np.packbits(high, axis=1, bitorder="little")
    fixed = np.empty((tile_count, tile_bytes), dtype=np.uint8)
    fixed[:, :X4T_TILE_ROWS] = bases.reshape(tile_count, X4T_TILE_ROWS)
    fixed[:, X4T_TILE_ROWS:] = selectors.reshape(tile_count, -1)

    coordinates = np.argwhere(~(low | high))
    if coordinates.size:
        positions = coordinates[:, 0].astype(np.uint32) * np.uint32(
            columns
        ) + coordinates[:, 1].astype(np.uint32)
        values = source[coordinates[:, 0], coordinates[:, 1]].astype(np.uint32)
        exceptions = positions | (values << np.uint32(X4T_POSITION_BITS))
    else:
        exceptions = np.empty((0,), dtype=np.uint32)
    return (
        torch.from_numpy(fixed.reshape(-1).copy()),
        torch.from_numpy(exceptions.copy()),
    )


def pack_x4t_scale_plane(scale: torch.Tensor) -> bytes:
    rows, columns = _validate_scale(scale)
    fixed, exceptions = pack_x4t_scale_components(scale)
    selector_bytes = math.ceil(columns / 8)
    tile_count = rows // X4T_TILE_ROWS
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)
    header = _SCALE_HEADER.pack(
        X4T_MAGIC,
        X4T_VERSION,
        X4T_TILE_ROWS,
        0,
        rows,
        columns,
        selector_bytes,
        tile_bytes,
        tile_count,
        fixed.numel(),
        exceptions.numel(),
        bytes(20),
    )
    return (
        header
        + fixed.numpy().tobytes()
        + exceptions.numpy().astype("<u4", copy=False).tobytes()
    )


def unpack_x4t_scale_plane(payload: bytes) -> torch.Tensor:
    if len(payload) < _SCALE_HEADER.size:
        raise ValueError("X4T scale is truncated before its header")
    (
        magic,
        version,
        tile_rows,
        flags,
        rows,
        columns,
        selector_bytes,
        tile_bytes,
        tile_count,
        fixed_bytes,
        exception_count,
        reserved,
    ) = _SCALE_HEADER.unpack_from(payload)
    expected_selector = math.ceil(columns / 8) if columns else 0
    expected_tiles = rows // X4T_TILE_ROWS if rows else 0
    expected_tile_bytes = X4T_TILE_ROWS * (1 + expected_selector)
    if magic != X4T_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T scale has unsupported magic or version")
    if tile_rows != X4T_TILE_ROWS or flags or any(reserved):
        raise ValueError("X4T scale header is noncanonical")
    if (
        rows <= 0
        or rows % X4T_TILE_ROWS
        or not 1 <= columns <= 255
        or rows * columns > X4T_POSITION_MASK
        or selector_bytes != expected_selector
        or tile_count != expected_tiles
        or tile_bytes != expected_tile_bytes
        or fixed_bytes != tile_count * tile_bytes
        or len(payload) != _SCALE_HEADER.size + fixed_bytes + 4 * exception_count
    ):
        raise ValueError("X4T scale geometry is noncanonical")

    fixed_start = _SCALE_HEADER.size
    fixed = torch.frombuffer(
        bytearray(payload[fixed_start : fixed_start + fixed_bytes]),
        dtype=torch.uint8,
    ).reshape(tile_count, tile_bytes)
    bases = fixed[:, :X4T_TILE_ROWS].reshape(rows)
    selectors = fixed[:, X4T_TILE_ROWS:].reshape(rows, selector_bytes)
    column = torch.arange(columns, dtype=torch.int64)
    selected = (
        selectors[:, column // 8].to(torch.int16) >> (column % 8).to(torch.int16)
    ) & 1
    result = (bases.to(torch.int16)[:, None] + selected).to(torch.uint8)
    if columns % 8 and bool((selectors[:, -1] >> (columns % 8)).any()):
        raise ValueError("X4T selector padding is nonzero")

    exception_start = fixed_start + fixed_bytes
    entries = (
        struct.unpack_from(f"<{exception_count}I", payload, exception_start)
        if exception_count
        else ()
    )
    previous = -1
    flat = result.view(-1)
    for entry in entries:
        position = entry & X4T_POSITION_MASK
        value = entry >> X4T_POSITION_BITS
        if position >= flat.numel() or position <= previous:
            raise ValueError("X4T exception positions are invalid")
        previous = position
        row = position // columns
        base = int(bases[row])
        if value in (base, base + 1):
            raise ValueError("X4T exception redundantly names a palette value")
        flat[position] = value
    if pack_x4t_scale_plane(result.contiguous()) != payload:
        raise ValueError("X4T scale payload is not canonical")
    return result


@dataclass(frozen=True)
class X4TMatrix:
    matrix: str
    packed: torch.Tensor
    scale: torch.Tensor

    def rank_shard(self, rank: int) -> X4TMatrix:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("X4T TP12 rank must be an integer")
        if not 0 <= rank < 12:
            raise ValueError("X4T TP12 rank must lie in 0..11")
        if self.matrix in ("w1", "w3"):
            begin = rank * 256
            end = begin + 256
            return X4TMatrix(
                self.matrix,
                self.packed[begin:end].contiguous(),
                self.scale[begin:end].contiguous(),
            )
        return X4TMatrix(
            self.matrix,
            self.packed[:, rank * 128 : (rank + 1) * 128].contiguous(),
            self.scale[:, rank * 8 : (rank + 1) * 8].contiguous(),
        )


@dataclass(frozen=True)
class X4TScaleComponents:
    """Rank-local X4T scale streams, without a dense scale materialization."""

    fixed: torch.Tensor
    exceptions: torch.Tensor
    rows: int
    columns: int

    def concatenate_rows(self, other: X4TScaleComponents) -> X4TScaleComponents:
        if self.columns != other.columns:
            raise ValueError("X4T row concatenation requires equal column counts")
        row_offset = self.rows
        if other.exceptions.numel():
            words = other.exceptions.to(torch.int64)
            positions = words & X4T_POSITION_MASK
            values = words >> X4T_POSITION_BITS
            shifted = (
                (values << X4T_POSITION_BITS) | (positions + row_offset * self.columns)
            ).to(torch.uint32)
        else:
            shifted = torch.empty((0,), dtype=torch.uint32)
        return X4TScaleComponents(
            fixed=torch.cat((self.fixed, other.fixed)).contiguous(),
            exceptions=torch.cat((self.exceptions, shifted)).contiguous(),
            rows=self.rows + other.rows,
            columns=self.columns,
        )


@dataclass(frozen=True)
class X4TRankComponents:
    """Packed rank-local nibbles and their directly sliced X4T scale stream."""

    matrix: str
    packed: torch.Tensor
    scale: X4TScaleComponents


@dataclass(frozen=True)
class X4TTP12RankShard:
    expert_ids: torch.Tensor
    w13_packed: torch.Tensor
    w2_packed: torch.Tensor
    w13_scales: tuple[X4TScaleComponents, ...]
    w2_scales: tuple[X4TScaleComponents, ...]


def x4t_tp12_rank_filename(layer: int, rank: int) -> str:
    if not 1 <= layer <= 92:
        raise ValueError("X4T TP12 cache layer must lie in 1..92")
    if not 0 <= rank < 12:
        raise ValueError("X4T TP12 cache rank must lie in 0..11")
    return f"x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"


def _validate_cache_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        tensor.dtype != dtype
        or tensor.device.type != "cpu"
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"X4T TP12 cache tensor {name} must be contiguous CPU {dtype} "
            f"{shape}, got {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
        )


def _cache_scale_components(
    *,
    name: str,
    fixed: torch.Tensor,
    offsets: torch.Tensor,
    exceptions: torch.Tensor,
    experts: int,
    rows: int,
    columns: int,
) -> tuple[X4TScaleComponents, ...]:
    selector_bytes = math.ceil(columns / 8)
    fixed_bytes = (rows // X4T_TILE_ROWS) * X4T_TILE_ROWS * (1 + selector_bytes)
    _validate_cache_tensor(
        f"{name}_fixed",
        fixed,
        dtype=torch.uint8,
        shape=(experts, fixed_bytes),
    )
    _validate_cache_tensor(
        f"{name}_exception_offsets",
        offsets,
        dtype=torch.int64,
        shape=(experts + 1,),
    )
    if (
        exceptions.dtype != torch.uint32
        or exceptions.device.type != "cpu"
        or exceptions.ndim != 1
        or not exceptions.is_contiguous()
    ):
        raise ValueError(
            f"X4T TP12 cache tensor {name}_exceptions must be contiguous "
            "one-dimensional CPU uint32"
        )
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != exceptions.numel()
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError(f"X4T TP12 cache {name} exception offsets are invalid")
    return tuple(
        X4TScaleComponents(
            fixed=fixed[index],
            exceptions=exceptions[int(offsets[index]) : int(offsets[index + 1])],
            rows=rows,
            columns=columns,
        )
        for index in range(experts)
    )


def read_x4t_tp12_rank_shard(
    path: str | Path,
    *,
    layer: int,
    rank: int,
    expected_expert_ids: Sequence[int],
) -> X4TTP12RankShard:
    """Map one prevalidated, rank-major X4T checkpoint shard."""

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected_metadata = {
            "format": X4T_TP12_CACHE_FORMAT,
            "version": str(X4T_TP12_CACHE_VERSION),
            "layer": str(layer),
            "rank": str(rank),
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"X4T TP12 cache metadata {name} mismatch: "
                    f"{metadata.get(name)!r} != {expected!r}"
                )
        if set(handle) != X4T_TP12_CACHE_KEYS:
            raise ValueError("X4T TP12 cache tensor inventory is noncanonical")
        tensors = {name: handle.get_tensor(name) for name in handle}

    expected_ids = tuple(map(int, expected_expert_ids))
    experts = len(expected_ids)
    expert_ids = tensors["expert_ids"]
    _validate_cache_tensor(
        "expert_ids",
        expert_ids,
        dtype=torch.int32,
        shape=(experts,),
    )
    if tuple(expert_ids.tolist()) != expected_ids:
        raise ValueError("X4T TP12 cache expert order disagrees with the slab")
    _validate_cache_tensor(
        "w13_packed",
        tensors["w13_packed"],
        dtype=torch.uint8,
        shape=(experts, 512, 1792),
    )
    _validate_cache_tensor(
        "w2_packed",
        tensors["w2_packed"],
        dtype=torch.uint8,
        shape=(experts, 3584, 128),
    )
    w13_scales = _cache_scale_components(
        name="w13",
        fixed=tensors["w13_fixed"],
        offsets=tensors["w13_exception_offsets"],
        exceptions=tensors["w13_exceptions"],
        experts=experts,
        rows=512,
        columns=112,
    )
    w2_scales = _cache_scale_components(
        name="w2",
        fixed=tensors["w2_fixed"],
        offsets=tensors["w2_exception_offsets"],
        exceptions=tensors["w2_exceptions"],
        experts=experts,
        rows=3584,
        columns=8,
    )
    return X4TTP12RankShard(
        expert_ids=expert_ids,
        w13_packed=tensors["w13_packed"],
        w2_packed=tensors["w2_packed"],
        w13_scales=w13_scales,
        w2_scales=w2_scales,
    )


@dataclass(frozen=True)
class X4TDirectoryEntry:
    offset: int = 0
    length: int = 0
    crc32: int = 0
    flags: int = 0

    @property
    def present(self) -> bool:
        return bool(self.flags & 1)


def _unpack_matrix_record(payload: bytes, expected_matrix: str) -> X4TMatrix:
    if len(payload) < _RECORD_HEADER.size:
        raise ValueError("X4T record is truncated before its header")
    (
        magic,
        version,
        header_bytes,
        matrix_id,
        tile_rows,
        flags,
        out_features,
        in_features,
        packed_bytes,
        scale_bytes,
        reserved,
    ) = _RECORD_HEADER.unpack_from(payload)
    if magic != X4T_RECORD_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T record has unsupported magic or version")
    if header_bytes != _RECORD_HEADER.size or flags or any(reserved):
        raise ValueError("X4T record header is noncanonical")
    if matrix_id >= len(X4T_MATRIX_ORDER):
        raise ValueError("X4T record matrix ID is invalid")
    matrix = X4T_MATRIX_ORDER[matrix_id]
    if matrix != expected_matrix or tile_rows != X4T_TILE_ROWS:
        raise ValueError("X4T record disagrees with its directory slot")
    if (out_features, in_features) != _MATRIX_SHAPES[matrix]:
        raise ValueError("X4T record has the wrong production matrix shape")
    if packed_bytes != out_features * in_features // 2:
        raise ValueError("X4T record packed byte count is invalid")
    if len(payload) != header_bytes + packed_bytes + scale_bytes:
        raise ValueError("X4T record component lengths do not close")
    packed_end = header_bytes + packed_bytes
    packed = torch.frombuffer(
        bytearray(payload[header_bytes:packed_end]), dtype=torch.uint8
    ).reshape(out_features, in_features // 2)
    scale = unpack_x4t_scale_plane(payload[packed_end:])
    if tuple(scale.shape) != (out_features, in_features // MXFP4_BLOCK):
        raise ValueError("X4T scale geometry disagrees with its matrix")
    return X4TMatrix(matrix, packed, scale)


def _parse_record_header(
    payload: bytes,
    *,
    expected_matrix: str,
    expected_length: int,
) -> tuple[int, int, int, int]:
    if len(payload) != _RECORD_HEADER.size:
        raise ValueError("X4T record header has the wrong byte length")
    (
        magic,
        version,
        header_bytes,
        matrix_id,
        tile_rows,
        flags,
        out_features,
        in_features,
        packed_bytes,
        scale_bytes,
        reserved,
    ) = _RECORD_HEADER.unpack(payload)
    if magic != X4T_RECORD_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T record has unsupported magic or version")
    if header_bytes != _RECORD_HEADER.size or flags or any(reserved):
        raise ValueError("X4T record header is noncanonical")
    if matrix_id >= len(X4T_MATRIX_ORDER):
        raise ValueError("X4T record matrix ID is invalid")
    matrix = X4T_MATRIX_ORDER[matrix_id]
    if matrix != expected_matrix or tile_rows != X4T_TILE_ROWS:
        raise ValueError("X4T record disagrees with its directory slot")
    if (out_features, in_features) != _MATRIX_SHAPES[matrix]:
        raise ValueError("X4T record has the wrong production matrix shape")
    if packed_bytes != out_features * in_features // 2:
        raise ValueError("X4T record packed byte count is invalid")
    if expected_length != header_bytes + packed_bytes + scale_bytes:
        raise ValueError("X4T record component lengths do not close")
    return out_features, in_features, packed_bytes, scale_bytes


def _parse_scale_header(
    payload: bytes,
    *,
    expected_rows: int,
    expected_columns: int,
    expected_bytes: int,
) -> tuple[int, int, int, int]:
    if len(payload) != _SCALE_HEADER.size:
        raise ValueError("X4T scale header has the wrong byte length")
    (
        magic,
        version,
        tile_rows,
        flags,
        rows,
        columns,
        selector_bytes,
        tile_bytes,
        tile_count,
        fixed_bytes,
        exception_count,
        reserved,
    ) = _SCALE_HEADER.unpack(payload)
    expected_selector = math.ceil(columns / 8) if columns else 0
    expected_tiles = rows // X4T_TILE_ROWS if rows else 0
    expected_tile_bytes = X4T_TILE_ROWS * (1 + expected_selector)
    if magic != X4T_MAGIC or version != X4T_VERSION:
        raise ValueError("X4T scale has unsupported magic or version")
    if tile_rows != X4T_TILE_ROWS or flags or any(reserved):
        raise ValueError("X4T scale header is noncanonical")
    if (
        rows != expected_rows
        or columns != expected_columns
        or rows <= 0
        or rows % X4T_TILE_ROWS
        or not 1 <= columns <= 255
        or rows * columns > X4T_POSITION_MASK
        or selector_bytes != expected_selector
        or tile_count != expected_tiles
        or tile_bytes != expected_tile_bytes
        or fixed_bytes != tile_count * tile_bytes
        or expected_bytes != _SCALE_HEADER.size + fixed_bytes + 4 * exception_count
    ):
        raise ValueError("X4T scale geometry is noncanonical")
    return selector_bytes, tile_bytes, fixed_bytes, exception_count


def _validate_exception_positions(
    exceptions: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    words = exceptions.numpy().astype(np.uint32, copy=False)
    positions = words & np.uint32(X4T_POSITION_MASK)
    if positions.size and (
        int(positions[-1]) >= rows * columns
        or bool(np.any(positions[1:] <= positions[:-1]))
    ):
        raise ValueError("X4T exception positions are invalid")
    return positions


def _canonical_layer_header(
    *,
    layer: int,
    file_bytes: int,
    record_count: int,
    directory_crc32: int,
    header_crc32: int,
) -> bytes:
    prefix = _LAYER_HEADER.pack(
        X4T_LAYER_MAGIC,
        X4T_VERSION,
        X4T_LAYER_HEADER_BYTES,
        layer,
        X4T_EXPERTS_PER_LAYER,
        len(X4T_MATRIX_ORDER),
        X4T_DIRECTORY_ENTRY_BYTES,
        0,
        X4T_LAYER_HEADER_BYTES,
        X4T_DIRECTORY_BYTES,
        X4T_DATA_OFFSET,
        file_bytes,
        record_count,
        directory_crc32,
        header_crc32,
        0,
    )
    return prefix + bytes(X4T_LAYER_HEADER_BYTES - len(prefix))


class X4TLayerReader:
    """Validated random-access reader for one X4T MoE layer sidecar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header = handle.read(X4T_LAYER_HEADER_BYTES)
            directory = handle.read(X4T_DIRECTORY_BYTES)
        if (
            len(header) != X4T_LAYER_HEADER_BYTES
            or len(directory) != X4T_DIRECTORY_BYTES
        ):
            raise ValueError("X4T layer is truncated before its data section")
        (
            magic,
            version,
            header_bytes,
            layer,
            experts,
            matrices,
            entry_bytes,
            flags,
            directory_offset,
            directory_bytes,
            data_offset,
            file_bytes,
            record_count,
            directory_crc32,
            header_crc32,
            reserved,
        ) = _LAYER_HEADER.unpack_from(header)
        if magic != X4T_LAYER_MAGIC or version != X4T_VERSION:
            raise ValueError("X4T layer has unsupported magic or version")
        if (
            header_bytes != X4T_LAYER_HEADER_BYTES
            or experts != X4T_EXPERTS_PER_LAYER
            or matrices != len(X4T_MATRIX_ORDER)
            or entry_bytes != X4T_DIRECTORY_ENTRY_BYTES
            or flags
            or directory_offset != X4T_LAYER_HEADER_BYTES
            or directory_bytes != X4T_DIRECTORY_BYTES
            or data_offset != X4T_DATA_OFFSET
            or reserved
            or not 1 <= layer <= 92
            or file_bytes != self.path.stat().st_size
        ):
            raise ValueError("X4T layer header is noncanonical")
        zero_header = _canonical_layer_header(
            layer=layer,
            file_bytes=file_bytes,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=0,
        )
        canonical = _canonical_layer_header(
            layer=layer,
            file_bytes=file_bytes,
            record_count=record_count,
            directory_crc32=directory_crc32,
            header_crc32=header_crc32,
        )
        if header != canonical or zlib.crc32(zero_header) != header_crc32:
            raise ValueError("X4T layer header checksum is invalid")
        if zlib.crc32(directory) != directory_crc32:
            raise ValueError("X4T layer directory checksum is invalid")
        entry_count = X4T_EXPERTS_PER_LAYER * len(X4T_MATRIX_ORDER)
        used = entry_count * X4T_DIRECTORY_ENTRY_BYTES
        if any(directory[used:]):
            raise ValueError("X4T layer directory padding is nonzero")
        self.entries = tuple(
            X4TDirectoryEntry(
                *_DIRECTORY_ENTRY.unpack_from(
                    directory, index * X4T_DIRECTORY_ENTRY_BYTES
                )
            )
            for index in range(entry_count)
        )
        if sum(entry.present for entry in self.entries) != record_count:
            raise ValueError("X4T layer record count disagrees with its directory")
        cursor = X4T_DATA_OFFSET
        for entry in self.entries:
            if not entry.present:
                if entry != X4TDirectoryEntry():
                    raise ValueError("absent X4T directory entry is noncanonical")
                continue
            if entry.flags != 1 or entry.offset != cursor or entry.length < 64:
                raise ValueError("present X4T directory entry is noncanonical")
            cursor = _align_up(entry.offset + entry.length)
            if cursor > file_bytes:
                raise ValueError("X4T directory entry exceeds the layer file")
        if cursor != file_bytes:
            raise ValueError("X4T layer has unreferenced trailing bytes")
        self.layer = layer
        self.file_bytes = file_bytes
        self.record_count = record_count
        self._descriptor: int | None = None
        self._mapping: mmap.mmap | None = None

    def __enter__(self) -> X4TLayerReader:
        if self._descriptor is not None or self._mapping is not None:
            raise RuntimeError("X4T layer reader is already open")
        self._descriptor = os.open(self.path, os.O_RDONLY)
        self._mapping = mmap.mmap(
            self._descriptor,
            length=0,
            access=mmap.ACCESS_READ,
        )
        if hasattr(self._mapping, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
            self._mapping.madvise(mmap.MADV_SEQUENTIAL)
        return self

    def __exit__(self, *_args: object) -> None:
        assert self._mapping is not None
        assert self._descriptor is not None
        self._mapping.close()
        os.close(self._descriptor)
        self._mapping = None
        self._descriptor = None

    def has(self, expert: int, matrix: str) -> bool:
        return self.entries[_entry_index(expert, matrix)].present

    def read(self, expert: int, matrix: str) -> X4TMatrix:
        entry = self.entries[_entry_index(expert, matrix)]
        if not entry.present:
            raise KeyError((expert, matrix))
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            payload = os.pread(descriptor, entry.length, entry.offset)
            padding = os.pread(
                descriptor,
                _align_up(entry.offset + entry.length) - entry.offset - entry.length,
                entry.offset + entry.length,
            )
        finally:
            os.close(descriptor)
        if len(payload) != entry.length or zlib.crc32(payload) != entry.crc32:
            raise ValueError("X4T matrix record checksum is invalid")
        if any(padding):
            raise ValueError("X4T matrix record padding is nonzero")
        return _unpack_matrix_record(payload, matrix)

    def read_rank(self, expert: int, matrix: str, rank: int) -> X4TMatrix:
        return self.read(expert, matrix).rank_shard(rank)

    def _read_rank_components(
        self,
        descriptor: int,
        *,
        expert: int,
        matrix: str,
        rank: int,
        mapping: mmap.mmap | None = None,
    ) -> X4TRankComponents:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("X4T TP12 rank must be an integer")
        if not 0 <= rank < 12:
            raise ValueError("X4T TP12 rank must lie in 0..11")
        entry = self.entries[_entry_index(expert, matrix)]
        if not entry.present:
            raise KeyError((expert, matrix))

        record_header = _pread_exact(descriptor, _RECORD_HEADER.size, entry.offset)
        out_features, in_features, packed_bytes, scale_bytes = _parse_record_header(
            record_header,
            expected_matrix=matrix,
            expected_length=entry.length,
        )
        packed_offset = entry.offset + _RECORD_HEADER.size
        packed_columns = in_features // 2
        if matrix in ("w1", "w3"):
            rank_rows = out_features // 12
            packed = _pread_tensor_exact(
                descriptor,
                dtype=torch.uint8,
                shape=(rank_rows, packed_columns),
                offset=packed_offset + rank * rank_rows * packed_columns,
            )
        else:
            # FC2 is column-sharded at TP12.  The V1 X4T sidecar stores the
            # nibble plane in ordinary row-major matrix order, so read it once
            # and perform the strided rank slice in compiled tensor code.  The
            # much smaller scale plane is sliced directly below without ever
            # being expanded to dense form.
            rank_columns = packed_columns // 12
            begin = rank * rank_columns
            if mapping is None:
                packed_full = _pread_tensor_exact(
                    descriptor,
                    dtype=torch.uint8,
                    shape=(out_features, packed_columns),
                    offset=packed_offset,
                )
                packed = packed_full[:, begin : begin + rank_columns].contiguous()
            else:
                packed_view = np.ndarray(
                    (out_features, packed_columns),
                    dtype=np.uint8,
                    buffer=mapping,
                    offset=packed_offset,
                )
                packed = torch.from_numpy(
                    packed_view[:, begin : begin + rank_columns].copy()
                )

        full_scale_rows = out_features
        full_scale_columns = in_features // MXFP4_BLOCK
        scale_offset = packed_offset + packed_bytes
        scale_header = _pread_exact(descriptor, _SCALE_HEADER.size, scale_offset)
        selector_bytes, tile_bytes, fixed_bytes, exception_count = _parse_scale_header(
            scale_header,
            expected_rows=full_scale_rows,
            expected_columns=full_scale_columns,
            expected_bytes=scale_bytes,
        )
        fixed = _pread_tensor_exact(
            descriptor,
            dtype=torch.uint8,
            shape=(full_scale_rows // X4T_TILE_ROWS, tile_bytes),
            offset=scale_offset + _SCALE_HEADER.size,
        )
        exceptions = _pread_tensor_exact(
            descriptor,
            dtype=torch.uint32,
            shape=(exception_count,),
            offset=scale_offset + _SCALE_HEADER.size + fixed_bytes,
        )
        positions = _validate_exception_positions(
            exceptions,
            rows=full_scale_rows,
            columns=full_scale_columns,
        )
        words = exceptions.numpy().astype(np.uint32, copy=False)
        values = words >> np.uint32(X4T_POSITION_BITS)

        if matrix in ("w1", "w3"):
            rank_rows = full_scale_rows // 12
            row_begin = rank * rank_rows
            row_end = row_begin + rank_rows
            tile_begin = row_begin // X4T_TILE_ROWS
            tile_end = row_end // X4T_TILE_ROWS
            rank_fixed = fixed[tile_begin:tile_end].contiguous().view(-1)
            rows = positions // np.uint32(full_scale_columns)
            keep = (rows >= row_begin) & (rows < row_end)
            local_positions = positions[keep] - np.uint32(
                row_begin * full_scale_columns
            )
            rank_words = local_positions | (
                values[keep] << np.uint32(X4T_POSITION_BITS)
            )
            rank_exceptions = torch.from_numpy(rank_words.copy())
            rank_scale_rows = rank_rows
            rank_scale_columns = full_scale_columns
        else:
            rank_scale_columns = full_scale_columns // 12
            column_begin = rank * rank_scale_columns
            # Bases remain exact for every rank slice.  Extract the one
            # selector byte belonging to this rank from each full K3 row.
            fixed_rows = fixed[:, X4T_TILE_ROWS:].reshape(
                -1, X4T_TILE_ROWS, selector_bytes
            )
            rank_fixed_2d = torch.empty(
                (fixed.shape[0], 2 * X4T_TILE_ROWS), dtype=torch.uint8
            )
            rank_fixed_2d[:, :X4T_TILE_ROWS].copy_(fixed[:, :X4T_TILE_ROWS])
            rank_fixed_2d[:, X4T_TILE_ROWS:].copy_(fixed_rows[:, :, rank])
            rank_fixed = rank_fixed_2d.view(-1)
            rows = positions // np.uint32(full_scale_columns)
            columns = positions % np.uint32(full_scale_columns)
            keep = (columns >= column_begin) & (
                columns < column_begin + rank_scale_columns
            )
            local_positions = (
                rows[keep] * np.uint32(rank_scale_columns)
                + columns[keep]
                - np.uint32(column_begin)
            )
            rank_words = local_positions | (
                values[keep] << np.uint32(X4T_POSITION_BITS)
            )
            rank_exceptions = torch.from_numpy(rank_words.copy())
            rank_scale_rows = full_scale_rows

        scale = X4TScaleComponents(
            fixed=rank_fixed.contiguous(),
            exceptions=rank_exceptions.contiguous(),
            rows=rank_scale_rows,
            columns=rank_scale_columns,
        )
        return X4TRankComponents(matrix=matrix, packed=packed, scale=scale)

    def read_rank_components(
        self, expert: int, matrix: str, rank: int
    ) -> X4TRankComponents:
        """Read a rank in its persistent GPU-facing X4T representation.

        Layer/directory and component geometry are checked here.  Expensive
        whole-record CRC and canonical re-encoding remain the responsibility
        of the offline artifact validator and :meth:`read`; serving should not
        reconstruct a dense scale plane merely to reproduce stored bytes.
        """

        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            return self._read_rank_components(
                descriptor,
                expert=expert,
                matrix=matrix,
                rank=rank,
            )
        finally:
            os.close(descriptor)

    def read_rank_triplet(
        self, expert: int, rank: int
    ) -> tuple[X4TRankComponents, X4TRankComponents, X4TRankComponents]:
        """Read w1/w3/w2 for one expert while sharing one file descriptor."""

        owned_descriptor = self._descriptor is None
        descriptor = (
            os.open(self.path, os.O_RDONLY) if owned_descriptor else self._descriptor
        )
        assert descriptor is not None
        mapping = self._mapping
        owned_mapping = False
        if mapping is None:
            mapping = mmap.mmap(descriptor, length=0, access=mmap.ACCESS_READ)
            owned_mapping = True
        try:
            return (
                self._read_rank_components(
                    descriptor, expert=expert, matrix="w1", rank=rank, mapping=mapping
                ),
                self._read_rank_components(
                    descriptor, expert=expert, matrix="w3", rank=rank, mapping=mapping
                ),
                self._read_rank_components(
                    descriptor, expert=expert, matrix="w2", rank=rank, mapping=mapping
                ),
            )
        finally:
            if owned_mapping:
                mapping.close()
            if owned_descriptor:
                os.close(descriptor)


__all__ = [
    "X4TLayerReader",
    "X4TMatrix",
    "X4TRankComponents",
    "X4TScaleComponents",
    "X4TTP12RankShard",
    "X4T_VERSION",
    "pack_x4t_scale_components",
    "read_x4t_tp12_rank_shard",
    "x4t_tp12_rank_filename",
]
