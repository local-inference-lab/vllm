# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""InstantTensor loader for canonical TP-independent X4T safetensors layers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

X4T_VERSION = 2
X4T_SCHEMA = "kquant_x4t_layer_v1"
X4T_TILE_ROWS = 16
X4T_POSITION_BITS = 24
X4T_POSITION_MASK = (1 << X4T_POSITION_BITS) - 1
X4T_EXPERTS_PER_LAYER = 896
X4T_MATRIX_ORDER = ("w1", "w3", "w2")
MXFP4_BLOCK = 32

_MATRIX_SHAPES = {
    "w1": (3072, 3584),
    "w3": (3072, 3584),
    "w2": (3584, 3072),
}


def _tensor_name(matrix: str, part: str) -> str:
    if matrix not in X4T_MATRIX_ORDER:
        raise ValueError(f"unsupported X4T matrix {matrix!r}")
    return f"{matrix}.{part}"


def balanced_group_partition(shard_count: int, shard_index: int) -> tuple[int, int]:
    groups = _MATRIX_SHAPES["w2"][1] // MXFP4_BLOCK
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("X4T shard_count must be an integer")
    if not 1 <= shard_count <= groups:
        raise ValueError(f"X4T shard_count must lie in 1..{groups}")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("X4T shard_index must be an integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"X4T shard_index must lie in 0..{shard_count - 1}")
    quotient, remainder = divmod(groups, shard_count)
    count = quotient + int(shard_index < remainder)
    first = shard_index * quotient + min(shard_index, remainder)
    return first, count


@dataclass(frozen=True)
class X4TScaleComponents:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    rows: int
    columns: int

    def concatenate_rows(self, other: X4TScaleComponents) -> X4TScaleComponents:
        if self.columns != other.columns or self.fixed.device != other.fixed.device:
            raise ValueError("X4T row concatenation requires matching geometry/device")
        if other.exceptions.numel():
            words = other.exceptions.to(torch.int64)
            positions = words & X4T_POSITION_MASK
            values = words >> X4T_POSITION_BITS
            shifted = (
                (values << X4T_POSITION_BITS) | (positions + self.rows * self.columns)
            ).to(torch.uint32)
        else:
            shifted = torch.empty((0,), dtype=torch.uint32, device=self.fixed.device)
        return X4TScaleComponents(
            fixed=torch.cat((self.fixed, other.fixed)).contiguous(),
            exceptions=torch.cat((self.exceptions, shifted)).contiguous(),
            rows=self.rows + other.rows,
            columns=self.columns,
        )


@dataclass(frozen=True)
class X4TShardComponents:
    matrix: str
    packed: torch.Tensor
    scale: X4TScaleComponents


@dataclass(frozen=True)
class X4TLayerMetadata:
    path: Path
    layer: int
    expert_ids: tuple[int, ...]
    exception_offsets: dict[str, tuple[int, ...]]

    @property
    def expert_to_slot(self) -> dict[int, int]:
        return {expert: slot for slot, expert in enumerate(self.expert_ids)}


def _fixed_bytes(matrix: str) -> int:
    rows, features = _MATRIX_SHAPES[matrix]
    columns = features // MXFP4_BLOCK
    return rows * (1 + math.ceil(columns / 8))


def _packed_shape(matrix: str, experts: int) -> tuple[int, ...]:
    rows, features = _MATRIX_SHAPES[matrix]
    if matrix == "w2":
        return (experts, features // MXFP4_BLOCK, rows, MXFP4_BLOCK // 2)
    return (experts, rows, features // 2)


def read_x4t_layer_metadata(path: str | Path) -> X4TLayerMetadata:
    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema") != X4T_SCHEMA:
            raise ValueError("X4T safetensors schema is unsupported")
        if metadata.get("version") != str(X4T_VERSION):
            raise ValueError("X4T safetensors version is unsupported")
        try:
            layer = int(metadata["layer"])
            experts = int(metadata["experts"])
        except (KeyError, ValueError) as exc:
            raise ValueError("X4T safetensors metadata is invalid") from exc
        if not 1 <= layer <= 92 or not 0 <= experts <= X4T_EXPERTS_PER_LAYER:
            raise ValueError("X4T layer or expert count is invalid")
        expected_metadata = {
            "format": "pt",
            "expert_capacity": str(X4T_EXPERTS_PER_LAYER),
            "matrix_order": ",".join(X4T_MATRIX_ORDER),
            "scale_codec": "x4t-adjacent-pair-fixed-stream-v1",
            "w2_packed_layout": "group-major-32-channel-v1",
            "exact_mxfp4_reconstruction": "true",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"X4T metadata {name!r} is noncanonical")
        expected_keys = {"expert_ids"}
        for matrix in X4T_MATRIX_ORDER:
            expected_keys.update(
                _tensor_name(matrix, part)
                for part in (
                    "packed",
                    "scale_fixed",
                    "scale_exceptions",
                    "scale_exception_offsets",
                )
            )
        if set(handle.keys()) != expected_keys:
            raise ValueError("X4T safetensors tensor inventory is noncanonical")
        expert_tensor = handle.get_tensor("expert_ids")
        expert_ids = tuple(map(int, expert_tensor.tolist()))
        if (
            expert_tensor.dtype != torch.int32
            or tuple(expert_tensor.shape) != (experts,)
            or expert_ids != tuple(sorted(expert_ids))
            or len(set(expert_ids)) != experts
            or any(not 0 <= expert < X4T_EXPERTS_PER_LAYER for expert in expert_ids)
        ):
            raise ValueError("X4T expert_ids tensor is noncanonical")
        offsets_by_matrix: dict[str, tuple[int, ...]] = {}
        for matrix in X4T_MATRIX_ORDER:
            offsets = handle.get_tensor(_tensor_name(matrix, "scale_exception_offsets"))
            values = tuple(map(int, offsets.tolist()))
            if (
                offsets.dtype != torch.int64
                or tuple(offsets.shape) != (experts + 1,)
                or not values
                or values[0] != 0
                or any(left > right for left, right in zip(values, values[1:]))
            ):
                raise ValueError("X4T exception offsets are noncanonical")
            if tuple(
                handle.get_slice(_tensor_name(matrix, "packed")).get_shape()
            ) != _packed_shape(matrix, experts):
                raise ValueError("X4T packed tensor shape is invalid")
            if tuple(
                handle.get_slice(_tensor_name(matrix, "scale_fixed")).get_shape()
            ) != (experts, _fixed_bytes(matrix)):
                raise ValueError("X4T fixed scale tensor shape is invalid")
            if tuple(
                handle.get_slice(_tensor_name(matrix, "scale_exceptions")).get_shape()
            ) != (4 * values[-1],):
                raise ValueError("X4T exception bytes disagree with their offsets")
            offsets_by_matrix[matrix] = values
    return X4TLayerMetadata(path, layer, expert_ids, offsets_by_matrix)


def _validate_exception_words(
    exceptions: torch.Tensor, *, rows: int, columns: int
) -> None:
    if exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
        raise TypeError("X4T exceptions must be one-dimensional uint32")
    if exceptions.numel():
        positions = exceptions.to(torch.int64) & X4T_POSITION_MASK
        if int(positions[-1]) >= rows * columns or bool(
            torch.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError("X4T exception positions are invalid")


def _slice_scale_components(
    *,
    matrix: str,
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    first_group: int,
    group_count: int,
) -> X4TScaleComponents:
    full_rows, features = _MATRIX_SHAPES[matrix]
    full_columns = features // MXFP4_BLOCK
    _validate_exception_words(exceptions, rows=full_rows, columns=full_columns)
    words = exceptions.to(torch.int64)
    positions = words & X4T_POSITION_MASK
    values = words >> X4T_POSITION_BITS
    if matrix in ("w1", "w3"):
        rows = group_count * MXFP4_BLOCK
        row_begin = first_group * MXFP4_BLOCK
        source_rows = positions // full_columns
        keep = (source_rows >= row_begin) & (source_rows < row_begin + rows)
        local = positions[keep] - row_begin * full_columns
        shard_exceptions = ((values[keep] << X4T_POSITION_BITS) | local).to(
            torch.uint32
        )
        return X4TScaleComponents(
            fixed=fixed.contiguous(),
            exceptions=shard_exceptions.contiguous(),
            rows=rows,
            columns=full_columns,
        )

    selector_bytes = math.ceil(full_columns / 8)
    tile_bytes = X4T_TILE_ROWS * (1 + selector_bytes)
    fixed_2d = fixed.reshape(full_rows // X4T_TILE_ROWS, tile_bytes)
    bases = fixed_2d[:, :X4T_TILE_ROWS].reshape(full_rows)
    selectors = fixed_2d[:, X4T_TILE_ROWS:].reshape(full_rows, selector_bytes)
    source_columns = torch.arange(
        first_group,
        first_group + group_count,
        dtype=torch.int64,
        device=fixed.device,
    )
    selected = (
        selectors[:, source_columns // 8].to(torch.int16)
        >> (source_columns % 8).to(torch.int16)
    ) & 1
    local_selector_bytes = math.ceil(group_count / 8)
    local_selectors = torch.zeros(
        (full_rows, local_selector_bytes), dtype=torch.uint8, device=fixed.device
    )
    for bit in range(group_count):
        local_selectors[:, bit // 8] |= selected[:, bit].to(torch.uint8) << (bit % 8)
    local_tile_bytes = X4T_TILE_ROWS * (1 + local_selector_bytes)
    local_fixed = torch.empty(
        (full_rows // X4T_TILE_ROWS, local_tile_bytes),
        dtype=torch.uint8,
        device=fixed.device,
    )
    local_fixed[:, :X4T_TILE_ROWS].copy_(bases.reshape(-1, X4T_TILE_ROWS))
    local_fixed[:, X4T_TILE_ROWS:].copy_(
        local_selectors.reshape(full_rows // X4T_TILE_ROWS, -1)
    )
    source_columns_all = positions % full_columns
    keep = (source_columns_all >= first_group) & (
        source_columns_all < first_group + group_count
    )
    local = (
        (positions[keep] // full_columns) * group_count
        + source_columns_all[keep]
        - first_group
    )
    shard_exceptions = ((values[keep] << X4T_POSITION_BITS) | local).to(torch.uint32)
    return X4TScaleComponents(
        fixed=local_fixed.reshape(-1).contiguous(),
        exceptions=shard_exceptions.contiguous(),
        rows=full_rows,
        columns=group_count,
    )


def _restrict_instanttensor_to_extents(
    opener: Any,
    extents: list[tuple[str, str, int, int]],
) -> None:
    required = (
        "filename",
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required if not hasattr(opener, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor cannot select X4T extents "
            f"(missing: {', '.join(missing)})"
        )
    if opener.loader_handle is not None:
        raise RuntimeError("X4T extent selection must occur before InstantTensor I/O")
    source_path = str(opener.filename[0])
    base_offsets = {
        name: int(opener.tensor_offsets[index][1])
        for name, index in opener.tensor_name_to_index.items()
    }
    filenames: list[str] = []
    metadata: list[tuple[str, dict[str, object]]] = []
    offsets: list[tuple[int, int]] = []
    sizes: list[int] = []
    for logical_name, tensor_name, relative_offset, size in extents:
        if size <= 0:
            continue
        file_index = len(filenames)
        begin = base_offsets[tensor_name] + relative_offset
        end = begin + size
        filenames.append(source_path)
        metadata.append(
            (
                logical_name,
                {"dtype": "U8", "shape": [size], "data_offsets": [0, size]},
            )
        )
        offsets.extend(((file_index, begin), (file_index, end)))
        sizes.append(size)
    if not metadata:
        raise RuntimeError("X4T InstantTensor selection is empty")
    opener.filename = filenames
    opener.ordered_tensor_metadatas = metadata
    opener.tensor_name_to_index = {
        name: index for index, (name, _) in enumerate(metadata)
    }
    opener.tensor_offsets = offsets
    opener.tensor_sizes = sizes
    opener.total_tensor_size = sum(sizes)
    opener._determine_buffer_size(None)


class X4TLayerReader:
    """Read balanced X4T shards via safetensors or direct-GPU InstantTensor."""

    def __init__(
        self,
        path: str | Path,
        *,
        shard_count: int = 1,
        shard_index: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.metadata = read_x4t_layer_metadata(path)
        self.path = self.metadata.path
        self.layer = self.metadata.layer
        self.expert_ids = self.metadata.expert_ids
        self._slots = self.metadata.expert_to_slot
        self.shard_count = shard_count
        self.shard_index = shard_index
        self.first_group, self.group_count = balanced_group_partition(
            shard_count, shard_index
        )
        self.device = torch.device(device)
        self._loaded: dict[str, torch.Tensor] | None = None

    def __enter__(self) -> X4TLayerReader:
        if self.device.type == "cuda":
            self._load_cuda_extents()
        return self

    def __exit__(self, *_args: object) -> None:
        self._loaded = None

    def has(self, expert: int, matrix: str) -> bool:
        _tensor_name(matrix, "packed")
        return expert in self._slots

    def _extent_plan(self) -> list[tuple[str, str, int, int]]:
        extents: list[tuple[str, str, int, int]] = []
        first_row = self.first_group * MXFP4_BLOCK
        shard_rows = self.group_count * MXFP4_BLOCK
        for expert, slot in self._slots.items():
            for matrix in ("w1", "w3"):
                rows, features = _MATRIX_SHAPES[matrix]
                packed_row_bytes = features // 2
                packed_per_expert = rows * packed_row_bytes
                extents.append(
                    (
                        f"{matrix}.{expert}.packed",
                        _tensor_name(matrix, "packed"),
                        slot * packed_per_expert + first_row * packed_row_bytes,
                        shard_rows * packed_row_bytes,
                    )
                )
                columns = features // MXFP4_BLOCK
                tile_bytes = X4T_TILE_ROWS * (1 + math.ceil(columns / 8))
                fixed_per_expert = _fixed_bytes(matrix)
                extents.append(
                    (
                        f"{matrix}.{expert}.fixed",
                        _tensor_name(matrix, "scale_fixed"),
                        slot * fixed_per_expert
                        + (first_row // X4T_TILE_ROWS) * tile_bytes,
                        (shard_rows // X4T_TILE_ROWS) * tile_bytes,
                    )
                )
            rows, features = _MATRIX_SHAPES["w2"]
            group_bytes = rows * (MXFP4_BLOCK // 2)
            packed_per_expert = (features // MXFP4_BLOCK) * group_bytes
            extents.append(
                (
                    f"w2.{expert}.packed",
                    _tensor_name("w2", "packed"),
                    slot * packed_per_expert + self.first_group * group_bytes,
                    self.group_count * group_bytes,
                )
            )
            fixed_per_expert = _fixed_bytes("w2")
            extents.append(
                (
                    f"w2.{expert}.fixed",
                    _tensor_name("w2", "scale_fixed"),
                    slot * fixed_per_expert,
                    fixed_per_expert,
                )
            )
            for matrix in X4T_MATRIX_ORDER:
                offsets = self.metadata.exception_offsets[matrix]
                first = offsets[slot]
                count = offsets[slot + 1] - first
                extents.append(
                    (
                        f"{matrix}.{expert}.exceptions",
                        _tensor_name(matrix, "scale_exceptions"),
                        4 * first,
                        4 * count,
                    )
                )
        return extents

    def _load_cuda_extents(self) -> None:
        import instanttensor

        device = self.device
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())

        opener = instanttensor.safe_open(
            str(self.path),
            framework="pt",
            device=device,
            load_now=False,
            copy=True,
        )
        _restrict_instanttensor_to_extents(opener, self._extent_plan())
        with opener as handle:
            self._loaded = dict(handle.tensors())

    def _cpu_extent(self, expert: int, matrix: str, part: str) -> torch.Tensor:
        slot = self._slots[expert]
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            if part == "packed":
                source = handle.get_slice(_tensor_name(matrix, "packed"))
                if matrix in ("w1", "w3"):
                    first_row = self.first_group * MXFP4_BLOCK
                    rows = self.group_count * MXFP4_BLOCK
                    return source[slot, first_row : first_row + rows].contiguous()
                return (
                    source[slot, self.first_group : self.first_group + self.group_count]
                    .contiguous()
                    .reshape(-1)
                )
            if part == "fixed":
                source = handle.get_slice(_tensor_name(matrix, "scale_fixed"))
                if matrix == "w2":
                    return source[slot].contiguous()
                _, features = _MATRIX_SHAPES[matrix]
                columns = features // MXFP4_BLOCK
                tile_bytes = X4T_TILE_ROWS * (1 + math.ceil(columns / 8))
                first_tile = self.first_group * MXFP4_BLOCK // X4T_TILE_ROWS
                tiles = self.group_count * MXFP4_BLOCK // X4T_TILE_ROWS
                return source[
                    slot,
                    first_tile * tile_bytes : (first_tile + tiles) * tile_bytes,
                ].contiguous()
            offsets = self.metadata.exception_offsets[matrix]
            first, end = offsets[slot : slot + 2]
            raw = handle.get_slice(_tensor_name(matrix, "scale_exceptions"))[
                4 * first : 4 * end
            ].contiguous()
            return raw.view(torch.uint32)

    def _component(self, expert: int, matrix: str, part: str) -> torch.Tensor:
        if expert not in self._slots:
            raise KeyError((expert, matrix))
        if self._loaded is None:
            return self._cpu_extent(expert, matrix, part)
        name = f"{matrix}.{expert}.{part if part != 'fixed' else 'fixed'}"
        value = self._loaded.get(name)
        if value is None:
            if part == "exceptions":
                return torch.empty((0,), dtype=torch.uint32, device=self.device)
            raise KeyError(name)
        if part == "exceptions":
            return value.view(torch.uint32)
        return value

    def read_shard_triplet(
        self,
        expert: int,
        shard_count: int | None = None,
        shard_index: int | None = None,
    ) -> tuple[X4TShardComponents, X4TShardComponents, X4TShardComponents]:
        if shard_count is not None and (
            shard_count != self.shard_count or shard_index != self.shard_index
        ):
            raise ValueError("X4T reader shard geometry is immutable")
        result = []
        for matrix in X4T_MATRIX_ORDER:
            packed = self._component(expert, matrix, "packed")
            rows, _ = _MATRIX_SHAPES[matrix]
            if matrix == "w2":
                packed = (
                    packed.reshape(self.group_count, rows, MXFP4_BLOCK // 2)
                    .permute(1, 0, 2)
                    .reshape(rows, self.group_count * MXFP4_BLOCK // 2)
                    .contiguous()
                )
            else:
                packed = packed.reshape(self.group_count * MXFP4_BLOCK, -1).contiguous()
            scale = _slice_scale_components(
                matrix=matrix,
                fixed=self._component(expert, matrix, "fixed"),
                exceptions=self._component(expert, matrix, "exceptions"),
                first_group=self.first_group,
                group_count=self.group_count,
            )
            result.append(X4TShardComponents(matrix, packed, scale))
        return tuple(result)  # type: ignore[return-value]


__all__ = [
    "X4TLayerMetadata",
    "X4TLayerReader",
    "X4TScaleComponents",
    "X4TShardComponents",
    "X4T_VERSION",
    "balanced_group_partition",
    "read_x4t_layer_metadata",
]
