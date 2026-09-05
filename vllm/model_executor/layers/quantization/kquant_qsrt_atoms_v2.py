# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reader for revision two of the TP-independent QSRT atom container."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
    _INSTANTTENSOR_IO_DEPTH,
    _INSTANTTENSOR_MAX_FREE_MEM_USAGE,
    _concrete_device,
    _select_instanttensor_extent,
    balanced_atom_partition,
)

SCHEMAS = {
    "kquant_kimi_k3_qsrt_atoms_v2",
    "qsrt_kimi_k3_qsrt_atoms_v2",
}
ENCODING = "qsrt_sqg_e4m3"
CODEBOOK = "sqg_xor_cheb_t12"
PROFILE = "k3x22_k4x2"
PURE_K2_PROFILE = "k2_coupled_h512_h128"
COUPLED_H308_PROFILE = "k3x22_k4x2_coupled_h512_h128"
VERSION = 2
PROFILE_ID = 2
PURE_K2_PROFILE_ID = 3
COUPLED_H308_PROFILE_ID = 4
EXPERTS = 896
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 3072
ATOM_CHANNELS = 32
ATOM_SLOTS = 96
P33_ATOM_BUNDLE_BYTES = 129216
P43_ATOM_BUNDLE_BYTES = 150720
P22_ATOM_BUNDLE_BYTES = 86208
COUPLED_H308_P33_P33_ATOM_BUNDLE_BYTES = 129216
COUPLED_H308_P43_P33_ATOM_BUNDLE_BYTES = 143552
COUPLED_H308_P43_P44_ATOM_BUNDLE_BYTES = 157888
H308_ATOM_SLOT_STRIDE_BYTES = 119005184
P22_ATOM_SLOT_STRIDE_BYTES = 77242368
FORMAT_SECTION_BYTES = 4096
FORMAT_TABLE_BYTES = EXPERTS
ROTATION_DRAW_OFFSET = EXPERTS
SHARED_SCALE_SECTION_BYTES = 24576
SHARED_SCALE_BYTES = 3 * HIDDEN_SIZE * torch.float16.itemsize
FORMAT_H308 = 0x33
ATOMS_PER_PAIR = 8
PAIRS = ATOM_SLOTS // ATOMS_PER_PAIR
ALIGNMENT_BYTES = 4096
P33_MATRIX_TRELLIS_BYTES = 43008
P43_MATRIX_TRELLIS_BYTES = 50176
P44_MATRIX_TRELLIS_BYTES = 57344
COUPLED_H308_PAIR_BUNDLE_BYTES = (
    *(COUPLED_H308_P33_P33_ATOM_BUNDLE_BYTES for _ in range(5)),
    COUPLED_H308_P43_P33_ATOM_BUNDLE_BYTES,
    *(COUPLED_H308_P33_P33_ATOM_BUNDLE_BYTES for _ in range(5)),
    COUPLED_H308_P43_P44_ATOM_BUNDLE_BYTES,
)

FORMAT_TENSOR = "_qsrt_format_section"
SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
ATOM_TENSOR = "qsrt_atoms"
TENSOR_INVENTORY = {FORMAT_TENSOR, SHARED_SCALE_TENSOR, ATOM_TENSOR}


@dataclass(frozen=True)
class QSRTAtomV2LayerMetadata:
    path: Path
    layer: int
    profile: str
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    rotation_draws: torch.Tensor | None
    atom_slot_stride_bytes: int | None
    atom_pair_bundle_bytes: tuple[int, ...] | None
    atom_slab_bytes: int

    @property
    def coupled_h308(self) -> bool:
        return self.profile == COUPLED_H308_PROFILE


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"QSRT metadata {name!r} is missing or invalid") from exc
    return value


def _atom_slot_stride_for_profile(profile: str) -> int:
    if profile == PROFILE:
        return H308_ATOM_SLOT_STRIDE_BYTES
    if profile == PURE_K2_PROFILE:
        return P22_ATOM_SLOT_STRIDE_BYTES
    raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")


def _align_up(value: int, alignment: int = ALIGNMENT_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def _coupled_pair_extent(physical_pair: int) -> tuple[int, int, int]:
    """Return ``(begin, extent, row_bytes)`` within the flat atom tensor."""

    if not 0 <= physical_pair < PAIRS:
        raise ValueError(f"physical_pair must lie in 0..{PAIRS - 1}")
    row_bytes = _align_up(EXPERTS * COUPLED_H308_PAIR_BUNDLE_BYTES[physical_pair])
    begin = sum(
        ATOMS_PER_PAIR * _align_up(EXPERTS * bundle_bytes)
        for bundle_bytes in COUPLED_H308_PAIR_BUNDLE_BYTES[:physical_pair]
    )
    return begin, ATOMS_PER_PAIR * row_bytes, row_bytes


def _coupled_h308_pair_for_rank(layer: int, shard_count: int, shard_index: int) -> int:
    """Assign one complete physical pair while balancing high-rate pairs."""

    if shard_count != PAIRS:
        raise ValueError(f"coupled H308 serving requires TP={PAIRS}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    if not 1 <= layer <= 92:
        raise ValueError("layer must identify a Kimi-K3 MoE layer")
    return (shard_index + layer - 1) % PAIRS


def _balanced_pure_k2_atom_partition(
    shard_count: int, shard_index: int
) -> tuple[int, int]:
    """Partition each preactivation half into complete H128 atom groups."""

    if shard_count == 1:
        return 0, ATOM_SLOTS
    if shard_count < 1 or shard_count > ATOM_SLOTS or shard_count % 2:
        raise ValueError("pure-K2 atom sharding requires an even shard count")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    ranks_per_half = shard_count // 2
    half = shard_index // ranks_per_half
    rank_in_half = shard_index % ranks_per_half
    blocks_per_half = (ATOM_SLOTS // 2) // 4
    quotient, remainder = divmod(blocks_per_half, ranks_per_half)
    block_count = quotient + int(rank_in_half < remainder)
    first_block = rank_in_half * quotient + min(rank_in_half, remainder)
    return half * (ATOM_SLOTS // 2) + 4 * first_block, 4 * block_count


COUPLED_H308_ATOM_SLAB_BYTES = sum(
    ATOMS_PER_PAIR * _align_up(EXPERTS * bundle_bytes)
    for bundle_bytes in COUPLED_H308_PAIR_BUNDLE_BYTES
)


def read_qsrt_atom_v2_layer_metadata(
    path: str | Path,
    *,
    layer: int,
    expected_bits: Sequence[int] | None = None,
) -> QSRTAtomV2LayerMetadata:
    """Validate atoms-v2 metadata without reading its large atom tensor."""

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != TENSOR_INVENTORY:
            raise ValueError("QSRT atoms-v2 tensor inventory is noncanonical")
        profile = metadata.get("profile")
        if profile not in {PROFILE, PURE_K2_PROFILE, COUPLED_H308_PROFILE}:
            raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")
        pure_k2 = profile == PURE_K2_PROFILE
        coupled_h308 = profile == COUPLED_H308_PROFILE
        profile_id = (
            PURE_K2_PROFILE_ID
            if pure_k2
            else COUPLED_H308_PROFILE_ID
            if coupled_h308
            else PROFILE_ID
        )
        expected_metadata = {
            "format": "pt",
            "version": str(VERSION),
            "encoding": ENCODING,
            "codebook": CODEBOOK,
            "profile": profile,
            "profile_id": str(profile_id),
            "layer": str(layer),
            "experts": str(EXPERTS),
            "intermediate_channels": str(INTERMEDIATE_SIZE),
            "latent_channels": str(HIDDEN_SIZE),
            "atom_channels": str(ATOM_CHANNELS),
            "atom_slots": str(ATOM_SLOTS),
            "alignment_bytes": "4096",
        }
        if metadata.get("schema") not in SCHEMAS:
            raise ValueError(
                "QSRT atoms-v2 metadata schema mismatch: "
                f"{metadata.get('schema')!r} not in {sorted(SCHEMAS)!r}"
            )
        if pure_k2 or coupled_h308:
            expected_metadata.update(
                {
                    "residual_hadamard_block_size": "512",
                    "preactivation_hadamard_block_size": "128",
                    "postactivation_hadamard_block_size": "128",
                    "intermediate_rotation_draws": "format_section[896:1792]",
                }
            )
        if pure_k2:
            expected_metadata["p22_atom_bundle_bytes"] = str(P22_ATOM_BUNDLE_BYTES)
        elif coupled_h308:
            expected_metadata.update(
                {
                    "atom_storage": "pair_variable_stride",
                    "atom_pair_bundle_bytes": ",".join(
                        str(value) for value in COUPLED_H308_PAIR_BUNDLE_BYTES
                    ),
                    "p33_matrix_trellis_bytes": str(P33_MATRIX_TRELLIS_BYTES),
                    "p43_matrix_trellis_bytes": str(P43_MATRIX_TRELLIS_BYTES),
                    "p44_matrix_trellis_bytes": str(P44_MATRIX_TRELLIS_BYTES),
                }
            )
        else:
            expected_metadata.update(
                {
                    "p33_atom_bundle_bytes": str(P33_ATOM_BUNDLE_BYTES),
                    "p43_atom_bundle_bytes": str(P43_ATOM_BUNDLE_BYTES),
                }
            )
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"QSRT atoms-v2 metadata {name} mismatch: "
                    f"{metadata.get(name)!r} != {expected!r}"
                )
        stride = None
        if not coupled_h308:
            stride = _metadata_int(metadata, "atom_slot_stride_bytes")
            expected_stride = _atom_slot_stride_for_profile(profile)
            if stride != expected_stride:
                raise ValueError(
                    "QSRT atoms-v2 atom-slot stride mismatch: "
                    f"{stride} != {expected_stride} for profile {profile!r}"
                )

        format_section = handle.get_tensor(FORMAT_TENSOR)
        expected_format = 0x44 if pure_k2 else FORMAT_H308
        has_rotation_draws = pure_k2 or coupled_h308
        padding_begin = (
            ROTATION_DRAW_OFFSET + EXPERTS if has_rotation_draws else FORMAT_TABLE_BYTES
        )
        if (
            format_section.dtype != torch.uint8
            or tuple(format_section.shape) != (FORMAT_SECTION_BYTES,)
            or bool(torch.any(format_section[:FORMAT_TABLE_BYTES] != expected_format))
            or bool(torch.any(format_section[padding_begin:] != 0))
        ):
            raise ValueError("QSRT atoms-v2 format section is malformed")
        expected_bit = 2 if pure_k2 else 3
        if expected_bits is not None and (
            len(expected_bits) != EXPERTS
            or any(bit != expected_bit for bit in expected_bits)
        ):
            raise ValueError(
                f"QSRT atoms-v2 profile {profile!r} requires all experts at "
                f"{expected_bit} bits"
            )
        rotation_draws = None
        if has_rotation_draws:
            rotation_draws = format_section[
                ROTATION_DRAW_OFFSET : ROTATION_DRAW_OFFSET + EXPERTS
            ].clone()
            if bool(torch.any(rotation_draws > 7)):
                raise ValueError("QSRT atoms-v2 contains an invalid rotation draw")

        shared = handle.get_tensor(SHARED_SCALE_TENSOR)
        if (
            shared.dtype != torch.uint8
            or tuple(shared.shape) != (SHARED_SCALE_SECTION_BYTES,)
            or bool(torch.any(shared[SHARED_SCALE_BYTES:] != 0))
        ):
            raise ValueError("QSRT atoms-v2 shared-scale section is malformed")
        vectors = (
            shared[:SHARED_SCALE_BYTES]
            .clone()
            .view(torch.float16)
            .reshape(3, HIDDEN_SIZE)
            .contiguous()
        )
        if not bool(torch.all(torch.isfinite(vectors))):
            raise ValueError("QSRT atoms-v2 shared scales contain non-finite values")
        atom_shape = handle.get_slice(ATOM_TENSOR).get_shape()
        expected_atom_shape = (
            [COUPLED_H308_ATOM_SLAB_BYTES] if coupled_h308 else [ATOM_SLOTS, stride]
        )
        if atom_shape != expected_atom_shape:
            raise ValueError(
                f"QSRT atoms-v2 slab shape {atom_shape} != {expected_atom_shape}"
            )

        if coupled_h308:
            atom_slab_bytes = COUPLED_H308_ATOM_SLAB_BYTES
        else:
            assert stride is not None
            atom_slab_bytes = ATOM_SLOTS * stride

    return QSRTAtomV2LayerMetadata(
        path=path,
        layer=layer,
        profile=profile,
        gate_suh=vectors[0].contiguous(),
        up_suh=vectors[1].contiguous(),
        down_svh=vectors[2].contiguous(),
        rotation_draws=rotation_draws,
        atom_slot_stride_bytes=stride,
        atom_pair_bundle_bytes=(
            COUPLED_H308_PAIR_BUNDLE_BYTES if coupled_h308 else None
        ),
        atom_slab_bytes=atom_slab_bytes,
    )


def _select_instanttensor_flat_extent(
    handle: Any,
    *,
    tensor_name: str,
    tensor_bytes: int,
    begin: int,
    rows: int,
    row_bytes: int,
) -> None:
    """Restrict an unopened InstantTensor handle to a flat byte extent."""

    required = (
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required if not hasattr(handle, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor cannot select a QSRT atom extent "
            f"(missing: {', '.join(missing)})"
        )
    if handle.loader_handle is not None:
        raise RuntimeError("QSRT extent selection must occur before InstantTensor I/O")
    try:
        tensor_index = handle.tensor_name_to_index[tensor_name]
    except KeyError as exc:
        raise ValueError(f"QSRT file omits tensor {tensor_name!r}") from exc
    file_index, tensor_start = handle.tensor_offsets[tensor_index]
    end_file_index, tensor_end = handle.tensor_offsets[tensor_index + 1]
    if file_index != end_file_index or tensor_end - tensor_start != tensor_bytes:
        raise RuntimeError("InstantTensor QSRT atom offsets disagree with metadata")
    extent = rows * row_bytes
    if begin < 0 or begin + extent > tensor_bytes:
        raise ValueError("QSRT atom extent lies outside the serialized tensor")
    absolute_begin = tensor_start + begin
    absolute_end = absolute_begin + extent
    selected_metadata = {
        "dtype": "U8",
        "shape": [rows, row_bytes],
        "data_offsets": [0, extent],
    }
    handle.ordered_tensor_metadatas = [(tensor_name, selected_metadata)]
    handle.tensor_name_to_index = {tensor_name: 0}
    handle.tensor_offsets = [
        (file_index, absolute_begin),
        (file_index, absolute_end),
    ]
    handle.tensor_sizes = [extent]
    handle.total_tensor_size = extent
    handle._determine_buffer_size(None)


@contextmanager
def open_qsrt_atom_v2_extent(
    metadata: QSRTAtomV2LayerMetadata,
    *,
    shard_count: int,
    shard_index: int,
    device: torch.device | str | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield one shard's contiguous padded atom rows as ``[A, stride]``."""

    if metadata.profile == PURE_K2_PROFILE and shard_count == 9:
        from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank

        extent = plan_qsrt_tp9_rank(metadata.layer, shard_index)
        first, rows = extent.first_atom, extent.atom_count
    else:
        first, rows = (
            _balanced_pure_k2_atom_partition(shard_count, shard_index)
            if metadata.profile == PURE_K2_PROFILE
            else balanced_atom_partition(shard_count, shard_index)
        )
    if metadata.coupled_h308:
        if rows != ATOMS_PER_PAIR or first % ATOMS_PER_PAIR:
            raise ValueError(
                "coupled H308 serving requires one aligned eight-atom pair per rank"
            )
        physical_pair = _coupled_h308_pair_for_rank(
            metadata.layer, shard_count, shard_index
        )
        first = physical_pair * ATOMS_PER_PAIR
        begin, extent, row_bytes = _coupled_pair_extent(physical_pair)
        if extent != rows * row_bytes:
            raise AssertionError("coupled H308 extent accounting drifted")
        if device is None or torch.device(device).type == "cpu":
            with safe_open(metadata.path, framework="pt", device="cpu") as handle:
                flat = handle.get_slice(ATOM_TENSOR)[begin : begin + extent]
                yield first, flat.contiguous().reshape(rows, row_bytes)
            return

        import instanttensor

        opener = instanttensor.safe_open(
            str(metadata.path),
            framework="pt",
            device=_concrete_device(device),
            concurrency=1,
            io_depth=_INSTANTTENSOR_IO_DEPTH,
            max_free_mem_usage=_INSTANTTENSOR_MAX_FREE_MEM_USAGE,
            load_now=False,
            copy=False,
        )
        _select_instanttensor_flat_extent(
            opener,
            tensor_name=ATOM_TENSOR,
            tensor_bytes=metadata.atom_slab_bytes,
            begin=begin,
            rows=rows,
            row_bytes=row_bytes,
        )
        with opener as handle:
            yield first, dict(handle.tensors())[ATOM_TENSOR]
        return

    assert metadata.atom_slot_stride_bytes is not None
    if device is None or torch.device(device).type == "cpu":
        with safe_open(metadata.path, framework="pt", device="cpu") as handle:
            yield (
                first,
                handle.get_slice(ATOM_TENSOR)[first : first + rows].contiguous(),
            )
        return

    import instanttensor

    opener = instanttensor.safe_open(
        str(metadata.path),
        framework="pt",
        device=_concrete_device(device),
        concurrency=1,
        io_depth=_INSTANTTENSOR_IO_DEPTH,
        max_free_mem_usage=_INSTANTTENSOR_MAX_FREE_MEM_USAGE,
        load_now=False,
        copy=False,
    )
    _select_instanttensor_extent(
        opener,
        tensor_name=ATOM_TENSOR,
        first_row=first,
        rows=rows,
        row_bytes=metadata.atom_slot_stride_bytes,
    )
    with opener as handle:
        yield first, dict(handle.tensors())[ATOM_TENSOR]


__all__ = [
    "COUPLED_H308_PROFILE",
    "QSRTAtomV2LayerMetadata",
    "PURE_K2_PROFILE",
    "_atom_slot_stride_for_profile",
    "_coupled_pair_extent",
    "open_qsrt_atom_v2_extent",
    "read_qsrt_atom_v2_layer_metadata",
]
