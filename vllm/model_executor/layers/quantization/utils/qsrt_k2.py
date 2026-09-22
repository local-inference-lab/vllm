# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read tensor-parallel-independent Kimi-K3 coupled QSRT-K2 atom extents."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_PROFILE = "k2_coupled_h512_h128"
_SCHEMAS = {"kquant_kimi_k3_qsrt_atoms_v2", "qsrt_kimi_k3_qsrt_atoms_v2"}
_EXPERTS = 896
_HIDDEN = 3584
_SLOTS = 96
_ROW_BYTES = 77_242_368
_FORMAT = "_qsrt_format_section"
_SCALES = "_qsrt_shared_scale_section"
_ATOMS = "qsrt_atoms"


@dataclass(frozen=True)
class QsrtK2Extent:
    """Contiguous source slots owned by one tensor-parallel rank."""

    first_slot: int
    slot_count: int

    @property
    def intermediate_size(self) -> int:
        return self.slot_count * 32


def plan_qsrt_k2_extent(layer: int, tp_size: int, tp_rank: int) -> QsrtK2Extent:
    """Partition both preactivation halves into complete 128-channel blocks.

    Rank ownership rotates between layers to balance unequal extents. The
    TP10 rotation advances two ranks per layer so 92 layers assign either
    880 or 884 slots per rank. This plans expert storage, not attention heads.
    """
    if not 1 <= layer <= 92:
        raise ValueError("QSRT-K2 layer must lie in 1..92")
    if not 2 <= tp_size <= 24 or not 0 <= tp_rank < tp_size:
        raise ValueError("QSRT-K2 requires TP in 2..24 and a rank within that group")
    step = 2 if tp_size == 10 else 1
    owner = (tp_rank - (layer - 1) * step) % tp_size
    first_half_ranks = (tp_size + 1) // 2
    half = int(owner >= first_half_ranks)
    half_rank = owner - half * first_half_ranks
    half_ranks = first_half_ranks if half == 0 else tp_size - first_half_ranks
    blocks, remainder = divmod(12, half_ranks)
    first_block = half_rank * blocks + min(half_rank, remainder)
    block_count = blocks + int(half_rank < remainder)
    return QsrtK2Extent(48 * half + 4 * first_block, 4 * block_count)


@dataclass(frozen=True)
class QsrtK2LayerSource:
    """Validated metadata; atom payload remains on disk until an extent is read."""

    path: Path
    layer: int
    scales: torch.Tensor
    rotation_draws: torch.Tensor

    def read_extent(self, tp_size: int, tp_rank: int) -> Any:
        """Lift one CPU extent into B12X's source-coordinate-aware weight format."""
        from b12x.moe._shared.kernels.w4a16.btx_compat import (
            lift_qsrt_atoms_v2_extent,
        )

        extent = plan_qsrt_k2_extent(self.layer, tp_size, tp_rank)
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            atoms = handle.get_slice(_ATOMS)[
                extent.first_slot : extent.first_slot + extent.slot_count
            ]
        if atoms.dtype != torch.uint8 or atoms.shape != (
            extent.slot_count,
            _ROW_BYTES,
        ):
            raise ValueError("QSRT-K2 atom extent has an invalid shape or dtype")
        return lift_qsrt_atoms_v2_extent(
            atoms,
            profile=_PROFILE,
            first_atom_slot=extent.first_slot,
            layer_index=self.layer,
            hidden_size=_HIDDEN,
            global_intermediate_size=3072,
            num_experts=_EXPERTS,
            gate_suh=self.scales[0],
            up_suh=self.scales[1],
            down_svh=self.scales[2],
            rotation_draws=self.rotation_draws,
        )


def read_qsrt_k2_source(root: Path, layer: int) -> QsrtK2LayerSource:
    """Validate a frozen Kimi-K3 K2 container without reading its atom payload."""
    plan_qsrt_k2_extent(layer, 2, 0)
    manifest = json.loads((root / "qsrt-manifest.json").read_text())
    if (
        manifest.get("codec") != "QSRT"
        or manifest.get("complete") is not True
        or manifest.get("storage_format") != "qsrt_atoms_v2"
        or manifest.get("storage_schema") not in _SCHEMAS
        or manifest.get("profile") != _PROFILE
        or manifest.get("all_experts_qsrt") is not True
    ):
        raise ValueError("Expected a complete uniform Kimi-K3 coupled QSRT-K2 manifest")
    entry = manifest.get("layers", {}).get(str(layer))
    if not isinstance(entry, dict):
        raise ValueError(f"QSRT-K2 manifest omits layer {layer}")
    name = entry.get("qsrt_atoms")
    if not isinstance(name, str) or name != f"qsrt-layer-{layer:05d}.safetensors":
        raise ValueError(f"QSRT-K2 manifest has an invalid layer-{layer} filename")
    path = root / name
    if path.stat().st_size != entry.get("atom_disk_bytes"):
        raise ValueError(f"QSRT-K2 layer-{layer} file size disagrees with its manifest")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {_FORMAT, _SCALES, _ATOMS}:
            raise ValueError("QSRT-K2 tensor inventory is noncanonical")
        metadata = handle.metadata() or {}
        required = {
            "format": "pt",
            "version": "2",
            "encoding": "qsrt_sqg_e4m3",
            "codebook": "sqg_xor_cheb_t12",
            "profile": _PROFILE,
            "profile_id": "3",
            "layer": str(layer),
            "experts": str(_EXPERTS),
            "intermediate_channels": "3072",
            "latent_channels": str(_HIDDEN),
            "atom_channels": "32",
            "atom_slots": str(_SLOTS),
            "atom_slot_stride_bytes": str(_ROW_BYTES),
            "p22_atom_bundle_bytes": "86208",
            "alignment_bytes": "4096",
            "residual_hadamard_block_size": "512",
            "preactivation_hadamard_block_size": "128",
            "postactivation_hadamard_block_size": "128",
            "intermediate_rotation_draws": "format_section[896:1792]",
        }
        if metadata.get("schema") not in _SCHEMAS:
            raise ValueError("QSRT-K2 atom container has an unsupported schema")
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"QSRT-K2 metadata {key!r}: expected {expected!r}, "
                    f"got {metadata.get(key)!r}"
                )
        formats = handle.get_tensor(_FORMAT)
        if (
            formats.dtype != torch.uint8
            or formats.shape != (4096,)
            or torch.any(formats[:_EXPERTS] != 0x44)
            or torch.any(formats[_EXPERTS : 2 * _EXPERTS] > 7)
            or torch.any(formats[2 * _EXPERTS :] != 0)
        ):
            raise ValueError("QSRT-K2 format section is malformed")
        shared = handle.get_tensor(_SCALES)
        scale_bytes = 3 * _HIDDEN * 2
        if (
            shared.dtype != torch.uint8
            or shared.shape != (24576,)
            or torch.any(shared[scale_bytes:] != 0)
        ):
            raise ValueError("QSRT-K2 shared scale section is malformed")
        scales = shared[:scale_bytes].clone().view(torch.float16).reshape(3, _HIDDEN)
        if not torch.isfinite(scales).all():
            raise ValueError("QSRT-K2 shared scales must be finite")
        if handle.get_slice(_ATOMS).get_shape() != [_SLOTS, _ROW_BYTES]:
            raise ValueError("QSRT-K2 atom slab has an invalid shape")
    return QsrtK2LayerSource(
        path=path,
        layer=layer,
        scales=scales,
        rotation_draws=formats[_EXPERTS : 2 * _EXPERTS].clone(),
    )
