# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel extents for EXL3 routed-expert checkpoints."""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXL3_MANIFEST_FILENAME = "exl3-manifest.json"


@dataclass(frozen=True)
class Exl3Extent:
    """Contiguous slots of one MoE layer owned by one tensor-parallel rank."""

    first_slot: int
    slot_count: int
    slot_channels: int

    @property
    def intermediate_size(self) -> int:
        return self.slot_count * self.slot_channels


@functools.cache
def load_exl3_manifest(root: str) -> Any:
    """Read and validate a checkpoint's EXL3 manifest once per process."""
    from b12x.moe.checkpoints.exl3 import read_exl3_manifest

    path = Path(root)
    if not (path / EXL3_MANIFEST_FILENAME).is_file():
        raise ValueError(
            "EXL3 experts require a local checkpoint directory containing "
            f"{EXL3_MANIFEST_FILENAME}"
        )
    return read_exl3_manifest(path)


def plan_exl3_extent(
    manifest: Any, layer: int, tp_size: int, tp_rank: int
) -> Exl3Extent:
    """Partition both intermediate halves into complete aligned blocks.

    The manifest's single extent barrier splits the intermediate axis into two
    halves that no rank extent may cross; each half is divided into blocks of
    ``extent_alignment_slots``. Ranks own contiguous block ranges, and
    ownership rotates between layers to balance unequal extents. At TP10 the
    rotation advances two ranks per layer so every rank holds 880 or 884 of the
    Kimi-K3 slots. This plans expert storage, not attention heads.
    """
    geometry = manifest.geometry
    layout = manifest.layout
    layers = tuple(geometry.moe_layer_indices)
    if layer not in layers:
        raise ValueError(f"EXL3 manifest does not declare MoE layer {layer}")
    if not 2 <= tp_size <= 24 or not 0 <= tp_rank < tp_size:
        raise ValueError(
            "EXL3 experts require TP in 2..24 and a rank within that group"
        )
    half_slots = geometry.num_slots // 2
    alignment = layout.extent_alignment_slots
    if (
        tuple(layout.extent_barriers) != (half_slots,)
        or 2 * half_slots != geometry.num_slots
        or half_slots % alignment
    ):
        raise NotImplementedError(
            "EXL3 extent planning requires one barrier between aligned halves"
        )
    blocks_per_half = half_slots // alignment
    step = 2 if tp_size == 10 else 1
    owner = (tp_rank - layers.index(layer) * step) % tp_size
    first_half_ranks = (tp_size + 1) // 2
    half = int(owner >= first_half_ranks)
    half_rank = owner - half * first_half_ranks
    half_ranks = first_half_ranks if half == 0 else tp_size - first_half_ranks
    blocks, remainder = divmod(blocks_per_half, half_ranks)
    first_block = half_rank * blocks + min(half_rank, remainder)
    block_count = blocks + int(half_rank < remainder)
    return Exl3Extent(
        half_slots * half + alignment * first_block,
        alignment * block_count,
        geometry.slot_channels,
    )
