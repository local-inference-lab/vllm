# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
    FRUIT_SCHEMA,
    balanced_atom_partition,
    open_qsrt_atom_extent,
    read_qsrt_atom_layer_metadata,
)


def test_reads_and_partitions_fruit_qsrt_atoms(tmp_path: Path) -> None:
    layer = 3
    experts = 3
    hidden_size = 1024
    intermediate_size = 512
    atom_slots = 16
    atom_bundle_bytes = 37_056
    alignment = 4096
    slot_payload_bytes = experts * atom_bundle_bytes
    slot_stride_bytes = (
        (slot_payload_bytes + alignment - 1) // alignment * alignment
    )

    format_section = torch.zeros(alignment, dtype=torch.uint8)
    format_section[:experts] = torch.tensor([0x00, 0x11, 0x21], dtype=torch.uint8)
    shared_scale_bytes = 3 * experts * hidden_size * 2
    shared_scale_section_bytes = (
        (shared_scale_bytes + alignment - 1) // alignment * alignment
    )
    shared_scale_section = torch.zeros(
        shared_scale_section_bytes, dtype=torch.uint8
    )
    shared_scale_section[:shared_scale_bytes].copy_(
        torch.ones((3, experts, hidden_size), dtype=torch.float16)
        .view(torch.uint8)
        .reshape(-1)
    )
    atom_slab = torch.zeros(
        (atom_slots, slot_stride_bytes), dtype=torch.uint8
    )
    path = tmp_path / "qsrt-layer-003.safetensors"
    save_file(
        {
            "_qsrt_format_section": format_section,
            "_qsrt_shared_scale_section": shared_scale_section,
            "qsrt_atoms": atom_slab,
        },
        path,
        metadata={
            "format": "pt",
            "schema": FRUIT_SCHEMA,
            "version": "1",
            "encoding": "qsrt_sqg_e4m3",
            "layer": str(layer),
            "profile_id": "1",
            "experts": str(experts),
            "compressed_experts": str(experts),
            "x4t_experts": "0",
            "intermediate_channels": str(intermediate_size),
            "latent_channels": str(hidden_size),
            "atom_channels": "32",
            "atom_slots": str(atom_slots),
            "atom_bundle_bytes": str(atom_bundle_bytes),
            "atom_slot_payload_bytes": str(slot_payload_bytes),
            "atom_slot_stride_bytes": str(slot_stride_bytes),
            "alignment_bytes": str(alignment),
            "pair_count": "2",
            "rotation_multiplier": "5",
            "shared_scale_section_bytes": str(shared_scale_section_bytes),
        },
    )

    metadata = read_qsrt_atom_layer_metadata(
        path,
        layer=layer,
        expected_experts=experts,
        expected_hidden_size=hidden_size,
        expected_intermediate_size=intermediate_size,
        expected_bits=[3, 3, 3],
    )
    assert metadata.schema == FRUIT_SCHEMA
    assert metadata.atom_slots == 16
    assert metadata.atom_bundle_bytes == 37_056
    assert metadata.rotation_multiplier == 5
    assert metadata.compressed_expert_ids.tolist() == [0, 1, 2]
    assert balanced_atom_partition(16, 2, 1) == (8, 8)

    with open_qsrt_atom_extent(
        metadata,
        shard_count=2,
        shard_index=1,
        device="cpu",
    ) as (first_atom_slot, atoms):
        assert first_atom_slot == 8
        assert atoms.shape == (8, 3, 37_056)
        assert atoms.is_contiguous()
