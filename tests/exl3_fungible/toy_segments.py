# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Toy fq-segment fixtures shared by the M4 swap tests.

Generates a deterministic per-expert EXL3 "checkpoint" (native trellis tiles
+ suh/svh rotations per projection, per bitrate) for a small mixed layer and
writes it out in the exact fq-segment/1 layout fq_repack produces
(``layer-LLL.kK.safetensors`` + ``index-kK.json``): per-expert contiguous
body, (expert, proj, rank, comp)-sorted tensors, canonical names. CPU-only,
plain file IO — usable from both the CPU contract tests and the T3/T4 GPU
harness.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

# Mirrors the proven pre-m4 occupancy harness geometry.
HIDDEN = 128
INTERMEDIATE = 128
TILE_CONFIG = (128, 128, 128, 128)
LAYER_ID = 3
MCG = -877912083  # the layer-wide MCG codebook word (fruit-segments value)

PROJ_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
COMP_ORDER = {"trellis": 0, "suh": 1, "svh": 2, "mcg": 3}


def _scales(shape, generator):
    return (0.875 + 0.25 * torch.rand(shape, generator=generator)).to(
        torch.float16)


def make_toy_checkpoint(num_experts: int, ks=(3, 4), *, hidden=HIDDEN,
                        intermediate=INTERMEDIATE, seed=20260810) -> dict:
    """Deterministic per-(expert, k) payload tensors, CPU.

    Returns ``{(expert, k): {comp_key: tensor}}`` with native trellis tiles
    ``gate``/``up`` ``[H/16, I/16, 16k]`` i16, ``down`` ``[I/16, H/16, 16k]``
    i16, and fp16 rotations: ``gate_suh``/``up_suh`` ``[H]``, ``down_svh``
    ``[H]``, ``gate_svh``/``up_svh``/``down_suh`` ``[I]``. Every (expert, k)
    pair gets distinct values so cross-expert or cross-bitrate mixups are
    detectable bitwise.
    """
    h16, i16 = hidden // 16, intermediate // 16
    out = {}
    for e in range(num_experts):
        for k in ks:
            g = torch.Generator().manual_seed(seed * 1000003 + e * 101 + k)
            words = 16 * k
            out[(e, k)] = {
                "gate": torch.randint(-2**15, 2**15, (h16, i16, words),
                                      dtype=torch.int16, generator=g),
                "up": torch.randint(-2**15, 2**15, (h16, i16, words),
                                    dtype=torch.int16, generator=g),
                "down": torch.randint(-2**15, 2**15, (i16, h16, words),
                                      dtype=torch.int16, generator=g),
                "gate_suh": _scales((hidden,), g),
                "up_suh": _scales((hidden,), g),
                "down_svh": _scales((hidden,), g),
                "gate_svh": _scales((intermediate,), g),
                "up_svh": _scales((intermediate,), g),
                "down_suh": _scales((intermediate,), g),
            }
    return out


def assemble_membership_tensors(checkpoint: dict, tier_globals, k: int) -> dict:
    """Stack one tier's members (slot order) into prepare-shaped tensors.

    Mirrors exl3.py ``_prepare_mixed_rank_sliced_weights``: w13 is
    ``[2(gate,up), E_t, H/16, I/16, 16k]`` i16, w2 ``[E_t, I/16, H/16, 16k]``
    i16, rotations fp16 with ``intermediate = [gate_svh|up_svh|down_suh]``.
    CPU tensors; callers move to device as needed.
    """
    members = [checkpoint[(int(g), k)] for g in tier_globals]
    return {
        "w13": torch.stack((
            torch.stack([m["gate"] for m in members]),
            torch.stack([m["up"] for m in members]),
        )).contiguous(),
        "w2": torch.stack([m["down"] for m in members]).contiguous(),
        "gate_suh": torch.stack([m["gate_suh"] for m in members]).contiguous(),
        "up_suh": torch.stack([m["up_suh"] for m in members]).contiguous(),
        "down_svh": torch.stack([m["down_svh"] for m in members]).contiguous(),
        "intermediate": torch.cat((
            torch.stack([m["gate_svh"] for m in members]),
            torch.stack([m["up_svh"] for m in members]),
            torch.stack([m["down_suh"] for m in members]),
        ), dim=1).contiguous(),
    }


def _expert_tensors(layer: int, expert: int, payload: dict, rank: int):
    """(name, tensor-or-mcg) in fq_repack's (proj, rank, comp) sort order."""
    base = f"model.layers.{layer}.mlp.experts.{expert}"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = proj.split("_")[0]
        yield f"{base}.{proj}.rank{rank}.trellis", payload[p]
        yield f"{base}.{proj}.rank{rank}.suh", payload[f"{p}_suh"]
        yield f"{base}.{proj}.rank{rank}.svh", payload[f"{p}_svh"]
        yield f"{base}.{proj}.rank{rank}.mcg", None  # scalar written as MCG


def write_toy_segments(root: str | Path, checkpoint: dict, *, layer=LAYER_ID,
                       ks=(3, 4), rank=0, mcg=MCG) -> Path:
    """Write ``layer-LLL.kK.safetensors`` + ``index-kK.json`` under root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dtypes = {torch.int16: "I16", torch.float16: "F16"}
    experts = sorted({e for e, _ in checkpoint})
    for k in ks:
        entries = []  # (name, dtype_tag, shape, payload_bytes)
        per_expert = {}
        off = 0
        for e in experts:
            start = off
            for name, t in _expert_tensors(layer, e, checkpoint[(e, k)], rank):
                if t is None:
                    raw = struct.pack("<i", mcg)
                    tag, shape = "I32", []
                else:
                    raw = t.numpy().tobytes()
                    tag, shape = dtypes[t.dtype], list(t.shape)
                entries.append((name, tag, shape, raw))
                off += len(raw)
            per_expert[str(e)] = [start, off]

        header = {"__metadata__": {
            "fq_schema": "fq-segment/1",
            "predicate": "repack-of",
            "k": str(k),
            "layer": str(layer),
            "layout": "rank_sliced_tp4",
            "num_experts": str(len(experts)),
            "source_file": "toy",
        }}
        pos = 0
        for name, tag, shape, raw in entries:
            header[name] = {"dtype": tag, "shape": shape,
                            "data_offsets": [pos, pos + len(raw)]}
            pos += len(raw)
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)

        seg_name = f"layer-{layer:03d}.k{k}.safetensors"
        with open(root / seg_name, "wb") as f:
            f.write(struct.pack("<Q", len(hj)) + hj)
            for _, _, _, raw in entries:
                f.write(raw)
        index = {str(layer): {
            "file": seg_name,
            "body_offset": 8 + len(hj),
            "size": (root / seg_name).stat().st_size,
            "experts": per_expert,
        }}
        (root / f"index-k{k}.json").write_text(json.dumps(index, indent=1))
    return root
