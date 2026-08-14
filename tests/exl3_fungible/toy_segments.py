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

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_PKG_DIR = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")


def load_tree_module(name: str):
    """Load one ``exl3_fungible`` module from THIS working tree, by path.

    The gg rootfs carries a copy of the package inside its site-packages that
    is periodically synced from the tree, so a plain
    ``from vllm... import swap`` can silently resolve to whatever was last
    copied there. Tests whose verdict is about tree code load the tree file.
    Cached under ``fq_<name>_tree`` so every test module that asks gets the
    SAME module object (class identity matters: ``SwapPlan.__eq__`` and the
    dataclasses are per-module).
    """
    alias = f"fq_{name}_tree"
    mod = sys.modules.get(alias)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            alias, _PKG_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return mod

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


def build_maps_reference(tier0_ids, tier1_ids, *, device):
    """Pure-torch replica of b12x ``build_tiered_maps`` (T3 proves parity)."""
    t0n, t1n = len(tier0_ids), len(tier1_ids)
    g2c = torch.full((t0n + t1n,), -1, dtype=torch.int32)
    for local, g in enumerate(tier0_ids):
        g2c[int(g)] = local
    for local, g in enumerate(tier1_ids):
        g2c[int(g)] = t0n + local
    desc = torch.tensor(
        [*range(t0n), *((1 << 8) | i for i in range(t1n))], dtype=torch.int32)
    return g2c.to(device), desc.to(device)


def cpu_layer_state(fq_swap, checkpoint: dict, t0_globals, t1_globals):
    """Hand-assembled ``MixedLayerState`` on CPU (fresh-build reference).

    ``fq_swap`` is passed in so the caller controls WHICH swap module the
    state belongs to (see :func:`load_tree_module`)."""
    def tier(globals_, k):
        t = assemble_membership_tensors(checkpoint, globals_, k)
        return SimpleNamespace(
            num_experts=len(globals_),
            w13=t["w13"].view(torch.int32).reshape(-1),
            w2=t["w2"].view(torch.int32).reshape(-1),
        ), t

    tier0, r0 = tier(t0_globals, 3)
    tier1, r1 = tier(t1_globals, 4)
    rotations = SimpleNamespace(
        intermediate=torch.cat((r0["intermediate"], r1["intermediate"])),
        gate_suh=torch.cat((r0["gate_suh"], r1["gate_suh"])),
        up_suh=torch.cat((r0["up_suh"], r1["up_suh"])),
        down_svh=torch.cat((r0["down_svh"], r1["down_svh"])),
    )
    g2c, desc = build_maps_reference(t0_globals, t1_globals,
                                     device=torch.device("cpu"))
    return fq_swap.MixedLayerState(
        tier0=tier0, tier1=tier1, rotations=rotations,
        global_to_combined=g2c, descriptor_map=desc,
        tier0_globals=list(t0_globals), tier1_globals=list(t1_globals))


def state_fingerprint(state) -> str:
    """sha256 over every byte a swap can touch — slabs, combined rotation /
    suh / svh tables, both maps. Two states with the same fingerprint are
    indistinguishable to the kernel."""
    import hashlib

    h = hashlib.sha256()
    for tier in (state.tier0, state.tier1):
        for slab in ("w13", "w2"):
            h.update(getattr(tier, slab).contiguous().view(torch.uint8)
                     .cpu().numpy().tobytes())
    for name in ("intermediate", "gate_suh", "up_suh", "down_svh"):
        h.update(getattr(state.rotations, name).contiguous().view(torch.uint8)
                 .cpu().numpy().tobytes())
    for m in (state.global_to_combined, state.descriptor_map):
        h.update(m.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def assert_states_equal(a, b) -> None:
    assert a.tier0_globals == b.tier0_globals
    assert a.tier1_globals == b.tier1_globals
    for name in ("tier0", "tier1"):
        for slab in ("w13", "w2"):
            assert torch.equal(getattr(getattr(a, name), slab),
                               getattr(getattr(b, name), slab)), (name, slab)
    for name in ("intermediate", "gate_suh", "up_suh", "down_svh"):
        assert torch.equal(getattr(a.rotations, name),
                           getattr(b.rotations, name)), name
    assert torch.equal(a.global_to_combined, b.global_to_combined)
    assert torch.equal(a.descriptor_map, b.descriptor_map)


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
