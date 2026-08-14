# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M4 swap-engine GPU tests — T3 and T4 (03-testing-validation.md).

T3 (the make-or-break): capture a mixed K3/K4 forward in a CUDA graph,
mutate MAP CONTENTS in place through the engine's map-rebuild path, replay —
the replayed output must be bitwise-equal to a fresh-built layer holding the
new membership. If this fails, maps are baked at capture and the atomic swap
path is dead (APPLY_MODE=reload is the ceiling).

T4: swap one expert pair end-to-end through SwapEngine against real
fq-segment/1 byte payloads (toy segments written by toy_segments.py in
fq_repack's exact layout), forward a probe batch, compare BITWISE against a
fresh-built layer with that membership; parametrized over slab rows
first/last/middle. Plus rollback (inverse plan restores the original output
bitwise) and apply() wall-time measurements for 1 and 8 pairs.

Run in the gg r33 env from a neutral cwd, e.g.::

    CUDA_VISIBLE_DEVICES=<free> .../runs/gg-env/gg-run.sh \
        python /home/mbelleau/src/gg-vllm/tests/exl3_fungible/test_swap_gpu.py
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_segments as toy  # noqa: E402

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (  # noqa: E402
        swap as fq_swap,
    )
except ImportError:  # gg-env: installed vllm predates exl3_fungible
    import importlib.util
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "swap.py")
    _spec = importlib.util.spec_from_file_location("fq_swap_standalone", _p)
    fq_swap = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = fq_swap
    _spec.loader.exec_module(fq_swap)

try:
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        build_tiered_maps,
        combine_trellis_rotations,
        compile_mixed_trellis,
        make_mixed_trellis_buffers,
        run_mixed_trellis,
    )
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_trellis256_moe_weights,
    )
    HAVE_B12X = True
except ImportError:
    HAVE_B12X = False


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


pytestmark = pytest.mark.skipif(
    not (HAVE_B12X and _sm12x_available()),
    reason="requires b12x and an SM120/SM121 GPU")

# ------------------------------------------------------------------ geometry
# The proven pre-m4 occupancy-harness toy layer: E=16 = 12 K3 + 4 K4.
M, TOPK, MOE_BLOCK = 8, 4, 8
TIER0_GLOBALS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
TIER1_GLOBALS = [3, 7, 11, 15]
# Every global id 0..15 appears; both tiers in most rows.
ROUTES16 = (
    (0, 3, 14, 5),
    (1, 11, 4, 8),
    (10, 12, 3, 14),
    (5, 0, 11, 1),
    (15, 8, 12, 10),
    (4, 7, 13, 0),
    (11, 14, 5, 12),
    (9, 10, 2, 6),
)

_TIMINGS: list[str] = []


# ------------------------------------------------------------------- helpers


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _build_state(ckpt, t0_globals, t1_globals, device) -> "fq_swap.MixedLayerState":
    """Fresh-built mixed layer from checkpoint tensors (exl3.py's assembly)."""
    prepared = []
    for globals_, bits in ((t0_globals, 3), (t1_globals, 4)):
        t = {k: v.to(device) for k, v in toy.assemble_membership_tensors(
            ckpt, globals_, bits).items()}
        prepared.append(prepare_trellis256_moe_weights(
            w13=t["w13"].contiguous(),
            w2=t["w2"].contiguous(),
            hidden_size=toy.HIDDEN,
            intermediate_size=toy.INTERMEDIATE,
            num_experts=len(globals_),
            activation="silu",
            fc1_tile_n=toy.TILE_CONFIG[1],
            fc2_tile_n=toy.TILE_CONFIG[3],
            params_dtype=torch.float16,
            w13_layout="trellis3_t256_proj",
            trellis_bits=bits,
            gate_suh=t["gate_suh"].contiguous(),
            up_suh=t["up_suh"].contiguous(),
            intermediate_rotations=t["intermediate"].contiguous(),
            down_svh=t["down_svh"].contiguous(),
            tile_config=toy.TILE_CONFIG,
        ))
    g2c, desc = build_tiered_maps(t0_globals, t1_globals, device=device)
    return fq_swap.MixedLayerState(
        tier0=prepared[0], tier1=prepared[1],
        rotations=combine_trellis_rotations(*prepared),
        global_to_combined=g2c, descriptor_map=desc,
        tier0_globals=list(t0_globals), tier1_globals=list(t1_globals),
        mcg=toy.MCG)


def _compile(device, cap0, cap1):
    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(M * TOPK, MOE_BLOCK, cap0 + cap1)
    return compile_mixed_trellis(
        size_m=M,
        hidden_size=toy.HIDDEN,
        intermediate_size=toy.INTERMEDIATE,
        tier0_num_experts=cap0,
        tier1_num_experts=cap1,
        top_k=TOPK,
        max_m_blocks=(route_slots + MOE_BLOCK - 1) // MOE_BLOCK,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=toy.TILE_CONFIG,
        moe_block_size=MOE_BLOCK,
        route_ids_dtype=torch.int32,
    ), int(props.multi_processor_count)


def _forward(state, launch, sms, x, topk_weights, topk_ids, *,
             g2c=None, descriptor=None, buffers=None):
    """One forward with fresh buffers (unless capturing into given ones)."""
    device = x.device
    if buffers is None:
        buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
    out = run_mixed_trellis(
        x, state.tier0, state.tier1, topk_weights, topk_ids,
        state.global_to_combined if g2c is None else g2c,
        state.descriptor_map if descriptor is None else descriptor,
        state.rotations, launch, buffers)
    return out


def _probe(device):
    torch.manual_seed(20260810)
    x = (torch.randn((M, toy.HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor(ROUTES16, dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn((M, TOPK), dtype=torch.float32, device=device), dim=-1)
    return x, topk_weights, topk_ids


@contextmanager
def _quiesce(device):
    """Test stand-in for the engine pause: nothing in flight; sync on exit
    so the measured window includes H2D completion."""
    yield
    torch.cuda.synchronize(device)


@pytest.fixture(scope="module")
def toy_gpu(tmp_path_factory):
    root = tmp_path_factory.mktemp("fq-toy-segments-gpu")
    ckpt = toy.make_toy_checkpoint(16)
    toy.write_toy_segments(root, ckpt)
    device = _device()
    launch, sms = _compile(device, len(TIER0_GLOBALS), len(TIER1_GLOBALS))
    return root, ckpt, launch, sms


def _make_engine(root, state, **kw):
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, fq_swap.LocalSegmentSource(root),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        expected_mcg=toy.MCG, **kw)


# ------------------------------------------------------------------------ T3


def test_t3_graph_replay_sees_map_mutation(toy_gpu):
    root, ckpt, launch, sms = toy_gpu
    device = _device()
    state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
    engine = _make_engine(root, state)
    x, topk_weights, topk_ids = _probe(device)

    # Sanity + warmup with the graph's own static buffers.
    graph_buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
    eager_a = _forward(state, launch, sms, x, topk_weights, topk_ids,
                       buffers=graph_buffers)
    torch.cuda.synchronize(device)
    eager_a = eager_a.clone()
    assert not torch.isnan(eager_a).any()
    repeat = _forward(state, launch, sms, x, topk_weights, topk_ids,
                      buffers=graph_buffers)
    torch.cuda.synchronize(device)
    assert torch.equal(repeat, eager_a), "harness is not deterministic"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward(state, launch, sms, x, topk_weights, topk_ids,
                            buffers=graph_buffers)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager_a), "capture drifted from eager"
    out_a = captured.clone()

    # New membership: within-tier reorders AND cross-tier moves.
    # 0 (K3 slot 0) <-> 3 (K4 slot 0); 1 <-> 14 within K3; 7 <-> 15 within K4.
    new_t0 = [3, 14, 2, 4, 5, 6, 8, 9, 10, 12, 13, 1]
    new_t1 = [0, 15, 11, 7]
    live_g2c_ptr = state.global_to_combined.data_ptr()
    live_desc_ptr = state.descriptor_map.data_ptr()
    engine.rebuild_maps(toy.LAYER_ID, new_t0, new_t1)
    torch.cuda.synchronize(device)
    assert state.global_to_combined.data_ptr() == live_g2c_ptr
    assert state.descriptor_map.data_ptr() == live_desc_ptr

    graph.replay()
    torch.cuda.synchronize(device)
    out_b = captured.clone()

    # Reference: fresh-built maps for the new membership, eager, fresh
    # buffers, same tier slabs (a pure re-attribution of globals to slots).
    ref_g2c, ref_desc = build_tiered_maps(new_t0, new_t1, device=device)
    ref_b = _forward(state, launch, sms, x, topk_weights, topk_ids,
                     g2c=ref_g2c, descriptor=ref_desc)
    torch.cuda.synchronize(device)

    assert torch.equal(state.global_to_combined, ref_g2c)
    assert torch.equal(state.descriptor_map, ref_desc)
    assert not torch.equal(out_b, out_a), \
        "map mutation did not change routing — T3 comparison is vacuous"
    assert torch.equal(out_b, ref_b), (
        "T3 FAIL: CUDA-graph replay did not see in-place map-content "
        "mutation — maps are baked at capture; the atomic swap path is "
        "dead and APPLY_MODE=reload is the ceiling (M3).")


# ------------------------------------------------------------------------ T4


@pytest.mark.parametrize(
    "slot1_out,slot0_in",
    [(0, 0), (3, 11), (1, 5)],  # first, last, middle slab rows of each tier
    ids=["rows-first", "rows-last", "rows-middle"])
def test_t4_swap_engine_row_write_fidelity(toy_gpu, slot1_out, slot0_in):
    root, ckpt, launch, sms = toy_gpu
    device = _device()
    state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
    engine = _make_engine(root, state)
    x, topk_weights, topk_ids = _probe(device)

    out_before = _forward(state, launch, sms, x, topk_weights, topk_ids)
    torch.cuda.synchronize(device)
    out_before = out_before.clone()

    e_out = TIER1_GLOBALS[slot1_out]
    e_in = TIER0_GLOBALS[slot0_in]
    plan = fq_swap.SwapPlan([(toy.LAYER_ID, e_out, e_in)])
    report = engine.apply(plan, quiesce=_quiesce(device))
    assert report.pairs == 1 and report.mcg == toy.MCG

    # Fresh-built reference with the post-swap membership (same slot layout).
    ref = _build_state(ckpt, state.tier0_globals, state.tier1_globals, device)
    assert ref.tier1_globals[slot1_out] == e_in
    assert ref.tier0_globals[slot0_in] == e_out
    for name in ("tier0", "tier1"):
        for slab in ("w13", "w2"):
            assert torch.equal(getattr(getattr(state, name), slab),
                               getattr(getattr(ref, name), slab)), (name, slab)
    for name in ("intermediate", "gate_suh", "up_suh", "down_svh"):
        assert torch.equal(getattr(state.rotations, name),
                           getattr(ref.rotations, name)), name
    assert torch.equal(state.global_to_combined, ref.global_to_combined)
    assert torch.equal(state.descriptor_map, ref.descriptor_map)

    out_swapped = _forward(state, launch, sms, x, topk_weights, topk_ids)
    ref_out = _forward(ref, launch, sms, x, topk_weights, topk_ids)
    torch.cuda.synchronize(device)
    assert not torch.equal(out_swapped, out_before), \
        "swap did not change the probe output — comparison is vacuous"
    assert torch.equal(out_swapped, ref_out), (
        "T4 FAIL: engine-swapped layer is not bitwise-equal to the "
        "fresh-built layer with the same membership")

    # Rollback (02 §Rollback): inverse plan restores the original bitwise.
    engine.apply(plan.inverse(), quiesce=_quiesce(device))
    out_rolled = _forward(state, launch, sms, x, topk_weights, topk_ids)
    torch.cuda.synchronize(device)
    assert torch.equal(out_rolled, out_before), \
        "rollback did not restore the pre-swap output bitwise"


# ------------------------------------------------------------------- timings


def test_apply_wall_time_1_and_8_pairs(tmp_path):
    device = _device()
    e, cap0, cap1, layer = 32, 24, 8, 12
    t1g = [3, 7, 11, 15, 19, 23, 27, 31]
    t0g = [i for i in range(e) if i not in t1g]
    ckpt = toy.make_toy_checkpoint(e)
    toy.write_toy_segments(tmp_path, ckpt, layer=layer)
    state = _build_state(ckpt, t0g, t1g, device)
    engine = fq_swap.SwapEngine(
        {layer: state}, fq_swap.LocalSegmentSource(tmp_path),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        expected_mcg=toy.MCG, max_pairs=8)

    for pairs in (1, 8):
        plan = fq_swap.SwapPlan(
            [(layer, t1g[i], t0g[i]) for i in range(pairs)])
        windows, stages = [], []
        for _ in range(3):
            for p in (plan, plan.inverse()):
                staged = engine.stage(p)
                t0 = time.perf_counter()
                report = engine.apply(staged=staged, quiesce=_quiesce(device))
                assert report.window_seconds <= time.perf_counter() - t0 + 1e-3
                windows.append(report.window_seconds)
                stages.append(report.stage_seconds)
        line = (f"apply {pairs} pair(s): window best {min(windows)*1e3:.3f} ms "
                f"(median {sorted(windows)[len(windows)//2]*1e3:.3f} ms), "
                f"stage best {min(stages)*1e3:.3f} ms, "
                f"{report.bytes_h2d/1e6:.3f} MB H2D")
        _TIMINGS.append(line)
        print(f"\n[timing] {line}")
    # Membership is restored (equal applies of plan and inverse).
    assert state.tier0_globals == t0g and state.tier1_globals == t1g


if __name__ == "__main__":
    code = pytest.main([
        __file__, "-v", "-s", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ])
    for line in _TIMINGS:
        print(f"[timing] {line}")
    sys.exit(int(code))
