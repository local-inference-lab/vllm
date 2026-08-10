# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the M4 swap engine: SwapPlan algebra, LocalSegmentSource
byte fidelity, and full engine apply()/rollback byte fidelity against
hand-assembled layer state (no GPU, no b12x — build_tiered_maps semantics
injected through the ``build_maps_fn`` seam and cross-checked on GPU by T3).
"""
import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_segments as toy  # noqa: E402

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (  # noqa: E402
        policy as fq_policy,
        store as fq_store,
        swap as fq_swap,
    )
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    from pathlib import Path as _P

    _dir = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"fq_{name}_standalone", _dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses resolve __module__
        spec.loader.exec_module(mod)
        return mod

    fq_policy = _load("policy")
    fq_store = _load("store")
    fq_swap = _load("swap")

E = 8  # toy global experts
T0_GLOBALS = [0, 2, 4, 5, 6]  # K3 tier, slot order
T1_GLOBALS = [3, 1, 7]  # K4 tier, slot order


def build_maps_reference(tier0_ids, tier1_ids, *, device):
    """Pure-torch replica of b12x build_tiered_maps (T3 proves parity)."""
    t0n, t1n = len(tier0_ids), len(tier1_ids)
    total = t0n + t1n
    g2c = torch.full((total,), -1, dtype=torch.int32)
    for local, g in enumerate(tier0_ids):
        g2c[int(g)] = local
    for local, g in enumerate(tier1_ids):
        g2c[int(g)] = t0n + local
    desc = torch.tensor(
        [*range(t0n), *((1 << 8) | i for i in range(t1n))], dtype=torch.int32)
    return g2c.to(device), desc.to(device)


# ------------------------------------------------------------------ SwapPlan


def test_plan_from_memberships_deterministic_and_ordered():
    old = np.array([[4, 4, 3, 3], [3, 4, 4, 3]])
    new = np.array([[3, 4, 3, 4], [4, 4, 3, 3]])
    plan = fq_swap.SwapPlan.from_memberships(old, new)
    assert plan.swaps == ((0, 0, 3), (1, 2, 0))
    assert plan == fq_swap.SwapPlan.from_memberships(old, new)
    assert plan.layers() == (0, 1)


def test_plan_inverse_restores_membership():
    rng = np.random.default_rng(7)
    L, En, n4 = 5, 16, 6
    old = np.full((L, En), 3)
    new = np.full((L, En), 3)
    for l in range(L):
        old[l, rng.choice(En, n4, replace=False)] = 4
        new[l, rng.choice(En, n4, replace=False)] = 4
    plan = fq_swap.SwapPlan.from_memberships(old, new)
    mid = fq_policy.apply_swaps(old, list(plan))
    assert (mid == new).all()
    back = fq_policy.apply_swaps(mid, list(plan.inverse()))
    assert (back == old).all()
    # double inverse is identity on the plan itself
    assert plan.inverse().inverse() == plan


def test_plan_rejects_cardinality_change_and_duplicates():
    with pytest.raises(ValueError, match="cardinality"):
        fq_swap.SwapPlan.from_memberships([[4, 3]], [[3, 3]])
    with pytest.raises(ValueError, match="twice"):
        fq_swap.SwapPlan([(0, 1, 2), (0, 1, 3)])


def test_plan_from_policies():
    def doc(bits):
        return {"schema": "fq-policy/2", "bits_per_expert": bits}

    old = doc({"3": [4, 3, 3, 4], "12": [3, 4, 3, 3]})
    new = doc({"3": [3, 4, 3, 4], "12": [3, 3, 4, 3]})
    plan = fq_swap.SwapPlan.from_policies(old, new)
    assert plan.swaps == ((3, 0, 1), (12, 1, 2))
    with pytest.raises(ValueError, match="different layers"):
        fq_swap.SwapPlan.from_policies(old, doc({"3": [3, 4, 3, 4]}))


# -------------------------------------------------------- LocalSegmentSource


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fq-toy-segments")
    ckpt = toy.make_toy_checkpoint(E)
    toy.write_toy_segments(root, ckpt)
    return root, ckpt


def test_segment_source_reads_expert_bitwise(toy_root):
    root, ckpt = toy_root
    src = fq_swap.LocalSegmentSource(root)
    for e, k in ((0, 3), (5, 3), (1, 4), (7, 4)):
        stage = fq_swap.ExpertStage(k, toy.HIDDEN, toy.INTERMEDIATE,
                                    pin_memory=False)
        src.read_expert(layer=toy.LAYER_ID, k=k, expert=e, rank=0, dest=stage)
        ref = ckpt[(e, k)]
        for proj in ("gate", "up", "down"):
            assert torch.equal(stage.trellis[proj], ref[proj])
            assert stage.mcg[proj] == toy.MCG
        assert torch.equal(stage.suh["gate"], ref["gate_suh"])
        assert torch.equal(stage.suh["up"], ref["up_suh"])
        assert torch.equal(stage.svh["down"], ref["down_svh"])
        expected_rot = torch.cat(
            (ref["gate_svh"], ref["up_svh"], ref["down_suh"]))
        assert torch.equal(stage.inter_rot, expected_rot)
    src.close()


def test_segment_source_fail_closed(toy_root):
    root, _ = toy_root
    src = fq_swap.LocalSegmentSource(root)
    stage = fq_swap.ExpertStage(3, toy.HIDDEN, toy.INTERMEDIATE,
                                pin_memory=False)
    with pytest.raises(KeyError, match="expert 99"):
        src.read_expert(layer=toy.LAYER_ID, k=3, expert=99, rank=0,
                        dest=stage)
    with pytest.raises(KeyError, match="layer 42"):
        src.read_expert(layer=42, k=3, expert=0, rank=0, dest=stage)
    with pytest.raises(ValueError, match="stage is K3"):
        src.read_expert(layer=toy.LAYER_ID, k=4, expert=0, rank=0,
                        dest=stage)
    # A tensor range escaping its indexed expert range is refused.
    idx_path = root / "index-k3.json"
    original = idx_path.read_text()
    index = json.loads(original)
    index[str(toy.LAYER_ID)]["experts"]["0"][1] -= 4
    idx_path.write_text(json.dumps(index))
    try:
        with pytest.raises(ValueError, match="escapes"):
            fq_swap.LocalSegmentSource(root).read_expert(
                layer=toy.LAYER_ID, k=3, expert=0, rank=0, dest=stage)
    finally:
        idx_path.write_text(original)
    src.close()


# ------------------------------------------------- engine apply, CPU tensors


def _cpu_state(ckpt, t0_globals, t1_globals):
    """Hand-assembled layer state on CPU (fresh-build reference shape)."""
    def tier(globals_, k):
        t = toy.assemble_membership_tensors(ckpt, globals_, k)
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


def _make_engine(root, state, **kw):
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, fq_swap.LocalSegmentSource(root),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        pin_memory=False, build_maps_fn=build_maps_reference, **kw)


def _assert_states_equal(a, b):
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


def test_apply_matches_fresh_build_and_rolls_back(toy_root):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    pristine = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = _make_engine(root, state)

    e_out, e_in = 1, 4  # K4 slot 1 out, K3 slot 2 in
    plan = fq_swap.SwapPlan([(toy.LAYER_ID, e_out, e_in)])
    steps = []
    report = engine.apply(plan, quiesce=nullcontext(),
                          step_hook=steps.append)
    assert steps == list(fq_swap.COMMIT_STEPS)
    assert report.pairs == 1 and report.generation == 1
    assert report.mcg == toy.MCG
    assert report.bytes_h2d > 0

    # e_in inherited e_out's tier1 slot; e_out inherited e_in's tier0 slot.
    expected = _cpu_state(ckpt, [0, 2, 1, 5, 6], [3, 4, 7])
    _assert_states_equal(state, expected)

    # Rollback: apply the inverse; every byte must return (02 §Rollback).
    engine.apply(plan.inverse(), quiesce=nullcontext())
    assert engine.generation == 2
    _assert_states_equal(state, pristine)


def test_apply_multi_pair_same_layer(toy_root):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = _make_engine(root, state)
    plan = fq_swap.SwapPlan(
        [(toy.LAYER_ID, 3, 0), (toy.LAYER_ID, 7, 6)])
    engine.apply(plan, quiesce=nullcontext())
    expected = _cpu_state(ckpt, [3, 2, 4, 5, 7], [0, 1, 6])
    _assert_states_equal(state, expected)
    engine.apply(plan.inverse(), quiesce=nullcontext())
    _assert_states_equal(state, _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS))


def test_apply_validates_residency_and_caps(toy_root):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = _make_engine(root, state, max_pairs=1)
    with pytest.raises(ValueError, match="resident"):
        engine.stage(fq_swap.SwapPlan([(toy.LAYER_ID, 0, 1)]))  # both wrong
    with pytest.raises(ValueError, match="staging holds 1"):
        engine.stage(fq_swap.SwapPlan(
            [(toy.LAYER_ID, 3, 0), (toy.LAYER_ID, 7, 6)]))
    with pytest.raises(KeyError, match="not registered"):
        engine.stage(fq_swap.SwapPlan([(99, 3, 0)]))


def test_engine_refuses_bad_geometry(toy_root):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    state.rotations.gate_suh = state.rotations.gate_suh[:1]  # broadcast-like
    with pytest.raises(ValueError, match="broadcast"):
        _make_engine(root, state)
    state2 = _cpu_state(ckpt, T0_GLOBALS, [3, 3])  # not a partition
    with pytest.raises(ValueError, match="partition"):
        _make_engine(root, state2)


def test_map_rebuild_fail_closed_on_holes(toy_root):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)

    def bad_maps(t0, t1, *, device):
        g2c, desc = build_maps_reference(t0, t1, device=device)
        g2c[0] = -1  # a hole the swap path must never introduce
        return g2c, desc

    engine = fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, fq_swap.LocalSegmentSource(root),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        pin_memory=False, build_maps_fn=bad_maps)
    with pytest.raises(ValueError, match="permutation"):
        engine.stage(fq_swap.SwapPlan([(toy.LAYER_ID, 3, 0)]))


def test_mcg_mismatch_refused(tmp_path):
    ckpt = toy.make_toy_checkpoint(E)
    toy.write_toy_segments(tmp_path, ckpt)
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = _make_engine(tmp_path, state, expected_mcg=12345)
    with pytest.raises(ValueError, match="mcg"):
        engine.stage(fq_swap.SwapPlan([(toy.LAYER_ID, 3, 0)]))


def test_policy_persist_on_commit(toy_root, tmp_path):
    root, ckpt = toy_root
    state = _cpu_state(ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = _make_engine(root, state)
    ps = fq_store.PolicyStore(tmp_path, "toy-manifest")
    doc = {
        "schema": "fq-policy/2",
        "manifest": "toy-manifest",
        "budget": {"n_k4_per_layer": {str(toy.LAYER_ID): 3}},
        "bits_per_expert": {str(toy.LAYER_ID): [3, 3, 3, 4, 4, 3, 3, 4]},
    }
    memo_calls = []
    report = engine.apply(
        fq_swap.SwapPlan([(toy.LAYER_ID, 1, 4)]), quiesce=nullcontext(),
        memo_hook=lambda: memo_calls.append(True),
        policy_store=ps, policy_doc=doc, policy_num_experts=E)
    assert memo_calls == [True]
    assert report.committed_policy_hash == fq_store.policy_hash(doc)
    assert ps.load_current(num_experts=E) == doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
