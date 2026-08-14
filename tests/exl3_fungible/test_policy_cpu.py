# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
try:
    from vllm.model_executor.layers.quantization.exl3_fungible import policy as fq
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "policy.py")
    _s = importlib.util.spec_from_file_location("fq_policy_standalone", _p)
    fq = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(fq)
"""T2 — policy engine property tests (CPU, stdlib + numpy only).

Covers research/fungible-quant/implementation/03-testing-validation.md T2:
determinism, budget invariant, pin/dwell/cap/hysteresis respect, inverse
property, and projection onto running cardinality. Runnable via pytest or as
a plain script (``python3 test_fq_policy.py``).
"""


import numpy as np


L, E = 6, 32  # small model: 6 MoE layers, 32 experts/layer


def make_inputs(seed=0, n_k4=12, dwell_ready=True, pin_frac=0.0):
    rng = np.random.default_rng(seed)
    stats = {
        "count": rng.integers(0, 5000, size=(L, E)).astype(np.float64),
        "mass": rng.uniform(0.0, 50.0, size=(L, E)),
    }
    eps = {3: rng.uniform(0.02, 0.30, size=(L, E)),
           4: rng.uniform(0.00, 0.02, size=(L, E))}  # eps3 > eps4: gains positive
    tier_of = np.full((L, E), fq.K3, dtype=np.int64)
    for l in range(L):
        tier_of[l, rng.permutation(E)[:n_k4]] = fq.K4
    pins = np.zeros((L, E), dtype=np.int64)
    if pin_frac:
        for l in range(L):
            k4_ids = np.nonzero(tier_of[l] == fq.K4)[0]
            k3_ids = np.nonzero(tier_of[l] == fq.K3)[0]
            n = max(1, int(pin_frac * E / 2))
            pins[l, rng.choice(k4_ids, n, replace=False)] = fq.PIN_K4
            pins[l, rng.choice(k3_ids, n, replace=False)] = fq.PIN_K3
    dwell = np.full((L, E), 10**9 if dwell_ready else 0, dtype=np.int64)
    cfg = {"n_k4": n_k4, "dwell_steps": 6000, "hysteresis": 1.25,
           "max_swaps_per_layer": 2, "max_swaps_total": 64}
    return stats, eps, tier_of, pins, dwell, cfg


def test_determinism():
    for seed in range(5):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed, pin_frac=0.1)
        a = fq.decide(stats, eps, tier, pins, dwell, cfg)
        b = fq.decide(stats, eps, tier, pins, dwell, cfg)
        c = fq.decide({k: v.copy() for k, v in stats.items()},
                      {k: v.copy() for k, v in eps.items()},
                      tier.copy(), pins.copy(), dwell.copy(), dict(cfg))
        assert a == b == c, f"non-deterministic decision at seed {seed}"
        assert all(isinstance(x, int) for sw in a for x in sw)


def test_budget_invariant_after_apply():
    for seed in range(8):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed, n_k4=10, pin_frac=0.1)
        swaps = fq.decide(stats, eps, tier, pins, dwell, cfg)
        new = fq.apply_swaps(tier, swaps)
        assert (fq.n_k4_of(new) == cfg["n_k4"]).all(), "budget broken after apply"
        assert set(np.unique(new)) <= {fq.K3, fq.K4}


def test_pin_respect():
    for seed in range(8):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed, pin_frac=0.25)
        swaps = fq.decide(stats, eps, tier, pins, dwell, cfg)
        for l, e_out, e_in in swaps:
            assert pins[l, e_out] != fq.PIN_K4, "pinned-K4 expert downgraded"
            assert pins[l, e_in] != fq.PIN_K3, "pinned-K3 expert upgraded"
        new = fq.apply_swaps(tier, swaps)
        assert (new[pins == fq.PIN_K4] == fq.K4).all()  # were K4, stayed K4
        assert (new[pins == fq.PIN_K3] == fq.K3).all()


def test_pin_forced_entry():
    # A pinned-K4 expert currently at K3 must be swapped in despite a low score.
    stats, eps, tier, pins, dwell, cfg = make_inputs(3)
    l = 0
    e_pin = int(np.nonzero(tier[l] == fq.K3)[0][0])
    pins[l, e_pin] = fq.PIN_K4
    stats["count"][l, e_pin] = 0.0  # worst possible score
    swaps = fq.decide(stats, eps, tier, pins, dwell, cfg)
    assert any(sw[0] == l and sw[2] == e_pin for sw in swaps), "pin-forced entry missing"


def test_dwell_respect():
    for seed in range(5):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed)
        rng = np.random.default_rng(seed + 100)
        young = rng.random((L, E)) < 0.5
        dwell[young] = cfg["dwell_steps"] - 1
        swaps = fq.decide(stats, eps, tier, pins, dwell, cfg)
        for l, e_out, e_in in swaps:  # no pins here -> dwell is absolute
            assert dwell[l, e_out] >= cfg["dwell_steps"], "young expert swapped out"
            assert dwell[l, e_in] >= cfg["dwell_steps"], "young expert swapped in"
    # All-young: no swaps at all.
    stats, eps, tier, pins, dwell, cfg = make_inputs(0, dwell_ready=False)
    assert fq.decide(stats, eps, tier, pins, dwell, cfg) == []


def test_cap_respect():
    stats, eps, tier, pins, dwell, cfg = make_inputs(1)
    # Adversarial scores: make every K3 expert vastly outscore every K4 one.
    eps = {3: np.where(tier == fq.K3, 10.0, 0.011), 4: np.full((L, E), 0.01)}
    for per_layer, total in [(2, 64), (5, 64), (5, 7), (1, 3)]:
        cfg2 = dict(cfg, max_swaps_per_layer=per_layer, max_swaps_total=total)
        swaps = fq.decide(stats, eps, tier, pins, dwell, cfg2)
        assert len(swaps) == min(L * per_layer, total), "caps not saturated"
        per = {}
        for l, _, _ in swaps:
            per[l] = per.get(l, 0) + 1
        assert max(per.values()) <= per_layer, "per-layer cap exceeded"


def test_hysteresis_threshold_and_monotonicity():
    # Entering expert outscores leaving by exactly 1.2x -> blocked at h=1.25,
    # admitted at h=1.1.
    stats = {"count": np.ones((L, E)), "mass": np.ones((L, E))}
    eps = {3: np.full((L, E), 0.01 + 1.0), 4: np.full((L, E), 0.01)}
    tier = np.full((L, E), fq.K3, dtype=np.int64)
    tier[:, :12] = fq.K4
    eps[3][:, 12] = 0.01 + 1.2  # candidate gain 1.2 vs incumbents' 1.0
    dwell = np.full((L, E), 10**9)
    pins = np.zeros((L, E), dtype=np.int64)
    base = {"n_k4": 12, "dwell_steps": 0, "max_swaps_per_layer": 8,
            "max_swaps_total": 640}
    assert fq.decide(stats, eps, tier, pins, dwell, dict(base, hysteresis=1.25)) == []
    admitted = fq.decide(stats, eps, tier, pins, dwell, dict(base, hysteresis=1.1))
    assert len(admitted) == L and all(sw[2] == 12 for sw in admitted)
    # Monotonicity on random positive-score inputs: raising h only removes swaps.
    for seed in range(5):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed)
        prev = None
        for h in (1.0, 1.25, 1.5, 2.0, 4.0):
            cur = set(fq.decide(stats, eps, tier, pins, dwell, dict(cfg, hysteresis=h)))
            if prev is not None:
                assert cur <= prev, f"h={h} produced swaps absent at lower h"
            prev = cur


def test_inverse_restores_membership():
    for seed in range(8):
        stats, eps, tier, pins, dwell, cfg = make_inputs(seed, pin_frac=0.1)
        swaps = fq.decide(stats, eps, tier, pins, dwell, cfg)
        if seed == 0:
            assert swaps, "want at least one non-empty swap list in this test"
        restored = fq.apply_swaps(fq.apply_swaps(tier, swaps), fq.inverse(swaps))
        assert (restored == tier).all(), "inverse did not restore membership"


def test_projection_onto_running_cardinality():
    rng = np.random.default_rng(42)
    running = np.array([12, 12, 12, 8, 16, 12])
    for policy_n in (6, 12, 20, 0, E):  # fewer, equal, more, extremes
        bits = np.full((L, E), fq.K3, dtype=np.int64)
        for l in range(L):
            bits[l, rng.permutation(E)[:policy_n]] = fq.K4
        order = rng.random((L, E))
        proj = fq.project(bits, running, order)
        assert (fq.n_k4_of(proj) == running).all(), "projection missed budget"
        for l in range(L):
            polk4 = set(np.nonzero(bits[l] == fq.K4)[0])
            projk4 = set(np.nonzero(proj[l] == fq.K4)[0])
            if policy_n >= running[l]:  # shrink: keep top-N of the policy's own K4s
                assert projk4 <= polk4
                keep = sorted(polk4, key=lambda e: (-order[l, e], e))[: running[l]]
                assert projk4 == set(keep)
            else:  # grow: keep all policy K4s, fill by the policy's ordering
                assert polk4 <= projk4
                fill = sorted(set(range(E)) - polk4,
                              key=lambda e: (-order[l, e], e))[: running[l] - policy_n]
                assert projk4 == polk4 | set(fill)
        # Projected membership is a valid decide() input (D1 restored).
        stats, eps, _, pins, dwell, cfg = make_inputs(7)
        fq.decide(stats, eps, proj, pins, dwell, dict(cfg, n_k4=running))
    # Equal-cardinality projection with the identity ordering is a no-op.
    bits = np.full((L, E), fq.K3, dtype=np.int64)
    bits[:, ::3] = fq.K4
    n = int((bits[0] == fq.K4).sum())
    assert (fq.project(bits, n) == bits).all()


def test_decide_rejects_budget_mismatch():
    stats, eps, tier, pins, dwell, cfg = make_inputs(0, n_k4=12)
    try:
        fq.decide(stats, eps, tier, pins, dwell, dict(cfg, n_k4=13))
    except ValueError:
        pass
    else:
        raise AssertionError("decide accepted membership violating the budget")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    sys.exit(1 if failures else 0)
