# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the fungible-quant MEMORY BUDGET (bytes <-> experts/layer).

Covers what the operator asked for: a ceiling expressible either as a
byte count / fraction of device memory or as a per-layer cardinality,
converted between the two honestly, enforced when proposing swaps, and
reported as headroom in the log table and on the Prometheus gauges.

The per-expert byte model is the load-bearing part: it is validated here
against the REAL tensor geometry of two assembled checkpoints, so a
regression in the derivation shows up as a failing test rather than as a
serve that quietly overruns its budget.
"""
import json
import logging

import numpy as np
import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        loop as FL,
        occupancy_table as OT,
        policy as P,
        store as S,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
except ImportError:  # standalone: load by path with a stub package
    import importlib.util
    import sys
    import types
    from pathlib import Path as _P

    _dir = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")
    _pkg_name = "vllm.model_executor.layers.quantization.exl3_fungible"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _pkg = types.ModuleType(_pkg_name)
    sys.modules[_pkg_name] = _pkg
    for _sub in ("policy", "stats", "store", "decision_log",
                 "occupancy_table", "loop"):
        _mod = _load(f"{_pkg_name}.{_sub}", _dir / f"{_sub}.py")
        setattr(_pkg, _sub, _mod)
    P, S, FL, OT = (_pkg.policy, _pkg.store, _pkg.loop, _pkg.occupancy_table)
    FqStatsCollector = _pkg.stats.FqStatsCollector

prometheus_client = pytest.importorskip("prometheus_client")


@pytest.fixture
def fq_caplog(caplog):
    """caplog, but able to see FQ records.

    vllm configures its ``vllm`` logger with ``propagate: False``, so
    records from ``vllm.model_executor...`` never reach the root handler
    caplog installs. Re-enable propagation for the duration of the test.
    """
    vllm_logger = logging.getLogger("vllm")
    before = vllm_logger.propagate
    vllm_logger.propagate = True
    try:
        yield caplog
    finally:
        vllm_logger.propagate = before


# ==================================================================== parsing

GIB = 1 << 30


@pytest.mark.parametrize("spec,want", [
    ("80000000000", 80_000_000_000),
    ("78g", 78 * GIB),
    ("78G", 78 * GIB),
    ("78gb", 78 * GIB),
    ("78GiB", 78 * GIB),
    ("512mb", 512 * (1 << 20)),
    ("2t", 2 * (1 << 40)),
    ("4096k", 4096 * 1024),
    ("1b", 1),
    ("  78 g  ", 78 * GIB),
    ("80_000_000_000", 80_000_000_000),
    (80_000_000_000, 80_000_000_000),
])
def test_parse_absolute_sizes(spec, want):
    assert P.parse_memory_budget(spec) == want


@pytest.mark.parametrize("spec,want", [
    ("0.80", int(0.80 * 100 * GIB)),
    ("0.9", int(0.9 * 100 * GIB)),
    ("1", 100 * GIB),          # bare 1 == the whole device, like gpu-mem-util
    ("1.0", 100 * GIB),
    ("80%", int(0.80 * 100 * GIB)),
    ("100%", 100 * GIB),
    (0.5, 50 * GIB),
])
def test_parse_fraction_of_device_memory(spec, want):
    assert P.parse_memory_budget(spec, device_total_bytes=100 * GIB) == want


def test_bare_one_is_a_fraction_but_one_byte_is_spellable():
    """The gpu-memory-utilization ergonomic has exactly one sharp edge."""
    assert P.parse_memory_budget("1", device_total_bytes=100) == 100
    assert P.parse_memory_budget("1b", device_total_bytes=100) == 1


@pytest.mark.parametrize("spec", [None, "", "none", "off", "unlimited",
                                  "UNBOUNDED"])
def test_budget_can_be_switched_off(spec):
    assert P.parse_memory_budget(spec) is None


@pytest.mark.parametrize("spec", [
    "abc", "12x", "-5", "0", "0g", "1.5.2", "%", "g", "78 gigs", "0%",
    "120%", "1.5%%", True,
])
def test_garbage_is_refused_not_guessed(spec):
    with pytest.raises(ValueError):
        P.parse_memory_budget(spec, device_total_bytes=100 * GIB)


def test_fraction_without_a_device_names_the_problem():
    with pytest.raises(ValueError, match="fraction of device memory"):
        P.parse_memory_budget("0.80")


# =============================================== per-expert byte model

def test_measured_glm52_points_fit_the_affine_trellis_model():
    eb = P.ExpertBytes.from_measurements(
        P.MEASURED_GLM52_TP4_PER_RANK, provenance="test")
    # The two measurements come back exactly.
    assert eb.bytes_for(3) == 3_542_028
    assert eb.bytes_for(4) == 4_721_676
    # ...and the promotion cost the operator measured.
    assert eb.promotion_cost(3, 4) == 1_179_648
    assert eb.promotion_cost(4, 3) == -1_179_648
    # K2/K5 are EVALUATED from the same line, not guessed.
    assert eb.bytes_for(2) == 2 * 1_179_648 + 3_084 == 2_362_380
    assert eb.bytes_for(5) == 5 * 1_179_648 + 3_084 == 5_901_324


def test_all_rank_figures_are_four_times_the_per_rank_ones():
    per = P.ExpertBytes.from_measurements(
        P.MEASURED_GLM52_TP4_PER_RANK, provenance="test")
    allr = P.ExpertBytes.from_measurements(
        P.MEASURED_GLM52_TP4_ALL_RANKS, provenance="test")
    for k in (2, 3, 4, 5):
        assert allr.bytes_for(k) == 4 * per.bytes_for(k)


def fruit_expert_tensor_table(k):
    """One expert, one rank, of the real GLM-5.2-arch proxy checkpoint.

    Transcribed from the safetensors headers of ``fruit-k3``/``fruit-k4``
    (H=1024, per-rank I=128): three ``[H/16, I/16, 16*K]`` int16 trellis
    slabs plus the K-independent rotations and mcg scalars.
    """
    words = 16 * k
    h16, i16 = 64, 8
    tre = h16 * i16 * words * 2
    out = []
    for proj, shape in (("gate_proj", (h16, i16, words)),
                        ("up_proj", (h16, i16, words)),
                        ("down_proj", (i16, h16, words))):
        base = f"model.layers.3.mlp.experts.0.{proj}.rank0"
        out.append((f"{base}.trellis", tre, shape))
        out.append((f"{base}.mcg", 4, ()))
    out += [
        ("model.layers.3.mlp.experts.0.gate_proj.rank0.suh", 2048, (1024,)),
        ("model.layers.3.mlp.experts.0.gate_proj.rank0.svh", 256, (128,)),
        ("model.layers.3.mlp.experts.0.up_proj.rank0.suh", 2048, (1024,)),
        ("model.layers.3.mlp.experts.0.up_proj.rank0.svh", 256, (128,)),
        ("model.layers.3.mlp.experts.0.down_proj.rank0.suh", 256, (128,)),
        ("model.layers.3.mlp.experts.0.down_proj.rank0.svh", 2048, (1024,)),
    ]
    return out


def test_model_derived_from_a_real_k3_header_predicts_the_real_k4_bytes():
    """Cross-checkpoint validation of the whole derivation.

    fruit-k3 and fruit-k4 are two independently encoded checkpoints. The
    model built from k3's shapes alone must reproduce k4's measured
    814,128 bytes/expert (across its 4 ranks) exactly — if it does not,
    the "last dim = 16*K" assumption is wrong and every headroom number
    downstream is fiction.
    """
    eb = P.ExpertBytes.from_tensor_table(
        fruit_expert_tensor_table(3), provenance="fruit-k3")
    assert eb.measured_ks == (3,)
    assert eb.bytes_for(3) * 4 == 617_520      # measured, fruit-k3
    assert eb.bytes_for(4) * 4 == 814_128      # measured, fruit-k4
    # And deriving from k4's header gives the identical model.
    eb4 = P.ExpertBytes.from_tensor_table(
        fruit_expert_tensor_table(4), provenance="fruit-k4")
    assert (eb4.trellis_bytes_per_k, eb4.fixed_bytes) == (
        eb.trellis_bytes_per_k, eb.fixed_bytes)


def test_tensor_table_reads_k_off_the_trellis_last_dim():
    for k in (2, 3, 4, 5):
        eb = P.ExpertBytes.from_tensor_table(
            fruit_expert_tensor_table(k), provenance="t")
        assert eb.measured_ks == (k,)
        assert f"K{k}" in eb.provenance


def test_tensor_table_refuses_a_non_trellis_last_dim():
    bad = [("x.trellis", 100, (4, 4, 50))]  # 50 is not 16*K
    with pytest.raises(ValueError, match="16"):
        P.ExpertBytes.from_tensor_table(bad, provenance="t")


def test_tensor_table_refuses_mixed_k():
    mixed = fruit_expert_tensor_table(3) + [
        ("model.layers.3.mlp.experts.0.gate_proj.rank1.trellis", 1,
         (64, 8, 64))]
    with pytest.raises(ValueError, match="exactly one K"):
        P.ExpertBytes.from_tensor_table(mixed, provenance="t")


def test_measurements_off_the_line_are_refused():
    with pytest.raises(ValueError, match="off the line"):
        P.ExpertBytes.from_measurements({3: 300, 4: 401, 5: 500},
                                        provenance="t")


def test_a_fractional_bytes_per_k_slope_is_refused():
    with pytest.raises(ValueError, match="fractional bytes-per-K slope"):
        P.ExpertBytes.from_measurements({3: 300, 5: 501}, provenance="t")


def test_one_measurement_is_not_enough():
    with pytest.raises(ValueError, match="at least two"):
        P.ExpertBytes.from_measurements({3: 300}, provenance="t")


def test_reference_model_is_flagged_as_not_this_checkpoint():
    eb = P.reference_expert_bytes()
    assert eb.is_reference is True
    assert "REFERENCE" in eb.provenance and "NOT derived" in eb.provenance
    assert "source:" in eb.describe()


# =================================================== bytes <-> cardinality

def simple_budget(limit=None, *, low=3, high=4):
    """slope 100 B/K, no fixed overhead: K3=300, K4=400, promotion=100."""
    eb = P.ExpertBytes.from_measurements({3: 300, 4: 400}, provenance="test")
    return P.MemoryBudget(eb, limit_bytes=limit, low_k=low, high_k=high)


def test_used_bytes_comes_from_actual_occupancy_including_off_ladder_tiers():
    b = simple_budget()
    tier = np.array([[2, 3, 4, 5], [3, 3, 4, 4]])
    # 1xK2 + 3xK3 + 3xK4 + 1xK5 = 200 + 900 + 1200 + 500
    assert b.counts_of(tier) == {2: 1, 3: 3, 4: 3, 5: 1}
    assert b.used_bytes(tier) == 200 + 900 + 1200 + 500


@pytest.mark.parametrize("n", range(0, 9))
def test_cardinality_bytes_round_trip(n):
    L, E = 5, 8
    b = simple_budget()
    limit = b.bytes_for_cardinality(L, E, n)
    assert P.MemoryBudget(b.expert_bytes, limit).n_high_per_layer(L, E) == n
    # And the tier array that budget describes costs exactly that.
    tier = np.full((L, E), 3)
    tier[:, :n] = 4
    assert b.used_bytes(tier) == limit


def test_n_high_per_layer_floors_a_budget_between_two_cardinalities():
    L, E = 5, 8
    b = simple_budget()
    exact = b.bytes_for_cardinality(L, E, 3)
    for slack in (0, 1, L * 100 - 1):
        assert P.MemoryBudget(
            b.expert_bytes, exact + slack).n_high_per_layer(L, E) == 3
    assert P.MemoryBudget(
        b.expert_bytes, exact + L * 100).n_high_per_layer(L, E) == 4


def test_n_high_per_layer_clamps_at_both_ends():
    L, E = 5, 8
    b = simple_budget()
    assert P.MemoryBudget(b.expert_bytes, 1).n_high_per_layer(L, E) == 0
    assert P.MemoryBudget(
        b.expert_bytes, 10 ** 12).n_high_per_layer(L, E) == E
    assert simple_budget().n_high_per_layer(L, E) is None  # unbounded


def test_glm52_budget_reads_back_as_experts_per_layer():
    """The operator's two spellings of the same ceiling must agree."""
    eb = P.ExpertBytes.from_measurements(
        P.MEASURED_GLM52_TP4_PER_RANK, provenance="test")
    L, E = 92, 160
    b = P.MemoryBudget(eb)
    for n in (0, 8, 40, 160):
        limit = b.bytes_for_cardinality(L, E, n)
        assert P.MemoryBudget(eb, limit).n_high_per_layer(L, E) == n


# ======================================================= headroom arithmetic

def test_headroom_counts_promotions_that_actually_fit():
    b = simple_budget()
    tier = np.full((2, 4), 3)          # 8 experts x 300 = 2400
    assert b.used_bytes(tier) == 2400
    for extra, want in [(0, 0), (99, 0), (100, 1), (250, 2), (1000, 8)]:
        bb = P.MemoryBudget(b.expert_bytes, 2400 + extra)
        assert bb.headroom_bytes(tier) == extra
        assert bb.promotions_headroom(tier) == want


def test_headroom_is_capped_by_the_low_tier_experts_that_exist():
    b = P.MemoryBudget(simple_budget().expert_bytes, 10 ** 9)
    tier = np.array([[3, 4, 4, 4]])    # only ONE promotable expert
    assert b.promotions_headroom(tier) == 1
    assert P.MemoryBudget(b.expert_bytes, 10 ** 9).promotions_headroom(
        np.full((2, 4), 4)) == 0       # nothing left to promote


def test_over_budget_headroom_is_reported_negative_not_clamped():
    b = simple_budget()
    tier = np.full((2, 4), 3)          # 2400 B
    bb = P.MemoryBudget(b.expert_bytes, 2150)
    assert bb.headroom_bytes(tier) == -250
    assert bb.promotions_headroom(tier) == -3   # floor(-250/100)
    s = bb.summary(tier)
    assert s["headroom_bytes"] == -250 and s["limit_bytes"] == 2150


def test_unbounded_budget_reports_none_not_zero():
    b = simple_budget()
    tier = np.full((2, 4), 3)
    assert b.headroom_bytes(tier) is None
    assert b.promotions_headroom(tier) is None
    s = b.summary(tier)
    assert s["limit_bytes"] is None and s["used_bytes"] == 2400
    assert s["mean_bits_per_expert"] == 3.0


def test_summary_always_carries_the_byte_model_provenance():
    s = simple_budget(3000).summary(np.full((2, 4), 3))
    assert s["source"] == "test" and s["is_reference"] is False
    assert s["per_k_bytes"][4] == 400
    assert s["promotion_cost_bytes"] == 100


# ========================================================== enforcement

def test_one_for_one_k3_k4_trades_are_byte_neutral_and_always_admitted():
    b = simple_budget(limit=None)
    tier = np.array([[4, 4, 3, 3, 3, 3]])
    b = P.MemoryBudget(b.expert_bytes, limit_bytes=int(b.used_bytes(tier)))
    kept, rej = P.budget_filter([(0, 0, 2), (0, 1, 3)], tier, b)
    assert kept == [(0, 0, 2), (0, 1, 3)] and rej == []


def test_budget_filter_rejects_a_swap_that_grows_the_pool():
    """A K2 expert entering K4 costs two steps and only frees one."""
    b = simple_budget()
    tier = np.array([[4, 4, 2, 3, 3, 3]])          # 400+400+200+300*3 = 1900
    limit = int(b.used_bytes(tier))                # exactly at the line
    bb = P.MemoryBudget(b.expert_bytes, limit)
    kept, rej = P.budget_filter([(0, 0, 2)], tier, bb)
    assert kept == []
    assert len(rej) == 1
    r = rej[0]
    assert r["kind"] == "swap" and r["ok"] is False
    assert (r["layer"], r["expert_in"], r["expert_out"]) == (0, 2, 0)
    assert r["budget_bytes"] == 1900
    assert r["current_bytes"] == 1900
    assert r["requested_bytes"] == 2000       # -100 (K4->K3) +200 (K2->K4)
    assert r["overshoot_bytes"] == 100
    # Raising the ceiling by exactly the overshoot admits it.
    kept2, rej2 = P.budget_filter(
        [(0, 0, 2)], tier, P.MemoryBudget(b.expert_bytes, limit + 100))
    assert kept2 == [(0, 0, 2)] and rej2 == []


def test_budget_filter_is_cumulative_over_the_ordered_list():
    b = simple_budget()
    tier = np.array([[4, 4, 2, 2, 3, 3]])          # 400*2+200*2+300*2 = 1800
    bb = P.MemoryBudget(b.expert_bytes, 1900)      # room for ONE +100 swap
    kept, rej = P.budget_filter([(0, 0, 2), (0, 1, 3)], tier, bb)
    assert kept == [(0, 0, 2)]
    assert [r["overshoot_bytes"] for r in rej] == [100]
    assert rej[0]["current_bytes"] == 1900         # after the first swap


def test_budget_filter_is_a_no_op_without_a_ceiling():
    swaps = [(0, 0, 2), (0, 1, 3)]
    tier = np.array([[4, 4, 3, 3]])
    assert P.budget_filter(swaps, tier, None) == (swaps, [])
    assert P.budget_filter(swaps, tier, simple_budget()) == (swaps, [])


# ------------------------------------------------------------ promotions

def promo_inputs(L=2, E=4):
    stats = {"count": np.tile(np.arange(E, 0, -1.0), (L, 1)),
             "mass": np.ones((L, E))}
    eps = {3: np.full((L, E), 1.0), 4: np.zeros((L, E))}
    return stats, eps


def test_promotions_spend_headroom_best_score_first():
    stats, eps = promo_inputs()
    tier = np.full((2, 4), 3)                      # 8 x 300 = 2400
    b = P.MemoryBudget(simple_budget().expert_bytes, 2400 + 300)
    promos, rej = P.plan_promotions(stats, eps, tier, b,
                                    cfg={"max_swaps_per_layer": 8})
    assert len(promos) == 3                        # 300 B / 100 B each
    # expert 0 scores highest in each layer, then expert 1.
    assert promos == [(0, 0), (1, 0), (0, 1)]
    assert P.apply_promotions(tier, promos).sum() == 2400 // 100 + 3


def test_promotion_rejection_carries_budget_current_requested_overshoot():
    stats, eps = promo_inputs()
    tier = np.full((2, 4), 3)
    b = P.MemoryBudget(simple_budget().expert_bytes, 2400 + 150)
    promos, rej = P.plan_promotions(stats, eps, tier, b,
                                    cfg={"max_swaps_per_layer": 8})
    assert len(promos) == 1
    assert rej, "an over-budget candidate must be reported, not dropped"
    r = rej[0]
    assert r["kind"] == "promotion" and r["ok"] is False
    assert r["budget_bytes"] == 2550
    assert r["current_bytes"] == 2500          # after the admitted promotion
    assert r["requested_bytes"] == 2600
    assert r["overshoot_bytes"] == 50
    assert r["headroom_bytes"] == -50


def test_promotions_respect_pins_dwell_and_zero_scores():
    stats, eps = promo_inputs()
    tier = np.full((2, 4), 3)
    b = P.MemoryBudget(simple_budget().expert_bytes, 10 ** 6)
    pins = np.zeros((2, 4), dtype=np.int64)
    pins[0, 0] = P.PIN_K3
    dwell = np.full((2, 4), 100)
    dwell[1, :] = 0
    stats["count"][0, 1] = 0.0                 # zero score -> no gain
    promos, _ = P.plan_promotions(
        stats, eps, tier, b, pins=pins, dwell=dwell,
        cfg={"max_swaps_per_layer": 8, "dwell_steps": 10})
    assert (0, 0) not in promos                # pinned to K3
    assert (0, 1) not in promos                # no routing pressure
    assert all(l == 0 for l, _ in promos)      # layer 1 is not dwell-ready
    assert set(promos) == {(0, 2), (0, 3)}


def test_promotions_respect_the_per_layer_cap():
    stats, eps = promo_inputs()
    tier = np.full((2, 4), 3)
    b = P.MemoryBudget(simple_budget().expert_bytes, 10 ** 6)
    promos, _ = P.plan_promotions(stats, eps, tier, b,
                                  cfg={"max_swaps_per_layer": 1})
    assert sorted(promos) == [(0, 0), (1, 0)]


def test_promotions_are_deterministic():
    stats, eps = promo_inputs(L=3, E=6)
    tier = np.full((3, 6), 3)
    b = P.MemoryBudget(simple_budget().expert_bytes, 10 ** 5)
    runs = [P.plan_promotions(stats, eps, tier, b,
                              cfg={"max_swaps_per_layer": 6})[0]
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_decide_with_budget_matches_decide_when_unbounded():
    stats, eps = promo_inputs(L=2, E=6)
    tier = np.full((2, 6), 3)
    tier[:, :2] = 4
    cfg = {"n_k4": 2, "dwell_steps": 0, "hysteresis": 1.0,
           "max_swaps_per_layer": 2, "max_swaps_total": 8}
    want = P.decide(stats, eps, tier, cfg=cfg)
    for budget in (None, simple_budget()):
        got, promos, rej = P.decide_with_budget(stats, eps, tier, budget,
                                                cfg=cfg)
        assert got == want and promos == [] and rej == []


# ============================================================ the loop

E = 8
LAYERS = [3, 4]


@pytest.fixture(autouse=True)
def clean_fq_env(monkeypatch):
    for env in (FL.FQ_INTERVAL_ENV, FL.FQ_APPLY_MODE_ENV,
                FL.FQ_ARTIFACT_DIR_ENV, FL.FQ_POLICY_ENV, FL.FQ_EPS_ROOT_ENV,
                FL.FQ_CACHE_ROOT_ENV, FL.FQ_MAX_SWAPS_LAYER_ENV,
                FL.FQ_MAX_SWAPS_TOTAL_ENV, FL.FQ_DWELL_ENV,
                FL.FQ_HYSTERESIS_ENV, FL.FQ_JACCARD_FLOOR_ENV,
                FL.FQ_MEMORY_BUDGET_ENV, FL.FQ_EXPERT_BYTES_ENV):
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


class FakeRouter:
    def __init__(self):
        self.capture_fn = None
        self.global_num_experts = E

    def set_capture_fn(self, fn):
        self.capture_fn = fn


def make_collector():
    c = FqStatsCollector(E, window_len=8, window_stride=2, decay=0.9,
                         device="cpu")
    routers = {lid: FakeRouter() for lid in LAYERS}
    for lid, r in routers.items():
        c.bind_router(lid, r)
    return c, routers


def boot_doc():
    bits = [4, 4, 3, 3, 3, 3, 3, 3]
    return {
        "schema": "fq-policy/2",
        "manifest": "m" * 64,
        "budget": {"mode": "fixed_cardinality",
                   "n_k4_per_layer": {str(lid): 2 for lid in LAYERS}},
        "bits_per_expert": {str(lid): list(bits) for lid in LAYERS},
        "pinned": {},
    }


def eps_hot4():
    e3 = np.full((2, E), 0.05)
    e3[:, 4] = 0.5
    return {P.K3: e3, P.K4: np.zeros((2, E))}


# Boot occupancy: 12 experts at K3 + 4 at K4 = 12*300 + 4*400 = 5200 B.
BOOT_BYTES = 5200


def make_state(tmp_path=None, *, limit=None, metrics=None, interval=4,
               **cfg_kw):
    cfg_kw.setdefault("dwell_steps", 0)
    cfg_kw.setdefault("jaccard_floor", 0.0)
    cfg = FL.FqLoopConfig(interval_steps=interval, **cfg_kw)
    collector, routers = make_collector()
    store = None
    if tmp_path is not None:
        store = S.PolicyStore(tmp_path, "m" * 64)
        store.commit(boot_doc(), num_experts=E)
    state = FL.FungibleQuantState(
        collector, boot_doc(), config=cfg, eps=eps_hot4(), store=store,
        metrics=metrics, budget=simple_budget(limit))
    return state, routers


def drive(state, routers, steps=None):
    for _ in range(steps or state.cfg.interval_steps):
        t = torch.tensor([[4, 4], [4, 1], [4, 0]], dtype=torch.int64)
        for r in routers.values():
            r.capture_fn(t)
        state.step()


def sample(registry, name, labels=None):
    v = registry.get_sample_value(name, labels or {})
    return 0.0 if v is None else v


def test_state_reports_boot_footprint_from_actual_occupancy():
    state, _ = make_state(limit=BOOT_BYTES + 500)
    s = state.budget.summary(state.tier_of)
    assert s["used_bytes"] == BOOT_BYTES
    assert s["headroom_bytes"] == 500
    assert s["headroom_promotions"] == 5


def test_gauges_expose_budget_used_and_headroom(tmp_path):
    registry = prometheus_client.CollectorRegistry()
    metrics = FL.FqMetrics(registry=registry)
    state, routers = make_state(tmp_path, limit=BOOT_BYTES + 250,
                                metrics=metrics)
    assert sample(registry, "fq_memory_budget_bytes") == BOOT_BYTES + 250
    assert sample(registry, "fq_memory_used_bytes") == BOOT_BYTES
    assert sample(registry, "fq_promotions_headroom") == 2.0
    # Dryrun never moves the running policy, so the gauges hold steady.
    drive(state, routers)
    assert sample(registry, "fq_memory_used_bytes") == BOOT_BYTES
    assert sample(registry, "fq_promotions_headroom") == 2.0


def test_gauges_mark_an_unconfigured_budget_with_documented_sentinels(tmp_path):
    registry = prometheus_client.CollectorRegistry()
    metrics = FL.FqMetrics(registry=registry)
    make_state(tmp_path, limit=None, metrics=metrics)
    assert sample(registry, "fq_memory_budget_bytes") == 0.0   # unconfigured
    assert sample(registry, "fq_promotions_headroom") == -1.0  # unbounded
    assert sample(registry, "fq_memory_used_bytes") == BOOT_BYTES


def test_gauges_go_negative_when_the_boot_policy_is_over_budget(tmp_path):
    registry = prometheus_client.CollectorRegistry()
    metrics = FL.FqMetrics(registry=registry)
    make_state(tmp_path, limit=BOOT_BYTES - 250, metrics=metrics)
    assert sample(registry, "fq_memory_used_bytes") == BOOT_BYTES
    assert sample(registry, "fq_promotions_headroom") == -3.0


def test_interval_rejects_promotions_over_budget_and_logs_the_numbers(
        tmp_path, fq_caplog):
    # Headroom for exactly one K3->K4 promotion (100 B each), plus 50 B.
    state, routers = make_state(tmp_path, limit=BOOT_BYTES + 150)
    with fq_caplog.at_level(logging.WARNING):
        drive(state, routers)

    rec = json.loads(next(
        (state.store.root / "decisions").glob("*.json")).read_text())
    b = rec["budget"]
    assert b["limit_bytes"] == BOOT_BYTES + 150
    assert b["used_bytes"] == BOOT_BYTES          # the RUNNING pool
    assert b["proposed_bytes"] == BOOT_BYTES + 100
    assert rec["totals"]["promotions"] == 1
    assert rec["totals"]["budget_rejections"] == len(b["rejections"]) > 0

    r = b["rejections"][0]
    assert r["kind"] == "promotion" and r["ok"] is False
    assert r["budget_bytes"] == BOOT_BYTES + 150
    assert r["current_bytes"] == BOOT_BYTES + 100
    assert r["requested_bytes"] == BOOT_BYTES + 200
    assert r["overshoot_bytes"] == 50

    logged = "\n".join(m for m in fq_caplog.messages
                        if "memory budget" in m)
    assert "REJECTED by memory budget" in logged
    for n in (b["limit_bytes"], r["current_bytes"], r["requested_bytes"], 50):
        assert str(n) in logged


def test_a_zero_headroom_budget_admits_swaps_but_no_promotions(tmp_path):
    state, routers = make_state(tmp_path, limit=BOOT_BYTES)
    drive(state, routers)
    rec = json.loads(next(
        (state.store.root / "decisions").glob("*.json")).read_text())
    # The byte-neutral 1-for-1 trades still go through...
    assert rec["totals"]["executed"] == 2
    assert all(sw["expert_in"] == 4 for sw in rec["swaps"])
    # ...and nothing was allowed to grow the pool.
    assert rec["totals"]["promotions"] == 0
    assert rec["budget"]["proposed_bytes"] == BOOT_BYTES
    assert rec["budget"]["headroom_promotions"] == 0


def test_proposal_with_promotions_declares_the_new_cardinality(tmp_path):
    state, routers = make_state(tmp_path, limit=BOOT_BYTES + 150)
    drive(state, routers)
    proposal = json.loads(next(
        (state.store.root / "history").glob("*-proposed.json")).read_text())
    caps = proposal["budget"]["n_k4_per_layer"]
    counts = {lid: sum(b == 4 for b in proposal["bits_per_expert"][lid])
              for lid in proposal["bits_per_expert"]}
    assert {k: int(v) for k, v in caps.items()} == counts
    assert sum(counts.values()) == 5            # 4 boot + 1 promotion
    # store.validate_policy enforces cap == n; the proposal must survive it.
    S.validate_policy(proposal, num_experts=E)
    assert proposal["budget"]["mode"] == "max_bytes"
    assert proposal["budget"]["max_bytes_per_rank"] == BOOT_BYTES + 150
    assert proposal["budget"]["bytes_per_expert_per_rank"]["4"] == 400
    assert proposal["budget"]["bytes_source"] == "test"


def test_unbounded_budget_leaves_the_decision_path_unchanged(tmp_path):
    state, routers = make_state(tmp_path, limit=None)
    drive(state, routers)
    rec = json.loads(next(
        (state.store.root / "decisions").glob("*.json")).read_text())
    assert rec["totals"]["executed"] == 2
    assert rec["totals"]["promotions"] == 0
    assert rec["budget"]["limit_bytes"] is None
    assert rec["budget"]["headroom_promotions"] is None


def test_composition_table_reports_the_budget(fq_caplog):
    with fq_caplog.at_level(logging.INFO):
        make_state(limit=BOOT_BYTES + 250)
    text = "\n".join(fq_caplog.messages)
    assert "memory budget:" in text
    assert "headroom" in text and "2 more K3->K4 promotions" in text
    assert "bytes/expert/rank:" in text


# ------------------------------------------------------ byte-model sourcing

def test_env_declared_expert_bytes_win(monkeypatch):
    eb = FL.resolve_expert_bytes(spec="k3=3542028,k4=4721676")
    assert eb.bytes_for(5) == 5_901_324
    assert "operator-declared" in eb.provenance and eb.is_reference is False


@pytest.mark.parametrize("spec", ["k3=1", "3542028", "k3:1,k4:2", "kx=1,ky=2"])
def test_bad_expert_bytes_spec_is_refused(spec):
    with pytest.raises(ValueError):
        FL.parse_expert_bytes_spec(spec)


def write_fake_checkpoint(root, k):
    """A safetensors file whose header carries one expert's real geometry."""
    import struct

    header, off = {}, 0
    for name, nbytes, shape in fruit_expert_tensor_table(k):
        dtype = "I16" if name.endswith("trellis") else (
            "I32" if name.endswith("mcg") else "F16")
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "model-layer-003.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)          # header only: the loop never reads the body
    return root


def test_expert_bytes_are_derived_from_the_loaded_checkpoints_shapes(tmp_path):
    root = write_fake_checkpoint(tmp_path / "ckpt", 3)
    eb = FL.expert_bytes_from_checkpoint(root)
    assert eb is not None and eb.is_reference is False
    assert eb.bytes_for(3) * 4 == 617_520 and eb.bytes_for(4) * 4 == 814_128
    assert "derived from loaded tensor shapes" in eb.provenance
    assert "model-layer-003.safetensors" in eb.provenance


def test_resolution_order_prefers_the_checkpoint_over_the_reference(tmp_path):
    root = write_fake_checkpoint(tmp_path / "ckpt", 4)
    eb = FL.resolve_expert_bytes(artifact_dir=str(root))
    assert eb.is_reference is False and eb.bytes_for(4) == 203_532
    # ...and the declared spec still outranks the checkpoint.
    eb2 = FL.resolve_expert_bytes(spec="k3=300,k4=400", artifact_dir=str(root))
    assert eb2.bytes_for(4) == 400


def test_missing_checkpoint_falls_back_loudly_never_silently(tmp_path,
                                                             fq_caplog):
    with fq_caplog.at_level(logging.WARNING):
        eb = FL.resolve_expert_bytes(artifact_dir=str(tmp_path / "nope"))
    assert eb.is_reference is True
    text = "\n".join(fq_caplog.messages)
    assert "REFERENCE" in text and FL.FQ_EXPERT_BYTES_ENV in text
    assert FL.expert_bytes_from_checkpoint(tmp_path / "nope") is None


def test_state_resolves_a_fractional_budget_against_the_device(monkeypatch):
    monkeypatch.setattr(FL, "device_total_bytes", lambda: 10_000)
    monkeypatch.setenv(FL.FQ_MEMORY_BUDGET_ENV, "0.80")
    monkeypatch.setenv(FL.FQ_EXPERT_BYTES_ENV, "k3=300,k4=400")
    cfg = FL.FqLoopConfig.from_env()
    cfg.interval_steps, cfg.dwell_steps, cfg.jaccard_floor = 4, 0, 0.0
    collector, _ = make_collector()
    state = FL.FungibleQuantState(collector, boot_doc(), config=cfg,
                                  eps=eps_hot4())
    assert state.budget.limit_bytes == 8_000
    assert state.budget.expert_bytes.bytes_for(4) == 400
    assert state.budget.used_bytes(state.tier_of) == BOOT_BYTES


def test_a_bad_budget_spec_fails_closed_rather_than_ignoring_the_ceiling(
        monkeypatch):
    monkeypatch.setenv(FL.FQ_MEMORY_BUDGET_ENV, "definitely-not-a-size")
    cfg = FL.FqLoopConfig.from_env()
    cfg.interval_steps, cfg.dwell_steps, cfg.jaccard_floor = 4, 0, 0.0
    collector, _ = make_collector()
    with pytest.raises(ValueError):
        FL.FungibleQuantState(collector, boot_doc(), config=cfg,
                              eps=eps_hot4())


def test_device_total_bytes_is_none_without_cuda():
    assert FL.device_total_bytes() is None or FL.device_total_bytes() > 0


# ================================================== occupancy table footer

def test_table_footer_shows_ceiling_use_and_headroom():
    summary = simple_budget(3000).summary(np.full((2, 4), 3))
    out = OT.render({0: {3: 4}, 1: {3: 4}}, budget=summary)
    assert "memory budget: 2.93 KiB" in out
    assert "used 2.34 KiB (80.0%)" in out
    assert "= 6 more K3->K4 promotions" in out
    assert "K3=300" in out and "K4=400" in out
    assert "byte model: test" in out


def test_table_footer_shouts_when_over_budget():
    summary = simple_budget(2000).summary(np.full((2, 4), 3))
    out = OT.render({0: {3: 4}}, budget=summary)
    assert "OVER BUDGET by 400 B" in out


def test_table_footer_flags_a_reference_byte_model():
    b = P.MemoryBudget(P.reference_expert_bytes(), 10 ** 12)
    out = OT.render({0: {3: 4}}, budget=b.summary(np.full((1, 4), 3)))
    assert "REFERENCE (not this checkpoint!)" in out


def test_table_without_a_budget_says_so_instead_of_implying_one():
    out = OT.render({0: {3: 4}}, budget=simple_budget().summary(
        np.full((1, 4), 3)))
    assert "no byte budget set" in out
    assert "headroom" not in out


def test_budget_survives_the_early_return_paths():
    summary = simple_budget(3000).summary(np.full((2, 4), 3))
    same = {0: {3: 4}}
    assert "headroom 6 promotions" in OT.render(
        same, same, diff_only=True, budget=summary)
    assert "2.34 KiB/2.93 KiB used" in OT.render({}, budget=summary)


def test_render_without_a_budget_is_unchanged():
    assert "memory" not in OT.render({0: {3: 4}})


@pytest.mark.parametrize("n,want", [
    (0, "0 B"), (512, "512 B"), (1024, "1.00 KiB"),
    (78 * GIB, "78.00 GiB"), (-1536, "-1.50 KiB"), (None, "unbounded"),
])
def test_format_bytes(n, want):
    assert OT.format_bytes(n) == want


# ============================== "experts/bpw per layer" budget spellings

def from_spec(spec, *, L=5, E=8, device=None):
    return P.MemoryBudget.from_spec(
        spec, simple_budget().expert_bytes, num_layers=L, num_experts=E,
        device_total_bytes=device)


@pytest.mark.parametrize("spec", ["3/layer", "3experts/layer",
                                  "3 experts / layer", "3EXPERT/layer"])
def test_budget_can_be_written_as_experts_per_layer(spec):
    b = from_spec(spec)
    # 5 layers x (5 K3 + 3 K4) = 5 * (1500 + 1200)
    assert b.limit_bytes == 5 * (5 * 300 + 3 * 400)
    assert b.n_high_per_layer(5, 8) == 3


def test_experts_per_layer_and_bytes_describe_the_same_ceiling():
    assert from_spec("3/layer").limit_bytes == from_spec(
        str(5 * (5 * 300 + 3 * 400))).limit_bytes


def test_budget_can_be_written_as_bits_per_weight():
    # 8 experts/layer, K3 base, K4 overlay: 3.5 bpw == 4 experts at K4.
    b = from_spec("3.5bpw")
    assert b.n_high_per_layer(5, 8) == 4
    assert b.limit_bytes == 5 * (4 * 300 + 4 * 400)
    # A bpw the ladder cannot subdivide rounds DOWN, never up.
    assert from_spec("3.6bpw").n_high_per_layer(5, 8) == 4
    assert from_spec("3.0bpw").n_high_per_layer(5, 8) == 0
    assert from_spec("4bpw").n_high_per_layer(5, 8) == 8
    assert from_spec("9bpw").n_high_per_layer(5, 8) == 8   # clamped at K4


def test_bpw_below_the_base_tier_is_refused():
    with pytest.raises(ValueError, match="below the base tier"):
        from_spec("2.5bpw")


def test_more_experts_than_a_layer_has_is_refused():
    with pytest.raises(ValueError, match="only has 8"):
        from_spec("9/layer")


def test_from_spec_still_takes_bytes_fractions_and_none():
    assert from_spec("78g").limit_bytes == 78 * GIB
    assert from_spec("0.5", device=1000).limit_bytes == 500
    assert from_spec(None).limit_bytes is None
    assert from_spec("off").limit_bytes is None
    with pytest.raises(ValueError):
        from_spec("nonsense")


def test_state_accepts_a_cardinality_budget_from_env(monkeypatch):
    monkeypatch.setenv(FL.FQ_MEMORY_BUDGET_ENV, "3/layer")
    monkeypatch.setenv(FL.FQ_EXPERT_BYTES_ENV, "k3=300,k4=400")
    cfg = FL.FqLoopConfig.from_env()
    cfg.interval_steps, cfg.dwell_steps, cfg.jaccard_floor = 4, 0, 0.0
    collector, _ = make_collector()
    state = FL.FungibleQuantState(collector, boot_doc(), config=cfg,
                                  eps=eps_hot4())
    # 2 layers x (5 K3 + 3 K4) = 2 * (1500 + 1200) = 5400
    assert state.budget.limit_bytes == 5400
    assert state.budget.used_bytes(state.tier_of) == BOOT_BYTES   # 5200
    assert state.budget.promotions_headroom(state.tier_of) == 2
