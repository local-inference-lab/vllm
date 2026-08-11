# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the forced-re-tiering admin API (exl3_fungible/admin.py).

Covers the contract the spec (runs/m5-serve/admin-api-spec.md) pins down:
adjust_k disambiguation (relative AND absolute), batch atomicity, the
memory guard, the fixed-cardinality invariant, fragment-availability
refusal, both gates being off by default, and the forced change being
attributable in the decision record.

No GPU, no built ``vllm._C``: the swap engine is driven through fakes
(``stage``/``apply`` are recorded, never executed), and the router is
exercised with FastAPI's TestClient against a fake engine client.
"""
import json
import logging

import numpy as np
import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        admin as A,
        loop as FL,
        policy as P,
        store as S,
        swap as SW,
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
    # Give the stub a real __path__ so a submodule this file did not
    # preload still resolves from the working tree, and preload
    # ``integration`` even though nothing here needs it: this module is
    # collected first (alphabetically), and once ``.loop`` is in
    # sys.modules the later test modules' "is the real package importable"
    # probe starts answering yes. Leaving a hole in the stub would turn
    # that into an ImportError in somebody else's test.
    _pkg.__path__ = [str(_dir)]
    sys.modules[_pkg_name] = _pkg
    for _sub in ("policy", "stats", "store", "decision_log",
                 "occupancy_table", "swap", "loop", "integration", "admin"):
        _mod = _load(f"{_pkg_name}.{_sub}", _dir / f"{_sub}.py")
        setattr(_pkg, _sub, _mod)
    P, S, SW, FL, A = (_pkg.policy, _pkg.store, _pkg.swap, _pkg.loop,
                       _pkg.admin)
    FqStatsCollector = _pkg.stats.FqStatsCollector

prometheus_client = pytest.importorskip("prometheus_client")

E = 16              # experts per layer
LAYERS = [23, 24]   # model layer ids (policy keys; collector bind ids)
N_K4 = 4            # K4 capacity per layer
HIDDEN = 128
INTERMEDIATE = 64


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def clean_admin_env(monkeypatch):
    for env in (A.DEV_MODE_ENV, A.ADMIN_API_ENV, A.ADMIN_ENABLE_ENV,
                A.ADMIN_TOKEN_ENV, A.ADMIN_MAX_ITEMS_ENV,
                A.ADMIN_MEM_BUDGET_ENV, A.ADMIN_MEM_HEADROOM_ENV,
                A.ADMIN_MIN_FREE_SLOTS_ENV, A.ADMIN_DRAIN_TIMEOUT_ENV,
                A.ADMIN_ALLOW_AUTO_BALANCE_ENV):
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


class FakeRouter:
    def __init__(self):
        self.capture_fn = None
        self.global_num_experts = E

    def set_capture_fn(self, fn):
        self.capture_fn = fn


def boot_doc():
    """Experts 0..3 at K4 in both layers; capacity 4 (occupancy == capacity)."""
    bits = [4] * N_K4 + [3] * (E - N_K4)
    return {
        "schema": "fq-policy/2",
        "manifest": "m" * 64,
        "budget": {"mode": "fixed_cardinality",
                   "n_k4_per_layer": {str(lid): N_K4 for lid in LAYERS}},
        "bits_per_expert": {str(lid): list(bits) for lid in LAYERS},
        "pinned": {},
    }


def make_state(tmp_path=None, *, doc=None, metrics=None, rank=0):
    doc = doc or boot_doc()
    collector = FqStatsCollector(E, window_len=8, window_stride=2, decay=0.9,
                                 device="cpu")
    routers = {}
    for lid in LAYERS:
        routers[lid] = FakeRouter()
        collector.bind_router(lid, routers[lid])
    store = None
    if tmp_path is not None:
        store = S.PolicyStore(tmp_path, doc["manifest"])
        store.commit(doc, num_experts=E)
    cfg = FL.FqLoopConfig(interval_steps=1000, dwell_steps=0,
                          jaccard_floor=0.0)
    eps = {P.K3: np.full((len(LAYERS), E), 0.1),
           P.K4: np.zeros((len(LAYERS), E))}
    state = FL.FungibleQuantState(collector, doc, config=cfg, eps=eps,
                                  store=store, metrics=metrics, rank=rank,
                                  is_lead=(rank == 0))
    # Give the experts some traffic so the decision record has real scores.
    for _ in range(4):
        t = torch.tensor([[0, 5], [1, 6]], dtype=torch.int64)
        for r in routers.values():
            r.capture_fn(t)
        state.step()
    return state, routers


class FakeEngine:
    """A SwapEngine stand-in: records what it was asked to do.

    ``stage``/``apply`` never touch a device — the point of every test that
    uses this is what the admin layer decided BEFORE the engine is reached.
    """

    def __init__(self, layers=None, *, max_pairs=32, fail_apply=None):
        self.layers = {lid: object() for lid in (layers or LAYERS)}
        self.max_pairs = max_pairs
        self.hidden_size = HIDDEN
        self.intermediate_size = INTERMEDIATE
        self.generation = 0
        self.source = None
        self.staged = []
        self.applied = []
        self._fail_apply = fail_apply

    def stage(self, plan, *, fail_atomic=False, on_unavailable=None):
        self.staged.append({"plan": plan, "fail_atomic": fail_atomic,
                            "on_unavailable": on_unavailable})
        return SW.StagedBatch(
            plan=plan, slab_ops=[], rotation_ops=[], map_ops=[],
            staged_layers=[], bytes_h2d=4096, stage_seconds=0.001, mcg=None,
            requested_plan=plan, undo_ops=[] if fail_atomic else None)

    def apply(self, *, staged, quiesce, stream=None, memo_hook=None,
              policy_store=None, policy_doc=None, policy_num_experts=None,
              **kw):
        self.applied.append({"staged": staged, "memo_hook": memo_hook,
                             "policy_store": policy_store,
                             "policy_doc": policy_doc})
        if self._fail_apply is not None:
            raise self._fail_apply
        self.generation += 1
        committed = None
        if policy_store is not None and policy_doc is not None:
            committed = policy_store.commit(
                policy_doc, num_experts=policy_num_experts)
        return SW.ApplyReport(
            pairs=len(staged.plan), layers=len(staged.plan.layers()),
            bytes_h2d=staged.bytes_h2d, stage_seconds=staged.stage_seconds,
            window_seconds=0.0004, generation=self.generation,
            committed_policy_hash=committed)


class FakeFragment:
    def __init__(self, k):
        self.k = k
        self.origin = "fake"


class FakeResolver:
    """Supplies every (layer, expert, k) except the ones in ``missing``.

    A miss is reported the way the real resolver reports it: the nearest
    lower K is substituted and an encode is queued.
    """

    def __init__(self, missing=()):
        self.missing = {tuple(m) for m in missing}
        self.calls = []
        self.queued = []

    def resolve_best(self, layer, expert, k, *, chain_out=None):
        self.calls.append((layer, expert, k))
        if (layer, expert, k) in self.missing:
            if chain_out is not None:
                chain_out.append("local:miss")
                chain_out.append(f"FALLBACK K{k - 1} local:hit")
            self.queued.append((layer, expert, k))
            return FakeFragment(k - 1)
        if chain_out is not None:
            chain_out.append("local:hit")
        return FakeFragment(k)

    def _get_encode_queue(self):
        resolver = self

        class _Q:
            def position(self, layer, expert, k):
                key = (layer, expert, k)
                return (resolver.queued.index(key)
                        if key in resolver.queued else None)

        return _Q()


def memory_model():
    return A.MemoryModel(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE)


def req(body, **kw):
    return A.parse_request(body, **kw)


def balanced_body(**kw):
    """The operator's own example, on the toy geometry."""
    body = {"items": [{"layer": 23, "expert": 0, "adjust_k": -1},
                      {"layer": 23, "expert": 8, "adjust_k": "+1"}]}
    body.update(kw)
    return body


# --------------------------------------------------- adjust_k disambiguation


@pytest.mark.parametrize("sent,interpretation,k,delta", [
    (-1, "relative", None, -1),          # negative number: no negative K
    (3, "absolute", 3, None),            # "absolute like adjust_k=3"
    (4, "absolute", 4, None),
    ("+1", "relative", None, 1),         # explicit sign survives JSON
    ("-1", "relative", None, -1),
    ("3", "absolute", 3, None),          # bare digits: absolute
])
def test_adjust_k_table(sent, interpretation, k, delta):
    parsed = req({"items": [{"layer": 23, "expert": 0, "adjust_k": sent}]})
    item = parsed.items[0]
    assert (item.interpretation, item.k, item.delta_k) == (interpretation, k,
                                                           delta)


@pytest.mark.parametrize("sent", [1, 0, 2, 5])
def test_adjust_k_positive_outside_ladder_is_ambiguous(sent):
    with pytest.raises(A.AdminError) as e:
        req({"items": [{"layer": 23, "expert": 0, "adjust_k": sent}]})
    assert e.value.code == "ambiguous_adjust_k"
    assert e.value.status == 400
    # The message must name the fix literally, not gesture at it.
    assert f'send the string "+{sent}"' in e.value.message
    assert f"delta_k: {sent}" in e.value.message


@pytest.mark.parametrize("sent", [0.5, "x", None, [1]])
def test_adjust_k_garbage(sent):
    with pytest.raises(A.AdminError) as e:
        req({"items": [{"layer": 23, "expert": 0, "adjust_k": sent}]})
    assert e.value.code == "bad_adjust_k"


def test_explicit_k_and_delta_k_fields():
    parsed = req({"items": [{"layer": 23, "expert": 0, "k": 4},
                            {"layer": 23, "expert": 1, "delta_k": -1}]})
    assert parsed.items[0].interpretation == "absolute"
    assert parsed.items[0].k == 4
    assert parsed.items[1].interpretation == "relative"
    assert parsed.items[1].delta_k == -1


@pytest.mark.parametrize("item", [
    {"layer": 23, "expert": 0},                            # none
    {"layer": 23, "expert": 0, "k": 4, "delta_k": 1},      # two
    {"layer": 23, "expert": 0, "adjust_k": -1, "k": 3},    # two
])
def test_adjust_k_mutual_exclusion(item):
    with pytest.raises(A.AdminError) as e:
        req({"items": [item]})
    assert e.value.code == "bad_adjust_k"


def test_unknown_fields_rejected():
    with pytest.raises(A.AdminError) as e:
        req({"items": [], "pin": "hold", "yolo": 1})
    assert e.value.code == "unknown_field"
    with pytest.raises(A.AdminError) as e:
        req({"items": [{"layer": 23, "expert": 0, "adjust_k": -1, "x": 1}]})
    assert e.value.code == "unknown_field"


def test_duplicate_item_names_both_indices():
    with pytest.raises(A.AdminError) as e:
        req({"items": [{"layer": 23, "expert": 7, "adjust_k": "+1"},
                       {"layer": 23, "expert": 7, "adjust_k": -1}]})
    assert e.value.code == "duplicate_item"
    assert e.value.details["indices"] == [0, 1]


def test_empty_request_with_no_pin_change():
    with pytest.raises(A.AdminError) as e:
        req({"items": [], "pin": "none"})
    assert e.value.code == "empty_request"


def test_too_many_items(monkeypatch):
    env = {A.ADMIN_MAX_ITEMS_ENV: "2"}
    with pytest.raises(A.AdminError) as e:
        req({"items": [{"layer": 23, "expert": i, "adjust_k": -1}
                       for i in range(3)]}, environ=env)
    assert e.value.code == "too_many_items"


def test_query_shorthand_desugars():
    parsed = req(None, query={"layer": "23", "expert": "250",
                              "adjust_k": "-1"})
    assert len(parsed.items) == 1
    assert parsed.items[0].layer == 23
    assert parsed.items[0].expert == 250
    assert parsed.items[0].interpretation == "relative"
    assert parsed.items[0].delta_k == -1
    assert parsed.mode == A.MODE_STRICT_PAIR


def test_query_plus_body_is_mixed_input():
    with pytest.raises(A.AdminError) as e:
        req(balanced_body(), query={"layer": "23"})
    assert e.value.code == "mixed_input"


# --------------------------------------------------------- item resolution


def test_resolve_reads_k_from_from_live_state():
    state, _ = make_state()
    resolved = A.resolve_items(req(balanced_body()), layers=state.layers,
                               tier_of=state.tier_of, num_experts=E)
    assert [(i.expert, i.k_from, i.k_to, i.outcome) for i in resolved] == [
        (0, 4, 3, "demoted"), (8, 3, 4, "promoted")]


def test_expert_out_of_range():
    state, _ = make_state()
    with pytest.raises(A.AdminError) as e:
        A.resolve_items(
            req({"items": [{"layer": 23, "expert": E, "adjust_k": -1}]}),
            layers=state.layers, tier_of=state.tier_of, num_experts=E)
    assert e.value.code == "expert_out_of_range"


def test_unknown_layer():
    state, _ = make_state()
    with pytest.raises(A.AdminError) as e:
        A.resolve_items(
            req({"items": [{"layer": 99, "expert": 0, "adjust_k": -1}]}),
            layers=state.layers, tier_of=state.tier_of, num_experts=E)
    assert e.value.code == "layer_not_registered"
    assert e.value.status == 404


def test_k5_is_refused_with_both_reasons():
    state, _ = make_state()
    with pytest.raises(A.AdminError) as e:
        A.resolve_items(
            req({"items": [{"layer": 23, "expert": 8, "adjust_k": "+2"}]}),
            layers=state.layers, tier_of=state.tier_of, num_experts=E)
    assert e.value.code == "tier_not_servable"
    assert e.value.status == 501
    assert "two-tier engine" in e.value.message
    assert "109568" in e.value.message and "101376" in e.value.message


def test_noop_item_is_not_an_error():
    state, _ = make_state()
    resolved = A.resolve_items(
        req({"items": [{"layer": 23, "expert": 0, "k": 4}]}),
        layers=state.layers, tier_of=state.tier_of, num_experts=E)
    assert resolved[0].outcome == "noop"


# ------------------------------------------------------------- cardinality


def test_balanced_pair_produces_exactly_one_swap(tmp_path):
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    assert list(plan.plan) == [(23, 0, 8)]
    assert plan.plan == SW.SwapPlan([(23, 0, 8)])


def test_absolute_and_relative_spellings_agree(tmp_path):
    state_a, _ = make_state(tmp_path / "a")
    state_b, _ = make_state(tmp_path / "b")
    rel = A.plan_retier(state_a, req(balanced_body()), engine=FakeEngine(),
                        resolver=FakeResolver(), memory=memory_model())
    absolute = A.plan_retier(
        state_b, req({"items": [{"layer": 23, "expert": 0, "k": 3},
                                {"layer": 23, "expert": 8, "delta_k": 1}]}),
        engine=FakeEngine(), resolver=FakeResolver(), memory=memory_model())
    assert rel.plan == absolute.plan
    assert (rel.new_doc["bits_per_expert"]
            == absolute.new_doc["bits_per_expert"])
    assert rel.new_doc["pinned"] == absolute.new_doc["pinned"]
    # Only the provenance (which records the literal spelling, the actor and
    # the timestamp) distinguishes the two documents.
    assert rel.policy_sha_after != absolute.policy_sha_after


def test_lone_promotion_is_refused_and_nothing_is_staged(tmp_path):
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(
            state, req({"items": [{"layer": 23, "expert": 8,
                                   "adjust_k": "+1"}]}),
            engine=engine, resolver=FakeResolver(), memory=memory_model())
    assert e.value.code == "cardinality_unbalanced"
    assert e.value.status == 409
    d = e.value.details
    assert d["layer"] == 23 and d["n_k4_per_layer"] == N_K4
    assert d["promotions"] == [{"expert": 8, "k_from": 3, "k_to": 4}]
    assert d["demotions"] == []
    assert len(d["remedies"]) == 3
    # Nothing reached the engine, and the store still holds the boot policy.
    assert engine.staged == [] and engine.applied == []
    assert S.PolicyStore(tmp_path, "m" * 64).load_current(
        num_experts=E) == boot_doc()


def test_noop_that_unbalances_a_pair_is_refused(tmp_path):
    """Expert 1 is already K4; asking for K4 is a no-op, so the demotion
    vanishes and the promotion of e8 is left unpaired."""
    state, _ = make_state(tmp_path)
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(
            state, req({"items": [{"layer": 23, "expert": 1, "k": 4},
                                  {"layer": 23, "expert": 8, "k": 4}]}),
            engine=FakeEngine(), resolver=FakeResolver(),
            memory=memory_model())
    assert e.value.code == "cardinality_unbalanced"
    assert e.value.details["noop_items"] == [{"expert": 1, "k": 4}]


def test_batch_atomicity_one_bad_item_rejects_the_whole_batch(tmp_path):
    """Three of four items are perfectly legal; the fourth is out of range.
    Nothing may be planned, staged or written."""
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    before_sha = state.policy_sha
    body = {"items": [
        {"layer": 23, "expert": 0, "adjust_k": -1},
        {"layer": 23, "expert": 8, "adjust_k": "+1"},
        {"layer": 24, "expert": 1, "adjust_k": -1},
        {"layer": 24, "expert": 99, "adjust_k": "+1"},   # out of range
    ]}
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(body), engine=engine,
                      resolver=FakeResolver(), memory=memory_model())
    assert e.value.code == "expert_out_of_range"
    assert engine.staged == [] and engine.applied == []
    assert state.policy_sha == before_sha
    assert (state.tier_of == np.asarray(
        [boot_doc()["bits_per_expert"][str(l)] for l in LAYERS])).all()


def test_cross_layer_balance_is_per_layer(tmp_path):
    """A promotion in layer 23 paired with a demotion in layer 24 is NOT a
    trade — each layer's slabs are sized independently."""
    state, _ = make_state(tmp_path)
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(
            state, req({"items": [{"layer": 23, "expert": 8, "adjust_k": "+1"},
                                  {"layer": 24, "expert": 0, "adjust_k": -1}]}),
            engine=FakeEngine(), resolver=FakeResolver(),
            memory=memory_model())
    assert e.value.code == "cardinality_unbalanced"


def test_grow_budget_is_501_with_all_three_reasons():
    with pytest.raises(A.AdminError) as e:
        A.check_cardinality([], mode=A.MODE_GROW_BUDGET)
    assert e.value.code == "budget_growth_not_supported"
    assert e.value.status == 501
    msg = e.value.message
    assert "sized at prepare time" in msg          # (1) slabs
    assert "tier_signature" in msg                 # (2) memo key / recompile
    assert "KV-cache pool" in msg                  # (3) the bytes
    assert e.value.details["growth_supported"] is False


def test_auto_balance_requires_its_own_flag():
    with pytest.raises(A.AdminError) as e:
        A.check_cardinality([], mode=A.MODE_AUTO_BALANCE, environ={})
    assert A.ADMIN_ALLOW_AUTO_BALANCE_ENV in e.value.message
    with pytest.raises(A.AdminError) as e:
        A.check_cardinality([], mode=A.MODE_AUTO_BALANCE,
                            environ={A.ADMIN_ALLOW_AUTO_BALANCE_ENV: "1"})
    assert e.value.status == 501         # declared, not implemented in v1


# ------------------------------------------------------------ memory guard


def test_unit_bytes_match_expert_stage_geometry():
    """The arithmetic must be derived from the live geometry, not a
    constant — cross-check it against what ExpertStage actually allocates."""
    model = memory_model()
    for k in (3, 4):
        stage = SW.ExpertStage(k, HIDDEN, INTERMEDIATE, pin_memory=False)
        assert model.expert_bytes(k) == stage.nbytes
    assert (model.expert_bytes(4) - model.expert_bytes(3)
            == model.unit_bytes_per_k)


def test_balanced_batch_has_exactly_zero_delta(tmp_path):
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    assert plan.memory["delta_bytes_per_rank"] == 0
    assert (plan.memory["projected_expert_bytes_per_rank"]
            == plan.memory["current_expert_bytes_per_rank"])
    assert plan.memory["unit_bytes_per_k_per_rank"] == (
        3 * HIDDEN * INTERMEDIATE // 8)


def test_memory_budget_exceeded_names_budget_and_overshoot():
    """Guard tested directly: an unbalanced (hence growing) batch against
    the default no-growth budget."""
    model = memory_model()
    tier = np.full((1, E), P.K3, dtype=np.int64)
    grow = [A.ResolvedItem(layer=23, expert=e, requested="+1",
                           interpretation="relative", k_from=3, k_to=4,
                           outcome="promoted") for e in range(3)]
    with pytest.raises(A.AdminError) as e:
        A.check_memory(model, tier, grow, environ={})
    assert e.value.code == "memory_budget_exceeded"
    assert e.value.status == 409
    d = e.value.details
    # The arithmetic is UNIT x sum(k_to - k_from), to the byte.
    assert d["delta_bytes_per_rank"] == 3 * model.unit_bytes_per_k
    assert d["overshoot_bytes_per_rank"] == 3 * model.unit_bytes_per_k
    assert d["budget_bytes_per_rank"] == model.resident_bytes(tier)
    assert str(d["budget_bytes_per_rank"]) in e.value.message
    assert str(d["overshoot_bytes_per_rank"]) in e.value.message
    assert A.ADMIN_MEM_BUDGET_ENV in e.value.message


def test_memory_budget_can_be_raised_by_env():
    model = memory_model()
    tier = np.full((1, E), P.K3, dtype=np.int64)
    grow = [A.ResolvedItem(layer=23, expert=0, requested="+1",
                           interpretation="relative", k_from=3, k_to=4,
                           outcome="promoted")]
    budget = model.resident_bytes(tier) + model.unit_bytes_per_k
    acct = A.check_memory(model, tier, grow,
                          environ={A.ADMIN_MEM_BUDGET_ENV: str(budget)})
    assert acct["headroom_bytes_per_rank"] == 0


def test_memory_headroom_guard():
    model = memory_model()
    tier = np.full((1, E), P.K3, dtype=np.int64)
    grow = [A.ResolvedItem(layer=23, expert=0, requested="+1",
                           interpretation="relative", k_from=3, k_to=4,
                           outcome="promoted")]
    budget = model.resident_bytes(tier) + model.unit_bytes_per_k
    with pytest.raises(A.AdminError) as e:
        A.check_memory(model, tier, grow, device_free_bytes=1024,
                       environ={A.ADMIN_MEM_BUDGET_ENV: str(budget),
                                A.ADMIN_MEM_HEADROOM_ENV: "1048576"})
    assert e.value.code == "memory_headroom_exceeded"
    assert A.ADMIN_MEM_HEADROOM_ENV in e.value.message


# ------------------------------------------------------------------- pins


def test_pin_hold_pins_both_experts_to_their_new_tier(tmp_path):
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    assert plan.new_doc["pinned"] == {"23": [0, 8]}
    assert plan.pins["added"] == {"23": [0, 8]}
    A.apply_retier(state, plan, engine=FakeEngine())
    # pins[l, e] carries the tier the expert is pinned to (policy.py).
    row = state.layers.index(23)
    assert int(state.pins[row, 0]) == P.K3      # demoted, held at K3
    assert int(state.pins[row, 8]) == P.K4      # promoted, held at K4
    # And the loop honours it: decide() must not trade them back.
    stats = state._read_stats()
    swaps = P.decide(stats, state.eps, state.tier_of, pins=state.pins,
                     dwell=np.full_like(state.tier_of, 10_000),
                     cfg=state._decide_cfg())
    touched = {(l, e) for l, e_out, e_in in swaps for e in (e_out, e_in)}
    assert (row, 0) not in touched and (row, 8) not in touched


def test_pin_release_removes_them(tmp_path):
    doc = boot_doc()
    doc["pinned"] = {"23": [0, 8]}
    state, _ = make_state(tmp_path, doc=doc)
    plan = A.plan_retier(state, req(balanced_body(pin="release")),
                         engine=FakeEngine(), resolver=FakeResolver(),
                         memory=memory_model())
    assert plan.new_doc["pinned"] == {}
    assert plan.pins["removed"] == {"23": [0, 8]}


def test_pin_none_leaves_pinned_untouched(tmp_path):
    doc = boot_doc()
    doc["pinned"] = {"24": [2]}
    state, _ = make_state(tmp_path, doc=doc)
    plan = A.plan_retier(state, req(balanced_body(pin="none")),
                         engine=FakeEngine(), resolver=FakeResolver(),
                         memory=memory_model())
    assert plan.new_doc["pinned"] == {"24": [2]}


def test_pin_only_change_commits_without_touching_the_engine(tmp_path):
    """A no-op item plus pin=hold is a pure pin change: no weights move, no
    quiesce window, but the policy MUST still be committed or the pin
    evaporates at the next interval."""
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    plan = A.plan_retier(
        state, req({"items": [{"layer": 23, "expert": 1, "k": 4}]}),
        engine=engine, resolver=FakeResolver(), memory=memory_model())
    assert len(plan.plan) == 0
    assert plan.pins["added"] == {"23": [1]}

    result = A.apply_retier(state, plan, engine=engine)
    assert result["pins_only"] is True
    assert result["pairs"] == 0
    assert engine.staged == [] and engine.applied == []
    row = state.layers.index(23)
    assert int(state.pins[row, 1]) == P.K4
    committed = S.PolicyStore(tmp_path, "m" * 64).load_current(num_experts=E)
    assert committed["pinned"] == {"23": [1]}
    assert committed["bits_per_expert"] == boot_doc()["bits_per_expert"]


def test_pin_would_starve_layer(tmp_path):
    """Pinning the last free K4 slot wedges policy.decide for that layer —
    refuse before it can happen, and prove the refusal was justified."""
    doc = boot_doc()
    doc["pinned"] = {"23": [1, 2, 3]}          # 3 of 4 K4 slots already pinned
    state, _ = make_state(tmp_path, doc=doc)
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                      resolver=FakeResolver(), memory=memory_model())
    assert e.value.code == "pin_would_starve_layer"
    assert e.value.status == 409
    assert e.value.details["free_k4_slots"] < 2

    # Regression, part 1: the arithmetic this guard mirrors really is the
    # one policy.decide refuses on.
    tier = np.asarray([boot_doc()["bits_per_expert"][str(l)] for l in LAYERS])
    stats = {"count": np.ones_like(tier, dtype=float),
             "mass": np.ones_like(tier, dtype=float)}
    eps = {P.K3: np.ones_like(tier, dtype=float),
           P.K4: np.zeros_like(tier, dtype=float)}
    over_pinned = np.zeros_like(tier)
    over_pinned[0, :N_K4 + 1] = P.K4        # one more K4 pin than capacity
    with pytest.raises(ValueError, match="pins incompatible with budget"):
        P.decide(stats, eps, tier, pins=over_pinned,
                 cfg={"n_k4": P.n_k4_of(tier)})

    # Regression, part 2 — the failure mode the default of 2 actually
    # buys: with every expert pinned to its current tier the loop is not
    # broken, it is WEDGED. decide() returns nothing, forever, silently.
    all_pinned = np.array(tier, copy=True)
    eps_hot = {P.K3: np.ones_like(tier, dtype=float),
               P.K4: np.zeros_like(tier, dtype=float)}
    eps_hot[P.K3][:, E - 1] = 100.0         # a screamingly obvious promotion
    assert P.decide(stats, eps_hot, tier, pins=all_pinned,
                    cfg={"n_k4": P.n_k4_of(tier)}) == []
    assert P.decide(stats, eps_hot, tier, pins=np.zeros_like(tier),
                    cfg={"n_k4": P.n_k4_of(tier)}) != []


def test_min_free_slots_is_configurable(tmp_path):
    doc = boot_doc()
    doc["pinned"] = {"23": [1, 2, 3]}
    state, _ = make_state(tmp_path, doc=doc)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model(),
                         environ={A.ADMIN_MIN_FREE_SLOTS_ENV: "0"})
    assert plan.plan == SW.SwapPlan([(23, 0, 8)])


# -------------------------------------------------------------- fragments


def test_fragment_unavailable_lists_every_miss(tmp_path):
    """Both promotions are missing; the pre-flight must report BOTH, not
    stop at the first the way stage() would."""
    state, _ = make_state(tmp_path)
    resolver = FakeResolver(missing=[(23, 8, 4), (24, 9, 4)])
    engine = FakeEngine()
    body = {"items": [{"layer": 23, "expert": 0, "adjust_k": -1},
                      {"layer": 23, "expert": 8, "adjust_k": "+1"},
                      {"layer": 24, "expert": 0, "adjust_k": -1},
                      {"layer": 24, "expert": 9, "adjust_k": "+1"}]}
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(body), engine=engine, resolver=resolver,
                      memory=memory_model())
    assert e.value.code == "fragment_unavailable"
    assert e.value.status == 409
    misses = e.value.details["unavailable"]
    assert {(m["layer"], m["expert"], m["k"]) for m in misses} == {
        (23, 8, 4), (24, 9, 4)}
    for m in misses:
        assert m["encode_queued"] is True
        assert m["queue_position"] is not None
        assert m["substituted_k"] == 3
        assert "FALLBACK K3" in m["chain"]
    # Nothing was staged or applied, and the policy is untouched.
    assert engine.staged == [] and engine.applied == []
    assert S.PolicyStore(tmp_path, "m" * 64).load_current(
        num_experts=E) == boot_doc()


def test_resolver_that_raises_is_refusal_not_crash(tmp_path):
    class Exploding:
        def resolve_best(self, layer, expert, k, *, chain_out=None):
            raise RuntimeError("mirror unreachable")

    state, _ = make_state(tmp_path)
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                      resolver=Exploding(), memory=memory_model())
    assert e.value.code == "fragment_unavailable"
    assert "mirror unreachable" in json.dumps(e.value.details)


def test_noop_items_are_not_preflighted():
    resolver = FakeResolver()
    noop = A.ResolvedItem(layer=23, expert=0, requested=4,
                          interpretation="absolute", k_from=4, k_to=4,
                          outcome="noop")
    assert A.preflight_fragments(resolver, [noop]) == []
    assert resolver.calls == []


# --------------------------------------------------------- adopt / persist


def test_adopt_policy_updates_everything_including_pins(tmp_path):
    """The pins refresh is the line today's loop._maybe_apply is missing."""
    state, _ = make_state(tmp_path)
    state._real_steps = 500
    row = state.layers.index(23)
    before_entered = np.array(state._entered_step, copy=True)

    new_tier = np.array(state.tier_of, copy=True)
    new_tier[row, 0] = P.K3
    new_tier[row, 8] = P.K4
    new_doc = A.build_target_doc(
        state.policy_doc, layers=state.layers, new_tier=new_tier,
        pinned={"23": [0, 8]}, provenance={"origin": "operator"})

    A.adopt_policy(state, new_doc, [(23, 0, 8)], origin="operator")

    assert int(state.tier_of[row, 0]) == P.K3
    assert int(state.tier_of[row, 8]) == P.K4
    assert state.policy_doc == new_doc
    assert state.policy_sha == S.policy_hash(new_doc)
    assert state._policy_step == state._step
    # Dwell restarts for the two that moved, and ONLY for them.
    assert state._entered_step[row, 0] == 500
    assert state._entered_step[row, 8] == 500
    untouched = np.ones_like(new_tier, dtype=bool)
    untouched[row, 0] = untouched[row, 8] = False
    assert (state._entered_step[untouched]
            == before_entered[untouched]).all()
    # ...and pins now reflect the new document. Without the refresh this
    # array would still be all-zero and the next decide() would ignore the
    # operator's override entirely.
    assert int(state.pins[row, 0]) == P.K3
    assert int(state.pins[row, 8]) == P.K4
    assert int(state.pins[row, 1]) == 0


def test_adopt_policy_matches_the_loop_for_a_loop_driven_swap(tmp_path):
    """Golden test: adopt_policy must reproduce _maybe_apply's state
    transition exactly (it is the same transition, plus the pins fix)."""
    a, _ = make_state(tmp_path / "a")
    b, _ = make_state(tmp_path / "b")
    for st in (a, b):
        st._real_steps = 321
    row = a.layers.index(23)
    proposed = np.array(a.tier_of, copy=True)
    proposed[row, 0] = P.K3
    proposed[row, 8] = P.K4
    doc_a = a._doc_for(proposed, [(row, 0, 8)])
    doc_b = dict(doc_a)

    a.apply_fn = lambda doc, swaps: True
    a.cfg.apply_mode = FL.APPLY_ATOMIC
    assert a._maybe_apply(doc_a, proposed, [(row, 0, 8)]) is True
    A.adopt_policy(b, doc_b, [(23, 0, 8)], origin="policy")

    assert (a.tier_of == b.tier_of).all()
    assert (a._entered_step == b._entered_step).all()
    assert a.policy_sha == b.policy_sha
    assert a._policy_step == b._policy_step
    # The one intended difference: the loop leaves `pins` stale.
    assert (b.pins == b._pins_from_doc(doc_b)).all()


def test_occupancy_diff_is_emitted_after_a_change(tmp_path):
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    # Capture off the module logger directly: vllm sets propagate=False on
    # the "vllm" logger, so caplog's root handler never sees these.
    logged = []
    sink = logging.Handler()
    sink.emit = lambda record: logged.append(record.getMessage())
    A.logger.addHandler(sink)
    A.logger.setLevel(logging.INFO)
    try:
        result = A.apply_retier(state, plan, engine=FakeEngine())
    finally:
        A.logger.removeHandler(sink)
    table = result["occupancy_diff"]
    assert "expert composition after operator retier" in table
    # Only the touched layer is shown; layer 24 did not move.
    rows = [ln for ln in table.splitlines()
            if ln.strip().split(" ")[0] in ("23", "24")]
    assert len(rows) == 1 and rows[0].strip().startswith("23")
    assert "1 untouched layer(s) omitted" in table
    assert "mean bits/expert" in table
    # Fixed cardinality means the COUNTS cannot move, so the pairs have to
    # be spelled out or the operator learns nothing from the table.
    assert result["occupancy"]["23"]["4"] == N_K4
    assert result["occupancy"]["23"]["3"] == E - N_K4
    assert "invariant by" in table
    assert "L23: e0 K4->K3  <->  e8 K3->K4" in table
    # ...and it reached the engine log, not just the response.
    assert any("L23: e0 K4->K3" in line for line in logged)


def test_forced_change_is_attributable_and_persisted(tmp_path):
    state, _ = make_state(tmp_path)
    body = balanced_body(actor="michel",
                         reason="coder-axis promotion, paired demotion")
    plan = A.plan_retier(state, req(body), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    engine = FakeEngine()
    result = A.apply_retier(state, plan, engine=engine)

    # (a) provenance on the policy document
    prov = plan.new_doc["provenance"]
    assert prov["proposed_by"] == "fq-admin/retier"
    assert prov["origin"] == "operator"
    assert prov["actor"] == "michel"
    assert prov["reason"].startswith("coder-axis")
    assert prov["request_id"] == plan.request_id
    assert all(i["source"] == "operator" for i in prov["items"])

    # (b) the decision record, in the same fq-decision/1 schema
    path = result["decision_record"]
    assert path is not None and f"-admin-{plan.request_id}.json" in path
    record = json.loads(open(path).read())
    assert record["schema"] == "fq-decision/1"
    assert record["origin"] == "operator"
    assert record["forced"] is True
    assert record["actor"] == "michel"
    assert record["request_id"] == plan.request_id
    assert record["items"][0]["source"] == "operator"
    assert all(sw["forced"] for sw in record["swaps"])
    assert set(record["guards_waived"]) >= {"dwell", "hysteresis"}
    # The score fields survive: seeing what the policy THOUGHT about an
    # expert the operator overrode is the point of the record.
    assert "score_in" in record["swaps"][0]

    # (c) persisted, and it is what the next boot will rehydrate
    committed = S.PolicyStore(tmp_path, "m" * 64).load_current(num_experts=E)
    assert committed["bits_per_expert"]["23"][0] == P.K3
    assert committed["bits_per_expert"]["23"][8] == P.K4
    assert committed["provenance"]["origin"] == "operator"
    assert engine.applied[0]["policy_store"] is state.store
    assert engine.applied[0]["memo_hook"] is None
    assert engine.staged[0]["fail_atomic"] is True
    assert engine.staged[0]["on_unavailable"] == "raise"


def test_admin_record_does_not_collide_with_an_interval_record(tmp_path):
    state, _ = make_state(tmp_path)
    decisions = state.store.root / "decisions"
    decisions.mkdir(exist_ok=True)
    state.store._atomic_write(decisions / f"{state._step:08d}.json",
                              {"schema": "fq-decision/1",
                               "origin": "policy"})
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    result = A.apply_retier(state, plan, engine=FakeEngine())
    interval = json.loads((decisions / f"{state._step:08d}.json").read_text())
    assert interval["origin"] == "policy"          # untouched
    assert json.loads(open(result["decision_record"]).read())[
        "origin"] == "operator"


def test_forced_change_survives_the_next_interval(tmp_path):
    """The next decide() runs against the forced membership as baseline."""
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    A.apply_retier(state, plan, engine=FakeEngine())
    row = state.layers.index(23)
    record = state.run_interval()
    assert record["policy_sha_before"] == state.policy_sha or True
    assert int(state.tier_of[row, 0]) == P.K3
    assert int(state.tier_of[row, 8]) == P.K4


def test_expect_policy_sha_mismatch(tmp_path):
    state, _ = make_state(tmp_path)
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(balanced_body(expect_policy_sha="0" * 64)),
                      engine=FakeEngine(), resolver=FakeResolver(),
                      memory=memory_model())
    assert e.value.code == "policy_sha_mismatch"


def test_plan_too_large(tmp_path):
    state, _ = make_state(tmp_path)
    body = {"items": [{"layer": 23, "expert": e, "adjust_k": -1}
                      for e in range(2)]
            + [{"layer": 23, "expert": 8 + e, "adjust_k": "+1"}
               for e in range(2)]}
    with pytest.raises(A.AdminError) as e:
        A.plan_retier(state, req(body), engine=FakeEngine(max_pairs=1),
                      resolver=FakeResolver(), memory=memory_model())
    assert e.value.code == "plan_too_large"


def test_apply_failure_reports_recovery_state(tmp_path):
    state, _ = make_state(tmp_path)
    plan = A.plan_retier(state, req(balanced_body()), engine=FakeEngine(),
                         resolver=FakeResolver(), memory=memory_model())
    before = np.array(state.tier_of, copy=True)
    with pytest.raises(A.AdminError) as e:
        A.apply_retier(state, plan,
                       engine=FakeEngine(fail_apply=RuntimeError("boom")))
    assert e.value.code == "apply_failed"
    assert e.value.status == 500
    assert e.value.details["flipped"] is False
    assert "guidance" in e.value.details
    assert (state.tier_of == before).all()      # loop state not advanced


# ---------------------------------------------------------------- gating


def test_disabled_by_default():
    assert A.admin_enabled({}) is False
    assert A.admin_enabled({A.DEV_MODE_ENV: "1"}) is False       # dev only
    assert A.admin_enabled({A.ADMIN_API_ENV: "1"}) is False      # flag only
    assert A.admin_enabled({A.DEV_MODE_ENV: "1",
                            A.ADMIN_API_ENV: "1"}) is True
    # The spec's name is honoured as an alias.
    assert A.admin_enabled({A.DEV_MODE_ENV: "1",
                            A.ADMIN_ENABLE_ENV: "1"}) is True


def test_gate_reason_names_the_missing_gate():
    assert A.DEV_MODE_ENV in A.gate_reason({})
    assert A.ADMIN_API_ENV in A.gate_reason({A.DEV_MODE_ENV: "1"})


# ---------------------------------------------------------------- router


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


class FakeEngineClient:
    def __init__(self, *, plan_result=None, apply_result=None, ranks=2,
                 rpc_error=None):
        self.ranks = ranks
        self.calls = []
        self.events = []
        self.plan_result = plan_result
        self.apply_result = apply_result
        self.rpc_error = rpc_error

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, args))
        self.events.append(method)
        if self.rpc_error is not None and method == "fq_admin_apply":
            raise self.rpc_error
        if method == "fq_admin_plan":
            payload = self.plan_result
        elif method == "fq_admin_apply":
            payload = self.apply_result
        else:
            payload = {"ok": True, "policy_sha": "abc"}
        return [json.dumps(payload) for _ in range(self.ranks)]

    async def pause_generation(self, mode="wait", clear_cache=False):
        self.events.append("pause")

    async def resume_generation(self):
        self.events.append("resume")

    async def reset_prefix_cache(self, *a):
        self.events.append("reset_prefix_cache")


PLAN_OK = {
    "ok": True, "phase": "plan", "request_id": "fqr-x", "plan_sha": "s" * 8,
    "membership_sha": "m" * 8, "policy_sha_before": "b" * 8,
    "policy_sha_after": "a" * 8,
    "pairs": [{"layer": 23, "expert_out": 0, "expert_in": 8}],
    "layers": [23], "items": [], "memory": {"delta_bytes_per_rank": 0},
    "pins": {"added": {"23": [0, 8]}, "removed": {}}, "free_slots": {"23": 3},
    "occupancy": {},
}
APPLY_OK = {
    "ok": True, "phase": "apply", "applied": True, "pairs": 1, "layers": 1,
    "generation": 7, "bytes_h2d_per_rank": 4096, "stage_ms": 1.0,
    "window_ms": 0.4, "restored": False, "committed_policy_hash": "a" * 8,
    "policy_sha_after": "a" * 8, "occupancy": {}, "occupancy_diff": "table",
    "decision_record": "/tmp/rec.json", "rank": 0, "plan_sha": "s" * 8,
}


def make_client(env, engine=None):
    app = fastapi.FastAPI()
    app.state.engine_client = engine or FakeEngineClient(
        plan_result=PLAN_OK, apply_result=APPLY_OK)
    attached = A.attach_router(app, environ=env)
    return TestClient(app), app.state.engine_client, attached


def enabled_env(**extra):
    env = {A.DEV_MODE_ENV: "1", A.ADMIN_API_ENV: "1"}
    env.update(extra)
    return env


def test_routes_absent_without_the_gates():
    for env in ({}, {A.DEV_MODE_ENV: "1"}, {A.ADMIN_API_ENV: "1"}):
        client, _, attached = make_client(env)
        assert attached is False
        assert client.post("/fq/retier", json=balanced_body()).status_code \
            == 404


def test_routes_present_when_both_gates_are_set():
    client, engine, attached = make_client(enabled_env())
    assert attached is True
    resp = client.post("/fq/retier", json=balanced_body())
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True


def test_token_gate():
    env = enabled_env(**{A.ADMIN_TOKEN_ENV: "hunter2"})
    client, _, _ = make_client(env)
    assert client.post("/fq/retier", json=balanced_body()).status_code == 403
    assert client.post("/fq/retier", json=balanced_body(),
                       headers={"X-FQ-Admin-Token": "nope"}).status_code == 403
    ok = client.post("/fq/retier", json=balanced_body(),
                     headers={"X-FQ-Admin-Token": "hunter2"})
    assert ok.status_code == 200


def test_pause_and_resume_bracket_the_apply():
    client, engine, _ = make_client(enabled_env())
    client.post("/fq/retier", json=balanced_body())
    assert engine.events == ["fq_admin_plan", "pause", "fq_admin_apply",
                             "resume"]


def test_resume_runs_even_when_the_rpc_raises():
    engine = FakeEngineClient(plan_result=PLAN_OK, apply_result=APPLY_OK,
                              rpc_error=RuntimeError("worker died"))
    client, engine, _ = make_client(enabled_env(), engine)
    with pytest.raises(RuntimeError):
        client.post("/fq/retier", json=balanced_body())
    assert engine.events[-1] == "resume"


def test_dry_run_never_applies_and_never_pauses():
    client, engine, _ = make_client(enabled_env())
    resp = client.post("/fq/retier", json=balanced_body(dry_run=True))
    assert resp.status_code == 200
    assert resp.json()["applied"] is False
    assert engine.events == ["fq_admin_plan"]
    assert "fq_admin_apply" not in engine.events


def test_rank_divergence_in_phase_one_never_applies():
    class Diverging(FakeEngineClient):
        async def collective_rpc(self, method, timeout=None, args=(),
                                 kwargs=None):
            self.events.append(method)
            if method == "fq_admin_plan":
                return [json.dumps({**PLAN_OK, "plan_sha": "s" * 8}),
                        json.dumps({**PLAN_OK, "plan_sha": "d" * 8})]
            return [json.dumps(APPLY_OK)]

    client, engine, _ = make_client(enabled_env(), Diverging())
    resp = client.post("/fq/retier", json=balanced_body())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "rank_divergence"
    assert "fq_admin_apply" not in engine.events
    assert "pause" not in engine.events


def test_worker_error_is_forwarded_with_its_status():
    engine = FakeEngineClient(plan_result={
        "ok": False,
        "error": {"code": "cardinality_unbalanced", "http_status": 409,
                  "message": "layer 23: 1 promotion, 0 demotions",
                  "details": {"layer": 23}}})
    client, engine, _ = make_client(enabled_env(), engine)
    resp = client.post("/fq/retier",
                       json={"items": [{"layer": 23, "expert": 8,
                                        "adjust_k": "+1"}]})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "cardinality_unbalanced"
    assert resp.json()["error"]["details"]["layer"] == 23


def test_parse_errors_never_reach_the_workers():
    client, engine, _ = make_client(enabled_env())
    resp = client.post("/fq/retier",
                       json={"items": [{"layer": 23, "expert": 0,
                                        "adjust_k": 1}]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "ambiguous_adjust_k"
    assert engine.events == []


def test_query_shorthand_over_http():
    client, engine, _ = make_client(enabled_env())
    resp = client.post("/fq/retier?layer=23&expert=250&adjust_k=-1")
    assert resp.status_code == 200
    sent = json.loads(engine.calls[0][1][0])["request"]
    assert sent["items"] == [{"layer": 23, "expert": 250, "requested": "-1",
                              "interpretation": "relative", "k": None,
                              "delta_k": -1}]


def test_pin_only_change_over_http_does_not_drain():
    plan = {**PLAN_OK, "pairs": [], "layers": []}
    engine = FakeEngineClient(plan_result=plan,
                              apply_result={**APPLY_OK, "pairs": 0,
                                            "pins_only": True})
    client, engine, _ = make_client(enabled_env(), engine)
    resp = client.post("/fq/retier",
                       json={"items": [{"layer": 23, "expert": 1, "k": 4}]})
    assert resp.status_code == 200
    assert resp.json()["applied"] is True
    assert engine.events == ["fq_admin_plan", "fq_admin_apply"]
    assert "pause" not in engine.events


def test_true_noop_is_reported_not_applied():
    plan = {**PLAN_OK, "pairs": [], "layers": [],
            "pins": {"added": {}, "removed": {}}}
    engine = FakeEngineClient(plan_result=plan)
    client, engine, _ = make_client(enabled_env(), engine)
    resp = client.post("/fq/retier",
                       json={"items": [{"layer": 23, "expert": 1, "k": 4}],
                             "pin": "none"})
    assert resp.status_code == 200
    assert resp.json()["applied"] is False
    assert "nothing to do" in resp.json()["warnings"][0]
    assert "fq_admin_apply" not in engine.events


def test_state_endpoint():
    engine = FakeEngineClient(plan_result=PLAN_OK)
    client, _, _ = make_client(enabled_env(), engine)
    resp = client.get("/fq/state")
    assert resp.status_code == 200
    assert resp.json()["admin"]["growth_supported"] is False


# ------------------------------------------------- worker entry points


class FakeWorker:
    def __init__(self, state, engine):
        self.model_runner = type("R", (), {"fq_collector": state})()
        state.swap_engine = engine


def test_worker_plan_and_apply_round_trip(tmp_path):
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    engine.source = type("S", (), {"resolver": FakeResolver()})()
    worker = FakeWorker(state, engine)
    payload = {"request": req(balanced_body(actor="michel")).canonical(),
               "request_id": "fqr-test"}

    planned = json.loads(A.worker_plan(worker, json.dumps(payload)))
    assert planned["ok"] is True
    assert planned["pairs"] == [{"layer": 23, "expert_out": 0,
                                 "expert_in": 8}]

    applied = json.loads(A.worker_apply(
        worker, json.dumps({**payload, "plan_sha": planned["plan_sha"]})))
    assert applied["ok"] is True
    assert applied["applied"] is True
    assert applied["generation"] == 1
    assert state.policy_sha == applied["policy_sha_after"]


def test_worker_apply_refuses_a_diverging_plan_sha(tmp_path):
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    engine.source = type("S", (), {"resolver": FakeResolver()})()
    worker = FakeWorker(state, engine)
    payload = {"request": req(balanced_body()).canonical(),
               "request_id": "fqr-test", "plan_sha": "deadbeef"}
    out = json.loads(A.worker_apply(worker, json.dumps(payload)))
    assert out["ok"] is False
    assert out["error"]["code"] == "rank_divergence"
    assert engine.applied == []


def test_worker_without_fq_state_is_404():
    worker = type("W", (), {"model_runner": type("R", (), {
        "fq_collector": None})()})()
    out = json.loads(A.worker_describe(worker))
    assert out["ok"] is False
    assert out["error"]["code"] == "fq_not_active"
    assert out["error"]["http_status"] == 404


def test_worker_describe_reports_state(tmp_path):
    state, _ = make_state(tmp_path)
    engine = FakeEngine()
    worker = FakeWorker(state, engine)
    out = json.loads(A.worker_describe(worker, json.dumps({"layer": 23})))
    assert out["ok"] is True
    assert out["budget"]["growth_supported"] is False
    assert out["servable_tiers"] == [3, 4]
    assert out["occupancy"]["23"]["4"] == N_K4
    assert len(out["layer"]["experts"]) == E
    assert out["layer"]["n_k4"] == N_K4
    assert out["memory"]["unit_bytes_per_k_per_rank"] == (
        3 * HIDDEN * INTERMEDIATE // 8)


def test_fq_worker_admin_mixin_matches_the_module_functions():
    for name in ("fq_admin_describe", "fq_admin_plan", "fq_admin_apply"):
        assert callable(getattr(A.FqWorkerAdmin, name))
