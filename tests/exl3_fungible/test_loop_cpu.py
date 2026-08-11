# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the M2 loop state machine (exl3_fungible/loop.py).

Covers the contract the live wiring depends on: interval firing, dummy
step semantics, dryrun persists-but-does-not-apply, metric increments,
decision determinism, eps loading, env config, and boot-glue policy
resolution/rehydration. No GPU, no built vllm required.
"""
import json

import numpy as np
import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        loop as FL,
        policy as P,
        store as S,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
    _PACKAGE_IMPORT = True
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
    # occupancy_table must precede loop: loop imports it, and a stub package
    # missing the attribute falls through to the real vllm import chain.
    for _sub in ("policy", "stats", "store", "decision_log",
                 "occupancy_table", "loop"):
        _mod = _load(f"{_pkg_name}.{_sub}", _dir / f"{_sub}.py")
        setattr(_pkg, _sub, _mod)
    P, S, FL = _pkg.policy, _pkg.store, _pkg.loop
    FqStatsCollector = _pkg.stats.FqStatsCollector
    _PACKAGE_IMPORT = False

prometheus_client = pytest.importorskip("prometheus_client")

E = 8               # experts per layer
LAYERS = [3, 4]     # model layer ids (policy keys; collector bind ids)


@pytest.fixture(autouse=True)
def clean_fq_env(monkeypatch):
    for env in (FL.FQ_INTERVAL_ENV, FL.FQ_APPLY_MODE_ENV,
                FL.FQ_ARTIFACT_DIR_ENV, FL.FQ_POLICY_ENV, FL.FQ_EPS_ROOT_ENV,
                FL.FQ_CACHE_ROOT_ENV, FL.FQ_MAX_SWAPS_LAYER_ENV,
                FL.FQ_MAX_SWAPS_TOTAL_ENV, FL.FQ_DWELL_ENV,
                FL.FQ_HYSTERESIS_ENV, FL.FQ_JACCARD_FLOOR_ENV):
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


class FakeRouter:
    def __init__(self):
        self.capture_fn = None
        self.global_num_experts = E

    def set_capture_fn(self, fn):
        self.capture_fn = fn


def make_collector(window_stride=2):
    c = FqStatsCollector(E, window_len=8, window_stride=window_stride,
                         decay=0.9, device="cpu")
    routers = {}
    for lid in LAYERS:
        routers[lid] = FakeRouter()
        c.bind_router(lid, routers[lid])
    return c, routers


def boot_doc():
    bits = [4, 4, 3, 3, 3, 3, 3, 3]  # experts 0,1 at K4
    return {
        "schema": "fq-policy/2",
        "manifest": "m" * 64,
        "budget": {"mode": "fixed_cardinality",
                   "n_k4_per_layer": {str(lid): 2 for lid in LAYERS}},
        "bits_per_expert": {str(lid): list(bits) for lid in LAYERS},
        "pinned": {},
    }


def eps_hot4():
    """Expert 4 has a 10x error gap: the clear upgrade candidate."""
    e3 = np.full((2, E), 0.05)
    e3[:, 4] = 0.5
    return {P.K3: e3, P.K4: np.zeros((2, E))}


def route(routers, ids):
    t = torch.tensor(ids, dtype=torch.int64)
    for r in routers.values():
        r.capture_fn(t)


def make_state(tmp_path=None, *, interval=4, metrics=None, eps=None,
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
        collector, boot_doc(), config=cfg, eps=eps or eps_hot4(),
        store=store, metrics=metrics)
    return state, routers


def drive_hot_interval(state, routers, steps=None):
    """Route heavy traffic to expert 4 for one interval's worth of steps."""
    for _ in range(steps or state.cfg.interval_steps):
        route(routers, [[4, 4], [4, 1], [4, 0]])
        state.step()


# ------------------------------------------------------------ interval firing

def test_interval_fires_on_schedule(monkeypatch):
    state, routers = make_state(interval=4)
    fired = []
    monkeypatch.setattr(state, "run_interval", lambda: fired.append(state._step))
    for _ in range(11):
        route(routers, [[0, 1]])
        state.step()
    assert fired == [4, 8]


def test_interval_exception_does_not_propagate(monkeypatch):
    state, routers = make_state(interval=2)
    monkeypatch.setattr(state, "run_interval",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    for _ in range(4):
        route(routers, [[0]])
        state.step()  # must not raise


def test_dummy_steps_advance_interval_but_not_dwell():
    state, _ = make_state(interval=4)
    fired = []
    state.run_interval = lambda: fired.append(state._step)
    for _ in range(4):
        state.step(is_dummy=True)
    # Dummy steps keep rank lockstep: the interval counter advances and
    # the boundary fires even on a dummy step (EPLB semantics)...
    assert fired == [4]
    # ...but dwell time is real-step based and must not have advanced.
    assert state._real_steps == 0
    # And the collector discarded the (nonexistent) dummy stats.
    assert state.collector._windows_rolled == 0


# --------------------------------------------------- dryrun persist semantics

def test_dryrun_decides_persists_but_does_not_apply(tmp_path):
    state, routers = make_state(tmp_path, interval=4)
    boot_tier = state.tier_of.copy()
    drive_hot_interval(state, routers)

    # The hot expert was proposed into K4 in every layer...
    decisions = sorted((state.store.root / "decisions").glob("*.json"))
    assert len(decisions) == 1
    rec = json.loads(decisions[0].read_text())
    assert rec["apply_mode"] == "dryrun" and rec["applied"] is False
    assert rec["totals"]["executed"] == 2
    assert all(sw["expert_in"] == 4 for sw in rec["swaps"])
    assert rec["layer_ids"] == LAYERS

    # ...the proposal landed in history WITHOUT touching current.json...
    proposals = sorted((state.store.root / "history").glob("*-proposed.json"))
    assert len(proposals) == 1
    proposed = json.loads(proposals[0].read_text())
    assert proposed["bits_per_expert"]["3"][4] == 4
    assert proposed["provenance"]["proposed_by"] == "fq-loop/dryrun"
    current = json.loads((state.store.root / "current.json").read_text())
    assert current["bits_per_expert"] == boot_doc()["bits_per_expert"]

    # ...and the running membership was not modified.
    assert (state.tier_of == boot_tier).all()


def test_no_swap_interval_writes_decision_but_no_proposal(tmp_path):
    # Uniform eps + uniform traffic: no score separation, desired == cur.
    state, routers = make_state(tmp_path, interval=4,
                                eps=FL.uniform_eps_stub(2, E))
    for _ in range(4):
        route(routers, [[0, 1, 2, 3, 4, 5, 6, 7]])
        state.step()
    rec_files = sorted((state.store.root / "decisions").glob("*.json"))
    assert len(rec_files) == 1
    assert json.loads(rec_files[0].read_text())["totals"]["executed"] == 0
    assert not list((state.store.root / "history").glob("*-proposed.json"))


def test_jaccard_guard_holds_swaps(tmp_path):
    state, routers = make_state(tmp_path, interval=4, jaccard_floor=0.99)
    # Seed a previous desired set that the hot-4 interval will contradict.
    prev = np.zeros((2, E), dtype=bool)
    prev[:, 6] = prev[:, 7] = True
    state._prev_desired = prev
    drive_hot_interval(state, routers)
    rec = json.loads(next(
        (state.store.root / "decisions").glob("*.json")).read_text())
    assert rec["jaccard_held"] is True
    assert rec["jaccard"] < 0.99
    assert rec["totals"]["executed"] == 0
    assert not list((state.store.root / "history").glob("*-proposed.json"))


# ------------------------------------------------------------------- metrics

def sample(registry, name, labels=None):
    v = registry.get_sample_value(name, labels or {})
    return 0.0 if v is None else v


def test_metrics_increment(tmp_path):
    registry = prometheus_client.CollectorRegistry()
    metrics = FL.FqMetrics(registry=registry)
    state, routers = make_state(tmp_path, interval=4, metrics=metrics)

    # Boot: occupancy exported, counters materialized at 0.
    assert sample(registry, "fq_tier_occupancy",
                  {"layer": "3", "tier": "k4"}) == 2.0
    assert sample(registry, "fq_tier_occupancy",
                  {"layer": "3", "tier": "k3"}) == 6.0
    assert sample(registry, "fq_swap_proposals_total", {"layer": "3"}) == 0.0
    assert sample(registry, "fq_swaps_applied_total", {"layer": "3"}) == 0.0
    assert sample(registry, "fq_rollbacks_total") == 0.0
    # No apply backend is bound in dryrun, and the gauge must say so: an
    # operator seeing proposals climb needs one glance to know whether any of
    # them can become a change.
    assert sample(registry, "fq_apply_bound") == 0.0

    drive_hot_interval(state, routers)
    assert sample(registry, "fq_swap_proposals_total", {"layer": "3"}) == 1.0
    assert sample(registry, "fq_swap_proposals_total", {"layer": "4"}) == 1.0
    assert sample(registry, "fq_policy_age_steps") == 4.0
    # THE distinction. A live serve decided 64 swaps across 39 layers with no
    # apply backend bound and exported them under a single "fq_swaps_total",
    # so the dashboard read as a working fungible loop while nothing whatever
    # was installed. Proposals must never imply applications.
    assert sample(registry, "fq_swaps_applied_total", {"layer": "3"}) == 0.0
    assert sample(registry, "fq_swaps_applied_total", {"layer": "4"}) == 0.0

    drive_hot_interval(state, routers)
    assert sample(registry, "fq_swap_proposals_total", {"layer": "3"}) == 2.0
    assert sample(registry, "fq_swaps_applied_total", {"layer": "3"}) == 0.0
    assert sample(registry, "fq_jaccard") == 1.0  # same desired set again
    # Dryrun: occupancy tracks the RUNNING policy, which never moved.
    assert sample(registry, "fq_tier_occupancy",
                  {"layer": "3", "tier": "k4"}) == 2.0


def test_non_lead_rank_does_not_persist_or_export(tmp_path):
    registry = prometheus_client.CollectorRegistry()
    metrics = FL.FqMetrics(registry=registry)
    cfg = FL.FqLoopConfig(interval_steps=4, dwell_steps=0, jaccard_floor=0.0)
    collector, routers = make_collector()
    store = S.PolicyStore(tmp_path, "m" * 64)
    state = FL.FungibleQuantState(
        collector, boot_doc(), config=cfg, eps=eps_hot4(),
        store=store, metrics=metrics, rank=2, is_lead=False)
    drive_hot_interval(state, routers)
    assert not (store.root / "decisions").exists()
    assert sample(registry, "fq_swaps_total", {"layer": "3"}) == 0.0


# --------------------------------------------------------------- determinism

def test_same_inputs_same_swaps_and_sha(tmp_path):
    records = []
    for i in range(2):
        state, routers = make_state(tmp_path / str(i), interval=4)
        drive_hot_interval(state, routers)
        rec = json.loads(next(
            (state.store.root / "decisions").glob("*.json")).read_text())
        records.append(rec)
    assert records[0]["swaps"] == records[1]["swaps"]
    assert records[0]["decision_sha"] == records[1]["decision_sha"]
    assert records[0]["policy_sha_after"] == records[1]["policy_sha_after"]


def test_decision_sha_is_order_and_content_sensitive():
    assert (FL.FungibleQuantState.decision_sha([(0, 1, 4)])
            != FL.FungibleQuantState.decision_sha([(0, 2, 4)]))
    assert (FL.FungibleQuantState.decision_sha([(0, 1, 4), (1, 1, 4)])
            != FL.FungibleQuantState.decision_sha([(1, 1, 4), (0, 1, 4)]))


# ---------------------------------------------------------------- eps source

def write_done_jsons(root, layers=LAYERS, ks=(3, 4), num_experts=E):
    for k in ks:
        d = root / f"work-k{k}-tr3"
        d.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            # Same per-layer base error, scaled down 1/k: K4 < K3 always.
            rng = np.random.default_rng(layer)
            mse = (rng.uniform(0.01, 0.05, num_experts) / k).tolist()
            (d / f"layer-{layer:03d}.done.json").write_text(
                json.dumps({"layer": layer, "expert_rel_rt_mse": mse}))


def test_eps_loader_reads_done_jsons(tmp_path):
    write_done_jsons(tmp_path)
    eps = FL.load_eps_from_work_root(tmp_path, LAYERS, E)
    assert set(eps) == {3, 4}
    assert eps[3].shape == (2, E) and eps[4].shape == (2, E)
    # K4 must be strictly better (smaller eps) than K3 in this fixture.
    assert (eps[3] > eps[4]).all()
    # Row order == requested layer order.
    d3 = json.loads(
        (tmp_path / "work-k3-tr3" / "layer-003.done.json").read_text())
    assert eps[3][0].tolist() == pytest.approx(d3["expert_rel_rt_mse"])


def test_eps_loader_falls_back_on_missing(tmp_path):
    write_done_jsons(tmp_path, layers=[3])  # layer 4 missing
    assert FL.load_eps_from_work_root(tmp_path, LAYERS, E) is None
    stub = FL.uniform_eps_stub(2, E)
    assert (stub[P.K3] == 1.0).all() and (stub[P.K4] == 0.0).all()


# -------------------------------------------------------------------- config

def test_config_defaults_match_spec():
    cfg = FL.FqLoopConfig.from_env()
    assert cfg.interval_steps == 3000
    assert cfg.apply_mode == "dryrun"
    assert cfg.dwell_steps == 6000          # 2 x interval
    assert cfg.hysteresis == 1.25
    assert cfg.max_swaps_per_layer == 2
    assert cfg.max_swaps_total == 64
    assert cfg.jaccard_floor == 0.95


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv(FL.FQ_INTERVAL_ENV, "200")
    monkeypatch.setenv(FL.FQ_APPLY_MODE_ENV, "reload")
    monkeypatch.setenv(FL.FQ_HYSTERESIS_ENV, "1.5")
    monkeypatch.setenv(FL.FQ_MAX_SWAPS_LAYER_ENV, "1")
    cfg = FL.FqLoopConfig.from_env()
    assert cfg.interval_steps == 200
    assert cfg.apply_mode == "reload"
    assert cfg.dwell_steps == 400           # follows the interval override
    assert cfg.hysteresis == 1.5
    assert cfg.max_swaps_per_layer == 1
    with pytest.raises(ValueError):
        FL.FqLoopConfig(apply_mode="yolo")


def test_reload_mode_without_apply_fn_records_only(tmp_path):
    state, routers = make_state(tmp_path, interval=4, apply_mode="reload")
    boot_tier = state.tier_of.copy()
    drive_hot_interval(state, routers)
    rec = json.loads(next(
        (state.store.root / "decisions").glob("*.json")).read_text())
    assert rec["apply_mode"] == "reload" and rec["applied"] is False
    assert (state.tier_of == boot_tier).all()


def test_apply_fn_path_commits_and_resets_dwell(tmp_path):
    applied = []
    state, routers = make_state(tmp_path, interval=4, apply_mode="reload")
    state.apply_fn = lambda doc, swaps: applied.append((doc, swaps)) or True
    drive_hot_interval(state, routers)
    assert len(applied) == 1
    # Membership moved, dwell reset for swapped experts, store committed.
    assert state.tier_of[0, 4] == P.K4
    assert state._entered_step[0, 4] == state._real_steps
    assert state._entered_step[0, 2] == 0
    cur = json.loads((state.store.root / "current.json").read_text())
    assert cur["bits_per_expert"]["3"][4] == 4
    assert state.policy_sha == S.policy_hash(cur)


# ----------------------------------------------------------------- boot glue

def test_build_from_env_policy_path_and_rehydration(tmp_path, monkeypatch):
    policy_path = tmp_path / "boot-policy.json"
    policy_path.write_text(json.dumps(boot_doc()))
    monkeypatch.setenv(FL.FQ_POLICY_ENV, str(policy_path))
    monkeypatch.setenv(FL.FQ_CACHE_ROOT_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(FL.FQ_INTERVAL_ENV, "4")

    collector, _ = make_collector()
    state = FL.build_from_env(collector)
    assert state is not None
    assert state.layers == LAYERS
    assert (state.store.root / "current.json").exists()
    boot_sha = state.policy_sha

    # Second boot rehydrates the committed policy (D8), same identity.
    collector2, _ = make_collector()
    state2 = FL.build_from_env(collector2)
    assert state2.policy_sha == boot_sha
    # Non-lead ranks compute but never persist.
    state3 = FL.build_from_env(make_collector()[0], rank=1)
    assert state3.store is None and state3.is_lead is False


def test_build_from_env_synthesizes_from_artifact_dir(tmp_path, monkeypatch):
    art = tmp_path / "ckpt"
    art.mkdir()
    bitmap = {str(lid): {"bits_per_expert": [4, 4, 3, 3, 3, 3, 3, 3]}
              for lid in LAYERS}
    (art / "tier_bitmap.json").write_text(json.dumps(bitmap))
    (art / "MANIFEST.sha256").write_text("deadbeef  model.safetensors\n")
    monkeypatch.setenv(FL.FQ_ARTIFACT_DIR_ENV, str(art))
    monkeypatch.setenv(FL.FQ_CACHE_ROOT_ENV, str(tmp_path / "cache"))

    state = FL.build_from_env(make_collector()[0])
    assert state is not None
    assert state.n_k4.tolist() == [2, 2]
    assert state.policy_doc["schema"] == "fq-policy/2"


def test_build_from_env_without_policy_source_returns_none(monkeypatch,
                                                           tmp_path):
    monkeypatch.setenv(FL.FQ_CACHE_ROOT_ENV, str(tmp_path))
    assert FL.build_from_env(make_collector()[0]) is None


@pytest.mark.skipif(not _PACKAGE_IMPORT,
                    reason="integration seam needs the real package")
def test_maybe_init_fq_state_degrades_to_collector(monkeypatch, tmp_path):
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        integration as fq_integration,
    )

    class FakeMoERunner:
        def __init__(self, layer_id, router):
            self.layer_id, self.router = layer_id, router

    class FakeModel:
        def __init__(self, mods):
            self._m = mods

        def modules(self):
            return iter(self._m)

    class FakeRunner:
        def __init__(self, mods):
            self.model = FakeModel(mods)
            self.device = "cpu"

    monkeypatch.setattr(fq_integration, "_moe_module_types",
                        lambda: (FakeMoERunner, FakeRouter))
    monkeypatch.setattr(fq_integration, "_collector_cls",
                        lambda: FqStatsCollector)
    monkeypatch.setenv(fq_integration.FQ_ENABLE_ENV, "1")
    monkeypatch.setenv(FL.FQ_CACHE_ROOT_ENV, str(tmp_path))
    runner = FakeRunner([FakeMoERunner(lid, FakeRouter()) for lid in LAYERS])

    # No policy source: falls back to the bare collector (M1-only).
    got = fq_integration.maybe_init_fq_state(runner)
    assert isinstance(got, FqStatsCollector)

    # With a policy source: the full loop state.
    policy_path = tmp_path / "p.json"
    policy_path.write_text(json.dumps(boot_doc()))
    monkeypatch.setenv(FL.FQ_POLICY_ENV, str(policy_path))
    runner2 = FakeRunner([FakeMoERunner(lid, FakeRouter()) for lid in LAYERS])
    got2 = fq_integration.maybe_init_fq_state(runner2)
    assert isinstance(got2, FL.FungibleQuantState)
    got2.step(is_dummy=True)  # runner call contract holds


# ------------------------------------------------------- composition table
class _CaptureLog:
    """Capture loop.py's own logger.

    vLLM configures its loggers with propagate=False, so pytest's caplog
    (which listens on the root logger) sees nothing — the table was being
    emitted correctly while the assertion read an empty string.
    """

    def __init__(self):
        self.lines = []

    def __enter__(self):
        import logging

        class _H(logging.Handler):
            def __init__(self, sink):
                super().__init__()
                self.sink = sink

            def emit(self, record):
                self.sink.append(record.getMessage())

        self._h = _H(self.lines)
        self._logger = FL.logger
        self._prev = self._logger.level
        self._logger.addHandler(self._h)
        self._logger.setLevel("INFO")
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._h)
        self._logger.setLevel(self._prev)
        return False

    @property
    def text(self):
        return "\n".join(self.lines)


def test_composition_table_printed_at_startup(tmp_path):
    """The boot shape must be on record, in full, before anything moves."""
    with _CaptureLog() as cap:
        make_state(tmp_path)
    assert "expert composition at startup" in cap.text
    assert "mean bits/expert" in cap.text
    for layer in LAYERS:
        assert any(line.strip().startswith(str(layer))
                   for line in cap.text.splitlines()), f"layer {layer} missing"


def test_composition_table_periodic_is_diff_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FQ_TABLE_EVERY_INTERVALS", "1")
    state, routers = make_state(tmp_path, interval=4)
    with _CaptureLog() as cap:
        drive_hot_interval(state, routers)
    assert "expert composition @ interval" in cap.text
    # dryrun proposes without mutating membership, so the table must report
    # "nothing moved" explicitly rather than printing an empty table that
    # would read as broken telemetry
    assert ("no tier changes" in cap.text
            or "unchanged layers omitted" in cap.text)


def test_composition_table_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FQ_TABLE_EVERY_INTERVALS", "0")
    state, routers = make_state(tmp_path, interval=4)
    with _CaptureLog() as cap:
        drive_hot_interval(state, routers)
    assert "expert composition @ interval" not in cap.text


def test_composition_table_reflects_a_real_membership_change(tmp_path):
    """A mutated membership must show up as a signed per-tier delta."""
    state, _ = make_state(tmp_path)
    state.log_composition(title="before", diff_only=False)
    state.tier_of[0][5] = P.K4          # promote expert 5 in the first layer
    with _CaptureLog() as cap:
        state.log_composition(title="after", diff_only=True)
    assert "-1 K3" in cap.text and "+1 K4" in cap.text


def test_composition_table_failure_never_kills_the_serve(tmp_path,
                                                         monkeypatch):
    """Telemetry is not allowed to take down inference."""
    state, _ = make_state(tmp_path)
    monkeypatch.setattr(FL.OT, "render",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    state.log_composition(title="x", diff_only=False)  # must not raise
