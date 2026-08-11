# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for REAL gate-mass capture — no GPU required.

Background: the collector always emitted a ``mass`` array, but nothing
in the live integration path ever bound a source of gate weights, so
``mass`` was aliased to ``count`` and a 75x256 live dump came back
byte-identical between the two. The policy is specified to score on
routing MASS (Σ gate weights), which ranks a confident route above a
marginal one — not on raw hit frequency.

These tests pin, on CPU:

* real mass differs from count when the gate weights are non-uniform,
* real mass equals count when every gate weight is 1.0,
* the padding sentinel (id == E) contributes to NEITHER,
* out-of-range ids never index past the accumulators,
* ``mass_is_real()`` is correct in both modes, including the downgrade
  when the runtime's router cannot hand over the weights,
* the ``mass_is_real`` flag reaches the ``VLLM_FQ_DUMP_STATS`` artifact.

The router side of the contract (``BaseRouter._select_experts`` passing
``topk_weights`` to capture fns tagged ``wants_topk_weights``) is
covered by ``test_base_router_gate_mass_cpu.py``.
"""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

_PKG = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
        / "layers" / "quantization" / "exl3_fungible")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        integration as fq_integration,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
except ImportError:  # standalone run against an env without built vllm._C
    FqStatsCollector = _load("fq_stats_gm", "stats.py").FqStatsCollector
    fq_integration = _load("fq_integration_gm", "integration.py")


E = 8


class FakeBaseRouter:
    """Duck-types the parts of ``BaseRouter`` the collector relies on.

    Crucially it reproduces the weight-passing contract: ``set_capture_fn``
    resolves ``capture_fn_wants_weights`` from the callback's
    ``wants_topk_weights`` tag, and ``route`` calls the callback with one
    or two arguments accordingly — the same branch as
    ``BaseRouter._select_experts``.
    """

    def __init__(self, global_num_experts: int = E):
        self.global_num_experts = global_num_experts
        self.capture_fn = None
        self.capture_fn_wants_weights = False

    def set_capture_fn(self, fn):
        self.capture_fn = fn
        self.capture_fn_wants_weights = bool(
            getattr(fn, "wants_topk_weights", False))

    def route(self, topk_ids, topk_weights):
        if self.capture_fn is None:
            return
        if self.capture_fn_wants_weights:
            self.capture_fn(topk_ids, topk_weights)
        else:
            self.capture_fn(topk_ids)


class LegacyRouter(FakeBaseRouter):
    """A runtime whose BaseRouter predates the weight-passing contract:
    it never exposes ``capture_fn_wants_weights`` and always calls the
    capture fn with ids only."""

    def __init__(self, global_num_experts: int = E):
        super().__init__(global_num_experts)
        del self.capture_fn_wants_weights

    def set_capture_fn(self, fn):
        self.capture_fn = fn

    def route(self, topk_ids, topk_weights):
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)


def make_collector(record_mass: bool, **kw) -> FqStatsCollector:
    kw.setdefault("device", "cpu")
    kw.setdefault("window_len", 4)
    kw.setdefault("window_stride", 1)
    kw.setdefault("decay", 1.0)
    return FqStatsCollector(E, record_mass=record_mass, **kw)


def bind(record_mass: bool, router=None, **kw):
    c = make_collector(record_mass, **kw)
    r = FakeBaseRouter() if router is None else router
    c.bind_router(0, r)
    return c, r


# ------------------------------------------------------- real mass recorded

def test_real_gate_mass_differs_from_count_for_nonuniform_weights():
    """The whole point: with non-uniform gate weights, mass must NOT be
    the hit histogram. Expert 1 is hit twice with tiny weights, expert 5
    once with a large one — count ranks 1 first, mass ranks 5 first."""
    c, r = bind(True)
    ids = torch.tensor([[1, 5], [1, 5]], dtype=torch.int64)
    w = torch.tensor([[0.05, 0.90], [0.05, 0.90]], dtype=torch.float32)
    r.route(ids, w)

    count = c.count_buf[0][:E]
    mass = c.mass_buf[0][:E]
    assert count[1].item() == pytest.approx(2.0)
    assert count[5].item() == pytest.approx(2.0)
    assert mass[1].item() == pytest.approx(0.10, abs=1e-6)
    assert mass[5].item() == pytest.approx(1.80, abs=1e-6)
    assert not torch.equal(count, mass), "mass is still aliased to count"
    assert c.mass_is_real(0) and c.mass_is_real()

    # And the ranking the policy reads actually flips.
    c.step()
    dc, dm = c.decayed(0)
    assert int(dc.argmax()) in (1, 5)          # tie on count
    assert int(dm.argmax()) == 5, "mass must prefer the confident route"


def test_real_gate_mass_equals_count_when_all_weights_are_one():
    """Degenerate check that the accumulation is a plain weighted sum:
    unit gate weights make real mass numerically equal the count."""
    c, r = bind(True)
    ids = torch.tensor([[0, 3], [3, 7]], dtype=torch.int64)
    r.route(ids, torch.ones_like(ids, dtype=torch.float32))

    assert torch.allclose(c.count_buf[0][:E], c.mass_buf[0][:E], atol=1e-6)
    assert c.count_buf[0][:E].tolist() == [1, 0, 0, 2, 0, 0, 0, 1]
    assert c.mass_is_real(0), "equal arrays here are real mass, not an alias"


def test_mass_accumulates_across_calls_and_windows():
    c, r = bind(True)
    w = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    ids = torch.tensor([[2, 4]], dtype=torch.int64)
    r.route(ids, w)
    r.route(ids, w)
    assert c.mass_buf[0][2].item() == pytest.approx(0.5, abs=1e-6)
    assert c.mass_buf[0][4].item() == pytest.approx(1.5, abs=1e-6)
    c.step()   # roll: accumulators -> window, then zeroed
    assert c.mass_buf[0].sum().item() == 0.0
    _, dm = c.decayed(0)
    assert dm[4].item() == pytest.approx(1.5, abs=1e-6)


# -------------------------------------------------------- sentinel and OOR

def test_padding_sentinel_contributes_to_neither_count_nor_mass():
    """id == E is the padding sentinel. torch.histc's last bin is CLOSED
    at max, so the count path bins over E+1 and slices; the mass path
    must drop it the same way (via the overflow slot), not fold it into
    expert E-1 or into a real expert."""
    c, r = bind(True)
    ids = torch.tensor([[3, E], [E, E]], dtype=torch.int64)
    w = torch.tensor([[0.4, 9.0], [9.0, 9.0]], dtype=torch.float32)
    r.route(ids, w)

    assert c.count_buf[0][:E].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert c.mass_buf[0][:E].tolist() == pytest.approx(
        [0, 0, 0, 0.4, 0, 0, 0, 0], abs=1e-6)
    assert c.mass_buf[0][E - 1].item() == 0.0, \
        "sentinel folded into the last real expert (closed-bin bug)"
    assert c.mass_buf[0][E].item() == pytest.approx(27.0, abs=1e-5), \
        "sentinel mass belongs in the overflow slot"
    # ... and the overflow slot never reaches the window or the policy.
    c.step()
    dc, dm = c.decayed(0)
    assert dc.shape[0] == E and dm.shape[0] == E
    assert dm.sum().item() == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize("bad", [-1, -1000, E, E + 1, E + 2, 300, 2 ** 20])
def test_out_of_range_ids_never_index_out_of_bounds(bad):
    """An OOR scatter_add_ is an illegal memory access that killed a live
    engine. Every OOR id must land in the overflow slot at index E, and
    the accumulators must stay exactly E+1 long."""
    c, r = bind(True)
    ids = torch.tensor([[0, bad]], dtype=torch.int64)
    w = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
    r.route(ids, w)

    assert c.count_buf[0].numel() == E + 1
    assert c.mass_buf[0].numel() == E + 1
    assert c.count_buf[0][:E].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert c.mass_buf[0][0].item() == pytest.approx(0.3, abs=1e-6)
    assert c.mass_buf[0][:E].sum().item() == pytest.approx(0.3, abs=1e-6)
    assert c.mass_buf[0][E].item() == pytest.approx(0.7, abs=1e-6)


def test_weighted_path_index_tensor_is_always_in_bounds():
    """Directly assert the property the guard exists for: the index fed
    to scatter_add_ is within [0, E] for arbitrary garbage ids."""
    c, r = bind(True)
    ids = torch.tensor(
        [[-2 ** 31, -1, 0, E - 1, E, E + 1, 2 ** 31 - 1, 12345]],
        dtype=torch.int64)
    w = torch.full_like(ids, 1, dtype=torch.float32)
    r.route(ids, w)          # would raise / corrupt memory if unguarded
    total = c.mass_buf[0].sum().item()
    assert total == pytest.approx(8.0, abs=1e-6), "no weight lost"
    assert c.mass_buf[0][:E].sum().item() == pytest.approx(2.0, abs=1e-6)
    assert c.mass_buf[0][E].item() == pytest.approx(6.0, abs=1e-6)


def test_count_is_identical_with_and_without_mass_recording():
    """Enabling gate mass must not perturb the count signal — both modes
    run the same histc — so an A/B of two runs stays comparable."""
    ids = torch.tensor([[0, 1, E, -3], [1, 1, 7, 300]], dtype=torch.int64)
    w = torch.rand(ids.shape, dtype=torch.float32)

    c_off, r_off = bind(False)
    r_off.route(ids, w)
    c_on, r_on = bind(True)
    r_on.route(ids, w)

    assert c_off.count_buf[0].tolist() == c_on.count_buf[0].tolist()


# --------------------------------------------------------- explicit flagging

def test_mass_is_real_flag_is_correct_in_both_modes():
    c_off, _ = bind(False)
    assert c_off.mass_is_real(0) is False
    assert c_off.mass_is_real() is False

    c_on, _ = bind(True)
    assert c_on.mass_is_real(0) is True
    assert c_on.mass_is_real() is True

    # Unbound layer is not "real"; a mixed model is not model-wide real.
    assert c_on.mass_is_real(99) is False
    c_on.bind_router(1, FakeBaseRouter(), record_mass=False)
    assert c_on.mass_is_real(0) is True
    assert c_on.mass_is_real(1) is False
    assert c_on.mass_is_real() is False, "one aliased layer taints the model"

    assert FqStatsCollector(E, device="cpu").mass_is_real() is False, \
        "nothing bound yet is not a claim of real mass"


def test_summary_carries_the_mass_is_real_flag():
    c_off, r_off = bind(False)
    r_off.route(torch.tensor([[1, 1]]), torch.tensor([[0.2, 0.8]]))
    c_off.step()
    s_off = c_off.summary()
    assert s_off["mass_is_real"] is False
    assert s_off["layers"][0]["mass_is_real"] is False
    assert s_off["layers"][0]["mass"] == s_off["layers"][0]["count"]

    c_on, r_on = bind(True)
    r_on.route(torch.tensor([[1, 1]]), torch.tensor([[0.2, 0.8]]))
    c_on.step()
    s_on = c_on.summary()
    assert s_on["mass_is_real"] is True
    assert s_on["layers"][0]["mass_is_real"] is True
    assert s_on["layers"][0]["mass"] != s_on["layers"][0]["count"]


def test_capture_fn_is_tagged_so_the_router_knows_to_pass_weights():
    _, r_on = bind(True)
    assert r_on.capture_fn.wants_topk_weights is True
    assert r_on.capture_fn_wants_weights is True
    _, r_off = bind(False)
    assert r_off.capture_fn.wants_topk_weights is False
    assert r_off.capture_fn_wants_weights is False


def test_unsupported_router_downgrades_instead_of_lying():
    """If the deployed runtime's BaseRouter cannot hand over the weights,
    the capture fn would be called with ids only and mass would stay at
    zero. Detect that at bind time and fall back to count-only, so
    mass_is_real() never claims mass we are not getting."""
    c, r = bind(True, router=LegacyRouter())
    assert c.mass_is_real(0) is False
    assert getattr(r.capture_fn, "wants_topk_weights", False) is False

    r.route(torch.tensor([[2, 2]], dtype=torch.int64),
            torch.tensor([[0.1, 0.9]]))
    assert c.count_buf[0][2].item() == pytest.approx(2.0)
    assert c.mass_buf[0].sum().item() == 0.0, "no real mass was collected"
    c.step()
    dc, dm = c.decayed(0)
    assert dm.tolist() == dc.tolist(), "aliased, and flagged as aliased"


def test_chained_previous_capture_fn_still_fires_in_weighted_mode():
    seen = []
    r = FakeBaseRouter()
    r.capture_fn = lambda ids: seen.append(tuple(ids.shape))
    c = make_collector(True)
    c.bind_router(0, r)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    r.route(ids, torch.tensor([[0.5, 0.5]]))
    assert seen == [(1, 2)], "previous capture fn must still fire"
    assert c.mass_buf[0][1].item() == pytest.approx(0.5, abs=1e-6)


# ------------------------------------------------------------- integration

def test_integration_hook_opts_in_from_env(monkeypatch):
    """VLLM_FQ_GATE_MASS=1 turns real mass on end-to-end; default is off."""

    class FakeMoERunner:
        def __init__(self, layer_id, router):
            self.layer_id = layer_id
            self.router = router

    class FakeModel:
        def __init__(self, mods):
            self._mods = mods

        def modules(self):
            return iter(self._mods)

    class FakeRunner:
        def __init__(self, mods):
            self.model = FakeModel(mods)
            self.device = "cpu"

    def runner():
        return FakeRunner([FakeMoERunner(0, FakeBaseRouter()),
                           FakeMoERunner(1, FakeBaseRouter())])

    monkeypatch.setattr(fq_integration, "_moe_module_types",
                        lambda: (FakeMoERunner, FakeBaseRouter))
    monkeypatch.setattr(fq_integration, "_collector_cls",
                        lambda: FqStatsCollector)
    monkeypatch.setenv(fq_integration.FQ_ENABLE_ENV, "1")

    monkeypatch.delenv(fq_integration.FQ_GATE_MASS_ENV, raising=False)
    c = fq_integration.maybe_init_fq_collector(runner())
    assert c is not None and c.mass_is_real() is False

    monkeypatch.setenv(fq_integration.FQ_GATE_MASS_ENV, "1")
    c = fq_integration.maybe_init_fq_collector(runner())
    assert c.mass_is_real() is True
    assert c.mass_is_real(0) and c.mass_is_real(1)


def _load_loop_module():
    """Load loop.py, which imports its siblings under the canonical
    package path. Registered and then UNREGISTERED so test module import
    order cannot change how the other test files resolve the package."""
    try:
        from vllm.model_executor.layers.quantization.exl3_fungible import (
            loop as fq_loop,
        )
        return fq_loop, lambda: None
    except ImportError:
        pass

    import sys
    import types

    name = "vllm.model_executor.layers.quantization.exl3_fungible"
    saved = {k: v for k, v in sys.modules.items() if k.startswith(name)}
    pkg = types.ModuleType(name)
    sys.modules[name] = pkg
    # occupancy_table must precede loop: loop imports it, and a stub
    # package missing the attribute falls through to the real vllm chain.
    for sub in ("policy", "stats", "store", "decision_log",
                "occupancy_table", "loop"):
        spec = importlib.util.spec_from_file_location(
            f"{name}.{sub}", _PKG / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{sub}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, sub, mod)

    def restore():
        for k in [k for k in sys.modules if k.startswith(name)]:
            del sys.modules[k]
        sys.modules.update(saved)

    return pkg.loop, restore


def test_dump_stats_records_whether_mass_is_real(tmp_path):
    """The VLLM_FQ_DUMP_STATS artifact must say which kind of mass it
    holds — the live 75x256 dump was byte-identical between count and
    mass and nothing in the file said why."""
    pytest.importorskip("prometheus_client")
    fq_loop, restore = _load_loop_module()
    try:
        _check_dump_stats(fq_loop, tmp_path)
    finally:
        restore()


def _check_dump_stats(fq_loop, tmp_path):

    class Stub:
        """Minimal stand-in for FungibleQuantState._dump_stats' self."""

        def __init__(self, collector, path):
            self.collector = collector
            self._dump_stats_path = str(path)
            self._step = 7
            self._intervals_run = 1
            self.layers = [0]
            self.tier_of = fq_loop.np.asarray([3])

    stats = {"count": [[1.0, 2.0]], "mass": [[0.5, 1.5]]}

    for record_mass, expected in ((False, False), (True, True)):
        c, _ = bind(record_mass)
        path = tmp_path / f"dump-{record_mass}.jsonl"
        fq_loop.FungibleQuantState._dump_stats(Stub(c, path), stats)
        rec = json.loads(path.read_text().strip())
        assert rec["mass_is_real"] is expected
        assert "count" in rec and "mass" in rec, "both stay available"
