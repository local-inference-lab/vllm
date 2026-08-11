# SPDX-License-Identifier: Apache-2.0
"""Opportunistic time-to-serve, and the honesty guarantees around it.

The feature trades initial quality for time-to-serve: boot on whatever tiers
the primed cache holds, then converge to the requested posture at runtime.
That is only defensible if two things hold, and these tests pin both:

  * a MISSING expert is fatal while a BELOW-TARGET expert is not, and
  * a converging model can never be read as a finished one.

The second matters most. A benchmark run mid-convergence that gets attributed
to the intended configuration is exactly the kind of quietly-wrong number that
would discredit the approach.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        convergence as CV,
    )
except ImportError:  # standalone: load by path
    import importlib.util

    _dir = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")
    _spec = importlib.util.spec_from_file_location(
        "fq_convergence_standalone", _dir / "convergence.py")
    CV = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = CV
    _spec.loader.exec_module(CV)


def _plan(**kw):
    return CV.ConvergencePlan(**kw)


# ------------------------------------------------------------------- opt-in
@pytest.mark.parametrize("val,on", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("no", False),
])
def test_opportunistic_is_opt_in(val, on):
    assert CV.opportunistic_enabled({"VLLM_FQ_OPPORTUNISTIC": val}) is on


def test_default_is_off():
    """Serving a knowingly-degraded model must never be the default."""
    assert CV.opportunistic_enabled({}) is False


def test_k_bounds_default_span_the_format():
    assert CV.k_bounds({}) == (2, 5)


def test_k_bounds_are_honoured():
    assert CV.k_bounds({"VLLM_FQ_K_MIN": "3", "VLLM_FQ_K_MAX": "4"}) == (3, 4)


def test_inverted_bounds_are_rejected_loudly():
    """min>max accepts nothing; failing at parse beats an inscrutable
    'no fragment acceptable' at expert 12,000."""
    with pytest.raises(ValueError, match="no tier is acceptable"):
        CV.k_bounds({"VLLM_FQ_K_MIN": "5", "VLLM_FQ_K_MAX": "3"})


def test_acceptable_respects_bounds():
    p = _plan(k_min=3, k_max=4)
    assert [p.acceptable(k) for k in (2, 3, 4, 5)] == [False, True, True, False]


# -------------------------------------------------------------- the gate
def test_below_target_is_servable():
    """The whole premise: K2 is a coarser version of the same expert, not a
    broken one."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    assert p.complete
    p.gate()                       # must not raise


def test_missing_expert_is_fatal():
    """Serving without an expert is incoherent, not merely coarse."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    p.observe_missing(3, 1)
    assert not p.complete
    with pytest.raises(ValueError, match="no fragment at ANY tier"):
        p.gate()


def test_gate_message_names_examples_and_bounds():
    p = _plan(k_min=3, k_max=4)
    for e in range(12):
        p.observe_missing(7, e)
    with pytest.raises(ValueError) as ei:
        p.gate()
    msg = str(ei.value)
    assert "12 expert(s)" in msg and "K3..K4" in msg and "L7/e0" in msg


def test_a_fully_satisfied_boot_needs_no_convergence():
    p = _plan()
    for e in range(8):
        p.observe(3, e, actual_k=3, target_k=3)
    assert p.pending_count == 0
    assert p.state is CV.ConvergenceState.PRISTINE
    assert p.state.is_final


# ------------------------------------------------------- honesty of the state
def test_converging_is_not_final():
    """The guarantee that keeps a mid-convergence benchmark from being
    attributed to the intended configuration."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=4)
    assert p.state is CV.ConvergenceState.CONVERGING
    assert p.state.is_final is False
    assert "NOT FINAL" in p.describe()


def test_converged_is_distinguishable_from_pristine():
    """Both are final, but 'started degraded and repaid' is a different fact
    from 'was correct all along', and an operator reading a dashboard after
    the fact deserves to know which happened."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    assert p.state is CV.ConvergenceState.CONVERGING
    p.repay(3, 0, new_k=3)
    assert p.state is CV.ConvergenceState.CONVERGED
    assert p.state.is_final
    assert p.state is not CV.ConvergenceState.PRISTINE


def test_ratio_reaches_one_only_when_actually_done():
    p = _plan()
    for e in range(4):
        p.observe(3, e, actual_k=2, target_k=3)
    assert p.converged_ratio == 0.0
    p.repay(3, 0, new_k=3)
    p.repay(3, 1, new_k=3)
    assert p.converged_ratio == 0.5
    p.repay(3, 2, new_k=3)
    p.repay(3, 3, new_k=3)
    assert p.converged_ratio == 1.0


def test_drift_bits_is_zero_exactly_when_posture_matches():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=4)     # owes 2
    p.observe(3, 1, actual_k=3, target_k=4)     # owes 1
    assert p.drift_bits == 3
    p.repay(3, 0, new_k=4)
    assert p.drift_bits == 1
    p.repay(3, 1, new_k=4)
    assert p.drift_bits == 0


def test_partial_climb_up_the_ladder_reduces_drift_without_closing():
    """K2 -> K3 against a K4 target is progress, not completion."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=4)
    assert p.repay(3, 0, new_k=3) is False
    assert p.pending_count == 1
    assert p.drift_bits == 1
    assert p.repay(3, 0, new_k=4) is True
    assert p.pending_count == 0


def test_overshoot_closes_the_deficit():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    assert p.repay(3, 0, new_k=5) is True


def test_repaying_an_unknown_expert_is_a_noop():
    assert _plan().repay(9, 9, new_k=5) is False


# ------------------------------------------------------------------ the queue
def test_worst_deficit_is_repaid_first():
    p = _plan()
    p.observe(3, 0, actual_k=3, target_k=4)     # gap 1
    p.observe(4, 1, actual_k=2, target_k=5)     # gap 3
    p.observe(5, 2, actual_k=2, target_k=3)     # gap 1
    assert [d.key() for d in p.pending()][0] == (4, 1)


def test_priority_hook_lets_traffic_drive_convergence():
    """Layer index is a poor proxy for importance; activation counts are the
    thing the caller knows and this module does not."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    p.observe(9, 7, actual_k=2, target_k=3)
    hot = {(9, 7): 5000.0, (3, 0): 1.0}
    assert [d.key() for d in p.pending(priority=lambda d: hot[d.key()])][0] \
        == (9, 7)


def test_pending_is_a_copy_not_the_live_dict():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    got = p.pending()
    got.clear()
    assert p.pending_count == 1


# -------------------------------------------------------------- unrepayable
def test_repeated_failures_become_stalled_not_silently_converging():
    """If the fetch cannot be satisfied -- the segment does not exist yet, the
    Hub is unreachable -- the operator must not see 'converging' forever."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    for _ in range(3):
        p.fail(3, 0)
    assert p.state is CV.ConvergenceState.STALLED
    assert p.state.is_final is False
    assert p.give_up() and p.give_up()[0].key() == (3, 0)


def test_one_failure_is_not_a_stall():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    p.fail(3, 0)
    assert p.state is CV.ConvergenceState.CONVERGING


def test_a_successful_repay_clears_the_failure_record():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    p.fail(3, 0)
    p.repay(3, 0, new_k=3)
    assert p.give_up() == []
    assert p.state is CV.ConvergenceState.CONVERGED


# --------------------------------------------------------------- the snapshot
def test_snapshot_carries_everything_a_scrape_needs():
    p = _plan(k_min=2, k_max=4)
    p.observe(3, 0, actual_k=2, target_k=4)
    p.observe(3, 1, actual_k=3, target_k=4)
    p.observe(9, 2, actual_k=4, target_k=4)
    s = p.snapshot()
    assert s["state"] == "converging" and s["is_final"] is False
    assert s["experts_total"] == 3 and s["experts_pending"] == 2
    assert s["drift_bits"] == 3
    assert s["k_min"] == 2 and s["k_max"] == 4
    assert s["layers_affected"] == 1
    assert s["worst_layers"][0] == (3, 2)


def test_snapshot_is_final_true_only_when_final():
    p = _plan()
    p.observe(3, 0, actual_k=3, target_k=3)
    assert p.snapshot()["is_final"] is True


def test_empty_plan_is_pristine_and_final():
    p = _plan()
    assert p.converged_ratio == 1.0
    assert p.state is CV.ConvergenceState.PRISTINE
    assert p.snapshot()["is_final"] is True


# ------------------------------------------------------------- thread safety
def test_concurrent_observe_and_repay_do_not_corrupt_counts():
    """The loader records from the weight-iterator thread while the
    convergence worker drains and a scrape reads."""
    import threading
    p = _plan()
    n = 200

    def writer():
        for e in range(n):
            p.observe(3, e, actual_k=2, target_k=3)

    def reader():
        for _ in range(n):
            p.snapshot()
            _ = p.drift_bits

    ts = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert p.pending_count == n
    for e in range(n):
        p.repay(3, e, new_k=3)
    assert p.pending_count == 0 and p.converged_ratio == 1.0


# --------------------------------------------------- loader integration guards
def _progressive_src() -> str:
    return (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible"
            / "progressive.py").read_text()


def test_loader_records_every_expert_not_just_substitutions():
    """converged_ratio needs the DENOMINATOR. Recording only substitutions
    would make a 90%-degraded boot and a 1%-degraded boot both read as
    'pending N' with no scale."""
    src = _progressive_src()
    assert "_plan.observe(layer, expert, actual_k, int(k))" in src


def test_loader_reports_missing_experts_to_the_plan():
    assert "_plan.observe_missing(layer, expert)" in _progressive_src()


def test_gate_runs_once_after_all_layers_not_per_layer():
    """Gating inside the layer loop would fail on the FIRST layer with a
    missing expert and report that layer's count as the whole extent."""
    src = _progressive_src()
    # the CALL, not a comment that mentions it
    calls = [ln for ln in src.splitlines()
             if ln.strip().startswith("_plan.gate()")]
    assert len(calls) == 1, f"expected exactly one gate call, got {calls}"
    indent = len(calls[0]) - len(calls[0].lstrip())
    # the per-layer loop body sits deeper than this; 12 is the post-loop level
    assert indent <= 12, f"gate is inside the layer loop (indent {indent})"


def test_opportunistic_is_off_unless_env_says_otherwise():
    src = _progressive_src()
    assert "opportunistic_enabled()" in src
    assert "_plan = None" in src


def test_missing_convergence_module_does_not_break_the_loader():
    """The loader must still work if this module is absent (older rootfs)."""
    src = _progressive_src()
    assert "except ImportError" in src


# ------------------------------------------------------------- the repay loop
def test_worker_repays_deficits_through_the_injected_swap():
    p = _plan()
    for e in range(3):
        p.observe(3, e, actual_k=2, target_k=3)
    calls = []

    def swap(layer, expert, target_k):
        calls.append((layer, expert, target_k))
        return target_k

    w = CV.ConvergenceWorker(p, swap)
    out = w.step()
    assert out["repaid"] == 3 and out["pending"] == 0
    assert p.state is CV.ConvergenceState.CONVERGED
    assert len(calls) == 3


def test_worker_respects_the_batch_so_a_tick_is_bounded():
    """19,200 deficits must not be attempted inside one engine tick."""
    p = _plan()
    for e in range(100):
        p.observe(3, e, actual_k=2, target_k=3)
    out = CV.ConvergenceWorker(p, lambda *a: 3, batch=16).step()
    assert out["attempted"] == 16 and out["pending"] == 84


def test_a_swap_that_raises_does_not_bring_down_a_serving_engine():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)

    def boom(*a):
        raise RuntimeError("fetch exploded")

    out = CV.ConvergenceWorker(p, boom).step()
    assert out["failed"] == 1 and out["repaid"] == 0
    assert p.pending_count == 1          # still owed, will retry


def test_worker_stops_spinning_on_written_off_deficits():
    """A segment that does not exist yet must not be retried every tick
    forever -- that is a busy loop against the Hub."""
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    w = CV.ConvergenceWorker(p, lambda *a: None, max_attempts=3)
    for _ in range(3):
        w.step()
    assert p.state is CV.ConvergenceState.STALLED
    before = p.snapshot()
    assert w.step()["attempted"] == 0, "kept retrying a written-off deficit"
    assert p.snapshot()["experts_pending"] == before["experts_pending"]


def test_partial_climb_is_kept_not_counted_as_repaid():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=4)
    out = CV.ConvergenceWorker(p, lambda l, e, k: 3).step()
    assert out["repaid"] == 0
    assert p.pending_count == 1 and p.drift_bits == 1


def test_worker_reports_drift_so_a_scrape_sees_convergence_moving():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=4)
    p.observe(3, 1, actual_k=2, target_k=4)
    w = CV.ConvergenceWorker(p, lambda l, e, k: k, batch=1)
    assert w.step()["drift_bits"] == 2
    assert w.step()["drift_bits"] == 0


def test_done_is_false_while_work_remains():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    w = CV.ConvergenceWorker(p, lambda *a: 3)
    assert w.done() is False
    w.step()
    assert w.done() is True


def test_priority_reaches_the_worker():
    p = _plan()
    p.observe(3, 0, actual_k=2, target_k=3)
    p.observe(9, 7, actual_k=2, target_k=3)
    order = []
    CV.ConvergenceWorker(
        p, lambda l, e, k: (order.append((l, e)), k)[1], batch=1,
        priority=lambda d: 100.0 if d.key() == (9, 7) else 1.0).step()
    assert order == [(9, 7)]
