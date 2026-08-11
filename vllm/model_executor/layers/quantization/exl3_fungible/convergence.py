# SPDX-License-Identifier: Apache-2.0
"""Opportunistic time-to-serve: boot on what is on disk, converge afterwards.

A progressive boot normally blocks until every expert can be served at the K
its policy asks for. That makes time-to-serve a function of the SLOWEST fetch,
and it turns a partially-primed cache into a long wait instead of an immediate
(slightly worse) model.

The observation this module implements: **every K is a valid quantisation of
the same expert.** A K2 fragment is not a broken K3 -- it is a coarser but
entirely servable version of the same weights. So if the primed cache holds a
complete set of experts at ANY mix of tiers within the operator's bounds, the
engine can start serving immediately and walk up to the intended posture at
runtime through the existing swap path.

Two invariants make that safe rather than reckless:

1. **Completeness, not adequacy, is the gate.** Serving with an expert missing
   is incoherent; serving it at K2 instead of K3 is merely worse. So a missing
   fragment at EVERY tier is a hard failure, while a fragment below target is
   a deficit to be repaid.
2. **A converging model must never be mistaken for a final one.** Anything
   reading these metrics has to be able to tell that the weights are still
   moving -- otherwise a benchmark run during convergence gets attributed to
   the intended configuration, which is exactly the kind of quietly-wrong
   number that discredits the whole approach.

This is OPT-IN (``VLLM_FQ_OPPORTUNISTIC=1``). The default remains: serve what
was asked for, or do not serve.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ConvergenceState",
    "Deficit",
    "ConvergencePlan",
    "opportunistic_enabled",
    "k_bounds",
]


def opportunistic_enabled(environ=None) -> bool:
    env = environ if environ is not None else os.environ
    return str(env.get("VLLM_FQ_OPPORTUNISTIC", "0")).strip().lower() in (
        "1", "true", "yes", "on")


def k_bounds(environ=None) -> tuple[int, int]:
    """Operator bounds on acceptable tiers, inclusive.

    A boot may substitute anything inside these; anything outside is refused
    even opportunistically. Defaults are deliberately wide (2..5, the tiers
    the format defines) so the gate is completeness, not taste.
    """
    env = environ if environ is not None else os.environ
    lo = int(env.get("VLLM_FQ_K_MIN", "2"))
    hi = int(env.get("VLLM_FQ_K_MAX", "5"))
    if lo > hi:
        raise ValueError(
            f"VLLM_FQ_K_MIN={lo} exceeds VLLM_FQ_K_MAX={hi}: no tier is "
            f"acceptable, so no boot is possible")
    return lo, hi


class ConvergenceState(str, Enum):
    """Never let a converging model be read as a finished one."""

    PRISTINE = "pristine"        # every expert already at target
    CONVERGING = "converging"    # serving, but below target somewhere
    CONVERGED = "converged"      # started below target, now repaid
    STALLED = "stalled"          # deficits remain and cannot be repaid

    @property
    def is_final(self) -> bool:
        return self in (ConvergenceState.PRISTINE, ConvergenceState.CONVERGED)


@dataclass(frozen=True)
class Deficit:
    """One expert serving below the tier its policy asked for."""

    layer: int
    expert: int
    actual_k: int
    target_k: int

    @property
    def gap(self) -> int:
        return self.target_k - self.actual_k

    def key(self) -> tuple[int, int]:
        return (self.layer, self.expert)


@dataclass
class ConvergencePlan:
    """The deficits a boot accepted, and the telemetry that admits to them.

    Thread-safe: the loader records from the weight-iterator thread while the
    convergence worker drains and a metrics scrape reads, all concurrently.
    """

    k_min: int = 2
    k_max: int = 5
    _deficits: dict[tuple[int, int], Deficit] = field(default_factory=dict)
    _missing: list[tuple[int, int]] = field(default_factory=list)
    _total_experts: int = 0
    _repaid: int = 0
    _failed: dict[tuple[int, int], int] = field(default_factory=dict)
    _started_dirty: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ---------------------------------------------------------------- record
    def observe(self, layer: int, expert: int, actual_k: int,
                target_k: int) -> None:
        """Record one expert as it was actually loaded."""
        with self._lock:
            self._total_experts += 1
            if actual_k < target_k:
                d = Deficit(layer, expert, actual_k, target_k)
                self._deficits[d.key()] = d
                # Latch it HERE. Deriving this lazily in the state getter
                # meant a plan whose deficits were all repaid before anything
                # read state reported PRISTINE -- erasing the fact that it
                # booted degraded, which is precisely the fact an operator
                # reading a dashboard afterwards needs.
                self._started_dirty = True

    def observe_missing(self, layer: int, expert: int) -> None:
        """No fragment at ANY tier. This is the one thing that is fatal."""
        with self._lock:
            self._total_experts += 1
            self._missing.append((layer, expert))

    def acceptable(self, k: int) -> bool:
        return self.k_min <= k <= self.k_max

    # ------------------------------------------------------------- the gate
    @property
    def complete(self) -> bool:
        """Every expert has SOME fragment. The only precondition for serving."""
        with self._lock:
            return not self._missing

    def missing(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._missing)

    def gate(self) -> None:
        """Raise unless the loaded set can coherently serve.

        Deliberately says nothing about tiers: below-target is the thing this
        whole module exists to permit.
        """
        with self._lock:
            if self._missing:
                head = self._missing[:8]
                raise ValueError(
                    f"opportunistic boot refused: {len(self._missing)} "
                    f"expert(s) have no fragment at ANY tier in "
                    f"[K{self.k_min}..K{self.k_max}], e.g. "
                    f"{['L%d/e%d' % le for le in head]}. A missing expert is "
                    f"incoherent, not merely coarse.")

    # ------------------------------------------------------------ the queue
    def pending(self, priority=None) -> list[Deficit]:
        """Deficits worth repaying, worst first.

        ``priority(deficit) -> float`` lets a caller order by something it
        knows and this module does not -- activation counts, for instance, so
        the experts that actually carry traffic converge first rather than
        whatever happened to sort low by layer index.
        """
        with self._lock:
            items = list(self._deficits.values())
        if priority is None:
            return sorted(items, key=lambda d: (-d.gap, d.layer, d.expert))
        return sorted(items, key=lambda d: (-float(priority(d)), -d.gap,
                                            d.layer, d.expert))

    def repay(self, layer: int, expert: int, new_k: int) -> bool:
        """Mark an expert as swapped up. True when it closed the deficit."""
        with self._lock:
            d = self._deficits.get((layer, expert))
            if d is None:
                return False
            if new_k >= d.target_k:
                del self._deficits[(layer, expert)]
                self._failed.pop((layer, expert), None)
                self._repaid += 1
                return True
            # partial progress up the ladder still counts
            self._deficits[(layer, expert)] = Deficit(
                layer, expert, new_k, d.target_k)
            return False

    def fail(self, layer: int, expert: int) -> int:
        """Record a repay attempt that could not be satisfied."""
        with self._lock:
            n = self._failed.get((layer, expert), 0) + 1
            self._failed[(layer, expert)] = n
            return n

    def give_up(self, max_attempts: int = 3) -> list[Deficit]:
        """Deficits that have failed enough times to call unrepayable."""
        with self._lock:
            return [d for key, d in self._deficits.items()
                    if self._failed.get(key, 0) >= max_attempts]

    # -------------------------------------------------------------- telemetry
    @property
    def state(self) -> ConvergenceState:
        with self._lock:
            if self._deficits:
                if self._failed and all(
                        self._failed.get(k, 0) >= 3 for k in self._deficits):
                    return ConvergenceState.STALLED
                return ConvergenceState.CONVERGING
            return (ConvergenceState.CONVERGED if self._started_dirty
                    else ConvergenceState.PRISTINE)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._deficits)

    @property
    def converged_ratio(self) -> float:
        """Fraction of experts at or above target. 1.0 only when truly done."""
        with self._lock:
            if not self._total_experts:
                return 1.0
            return 1.0 - len(self._deficits) / self._total_experts

    @property
    def drift_bits(self) -> int:
        """Total tier-steps still owed -- the headline 'how wrong is it now'.

        Zero exactly when the served weights match the requested posture.
        """
        with self._lock:
            return sum(d.gap for d in self._deficits.values())

    def snapshot(self) -> dict:
        """One dict for /fq/convergence, Prometheus and the boot log."""
        with self._lock:
            deficits = list(self._deficits.values())
            total = self._total_experts
            repaid = self._repaid
            missing = len(self._missing)
        state = self.state
        by_layer: dict[int, int] = {}
        for d in deficits:
            by_layer[d.layer] = by_layer.get(d.layer, 0) + 1
        return {
            "state": state.value,
            "is_final": state.is_final,
            "experts_total": total,
            "experts_pending": len(deficits),
            "experts_repaid": repaid,
            "experts_missing": missing,
            "converged_ratio": round(self.converged_ratio, 6),
            "drift_bits": self.drift_bits,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "layers_affected": len(by_layer),
            "worst_layers": sorted(by_layer.items(),
                                   key=lambda kv: -kv[1])[:8],
        }

    def describe(self) -> str:
        s = self.snapshot()
        if s["state"] == ConvergenceState.PRISTINE.value:
            return (f"FQ convergence: PRISTINE — all {s['experts_total']} "
                    f"experts at target")
        return (
            f"FQ convergence: {s['state'].upper()} — "
            f"{s['experts_pending']}/{s['experts_total']} experts below "
            f"target ({100 * s['converged_ratio']:.1f}% converged, "
            f"drift={s['drift_bits']} tier-steps across "
            f"{s['layers_affected']} layers). "
            f"SERVED WEIGHTS ARE NOT FINAL.")


class ConvergenceWorker:
    """Repays the deficits an opportunistic boot accepted.

    Deliberately takes ``swap_fn`` rather than reaching for the engine: the
    swap path already exists (M4 atomic swap / fq_reload), and the thing that
    was missing was never the mechanism but the BOOKKEEPING -- knowing which
    experts owe an upgrade, in what order, and when to stop trying.

    ``swap_fn(layer, expert, target_k) -> int | None`` returns the K actually
    installed (which may be an intermediate rung of the ladder), or None if it
    could not be satisfied right now.
    """

    def __init__(self, plan: ConvergencePlan, swap_fn, *, batch: int = 16,
                 max_attempts: int = 3, priority=None):
        self.plan = plan
        self.swap_fn = swap_fn
        self.batch = max(1, int(batch))
        self.max_attempts = max_attempts
        self.priority = priority

    def step(self) -> dict:
        """Repay up to ``batch`` deficits. Safe to call from a loop tick.

        Never raises: convergence is best-effort by construction, and an
        engine that is serving must not be brought down by an upgrade that
        could have been retried on the next tick.
        """
        repaid = failed = attempted = 0
        giving_up = {d.key() for d in self.plan.give_up(self.max_attempts)}
        for d in self.plan.pending(priority=self.priority):
            if attempted >= self.batch:
                break
            if d.key() in giving_up:
                continue          # already written off; do not spin on it
            attempted += 1
            try:
                got = self.swap_fn(d.layer, d.expert, d.target_k)
            except Exception:  # noqa: BLE001 — a serving engine must survive
                got = None
            if got is None:
                self.plan.fail(d.layer, d.expert)
                failed += 1
            elif self.plan.repay(d.layer, d.expert, int(got)):
                repaid += 1
            else:
                # climbed a rung but not to target: progress, still pending
                repaid += 0
        return {
            "attempted": attempted,
            "repaid": repaid,
            "failed": failed,
            "pending": self.plan.pending_count,
            "state": self.plan.state.value,
            "drift_bits": self.plan.drift_bits,
        }

    def done(self) -> bool:
        """True when there is nothing left worth attempting."""
        return self.plan.state.is_final or self.plan.state is (
            ConvergenceState.STALLED)
