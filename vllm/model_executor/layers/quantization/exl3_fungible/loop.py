# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant M2 loop: observe -> decide -> explain -> persist.

:class:`FungibleQuantState` is the interval state machine that turns the
M1 stats collector into the M2 shadow evaluator. The model runner calls
``state.step()`` once per engine step at the EPLB step sites (via
``integration.maybe_init_fq_state``, which wraps the collector); every
``VLLM_FQ_INTERVAL_STEPS`` steps it:

1. reads the decayed routing window from the collector (observe),
2. runs ``policy.decide`` on (stats, eps, current membership) (decide),
3. builds the explainable decision record and logs the EPLB-style
   interval + per-swap rationale lines (explain),
4. persists the decision JSON and — in ``dryrun`` mode — the PROPOSED
   policy document into the store's history WITHOUT touching
   ``current.json`` or any weights (persist),
5. bumps the Prometheus instruments of 03-testing-validation
   §Instrumentation (``fq_swaps_total{layer}``, ``fq_rollbacks_total``,
   ``fq_probe_kld``, ``fq_jaccard``, ``fq_policy_age_steps``, per-tier
   occupancy gauges).

Apply modes (01-artifacts-policy-stats.md §4): ``dryrun`` (M2's shipping
mode — decide + log only, permanently useful as a shadow evaluator),
``reload`` (delegates to a bound ``apply_fn``; the proven path is the M3
``fq_reload`` worker extension driven from the persisted proposal),
``atomic`` (M4 swap engine; not yet bound to live layers — treated as
record-only with a warning until that wiring lands).

Determinism: ``policy.decide`` is a pure function of its inputs; each
rank logs a T6-style ``decision sha`` over the swap list so cross-rank
agreement is auditable from the serve log alone.

Hot-path contract (PERFORMANCE.md): ``step()`` between intervals is a
few integer ops on top of ``collector.step()``; all tensor reads, numpy
work, file IO and metric updates happen only on interval boundaries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vllm.model_executor.layers.quantization.exl3_fungible import policy as P
from vllm.model_executor.layers.quantization.exl3_fungible import (
    occupancy_table as OT,
)
from vllm.model_executor.layers.quantization.exl3_fungible import (
    decision_log as DL,
)
from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
    FqStatsCollector,
)
from vllm.model_executor.layers.quantization.exl3_fungible.store import (
    PolicyStore,
    policy_hash,
    validate_policy,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ env knobs
# (01-artifacts-policy-stats.md §4; read at init, GG-style raw os.environ —
# same pattern as integration.py.)
FQ_INTERVAL_ENV = "VLLM_FQ_INTERVAL_STEPS"
FQ_APPLY_MODE_ENV = "VLLM_FQ_APPLY_MODE"
FQ_ARTIFACT_DIR_ENV = "VLLM_FQ_ARTIFACT_DIR"
FQ_POLICY_ENV = "VLLM_FQ_POLICY"
FQ_EPS_ROOT_ENV = "VLLM_FQ_EPS_ROOT"
FQ_CACHE_ROOT_ENV = "VLLM_FQ_CACHE_ROOT"
FQ_MAX_SWAPS_LAYER_ENV = "VLLM_FQ_MAX_SWAPS_LAYER"
FQ_MAX_SWAPS_TOTAL_ENV = "VLLM_FQ_MAX_SWAPS_TOTAL"
FQ_DWELL_ENV = "VLLM_FQ_DWELL_STEPS"
FQ_HYSTERESIS_ENV = "VLLM_FQ_HYSTERESIS"
FQ_JACCARD_FLOOR_ENV = "VLLM_FQ_JACCARD_FLOOR"
FQ_DUMP_STATS_ENV = "VLLM_FQ_DUMP_STATS"

APPLY_DRYRUN, APPLY_RELOAD, APPLY_ATOMIC = "dryrun", "reload", "atomic"


class FqLoopConfig:
    """Loop knobs resolved from the environment (defaults per spec §4)."""

    def __init__(
        self,
        *,
        interval_steps: int = 3000,
        apply_mode: str = APPLY_DRYRUN,
        dwell_steps: int | None = None,
        hysteresis: float = 1.25,
        max_swaps_per_layer: int = 2,
        max_swaps_total: int = 64,
        jaccard_floor: float = 0.95,
        artifact_dir: str | None = None,
        policy_path: str | None = None,
        eps_root: str | None = None,
        cache_root: str | None = None,
    ) -> None:
        if apply_mode not in (APPLY_DRYRUN, APPLY_RELOAD, APPLY_ATOMIC):
            raise ValueError(f"unknown {FQ_APPLY_MODE_ENV}: {apply_mode!r}")
        self.interval_steps = int(interval_steps)
        self.apply_mode = apply_mode
        # Spec default: DWELL = 2 x interval.
        self.dwell_steps = (2 * self.interval_steps
                            if dwell_steps is None else int(dwell_steps))
        self.hysteresis = float(hysteresis)
        self.max_swaps_per_layer = int(max_swaps_per_layer)
        self.max_swaps_total = int(max_swaps_total)
        self.jaccard_floor = float(jaccard_floor)
        self.artifact_dir = artifact_dir
        self.policy_path = policy_path
        self.eps_root = eps_root
        self.cache_root = cache_root or os.path.expanduser("~/.cache/vllm")

    @classmethod
    def from_env(cls) -> "FqLoopConfig":
        env = os.environ.get
        kwargs: dict[str, Any] = {}
        if env(FQ_INTERVAL_ENV):
            kwargs["interval_steps"] = int(env(FQ_INTERVAL_ENV))
        kwargs["apply_mode"] = env(FQ_APPLY_MODE_ENV, APPLY_DRYRUN)
        if env(FQ_DWELL_ENV):
            kwargs["dwell_steps"] = int(env(FQ_DWELL_ENV))
        if env(FQ_HYSTERESIS_ENV):
            kwargs["hysteresis"] = float(env(FQ_HYSTERESIS_ENV))
        if env(FQ_MAX_SWAPS_LAYER_ENV):
            kwargs["max_swaps_per_layer"] = int(env(FQ_MAX_SWAPS_LAYER_ENV))
        if env(FQ_MAX_SWAPS_TOTAL_ENV):
            kwargs["max_swaps_total"] = int(env(FQ_MAX_SWAPS_TOTAL_ENV))
        if env(FQ_JACCARD_FLOOR_ENV):
            kwargs["jaccard_floor"] = float(env(FQ_JACCARD_FLOOR_ENV))
        kwargs["artifact_dir"] = env(FQ_ARTIFACT_DIR_ENV)
        kwargs["policy_path"] = env(FQ_POLICY_ENV)
        kwargs["eps_root"] = env(FQ_EPS_ROOT_ENV)
        kwargs["cache_root"] = env(FQ_CACHE_ROOT_ENV)
        return cls(**kwargs)


# ------------------------------------------------------------------ eps source
def load_eps_from_work_root(
    work_root: str | Path,
    layers: list[int],
    num_experts: int,
    *,
    ks: tuple[int, int] = (P.K3, P.K4),
    dir_pattern: str = "work-k{k}-tr3",
) -> dict[int, np.ndarray] | None:
    """Per-expert eps from the encoder's per-layer done-JSONs.

    Minimal reimplementation of the ``tools/fq_eps.py`` parsing (research
    repo — not importable from here): ``<work_root>/work-k{k}-tr3/
    layer-NNN.done.json`` -> ``expert_rel_rt_mse`` (per-expert round-trip
    MSE). Returns ``{k: [L, E] float64}`` with rows ordered like
    ``layers``, or None when any layer/K is missing (caller falls back to
    the uniform stub).
    """
    root = Path(work_root)
    out: dict[int, np.ndarray] = {}
    for k in ks:
        rows = []
        for layer in layers:
            p = root / dir_pattern.format(k=k) / f"layer-{layer:03d}.done.json"
            if not p.is_file():
                logger.warning("FQ eps: missing %s — using uniform stub", p)
                return None
            try:
                doc = json.loads(p.read_text())
                mse = doc["expert_rel_rt_mse"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("FQ eps: unreadable %s (%s) — uniform stub",
                               p, e)
                return None
            if len(mse) != num_experts:
                logger.warning(
                    "FQ eps: %s has %d experts, expected %d — uniform stub",
                    p, len(mse), num_experts)
                return None
            rows.append(mse)
        out[k] = np.asarray(rows, dtype=np.float64)
    return out


def uniform_eps_stub(num_layers: int, num_experts: int) -> dict[int, np.ndarray]:
    """No eps data: unit K3->K4 error gap everywhere, so the score
    degrades to pure routing pressure (count x mass) instead of
    silently vanishing."""
    return {
        P.K3: np.ones((num_layers, num_experts), dtype=np.float64),
        P.K4: np.zeros((num_layers, num_experts), dtype=np.float64),
    }


# ------------------------------------------------------------------ metrics
class FqMetrics:
    """03-testing-validation §Instrumentation, prometheus_client-native
    (multiprocess-aware exactly like vllm's own metrics: with
    ``PROMETHEUS_MULTIPROC_DIR`` set, worker-process writes aggregate into
    the API server's ``/metrics`` via ``MultiProcessCollector``)."""

    def __init__(self, registry=None):
        import prometheus_client as pc
        kw = {} if registry is None else {"registry": registry}
        self.swaps_total = pc.Counter(
            "fq_swaps_total",
            "Fungible-quant expert tier swaps decided per MoE layer "
            "(proposals in dryrun mode, applied swaps otherwise).",
            ["layer"], **kw)
        self.rollbacks_total = pc.Counter(
            "fq_rollbacks_total",
            "Fungible-quant probe-triggered policy rollbacks.", **kw)
        self.probe_kld = pc.Gauge(
            "fq_probe_kld",
            "Held-out probe KL divergence at the last probe run.",
            multiprocess_mode="mostrecent", **kw)
        self.jaccard = pc.Gauge(
            "fq_jaccard",
            "Mean per-layer Jaccard similarity of the desired-K4 set "
            "between consecutive decision intervals (router-shift guard).",
            multiprocess_mode="mostrecent", **kw)
        self.policy_age_steps = pc.Gauge(
            "fq_policy_age_steps",
            "Engine steps since the running policy was committed.",
            multiprocess_mode="mostrecent", **kw)
        self.tier_occupancy = pc.Gauge(
            "fq_tier_occupancy",
            "Experts resident per (layer, K tier) under the running policy.",
            ["layer", "tier"], multiprocess_mode="mostrecent", **kw)
        # Materialize the counters at 0 so they are scrapeable before the
        # first increment (multiprocess files are created on first touch).
        self.rollbacks_total.inc(0)


_METRICS: FqMetrics | None = None


def get_metrics() -> FqMetrics | None:
    """Process-wide singleton; None when prometheus_client is absent."""
    global _METRICS
    if _METRICS is None:
        try:
            _METRICS = FqMetrics()
        except ImportError:
            logger.warning("prometheus_client unavailable — FQ metrics off")
    return _METRICS


# ------------------------------------------------------------------ the loop
class FungibleQuantState:
    """Observe -> decide -> explain -> persist, one cycle per interval.

    Args:
        collector: the bound M1 stats collector.
        policy_doc: the running ``fq-policy/2`` document (defines layer
            rows, membership, budget, pins).
        config: loop knobs (default: from env).
        eps: ``{3: [L,E], 4: [L,E]}`` per-expert error at each tier
            (default: loaded from ``config.eps_root``, else uniform stub).
        store: policy store for persistence (None disables persistence,
            e.g. on non-lead ranks).
        metrics: FqMetrics (None disables instrument updates).
        rank: this worker's rank, for the T6-style decide-sha log line.
        is_lead: only the lead rank persists and bumps metrics, so
            counters are not multiplied by world size.
        apply_fn: ``fn(proposed_doc, swaps) -> bool`` bound by an apply
            backend (M3 reload / M4 atomic). Required for any mode other
            than dryrun to actually apply.
    """

    def __init__(
        self,
        collector: FqStatsCollector,
        policy_doc: dict,
        *,
        config: FqLoopConfig | None = None,
        eps: dict[int, np.ndarray] | None = None,
        store: PolicyStore | None = None,
        metrics: FqMetrics | None = None,
        rank: int = 0,
        is_lead: bool = True,
        apply_fn: Callable[[dict, list], bool] | None = None,
    ) -> None:
        self.collector = collector
        self.cfg = config or FqLoopConfig.from_env()
        self.store = store
        self.metrics = metrics
        self.rank = int(rank)
        self.is_lead = bool(is_lead)
        self.apply_fn = apply_fn

        validate_policy(policy_doc, num_experts=collector.num_experts)
        self.policy_doc = policy_doc
        self.policy_sha = policy_hash(policy_doc)
        # Policy layer rows: sorted numeric layer ids from the document.
        self.layers = sorted(int(k) for k in policy_doc["bits_per_expert"])
        self.num_experts = collector.num_experts
        # Operator-facing composition table: remembered so each print
        # can show what moved since the last one.
        self._last_composition: dict | None = None
        self._table_every = int(
            os.environ.get('VLLM_FQ_TABLE_EVERY_INTERVALS', '10'))
        # Per-expert routing mass/count dump. The decision record keeps
        # only the swaps it chose; reconstructing WHICH experts the
        # traffic actually favoured -- e.g. to compare our selection
        # against a human-built mixed quant -- needs the raw ranking.
        self._dump_stats_path = os.environ.get(FQ_DUMP_STATS_ENV)
        self._collector_layer_map = self._map_collector_layers()

        self.tier_of = np.asarray(
            [policy_doc["bits_per_expert"][str(layer)]
             for layer in self.layers], dtype=np.int64)
        self.n_k4 = P.n_k4_of(self.tier_of)
        self.pins = self._pins_from_doc(policy_doc)

        # Step machinery. ``_step`` counts every engine step including
        # dummy ones (collector._step semantics: rank lockstep); dwell is
        # measured in real steps via per-expert entry marks.
        self._step = 0
        self._real_steps = 0
        self._entered_step = np.zeros_like(self.tier_of)
        self._policy_step = 0          # step at which policy_doc took effect
        self._intervals_run = 0
        self._prev_desired: np.ndarray | None = None  # [L,E] bool
        self._atomic_warned = False
        # Apply attempts the backend refused/crashed on. A missing fragment
        # at the required K is the expected cause: the incumbent tiering
        # stays live and the next interval retries.
        self.apply_failures = 0

        if eps is None:
            eps = self._resolve_eps()
        self.eps = eps

        if self.metrics is not None and self.is_lead:
            for layer in self.layers:
                self.metrics.swaps_total.labels(layer=str(layer)).inc(0)
            self._export_occupancy()

        # The composition the checkpoint actually booted with. Printed in
        # full (not diff-only): there is nothing to diff against yet, and an
        # operator needs the starting shape on record in the log so a later
        # diff can be interpreted without re-deriving it.
        self.log_composition(title="expert composition at startup",
                             diff_only=False)

    # ---------------------------------------------------------------- setup

    def _map_collector_layers(self) -> dict[int, int]:
        """policy layer id -> collector layer id.

        Identity when the id sets match (MoERunner.layer_id is the model
        layer index, as are the policy keys); positional (sorted order)
        with a warning otherwise.
        """
        cids = sorted(self.collector.count_buf)
        if not cids:
            # Tests may drive a collector before any bind; treat as
            # identity and let decayed() fail loudly if actually read.
            return {layer: layer for layer in self.layers}
        if set(cids) >= set(self.layers):
            return {layer: layer for layer in self.layers}
        if len(cids) == len(self.layers):
            logger.warning(
                "FQ loop: collector layer ids %s != policy layers %s — "
                "mapping positionally by sorted order", cids, self.layers)
            return dict(zip(self.layers, cids))
        raise ValueError(
            f"FQ loop: cannot map policy layers {self.layers} onto "
            f"collector layers {cids}")

    def _pins_from_doc(self, doc: dict) -> np.ndarray:
        """Minimal ``pinned`` semantics: ``{layer: "all" | [expert ids]}``
        pins the named experts to their *current* tier (v1's use case is
        pinning a whole layer, e.g. MTP-78, out of the swap set)."""
        pins = np.zeros_like(self.tier_of)
        for key, val in (doc.get("pinned") or {}).items():
            row = self.layers.index(int(key))
            experts = (range(self.num_experts) if val == "all" else
                       [int(e) for e in val])
            for e in experts:
                pins[row, e] = self.tier_of[row, e]
        return pins

    def _resolve_eps(self) -> dict[int, np.ndarray]:
        if self.cfg.eps_root:
            eps = load_eps_from_work_root(
                self.cfg.eps_root, self.layers, self.num_experts)
            if eps is not None:
                logger.info("FQ loop: eps loaded from %s (%d layers)",
                            self.cfg.eps_root, len(self.layers))
                return eps
        logger.info("FQ loop: no eps source — uniform stub (score degrades "
                    "to routing pressure)")
        return uniform_eps_stub(len(self.layers), self.num_experts)

    # ---------------------------------------------------------------- step

    def step(self, *, is_dummy: bool = False) -> None:
        """Advance one engine step. Same call contract as
        ``FqStatsCollector.step`` — the model runner needs no other API.

        Between intervals this is collector.step plus two integer
        increments and one modulo (hot-path contract)."""
        self.collector.step(is_dummy=is_dummy)
        self._step += 1
        if not is_dummy:
            self._real_steps += 1
        if self._step % self.cfg.interval_steps:
            return
        try:
            self.run_interval()
        except Exception:
            # A loop bug must never take the serve down; the collector
            # keeps recording and the next interval retries.
            logger.exception("FQ interval at step %d failed — continuing",
                             self._step)

    # ---------------------------------------------------------------- cycle

    def _read_stats(self) -> dict[str, np.ndarray]:
        counts, masses = [], []
        for layer in self.layers:
            c, m = self.collector.decayed(self._collector_layer_map[layer])
            counts.append(c.cpu().numpy())
            masses.append(m.cpu().numpy())
        return {"count": np.asarray(counts), "mass": np.asarray(masses)}

    def _decide_cfg(self) -> dict:
        return {
            "n_k4": self.n_k4,
            "hysteresis": self.cfg.hysteresis,
            "dwell_steps": self.cfg.dwell_steps,
            "max_swaps_per_layer": self.cfg.max_swaps_per_layer,
            "max_swaps_total": self.cfg.max_swaps_total,
        }

    def _desired_sets(self, stats: dict) -> np.ndarray:
        """[L,E] bool: per layer, the top-n_k4 experts by score (the
        router-shift signal of spec 0e, independent of guards)."""
        s = P.score(stats, self.eps)
        desired = np.zeros(s.shape, dtype=bool)
        for row in range(s.shape[0]):
            order = np.lexsort((np.arange(s.shape[1]), -s[row]))
            desired[row, order[: int(self.n_k4[row])]] = True
        return desired

    @staticmethod
    def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
        inter = (a & b).sum(axis=1).astype(np.float64)
        union = np.maximum((a | b).sum(axis=1), 1).astype(np.float64)
        return float((inter / union).mean())

    @staticmethod
    def decision_sha(swaps: list[tuple[int, int, int]]) -> str:
        """T6-style cross-rank agreement digest of the ordered swap list."""
        return hashlib.sha256(
            json.dumps(swaps, separators=(",", ":")).encode()).hexdigest()

    def run_interval(self) -> dict:
        """One observe->decide->explain->persist cycle; returns the
        decision record."""
        self._intervals_run += 1
        if (self._table_every > 0
                and self._intervals_run % self._table_every == 0):
            self.log_composition(
                title=f'expert composition @ interval {self._intervals_run}',
                diff_only=True)
        stats = self._read_stats()
        if self._dump_stats_path and self.rank in (0, None):
            self._dump_stats(stats)
        dwell = self._real_steps - self._entered_step

        swaps = P.decide(stats, self.eps, self.tier_of, pins=self.pins,
                         dwell=dwell, cfg=self._decide_cfg())

        # Router-shift guard (spec 0e / VLLM_FQ_JACCARD_FLOOR): if the
        # desired top set is churning faster than the floor allows, hold
        # this interval's proposals — the signal is not trustworthy yet.
        desired = self._desired_sets(stats)
        jac = (None if self._prev_desired is None
               else self._jaccard(desired, self._prev_desired))
        self._prev_desired = desired
        jaccard_held = jac is not None and jac < self.cfg.jaccard_floor
        if jaccard_held and swaps:
            logger.warning(
                "FQ interval step=%d: jaccard %.3f < floor %.3f — holding "
                "%d proposed swaps", self._step, jac,
                self.cfg.jaccard_floor, len(swaps))
            swaps = []

        proposed_tier = P.apply_swaps(self.tier_of, swaps)
        proposed_doc = self._doc_for(proposed_tier, swaps)
        proposed_sha = policy_hash(proposed_doc)

        record = DL.explain(
            stats, self.eps, self.tier_of, swaps, pins=self.pins,
            dwell=dwell, cfg=self._decide_cfg(), step=self._step,
            policy_sha_before=self.policy_sha,
            policy_sha_after=proposed_sha if swaps else self.policy_sha)
        record["apply_mode"] = self.cfg.apply_mode
        record["applied"] = False
        # decide()/explain() speak in row indices; record the row->model
        # layer id mapping so the persisted JSON is self-describing.
        record["layer_ids"] = [int(x) for x in self.layers]
        record["jaccard"] = jac
        record["jaccard_held"] = bool(jaccard_held)
        record["decision_sha"] = self.decision_sha(swaps)
        record["real_steps"] = int(self._real_steps)

        DL.log_decision(record)
        logger.info(
            "FQ decide rank=%d step=%d interval=%d swaps=%d sha=%s",
            self.rank, self._step, self._intervals_run, len(swaps),
            record["decision_sha"][:16])

        applied = self._maybe_apply(proposed_doc, proposed_tier, swaps)
        record["applied"] = applied
        record["apply_failures"] = int(self.apply_failures)

        if self.is_lead:
            self._persist(record, proposed_doc, swaps)
            self._export_metrics(record, swaps, jac)
        return record

    # ---------------------------------------------------------------- apply

    def _doc_for(self, tier: np.ndarray, swaps: list) -> dict:
        doc = {k: v for k, v in self.policy_doc.items()
               if k not in ("bits_per_expert", "provenance")}
        doc["bits_per_expert"] = {
            str(layer): [int(b) for b in tier[row]]
            for row, layer in enumerate(self.layers)}
        doc["provenance"] = {
            "proposed_by": f"fq-loop/{self.cfg.apply_mode}",
            "step": int(self._step),
            "base_policy": self.policy_sha,
            "num_swaps": len(swaps),
        }
        return doc

    def _maybe_apply(self, proposed_doc: dict, proposed_tier: np.ndarray,
                     swaps: list) -> bool:
        """dryrun: never. reload/atomic: only through a bound apply_fn
        (M3's fq_reload path / M4's swap engine); record-only otherwise.

        The apply backend is treated as untrusted: a fragment that is not
        available at the required K, an unreachable mirror, an aborted
        stage — anything it raises is caught here and downgraded to "not
        applied". The incumbent tiering stays live and authoritative
        (``tier_of``/``policy_doc``/``store`` are only advanced on a clean
        ``True``), the rest of the interval still explains and persists its
        decision, and the next interval retries. Staging is host-only and
        happens before the swap engine's quiesce window, so a supply
        failure cannot have left a layer half-updated."""
        if self.cfg.apply_mode == APPLY_DRYRUN or not swaps:
            return False
        if self.apply_fn is None:
            if self.cfg.apply_mode == APPLY_ATOMIC and not self._atomic_warned:
                logger.warning(
                    "FQ apply_mode=atomic without a bound apply_fn — "
                    "recording proposals only (M4 live wiring pending)")
                self._atomic_warned = True
            elif self.cfg.apply_mode == APPLY_RELOAD:
                logger.warning(
                    "FQ apply_mode=reload without a bound apply_fn — "
                    "recording proposal only; drive the proven M3 path "
                    "(fq_assemble + fq_reload swap) from the persisted "
                    "proposal JSON")
            return False
        try:
            ok = bool(self.apply_fn(proposed_doc, swaps))
        except Exception:  # noqa: BLE001 — a supply failure is not a crash
            self.apply_failures += 1
            logger.exception(
                "FQ apply failed at step %d (%d swaps) — keeping the "
                "incumbent tiering and retrying next interval "
                "(apply_failures=%d)",
                self._step, len(swaps), self.apply_failures)
            return False
        if ok:
            swapped = self.tier_of != proposed_tier
            self.tier_of = proposed_tier
            self._entered_step = np.where(
                swapped, self._real_steps, self._entered_step)
            self.policy_doc = proposed_doc
            self.policy_sha = policy_hash(proposed_doc)
            self._policy_step = self._step
            if self.store is not None and self.is_lead:
                self.store.commit(proposed_doc, num_experts=self.num_experts)
        return ok

    # ---------------------------------------------------------------- persist

    def _persist(self, record: dict, proposed_doc: dict, swaps: list) -> None:
        if self.store is None:
            return
        decisions = self.store.root / "decisions"
        decisions.mkdir(exist_ok=True)
        self.store._atomic_write(
            decisions / f"{self._step:08d}.json", record)
        if swaps and not record["applied"]:
            # The dryrun contract: the proposal lands in history/ for
            # audit and for the out-of-band M3 reload path, while
            # current.json (the running policy) stays untouched.
            validate_policy(proposed_doc, num_experts=self.num_experts)
            self.store._atomic_write(
                self.store.root / "history"
                / f"{self._step:08d}-proposed.json", proposed_doc)

    def _dump_stats(self, stats: dict) -> None:
        """Append one JSON line of per-expert routing signal.

        Rank 0 only and best-effort: this is analysis telemetry, and a
        full disk must not take down inference.

        ``mass_is_real`` says whether "mass" is REAL gate mass or just a
        copy of "count" (the collector aliases it when gate-weight
        capture is off). Without it the alias is only detectable by the
        arrays happening to be identical — which a uniform router would
        also produce.
        """
        try:
            rec = {
                "step": int(self._step),
                "interval": int(self._intervals_run),
                "layers": [int(x) for x in self.layers],
                "tier_of": self.tier_of.tolist(),
                "mass_is_real": bool(getattr(
                    self.collector, "mass_is_real", lambda: False)()),
            }
            for key in ("count", "mass"):
                arr = stats.get(key)
                if arr is not None:
                    rec[key] = np.asarray(arr).tolist()
            with open(self._dump_stats_path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        except Exception:  # noqa: BLE001
            logger.exception("FQ stats dump failed (continuing)")

    def _export_occupancy(self) -> None:
        for row, layer in enumerate(self.layers):
            for tier in (P.K3, P.K4):
                self.metrics.tier_occupancy.labels(
                    layer=str(layer), tier=f"k{tier}").set(
                        int((self.tier_of[row] == tier).sum()))

    def _occupancy_map(self) -> dict[int, dict[int, int]]:
        """``{layer: {k: expert_count}}`` for the operator-facing table."""
        out: dict[int, dict[int, int]] = {}
        for row, layer in enumerate(self.layers):
            counts: dict[int, int] = {}
            for tier in OT.TIERS:
                n = int((self.tier_of[row] == tier).sum())
                if n or tier in (P.K3, P.K4):
                    counts[tier] = n
            out[int(layer)] = counts
        return out

    def log_composition(self, *, title: str, diff_only: bool) -> None:
        """Print the layer x K-tier matrix to the engine log.

        A scrape target answers "what is the occupancy" but not "what changed
        since I last looked", and an operator cannot eyeball Prometheus at
        3am. Rank 0 only: all TP ranks hold the same policy, so N identical
        90-row tables would be pure noise.
        """
        if self.rank not in (0, None):
            return
        cur = self._occupancy_map()
        try:
            text = OT.render(cur, self._last_composition, title=title,
                             num_experts=int(self.num_experts),
                             diff_only=diff_only)
        except Exception:  # noqa: BLE001 - telemetry must never kill a serve
            logger.exception("FQ composition table failed to render")
            return
        for line in text.splitlines():
            logger.info("%s", line)
        self._last_composition = cur

    def _export_metrics(self, record: dict, swaps: list,
                        jac: float | None) -> None:
        if self.metrics is None:
            return
        for layer, _, _ in swaps:
            self.metrics.swaps_total.labels(
                layer=str(self.layers[layer])).inc()
        if jac is not None:
            self.metrics.jaccard.set(jac)
        self.metrics.policy_age_steps.set(self._step - self._policy_step)
        self._export_occupancy()


# ------------------------------------------------------------------ boot glue
def _synthesize_policy_from_artifact(artifact_dir: str) -> dict | None:
    """fq-policy/2 document from an assembled checkpoint's
    ``tier_bitmap.json`` (the live membership), manifest-keyed by the
    checkpoint's MANIFEST.sha256 digest."""
    root = Path(artifact_dir)
    bitmap_path = root / "tier_bitmap.json"
    if not bitmap_path.is_file():
        return None
    bitmap = json.loads(bitmap_path.read_text())
    bits = {str(layer): [int(b) for b in bitmap[layer]["bits_per_expert"]]
            for layer in sorted(bitmap, key=int)}
    manifest_path = root / "MANIFEST.sha256"
    manifest = (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_path.is_file()
                else hashlib.sha256(str(root).encode()).hexdigest())
    return {
        "schema": "fq-policy/2",
        "manifest": manifest,
        "budget": {
            "mode": "fixed_cardinality",
            "n_k4_per_layer": {
                layer: sum(b == P.K4 for b in row)
                for layer, row in bits.items()},
        },
        "bits_per_expert": bits,
        "pinned": {},
        "provenance": {"synthesized_from": str(bitmap_path)},
    }


def build_from_env(
    collector: FqStatsCollector,
    *,
    rank: int = 0,
) -> FungibleQuantState | None:
    """Construct the loop state for a live engine (integration.py seam).

    Policy resolution order: the store's committed ``current.json``
    (crash/restart rehydration, D8) > ``VLLM_FQ_POLICY`` document >
    synthesis from ``VLLM_FQ_ARTIFACT_DIR``'s tier bitmap. Returns None
    (M1 collector-only observability) when no policy source exists.
    """
    cfg = FqLoopConfig.from_env()

    doc: dict | None = None
    if cfg.policy_path:
        doc = json.loads(Path(cfg.policy_path).read_text())
    elif cfg.artifact_dir:
        doc = _synthesize_policy_from_artifact(cfg.artifact_dir)
    if doc is None:
        logger.warning(
            "FQ loop: no policy source (%s/%s unset or unreadable) — "
            "running collector-only", FQ_POLICY_ENV, FQ_ARTIFACT_DIR_ENV)
        return None

    is_lead = rank == 0
    store = PolicyStore(cfg.cache_root, doc["manifest"])
    committed = store.load_current(num_experts=collector.num_experts)
    if committed is not None:
        doc = committed
        logger.info("FQ loop: rehydrated committed policy %s",
                    policy_hash(doc)[:16])
    elif is_lead:
        store.commit(doc, num_experts=collector.num_experts)
        logger.info("FQ loop: boot policy %s committed as current",
                    policy_hash(doc)[:16])

    state = FungibleQuantState(
        collector, doc, config=cfg,
        store=store if is_lead else None,
        metrics=get_metrics() if is_lead else None,
        rank=rank, is_lead=is_lead)
    logger.info(
        "FQ loop: armed — mode=%s interval=%d dwell=%d hysteresis=%.2f "
        "caps=(%d/layer, %d total) layers=%s policy=%s",
        cfg.apply_mode, cfg.interval_steps, cfg.dwell_steps, cfg.hysteresis,
        cfg.max_swaps_per_layer, cfg.max_swaps_total, state.layers,
        state.policy_sha[:16])
    return state
