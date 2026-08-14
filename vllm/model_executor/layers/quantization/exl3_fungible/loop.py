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

Memory budget (``VLLM_FQ_MEMORY_BUDGET``): the fixed-cardinality budget
is only a PROXY for memory. A byte ceiling — absolute, a fraction of
device memory, or the equivalent experts/bpw per layer — is resolved at
init, enforced as Guard 5 when proposing, and reported as headroom on
``fq_memory_budget_bytes`` / ``fq_memory_used_bytes`` /
``fq_promotions_headroom`` plus the composition table. Every byte figure
is computed from the ACTUAL tier occupancy using per-expert sizes read
off the loaded checkpoint's tensor shapes; see ``policy.ExpertBytes``.

Hot-path contract (PERFORMANCE.md): ``step()`` between intervals is a
few integer ops on top of ``collector.step()``; all tensor reads, numpy
work, file IO and metric updates happen only on interval boundaries.
The budget adds nothing between intervals: the byte model is built once
at init (one safetensors HEADER read, no payload), and per-interval
accounting is one reduction over the [L,E] tier array plus integer
arithmetic per candidate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
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
# Ceiling on the per-device expert pool. Accepts an absolute size
# ("78g", "80000000000"), a fraction of device memory ("0.80", "80%")
# mirroring --gpu-memory-utilization, or the cardinality spellings of the
# same ceiling ("24/layer", "3.5bpw") — all resolved to bytes.
# Unset -> no byte ceiling; the fixed-cardinality budget alone applies.
FQ_MEMORY_BUDGET_ENV = "VLLM_FQ_MEMORY_BUDGET"
# Operator override for the per-expert byte model, as measured K points:
# "k3=3542028,k4=4721676" (per rank). Only needed when the checkpoint's
# real geometry cannot be read at boot.
FQ_EXPERT_BYTES_ENV = "VLLM_FQ_EXPERT_BYTES"

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
        memory_budget: str | int | float | None = None,
        expert_bytes: str | None = None,
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
        # Kept as the raw spec: resolving a fraction needs the device,
        # which is not knowable at config-parse time.
        self.memory_budget = memory_budget
        self.expert_bytes = expert_bytes

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
        kwargs["memory_budget"] = env(FQ_MEMORY_BUDGET_ENV)
        kwargs["expert_bytes"] = env(FQ_EXPERT_BYTES_ENV)
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


# ------------------------------------------------------------ memory budget
# The byte budget is only as honest as the per-expert byte figures it is
# computed from, so this section is written to prefer REAL geometry from
# the loaded checkpoint over any constant, and to state which one it got.

_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.rank(\d+)\.(\w+)$")


def device_total_bytes() -> int | None:
    """Total bytes of the current CUDA device, or None off-GPU.

    Monkeypatch seam for CPU tests; a fractional ``VLLM_FQ_MEMORY_BUDGET``
    resolves against this.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[1])
    except Exception:  # noqa: BLE001 — telemetry-grade probe, never fatal
        logger.debug("FQ budget: device memory unreadable", exc_info=True)
        return None


def _read_safetensors_header(path: Path) -> dict:
    """Header dict of a safetensors file (8-byte length prefix + JSON).

    Inlined rather than imported from ``fragments`` to keep this module's
    import graph unchanged; it is 4 lines and the format is frozen.
    """
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(hlen))


def expert_bytes_from_checkpoint(artifact_dir: str | Path
                                 ) -> "P.ExpertBytes | None":
    """Derive the per-expert byte model from the checkpoint's REAL shapes.

    Reads one assembled layer's safetensors header (header only — no
    payload, no allocation) and takes the tensor table of a single
    (expert, rank). Byte counts come from each tensor's own
    ``data_offsets``, so they are the true spans, and the bitrate is read
    off the trellis last dim (``16 * K``). Returns None when the artifact
    dir has no readable layer file — the caller then has to say out loud
    that it is falling back to a reference constant.
    """
    root = Path(artifact_dir)
    for path in sorted(root.glob("model-layer-*.safetensors"))[:4]:
        try:
            header = _read_safetensors_header(path)
        except Exception:  # noqa: BLE001
            logger.debug("FQ budget: unreadable header %s", path, exc_info=True)
            continue
        best: tuple[int, int] | None = None
        ranks: set[int] = set()
        for name in header:
            m = _EXPERT_TENSOR_RE.match(name)
            if m is None:
                continue
            expert, rank = int(m.group(2)), int(m.group(4))
            ranks.add(rank)
            if best is None or (expert, rank) < best:
                best = (expert, rank)
        if best is None:
            continue
        expert, rank = best
        entries = []
        for name, t in header.items():
            m = _EXPERT_TENSOR_RE.match(name)
            if m is None or int(m.group(2)) != expert or int(m.group(4)) != rank:
                continue
            lo, hi = t["data_offsets"]
            entries.append((name, int(hi) - int(lo), tuple(t["shape"])))
        try:
            return P.ExpertBytes.from_tensor_table(
                entries,
                provenance=f"derived from loaded tensor shapes: {path.name} "
                           f"expert {expert} rank {rank} of "
                           f"{len(ranks)} rank(s)")
        except ValueError:
            logger.warning("FQ budget: %s expert %d rank %d does not fit the "
                           "affine trellis byte model", path.name, expert,
                           rank, exc_info=True)
            return None
    return None


def parse_expert_bytes_spec(spec: str) -> "P.ExpertBytes":
    """``"k3=3542028,k4=4721676"`` -> an operator-declared byte model."""
    points: dict[int, int] = {}
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition("=")
        key = key.strip().lower().lstrip("k")
        if not sep or not key.isdigit():
            raise ValueError(
                f"unparseable {FQ_EXPERT_BYTES_ENV} entry {part!r}: want "
                f"'k3=3542028,k4=4721676' (bytes per expert per rank)")
        points[int(key)] = int(val.strip())
    return P.ExpertBytes.from_measurements(
        points, provenance=f"operator-declared {FQ_EXPERT_BYTES_ENV}={spec}")


def resolve_expert_bytes(*, spec: str | None = None,
                         artifact_dir: str | None = None,
                         enforcing: bool = True) -> "P.ExpertBytes":
    """Per-expert byte model, most trustworthy source first.

    1. an explicit operator declaration (``VLLM_FQ_EXPERT_BYTES``),
    2. the loaded checkpoint's own tensor shapes,
    3. the built-in GLM-5.2/TP4 reference measurement — flagged
       ``is_reference`` and said out loud, because it describes a
       different checkpoint and may not match this one.

    ``enforcing`` is True when a byte ceiling is actually gating
    decisions; the fallback is then a WARNING rather than an INFO,
    because a wrong constant would be silently deciding what stays
    resident.
    """
    if spec:
        return parse_expert_bytes_spec(spec)
    if artifact_dir:
        eb = expert_bytes_from_checkpoint(artifact_dir)
        if eb is not None:
            return eb
    eb = P.reference_expert_bytes()
    (logger.warning if enforcing else logger.info)(
        "FQ budget: could not read per-expert byte sizes from the loaded "
        "checkpoint (%s unset/unreadable) — falling back to the %s. Set %s "
        "(e.g. k3=3542028,k4=4721676) if this checkpoint's geometry differs.",
        FQ_ARTIFACT_DIR_ENV, eb.provenance, FQ_EXPERT_BYTES_ENV)
    return eb


# ------------------------------------------------------------------ metrics
class FqMetrics:
    """03-testing-validation §Instrumentation, prometheus_client-native
    (multiprocess-aware exactly like vllm's own metrics: with
    ``PROMETHEUS_MULTIPROC_DIR`` set, worker-process writes aggregate into
    the API server's ``/metrics`` via ``MultiProcessCollector``)."""

    def __init__(self, registry=None):
        import prometheus_client as pc
        kw = {} if registry is None else {"registry": registry}
        # PROPOSED and APPLIED are separate series on purpose. One counter
        # covering both read "fq_swaps_total{layer=3} 2.0" on a serve where
        # the apply path was not wired at all: 64 swaps decided, zero
        # installed, and a dashboard that said the fungible loop was working.
        # A metric an operator cannot distinguish from success IS a false
        # success.
        self.swap_proposals_total = pc.Counter(
            "fq_swap_proposals_total",
            "Fungible-quant expert tier swaps DECIDED per MoE layer. A "
            "proposal is not a change: compare against "
            "fq_swaps_applied_total.",
            ["layer"], **kw)
        self.swaps_applied_total = pc.Counter(
            "fq_swaps_applied_total",
            "Fungible-quant expert tier swaps ACTUALLY INSTALLED per MoE "
            "layer, counted only after the apply backend confirmed.",
            ["layer"], **kw)
        self.apply_bound = pc.Gauge(
            "fq_apply_bound",
            "1 when an apply backend is bound, 0 when the loop can only "
            "record proposals. Zero here means no proposal can ever become "
            "a change, however many are decided.",
            multiprocess_mode="mostrecent", **kw)
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
        # Staging and quiesce cost. These three decide how fast the loop can
        # converge and how much serving it disturbs, and all three were
        # computed but never surfaced: staging turned out to be the binding
        # constraint (IO-bound, ~10.5 MB/s of small ranged reads) and that was
        # only discovered by differencing log timestamps.
        self.stage_seconds = pc.Gauge(
            "fq_stage_seconds",
            "Wall time to stage the last swap batch (fragment IO, off-step).",
            multiprocess_mode="mostrecent", **kw)
        self.stage_bytes = pc.Gauge(
            "fq_stage_bytes",
            "Host-to-device bytes in the last staged swap batch.",
            multiprocess_mode="mostrecent", **kw)
        self.apply_window_seconds = pc.Gauge(
            "fq_apply_window_seconds",
            "Quiesce window of the last committed apply — the only interval "
            "in which serving is actually disturbed.",
            multiprocess_mode="mostrecent", **kw)
        self.tier_occupancy = pc.Gauge(
            "fq_tier_occupancy",
            "Experts resident per (layer, K tier) under the running policy.",
            ["layer", "tier"], multiprocess_mode="mostrecent", **kw)
        self.memory_budget_bytes = pc.Gauge(
            "fq_memory_budget_bytes",
            "Configured per-device byte ceiling for the fungible-quant "
            "expert pool (VLLM_FQ_MEMORY_BUDGET). 0 means no byte budget "
            "is configured and only the fixed-cardinality budget applies.",
            multiprocess_mode="mostrecent", **kw)
        self.memory_used_bytes = pc.Gauge(
            "fq_memory_used_bytes",
            "Per-device bytes the CURRENT expert tier occupancy actually "
            "costs, computed from the live layer x K matrix and the "
            "per-expert byte model (see fq_memory_budget_bytes).",
            multiprocess_mode="mostrecent", **kw)
        self.promotions_headroom = pc.Gauge(
            "fq_promotions_headroom",
            "How many further K3->K4 expert promotions fit under the byte "
            "ceiling at the current occupancy. Negative means the pool is "
            "already over budget; -1 with fq_memory_budget_bytes == 0 "
            "means unbounded (no byte budget configured).",
            multiprocess_mode="mostrecent", **kw)
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
        budget: "P.MemoryBudget | None" = None,
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
        # A policy legitimately covers layers the collector does not
        # instrument. GLM-5.2's layer 78 is the MTP layer: the EXL3 loader
        # REQUIRES a bitrate entry for it ("rank-sliced EXL3 bitrate map is
        # missing layer 78") because it is a real MoE layer in the
        # checkpoint, but it is not bound as a main-model MoERunner so the
        # collector never sees it. Refusing to start on that mismatch made
        # the two requirements unsatisfiable at once. Restrict the DECISION
        # domain to instrumented layers and say so; the loader keeps its
        # full map.
        _policy_layers = sorted(int(k) for k in policy_doc["bits_per_expert"])
        _bound = set(collector.count_buf)
        if _bound and not _bound.issuperset(_policy_layers):
            _skip = [x for x in _policy_layers if x not in _bound]
            if _skip and len(_skip) < len(_policy_layers):
                logger.warning(
                    "FQ loop: %d policy layer(s) are not instrumented by the "
                    "collector and are excluded from decisions: %s",
                    len(_skip), _skip)
                _policy_layers = [x for x in _policy_layers if x in _bound]
        self.layers = _policy_layers
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
        # The online score/policy objective is calibrated specifically for
        # K3->K4 error reduction. Other binary pairs remain fully observable
        # and operator-swappable, but must not be fed through that policy as
        # though K2 were K3.
        self._auto_decisions_supported = (
            set(int(k) for k in np.unique(self.tier_of)) <= {P.K3, P.K4})
        if not self._auto_decisions_supported:
            logger.warning(
                "FQ loop: automatic decisions disabled for tier set %s; "
                "collector and admin re-tiering remain active",
                sorted(int(k) for k in np.unique(self.tier_of)))
        self.pins = self._pins_from_doc(policy_doc)
        self.budget = budget if budget is not None else self._resolve_budget()
        self._log_budget()

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
        self._promotion_apply_warned = False
        # Apply attempts the backend refused/crashed on. A missing fragment
        # at the required K is the expected cause: the incumbent tiering
        # stays live and the next interval retries.
        self.apply_failures = 0

        if eps is None:
            eps = self._resolve_eps()
        self.eps = eps

        if self.metrics is not None and self.is_lead:
            for layer in self.layers:
                self.metrics.swap_proposals_total.labels(
                    layer=str(layer)).inc(0)
                self.metrics.swaps_applied_total.labels(
                    layer=str(layer)).inc(0)
            self._export_occupancy()
            self._export_budget()

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

    def _resolve_budget(self) -> P.MemoryBudget:
        """Build the memory budget from env + the checkpoint's geometry.

        A malformed ``VLLM_FQ_MEMORY_BUDGET`` is allowed to propagate: the
        integration seam catches loop-init failures and degrades to a
        collector-only serve, which performs no swaps and therefore
        cannot grow past a ceiling. Failing OPEN on a memory budget — by
        quietly dropping it — would be the unsafe choice.
        """
        expert_bytes = resolve_expert_bytes(
            spec=self.cfg.expert_bytes, artifact_dir=self.cfg.artifact_dir,
            enforcing=bool(self.cfg.memory_budget))
        return P.MemoryBudget.from_spec(
            self.cfg.memory_budget, expert_bytes,
            num_layers=len(self.layers), num_experts=self.num_experts,
            device_total_bytes=device_total_bytes())

    def _log_budget(self) -> None:
        summary = self.budget.summary(self.tier_of)
        if self.budget.limit_bytes is None:
            logger.info(
                "FQ memory budget: none (%s unset) — fixed cardinality only; "
                "current pool %d B/rank; %s", FQ_MEMORY_BUDGET_ENV,
                summary["used_bytes"], self.budget.expert_bytes.describe())
            return
        n_high = self.budget.n_high_per_layer(len(self.layers),
                                              self.num_experts)
        logger.info(
            "FQ memory budget: %d B/rank (%s) | used %d B | headroom %d B = "
            "%s more K3->K4 promotions | equivalent fixed cardinality "
            "<= %s K4/layer over %d layers | %s",
            self.budget.limit_bytes, self.cfg.memory_budget,
            summary["used_bytes"], summary["headroom_bytes"],
            summary["headroom_promotions"], n_high, len(self.layers),
            self.budget.expert_bytes.describe())
        if summary["headroom_bytes"] is not None and summary["headroom_bytes"] < 0:
            logger.error(
                "FQ memory budget: the BOOT policy is already %d B over the "
                "%d B ceiling — no promotion will be admitted until the "
                "occupancy comes down", -summary["headroom_bytes"],
                self.budget.limit_bytes)

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
        if (self._step % self.cfg.interval_steps
                or not self._auto_decisions_supported):
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
        """Mean per-layer Jaccard over layers that HAVE a desired set.

        A layer whose n_k4 is 0 has an empty desired set in both snapshots.
        The old code clamped the union to 1, scoring that layer 0.0 and
        averaging it in — so 27 of GLM-5.2's 75 layers, which cannot hold a
        K4 expert at all and therefore cannot churn, dragged the mean from
        0.879 down to 0.562 against a 0.950 floor. The guard was measuring
        the stability of sets that do not exist, and blocked every swap for
        it.

        Empty-vs-empty is not "maximally unstable"; it is no evidence. Skip
        those rows and average the rest. If NO layer has a desired set there
        is nothing to guard, so report perfect agreement rather than zero.
        """
        inter = (a & b).sum(axis=1).astype(np.float64)
        union = (a | b).sum(axis=1).astype(np.float64)
        live = union > 0
        if not live.any():
            return 1.0
        return float((inter[live] / union[live]).mean())

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

        # Guard 5 — the byte ceiling. A 1-for-1 K3<->K4 trade is
        # byte-neutral so this normally passes everything through; when
        # the ceiling leaves room, the surplus is spent on unpaired
        # promotions, which is the thing fixed cardinality cannot express.
        promotions: list[tuple[int, int]] = []
        rejections: list[dict] = []
        if self.budget.limit_bytes is not None:
            swaps, rejections = P.budget_filter(swaps, self.tier_of,
                                                self.budget)
            if not jaccard_held:
                promotions, prej = P.plan_promotions(
                    stats, self.eps, P.apply_swaps(self.tier_of, swaps),
                    self.budget, pins=self.pins, dwell=dwell,
                    cfg=self._decide_cfg(),
                    exclude=P.swap_touched(swaps))
                rejections = rejections + prej
            for rej in rejections:
                logger.warning("FQ budget step=%d: %s", self._step,
                               P.rejection_message(rej))

        # Two memberships, kept apart on purpose. The PROPOSAL includes the
        # headroom promotions and is what gets explained, logged and
        # archived; the swaps-only membership is the executable part (a
        # 1-for-1 K3<->K4 trade), and it is the only thing the apply
        # backend may ever be handed — see _maybe_apply.
        swaps_tier = P.apply_swaps(self.tier_of, swaps)
        proposed_tier = P.apply_promotions(swaps_tier, promotions,
                                           self.budget.high_k)
        proposed_doc = self._doc_for(proposed_tier, swaps)
        proposed_sha = policy_hash(proposed_doc)

        record = DL.explain(
            stats, self.eps, self.tier_of, swaps, pins=self.pins,
            dwell=dwell, cfg=self._decide_cfg(), step=self._step,
            policy_sha_before=self.policy_sha,
            policy_sha_after=(proposed_sha if (swaps or promotions)
                              else self.policy_sha))
        record["apply_mode"] = self.cfg.apply_mode
        record["applied"] = False
        # Budget state travels WITH the decision: a reviewer reading the
        # persisted JSON must be able to see the ceiling, what the pool
        # actually costs, what was refused, and by how much.
        record["budget"] = self.budget.summary(self.tier_of)
        record["budget"]["proposed_bytes"] = self.budget.used_bytes(
            proposed_tier)
        record["budget"]["rejections"] = rejections
        record["promotions"] = [{"layer": int(self.layers[l]),
                                 "layer_row": int(l), "expert": int(e),
                                 "to_k": int(self.budget.high_k)}
                                for l, e in promotions]
        record["totals"]["promotions"] = len(promotions)
        record["totals"]["budget_rejections"] = len(rejections)
        # decide()/explain() speak in row indices; record the row->model
        # layer id mapping so the persisted JSON is self-describing.
        record["layer_ids"] = [int(x) for x in self.layers]
        record["jaccard"] = jac
        record["jaccard_held"] = bool(jaccard_held)
        record["decision_sha"] = self.decision_sha(swaps)
        record["real_steps"] = int(self._real_steps)

        DL.log_decision(record)
        logger.info(
            "FQ decide rank=%d step=%d interval=%d swaps=%d promotions=%d "
            "budget_rejections=%d used=%d B headroom=%s sha=%s",
            self.rank, self._step, self._intervals_run, len(swaps),
            len(promotions), len(rejections),
            record["budget"]["used_bytes"],
            record["budget"]["headroom_promotions"],
            record["decision_sha"][:16])

        # Unpaired promotions can never be executed live, but the paired
        # swaps decided in the SAME interval can. Discarding those too
        # would freeze the serve: promotions are re-proposed every
        # interval for as long as any headroom exists, nothing is applied,
        # so the occupancy — and therefore the headroom — never changes.
        # Apply the executable subset from a swaps-only document.
        if promotions:
            applied = self._maybe_apply(self._doc_for(swaps_tier, swaps),
                                        swaps_tier, swaps, promotions)
        else:
            applied = self._maybe_apply(proposed_doc, proposed_tier,
                                        swaps, promotions)
        record["applied"] = applied
        # ``applied`` covers the swap list only. Promotions are structurally
        # inapplicable, so say so rather than leaving a reader to infer that
        # record["promotions"] went live alongside record["swaps"].
        record["promotions_applied"] = False
        record["apply_failures"] = int(self.apply_failures)

        if self.is_lead:
            self._persist(record, proposed_doc, swaps or promotions)
            self._export_metrics(record, swaps, jac,
                                 applied=bool(record.get("applied")))
        return record

    # ---------------------------------------------------------------- apply

    def _doc_for(self, tier: np.ndarray, swaps: list) -> dict:
        doc = {k: v for k, v in self.policy_doc.items()
               if k not in ("bits_per_expert", "provenance", "budget")}
        # Same hazard as admin.build_target_doc: rebuilding bits_per_expert
        # from self.layers alone silently drops every layer outside the
        # decision domain (MTP layer 78 on GLM-5.2 — required by the loader's
        # bitrate map, unbindable by the collector). The proposed document
        # would then cover fewer layers than the running one and the swap
        # engine would refuse it as "policies cover different layers".
        bpe = {str(k): list(v) for k, v in
               (self.policy_doc.get("bits_per_expert") or {}).items()}
        for row, layer in enumerate(self.layers):
            bpe[str(layer)] = [int(b) for b in tier[row]]
        doc["bits_per_expert"] = bpe
        # The declared cardinality is RECOMPUTED from the proposed tiers,
        # not copied: promotions change occupancy, and store.validate_policy
        # enforces cap == n. The byte ceiling rides along so a persisted
        # proposal records the budget it was decided under.
        budget = dict(self.policy_doc.get("budget") or {})
        budget["n_k4_per_layer"] = {
            str(layer): int((tier[row] == P.K4).sum())
            for row, layer in enumerate(self.layers)}
        if self.budget.limit_bytes is not None:
            budget["mode"] = "max_bytes"
            budget["max_bytes_per_rank"] = int(self.budget.limit_bytes)
            budget["bytes_per_expert_per_rank"] = {
                str(k): int(v)
                for k, v in self.budget.expert_bytes.table().items()}
            budget["bytes_source"] = self.budget.expert_bytes.provenance
        else:
            # The inherited document may have been committed under a byte
            # ceiling that is NOT configured on this run (the operator
            # dropped VLLM_FQ_MEMORY_BUDGET and restarted). Carrying its
            # mode/ceiling forward would have every document this loop
            # emits advertise a limit nothing is enforcing, so the byte
            # fields are cleared rather than left to setdefault.
            budget["mode"] = "fixed_cardinality"
            for key in ("max_bytes_per_rank", "bytes_per_expert_per_rank",
                        "bytes_source"):
                budget.pop(key, None)
        doc["budget"] = budget
        doc["provenance"] = {
            "proposed_by": f"fq-loop/{self.cfg.apply_mode}",
            "step": int(self._step),
            "base_policy": self.policy_sha,
            "num_swaps": len(swaps),
        }
        return doc

    def _maybe_apply(self, proposed_doc: dict, proposed_tier: np.ndarray,
                     swaps: list, promotions: list | None = None) -> bool:
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
        promotions = promotions or []
        if self.cfg.apply_mode == APPLY_DRYRUN or not (swaps or promotions):
            return False
        if promotions:
            # An unpaired promotion CHANGES the per-layer cardinality, and
            # nothing downstream can execute that: SwapPlan.from_memberships
            # refuses a cardinality change (D1), SwapEngine._validate_layer
            # derives the slab word counts from the tier-1 globals fixed at
            # prepare time, and the mixed-kernel memo key carries the
            # per-tier COUNTS, so a count change forces a recompile that is
            # refused under CUDA-graph capture. admin.check_cardinality
            # answers the same request with 501; the loop must not quietly
            # do what the admin API refuses to pretend it can do.
            #
            # apply_fn is handed the SWAP LIST, so a promotion would never
            # reach it anyway — advancing tier_of/policy_doc/store past it
            # would make the loop, the gauges and the committed policy all
            # claim a tier the device never received.
            #
            # The promotions are therefore DROPPED (still explained and
            # persisted to history/, which is exactly the "raise
            # n_k4_per_layer and restart" path) — but only the promotions.
            # The interval's paired swaps are byte-neutral and
            # cardinality-preserving, so they stay applicable; refusing
            # them as well would stall the serve permanently, because a
            # promotion is re-proposed every interval for as long as the
            # ceiling leaves headroom and nothing ever consumes it.
            if not self._promotion_apply_warned:
                logger.error(
                    "FQ budget: %d headroom promotion(s) proposed at step %d "
                    "but runtime cardinality growth cannot be applied live "
                    "(fixed-capacity slabs, D1) — dropping them and applying "
                    "only the %d paired swap(s). Raise budget.n_k4_per_layer "
                    "in the policy and restart to bank the headroom.",
                    len(promotions), self._step, len(swaps))
                self._promotion_apply_warned = True
            if not swaps:
                return False
        # Last line of defence for D1, independent of what the caller
        # passed: the backend is only ever given a swap list, so a
        # membership whose per-layer K4 cardinality moved is one it cannot
        # reach. Refuse rather than commit a policy the device never got.
        if not np.array_equal(P.n_k4_of(proposed_tier), self.n_k4):
            logger.error(
                "FQ apply refused at step %d: proposed membership changes the "
                "per-layer K4 cardinality (%s -> %s), which the swap backend "
                "cannot execute (D1)", self._step, list(map(int, self.n_k4)),
                list(map(int, P.n_k4_of(proposed_tier))))
            return False
        if self.apply_fn is None:
            if self.cfg.apply_mode == APPLY_ATOMIC and not self._atomic_warned:
                logger.warning(
                    "FQ apply_mode=atomic with no apply backend bound — "
                    "recording proposals only. This is the DEFAULT since "
                    "inline apply deadlocked a TP serve (stage() does "
                    "synchronous fragment IO inside the step, per-rank, so "
                    "the ranks desynchronise and the shm broadcast starves). "
                    "The swap engine itself is proven; what is missing is "
                    "out-of-step coordination. VLLM_FQ_LIVE_APPLY=1 binds it "
                    "anyway for experiments.")
                self._atomic_warned = True
            elif self.cfg.apply_mode == APPLY_RELOAD:
                logger.warning(
                    "FQ apply_mode=reload without a bound apply_fn — "
                    "recording proposal only; drive the proven M3 path "
                    "(fq_assemble + fq_reload swap) from the persisted "
                    "proposal JSON")
            return False
        try:
            # apply_fn may return the swaps it ACTUALLY installed instead of
            # a bare bool. That matters: an asynchronous backend can install a
            # batch staged an interval earlier, which is NOT the same set as
            # `swaps`. Trusting `proposed_tier` in that case makes tier_of
            # describe a device state that never existed, and the very next
            # interval proposes a move off an expert that is not there:
            #   ValueError: invalid swap (5, 82, 3): e_out must be resident K4
            result = self.apply_fn(proposed_doc, swaps)
            ok = bool(result)
            installed = result if isinstance(result, (list, tuple)) else None
        except Exception:  # noqa: BLE001 — a supply failure is not a crash
            self.apply_failures += 1
            logger.exception(
                "FQ apply failed at step %d (%d swaps) — keeping the "
                "incumbent tiering and retrying next interval "
                "(apply_failures=%d)",
                self._step, len(swaps), self.apply_failures)
            return False
        if ok:
            if installed is not None:
                # Rebuild the incumbent from what the device actually took, so
                # tier_of and the layer states cannot drift apart.
                applied_tier = self.tier_of.copy()
                for row, e_out, e_in in installed:
                    applied_tier[int(row), int(e_out)] = P.K3
                    applied_tier[int(row), int(e_in)] = P.K4
                if not np.array_equal(applied_tier, proposed_tier):
                    logger.info(
                        "FQ apply installed %d swap(s) that differ from this "
                        "interval's proposal — adopting the INSTALLED "
                        "membership, not the proposed one", len(installed))
                proposed_tier = applied_tier
                proposed_doc = self._doc_for(applied_tier, list(installed))
            swapped = self.tier_of != proposed_tier
            self.tier_of = proposed_tier
            self._entered_step = np.where(
                swapped, self._real_steps, self._entered_step)
            self.policy_doc = proposed_doc
            self.policy_sha = policy_hash(proposed_doc)
            self._policy_step = self._step
            # Promotions change the running cardinality; decide() refuses
            # a membership that disagrees with cfg["n_k4"], so the budget
            # must follow the occupancy it actually applied.
            self.n_k4 = P.n_k4_of(self.tier_of)
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
        # A proposal carrying promotions is ALWAYS archived, even when the
        # interval's swaps were applied: the promotions themselves never
        # are, and history/ is the only record of the "raise
        # n_k4_per_layer and restart" recommendation.
        unapplied_growth = bool(record.get("totals", {}).get("promotions"))
        if swaps and (unapplied_growth or not record["applied"]):
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

    def _export_budget(self) -> None:
        """fq_memory_budget_bytes / fq_memory_used_bytes /
        fq_promotions_headroom, from the ACTUAL live occupancy."""
        summary = self.budget.summary(self.tier_of)
        self.metrics.memory_budget_bytes.set(
            0 if summary["limit_bytes"] is None else summary["limit_bytes"])
        self.metrics.memory_used_bytes.set(summary["used_bytes"])
        # -1 is the documented "unbounded" sentinel; a real over-budget
        # pool reports its own (negative) promotion count instead.
        self.metrics.promotions_headroom.set(
            -1 if summary["headroom_promotions"] is None
            else summary["headroom_promotions"])

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
                             diff_only=diff_only,
                             budget=self.budget.summary(self.tier_of))
        except Exception:  # noqa: BLE001 - telemetry must never kill a serve
            logger.exception("FQ composition table failed to render")
            return
        for line in text.splitlines():
            logger.info("%s", line)
        self._last_composition = cur

        # Membership diff, because the table above CANNOT show a swap.
        #
        # `_occupancy_map` reports per-tier COUNTS, and D1 fixed cardinality
        # makes every count invariant under a K3<->K4 exchange. So the table
        # printed "no tier changes across 75 layers" at every single interval
        # of a run in which 256 experts demonstrably moved (verified by
        # diffing the committed policy against the boot policy file). It was
        # literally true and the worst kind of instrument: it reads as
        # evidence that nothing happened.
        #
        # Counts still answer "what is the occupancy", so they stay; this adds
        # the question they cannot answer.
        prev = getattr(self, "_last_tier_of", None)
        curr = np.array(self.tier_of, copy=True)
        if prev is not None and getattr(prev, "shape", None) == curr.shape:
            moved = prev != curr
            n_moved = int(moved.sum())
            if n_moved:
                rows = np.flatnonzero(moved.any(axis=1))
                sample = ", ".join(f"L{int(self.layers[r])}:{int(moved[r].sum())}"
                                   for r in rows[:6])
                logger.info(
                    "  membership: %d expert(s) moved across %d layer(s) "
                    "since the last table (%s%s)",
                    n_moved, int(rows.size), sample,
                    ", ..." if rows.size > 6 else "")
            else:
                logger.info("  membership: unchanged since the last table")
        self._last_tier_of = curr

    def _export_metrics(self, record: dict, swaps: list,
                        jac: float | None, applied: bool = False) -> None:
        if self.metrics is None:
            return
        for layer, _, _ in swaps:
            self.metrics.swap_proposals_total.labels(
                layer=str(self.layers[layer])).inc()
            if applied:
                self.metrics.swaps_applied_total.labels(
                    layer=str(self.layers[layer])).inc()
        self.metrics.apply_bound.set(1 if self.apply_fn is not None else 0)
        if jac is not None:
            self.metrics.jaccard.set(jac)
        self.metrics.policy_age_steps.set(self._step - self._policy_step)
        self._export_occupancy()
        self._export_budget()


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
        "caps=(%d/layer, %d total) memory_budget=%s layers=%s policy=%s",
        cfg.apply_mode, cfg.interval_steps, cfg.dwell_steps, cfg.hysteresis,
        cfg.max_swaps_per_layer, cfg.max_swaps_total,
        state.budget.limit_bytes if state.budget.limit_bytes is not None
        else "none", state.layers, state.policy_sha[:16])
    return state
