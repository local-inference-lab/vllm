# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant policy engine prototype — pure NumPy, CPU-only, deterministic.

Implements the decision function of research/fungible-quant/implementation/ (vllm-voipmonitor research branch) 
01-artifacts-policy-stats.md §3 (score -> desired set -> symmetric difference
-> guards -> ordered swap list), plus apply/inverse helpers and the §1.2
projection of a policy with different N_L onto the running cardinality.

Domain conventions
------------------
- All arrays are [L, E] (num MoE layers x num logical experts per layer).
- Tiers are the literal K values: 3 (base) and 4 (overlay).
- ``pins[l, e]``: 0 = free, 3 = pinned to K3 (never upgraded), 4 = pinned to
  K4 (never downgraded; forced into the desired set).
- ``dwell[l, e]``: steps the expert has been resident in its *current* tier.
- A swap is ``(layer, e_out, e_in)``: e_out K4->K3, e_in K3->K4.

Determinism: only stable sorts with total-order keys (score, then expert id);
no dict iteration over unordered inputs; pure functions of the arguments.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

K2, K3, K4, K5 = 2, 3, 4, 5
K_UNIVERSE = (K2, K3, K4, K5)
PIN_NONE, PIN_K3, PIN_K4 = 0, 3, 4

DEFAULT_CFG = {
    "alpha": 1.0,           # count exponent   (free parameter until Phase 0a)
    "beta": 1.0,            # gate-mass exponent (free parameter until Phase 0a)
    "hysteresis": 1.25,     # VLLM_FQ_HYSTERESIS
    "dwell_steps": 6000,    # VLLM_FQ_DWELL_STEPS (2 x interval default)
    "max_swaps_per_layer": 2,   # VLLM_FQ_MAX_SWAPS_LAYER
    "max_swaps_total": 64,      # VLLM_FQ_MAX_SWAPS_TOTAL
}


def _full_cfg(cfg):
    out = dict(DEFAULT_CFG)
    out.update(cfg or {})
    if "n_k4" not in out:
        raise ValueError("cfg must provide 'n_k4' (per-layer K4 cardinality)")
    return out


def _as_eps(eps, L, E):
    """Accept eps as {3: [L,E], 4: [L,E]} or an [L,E,2] array -> (eps3, eps4)."""
    if isinstance(eps, dict):
        e3 = np.asarray(eps[K3], dtype=np.float64)
        e4 = np.asarray(eps[K4], dtype=np.float64)
    else:
        arr = np.asarray(eps, dtype=np.float64)
        if arr.shape != (L, E, 2):
            raise ValueError(f"eps must be [L,E,2] or {{3:..,4:..}}, got {arr.shape}")
        e3, e4 = arr[..., 0], arr[..., 1]
    if e3.shape != (L, E) or e4.shape != (L, E):
        raise ValueError("eps arrays must be [L,E]")
    return e3, e4


def score(stats, eps, cfg=None):
    """score[l,e] = (eps_k3 - eps_k4) * mass**beta * count**alpha  (spec §3.2)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    count = np.asarray(stats["count"], dtype=np.float64)
    mass = np.asarray(stats["mass"], dtype=np.float64)
    L, E = count.shape
    e3, e4 = _as_eps(eps, L, E)
    with np.errstate(invalid="ignore"):
        s = (e3 - e4) * np.power(mass, cfg["beta"]) * np.power(count, cfg["alpha"])
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def n_k4_of(tier_of):
    """Per-layer K4 occupancy of a membership array."""
    return (np.asarray(tier_of) == K4).sum(axis=1)


def decide(stats, eps, tier_of, pins=None, dwell=None, cfg=None):
    """Compute the ordered swap list for one policy interval (spec §3.2).

    Returns [(layer, e_out, e_in), ...] sorted by (layer, score-gap desc, ids).
    Pin-forced pairs bypass dwell/hysteresis for the forced member and sort
    ahead of free pairs under the caps.
    """
    cfg = _full_cfg(cfg)
    tier_of = np.asarray(tier_of)
    L, E = tier_of.shape
    s = score(stats, eps, cfg)
    n_k4 = np.broadcast_to(np.asarray(cfg["n_k4"], dtype=np.int64), (L,))
    pins = np.zeros((L, E), dtype=np.int64) if pins is None else np.asarray(pins)
    if dwell is None:
        dwell_ok = np.ones((L, E), dtype=bool)
    else:
        dwell_ok = np.asarray(dwell) >= cfg["dwell_steps"]
    h = float(cfg["hysteresis"])

    candidates = []  # (gap, layer, e_out, e_in)
    for l in range(L):
        cur = tier_of[l] == K4
        if int(cur.sum()) != int(n_k4[l]):
            raise ValueError(
                f"layer {l}: membership has {int(cur.sum())} K4, budget is "
                f"{int(n_k4[l])} — project() the policy first (D1)")
        pin4 = pins[l] == PIN_K4
        pin3 = pins[l] == PIN_K3
        if int(pin4.sum()) > int(n_k4[l]) or E - int(pin3.sum()) < int(n_k4[l]):
            raise ValueError(f"layer {l}: pins incompatible with budget")

        # Desired set: pinned-K4 forced in, pinned-K3 forced out, rest by score.
        desired = pin4.copy()
        slots = int(n_k4[l] - pin4.sum())
        free_order = np.lexsort((np.arange(E), -s[l]))  # score desc, id asc
        for e in free_order:
            if slots == 0:
                break
            if not pin4[e] and not pin3[e]:
                desired[e] = True
                slots -= 1

        # Symmetric difference -> paired swap candidates.
        enter = sorted(np.nonzero(desired & ~cur)[0],
                       key=lambda e: (not pin4[e], -s[l, e], e))
        leave = sorted(np.nonzero(cur & ~desired)[0],
                       key=lambda e: (not pin3[e], s[l, e], e))
        layer_pairs = []
        for e_in, e_out in zip(enter, leave):
            forced_in, forced_out = bool(pin4[e_in]), bool(pin3[e_out])
            # Guard 1 — dwell (forced member exempt, its counterpart is not).
            if not forced_in and not dwell_ok[l, e_in]:
                continue
            if not forced_out and not dwell_ok[l, e_out]:
                continue
            # Guard 2 — hysteresis (waived when either side is pin-forced).
            if not (forced_in or forced_out) and not (s[l, e_in] > h * s[l, e_out]):
                continue
            gap = math.inf if (forced_in or forced_out) else float(s[l, e_in] - s[l, e_out])
            layer_pairs.append((gap, l, int(e_out), int(e_in)))
        # Guard 3a — per-layer cap, highest score-gap first.
        layer_pairs.sort(key=lambda p: (-p[0], p[2], p[3]))
        candidates.extend(layer_pairs[: int(cfg["max_swaps_per_layer"])])

    # Guard 3b — model-wide cap, highest score-gap first, deterministic ties.
    candidates.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
    candidates = candidates[: int(cfg["max_swaps_total"])]
    # Guard 4 — emit ordered swap list.
    candidates.sort(key=lambda p: (p[1], -p[0], p[2], p[3]))
    return [(l, e_out, e_in) for _, l, e_out, e_in in candidates]


def apply_swaps(tier_of, swaps):
    """Apply an ordered swap list to a membership array (returns a copy)."""
    out = np.array(tier_of, copy=True)
    for l, e_out, e_in in swaps:
        if out[l, e_out] != K4 or out[l, e_in] != K3:
            raise ValueError(f"invalid swap ({l}, {e_out}, {e_in}) for current tiers")
        out[l, e_out] = K3
        out[l, e_in] = K4
    return out


def inverse(swaps):
    """Inverse swap list: applying swaps then inverse(swaps) is identity (§3.3)."""
    return [(l, e_in, e_out) for (l, e_out, e_in) in reversed(swaps)]


def project(bits_per_expert, n_k4, order=None):
    """Project a policy's membership onto the running cardinality (spec §1.2).

    ``bits_per_expert``: [L,E] ints in {3,4} from the policy artifact.
    ``n_k4``: running per-layer budget (scalar or [L]).
    ``order``: optional [L,E] ranking (the policy's own ordering, e.g. its
    scores); higher = more deserving of K4. Defaults to expert-id order.
    Take the policy's top-N_L per layer: its K4 members first (by order),
    then, if the running budget is larger, the best of its K3 members.
    """
    bits = np.asarray(bits_per_expert)
    L, E = bits.shape
    n_k4 = np.broadcast_to(np.asarray(n_k4, dtype=np.int64), (L,))
    order = np.zeros((L, E)) if order is None else np.asarray(order, dtype=np.float64)
    out = np.full((L, E), K3, dtype=bits.dtype)
    for l in range(L):
        is4 = (bits[l] == K4).astype(np.int64)
        rank = np.lexsort((np.arange(E), -order[l], -is4))  # K4 first, then order desc, id asc
        out[l, rank[: int(n_k4[l])]] = K4
    return out


# =====================================================================
# Memory budget — bytes <-> experts/bpw/layer
# =====================================================================
# The fixed-cardinality budget above (``n_k4``) is a PROXY for memory: it
# holds the expert pool's footprint constant by construction, because a
# 1-for-1 K3<->K4 trade is byte-neutral. It cannot express what an
# operator actually wants to say, which is "stay under N bytes on this
# device". This section adds that, and — crucially — makes the two
# representations inter-convertible so a byte ceiling can be read back as
# "how many experts per layer may sit at K4".
#
# Everything here is per DEVICE (i.e. per TP rank), because that is the
# thing a byte ceiling is actually bounded by and the thing
# ``gpu_memory_utilization`` is a fraction of.

_SIZE_SPEC_RE = re.compile(r"^([0-9]*\.?[0-9]+)([a-z%]*)$")

# Binary multipliers throughout: an operator writing "78g" for a device
# with 79.6 GiB free means gibibytes, and mixing SI and binary in one
# knob is how you end up 7% over the line.
_SIZE_UNITS: dict[str, int] = {
    "": 1, "b": 1,
    "k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10,
    "m": 1 << 20, "mb": 1 << 20, "mib": 1 << 20,
    "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30,
    "t": 1 << 40, "tb": 1 << 40, "tib": 1 << 40,
}
_BUDGET_OFF = ("", "none", "off", "unlimited", "unbounded")

# The "experts/bpw per layer" spellings of the same ceiling — see
# MemoryBudget.from_spec, which converts them to bytes.
_PER_LAYER_RE = re.compile(r"^([0-9]+)(?:experts?)?/layer$")
_BPW_RE = re.compile(r"^([0-9]*\.?[0-9]+)bpw$")


def parse_memory_budget(spec, *, device_total_bytes: int | None = None
                        ) -> int | None:
    """Resolve a memory-budget spec to an absolute byte count.

    Accepted forms, mirroring ``--gpu-memory-utilization`` ergonomics so
    the operator has one mental model for "how much of the card may this
    use":

    - ``None`` / ``""`` / ``"none"`` / ``"off"`` / ``"unlimited"`` -> None
      (no byte ceiling; the fixed-cardinality budget alone applies).
    - an explicit size: ``"78g"``, ``"78GiB"``, ``"80000000000"``,
      ``"512mb"``, ``"1b"``. Suffixes are BINARY (1g == 1 GiB).
    - a fraction of device memory: ``"0.80"`` or ``"80%"``. A bare number
      in ``(0, 1]`` is read as a fraction — exactly like
      ``gpu_memory_utilization`` — so ``"1"`` means the whole device and
      one single byte must be written ``"1b"``.

    ``device_total_bytes`` is required for the fractional forms; without
    it they raise rather than silently guessing a device size.
    """
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError(f"memory budget must be a size or fraction, got {spec!r}")
    if isinstance(spec, (int, float)):
        num, unit = float(spec), ""
    else:
        s = str(spec).strip().lower().replace("_", "").replace(" ", "")
        if s in _BUDGET_OFF:
            return None
        m = _SIZE_SPEC_RE.match(s)
        if m is None:
            raise ValueError(
                f"unparseable memory budget {spec!r}: want a size "
                f"('78g', '80000000000', '512mib'), a fraction of device "
                f"memory ('0.80', '80%'), or 'none'")
        num, unit = float(m.group(1)), m.group(2)

    if unit == "%":
        frac = num / 100.0
    elif unit == "":
        # gpu-memory-utilization semantics: (0, 1] is a fraction, above 1
        # is an absolute byte count.
        frac = num if 0.0 < num <= 1.0 else None
        if frac is None:
            if num <= 0:
                raise ValueError(
                    f"memory budget must be positive, got {spec!r}")
            return int(num)
    else:
        mult = _SIZE_UNITS.get(unit)
        if mult is None:
            raise ValueError(
                f"unknown size unit {unit!r} in memory budget {spec!r}; "
                f"known: {', '.join(sorted(u for u in _SIZE_UNITS if u))}")
        if num <= 0:
            raise ValueError(f"memory budget must be positive, got {spec!r}")
        return int(num * mult)

    if not 0.0 < frac <= 1.0:
        raise ValueError(
            f"fractional memory budget {spec!r} resolves to {frac} — must "
            f"be in (0, 1]")
    if device_total_bytes is None:
        raise ValueError(
            f"memory budget {spec!r} is a fraction of device memory, but no "
            f"device total is available (no CUDA device visible?) — give an "
            f"absolute size such as '78g' instead")
    return int(frac * int(device_total_bytes))


@dataclass(frozen=True)
class ExpertBytes:
    """How many bytes ONE expert occupies at a given K, on ONE rank.

    The exl3 trellis layout makes this exactly affine in K. An expert is
    three projections; each projection's ``trellis`` tensor has shape
    ``[H/16, I/16, 16*K]`` (see ``fragments.ExpertStage``), so its byte
    count is strictly proportional to K, while everything else the expert
    carries — the ``suh``/``svh`` rotation vectors and the ``mcg`` scalar
    — has no K in its shape at all. Hence::

        bytes(K) = trellis_bytes_per_k * K + fixed_bytes

    Two points determine that line exactly. That is the ONLY reason this
    module is allowed to state a K2 or K5 figure it has never measured:
    it is evaluating a known-affine function, not extrapolating a trend.

    ``provenance`` records where the numbers came from and is surfaced in
    the log, the decision record and the composition table — a byte
    budget computed from the wrong checkpoint's geometry is worse than no
    byte budget, so the source is never allowed to go unstated.
    """

    trellis_bytes_per_k: int
    fixed_bytes: int
    provenance: str
    measured_ks: tuple[int, ...] = ()
    # True when these numbers describe some OTHER checkpoint (the built-in
    # reference measurement) rather than the one actually loaded.
    is_reference: bool = False

    def __post_init__(self) -> None:
        if self.trellis_bytes_per_k <= 0:
            raise ValueError(
                f"trellis_bytes_per_k must be positive, got "
                f"{self.trellis_bytes_per_k}")
        if self.fixed_bytes < 0:
            raise ValueError(f"fixed_bytes must be >= 0, got {self.fixed_bytes}")

    def bytes_for(self, k: int) -> int:
        """Bytes one expert occupies at bitrate ``k`` (one rank)."""
        k = int(k)
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        return self.trellis_bytes_per_k * k + self.fixed_bytes

    def promotion_cost(self, k_from: int = K3, k_to: int = K4) -> int:
        """Byte delta of moving one expert from ``k_from`` to ``k_to``.

        Signed: a demotion returns a negative number.
        """
        return self.trellis_bytes_per_k * (int(k_to) - int(k_from))

    def table(self, ks: Sequence[int] = K_UNIVERSE) -> dict[int, int]:
        return {int(k): self.bytes_for(k) for k in ks}

    def describe(self, ks: Sequence[int] = K_UNIVERSE) -> str:
        per_k = " ".join(f"K{k}={self.bytes_for(k)}" for k in ks)
        measured = (",".join(f"K{k}" for k in self.measured_ks)
                    if self.measured_ks else "-")
        return (f"{per_k} bytes/expert/rank (trellis {self.trellis_bytes_per_k}"
                f" B per K + {self.fixed_bytes} B fixed; measured at "
                f"{measured}; source: {self.provenance})")

    # ------------------------------------------------------ constructors

    @classmethod
    def from_measurements(cls, points: Mapping[int, int], *, provenance: str,
                          is_reference: bool = False) -> "ExpertBytes":
        """Fit the affine model to >= 2 measured ``{K: bytes}`` points.

        Every supplied point must lie exactly on the fitted line; a set
        that does not is a measurement (or geometry) error and is refused
        rather than least-squares'd into a plausible-looking lie.
        """
        pts = sorted((int(k), int(v)) for k, v in points.items())
        if len(pts) < 2:
            raise ValueError(
                "need at least two measured K points to separate the "
                "K-proportional trellis bytes from the fixed bytes; got "
                f"{pts}")
        (k0, b0), (k1, b1) = pts[0], pts[-1]
        span = k1 - k0
        if (b1 - b0) % span:
            raise ValueError(
                f"measurements {pts} imply a fractional bytes-per-K slope "
                f"({b1 - b0}/{span}) — not a valid trellis geometry")
        slope = (b1 - b0) // span
        fixed = b0 - slope * k0
        for k, b in pts:
            if slope * k + fixed != b:
                raise ValueError(
                    f"measurement K{k}={b} is off the line fitted through "
                    f"{pts} (expected {slope * k + fixed}) — the points do "
                    f"not describe one checkpoint geometry")
        return cls(slope, fixed, provenance, tuple(k for k, _ in pts),
                   is_reference)

    @classmethod
    def from_tensor_table(
        cls,
        entries: Iterable[tuple[str, int, Sequence[int]]],
        *,
        provenance: str,
    ) -> "ExpertBytes":
        """Derive the model from ONE expert's REAL tensor table.

        ``entries`` are ``(name, nbytes, shape)`` triples for a single
        (expert, rank) — exactly what a safetensors header yields, where
        ``nbytes`` comes from the tensor's own ``data_offsets`` and so is
        the true on-disk/in-memory span, not a recomputation.

        A tensor is trellis iff its name's last dotted component is
        ``trellis``; its K is read straight off the last dim
        (``shape[-1] == 16 * K``). Everything else is K-independent
        overhead. This is the honest runtime path: no constant is
        assumed, the geometry comes from the checkpoint that is actually
        loaded.
        """
        trellis_bytes = 0
        fixed_bytes = 0
        ks: set[int] = set()
        seen = 0
        for name, nbytes, shape in entries:
            seen += 1
            nbytes = int(nbytes)
            if str(name).rsplit(".", 1)[-1] == "trellis":
                shape = tuple(int(d) for d in shape)
                if not shape or shape[-1] <= 0 or shape[-1] % 16:
                    raise ValueError(
                        f"{name}: trellis last dim {shape[-1] if shape else None}"
                        f" is not 16*K — cannot read K off this tensor")
                ks.add(shape[-1] // 16)
                trellis_bytes += nbytes
            else:
                fixed_bytes += nbytes
        if not seen:
            raise ValueError("empty tensor table — nothing to derive from")
        if len(ks) != 1:
            raise ValueError(
                f"expected exactly one K across the expert's trellis "
                f"tensors, found {sorted(ks)}")
        k = ks.pop()
        if trellis_bytes % k:
            raise ValueError(
                f"trellis bytes {trellis_bytes} are not divisible by K={k} — "
                f"the layout is not the affine trellis this model assumes")
        return cls(trellis_bytes // k, fixed_bytes,
                   f"{provenance} (K{k} tensor table, {seen} tensors)", (k,))


# Measured on the live GLM-5.2 serve (TP4, SM120), one MoE expert:
#   K3 = 14,168,112 B across all 4 ranks -> 3,542,028 B per rank
#   K4 = 18,886,704 B across all 4 ranks -> 4,721,676 B per rank
# so a K3->K4 promotion costs 1,179,648 B per rank. Those two points fit
# the affine model above exactly (slope 1,179,648 B/K, fixed 3,084 B),
# and K2/K5 are then EVALUATED from it rather than guessed:
#   K2 = 2,362,380 B/rank (9,449,520 across TP4)
#   K5 = 5,901,324 B/rank (23,605,296 across TP4)
# The same construction was validated end-to-end against two real
# checkpoints of the small GLM-5.2-arch proxy: deriving the model from
# fruit-k3's header alone predicts fruit-k4's measured 814,128 B/expert
# to the byte (see test_memory_budget_cpu.py).
MEASURED_GLM52_TP4_PER_RANK: dict[int, int] = {K3: 3_542_028, K4: 4_721_676}
MEASURED_GLM52_TP4_ALL_RANKS: dict[int, int] = {K3: 14_168_112, K4: 18_886_704}


def reference_expert_bytes(*, per_rank: bool = True) -> ExpertBytes:
    """The built-in GLM-5.2/TP4 measurement — a LAST RESORT.

    Marked ``is_reference``: callers must say out loud that the budget is
    being computed from another checkpoint's geometry, because it may not
    match the one actually loaded (requirement: never silently use a
    constant that might be wrong).
    """
    points = (MEASURED_GLM52_TP4_PER_RANK if per_rank
              else MEASURED_GLM52_TP4_ALL_RANKS)
    scope = "per rank" if per_rank else "all TP4 ranks"
    return ExpertBytes.from_measurements(
        points,
        provenance=f"REFERENCE measurement GLM-5.2 TP4 SM120 ({scope}) — "
                   f"NOT derived from the loaded checkpoint",
        is_reference=True)


@dataclass(frozen=True)
class BudgetVerdict:
    """Outcome of checking one proposal against the byte ceiling."""

    ok: bool
    limit_bytes: int | None
    current_bytes: int
    requested_bytes: int
    overshoot_bytes: int
    headroom_bytes: int | None
    headroom_promotions: int | None
    kind: str = "proposal"
    layer: int | None = None
    expert_in: int | None = None
    expert_out: int | None = None

    def as_dict(self) -> dict:
        d = {
            "ok": bool(self.ok),
            "kind": self.kind,
            "budget_bytes": self.limit_bytes,
            "current_bytes": int(self.current_bytes),
            "requested_bytes": int(self.requested_bytes),
            "overshoot_bytes": int(self.overshoot_bytes),
            "headroom_bytes": self.headroom_bytes,
            "headroom_promotions": self.headroom_promotions,
        }
        for key in ("layer", "expert_in", "expert_out"):
            if getattr(self, key) is not None:
                d[key] = int(getattr(self, key))
        return d

    def message(self) -> str:
        return rejection_message(self.as_dict())


def rejection_message(rej: Mapping) -> str:
    """One log line carrying every number a reviewer needs."""
    where = ""
    if rej.get("layer") is not None:
        where = f" L{rej['layer']}"
        if rej.get("expert_in") is not None:
            where += f" e{rej['expert_in']}"
        if rej.get("expert_out") is not None:
            where += f"<-e{rej['expert_out']}"
    return (
        f"{rej.get('kind', 'proposal')}{where} REJECTED by memory budget: "
        f"budget={rej['budget_bytes']} B current={rej['current_bytes']} B "
        f"requested={rej['requested_bytes']} B "
        f"overshoot={rej['overshoot_bytes']} B")


class MemoryBudget:
    """A byte ceiling on the per-device expert pool, plus the conversion
    to and from the "N experts at K4 per layer" form.

    ``limit_bytes is None`` means no byte ceiling — the fixed-cardinality
    budget alone governs, which is exactly today's behaviour.

    Every byte figure this object reports is computed from the ACTUAL
    tier occupancy it is handed, never from an assumed shape: pass the
    live ``tier_of`` array and you get the live footprint, including
    experts sitting at tiers outside the K3/K4 ladder.
    """

    def __init__(
        self,
        expert_bytes: ExpertBytes,
        limit_bytes: int | None = None,
        *,
        low_k: int = K3,
        high_k: int = K4,
        spec: Any = None,
    ) -> None:
        self.expert_bytes = expert_bytes
        self.limit_bytes = None if limit_bytes is None else int(limit_bytes)
        if self.limit_bytes is not None and self.limit_bytes <= 0:
            raise ValueError(f"memory budget must be positive, got {limit_bytes}")
        self.low_k = int(low_k)
        self.high_k = int(high_k)
        self.spec = spec

    @classmethod
    def from_spec(
        cls,
        spec,
        expert_bytes: ExpertBytes,
        *,
        num_layers: int,
        num_experts: int,
        device_total_bytes: int | None = None,
        low_k: int = K3,
        high_k: int = K4,
    ) -> "MemoryBudget":
        """Build a budget from any of the operator's three spellings.

        - bytes / fraction of device memory -> :func:`parse_memory_budget`
        - ``"24/layer"`` / ``"24experts/layer"`` -> that many ``high_k``
          experts per layer, converted to the byte ceiling it implies
        - ``"3.5bpw"`` -> the mean-bits-per-expert the pool may average,
          converted the same way (rounded DOWN, so the resulting ceiling
          never exceeds what was asked for)

        The last two are the "experts/bpw/layer" form; they are stored as
        bytes, so enforcement, headroom and reporting all run through one
        code path regardless of how the ceiling was written.
        """
        probe = cls(expert_bytes, None, low_k=low_k, high_k=high_k, spec=spec)
        s = ("" if spec is None else
             str(spec).strip().lower().replace("_", "").replace(" ", ""))
        n_high: int | None = None
        m = _PER_LAYER_RE.match(s)
        if m is not None:
            n_high = int(m.group(1))
            if n_high > num_experts:
                raise ValueError(
                    f"memory budget {spec!r} asks for {n_high} {high_k}-bit "
                    f"experts per layer but each layer only has {num_experts}")
        else:
            m = _BPW_RE.match(s)
            if m is not None:
                bpw = float(m.group(1))
                if bpw < low_k:
                    raise ValueError(
                        f"memory budget {spec!r} is below the base tier "
                        f"K{low_k}: even an all-K{low_k} pool averages "
                        f"{low_k} bits/expert")
                n_high = min(num_experts, int(math.floor(
                    num_experts * (bpw - low_k) / (high_k - low_k))))
        if n_high is not None:
            limit = probe.bytes_for_cardinality(num_layers, num_experts,
                                                n_high)
        else:
            limit = parse_memory_budget(
                spec, device_total_bytes=device_total_bytes)
        return cls(expert_bytes, limit, low_k=low_k, high_k=high_k, spec=spec)

    # ------------------------------------------------------ measurement

    def bytes_of_counts(self, counts: Mapping[int, int]) -> int:
        """Bytes for ``{K: number_of_experts}`` (one rank)."""
        return sum(self.expert_bytes.bytes_for(k) * int(n)
                   for k, n in counts.items() if n)

    def counts_of(self, tier_of) -> dict[int, int]:
        """ACTUAL ``{K: expert_count}`` of a membership array."""
        vals, cnts = np.unique(np.asarray(tier_of), return_counts=True)
        return {int(k): int(n) for k, n in zip(vals, cnts)}

    def used_bytes(self, tier_of) -> int:
        """Per-rank bytes the given occupancy actually costs."""
        return self.bytes_of_counts(self.counts_of(tier_of))

    def promotion_cost(self) -> int:
        return self.expert_bytes.promotion_cost(self.low_k, self.high_k)

    def headroom_bytes(self, tier_of) -> int | None:
        if self.limit_bytes is None:
            return None
        return self.limit_bytes - self.used_bytes(tier_of)

    def promotions_headroom(self, tier_of) -> int | None:
        """How many more ``low_k -> high_k`` promotions fit.

        None when unbounded. Never negative-by-rounding: an over-budget
        occupancy reports a negative count, which is the honest answer
        ("you are this many promotions past the line"), and it is capped
        by the number of ``low_k`` experts that actually exist.
        """
        free = self.headroom_bytes(tier_of)
        if free is None:
            return None
        cost = self.promotion_cost()
        if cost <= 0:
            return 0
        n = int(math.floor(free / cost))
        if n <= 0:
            return n
        return min(n, int(self.counts_of(tier_of).get(self.low_k, 0)))

    def mean_bits(self, tier_of) -> float:
        counts = self.counts_of(tier_of)
        n = sum(counts.values())
        return (sum(k * v for k, v in counts.items()) / n) if n else 0.0

    # ------------------------------------------------------ conversion

    def n_high_per_layer(self, num_layers: int, num_experts: int) -> int | None:
        """Byte ceiling -> the equivalent fixed-cardinality budget.

        The largest uniform per-layer ``high_k`` count that fits, assuming
        every other expert sits at ``low_k``. None when unbounded. This is
        the exact inverse of :meth:`bytes_for_cardinality`.
        """
        if self.limit_bytes is None:
            return None
        num_layers, num_experts = int(num_layers), int(num_experts)
        base = num_layers * num_experts * self.expert_bytes.bytes_for(self.low_k)
        cost = num_layers * self.promotion_cost()
        if cost <= 0:
            return num_experts
        n = (self.limit_bytes - base) // cost
        return int(max(0, min(num_experts, n)))

    def bytes_for_cardinality(self, num_layers: int, num_experts: int,
                              n_high_per_layer: int) -> int:
        """Fixed-cardinality budget -> bytes (one rank)."""
        num_layers, num_experts = int(num_layers), int(num_experts)
        n_high = int(n_high_per_layer)
        if not 0 <= n_high <= num_experts:
            raise ValueError(
                f"n_high_per_layer {n_high} outside [0, {num_experts}]")
        return num_layers * (
            (num_experts - n_high) * self.expert_bytes.bytes_for(self.low_k)
            + n_high * self.expert_bytes.bytes_for(self.high_k))

    # ------------------------------------------------------ enforcement

    def check(self, current_tier, proposed_tier, *, kind: str = "proposal",
              layer: int | None = None, expert_in: int | None = None,
              expert_out: int | None = None) -> BudgetVerdict:
        """Is ``proposed_tier`` affordable? Carries the numbers either way."""
        cur = self.used_bytes(current_tier)
        req = self.used_bytes(proposed_tier)
        over = 0 if self.limit_bytes is None else max(0, req - self.limit_bytes)
        return BudgetVerdict(
            ok=(self.limit_bytes is None or req <= self.limit_bytes),
            limit_bytes=self.limit_bytes,
            current_bytes=cur, requested_bytes=req, overshoot_bytes=over,
            headroom_bytes=(None if self.limit_bytes is None
                            else self.limit_bytes - req),
            headroom_promotions=self.promotions_headroom(proposed_tier),
            kind=kind, layer=layer, expert_in=expert_in, expert_out=expert_out)

    def summary(self, tier_of) -> dict:
        """Operator-facing state, for the log table / metrics / record."""
        used = self.used_bytes(tier_of)
        return {
            "limit_bytes": self.limit_bytes,
            "used_bytes": used,
            "headroom_bytes": (None if self.limit_bytes is None
                               else self.limit_bytes - used),
            "headroom_promotions": self.promotions_headroom(tier_of),
            "promotion_cost_bytes": self.promotion_cost(),
            "per_k_bytes": self.expert_bytes.table(),
            "mean_bits_per_expert": self.mean_bits(tier_of),
            "source": self.expert_bytes.provenance,
            "is_reference": bool(self.expert_bytes.is_reference),
            "spec": None if self.spec is None else str(self.spec),
        }


def swap_byte_delta(tier_of, layer: int, e_out: int, e_in: int,
                    budget: MemoryBudget) -> int:
    """Signed byte cost of ``(layer, e_out, e_in)`` given the CURRENT tiers.

    ``e_out`` lands on ``budget.low_k`` and ``e_in`` on ``budget.high_k``;
    what each of them costs today is read off ``tier_of`` rather than
    assumed, so a member sitting at K2 or K5 is priced correctly.

    Note the consequence for today's engine: ``apply_swaps`` only accepts
    a K4 leaving and a K3 entering, so every swap it can execute has a
    delta of exactly zero. The fixed-cardinality budget is therefore an
    EXACT memory invariant on the K3/K4 ladder — and this function is
    what keeps that true by construction instead of by assumption once
    the ladder widens.
    """
    tier_of = np.asarray(tier_of)
    eb = budget.expert_bytes
    return ((eb.bytes_for(budget.low_k) - eb.bytes_for(tier_of[layer, e_out]))
            + (eb.bytes_for(budget.high_k) - eb.bytes_for(tier_of[layer, e_in])))


def budget_filter(swaps: list, tier_of, budget: MemoryBudget | None,
                  ) -> tuple[list, list[dict]]:
    """Guard 5 — drop swaps that would push the pool over the byte ceiling.

    Applied to the ORDERED list cumulatively, so the surviving prefix is
    affordable *together*, not merely one at a time.

    A swap that does not GROW the pool (``delta <= 0``) is always
    admitted, including when the occupancy is already over the ceiling.
    Refusing those would be a deadlock, not a guard: every executable
    swap on the K3/K4 ladder is byte-neutral, so an over-budget pool
    would never be allowed to re-tier again — and, because demotions
    only ever happen paired with a promotion, its occupancy could never
    come back down either. The ceiling is a bound on growth; a
    non-increasing move cannot violate it any harder than the state it
    started from.

    Returns ``(kept, rejections)``; each rejection is a
    :meth:`BudgetVerdict.as_dict` carrying budget/current/requested/
    overshoot.
    """
    if budget is None or budget.limit_bytes is None or not swaps:
        return list(swaps), []
    tier_of = np.asarray(tier_of)
    used = budget.used_bytes(tier_of)
    kept: list = []
    rejections: list[dict] = []
    for l, e_out, e_in in swaps:
        delta = swap_byte_delta(tier_of, l, e_out, e_in, budget)
        requested = used + delta
        if delta <= 0 or requested <= budget.limit_bytes:
            used = requested
            kept.append((l, e_out, e_in))
        else:
            rejections.append(BudgetVerdict(
                ok=False, limit_bytes=budget.limit_bytes, current_bytes=used,
                requested_bytes=requested,
                overshoot_bytes=requested - budget.limit_bytes,
                headroom_bytes=budget.limit_bytes - requested,
                headroom_promotions=int(math.floor(
                    (budget.limit_bytes - used) / budget.promotion_cost()))
                if budget.promotion_cost() > 0 else 0,
                kind="swap", layer=int(l), expert_in=int(e_in),
                expert_out=int(e_out)).as_dict())
    return kept, rejections


def swap_touched(swaps: Iterable[tuple[int, int, int]]
                 ) -> list[tuple[int, int]]:
    """Every ``(layer, expert)`` an ordered swap list moves — BOTH sides.

    ``plan_promotions`` must exclude all of them, not just the entering
    ones. ``decide`` demotes the weakest K4 resident, but "weakest of the
    incumbents" is routinely still stronger than the best K3 expert
    elsewhere, so a promotion planner ranking on raw score will pick that
    expert straight back up in the SAME interval. The result is not
    merely churn: the emitted swap list says ``e_out -> K3`` while the
    proposed membership says ``e_out`` is still K4, so the instruction
    handed to the apply backend contradicts the state the loop then
    records. One expert moves once per interval, or not at all.
    """
    out: list[tuple[int, int]] = []
    for l, e_out, e_in in swaps:
        out.append((int(l), int(e_out)))
        out.append((int(l), int(e_in)))
    return out


def apply_promotions(tier_of, promotions: Iterable[tuple[int, int]],
                     k: int = K4):
    """Apply unpaired ``(layer, expert) -> k`` promotions (returns a copy)."""
    out = np.array(tier_of, copy=True)
    for l, e in promotions:
        out[l, e] = k
    return out


def plan_promotions(
    stats,
    eps,
    tier_of,
    budget: MemoryBudget,
    *,
    pins=None,
    dwell=None,
    cfg=None,
    exclude: Iterable[tuple[int, int]] = (),
    max_rejections: int = 8,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Spend byte headroom on UNPAIRED ``low_k -> high_k`` promotions.

    This is the capability the byte budget buys that fixed cardinality
    cannot express: when the ceiling leaves room, the best-scoring
    low-tier experts are promoted outright instead of having to displace
    an incumbent. Candidates are taken in global score order (ties broken
    by layer then expert id, so the plan is deterministic), must have a
    strictly positive score, must be dwell-ready and not pinned to
    ``low_k``, and are capped per layer and in total by the same knobs
    that cap swaps.

    Every candidate that was otherwise eligible and did not fit is
    reported as a rejection carrying budget/current/requested/overshoot
    — that is what lands in the decision log.

    Returns ``(promotions, rejections)`` where a promotion is
    ``(layer_row, expert)``.
    """
    full = _full_cfg(dict(cfg or {}, n_k4=0))
    tier_of = np.asarray(tier_of)
    L, E = tier_of.shape
    if budget.limit_bytes is None:
        return [], []
    s = score(stats, eps, cfg)
    pins = np.zeros((L, E), dtype=np.int64) if pins is None else np.asarray(pins)
    dwell_ok = (np.ones((L, E), dtype=bool) if dwell is None
                else np.asarray(dwell) >= full["dwell_steps"])
    blocked = {(int(l), int(e)) for l, e in exclude}

    cand: list[tuple[float, int, int]] = []
    for l in range(L):
        for e in range(E):
            if tier_of[l, e] != budget.low_k or pins[l, e] == PIN_K3:
                continue
            if not dwell_ok[l, e] or (int(l), int(e)) in blocked:
                continue
            if not s[l, e] > 0:
                continue
            cand.append((float(s[l, e]), int(l), int(e)))
    cand.sort(key=lambda c: (-c[0], c[1], c[2]))

    per_layer_cap = int(full["max_swaps_per_layer"])
    total_cap = int(cfg.get("max_promotions_total", full["max_swaps_total"])
                    if cfg else full["max_swaps_total"])
    # Byte accounting is incremental: the footprint is measured ONCE from
    # the real occupancy and then moved by the known per-promotion cost,
    # so a 92x160 tier array is not re-reduced per candidate.
    used = budget.used_bytes(tier_of)
    limit = budget.limit_bytes
    n_low_left = int(budget.counts_of(tier_of).get(budget.low_k, 0))
    promotions: list[tuple[int, int]] = []
    rejections: list[dict] = []
    per_layer: dict[int, int] = {}
    for _, l, e in cand:
        if len(promotions) >= total_cap:
            break
        if per_layer.get(l, 0) >= per_layer_cap:
            continue
        cost = budget.expert_bytes.promotion_cost(
            int(tier_of[l, e]), budget.high_k)
        requested = used + cost
        if requested <= limit:
            used = requested
            n_low_left -= 1
            promotions.append((l, e))
            per_layer[l] = per_layer.get(l, 0) + 1
        elif len(rejections) < max_rejections:
            step = budget.promotion_cost()
            fits = (min(int(math.floor((limit - used) / step)), n_low_left)
                    if step > 0 else 0)
            rejections.append(BudgetVerdict(
                ok=False, limit_bytes=limit, current_bytes=used,
                requested_bytes=requested, overshoot_bytes=requested - limit,
                headroom_bytes=limit - requested, headroom_promotions=fits,
                kind="promotion", layer=l, expert_in=e).as_dict())
    return promotions, rejections


def decide_with_budget(
    stats,
    eps,
    tier_of,
    budget: MemoryBudget | None,
    *,
    pins=None,
    dwell=None,
    cfg=None,
    allow_promotions: bool = True,
) -> tuple[list, list[tuple[int, int]], list[dict]]:
    """``decide`` + Guard 5 + headroom promotions, in one call.

    Returns ``(swaps, promotions, rejections)``. With ``budget`` None or
    unbounded this is exactly ``decide`` with two empty lists, so the
    byte budget is strictly additive to the existing behaviour.
    """
    swaps = decide(stats, eps, tier_of, pins=pins, dwell=dwell, cfg=cfg)
    if budget is None or budget.limit_bytes is None:
        return swaps, [], []
    swaps, rejections = budget_filter(swaps, tier_of, budget)
    promotions: list[tuple[int, int]] = []
    if allow_promotions:
        touched = swap_touched(swaps)
        promotions, prej = plan_promotions(
            stats, eps, apply_swaps(tier_of, swaps), budget, pins=pins,
            dwell=dwell, cfg=cfg, exclude=touched)
        rejections = rejections + prej
    return swaps, promotions, rejections
