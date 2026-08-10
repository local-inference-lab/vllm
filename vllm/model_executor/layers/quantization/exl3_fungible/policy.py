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

import numpy as np

K3, K4 = 3, 4
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
