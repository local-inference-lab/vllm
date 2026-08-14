# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explainable decision records for fungible-quant swaps (M2 gate).

`policy.decide()` stays a pure function; this module recomputes the same
deterministic quantities to explain a decision after the fact — one
structured record per interval answering WHEN and WHY for every swap
(scores, ε-gap, routing mass, hysteresis ratio) and why candidates were
NOT swapped (dwell/hysteresis/cap tallies). Emits:

- one EPLB-style INFO line per interval (totals + budget state),
- one INFO line per executed swap with its full rationale,
- a JSON record for persistence beside the policy history
  (store.PolicyStore.dump_stats-compatible), enabling the M2 "manual
  review of explainable proposals" gate and post-hoc audit.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from vllm.model_executor.layers.quantization.exl3_fungible import policy as P

logger = logging.getLogger(__name__)


def explain(
    stats: dict,
    eps: Any,
    tier_of: Any,
    swaps: list[tuple[int, int, int]],
    *,
    pins: Any = None,
    dwell: Any = None,
    cfg: dict | None = None,
    step: int | None = None,
    policy_sha_before: str | None = None,
    policy_sha_after: str | None = None,
) -> dict:
    """Build the decision record for one interval.

    Inputs mirror policy.decide(); ``swaps`` is decide()'s output for the
    same inputs. All quantities recomputed here are byte-deterministic
    reproductions of decide()'s internals.
    """
    full = dict(P.DEFAULT_CFG, **(cfg or {}))
    s = P.score(stats, eps, cfg)
    tier_of = np.asarray(tier_of)
    L, E = tier_of.shape
    count = np.asarray(stats["count"], dtype=np.float64)
    mass = np.asarray(stats["mass"], dtype=np.float64)
    e3, e4 = P._as_eps(eps, L, E)
    h = float(full.get("hysteresis", P.DEFAULT_CFG["hysteresis"]))

    swap_records = []
    for layer, e_out, e_in in swaps:
        swap_records.append({
            "layer": int(layer), "expert_out": int(e_out),
            "expert_in": int(e_in),
            "score_in": float(s[layer, e_in]),
            "score_out": float(s[layer, e_out]),
            "score_gap": float(s[layer, e_in] - s[layer, e_out]),
            "hysteresis_ratio": (
                float(s[layer, e_in] / s[layer, e_out])
                if s[layer, e_out] > 0 else float("inf")),
            "eps_gap_in": float(e3[layer, e_in] - e4[layer, e_in]),
            "eps_gap_out": float(e3[layer, e_out] - e4[layer, e_out]),
            "count_in": float(count[layer, e_in]),
            "count_out": float(count[layer, e_out]),
            "mass_in": float(mass[layer, e_in]),
            "mass_out": float(mass[layer, e_out]),
        })

    # Why-not tallies: candidates that the guards held back, recomputed
    # with decide()'s exact ordering semantics.
    dwell_ok = (np.ones((L, E), bool) if dwell is None
                else np.asarray(dwell) >= full["dwell_steps"])
    blocked = {"dwell": 0, "hysteresis": 0, "layer_cap": 0}
    per_layer_pairs = {}
    for l in range(L):
        cur = tier_of[l] == P.K4
        n_k4 = int(cur.sum())
        order = np.lexsort((np.arange(E), -s[l]))
        desired = np.zeros(E, bool)
        desired[order[:n_k4]] = True
        enter = sorted(np.nonzero(desired & ~cur)[0], key=lambda e: (-s[l, e], e))
        leave = sorted(np.nonzero(cur & ~desired)[0], key=lambda e: (s[l, e], e))
        eligible = 0
        for e_in, e_out in zip(enter, leave):
            if not dwell_ok[l, e_in] or not dwell_ok[l, e_out]:
                blocked["dwell"] += 1
                continue
            if not (s[l, e_in] > h * s[l, e_out]):
                blocked["hysteresis"] += 1
                continue
            eligible += 1
        executed = sum(1 for (ll, _, _) in swaps if ll == l)
        blocked["layer_cap"] += max(0, eligible - executed)
        if eligible or executed:
            per_layer_pairs[str(l)] = {"eligible": eligible, "executed": executed}

    record = {
        "schema": "fq-decision/1",
        "step": step,
        "policy_sha_before": policy_sha_before,
        "policy_sha_after": policy_sha_after,
        "config": {k: full[k] for k in
                   ("hysteresis", "dwell_steps", "max_swaps_per_layer",
                    "max_swaps_total") if k in full},
        "swaps": swap_records,
        "blocked": blocked,
        "per_layer": per_layer_pairs,
        "totals": {"executed": len(swaps),
                   "layers_touched": len({sw[0] for sw in swaps})},
    }
    return record


def log_decision(record: dict) -> None:
    """Emit the EPLB-style interval line + one line per swap."""
    t, b = record["totals"], record["blocked"]
    logger.info(
        "FQ interval step=%s: %d swaps across %d layers "
        "(blocked: dwell=%d hysteresis=%d cap=%d) policy %s -> %s",
        record.get("step"), t["executed"], t["layers_touched"],
        b["dwell"], b["hysteresis"], b["layer_cap"],
        (record.get("policy_sha_before") or "?")[:8],
        (record.get("policy_sha_after") or "?")[:8])
    for sw in record["swaps"]:
        logger.info(
            "FQ swap L%d: K3->K4 e%d (score %.3e, eps_gap %.3e, mass %.1f) "
            "displaces e%d (score %.3e) ratio=%.2f",
            sw["layer"], sw["expert_in"], sw["score_in"], sw["eps_gap_in"],
            sw["mass_in"], sw["expert_out"], sw["score_out"],
            sw["hysteresis_ratio"])


def to_json(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))
