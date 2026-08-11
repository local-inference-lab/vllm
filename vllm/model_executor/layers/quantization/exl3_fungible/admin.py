# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant admin API: operator-forced expert re-tiering.

Implements ``runs/m5-serve/admin-api-spec.md``. The operator's ask was
"expose (optionally) an admin api ... that would allow to force
upgrade/downgrade of specific experts K -- for example
layer=23,expert=250,adjust_k=-1 (relative, or absolute like adjust_k=3)
and as a batch ... this being guarded by maximum memory usage".

Design stance, restated because it is the whole point: this module is
**glue plus guards**. It does not invent a second way to move an expert
between tiers. The target ``fq-policy/2`` document is built, handed to
:meth:`SwapPlan.from_policies` (which already does pairing, ordering and
the same-cardinality check) and then to :meth:`SwapEngine.apply` (which
already does the six-step commit protocol and persists the policy as
step 5 inside the quiesce window). ``swap.py``, ``store.py``,
``policy.py``, ``fragments.py`` and ``occupancy_table.py`` are unchanged.

Gating — OFF by default, twice over:

1. ``VLLM_SERVER_DEV_MODE`` — vLLM's own dev-endpoint gate. Without it
   ``register_vllm_dev_api_routers`` never runs, so the router is never
   attached.
2. ``VLLM_FQ_ADMIN_API=1`` — a second, FQ-specific gate. Dev mode gets
   turned on for lots of reasons; forced mutation of live weights should
   require saying so explicitly. (``VLLM_FQ_ADMIN_ENABLE`` is accepted as
   an alias, which is the name the spec used.)

Both are checked in three places, because the router is not the only way
in: at attach time (the routes do not exist otherwise), on every request
in the router's ``_guard``, and — the one that actually matters — in
``worker_describe``/``worker_plan``/``worker_apply``. Dev mode also
attaches vLLM's generic ``POST /collective_rpc``, which forwards any
``method`` name to every worker, and ``Worker.fq_admin_apply`` is an
unconditional method; without the worker-side check
``VLLM_SERVER_DEV_MODE`` alone would mutate live weights and
``VLLM_FQ_ADMIN_TOKEN`` would never be consulted. See
:func:`_require_gates`.

Fixed cardinality is not negotiable here. Occupancy == capacity is
enforced in four independent places (``store.validate_policy``,
``SwapPlan.from_memberships``, ``SwapEngine._validate_layer``,
``policy.decide``), and runtime budget *growth* is structurally
impossible: tier slabs are sized at prepare time and the mixed-kernel
memo key carries per-tier expert counts, so changing a count forces a
recompile that is refused under CUDA-graph capture. ``mode=grow_budget``
therefore returns 501 with all three reasons rather than pretending.

Laziness contract (same as ``integration.py``): nothing from ``vllm`` and
nothing from ``fastapi`` is imported at module level, so this module is
importable — and CPU-testable — without a built ``vllm._C``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vllm.model_executor.layers.quantization.exl3_fungible import (
    decision_log as DL,
)
from vllm.model_executor.layers.quantization.exl3_fungible import (
    occupancy_table as OT,
)
from vllm.model_executor.layers.quantization.exl3_fungible import policy as P
from vllm.model_executor.layers.quantization.exl3_fungible import swap as SW
from vllm.model_executor.layers.quantization.exl3_fungible.store import (
    policy_hash,
    validate_policy,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ env knobs
DEV_MODE_ENV = "VLLM_SERVER_DEV_MODE"
# The task-mandated primary gate. VLLM_FQ_ADMIN_ENABLE is the name the
# spec used; both are honoured so neither document is wrong.
ADMIN_API_ENV = "VLLM_FQ_ADMIN_API"
ADMIN_ENABLE_ENV = "VLLM_FQ_ADMIN_ENABLE"
ADMIN_TOKEN_ENV = "VLLM_FQ_ADMIN_TOKEN"
ADMIN_MAX_ITEMS_ENV = "VLLM_FQ_ADMIN_MAX_ITEMS"
ADMIN_MEM_BUDGET_ENV = "VLLM_FQ_ADMIN_MEM_BUDGET_BYTES"
ADMIN_MEM_HEADROOM_ENV = "VLLM_FQ_ADMIN_MEM_HEADROOM_BYTES"
ADMIN_MIN_FREE_SLOTS_ENV = "VLLM_FQ_ADMIN_MIN_FREE_SLOTS"
ADMIN_DRAIN_TIMEOUT_ENV = "VLLM_FQ_ADMIN_DRAIN_TIMEOUT_S"
ADMIN_ALLOW_AUTO_BALANCE_ENV = "VLLM_FQ_ADMIN_ALLOW_AUTO_BALANCE"

DEFAULT_MAX_ITEMS = 32
DEFAULT_MEM_HEADROOM_BYTES = 1 << 30
DEFAULT_MIN_FREE_SLOTS = 2
DEFAULT_DRAIN_TIMEOUT_S = 120.0

#: The tiers the v1 swap engine can actually serve as a MIXED pair.
#: ``MixedLayerState.from_exl3_mixed_trellis`` refuses anything but (3, 4),
#: and K5 as a mixed tier does not fit SM120 shared memory at all
#: (109568 > 101376; K4 is exactly 101376, zero headroom) — see
#: runs/m5-serve/k5-shared-memory-limit.md.
SERVABLE_TIERS: tuple[int, ...] = (P.K3, P.K4)

MODE_STRICT_PAIR = "strict_pair"
MODE_AUTO_BALANCE = "auto_balance"
MODE_GROW_BUDGET = "grow_budget"

PIN_HOLD, PIN_NONE, PIN_RELEASE = "hold", "none", "release"

_REQUEST_FIELDS = frozenset({
    "items", "mode", "dry_run", "pin", "drain_mode", "reset_prefix_cache",
    "expect_policy_sha", "timeout_s", "actor", "reason",
})
_ITEM_FIELDS = frozenset({"layer", "expert", "adjust_k", "k", "delta_k"})


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _env_int(environ, name: str, default: int) -> int:
    raw = _env(environ).get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("FQ admin: %s=%r is not an int — using %d",
                       name, raw, default)
        return default


def _env_float(environ, name: str, default: float) -> float:
    raw = _env(environ).get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("FQ admin: %s=%r is not a float — using %s",
                       name, raw, default)
        return default


def dev_mode_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return str(_env(environ).get(DEV_MODE_ENV, "0")).strip() in ("1", "true",
                                                                 "True")


def admin_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Both gates. OFF unless dev mode AND the FQ admin flag are set."""
    env = _env(environ)
    flag = (str(env.get(ADMIN_API_ENV, "")).strip()
            or str(env.get(ADMIN_ENABLE_ENV, "")).strip())
    return dev_mode_enabled(environ) and flag in ("1", "true", "True")


def gate_reason(environ: Mapping[str, str] | None = None) -> str:
    """Human-readable reason the API is off (for the 404 body)."""
    if not dev_mode_enabled(environ):
        return (f"{DEV_MODE_ENV} is not set — vLLM development endpoints are "
                "disabled")
    return (f"{ADMIN_API_ENV}=1 is required to expose forced expert "
            "re-tiering (dev mode alone is not enough)")


# ------------------------------------------------------------------- errors


class AdminError(Exception):
    """A refusal with a machine-readable body.

    An admin API is machine-driven; a bare ``detail`` string is not
    parseable. Every refusal carries ``code`` (stable), ``message``
    (actionable prose) and ``details`` (the numbers to act on).
    """

    def __init__(self, code: str, message: str, *, status: int = 400,
                 details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = int(status)
        self.details = details or {}

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message,
                          "details": self.details}}

    def wire(self) -> dict:
        """Serialisable form for the worker -> router hop."""
        return {"code": self.code, "message": self.message,
                "http_status": self.status, "details": self.details}

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "AdminError":
        return cls(str(payload.get("code", "internal_error")),
                   str(payload.get("message", "")),
                   status=int(payload.get("http_status", 500)),
                   details=dict(payload.get("details") or {}))


def _fmt_bytes(n: int) -> str:
    n = int(n)
    sign = "-" if n < 0 else ""
    a = abs(n)
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if a >= scale:
            return f"{sign}{a / scale:.3f} {unit}"
    return f"{sign}{a} B"


# ------------------------------------------------------------------ requests


@dataclass(frozen=True)
class RetierItem:
    """One parsed ``items[]`` entry, before it meets live state."""

    layer: int
    expert: int
    requested: Any                 # verbatim, echoed back to the operator
    interpretation: str            # "absolute" | "relative"
    k: int | None = None
    delta_k: int | None = None


@dataclass(frozen=True)
class RetierRequest:
    items: tuple[RetierItem, ...]
    mode: str = MODE_STRICT_PAIR
    dry_run: bool = False
    pin: str = PIN_HOLD
    drain_mode: str = "wait"
    reset_prefix_cache: bool = False
    expect_policy_sha: str | None = None
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S
    actor: str | None = None
    reason: str | None = None

    def canonical(self) -> dict:
        return {
            "items": [
                {"layer": it.layer, "expert": it.expert,
                 "requested": it.requested,
                 "interpretation": it.interpretation,
                 "k": it.k, "delta_k": it.delta_k}
                for it in self.items
            ],
            "mode": self.mode,
            "dry_run": self.dry_run,
            "pin": self.pin,
            "drain_mode": self.drain_mode,
            "reset_prefix_cache": self.reset_prefix_cache,
            "expect_policy_sha": self.expect_policy_sha,
            "timeout_s": self.timeout_s,
            "actor": self.actor,
            "reason": self.reason,
        }

    @classmethod
    def from_canonical(cls, doc: Mapping[str, Any]) -> "RetierRequest":
        items = tuple(
            RetierItem(layer=int(d["layer"]), expert=int(d["expert"]),
                       requested=d.get("requested"),
                       interpretation=str(d["interpretation"]),
                       k=None if d.get("k") is None else int(d["k"]),
                       delta_k=(None if d.get("delta_k") is None
                                else int(d["delta_k"])))
            for d in doc.get("items", ())
        )
        return cls(
            items=items,
            mode=str(doc.get("mode", MODE_STRICT_PAIR)),
            dry_run=bool(doc.get("dry_run", False)),
            pin=str(doc.get("pin", PIN_HOLD)),
            drain_mode=str(doc.get("drain_mode", "wait")),
            reset_prefix_cache=bool(doc.get("reset_prefix_cache", False)),
            expect_policy_sha=doc.get("expect_policy_sha"),
            timeout_s=float(doc.get("timeout_s", DEFAULT_DRAIN_TIMEOUT_S)),
            actor=doc.get("actor"),
            reason=doc.get("reason"),
        )


_AMBIGUOUS_MSG = (
    'adjust_k={n} is not a valid absolute tier (ladder is K3/K4). For a '
    'relative +{n}, send the string "+{n}", or use the explicit field '
    'delta_k: {n}.'
)


def _parse_adjust(raw: Any, index: int) -> tuple[str, int | None, int | None]:
    """``adjust_k`` disambiguation (spec §3.2).

    ``+1`` is not legal JSON, so the sign has to survive some other way:

    ==========  ======================  ====================================
    sent        interpretation          why it is unambiguous
    ==========  ======================  ====================================
    ``-1``      relative ``delta=-1``   no negative K exists
    ``3``       absolute ``k=3``        matches "absolute like adjust_k=3"
    ``"+1"``    relative ``delta=+1``   explicit sign
    ``"-1"``    relative ``delta=-1``   explicit sign
    ``"3"``     absolute ``k=3``        no sign
    ``1``/``0`` 400 ambiguous_adjust_k  K1/K0 do not exist
    ==========  ======================  ====================================
    """
    if isinstance(raw, bool):
        raise AdminError("bad_adjust_k",
                         f"items[{index}].adjust_k must be a number or a "
                         f"signed string, got a boolean",
                         details={"index": index, "adjust_k": raw})
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise AdminError("bad_adjust_k",
                             f"items[{index}].adjust_k is empty",
                             details={"index": index})
        signed = text[0] in "+-"
        try:
            value = int(text, 10)
        except ValueError:
            raise AdminError("bad_adjust_k",
                             f"items[{index}].adjust_k={raw!r} is not an "
                             f"integer",
                             details={"index": index, "adjust_k": raw}) from None
        if signed:
            return "relative", None, value
        return "absolute", value, None
    if isinstance(raw, int):
        if raw < 0:
            return "relative", None, raw
        if raw in SERVABLE_TIERS:
            return "absolute", raw, None
        raise AdminError(
            "ambiguous_adjust_k",
            _AMBIGUOUS_MSG.format(n=raw),
            details={"index": index, "adjust_k": raw,
                     "ladder": list(SERVABLE_TIERS)})
    raise AdminError("bad_adjust_k",
                     f"items[{index}].adjust_k={raw!r} is neither a number "
                     f"nor a signed string",
                     details={"index": index, "adjust_k": raw})


def _parse_item(raw: Any, index: int) -> RetierItem:
    if not isinstance(raw, Mapping):
        raise AdminError("bad_json", f"items[{index}] is not an object",
                         details={"index": index})
    unknown = sorted(set(raw) - _ITEM_FIELDS)
    if unknown:
        raise AdminError("unknown_field",
                         f"items[{index}] has unrecognised key(s) {unknown}",
                         details={"index": index, "unknown": unknown,
                                  "allowed": sorted(_ITEM_FIELDS)})
    for name in ("layer", "expert"):
        if name not in raw:
            raise AdminError("bad_json", f"items[{index}].{name} is required",
                             details={"index": index})
    try:
        layer = int(raw["layer"])
        expert = int(raw["expert"])
    except (TypeError, ValueError):
        raise AdminError("bad_json",
                         f"items[{index}]: layer/expert must be integers",
                         details={"index": index}) from None

    present = [n for n in ("adjust_k", "k", "delta_k") if n in raw]
    if len(present) != 1:
        raise AdminError(
            "bad_adjust_k",
            f"items[{index}]: exactly one of adjust_k / k / delta_k is "
            f"required, got {present or 'none'}",
            details={"index": index, "present": present})
    name = present[0]
    value = raw[name]
    if name == "adjust_k":
        interpretation, k, delta = _parse_adjust(value, index)
    elif name == "k":
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdminError("bad_adjust_k",
                             f"items[{index}].k must be an integer",
                             details={"index": index, "k": value})
        interpretation, k, delta = "absolute", int(value), None
    else:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdminError("bad_adjust_k",
                             f"items[{index}].delta_k must be an integer",
                             details={"index": index, "delta_k": value})
        interpretation, k, delta = "relative", None, int(value)
    return RetierItem(layer=layer, expert=expert, requested=value,
                      interpretation=interpretation, k=k, delta_k=delta)


def parse_request(body: Mapping[str, Any] | None, *,
                  query: Mapping[str, Any] | None = None,
                  actor_header: str | None = None,
                  environ: Mapping[str, str] | None = None) -> RetierRequest:
    """Validate and desugar a ``POST /fq/retier`` payload.

    Accepts the single-item query-string shorthand the operator wrote
    (``?layer=23&expert=250&adjust_k=-1``) when there is no body; a body
    and query params together is ``400 mixed_input`` rather than a guess.

    Anything else in the query string is a ``400 unknown_field``, never a
    silent drop. Filtering it away instead made
    ``?layer=23&expert=250&adjust_k=-1&dry_run=true`` — the obvious
    spelling of the operator's "will this work?" button — perform a real,
    unbracketed weight mutation and answer ``applied: true``. The same
    silence disarmed ``expect_policy_sha`` (the optimistic-concurrency
    guard), ``mode``, ``pin``, ``timeout_s`` and the ``actor``/``reason``
    that make the audit record attributable.
    """
    query = dict(query or {})
    unknown_query = sorted(set(query) - _ITEM_FIELDS)
    if unknown_query:
        raise AdminError(
            "unknown_field",
            f"unrecognised query parameter(s) {unknown_query}. The "
            f"query-string shorthand carries one item and nothing else "
            f"({sorted(_ITEM_FIELDS)}); everything that changes what the "
            f"request DOES — dry_run, mode, pin, expect_policy_sha, "
            f"timeout_s, actor, reason — must go in the JSON body, or it "
            f"would be silently ignored.",
            details={"unknown": unknown_query,
                     "allowed_query": sorted(_ITEM_FIELDS),
                     "allowed_body": sorted(_REQUEST_FIELDS)})
    if query and body:
        raise AdminError("mixed_input",
                         "send either the query-string shorthand or a JSON "
                         "body, not both",
                         details={"query": sorted(query)})
    if query:
        # Query values arrive as strings. ``adjust_k`` keeps its string
        # form on purpose (that is how "+1" survives the wire at all), but
        # the explicit machine fields are numbers.
        shorthand = dict(query)
        for name in ("layer", "expert", "k", "delta_k"):
            if name in shorthand:
                try:
                    shorthand[name] = int(str(shorthand[name]).strip())
                except ValueError:
                    raise AdminError(
                        "bad_json",
                        f"query parameter {name}={shorthand[name]!r} is not "
                        f"an integer", details={name: shorthand[name]}
                    ) from None
        body = {"items": [shorthand]}
    if body is None:
        raise AdminError("bad_json", "a JSON body is required")
    if not isinstance(body, Mapping):
        raise AdminError("bad_json", "body must be a JSON object")

    unknown = sorted(set(body) - _REQUEST_FIELDS)
    if unknown:
        raise AdminError("unknown_field",
                         f"unrecognised top-level key(s) {unknown}",
                         details={"unknown": unknown,
                                  "allowed": sorted(_REQUEST_FIELDS)})

    raw_items = body.get("items", [])
    if not isinstance(raw_items, (list, tuple)):
        raise AdminError("bad_json", "items must be an array")
    max_items = _env_int(environ, ADMIN_MAX_ITEMS_ENV, DEFAULT_MAX_ITEMS)
    if len(raw_items) > max_items:
        raise AdminError("too_many_items",
                         f"{len(raw_items)} items exceeds "
                         f"{ADMIN_MAX_ITEMS_ENV}={max_items}",
                         details={"items": len(raw_items),
                                  "max_items": max_items})

    pin = str(body.get("pin", PIN_HOLD))
    if pin not in (PIN_HOLD, PIN_NONE, PIN_RELEASE):
        raise AdminError("unknown_field",
                         f"pin must be hold|none|release, got {pin!r}",
                         details={"pin": pin})
    if not raw_items and pin == PIN_NONE:
        raise AdminError("empty_request",
                         "no items and no pin change — nothing to do")

    items = tuple(_parse_item(raw, i) for i, raw in enumerate(raw_items))
    seen: dict[tuple[int, int], int] = {}
    for i, it in enumerate(items):
        key = (it.layer, it.expert)
        if key in seen:
            raise AdminError(
                "duplicate_item",
                f"(layer {it.layer}, expert {it.expert}) appears at items"
                f"[{seen[key]}] and items[{i}] — one entry per expert",
                details={"layer": it.layer, "expert": it.expert,
                         "indices": [seen[key], i]})
        seen[key] = i

    mode = str(body.get("mode", MODE_STRICT_PAIR))
    if mode not in (MODE_STRICT_PAIR, MODE_AUTO_BALANCE, MODE_GROW_BUDGET):
        raise AdminError("unknown_field",
                         f"mode must be strict_pair|auto_balance|grow_budget,"
                         f" got {mode!r}", details={"mode": mode})
    drain_mode = str(body.get("drain_mode", "wait"))
    if drain_mode not in ("wait", "abort", "keep"):
        raise AdminError("unknown_field",
                         f"drain_mode must be wait|abort|keep, got "
                         f"{drain_mode!r}", details={"drain_mode": drain_mode})

    timeout_s = body.get("timeout_s")
    if timeout_s is None:
        timeout_s = _env_float(environ, ADMIN_DRAIN_TIMEOUT_ENV,
                               DEFAULT_DRAIN_TIMEOUT_S)
    try:
        timeout_s = float(timeout_s)
    except (TypeError, ValueError):
        raise AdminError("bad_json", "timeout_s must be a number") from None

    return RetierRequest(
        items=items, mode=mode, dry_run=bool(body.get("dry_run", False)),
        pin=pin, drain_mode=drain_mode,
        reset_prefix_cache=bool(body.get("reset_prefix_cache", False)),
        expect_policy_sha=body.get("expect_policy_sha"),
        timeout_s=timeout_s,
        actor=body.get("actor") or actor_header,
        reason=body.get("reason"))


# ------------------------------------------------------------- resolved items


@dataclass(frozen=True)
class ResolvedItem:
    layer: int
    expert: int
    requested: Any
    interpretation: str
    k_from: int
    k_to: int
    outcome: str                   # promoted | demoted | noop
    source: str = "operator"

    def as_dict(self) -> dict:
        return {"layer": self.layer, "expert": self.expert,
                "requested": self.requested,
                "interpretation": self.interpretation,
                "k_from": self.k_from, "k_to": self.k_to,
                "outcome": self.outcome, "source": self.source}


def resolve_items(request: RetierRequest, *, layers: Sequence[int],
                  tier_of: Any, num_experts: int,
                  registered_layers: Sequence[int] | None = None,
                  ) -> list[ResolvedItem]:
    """Turn parsed items into (k_from, k_to) against the **live** state.

    ``k_from`` is always read from ``tier_of`` — never from anything the
    client sent — so a stale client view cannot silently mis-target.
    """
    tier_of = np.asarray(tier_of)
    rows = {int(l): i for i, l in enumerate(layers)}
    registered = None if registered_layers is None else set(
        int(x) for x in registered_layers)
    out: list[ResolvedItem] = []
    for item in request.items:
        row = rows.get(item.layer)
        if row is None:
            raise AdminError(
                "layer_not_registered",
                f"layer {item.layer} is not in the running policy "
                f"(layers {sorted(rows)[:8]}{'...' if len(rows) > 8 else ''})",
                status=404,
                details={"layer": item.layer, "known_layers": sorted(rows)})
        if registered is not None and item.layer not in registered:
            raise AdminError(
                "layer_not_registered",
                f"layer {item.layer} is in the policy but has no live "
                f"mixed-trellis state registered with the swap engine",
                status=404,
                details={"layer": item.layer,
                         "registered_layers": sorted(registered)})
        if not 0 <= item.expert < int(num_experts):
            raise AdminError(
                "expert_out_of_range",
                f"expert {item.expert} is outside [0, {int(num_experts)})",
                details={"layer": item.layer, "expert": item.expert,
                         "num_experts": int(num_experts)})
        k_from = int(tier_of[row, item.expert])
        k_to = int(item.k) if item.k is not None else k_from + int(item.delta_k)
        if k_to == k_from:
            outcome = "noop"
        elif k_to > k_from:
            outcome = "promoted"
        else:
            outcome = "demoted"
        if outcome != "noop" and k_to not in SERVABLE_TIERS:
            raise AdminError(
                "tier_not_servable",
                f"L{item.layer}/e{item.expert}: K{k_to} cannot be served as a "
                f"mixed tier. Two independent reasons: (a) the v1 swap engine "
                f"is a two-tier engine — MixedLayerState.from_exl3_mixed_"
                f"trellis refuses anything but "
                f"{tuple(SERVABLE_TIERS)}; (b) on SM120 a K5 mixed tier needs "
                f"109568 bytes of shared memory against a 101376 opt-in limit "
                f"(K4 is exactly 101376, zero headroom) — see "
                f"runs/m5-serve/k5-shared-memory-limit.md.",
                status=501,
                details={"layer": item.layer, "expert": item.expert,
                         "k_from": k_from, "k_to": k_to,
                         "ladder": list(SERVABLE_TIERS)})
        out.append(ResolvedItem(
            layer=item.layer, expert=item.expert, requested=item.requested,
            interpretation=item.interpretation, k_from=k_from, k_to=k_to,
            outcome=outcome))
    return out


def check_cardinality(resolved: Sequence[ResolvedItem], *, mode: str,
                      n_k4_per_layer: Mapping[str, int] | None = None,
                      environ: Mapping[str, str] | None = None) -> None:
    """The fixed-cardinality invariant, per layer, before anything moves.

    ``strict_pair`` (the default and the only mode v1 ships) requires
    ``#promotions == #demotions`` per layer. An override that silently
    makes a second, unrequested policy decision — which expert to
    sacrifice — is exactly the surprise an operator issues an override to
    avoid.
    """
    if mode == MODE_GROW_BUDGET:
        raise AdminError(
            "budget_growth_not_supported",
            "runtime budget growth is not supported, and not merely "
            "unbudgeted — it is structurally impossible. (1) The tier slabs "
            "are sized at prepare time: SwapEngine._validate_layer derives "
            "the slab word counts from len(tierN_globals) and refuses a "
            "mismatch, so one more K4 expert means reallocating tier1.w13/w2. "
            "(2) The mixed-kernel memo key carries per-tier expert COUNTS "
            "(tier_signature), so changing a count forces "
            "compile_mixed_trellis, which is refused under CUDA-graph capture "
            "and which on SM120 re-enters the tile-fit trap that blocks K5. "
            "(3) The bytes are not there: the KV-cache pool is sized from "
            "post-load free memory at startup. Raise n_k4_per_layer in the "
            "policy and restart — the committed policy steers the next boot.",
            status=501,
            details={"mode": mode, "growth_supported": False})
    if mode == MODE_AUTO_BALANCE:
        allow = str(_env(environ).get(ADMIN_ALLOW_AUTO_BALANCE_ENV, "0"))
        if allow.strip() not in ("1", "true", "True"):
            raise AdminError(
                "unknown_field",
                f"mode=auto_balance requires "
                f"{ADMIN_ALLOW_AUTO_BALANCE_ENV}=1 (it lets the server pick "
                f"which expert to sacrifice, which an override normally "
                f"should not do)",
                details={"mode": mode})
        raise AdminError(
            "unknown_field",
            "mode=auto_balance is declared but not implemented in v1 — use "
            "strict_pair and name the partner explicitly (GET /fq/layer/"
            "{layer} ranks the coldest K4 residents)",
            status=501, details={"mode": mode})

    per_layer: dict[int, dict[str, list]] = {}
    for it in resolved:
        bucket = per_layer.setdefault(
            it.layer, {"promotions": [], "demotions": [], "noop": []})
        if it.outcome == "promoted":
            bucket["promotions"].append(it)
        elif it.outcome == "demoted":
            bucket["demotions"].append(it)
        else:
            bucket["noop"].append(it)

    for layer in sorted(per_layer):
        bucket = per_layer[layer]
        ups, downs = bucket["promotions"], bucket["demotions"]
        if len(ups) == len(downs):
            continue
        cap = None
        if n_k4_per_layer is not None:
            cap = n_k4_per_layer.get(str(layer), n_k4_per_layer.get(layer))
        detail_ups = [{"expert": i.expert, "k_from": i.k_from,
                       "k_to": i.k_to} for i in ups]
        detail_downs = [{"expert": i.expert, "k_from": i.k_from,
                         "k_to": i.k_to} for i in downs]
        pretty = ", ".join(f"e{i.expert}: K{i.k_from}->K{i.k_to}"
                           for i in ups + downs) or "nothing"
        raise AdminError(
            "cardinality_unbalanced",
            f"layer {layer}: {len(ups)} promotion(s), {len(downs)} "
            f"demotion(s) ({pretty}). Fixed-cardinality invariant requires a "
            f"1-for-1 trade.",
            status=409,
            details={
                "layer": layer,
                "n_k4_per_layer": cap,
                "promotions": detail_ups,
                "demotions": detail_downs,
                "noop_items": [{"expert": i.expert, "k": i.k_from}
                               for i in bucket["noop"]],
                "remedies": [
                    f"add a matching {'demotion' if len(ups) > len(downs) else 'promotion'} "
                    f"in layer {layer} (see GET /fq/layer/{layer} for the "
                    f"coldest K4 residents)",
                    f"retry with mode=auto_balance (requires "
                    f"{ADMIN_ALLOW_AUTO_BALANCE_ENV}=1)",
                    f"raise n_k4_per_layer['{layer}'] and restart — runtime "
                    f"budget growth is not supported (see "
                    f"/fq/state.budget.growth_supported)",
                ],
            })


# ------------------------------------------------------------- memory guard


@dataclass(frozen=True)
class MemoryModel:
    """Per-rank expert payload arithmetic, derived from live geometry.

    Sizes come straight from :class:`swap.ExpertStage`, which is the
    authority on what a swap actually moves:

    * trellis (3 slab rows) — ``3 * H * I * k / 8`` bytes, i.e. exactly
      ``unit_bytes_per_k`` per bit of K,
    * rotations (gate_suh, up_suh, down_svh, the 3I combined
      intermediate row) — ``6 * (H + I)`` bytes, independent of K.

    So the *delta* of a re-tier is ``unit_bytes_per_k * Σ(k_to - k_from)``
    exactly, and it is 0 by construction under ``strict_pair``. Derived
    rather than hard-coded so the response can show the operator the
    arithmetic instead of asking them to trust a constant.
    """

    hidden_size: int
    intermediate_size: int

    @property
    def unit_bytes_per_k(self) -> int:
        return 3 * int(self.hidden_size) * int(self.intermediate_size) // 8

    @property
    def rotation_bytes(self) -> int:
        return 6 * (int(self.hidden_size) + int(self.intermediate_size))

    def expert_bytes(self, k: int) -> int:
        return self.unit_bytes_per_k * int(k) + self.rotation_bytes

    def resident_bytes(self, tier_of: Any) -> int:
        tier = np.asarray(tier_of, dtype=np.int64)
        return int(self.unit_bytes_per_k * int(tier.sum())
                   + self.rotation_bytes * int(tier.size))

    def delta_bytes(self, resolved: Sequence[ResolvedItem]) -> int:
        return int(self.unit_bytes_per_k
                   * sum(int(i.k_to) - int(i.k_from) for i in resolved))


def check_memory(model: MemoryModel, tier_of: Any,
                 resolved: Sequence[ResolvedItem], *,
                 budget_bytes: int | None = None,
                 device_free_bytes: int | None = None,
                 headroom_bytes: int | None = None,
                 environ: Mapping[str, str] | None = None) -> dict:
    """Refuse a re-tier that would grow the resident expert payload.

    Under ``strict_pair`` ``delta_bytes`` is exactly 0, so this never
    fires. It is evaluated and reported anyway: an admin API that asserts
    an invariant it never checks is an admin API that will violate it the
    first time someone adds a mode.
    """
    current = model.resident_bytes(tier_of)
    delta = model.delta_bytes(resolved)
    projected = current + delta
    if budget_bytes is None:
        budget_bytes = _env_int(environ, ADMIN_MEM_BUDGET_ENV, current)
    budget_bytes = int(budget_bytes)
    if headroom_bytes is None:
        headroom_bytes = _env_int(environ, ADMIN_MEM_HEADROOM_ENV,
                                  DEFAULT_MEM_HEADROOM_BYTES)
    headroom_bytes = int(headroom_bytes)

    accounting = {
        "unit_bytes_per_k_per_rank": model.unit_bytes_per_k,
        "rotation_bytes_per_expert_per_rank": model.rotation_bytes,
        "hidden_size": int(model.hidden_size),
        "intermediate_size_per_partition": int(model.intermediate_size),
        "delta_bytes_per_rank": delta,
        "current_expert_bytes_per_rank": current,
        "projected_expert_bytes_per_rank": projected,
        "budget_bytes_per_rank": budget_bytes,
        "headroom_bytes_per_rank": budget_bytes - projected,
        "device_free_bytes_min_rank": device_free_bytes,
        "required_free_headroom_bytes": headroom_bytes,
    }
    if projected > budget_bytes:
        over = projected - budget_bytes
        raise AdminError(
            "memory_budget_exceeded",
            f"resident expert payload would grow to {projected} B/rank "
            f"({_fmt_bytes(projected)}), which exceeds the budget of "
            f"{budget_bytes} B/rank ({_fmt_bytes(budget_bytes)}) by "
            f"{over} B ({_fmt_bytes(over)}). The change asks for "
            f"{delta:+d} B/rank at {model.unit_bytes_per_k} B per expert per "
            f"bit of K. Rebalance the batch so every promotion is paired with "
            f"a demotion, or raise {ADMIN_MEM_BUDGET_ENV}.",
            status=409,
            details={**accounting, "overshoot_bytes_per_rank": over,
                     "budget_env": ADMIN_MEM_BUDGET_ENV})
    if delta > 0 and device_free_bytes is not None:
        remaining = int(device_free_bytes) - delta
        if remaining < headroom_bytes:
            raise AdminError(
                "memory_headroom_exceeded",
                f"the change needs {delta} B/rank ({_fmt_bytes(delta)}) but "
                f"the tightest rank has only {int(device_free_bytes)} B free "
                f"({_fmt_bytes(int(device_free_bytes))}); that would leave "
                f"{remaining} B ({_fmt_bytes(remaining)}) against a required "
                f"headroom of {headroom_bytes} B "
                f"({_fmt_bytes(headroom_bytes)}). "
                f"{ADMIN_MEM_HEADROOM_ENV} governs the floor.",
                status=409,
                details={**accounting, "remaining_after_bytes": remaining,
                         "headroom_env": ADMIN_MEM_HEADROOM_ENV})
    return accounting


# ------------------------------------------------------------------- pins


def _pinned_from_doc(doc: Mapping[str, Any]) -> dict:
    raw = doc.get("pinned") or {}
    out: dict[str, Any] = {}
    for key, val in raw.items():
        out[str(key)] = "all" if val == "all" else sorted(
            {int(e) for e in val})
    return out


def apply_pin_change(policy_doc: Mapping[str, Any],
                     resolved: Sequence[ResolvedItem], *, pin: str) -> dict:
    """New ``pinned`` block for the target document.

    ``hold`` (the default) pins every moved expert to its NEW tier.
    Without it, the next interval sees the operator's demoted expert as a
    high-score K3 candidate and can trade it straight back once dwell
    expires — an operator would reasonably file that as a bug. ``release``
    un-does it; ``none`` leaves ``pinned`` alone.
    """
    pinned = _pinned_from_doc(policy_doc)
    if pin == PIN_NONE:
        return pinned
    # No-op items are pinned too. "expert 5 should be K4" about an expert
    # that is already K4 is still an explicit operator statement about where
    # it belongs, and reading it as "...and keep it there" is what makes a
    # pure pin change expressible without a second endpoint.
    for item in resolved:
        key = str(item.layer)
        cur = pinned.get(key)
        if cur == "all":
            if pin == PIN_RELEASE:
                raise AdminError(
                    "pin_release_from_all",
                    f'layer {item.layer} is pinned as "all"; releasing a '
                    f"single expert from it is not representable in "
                    f"fq-policy/2. Rewrite pinned['{item.layer}'] as an "
                    f"explicit expert list first.",
                    status=409, details={"layer": item.layer})
            continue
        ids = set(cur or ())
        if pin == PIN_HOLD:
            ids.add(int(item.expert))
        else:
            ids.discard(int(item.expert))
        if ids:
            pinned[key] = sorted(ids)
        else:
            pinned.pop(key, None)
    return pinned


def check_pin_starvation(new_doc: Mapping[str, Any], *,
                         layers: Sequence[int], new_tier: Any,
                         num_experts: int,
                         touched_layers: Sequence[int] | None = None,
                         min_free_slots: int | None = None,
                         environ: Mapping[str, str] | None = None) -> dict:
    """Refuse a pin set that would wedge the next ``policy.decide``.

    ``policy.decide`` raises ``"layer {l}: pins incompatible with budget"``
    when ``pin4.sum() > n_k4[l]`` or ``E - pin3.sum() < n_k4[l]`` — and
    ``step()``'s blanket handler would swallow it, so the loop would
    silently stop deciding. Keep at least ``max_swaps_per_layer`` worth of
    room on both sides so the loop always has an interval's work available.
    """
    if min_free_slots is None:
        min_free_slots = _env_int(environ, ADMIN_MIN_FREE_SLOTS_ENV,
                                  DEFAULT_MIN_FREE_SLOTS)
    min_free_slots = int(min_free_slots)
    new_tier = np.asarray(new_tier)
    rows = {int(l): i for i, l in enumerate(layers)}
    pinned = _pinned_from_doc(new_doc)
    caps = (new_doc.get("budget") or {}).get("n_k4_per_layer") or {}
    check = (sorted(rows) if touched_layers is None
             else sorted({int(x) for x in touched_layers}))
    free_slots: dict[str, int] = {}
    for layer in check:
        row = rows[layer]
        n_k4 = int(caps.get(str(layer), int((new_tier[row] == P.K4).sum())))
        val = pinned.get(str(layer))
        if val == "all":
            ids = list(range(int(num_experts)))
        else:
            ids = [int(e) for e in (val or ())]
        pin4 = sum(1 for e in ids if int(new_tier[row, e]) == P.K4)
        pin3 = sum(1 for e in ids if int(new_tier[row, e]) == P.K3)
        k4_free = n_k4 - pin4
        k3_free = (int(num_experts) - pin3) - n_k4
        free_slots[str(layer)] = int(min(k4_free, k3_free))
        if k4_free < min_free_slots or k3_free < min_free_slots:
            raise AdminError(
                "pin_would_starve_layer",
                f"layer {layer}: after this pin change the decision loop "
                f"would have {k4_free} free K4 slot(s) and {k3_free} free K3 "
                f"slot(s), below the required {min_free_slots} "
                f"({ADMIN_MIN_FREE_SLOTS_ENV}). policy.decide would refuse "
                f"the layer outright ('pins incompatible with budget') and "
                f"the loop would stop deciding. Send pin=\"release\" for some "
                f"experts in this layer, or pin=\"none\" for this request.",
                status=409,
                details={"layer": layer, "n_k4_per_layer": n_k4,
                         "pinned_k4": pin4, "pinned_k3": pin3,
                         "free_k4_slots": k4_free, "free_k3_slots": k3_free,
                         "min_free_slots": min_free_slots,
                         "min_free_slots_env": ADMIN_MIN_FREE_SLOTS_ENV})
    return free_slots


# ------------------------------------------------------- fragment pre-flight


def preflight_fragments(resolver: Any, resolved: Sequence[ResolvedItem],
                        *, rank: int = 0) -> list[dict]:
    """Reject, never degrade — and find EVERY miss in one pass.

    A K3 payload cannot be written into a K4 slab row (the row is
    ``16 x bits`` int16 words wide), so "accept and give them K3 instead"
    is not merely dishonest, it is unrepresentable. ``stage()`` would stop
    at the first miss; this walks the whole batch so one round-trip tells
    the operator everything that is missing. ``resolve``/``resolve_best``
    enqueue the lazy encode themselves, so a refusal is also the fix
    being started.

    Returns the list of unavailable targets (empty when everything is
    supplied). Never raises for a resolver failure — a crashed resolver is
    reported as "unavailable", which is the safe reading.
    """
    if resolver is None:
        return []
    misses: list[dict] = []
    best = getattr(resolver, "resolve_best", None)
    takes_chain = False
    if callable(best):
        try:
            import inspect

            takes_chain = "chain_out" in inspect.signature(best).parameters
        except (TypeError, ValueError):
            takes_chain = False
    for item in resolved:
        if item.outcome == "noop":
            continue
        chain: list[str] = []
        fragment = None
        reason = None
        try:
            if callable(best):
                fragment = (best(item.layer, item.expert, item.k_to,
                                 chain_out=chain) if takes_chain
                            else best(item.layer, item.expert, item.k_to))
            else:
                fragment = resolver.resolve(item.layer, item.expert,
                                            item.k_to)
        except Exception as exc:  # noqa: BLE001 — a refusal, never a crash
            reason = f"{type(exc).__name__}: {exc}"
        if fragment is None and reason is None:
            reason = "no source has it (resolver returned nothing)"
        got_k = None
        if fragment is not None:
            got_k = int(getattr(fragment, "k", item.k_to))
            if got_k != item.k_to:
                reason = (f"resolver substituted K{got_k} (the requested "
                          f"encode has not landed)")
        if reason is None:
            continue
        queued, position = _encode_queue_state(resolver, item)
        misses.append({
            "layer": item.layer, "expert": item.expert, "k": item.k_to,
            "k_from": item.k_from, "substituted_k": got_k, "reason": reason,
            "encode_queued": queued, "queue_position": position,
            "chain": "; ".join(chain) or None,
        })
    return misses


def _encode_queue_state(resolver: Any, item: ResolvedItem
                        ) -> tuple[bool, int | None]:
    """Best-effort read of the lazy-encode queue position ``resolve`` created."""
    getter = getattr(resolver, "_get_encode_queue", None)
    queue = None
    if callable(getter):
        try:
            queue = getter()
        except Exception:  # noqa: BLE001 — telemetry, never a failure path
            queue = None
    if queue is None:
        return False, None
    for name in ("position", "position_of", "lookup"):
        fn = getattr(queue, name, None)
        if callable(fn):
            try:
                pos = fn(item.layer, item.expert, item.k_to)
            except Exception:  # noqa: BLE001
                continue
            if pos is not None:
                return True, int(pos)
    return True, None


def raise_if_unavailable(misses: Sequence[Mapping[str, Any]],
                         *, requested: int) -> None:
    if not misses:
        return
    raise AdminError(
        "fragment_unavailable",
        f"{len(misses)} of {requested} requested tier(s) are not available; "
        f"nothing was applied. The missing encodes have been queued — drain "
        f"the lazy-encode queue "
        f"(python -m vllm.model_executor.layers.quantization.exl3_fungible."
        f"lazy_encode --drain) and retry.",
        status=409,
        details={"unavailable": [dict(m) for m in misses],
                 "requested": int(requested),
                 "retry_after_hint": "drain the lazy-encode queue"})


# --------------------------------------------------------- target document


def build_target_doc(policy_doc: Mapping[str, Any], *, layers: Sequence[int],
                     new_tier: Any, pinned: Mapping[str, Any],
                     provenance: Mapping[str, Any]) -> dict:
    """The ``fq-policy/2`` document the swap engine will commit.

    Same shape as ``loop._doc_for`` — every key of the running document is
    carried through untouched except the three this change owns.
    """
    doc = {k: v for k, v in policy_doc.items()
           if k not in ("bits_per_expert", "provenance", "pinned")}
    tier = np.asarray(new_tier)
    # START from the running document and overwrite only the rows this change
    # owns. Building bits_per_expert from `layers` alone DROPS every layer the
    # decision domain does not cover -- on GLM-5.2 that is MTP layer 78, which
    # the progressive loader requires in the bitrate map but the stats
    # collector cannot bind. The resulting document described 75 layers
    # against a running document of 76, and SwapPlan.from_policies refused it
    # with "policies cover different layers" -- an accurate message about a
    # document we had malformed ourselves.
    bpe = {str(k): list(v) for k, v in
           (policy_doc.get("bits_per_expert") or {}).items()}
    for row, layer in enumerate(layers):
        bpe[str(layer)] = [int(b) for b in tier[row]]
    doc["bits_per_expert"] = bpe
    doc["pinned"] = {str(k): v for k, v in pinned.items()}
    doc["provenance"] = dict(provenance)
    return doc


def build_plan(current_doc: Mapping[str, Any], new_doc: Mapping[str, Any]
               ) -> Any:
    """``SwapPlan.from_policies`` with the cardinality failure translated.

    One call does pairing, ordering and the D1 same-cardinality check, all
    already covered by ``tests/exl3_fungible/test_swap_cpu.py``. The
    endpoint must not hand-assemble ``(layer, e_out, e_in)`` triples.
    """
    try:
        return SW.SwapPlan.from_policies(dict(current_doc), dict(new_doc))
    except ValueError as exc:
        raise AdminError(
            "cardinality_unbalanced",
            f"the target membership is not a same-cardinality trade: {exc}",
            status=409, details={"swap_plan_error": str(exc)}) from exc


# --------------------------------------------------------------- loop state


def _state_rows(state: Any) -> dict[int, int]:
    return {int(l): i for i, l in enumerate(state.layers)}


def occupancy_map(state: Any) -> dict[int, dict[int, int]]:
    fn = getattr(state, "_occupancy_map", None)
    if callable(fn):
        return fn()
    tier = np.asarray(state.tier_of)
    return {int(layer): {int(t): int((tier[row] == t).sum())
                         for t in OT.TIERS if t in (P.K3, P.K4)
                         or int((tier[row] == t).sum())}
            for row, layer in enumerate(state.layers)}


def render_retier_table(before: Mapping[int, Mapping[int, int]],
                        after: Mapping[int, Mapping[int, int]],
                        swaps: Sequence[tuple[int, int, int]], *,
                        before_tier: Any, after_tier: Any,
                        layers: Sequence[int], title: str,
                        num_experts: int) -> str:
    """The operator-facing "what moved" view.

    Worth being explicit about, because the obvious implementation is
    wrong: ``occupancy_table.render(..., diff_only=True)`` counts experts
    per tier, and a fixed-cardinality 1-for-1 trade leaves every count
    **identical** — so the diff-only table for a successful retier is
    literally "no tier changes across N layers". Truthful and useless.

    So: render the touched layers in full (their shape is what the
    operator wants confirmed), then name the pairs underneath, since the
    pairs are the only place the change is visible at all.

    ``swaps`` are ``(model layer id, e_out, e_in)`` — NOT the ``(row,
    e_out, e_in)`` triples ``policy.decide``/``decision_log.explain``
    speak in. Both conventions exist in this package, so the whole body
    is inside the never-raise guard: a caller that passes rows would
    otherwise reach ``rows[0] -> KeyError`` and take the serve down from
    a logging helper, after the weights had already moved.
    """
    touched = sorted({int(l) for l, _, _ in swaps})
    try:
        if touched:
            text = OT.render(
                {l: after[l] for l in touched if l in after},
                {l: before[l] for l in touched if l in before},
                title=title, num_experts=int(num_experts), diff_only=False)
        else:
            return OT.render(after, before, title=title,
                             num_experts=int(num_experts), diff_only=True)

        rows = {int(l): i for i, l in enumerate(layers)}
        before_tier = np.asarray(before_tier)
        after_tier = np.asarray(after_tier)
        lines = [text,
                 "  moved (fixed cardinality: the per-tier COUNTS above are "
                 "invariant by",
                 "  construction, so a 1-for-1 trade cannot show up as a "
                 "delta — these are",
                 f"  the {len(swaps)} pair(s) that actually moved):"]
        for layer, e_out, e_in in swaps:
            row = rows[int(layer)]
            lines.append(
                f"    L{int(layer)}: e{int(e_out)} "
                f"K{int(before_tier[row, e_out])}->"
                f"K{int(after_tier[row, e_out])}"
                f"  <->  e{int(e_in)} "
                f"K{int(before_tier[row, e_in])}->"
                f"K{int(after_tier[row, e_in])}")
        omitted = len(layers) - len(touched)
        if omitted > 0:
            lines.append(f"  ({omitted} untouched layer(s) omitted)")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — telemetry must never kill a serve
        logger.exception("FQ admin: occupancy table failed to render")
        return ""


def adopt_policy(state: Any, new_doc: Mapping[str, Any],
                 swaps: Sequence[tuple[int, int, int]], *,
                 origin: str = "policy", record: Mapping | None = None,
                 title: str | None = None) -> dict:
    """Adopt a committed membership change into the live loop state.

    One code path for both callers (the interval loop and this admin API),
    because there is exactly one correct order:

    1. recompute ``tier_of`` from the document,
    2. **restart dwell for the experts that moved** (``_entered_step``), so
       the next interval cannot instantly trade them back,
    3. advance ``policy_doc`` / ``policy_sha`` / ``_policy_step``,
    4. **refresh ``pins`` from the new document** — this line is missing
       from today's ``loop._maybe_apply``. Benign for the loop, which never
       changes ``pinned``; a live bug the moment the admin path writes
       pins, because the stale ``pins`` array would be handed to the next
       ``policy.decide``,
    5. log the occupancy diff so the operator sees exactly what moved,
    6. bump the instruments.

    ``loop.py`` is owned by another change this round, so this lives here
    and takes the state as a parameter rather than being a method on
    :class:`FungibleQuantState`. It is deliberately duck-typed: anything
    with the loop-state attribute surface works, which is also what makes
    it CPU-testable.
    """
    layers = list(state.layers)
    new_tier = np.asarray(
        [[int(b) for b in new_doc["bits_per_expert"][str(layer)]]
         for layer in layers], dtype=np.int64)
    before_tier = np.asarray(state.tier_of)
    before_occupancy = occupancy_map(state)
    moved = before_tier != new_tier

    state.tier_of = new_tier
    entered = getattr(state, "_entered_step", None)
    if entered is not None:
        state._entered_step = np.where(
            moved, int(getattr(state, "_real_steps", 0)), entered)
    state.policy_doc = dict(new_doc)
    state.policy_sha = policy_hash(dict(new_doc))
    state._policy_step = int(getattr(state, "_step", 0))
    # (4) — the line today's _maybe_apply is missing.
    pins_fn = getattr(state, "_pins_from_doc", None)
    if callable(pins_fn):
        state.pins = pins_fn(dict(new_doc))
    # Fixed cardinality means n_k4 is invariant; recompute anyway so a
    # future mode that does change it cannot leave a stale budget behind.
    state.n_k4 = P.n_k4_of(new_tier)

    after_occupancy = occupancy_map(state)
    if title is None:
        title = f"expert composition after {origin} retier"
    diff_text = render_retier_table(
        before_occupancy, after_occupancy, swaps, before_tier=before_tier,
        after_tier=new_tier, layers=layers, title=title,
        num_experts=int(state.num_experts))

    if int(getattr(state, "rank", 0) or 0) == 0:
        for line in diff_text.splitlines():
            logger.info("%s", line)
        # Re-arm the loop's own periodic diff from the post-change shape,
        # so its next print is measured against what the operator did.
        state._last_composition = after_occupancy

    _bump_metrics(state, swaps, origin=origin)
    return {"occupancy_before": before_occupancy,
            "occupancy_after": after_occupancy,
            "occupancy_diff": diff_text,
            "moved": int(moved.sum())}


class _AdminMetrics:
    """The two instruments the admin path adds, registered once."""

    def __init__(self):
        import prometheus_client as pc

        self.forced_swaps_total = pc.Counter(
            "fq_forced_swaps_total",
            "Operator-forced fungible-quant expert tier changes.",
            ["layer", "direction"])
        self.admin_requests_total = pc.Counter(
            "fq_admin_requests_total",
            "Fungible-quant admin API requests by outcome.", ["outcome"])


_ADMIN_METRICS: _AdminMetrics | None = None
_ADMIN_METRICS_TRIED = False


def get_admin_metrics() -> _AdminMetrics | None:
    global _ADMIN_METRICS, _ADMIN_METRICS_TRIED
    if not _ADMIN_METRICS_TRIED:
        _ADMIN_METRICS_TRIED = True
        try:
            _ADMIN_METRICS = _AdminMetrics()
        except Exception:  # noqa: BLE001 — metrics are never load-bearing
            logger.warning("FQ admin metrics unavailable", exc_info=True)
            _ADMIN_METRICS = None
    return _ADMIN_METRICS


def record_admin_outcome(outcome: str) -> None:
    metrics = get_admin_metrics()
    if metrics is not None:
        try:
            metrics.admin_requests_total.labels(outcome=outcome).inc()
        except Exception:  # noqa: BLE001
            pass


def _bump_metrics(state: Any, swaps: Sequence[tuple[int, int, int]], *,
                  origin: str) -> None:
    metrics = getattr(state, "metrics", None)
    if metrics is not None and getattr(state, "is_lead", True):
        try:
            for layer, _, _ in swaps:
                # A forced admin swap IS installed by the time we get here.
                metrics.swap_proposals_total.labels(
                    layer=str(int(layer))).inc()
                metrics.swaps_applied_total.labels(
                    layer=str(int(layer))).inc()
            metrics.policy_age_steps.set(
                int(getattr(state, "_step", 0))
                - int(getattr(state, "_policy_step", 0)))
            export = getattr(state, "_export_occupancy", None)
            if callable(export):
                export()
        except Exception:  # noqa: BLE001
            logger.warning("FQ admin: loop metric update failed",
                           exc_info=True)
    if origin != "operator":
        return
    admin = get_admin_metrics()
    if admin is None or not getattr(state, "is_lead", True):
        return
    try:
        for layer, e_out, e_in in swaps:
            admin.forced_swaps_total.labels(
                layer=str(int(layer)), direction="demote").inc()
            admin.forced_swaps_total.labels(
                layer=str(int(layer)), direction="promote").inc()
    except Exception:  # noqa: BLE001
        logger.warning("FQ admin: forced-swap metric failed", exc_info=True)


# ------------------------------------------------------- decision record


def explain_forced(state: Any, swaps_rows: Sequence[tuple[int, int, int]], *,
                   tier_before: Any, items: Sequence[ResolvedItem],
                   actor: str | None, reason: str | None, request_id: str,
                   policy_sha_before: str, policy_sha_after: str) -> dict:
    """An ``fq-decision/1`` record marked operator-forced.

    Same schema as the interval path so existing consumers keep working,
    with ``origin: "operator"``, ``forced: true`` on every swap, the
    guards that were waived spelled out, and the actor/reason/request_id
    that make it attributable. The score fields are still populated from
    the live stats — seeing *what the policy thought* about an expert the
    operator overrode is the whole value of the record.
    """
    record: dict[str, Any]
    try:
        stats = state._read_stats()
        dwell = (np.asarray(state._real_steps)
                 - np.asarray(state._entered_step))
        record = DL.explain(
            stats, state.eps, np.asarray(tier_before), list(swaps_rows),
            pins=getattr(state, "pins", None), dwell=dwell,
            cfg=state._decide_cfg(), step=int(getattr(state, "_step", 0)),
            policy_sha_before=policy_sha_before,
            policy_sha_after=policy_sha_after)
    except Exception:  # noqa: BLE001 — the weights already moved; still log
        logger.warning("FQ admin: score recomputation failed — emitting a "
                       "minimal decision record", exc_info=True)
        record = {
            "schema": "fq-decision/1",
            "step": int(getattr(state, "_step", 0)),
            "policy_sha_before": policy_sha_before,
            "policy_sha_after": policy_sha_after,
            "config": {},
            "swaps": [{"layer": int(l), "expert_out": int(o),
                       "expert_in": int(i)} for l, o, i in swaps_rows],
            "blocked": {"dwell": 0, "hysteresis": 0, "layer_cap": 0},
            "per_layer": {},
            "totals": {"executed": len(swaps_rows),
                       "layers_touched": len({s[0] for s in swaps_rows})},
            "scores_unavailable": True,
        }
    record["origin"] = "operator"
    record["forced"] = True
    record["actor"] = actor
    record["reason"] = reason
    record["request_id"] = request_id
    record["apply_mode"] = getattr(getattr(state, "cfg", None), "apply_mode",
                                   None)
    record["applied"] = True
    record["layer_ids"] = [int(x) for x in state.layers]
    record["real_steps"] = int(getattr(state, "_real_steps", 0))
    record["guards_waived"] = ["dwell", "hysteresis", "jaccard",
                               "max_swaps_per_layer", "max_swaps_total"]
    record["blocked"] = {"dwell": 0, "hysteresis": 0, "layer_cap": 0}
    record["items"] = [i.as_dict() for i in items]
    for sw in record.get("swaps", ()):
        sw["forced"] = True
    record["decision_sha"] = hashlib.sha256(
        json.dumps([[int(x) for x in s] for s in swaps_rows],
                   separators=(",", ":")).encode()).hexdigest()
    return record


def persist_decision_record(state: Any, record: Mapping[str, Any], *,
                            request_id: str) -> str | None:
    """Write the record beside the interval records, lead rank only.

    The ``-admin-`` infix keeps a forced retier that lands on an interval
    boundary from colliding with that interval's own ``{step:08d}.json``.
    """
    store = getattr(state, "store", None)
    if store is None or not getattr(state, "is_lead", True):
        return None
    try:
        decisions = store.root / "decisions"
        decisions.mkdir(exist_ok=True)
        path = decisions / f"{int(getattr(state, '_step', 0)):08d}-admin-{request_id}.json"
        store._atomic_write(path, dict(record))
        return str(path)
    except Exception:  # noqa: BLE001 — audit trail must not fail the apply
        logger.exception("FQ admin: decision record write failed")
        return None


# ------------------------------------------------------------ plan + apply


@dataclass
class RetierPlan:
    """Phase-1 output: everything decided, nothing mutated."""

    request_id: str
    request: RetierRequest
    items: list[ResolvedItem]
    effective: list[ResolvedItem]
    plan: Any                                   # SwapPlan
    new_doc: dict
    policy_sha_before: str
    policy_sha_after: str
    memory: dict
    pins: dict
    free_slots: dict
    occupancy_before: dict
    membership_sha: str
    plan_sha: str
    unavailable: list[dict] = field(default_factory=list)

    def wire(self) -> dict:
        return {
            "request_id": self.request_id,
            "plan_sha": self.plan_sha,
            "membership_sha": self.membership_sha,
            "policy_sha_before": self.policy_sha_before,
            "policy_sha_after": self.policy_sha_after,
            "pairs": [{"layer": int(l), "expert_out": int(o),
                       "expert_in": int(i)} for l, o, i in self.plan],
            "layers": [int(x) for x in self.plan.layers()],
            "items": [i.as_dict() for i in self.items],
            "memory": self.memory,
            "pins": self.pins,
            "free_slots": self.free_slots,
            "occupancy": {str(k): {str(kk): vv for kk, vv in v.items()}
                          for k, v in self.occupancy_before.items()},
        }


def _membership_sha(tier_of: Any, layers: Sequence[int],
                    touched: Sequence[int]) -> str:
    tier = np.asarray(tier_of)
    rows = {int(l): i for i, l in enumerate(layers)}
    payload = {str(l): [int(b) for b in tier[rows[int(l)]]]
               for l in sorted({int(x) for x in touched})}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def plan_retier(state: Any, request: RetierRequest, *,
                engine: Any = None, resolver: Any = None,
                memory: MemoryModel | None = None,
                device_free_bytes: int | None = None,
                request_id: str | None = None,
                environ: Mapping[str, str] | None = None) -> RetierPlan:
    """Phase 1: resolve, guard, build the target document and the plan.

    Pure with respect to live state — nothing is staged, nothing is
    written, no device memory is touched. ``dry_run`` is simply "stop
    after this", which makes it the operator's "will this work?" button.
    """
    request_id = request_id or new_request_id()
    policy_sha_before = policy_hash(dict(state.policy_doc))
    if request.expect_policy_sha and \
            request.expect_policy_sha != policy_sha_before:
        raise AdminError(
            "policy_sha_mismatch",
            f"expect_policy_sha {request.expect_policy_sha[:16]}... does not "
            f"match the live policy {policy_sha_before[:16]}... — a policy "
            f"interval committed between your read and this write. GET "
            f"/fq/state and retry.",
            status=409,
            details={"expected": request.expect_policy_sha,
                     "actual": policy_sha_before})

    registered = None
    if engine is not None:
        registered = sorted(int(x) for x in engine.layers)
    resolved = resolve_items(
        request, layers=state.layers, tier_of=state.tier_of,
        num_experts=int(state.num_experts), registered_layers=registered)
    effective = [i for i in resolved if i.outcome != "noop"]

    caps = (dict(state.policy_doc).get("budget") or {}).get("n_k4_per_layer")
    check_cardinality(resolved, mode=request.mode, n_k4_per_layer=caps,
                      environ=environ)

    if memory is None:
        memory = memory_model_from_engine(engine)
    accounting: dict = {}
    if memory is not None:
        accounting = check_memory(memory, state.tier_of, effective,
                                  device_free_bytes=device_free_bytes,
                                  environ=environ)

    misses = preflight_fragments(resolver, effective,
                                 rank=int(getattr(state, "rank", 0) or 0))
    raise_if_unavailable(misses, requested=len(effective))

    rows = _state_rows(state)
    new_tier = np.array(state.tier_of, copy=True)
    for item in effective:
        new_tier[rows[item.layer], item.expert] = item.k_to

    pinned = apply_pin_change(state.policy_doc, resolved, pin=request.pin)
    touched = sorted({i.layer for i in resolved})
    new_doc = build_target_doc(
        state.policy_doc, layers=state.layers, new_tier=new_tier,
        pinned=pinned,
        provenance=_provenance(state, request, resolved, request_id,
                               policy_sha_before))
    free_slots = check_pin_starvation(
        new_doc, layers=state.layers, new_tier=new_tier,
        num_experts=int(state.num_experts), touched_layers=touched,
        environ=environ)

    plan = build_plan(state.policy_doc, new_doc)
    # The fixed-cardinality invariant, independently, before any device
    # write. store.commit would catch it too — but at step 5, after the
    # weights have already moved.
    try:
        validate_policy(new_doc, num_experts=int(state.num_experts))
    except ValueError as exc:
        raise AdminError("cardinality_unbalanced",
                         f"the target policy document is invalid: {exc}",
                         status=409,
                         details={"validate_policy_error": str(exc)}) from exc

    if engine is not None and len(plan) > int(engine.max_pairs):
        raise AdminError(
            "plan_too_large",
            f"the plan has {len(plan)} pairs but staging holds "
            f"{int(engine.max_pairs)} ({ADMIN_MAX_ITEMS_ENV} sizes it) — "
            f"split the batch",
            status=409, details={"pairs": len(plan),
                                 "max_pairs": int(engine.max_pairs)})

    policy_sha_after = policy_hash(new_doc)
    # plan_sha must be what its docstring claims -- "a pure function of the
    # request and the live membership" -- so that a mismatch means the ranks
    # genuinely disagree about the model.
    #
    # policy_sha_after is NOT that. It covers the whole document including
    # provenance, which carries a per-rank wall clock ("utc") and a per-rank
    # step counter. Four ranks that straddle a second boundary therefore
    # derive four different plan_shas for an identical change, and the
    # cross-check refuses with "ranks disagree on plan_sha — nothing applied":
    # an alarming message about membership corruption, raised by a clock tick.
    # Observed live -- rank 3 diverged once, then three identical retries
    # passed -- which is the worst shape for a safety check, because it fires
    # rarely and points somewhere else.
    #
    # Hash the MEMBERSHIP the document encodes instead. Two ranks that agree
    # on every expert's tier agree on the plan, whatever second they computed
    # it in.
    membership_after = hashlib.sha256(json.dumps(
        {str(k): [int(b) for b in v]
         for k, v in (new_doc.get("bits_per_expert") or {}).items()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan_sha = hashlib.sha256(json.dumps(
        {"pairs": [[int(l), int(o), int(i)] for l, o, i in plan],
         "policy_sha_before": policy_sha_before,
         "membership_after": membership_after},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    return RetierPlan(
        request_id=request_id, request=request, items=resolved,
        effective=effective, plan=plan, new_doc=new_doc,
        policy_sha_before=policy_sha_before,
        policy_sha_after=policy_sha_after, memory=accounting,
        pins=_pin_summary(state.policy_doc, new_doc),
        free_slots=free_slots, occupancy_before=occupancy_map(state),
        membership_sha=_membership_sha(state.tier_of, state.layers, touched),
        plan_sha=plan_sha, unavailable=misses)


def _pin_summary(old_doc: Mapping[str, Any], new_doc: Mapping[str, Any]
                 ) -> dict:
    old = _pinned_from_doc(old_doc)
    new = _pinned_from_doc(new_doc)
    added: dict[str, list[int]] = {}
    removed: dict[str, list[int]] = {}
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a == "all" or b == "all":
            continue
        aset, bset = set(a or ()), set(b or ())
        if bset - aset:
            added[key] = sorted(bset - aset)
        if aset - bset:
            removed[key] = sorted(aset - bset)
    return {"added": added, "removed": removed, "pinned": new}


def _provenance(state: Any, request: RetierRequest,
                resolved: Sequence[ResolvedItem], request_id: str,
                base_policy: str) -> dict:
    return {
        "proposed_by": "fq-admin/retier",
        "origin": "operator",
        "request_id": request_id,
        "actor": request.actor,
        "reason": request.reason,
        # Stamped from the REQUEST when the router supplied it, so all four
        # ranks commit an identical document. A per-rank time.strftime() here
        # made the committed policies differ by provenance alone, which shows
        # up later as /fq/state agreed=False after a swap that was in fact
        # applied identically everywhere.
        "utc": getattr(request, "utc", None) or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_policy": base_policy,
        "num_swaps": sum(1 for i in resolved if i.outcome == "promoted"),
        "mode": request.mode,
        "items": [i.as_dict() for i in resolved],
    }


def new_request_id() -> str:
    return "fqr-" + uuid.uuid4().hex[:12]


def apply_retier(state: Any, retier: RetierPlan, *, engine: Any,
                 stream: Any = None, quiesce: Any = None) -> dict:
    """Phase 2: stage, commit, adopt.

    ``fail_atomic=True`` is mandatory here: without it an abort before the
    visibility flip leaves the layer genuinely torn. With it the pre-swap
    rows and maps are restored inside the same quiesce window and the
    serve stays healthy — for an operator action that trade is obvious.

    ``on_unavailable="raise"`` (never ``"drop"``): availability was
    pre-flighted, so this is belt and braces; if it fires anyway nothing
    has been written, because staging mutates no live state.

    ``memo_hook=None`` is correct, not an omission: the mixed-runtime memo
    key contains ``tier_signature`` = *(bits, count)* per tier, not
    membership. A 1-for-1 trade leaves both counts unchanged, so the key
    is unchanged and the compiled runtime stays valid.

    ``policy_store`` only on the lead rank — passing a store on every rank
    would race four writers onto one ``current.json``.
    """
    tier_before = np.array(state.tier_of, copy=True)
    if len(retier.plan) == 0:
        # A pin-only change. No weights move, so there is nothing for the
        # swap engine to do and no reason to open a quiesce window — but
        # the document still has to be committed, or the pin evaporates at
        # the next interval and the operator's instruction was a no-op with
        # a 200 status code.
        return _apply_pins_only(state, retier)

    t0 = time.perf_counter()
    staged = engine.stage(retier.plan, fail_atomic=True,
                          on_unavailable="raise")
    stage_ms = (time.perf_counter() - t0) * 1e3

    if getattr(staged, "dropped", ()):  # defensive: raise mode should not
        raise AdminError(
            "fragment_unavailable",
            "staging dropped pairs despite on_unavailable=raise — nothing "
            "was applied", status=409,
            details={"dropped": [str(d) for d in staged.dropped]})

    is_lead = bool(getattr(state, "is_lead", True))
    store = getattr(state, "store", None)
    generation_before = int(getattr(engine, "generation", 0))
    try:
        report = engine.apply(
            staged=staged,
            quiesce=contextlib.nullcontext() if quiesce is None else quiesce,
            stream=stream,
            memo_hook=None,
            policy_store=store if is_lead else None,
            policy_doc=retier.new_doc,
            policy_num_experts=int(state.num_experts),
        )
    except Exception as exc:  # noqa: BLE001 — must report recovery state
        # The visibility flip is the only commit point, and apply() bumps
        # the generation counter WITH it (in a finally), so the counter is
        # the honest witness for which side of the flip this died on.
        generation = int(getattr(engine, "generation", generation_before))
        flipped = generation > generation_before
        restored = bool(getattr(staged, "restored", False))
        torn = not flipped and not restored
        if flipped:
            # The weights ARE new. The loop state must not keep claiming
            # otherwise, or the next interval decides against fiction.
            with contextlib.suppress(Exception):
                adopt_policy(state, retier.new_doc,
                             [(int(l), int(o), int(i))
                              for l, o, i in staged.plan],
                             origin="operator",
                             title=f"expert composition after PARTIAL "
                                   f"operator retier {retier.request_id}")
            guidance = ("the swap was committed (the visibility flip "
                        "succeeded) and the loop state was advanced to match; "
                        "a later commit-protocol step failed, so the policy "
                        "may not have been persisted — re-issue to roll "
                        "forward, or apply the inverse plan to roll back")
        elif restored:
            guidance = ("fail-atomic staging restored the pre-swap rows and "
                        "maps inside the quiesce window; the layer is "
                        "fully-old and the serve is healthy")
        else:
            guidance = ("the layer is torn — re-issue the same request (roll "
                        "forward) or restart (slabs are a cache, rebuilt from "
                        "artifacts)")
        if torn:
            logger.error("FQ ADMIN apply failed request=%s and the batch was "
                         "NOT restored — %s", retier.request_id, guidance)
        raise AdminError(
            "apply_failed", f"SwapEngine.apply raised: {exc}", status=500,
            details={"flipped": flipped, "restored": restored, "torn": torn,
                     "generation": generation,
                     "guidance": guidance}) from exc

    swaps_model = [(int(l), int(o), int(i)) for l, o, i in staged.plan]
    rows = _state_rows(state)
    swaps_rows = [(rows[l], o, i) for l, o, i in swaps_model]

    adopted = adopt_policy(
        state, retier.new_doc, swaps_model, origin="operator",
        title=f"expert composition after operator retier "
              f"{retier.request_id}")

    record = explain_forced(
        state, swaps_rows, tier_before=tier_before, items=retier.items,
        actor=retier.request.actor, reason=retier.request.reason,
        request_id=retier.request_id,
        policy_sha_before=retier.policy_sha_before,
        policy_sha_after=retier.policy_sha_after)
    record_path = persist_decision_record(state, record,
                                          request_id=retier.request_id)
    DL.log_decision(record)

    logger.warning(
        "FQ ADMIN applied request=%s pairs=%d gen=%d window=%.3fms "
        "delta_bytes/rank=%s policy %s -> %s pins+%d",
        retier.request_id, report.pairs, report.generation,
        report.window_seconds * 1e3,
        retier.memory.get("delta_bytes_per_rank"),
        retier.policy_sha_before[:8], retier.policy_sha_after[:8],
        sum(len(v) for v in retier.pins.get("added", {}).values()))

    return {
        "applied": True,
        "pairs": report.pairs,
        "layers": report.layers,
        "generation": int(report.generation),
        "bytes_h2d_per_rank": int(report.bytes_h2d),
        "stage_ms": stage_ms,
        "window_ms": report.window_seconds * 1e3,
        "restored": bool(getattr(staged, "restored", False)),
        "committed_policy_hash": report.committed_policy_hash,
        "policy_sha_after": state.policy_sha,
        "occupancy": {str(k): {str(kk): vv for kk, vv in v.items()}
                      for k, v in adopted["occupancy_after"].items()},
        "occupancy_diff": adopted["occupancy_diff"],
        "decision_record": record_path,
        "decision_sha": record.get("decision_sha"),
    }


# ------------------------------------------------------------ live binding


def _apply_pins_only(state: Any, retier: RetierPlan) -> dict:
    """Commit a document whose only change is ``pinned``."""
    committed = None
    if getattr(state, "store", None) is not None and \
            getattr(state, "is_lead", True):
        committed = state.store.commit(retier.new_doc,
                                       num_experts=int(state.num_experts))
    adopted = adopt_policy(
        state, retier.new_doc, [], origin="operator",
        title=f"expert composition after operator pin change "
              f"{retier.request_id}")
    record = explain_forced(
        state, [], tier_before=np.asarray(state.tier_of),
        items=retier.items, actor=retier.request.actor,
        reason=retier.request.reason, request_id=retier.request_id,
        policy_sha_before=retier.policy_sha_before,
        policy_sha_after=retier.policy_sha_after)
    record_path = persist_decision_record(state, record,
                                          request_id=retier.request_id)
    logger.warning(
        "FQ ADMIN pins applied request=%s policy %s -> %s pins+%d pins-%d",
        retier.request_id, retier.policy_sha_before[:8],
        retier.policy_sha_after[:8],
        sum(len(v) for v in retier.pins.get("added", {}).values()),
        sum(len(v) for v in retier.pins.get("removed", {}).values()))
    return {
        "applied": True, "pairs": 0, "layers": 0,
        "generation": int(getattr(state, "_policy_step", 0)),
        "bytes_h2d_per_rank": 0, "stage_ms": 0.0, "window_ms": 0.0,
        "restored": False, "committed_policy_hash": committed,
        "policy_sha_after": state.policy_sha,
        "occupancy": {str(k): {str(kk): vv for kk, vv in v.items()}
                      for k, v in adopted["occupancy_after"].items()},
        "occupancy_diff": adopted["occupancy_diff"],
        "decision_record": record_path,
        "decision_sha": record.get("decision_sha"),
        "pins_only": True,
    }


def memory_model_from_engine(engine: Any) -> MemoryModel | None:
    if engine is None:
        return None
    try:
        return MemoryModel(hidden_size=int(engine.hidden_size),
                           intermediate_size=int(engine.intermediate_size))
    except (AttributeError, TypeError, ValueError):
        return None


_LAYER_IN_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _module_layer_id(module: Any) -> int | None:
    """The MoE layer index of a module carrying ``exl3_mixed_trellis``.

    ``layer_id`` first, but it is NOT present on the module we need: exl3.py
    attaches the mixed-trellis dict to the fused-experts module, and
    fused_moe/layer.py identifies that module by ``layer_name = prefix``
    ("model.layers.10.mlp.experts") with no numeric id. Requiring ``layer_id``
    therefore skipped every mixed layer on a real serve and
    :func:`build_swap_engine` returned None, so POST /fq/retier answered
    "fq_not_active — the serve is uniform-K" on a checkpoint with 75 mixed
    layers. The router modules the stats collector binds DO carry ``layer_id``,
    which is why the collector worked and the swap engine did not.
    """
    layer_id = getattr(module, "layer_id", None)
    if layer_id is not None:
        return int(layer_id)
    name = getattr(module, "layer_name", None) or getattr(
        module, "prefix", None)
    if not name:
        return None
    m = _LAYER_IN_NAME.search(str(name))
    return int(m.group(1)) if m else None


def build_swap_engine(model_runner: Any, *, rank: int = 0,
                      max_pairs: int = DEFAULT_MAX_ITEMS,
                      environ: Mapping[str, str] | None = None) -> Any:
    """Bind a :class:`SwapEngine` over the live mixed-trellis layers.

    ``loop.py`` records that atomic mode "is not yet bound to live layers";
    this is that binding. Construction runs ``_validate_layer`` on every
    layer, which is the honest boot gate — a checkpoint whose rotations use
    the broadcast layout, or whose tier orderings are not a partition,
    fails here with a clear message instead of at the first swap.

    A uniform-K serve has no mixed layers at all, so this returns None and
    the caller answers ``404 fq_not_active``.
    """
    layers: dict[int, Any] = {}
    hidden = intermediate = None
    for module in model_runner.model.modules():
        mixed = getattr(module, "exl3_mixed_trellis", None)
        if mixed is None:
            continue
        layer_id = _module_layer_id(module)
        if layer_id is None:
            continue
        layers[int(layer_id)] = SW.MixedLayerState.from_exl3_mixed_trellis(
            mixed)
        hidden = int(getattr(module, "exl3_hidden_size", hidden or 0))
        intermediate = int(
            getattr(module, "exl3_intermediate_size_per_partition",
                    intermediate or 0))
    if not layers or not hidden or not intermediate:
        return None

    from vllm.model_executor.layers.quantization.exl3_fungible.progressive import (  # noqa: E501
        ProgressiveSpec,
    )

    spec = ProgressiveSpec.from_env(environ=dict(_env(environ)))
    source = SW.ResolverFragmentSource(spec.make_resolver())
    return SW.SwapEngine(
        layers, source, hidden_size=hidden, intermediate_size=intermediate,
        tier_bits=(P.K3, P.K4), rank=int(rank), max_pairs=int(max_pairs))


def _loop_state(worker: Any) -> Any:
    runner = getattr(worker, "model_runner", None)
    state = getattr(runner, "fq_collector", None)
    if state is None or not hasattr(state, "tier_of"):
        raise AdminError(
            "fq_not_active",
            "no fungible-quant loop state on this worker — the serve is "
            "running collector-only or VLLM_FQ_ENABLE is not set",
            status=404, details={"has_runner": runner is not None})
    return state


def _bound_engine(worker: Any, state: Any,
                  environ: Mapping[str, str] | None = None) -> Any:
    engine = getattr(state, "swap_engine", None)
    if engine is not None:
        return engine
    max_pairs = max(8, _env_int(environ, ADMIN_MAX_ITEMS_ENV,
                                DEFAULT_MAX_ITEMS))
    engine = build_swap_engine(worker.model_runner,
                               rank=int(getattr(state, "rank", 0) or 0),
                               max_pairs=max_pairs, environ=environ)
    if engine is None:
        raise AdminError(
            "fq_not_active",
            "no mixed-trellis layers are registered; the serve is uniform-K "
            "and there is nothing to re-tier",
            status=404, details={"reason": "no mixed-trellis layers"})
    state.swap_engine = engine
    return engine


def _device_free_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        return None


_WORKER_BUSY = threading.Lock()


def _ok(payload: Mapping[str, Any]) -> str:
    return json.dumps({"ok": True, **payload}, sort_keys=True,
                      separators=(",", ":"), default=str)


def _err(exc: BaseException) -> str:
    if isinstance(exc, AdminError):
        return json.dumps({"ok": False, "error": exc.wire()}, sort_keys=True,
                          separators=(",", ":"), default=str)
    logger.exception("FQ admin worker call failed")
    return json.dumps({"ok": False, "error": {
        "code": "internal_error", "http_status": 500,
        "message": f"{type(exc).__name__}: {exc}", "details": {}}},
        sort_keys=True, separators=(",", ":"), default=str)


def worker_describe(worker: Any, request_json: str = "{}") -> str:
    """Read-only view of this rank's FQ state (``GET /fq/state``)."""
    try:
        _require_gates()
        query = json.loads(request_json or "{}")
        state = _loop_state(worker)
        engine = None
        with contextlib.suppress(AdminError):
            engine = _bound_engine(worker, state)
        memory = memory_model_from_engine(engine)
        tier = np.asarray(state.tier_of)
        caps = (dict(state.policy_doc).get("budget") or {}).get(
            "n_k4_per_layer") or {}
        payload: dict[str, Any] = {
            "rank": int(getattr(state, "rank", 0) or 0),
            "policy_sha": policy_hash(dict(state.policy_doc)),
            "layers": [int(x) for x in state.layers],
            "num_experts": int(state.num_experts),
            "step": int(getattr(state, "_step", 0)),
            "policy_step": int(getattr(state, "_policy_step", 0)),
            "apply_mode": getattr(getattr(state, "cfg", None), "apply_mode",
                                  None),
            "budget": {"mode": "fixed_cardinality",
                       "n_k4_per_layer": {str(k): int(v)
                                          for k, v in caps.items()},
                       "growth_supported": False},
            "occupancy": {str(k): {str(kk): vv for kk, vv in v.items()}
                          for k, v in occupancy_map(state).items()},
            "pinned": _pinned_from_doc(state.policy_doc),
            "servable_tiers": list(SERVABLE_TIERS),
            "engine_bound": engine is not None,
            "registered_layers": (sorted(int(x) for x in engine.layers)
                                  if engine is not None else []),
            "max_pairs": int(getattr(engine, "max_pairs", 0)) if engine
            else 0,
            "generation": int(getattr(engine, "generation", 0)) if engine
            else 0,
        }
        if memory is not None:
            payload["memory"] = {
                "unit_bytes_per_k_per_rank": memory.unit_bytes_per_k,
                "rotation_bytes_per_expert_per_rank": memory.rotation_bytes,
                "current_expert_bytes_per_rank": memory.resident_bytes(tier),
                "device_free_bytes": _device_free_bytes(),
            }
        layer = query.get("layer")
        if layer is not None:
            payload["layer"] = _layer_view(state, int(layer))
        return _ok(payload)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def _layer_view(state: Any, layer: int) -> dict:
    rows = _state_rows(state)
    if int(layer) not in rows:
        raise AdminError("layer_not_registered",
                         f"layer {layer} is not in the running policy",
                         status=404, details={"layer": int(layer)})
    row = rows[int(layer)]
    tier = np.asarray(state.tier_of)[row]
    pins = np.asarray(getattr(state, "pins", np.zeros_like(tier)))[row]
    dwell = (int(getattr(state, "_real_steps", 0))
             - np.asarray(getattr(state, "_entered_step",
                                  np.zeros_like(tier)))[row])
    scores = None
    try:
        s = P.score(state._read_stats(), state.eps)
        scores = [float(x) for x in np.asarray(s)[row]]
    except Exception:  # noqa: BLE001 — the table is still useful without it
        logger.debug("FQ admin: live scores unavailable", exc_info=True)
    experts = []
    for e in range(int(state.num_experts)):
        experts.append({
            "expert": e, "k": int(tier[e]), "pin": int(pins[e]),
            "dwell_steps": int(dwell[e]),
            "score": None if scores is None else scores[e],
        })
    if scores is not None:
        order = sorted(range(len(scores)), key=lambda e: (-scores[e], e))
        rank_of = {e: i for i, e in enumerate(order)}
        for row_dict in experts:
            row_dict["score_rank"] = rank_of[row_dict["expert"]]
    return {"layer": int(layer), "experts": experts,
            "n_k4": int((tier == P.K4).sum())}


def worker_plan(worker: Any, request_json: str) -> str:
    """Phase 1 on this rank: resolve + guard, mutate nothing."""
    try:
        _require_gates()
        payload = json.loads(request_json)
        state = _loop_state(worker)
        engine = _bound_engine(worker, state)
        request = RetierRequest.from_canonical(payload["request"])
        retier = plan_retier(state, request, engine=engine,
                             resolver=_resolver_of(engine),
                             device_free_bytes=_device_free_bytes(),
                             request_id=payload.get("request_id"))
        return _ok({"phase": "plan", **retier.wire()})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def worker_apply(worker: Any, plan_json: str) -> str:
    """Phase 2 on this rank: re-derive the plan, then commit it.

    Re-deriving rather than trusting the wire is deliberate: under
    ``strict_pair`` the plan is a pure function of the request and the
    live membership, so every rank lands on the same pairs — and the
    ``plan_sha`` cross-check turns any disagreement into a refusal instead
    of divergent weights.
    """
    try:
        _require_gates()
    except AdminError as exc:
        return _err(exc)
    if not _WORKER_BUSY.acquire(blocking=False):
        return _err(AdminError(
            "retier_in_flight",
            "another re-tier is already staging on this worker; SwapEngine "
            "holds one staged batch at a time", status=409))
    try:
        payload = json.loads(plan_json)
        state = _loop_state(worker)
        engine = _bound_engine(worker, state)
        request = RetierRequest.from_canonical(payload["request"])
        retier = plan_retier(state, request, engine=engine,
                             resolver=_resolver_of(engine),
                             device_free_bytes=_device_free_bytes(),
                             request_id=payload.get("request_id"))
        expected = payload.get("plan_sha")
        if expected and expected != retier.plan_sha:
            raise AdminError(
                "rank_divergence",
                f"rank {getattr(state, 'rank', '?')} derived plan_sha "
                f"{retier.plan_sha[:16]}... but the router expects "
                f"{str(expected)[:16]}... — nothing applied",
                status=500,
                details={"expected": expected, "actual": retier.plan_sha,
                         "rank": int(getattr(state, "rank", 0) or 0)})
        result = apply_retier(state, retier, engine=engine)
        result["rank"] = int(getattr(state, "rank", 0) or 0)
        result["plan_sha"] = retier.plan_sha
        return _ok({"phase": "apply", **result})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)
    finally:
        _WORKER_BUSY.release()


def _resolver_of(engine: Any) -> Any:
    source = getattr(engine, "source", None)
    return getattr(source, "resolver", None)


def _require_gates(environ: Mapping[str, str] | None = None) -> None:
    """The second gate, enforced at the WORKER, not only at the router.

    The router is not the only way in. ``register_vllm_dev_api_routers``
    also attaches vLLM's own ``POST /collective_rpc``
    (``vllm/entrypoints/serve/dev/rpc/api_router.py``), which forwards an
    arbitrary ``method`` name straight to every worker — and
    ``Worker.fq_admin_apply`` is an unconditional method on ``Worker``. So
    ``VLLM_SERVER_DEV_MODE=1`` alone was enough to mutate live weights
    through ``{"method": "fq_admin_apply", "args": ["<payload>"]}``,
    bypassing ``VLLM_FQ_ADMIN_API`` *and* ``VLLM_FQ_ADMIN_TOKEN``
    entirely. Both gates are re-checked here so "OFF by default, twice
    over" is true of every entry point rather than only of the one this
    module owns.

    Worker processes inherit the server's environment (the multiproc
    executor forks/spawns from it; the Ray path copies all of
    ``os.environ`` via ``ray_env_utils.get_env_vars_to_copy``), so this
    never refuses a request the router would have allowed.
    """
    if not admin_enabled(environ):
        raise AdminError("fq_admin_disabled", gate_reason(environ),
                         status=404,
                         details={"dev_mode": dev_mode_enabled(environ),
                                  "admin_flag_env": ADMIN_API_ENV})


class FqWorkerAdmin:
    """``--worker-extension-cls`` mixin, for trees without the core hooks.

    ``vllm/v1/worker/worker_base.py`` injects the resolved class into
    ``Worker.__bases__`` after asserting no attribute collides — so use
    EITHER this mixin OR the thin ``fq_admin_*`` methods on ``Worker``,
    never both. vLLM supports exactly one extension class, so the shipping
    binding is the core methods; this exists so an RLHF-free deployment can
    get the admin API with zero core-file changes.
    """

    def fq_admin_describe(self, request_json: str = "{}") -> str:
        return worker_describe(self, request_json)

    def fq_admin_plan(self, request_json: str) -> str:
        return worker_plan(self, request_json)

    def fq_admin_apply(self, plan_json: str) -> str:
        return worker_apply(self, plan_json)


# ------------------------------------------------------------------ router


def _rank_results(raw: Sequence[Any]) -> list[dict]:
    out = []
    for entry in raw:
        if isinstance(entry, (bytes, bytearray)):
            entry = entry.decode()
        if isinstance(entry, str):
            entry = json.loads(entry)
        out.append(dict(entry))
    return out


def _first_error(results: Sequence[Mapping[str, Any]]) -> AdminError | None:
    """The first rank's refusal — with the OTHER ranks' verdicts attached.

    A collective call is not atomic across ranks. Returning only the
    first failure threw away the fact that the others succeeded, so a TP4
    apply where rank 2 dies reported that one rank's ``torn: true,
    flipped: false`` — "nothing was applied" — while ranks 0/1/3 had
    committed the trade and advanced their loop state. ``per_rank`` and
    ``ranks_ok`` make the split visible; the caller decides how loud to
    be about it.
    """
    err = None
    for res in results:
        if not res.get("ok", False):
            err = AdminError.from_wire(res.get("error") or {})
            break
    if err is None:
        return None
    err.details["per_rank"] = [
        {"index": i, "ok": bool(res.get("ok", False)),
         "code": (res.get("error") or {}).get("code") if not res.get("ok")
         else None,
         "generation": res.get("generation"),
         "policy_sha_after": res.get("policy_sha_after")}
        for i, res in enumerate(results)]
    err.details["ranks_ok"] = sum(1 for r in results if r.get("ok", False))
    err.details["ranks_total"] = len(results)
    return err


def _agree(results: Sequence[Mapping[str, Any]], key: str) -> bool:
    values = [res.get(key) for res in results]
    return len(set(map(str, values))) <= 1


def attach_router(app: Any, *, environ: Mapping[str, str] | None = None
                  ) -> bool:
    """Attach ``/fq/*`` to a FastAPI app, if and only if both gates are on.

    Returns True when the routes were attached. Called from
    ``register_vllm_dev_api_routers``, which already runs only under
    ``VLLM_SERVER_DEV_MODE``; the second gate is re-checked here so the
    routes cannot appear on a production serve by accident, and again per
    request so nothing that attaches the router by another path can slip
    a live weight mutation through.
    """
    if not admin_enabled(environ):
        logger.debug("FQ admin API not attached: %s", gate_reason(environ))
        return False
    app.include_router(build_router(environ=environ))
    logger.warning(
        "SECURITY WARNING: FQ admin API is ENABLED (%s=1). POST /fq/retier "
        "mutates live model weights. Set %s to require a header token.",
        ADMIN_API_ENV, ADMIN_TOKEN_ENV)
    return True


def build_router(*, environ: Mapping[str, str] | None = None) -> Any:
    """The ``/fq`` router. FastAPI is imported here, never at module scope."""
    import asyncio

    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    # ``from __future__ import annotations`` turns every annotation in this
    # module into a string, and FastAPI resolves handler annotations with
    # typing.get_type_hints against the *module* globals. fastapi is
    # imported lazily here (the laziness contract: nothing web at import
    # time), so ``Request`` is a local — publish it into the module
    # namespace or FastAPI reads ``raw_request: "Request"`` as an
    # unresolvable name, decides it must be a query parameter, and answers
    # 422 on every route.
    globals().setdefault("Request", Request)
    globals().setdefault("JSONResponse", JSONResponse)

    router = APIRouter(prefix="/fq", tags=["fungible-quant admin"])
    lock = asyncio.Lock()

    def engine_client(request: Request):
        return request.app.state.engine_client

    def _fail(exc: AdminError) -> JSONResponse:
        record_admin_outcome(exc.code)
        return JSONResponse(status_code=exc.status, content=exc.body())

    def _guard(request: Request) -> AdminError | None:
        if not admin_enabled(environ):
            return AdminError("fq_admin_disabled", gate_reason(environ),
                              status=404)
        token = str(_env(environ).get(ADMIN_TOKEN_ENV, "")).strip()
        if token:
            import hmac

            sent = request.headers.get("X-FQ-Admin-Token", "")
            # compare_digest refuses str operands with non-ASCII code
            # points (TypeError). Starlette decodes headers as latin-1, so
            # a single 0x80..0xff byte in the header turned an
            # unauthenticated 403 into an unhandled 500 with a traceback.
            # Compare bytes: same constant-time property, total function.
            if not hmac.compare_digest(sent.encode("utf-8", "surrogateescape"),
                                       token.encode("utf-8",
                                                    "surrogateescape")):
                return AdminError(
                    "fq_admin_forbidden",
                    f"X-FQ-Admin-Token missing or wrong ({ADMIN_TOKEN_ENV} is "
                    f"set on this serve)", status=403)
        return None

    async def _collective(request: Request, method: str, payload: dict,
                          timeout: float | None = None) -> list[dict]:
        engine = engine_client(request)
        raw = await engine.collective_rpc(
            method, timeout=timeout,
            args=(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             default=str),))
        if raw is None:
            raise AdminError("rank_divergence",
                             f"{method} returned nothing", status=500)
        results = _rank_results(raw)
        err = _first_error(results)
        if err is not None:
            raise err
        return results

    @router.get("/state")
    async def fq_state(raw_request: Request):
        blocked = _guard(raw_request)
        if blocked is not None:
            return _fail(blocked)
        try:
            results = await _collective(raw_request, "fq_admin_describe", {})
        except AdminError as exc:
            return _fail(exc)
        record_admin_outcome("state")
        return JSONResponse(content={
            "ranks": len(results),
            "agreed": _agree(results, "policy_sha"),
            "state": results[0],
            "per_rank": results,
            "admin": {"enabled": True,
                      "max_items": _env_int(environ, ADMIN_MAX_ITEMS_ENV,
                                            DEFAULT_MAX_ITEMS),
                      "token_required": bool(
                          str(_env(environ).get(ADMIN_TOKEN_ENV, "")).strip()),
                      "auto_balance": False,
                      "growth_supported": False}})

    @router.get("/layer/{layer}")
    async def fq_layer(layer: int, raw_request: Request):
        blocked = _guard(raw_request)
        if blocked is not None:
            return _fail(blocked)
        try:
            results = await _collective(raw_request, "fq_admin_describe",
                                        {"layer": int(layer)})
        except AdminError as exc:
            return _fail(exc)
        record_admin_outcome("layer")
        return JSONResponse(content=results[0].get("layer", {}))

    @router.post("/retier")
    async def fq_retier(raw_request: Request):
        blocked = _guard(raw_request)
        if blocked is not None:
            return _fail(blocked)
        t0 = time.perf_counter()
        try:
            body = None
            if (await raw_request.body()).strip():
                try:
                    body = await raw_request.json()
                except Exception:  # noqa: BLE001
                    raise AdminError("bad_json",
                                     "body is not valid JSON") from None
            request = parse_request(
                body, query=dict(raw_request.query_params),
                actor_header=raw_request.headers.get("X-FQ-Actor"),
                environ=environ)
        except AdminError as exc:
            return _fail(exc)

        request_id = new_request_id()
        logger.warning(
            "FQ ADMIN retier request=%s actor=%s mode=%s items=%d layers=%s "
            "dry_run=%s reason=%r", request_id, request.actor, request.mode,
            len(request.items), sorted({i.layer for i in request.items}),
            request.dry_run, request.reason)

        if lock.locked():
            return _fail(AdminError(
                "retier_in_flight",
                "another re-tier is in flight; SwapEngine holds one staged "
                "batch at a time", status=409))
        async with lock:
            payload = {"request": request.canonical(),
                       "request_id": request_id}
            try:
                plans = await _collective(raw_request, "fq_admin_plan",
                                          payload, timeout=request.timeout_s)
            except AdminError as exc:
                return _fail(exc)
            for key in ("plan_sha", "membership_sha", "policy_sha_before"):
                if not _agree(plans, key):
                    return _fail(AdminError(
                        "rank_divergence",
                        f"ranks disagree on {key} — nothing applied",
                        status=500,
                        details={key: [p.get(key) for p in plans]}))
            head = plans[0]
            base = {
                "request_id": request_id,
                "dry_run": request.dry_run,
                "mode": request.mode,
                "plan": {"pairs": head.get("pairs", []),
                         "layers": head.get("layers", []),
                         "plan_sha": head.get("plan_sha")},
                "items": head.get("items", []),
                "memory": head.get("memory", {}),
                "pins": head.get("pins", {}),
                "free_slots": head.get("free_slots", {}),
                "policy": {"sha_before": head.get("policy_sha_before"),
                           "sha_after": head.get("policy_sha_after")},
                "ranks": {"count": len(plans), "agreed": True},
                "warnings": [],
            }
            if request.dry_run:
                record_admin_outcome("dry_run")
                base["applied"] = False
                base["timing"] = {"total_ms": (time.perf_counter() - t0) * 1e3}
                return JSONResponse(content=base)
            pairs = head.get("pairs") or []
            pin_delta = head.get("pins") or {}
            pin_change = bool(pin_delta.get("added")
                              or pin_delta.get("removed"))
            if not pairs and not pin_change:
                record_admin_outcome("noop")
                base["applied"] = False
                base["warnings"].append(
                    "every item was a no-op and no pin changed; nothing to "
                    "do")
                base["timing"] = {"total_ms": (time.perf_counter() - t0) * 1e3}
                return JSONResponse(content=base)

            engine = engine_client(raw_request)
            # A pin-only change moves no weights, so it needs no drain —
            # opening a quiesce window for it would stall the serve for a
            # metadata edit.
            needs_drain = bool(pairs)
            drain_t = time.perf_counter()
            paused = False
            if needs_drain:
                try:
                    await engine.pause_generation(mode=request.drain_mode,
                                                  clear_cache=False)
                    paused = True
                except TimeoutError:
                    return _fail(AdminError(
                        "drain_timeout",
                        f"generation did not drain within "
                        f"{request.timeout_s}s", status=408))
                except AdminError as exc:
                    return _fail(exc)
            drain_ms = (time.perf_counter() - drain_t) * 1e3
            try:
                apply_payload = {**payload, "plan_sha": head.get("plan_sha")}
                results = await _collective(raw_request, "fq_admin_apply",
                                            apply_payload,
                                            timeout=request.timeout_s)
            except AdminError as exc:
                # A collective apply is not atomic ACROSS ranks. If some
                # rank committed the trade and another did not, the ranks
                # are now serving different weights, and reporting the
                # loser's "nothing was applied" would be a lie that reads
                # like reassurance.
                applied_ranks = int(exc.details.get("ranks_ok") or 0)
                if applied_ranks:
                    total = exc.details.get("ranks_total", "?")
                    logger.error(
                        "FQ ADMIN request=%s PARTIALLY applied: %s of %s "
                        "rank(s) committed, the rest refused (%s) — the "
                        "ranks' weights now differ",
                        request_id, applied_ranks, total, exc.code)
                    exc = AdminError(
                        "partial_apply",
                        f"{applied_ranks} of {total} rank(s) committed this "
                        f"re-tier and {exc.code} on the rest — the ranks' "
                        f"weights may now differ; GET /fq/state to compare "
                        f"policy_sha per rank and consider a restart. "
                        f"First failure: {exc.message}",
                        status=500, details=dict(exc.details,
                                                 first_error=exc.code))
                return _fail(exc)
            finally:
                if paused:
                    with contextlib.suppress(Exception):
                        await engine.resume_generation()

            if request.reset_prefix_cache:
                with contextlib.suppress(Exception):
                    await engine.reset_prefix_cache(False, False)

            for key in ("generation", "policy_sha_after"):
                if not _agree(results, key):
                    logger.error(
                        "FQ ADMIN rank divergence on %s after apply: %s",
                        key, [r.get(key) for r in results])
                    record_admin_outcome("rank_divergence")
                    return JSONResponse(status_code=500, content=AdminError(
                        "rank_divergence",
                        f"ranks disagree on {key} AFTER the apply — the "
                        f"ranks' weights may now differ; consider a restart",
                        status=500,
                        details={key: [r.get(key) for r in results]}).body())

            lead = results[0]
            record_admin_outcome("applied")
            base.update({
                "applied": True,
                "occupancy": lead.get("occupancy", {}),
                "occupancy_diff": lead.get("occupancy_diff", ""),
                "decision_record": lead.get("decision_record"),
                "reset_prefix_cache": request.reset_prefix_cache,
                "timing": {"drain_ms": drain_ms,
                           "stage_ms": lead.get("stage_ms"),
                           "window_ms": lead.get("window_ms"),
                           "total_ms": (time.perf_counter() - t0) * 1e3},
            })
            base["plan"]["generation"] = lead.get("generation")
            base["policy"]["sha_after"] = lead.get("policy_sha_after")
            base["policy"]["committed"] = bool(
                lead.get("committed_policy_hash"))
            base["ranks"].update({
                "generation": [r.get("generation") for r in results],
                "policy_sha_after": [r.get("policy_sha_after")
                                     for r in results],
                "bytes_h2d_per_rank": [r.get("bytes_h2d_per_rank")
                                       for r in results],
                "restored": [r.get("restored") for r in results],
            })
            return JSONResponse(content=base)

    return router
