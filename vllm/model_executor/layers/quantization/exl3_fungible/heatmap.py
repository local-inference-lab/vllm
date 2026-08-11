# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant activation-matrix endpoint (``GET /fq/heatmap``).

The operator-facing question this answers: *which experts is the router
actually sending traffic to, right now, per layer, with the current K
tier overlaid.* Implements ``runs/m5-serve/heatmap/ENDPOINT-SPEC.md``.

Three facts about the data drive every decision in this file, and all
three are properties of :class:`~.stats.FqStatsCollector`, not of the
transport:

1. **It is 19,200 float64 cells.** 75 instrumented MoE layers x 256
   experts on GLM-5.2. Serialised the way ``loop._dump_stats`` does it
   that is ~800 KB per sample (measured: 807,073 B/record mean over the
   archived dumps). Base64 bf16 + ``Content-Encoding: gzip`` puts the
   default sample at **33,163 B** on the wire, a 24x reduction, at a
   measured 0.38898 % worst-case relative error — far below anything a
   colour ramp resolves.
2. **It is piecewise constant between window rolls.** ``decayed()``
   reads only ``_count_win`` / ``_win_pos`` / ``_windows_rolled``
   (stats.py:320-341), and all three are written only at a roll
   (stats.py:287-298). At the observed ~2.1 engine steps/s and
   ``window_stride=32`` that is one change every ~15 s, so a weak
   ``ETag`` turns six polls out of seven into a ~200 B ``304``.
3. **Reading it stalls the engine.** The utility RPC is dispatched
   between engine steps (core.py:1387-1394) and ``collective_rpc``
   blocks (multiproc_executor.py:364,421). Hence the single-flight lock
   + TTL cache here, and hence the worker does the cheapest possible
   work: it casts and base64s, and never compresses.

**Cross-rank merge rule: rank 0 is canonical. Counts are NOT summed.**
See :data:`MERGE_RULE` for the derivation from ``stats.py`` — getting
this wrong inflates every number by the TP degree (4x on this serve).

Ownership: this module reads the collector through its public surface
(``decayed``, ``mass_is_real``, ``count_buf``, ``num_experts``,
``window_len``/``window_stride``/``decay``) plus the documented window
bookkeeping (``_windows_rolled``, ``_win_pos``, ``_step``,
``_count_win``). It reuses the admin API's rank helpers by import. It
edits nothing in ``stats.py``, ``loop.py``, ``policy.py``,
``admin.py`` or ``occupancy_table.py``.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vllm.model_executor.layers.quantization.exl3_fungible.admin import (
    DEV_MODE_ENV,
    AdminError,
    _agree,
    _env,
    _env_float,
    _env_int,
    _first_error,
    _rank_results,
    dev_mode_enabled,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ env knobs

#: Second gate, specific to this surface. Deliberately NOT
#: ``VLLM_FQ_ADMIN_API``: the routing matrix is a fingerprint of what the
#: model is being asked to do, but it is read-only, so a dashboard should
#: be able to see it without holding the credential that mutates live
#: weights.
HEATMAP_API_ENV = "VLLM_FQ_HEATMAP"
HEATMAP_API_ALIAS_ENV = "VLLM_FQ_HEATMAP_API"
HEATMAP_TOKEN_ENV = "VLLM_FQ_HEATMAP_TOKEN"
HEATMAP_MIN_PERIOD_MS_ENV = "VLLM_FQ_HEATMAP_MIN_PERIOD_MS"
HEATMAP_TIMEOUT_S_ENV = "VLLM_FQ_HEATMAP_TIMEOUT_S"
HEATMAP_ALLOW_COLLECTOR_ZERO_ENV = "VLLM_FQ_HEATMAP_ALLOW_COLLECTOR_ZERO"

TOKEN_HEADER = "X-FQ-Heatmap-Token"

DEFAULT_MIN_PERIOD_MS = 500
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_AGE_MS = 1000
GZIP_LEVEL = 1  # measured 2.20 ms vs 5.23 ms at level 6, for 4 % more bytes

SCHEMA = "fq-heatmap/1"
SCHEMA_VERSION = 1
WORKER_METHOD = "fq_heatmap_sample"

INCLUDE_TOKENS = frozenset({"live", "cum", "mass"})
PRECISIONS = frozenset({"bf16", "f32"})
REDUCERS = frozenset({"rank0", "all"})
RESET_SCOPES = frozenset({"client", "heatmap", "collector", "poison"})

#: Why rank 0 and not a sum, a max, or a mean.
#:
#: ``integration.maybe_init_fq_collector`` sizes the collector to
#: ``routers[0][1].global_num_experts`` (integration.py:157) — the GLOBAL
#: logical expert count, identical on every rank, never a per-rank shard
#: — and binds it to ``module.router``, a ``BaseRouter``
#: (integration.py:149-162). The capture fn histograms
#: ``topk_ids.flatten()`` over ``[0, num_experts)``
#: (stats.py:211-231,236-268), i.e. the GATE's output.
#:
#: Under tensor parallelism the gate is REPLICATED: every rank runs the
#: same gate over the same hidden states and produces the same
#: ``topk_ids``. So each rank's ``count_buf`` is an independent
#: observation of the SAME routing events, not a partition of them. The
#: ranks are replicas, and the correct merge of replicas is "take one and
#: check the others agree" — summing would report 4x the real traffic on
#: a TP4 serve, and a max would silently paper over divergence.
#:
#: Expert parallelism does not change this: EP shards which expert each
#: rank *computes*, not which expert the (replicated) gate *chooses*.
#:
#: Data parallelism DOES change it — each replica sees different requests,
#: so a sum would be correct there — but ``DPLBAsyncMPClient
#: .call_utility_async`` gathers from every engine and returns only
#: ``[0]`` (core_client.py:1449-1458), so the endpoint cannot see the
#: other replicas to sum them and would silently report one replica's
#: traffic as the whole serve's. Hence ``501 dp_not_supported``.
MERGE_RULE = "rank0-canonical"


def heatmap_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Both gates. OFF unless dev mode AND the heatmap flag are set."""
    env = _env(environ)
    flag = (str(env.get(HEATMAP_API_ENV, "")).strip()
            or str(env.get(HEATMAP_API_ALIAS_ENV, "")).strip())
    return dev_mode_enabled(environ) and flag in ("1", "true", "True")


def collector_zero_allowed(
        environ: Mapping[str, str] | None = None) -> bool:
    """The third gate: the destructive reset scope (see §7.5)."""
    return str(_env(environ).get(HEATMAP_ALLOW_COLLECTOR_ZERO_ENV,
                                 "0")).strip() in ("1", "true", "True")


def gate_reason(environ: Mapping[str, str] | None = None) -> str:
    if not dev_mode_enabled(environ):
        return (f"{DEV_MODE_ENV} is not set — vLLM development endpoints "
                "are disabled")
    return (f"{HEATMAP_API_ENV}=1 is required to expose the routing "
            "activation matrix (dev mode alone is not enough)")


def _require_gates(environ: Mapping[str, str] | None = None) -> None:
    """The gates, enforced at the WORKER as well as at the router.

    ``register_vllm_dev_api_routers`` also attaches vLLM's own
    ``POST /collective_rpc``, which forwards an arbitrary ``method`` name
    to every worker, and ``Worker.fq_heatmap_sample`` is an
    unconditional method. Without this check ``VLLM_SERVER_DEV_MODE=1``
    alone would be enough to read the routing matrix — and, worse, to
    reach ``op=zero_collector`` — bypassing ``VLLM_FQ_HEATMAP`` and the
    token entirely. Worker processes inherit the server environment, so
    the same env is visible here.
    """
    if not heatmap_enabled(environ):
        raise AdminError("fq_heatmap_disabled", gate_reason(environ),
                         status=404)


# ------------------------------------------------------------------ encoding


def f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bfloat16 bit patterns, round-to-nearest-even.

    bf16 IS the top half of a float32, so this is a shift plus the RNE
    correction; the client decodes it by shifting back (four lines of
    JavaScript, no library). RNE rather than truncation because it halves
    the error for free: measured over all 19,200 cells of ``stats.jsonl``
    record 9, RNE's worst-case relative error is **0.38898 %** against
    truncation's 0.77261 %.
    """
    a = np.ascontiguousarray(arr, dtype=np.float32)
    u = a.view(np.uint32)
    # NaN must stay NaN: the round-half-to-even carry can push a NaN
    # pattern into the infinity encoding.
    nan = np.isnan(a)
    rounded = ((u.astype(np.uint64) + 0x7FFF
                + ((u.astype(np.uint64) >> 16) & 1)) >> 16)
    bits = rounded.astype(np.uint16)
    if nan.any():
        bits = np.where(nan, np.uint16(0x7FC0), bits).astype(np.uint16)
    return bits


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 bfloat16 bit patterns -> float32 (exact, no rounding)."""
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def encode_floats(arr: np.ndarray, precision: str) -> str:
    """Base64 of little-endian raw bytes at the requested wire dtype."""
    if precision == "bf16":
        payload = f32_to_bf16_bits(np.asarray(arr, dtype=np.float32))
    elif precision == "f32":
        payload = np.ascontiguousarray(arr, dtype=np.float32)
    else:
        raise ValueError(f"unknown precision {precision!r}")
    return _b64(_le(payload))


def decode_floats(blob: str, precision: str,
                  cells: int | None = None) -> np.ndarray:
    """Inverse of :func:`encode_floats`, as float32. Client-side check."""
    raw = _unb64(blob)
    if precision == "bf16":
        out = bf16_bits_to_f32(np.frombuffer(raw, dtype="<u2"))
    elif precision == "f32":
        out = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unknown precision {precision!r}")
    _check_cells(out, cells)
    return out


def encode_tier(arr: np.ndarray) -> str:
    """Base64 of one u8 per cell. 19,200 bytes of ``3`` gzip to 168 B.

    Validates the domain: a value outside ``occupancy_table.TIERS``
    means the tier table has been corrupted, and truncating it into a
    u8 would ship a plausible-looking lie.
    """
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        occupancy_table as OT,
    )

    a = np.asarray(arr)
    bad = np.setdiff1d(np.unique(a), np.asarray(OT.TIERS))
    if bad.size:
        raise ValueError(
            f"tier values {bad.tolist()} are outside "
            f"occupancy_table.TIERS={list(OT.TIERS)}")
    return _b64(_le(a.astype(np.uint8)))


def decode_tier(blob: str, cells: int | None = None) -> np.ndarray:
    out = np.frombuffer(_unb64(blob), dtype=np.uint8)
    _check_cells(out, cells)
    return out


def encode_u32(arr: np.ndarray) -> str:
    return _b64(_le(np.asarray(arr).astype(np.uint32)))


def decode_u32(blob: str, cells: int | None = None) -> np.ndarray:
    out = np.frombuffer(_unb64(blob), dtype="<u4")
    _check_cells(out, cells)
    return out


def _le(arr: np.ndarray) -> np.ndarray:
    """Force little-endian byte order — the wire contract says so.

    ``dtype.byteorder`` is ``'='`` for native dtypes, so testing for
    ``'>'`` would silently ship big-endian bytes on a big-endian host.
    Cast to an explicitly little-endian dtype instead; on x86 that is a
    no-op.
    """
    a = np.ascontiguousarray(arr)
    return a.astype(a.dtype.newbyteorder("<"), copy=False)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _unb64(blob: str) -> bytes:
    try:
        return base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not valid base64: {exc}") from None


def _check_cells(arr: np.ndarray, cells: int | None) -> None:
    if cells is not None and arr.size != cells:
        raise ValueError(
            f"array decodes to {arr.size} elements, expected {cells} — "
            "a truncated blob must never be rendered as a shifted heatmap")


def gzip_bytes(payload: bytes, level: int = GZIP_LEVEL) -> bytes:
    import zlib

    co = zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return co.compress(payload) + co.flush()


def rate_denominator(stride: int, decay: float, n_effective: int) -> float:
    """``count / denom`` is hits-per-step over the decayed horizon.

    If the per-step hit rate ``r`` were constant then
    ``count = r * stride * (1 - decay**n) / (1 - decay)``. At the shipped
    defaults (stride 32, decay 0.95, n 64) this is 615.99.
    """
    if decay >= 1.0 or decay < 0.0:
        return float(stride) * max(int(n_effective), 1)
    if n_effective <= 0:
        return float(stride)
    return float(stride) * (1.0 - decay ** int(n_effective)) / (1.0 - decay)


# -------------------------------------------------------------- worker state


def _worker_state(worker: Any) -> tuple[Any, Any, bool]:
    """``(state, collector, has_loop)`` or ``fq_not_active``.

    ``maybe_init_fq_state`` degrades to a bare collector when the policy
    loop fails to boot (integration.py:119-125), and the heatmap is still
    perfectly meaningful in that mode — you just get no ``tier``, no
    ``interval`` and no ``policy_sha``. So unlike ``admin._loop_state``
    this accepts either object.
    """
    runner = getattr(worker, "model_runner", None)
    state = getattr(runner, "fq_collector", None)
    if state is None:
        raise AdminError(
            "fq_not_active",
            "no fungible-quant collector on this worker — VLLM_FQ_ENABLE "
            "is not set, or the model has no MoE routers to instrument",
            status=404, details={"has_runner": runner is not None})
    collector = getattr(state, "collector", state)
    if not hasattr(collector, "decayed"):
        raise AdminError(
            "fq_not_active",
            "the object on model_runner.fq_collector is not an "
            "FqStatsCollector (no decayed()); refusing to guess",
            status=404,
            details={"type": type(state).__name__})
    return state, collector, hasattr(state, "tier_of")


def _row_layers(state: Any, collector: Any, has_loop: bool) -> list[int]:
    """The row axis: policy layer ids when there is a loop, else the
    collector's bound layer ids. Shipped as data — a client that assumes
    ``layer = row + 3`` breaks on any other MoE span."""
    if has_loop:
        return [int(x) for x in state.layers]
    return sorted(int(x) for x in collector.count_buf)


def _cid_of(state: Any, has_loop: bool, layers: Sequence[int]) -> dict:
    """policy layer id -> collector layer id, the loop's own mapping."""
    if has_loop:
        mapping = getattr(state, "_collector_layer_map", None)
        if mapping:
            return {int(k): int(v) for k, v in mapping.items()}
    return {int(x): int(x) for x in layers}


# --------------------------------------------- cumulative accumulator (§7.4)


class _CumAccumulator:
    """A true cumulative counter, at zero hot-path cost.

    The collector has none: ``count_buf`` is zeroed at every roll
    (stats.py:295-296) and the window is decayed on read. But the ring
    itself is exact, undecayed int64 (stats.py:131-133), so integrating
    the slots that have rolled since the previous sample reconstructs an
    exact total — while the gap stays under ``window_len`` slots
    (``64 * 32 = 2,048`` engine steps, ~17 min at the observed cadence).
    Beyond that the ring has wrapped and slots are genuinely gone:
    :attr:`lossy` says so rather than reporting a plausible undercount.
    """

    def __init__(self) -> None:
        self.cum: np.ndarray | None = None
        self.rolled: int = 0
        self.since_step: int = 0
        self.lossy: bool = False
        self.dropped_slots: int = 0
        self.overflow_rebased: bool = False
        self.layers: tuple[int, ...] = ()

    def rebase(self, rolled: int, step: int,
               layers: Sequence[int], num_experts: int) -> int:
        previous = self.since_step
        self.cum = np.zeros((len(layers), int(num_experts)), dtype=np.int64)
        self.rolled = int(rolled)
        self.since_step = int(step)
        self.lossy = False
        self.dropped_slots = 0
        self.overflow_rebased = False
        self.layers = tuple(int(x) for x in layers)
        return previous

    def integrate(self, collector: Any, cids: Sequence[int],
                  layers: Sequence[int], rolled: int, win_pos: int,
                  step: int) -> None:
        import torch

        num_experts = int(collector.num_experts)
        if self.cum is None or self.layers != tuple(int(x) for x in layers):
            # First sample (or the row axis changed): start at zero here
            # rather than pretending to know the history.
            self.rebase(rolled, step, layers, num_experts)
            return
        window_len = int(collector.window_len)
        n_new = int(rolled) - int(self.rolled)
        if n_new <= 0:
            return
        if n_new > window_len:
            self.dropped_slots += n_new - window_len
            self.lossy = True
            n_new = window_len
        slots = [(int(win_pos) - 1 - i) % window_len for i in range(n_new)]
        rows = [collector._count_win[int(cid)][slots].sum(0) for cid in cids]
        self.cum += torch.stack(rows).cpu().numpy().astype(np.int64)
        self.rolled = int(rolled)
        # u32 on the wire; rebase long before the cast would wrap. At the
        # observed peak per-cell rate this is ~23 days of continuous
        # decode, so it is a correctness backstop, not a hot path.
        if self.cum.size and int(self.cum.max()) >= (1 << 31):
            self.rebase(rolled, step, layers, num_experts)
            self.overflow_rebased = True


#: One accumulator per worker process (the collector is per worker too).
_CUM = _CumAccumulator()


def reset_worker_state() -> None:
    """Test seam: forget the per-process cumulative accumulator."""
    global _CUM
    _CUM = _CumAccumulator()


# -------------------------------------------------------------- worker entry


def _parse_worker_request(req: Mapping[str, Any]) -> dict:
    include = set(req.get("include") or [])
    unknown = sorted(include - INCLUDE_TOKENS)
    if unknown:
        raise AdminError(
            "unknown_include",
            f"unknown include token(s) {unknown}; valid: "
            f"{sorted(INCLUDE_TOKENS)}", status=400)
    precision = str(req.get("precision") or "bf16")
    if precision not in PRECISIONS:
        raise AdminError(
            "bad_precision",
            f"precision={precision!r} is not one of {sorted(PRECISIONS)}",
            status=400)
    return {
        "op": str(req.get("op") or "sample"),
        "include": include,
        "precision": precision,
        "all_ranks": bool(req.get("all_ranks")),
        "layers": req.get("layers"),
    }


def _select_rows(all_layers: Sequence[int],
                 wanted: Sequence[int] | None) -> tuple[list[int], list[int]]:
    """``(row indices, layer ids)`` for the requested subset."""
    if not wanted:
        return list(range(len(all_layers))), [int(x) for x in all_layers]
    index = {int(v): i for i, v in enumerate(all_layers)}
    rows, ids = [], []
    missing = []
    for layer in wanted:
        i = index.get(int(layer))
        if i is None:
            missing.append(int(layer))
            continue
        rows.append(i)
        ids.append(int(layer))
    if missing:
        raise AdminError(
            "bad_layers",
            f"layer(s) {missing} are not instrumented on this serve; "
            f"instrumented layers are {int(all_layers[0])}.."
            f"{int(all_layers[-1])} ({len(all_layers)} rows)",
            status=400, details={"missing": missing})
    if not rows:
        raise AdminError("bad_layers", "empty layer selection", status=400)
    return rows, ids


def _rank_of(worker: Any, state: Any) -> int:
    rank = getattr(worker, "rank", None)
    if rank is None:
        rank = getattr(state, "rank", 0)
    try:
        return int(rank or 0)
    except (TypeError, ValueError):
        return 0


def worker_sample(worker: Any, request_json: str = "{}") -> str:
    """One rank's contribution to a sample. JSON string in, JSON out.

    Rank 0 returns the arrays; every other rank returns a ~200 B digest
    (see :data:`MERGE_RULE`). ``collective_rpc`` runs on every rank
    regardless — the async ``EngineClient`` path exposes no
    ``unique_reply_rank`` (protocol.py:227-235) — so the saving is on the
    socket, not in the workers.
    """
    try:
        _require_gates()
        req = _parse_worker_request(json.loads(request_json or "{}"))
        op = req["op"]
        if op == "sample":
            return _ok(_worker_do_sample(worker, req))
        if op == "meta":
            return _ok(_worker_do_meta(worker, req))
        if op == "reset_cum":
            return _ok(_worker_do_reset_cum(worker))
        if op == "zero_collector":
            return _ok(_worker_do_zero_collector(worker))
        raise AdminError("bad_scope", f"unknown worker op {op!r}", status=400)
    except AdminError as exc:
        return json.dumps({"ok": False, "error": exc.wire()},
                          sort_keys=True, separators=(",", ":"), default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("FQ heatmap worker call failed")
        return json.dumps({"ok": False, "error": {
            "code": "internal_error", "http_status": 500,
            "message": f"{type(exc).__name__}: {exc}", "details": {}}},
            sort_keys=True, separators=(",", ":"), default=str)


def _ok(payload: Mapping[str, Any]) -> str:
    return json.dumps({"ok": True, **payload}, sort_keys=True,
                      separators=(",", ":"), default=str)


def _window_meta(collector: Any) -> dict:
    rolled = int(getattr(collector, "_windows_rolled", 0))
    window_len = int(collector.window_len)
    stride = int(collector.window_stride)
    decay = float(collector.decay)
    n_eff = min(rolled, window_len)
    return {
        "len": window_len,
        "stride": stride,
        "decay": decay,
        "rolled": rolled,
        "n_effective": n_eff,
        "steps_since_roll": int(getattr(collector, "_step", 0)) % stride
        if stride else 0,
        "horizon_steps": stride / (1.0 - decay) if decay < 1.0 else None,
        "rate_denominator": rate_denominator(stride, decay, n_eff),
    }


def _worker_do_meta(worker: Any, req: Mapping[str, Any]) -> dict:
    state, collector, has_loop = _worker_state(worker)
    layers = _row_layers(state, collector, has_loop)
    return {
        "rank": _rank_of(worker, state),
        "layers": layers,
        "num_layers": len(layers),
        "num_experts": int(collector.num_experts),
        "cells": len(layers) * int(collector.num_experts),
        "mass_is_real": bool(collector.mass_is_real()),
        "has_loop": bool(has_loop),
        "window": _window_meta(collector),
        "step": int(getattr(collector, "_step", 0)),
        "policy_sha": getattr(state, "policy_sha", None) if has_loop else None,
    }


def _read_arrays(state: Any, collector: Any, has_loop: bool,
                 rows: Sequence[int], cids: Sequence[int],
                 include: set[str]) -> dict:
    """The device reads, in one place so the torn-read retry can redo
    exactly them and nothing else."""
    import torch

    counts, masses = [], []
    for cid in cids:
        c, m = collector.decayed(int(cid))
        counts.append(c.detach().cpu().numpy())
        masses.append(m.detach().cpu().numpy())
    out: dict[str, Any] = {
        "count": np.asarray(counts, dtype=np.float64),
        "mass": np.asarray(masses, dtype=np.float64),
    }
    if "live" in include:
        e = int(collector.num_experts)
        live = torch.stack([collector.count_buf[int(cid)][:e]
                            for cid in cids])
        out["live"] = live.detach().cpu().numpy().astype(np.float32)
    if has_loop:
        tier = np.asarray(state.tier_of)
        out["tier"] = tier[list(rows), :]
    return out


def _worker_do_sample(worker: Any, req: Mapping[str, Any]) -> dict:
    state, collector, has_loop = _worker_state(worker)
    all_layers = _row_layers(state, collector, has_loop)
    if not all_layers:
        raise AdminError(
            "fq_not_active",
            "the collector has no bound MoE layers yet — nothing to "
            "sample", status=404)
    rows, layers = _select_rows(all_layers, req.get("layers"))
    cid_map = _cid_of(state, has_loop, all_layers)
    cids = [cid_map.get(int(x), int(x)) for x in layers]
    include = set(req["include"])
    precision = req["precision"]
    warnings: list[str] = []

    # No tearing is possible in production — the RPC handler and
    # execute_model (which calls collector.step(), i.e. the roll) are
    # dispatched from the same worker loop over rpc_broadcast_mq, so they
    # are serialised. Verify it anyway: it costs two int reads and it
    # makes the failure legible if that ever stops being true.
    torn_retries = 0
    for attempt in range(2):
        rolled0 = int(getattr(collector, "_windows_rolled", 0))
        win_pos0 = int(getattr(collector, "_win_pos", 0))
        arrays = _read_arrays(state, collector, has_loop, rows, cids,
                              include)
        rolled1 = int(getattr(collector, "_windows_rolled", 0))
        if rolled1 == rolled0:
            break
        torn_retries += 1
        if attempt == 1:
            warnings.append(
                f"the window rolled during the read twice "
                f"({rolled0} -> {rolled1}); the sample is from the second "
                "attempt and may mix two windows")
    if torn_retries:
        warnings.append(f"torn read retried {torn_retries}x")

    window = _window_meta(collector)
    window["rolled"] = rolled0
    window["n_effective"] = min(rolled0, window["len"])
    window["rate_denominator"] = rate_denominator(
        window["stride"], window["decay"], window["n_effective"])
    step = int(getattr(collector, "_step", 0))
    mass_is_real = bool(collector.mass_is_real())

    payload: dict[str, Any] = {
        "rank": _rank_of(worker, state),
        "step": step,
        "real_steps": int(getattr(state, "_real_steps", 0))
        if has_loop else None,
        "interval": int(getattr(state, "_intervals_run", 0))
        if has_loop else None,
        "window": window,
        "layers": layers,
        "num_layers": len(layers),
        "num_experts": int(collector.num_experts),
        "cells": len(layers) * int(collector.num_experts),
        "mass_is_real": mass_is_real,
        "has_loop": bool(has_loop),
        "policy_sha": getattr(state, "policy_sha", None) if has_loop else None,
        "apply_mode": getattr(getattr(state, "cfg", None), "apply_mode", None)
        if has_loop else None,
        "precision": precision,
        "merge_rule": MERGE_RULE,
        "cum_since_step": None,
        "cum_lossy": False,
        "cum_dropped_slots": 0,
        "cum_overflow_rebased": False,
        "warnings": warnings,
    }

    count_b64 = encode_floats(arrays["count"].reshape(-1), precision)
    tier_b64 = None
    if has_loop:
        try:
            tier_b64 = encode_tier(arrays["tier"].reshape(-1))
        except ValueError as exc:
            raise AdminError("tier_out_of_domain", str(exc),
                             status=500) from None

    mass_b64 = None
    if mass_is_real and "mass" in include:
        mass_b64 = encode_floats(arrays["mass"].reshape(-1), precision)
    elif not mass_is_real:
        # Shipping a byte-identical second array to say "same as the
        # other one" is the easiest 50 % saving on the wire, and it is
        # free. NEVER inferred from array equality: a uniform router
        # produces equal arrays legitimately (stats.py:302-318).
        payload["warnings"].append(
            "mass is aliased to count (collector.mass_is_real() == false); "
            "set VLLM_FQ_GATE_MASS=1 at boot to record real gate mass")

    live_b64 = None
    if "live" in include:
        live_b64 = encode_floats(arrays["live"].reshape(-1), precision)

    cum_b64 = None
    if "cum" in include:
        _CUM.integrate(collector, cids, layers, rolled0, win_pos0, step)
        cum = _CUM.cum
        cum_b64 = encode_u32(np.clip(cum, 0, (1 << 32) - 1).reshape(-1))
        payload["cum_since_step"] = int(_CUM.since_step)
        payload["cum_lossy"] = bool(_CUM.lossy)
        payload["cum_dropped_slots"] = int(_CUM.dropped_slots)
        payload["cum_overflow_rebased"] = bool(_CUM.overflow_rebased)
        if _CUM.lossy:
            payload["warnings"].append(
                f"cum_count is LOSSY: {_CUM.dropped_slots} window slot(s) "
                f"rolled out of the {int(collector.window_len)}-slot ring "
                "between polls and are gone; poll more often than "
                f"{int(collector.window_len) * int(collector.window_stride)} "
                "engine steps for an exact cumulative")

    # The digest is over the CANONICAL WIRE BYTES, so every rank computes
    # it identically and cross-rank agreement costs 32 bytes per rank
    # instead of 76 KB. Over bf16 bytes it tolerates last-mantissa-bit
    # differences in the float64 accumulation, which is intentional.
    digest = hashlib.sha256()
    digest.update(count_b64.encode())
    digest.update((tier_b64 or "").encode())
    digest.update(f"|{step}|{rolled0}|{win_pos0}".encode())
    payload["digest"] = digest.hexdigest()[:16]
    payload["tier_digest"] = hashlib.sha256(
        (tier_b64 or "").encode()).hexdigest()[:8]
    payload["win_pos"] = win_pos0

    canonical = payload["rank"] == 0 or bool(req.get("all_ranks"))
    if not canonical:
        # ~180 B instead of ~77 KB. The merge only needs the digest and
        # the clocks; shipping this rank's metadata four times over would
        # be four copies of an answer the canonical rank already gave.
        return {k: payload[k] for k in
                ("rank", "step", "cells", "mass_is_real", "digest")} | {
            "canonical": False,
            "rolled": payload["window"]["rolled"],
        }
    payload["canonical"] = True
    payload["count"] = count_b64
    payload["mass"] = mass_b64
    payload["tier"] = tier_b64
    payload["live_count"] = live_b64
    payload["cum_count"] = cum_b64
    return payload


def _worker_do_reset_cum(worker: Any) -> dict:
    state, collector, has_loop = _worker_state(worker)
    layers = _row_layers(state, collector, has_loop)
    step = int(getattr(collector, "_step", 0))
    rolled = int(getattr(collector, "_windows_rolled", 0))
    previous = _CUM.rebase(rolled, step, layers, int(collector.num_experts))
    return {
        "rank": _rank_of(worker, state),
        "scope": "heatmap",
        "applied": True,
        "cum_since_step": step,
        "previous_cum_since_step": int(previous),
        "digest": f"{step}:{rolled}",
    }


def _worker_do_zero_collector(worker: Any) -> dict:
    """The destructive scope. Gated three times over; see §7.5.

    Zeroing the ring makes ``decayed()`` return all zeros
    (stats.py:326-330) and the POLICY LOOP reads exactly those arrays
    every interval (loop.py:661-667,709). The consequences are in the
    response body, not only in a doc, because the operator who reached
    this is the one who needs them.
    """
    if not collector_zero_allowed():
        raise AdminError(
            "collector_zero_not_allowed",
            f"zeroing the collector is refused unless "
            f"{HEATMAP_ALLOW_COLLECTOR_ZERO_ENV}=1; use "
            'scope="heatmap" (rebases this endpoint\'s own cumulative, '
            "harmless) or scope=\"client\" (snapshot and subtract)",
            status=409)
    state, collector, has_loop = _worker_state(worker)
    for lid in list(collector.count_buf):
        collector.count_buf[lid].zero_()
        collector.mass_buf[lid].zero_()
        collector._count_win[lid].zero_()
        collector._mass_win[lid].zero_()
    collector._win_pos = 0
    collector._windows_rolled = 0
    rank = _rank_of(worker, state)
    logger.warning(
        "FQ heatmap: COLLECTOR ZEROED on rank %d (%s=1). The policy loop "
        "reads these counters (loop.py:661-667); the next 1-2 intervals "
        "will score on a zeroed window.", rank,
        HEATMAP_ALLOW_COLLECTOR_ZERO_ENV)
    return {
        "rank": rank,
        "scope": "collector",
        "applied": True,
        "digest": "zeroed",
        "warnings": [
            ("the policy loop reads these counters (loop.py:661-667); the "
             "next 1-2 intervals will score on a zeroed window, the "
             "Jaccard router-shift guard will hold all proposals, and "
             "pin-forced pairs bypass hysteresis (policy.py:142-144) and "
             "may swap against index order rather than traffic"),
        ],
    }


class FqHeatmapWorker:
    """``--worker-extension-cls`` route to the same worker methods.

    vLLM injects exactly ONE extension class into ``Worker.__bases__``
    after asserting that no attribute collides
    (worker_base.py:261-286), so use this mixin **or** the core
    ``Worker.fq_heatmap_sample`` method, never both. On the GLM-5.2
    serve the extension slot is already held by
    ``fq_reload.FqReloadWorker``, so the core method is the only usable
    route there and this class exists for other deployments.
    """

    def fq_heatmap_sample(self, request_json: str = "{}") -> str:
        return worker_sample(self, request_json)


# ------------------------------------------------------------------- poison


class _PoisonState:
    """A ``collective_rpc`` timeout is not a transient — it desynchronises
    the channel.

    The broadcast is already enqueued (multiproc_executor.py:392) and the
    slow worker still executes it and still writes its response. Nothing
    drains that orphaned response, so the NEXT collective RPC on the
    queue can read the stale one. Retrying into that is how a telemetry
    endpoint corrupts the channel the admin API needs. So a timeout
    disables sampling until the serve restarts, or until an operator
    explicitly accepts the risk with ``POST /fq/heatmap/reset
    {"scope":"poison"}``.
    """

    def __init__(self) -> None:
        self.poisoned = False
        self.reason: str | None = None
        self.at_ms: int | None = None

    def poison(self, reason: str) -> None:
        first = not self.poisoned
        self.poisoned = True
        self.reason = reason
        self.at_ms = int(time.time() * 1000)
        if first:
            logger.error(
                "FQ heatmap: collective-RPC timeout (%s). The broadcast was "
                "already enqueued and the slow worker will still write its "
                "response, so the RPC channel may be desynchronised. "
                "Heatmap sampling is DISABLED until the serve restarts.",
                reason)

    def clear(self) -> None:
        self.poisoned = False
        self.reason = None
        self.at_ms = None


_POISON = _PoisonState()
_LAST_DIVERGENCE_LOG_MS = 0.0


def reset_state() -> None:
    """Test seam: clear the module-level poison flag and accumulator."""
    global _LAST_DIVERGENCE_LOG_MS
    _POISON.clear()
    _LAST_DIVERGENCE_LOG_MS = 0.0
    reset_worker_state()


# -------------------------------------------------------------- router utils


def _parse_include(raw: str | None) -> set[str]:
    if raw is None:
        return {"mass"}
    tokens = {t.strip() for t in str(raw).split(",") if t.strip()}
    unknown = sorted(tokens - INCLUDE_TOKENS)
    if unknown:
        raise AdminError(
            "unknown_include",
            f"unknown include token(s) {unknown}; valid: "
            f"{sorted(INCLUDE_TOKENS)}", status=400,
            details={"unknown": unknown, "valid": sorted(INCLUDE_TOKENS)})
    return tokens


def parse_layers(raw: str | None) -> list[int] | None:
    """``"20-40"`` or ``"3,7,9"`` -> a sorted list of layer ids."""
    if raw is None or not str(raw).strip():
        return None
    out: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part.lstrip("-"):
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if hi < lo:
                    raise ValueError(f"reversed range {part!r}")
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
        except ValueError as exc:
            raise AdminError(
                "bad_layers",
                f"malformed layers selector {raw!r} at {part!r}: {exc}",
                status=400) from None
    if not out:
        raise AdminError("bad_layers",
                         f"layers selector {raw!r} selected nothing",
                         status=400)
    return sorted(out)


def _dp_size(request: Any) -> int:
    for holder in (getattr(request.app, "state", None),
                   getattr(getattr(request.app, "state", None),
                           "engine_client", None)):
        cfg = getattr(holder, "vllm_config", None)
        parallel = getattr(cfg, "parallel_config", None)
        size = getattr(parallel, "data_parallel_size", None)
        if size is not None:
            try:
                return int(size)
            except (TypeError, ValueError):
                return 1
    return 1


def compute_etag(*, rolled: int, win_pos: int, tier_digest: str,
                 mass_is_real: bool, layers: Sequence[int], num_experts: int,
                 include: Sequence[str], precision: str, reduce: str,
                 step: int | None = None) -> str:
    """Weak ``ETag`` over everything that can change the arrays.

    Weak, not strong: ``sample_id`` / ``sampled_at_unix_ms`` /
    ``cache_age_ms`` differ between two bodies with the same basis, so a
    strong ETag would be a lie. ``step`` enters the basis only when
    ``include=live`` — ``live_count`` changes every step, which defeats
    revalidation by design.
    """
    basis = json.dumps({
        "v": SCHEMA_VERSION, "rolled": int(rolled), "win_pos": int(win_pos),
        "tier": tier_digest, "mass_is_real": bool(mass_is_real),
        "layers": hashlib.sha256(
            ",".join(str(int(x)) for x in layers).encode()).hexdigest()[:16],
        "E": int(num_experts), "include": sorted(include),
        "precision": precision, "reduce": reduce,
        "step": int(step) if step is not None else None,
    }, sort_keys=True, separators=(",", ":"))
    short = hashlib.sha256(basis.encode()).hexdigest()[:16]
    return f'W/"fqhm-{SCHEMA_VERSION}-{short}"'


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    if header.strip() == "*":
        return True
    for candidate in header.split(","):
        c = candidate.strip()
        if c.startswith("W/"):
            c = c[2:]
        bare = etag[2:] if etag.startswith("W/") else etag
        if c in (bare, etag):
            return True
    return False


def _map_worker_error(exc: AdminError) -> AdminError:
    """A worker refusal keeps its code; a worker *crash* becomes 503."""
    if exc.code == "internal_error":
        return AdminError(
            "fq_heatmap_worker_error",
            f"a worker raised while sampling: {exc.message}",
            status=503, details=exc.details)
    return exc


def _map_rpc_exception(exc: BaseException) -> AdminError:
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if isinstance(exc, TimeoutError) or name in ("TimeoutError",
                                                 "asyncio.TimeoutError"):
        _POISON.poison(text or "collective_rpc deadline exceeded")
        return AdminError(
            "fq_heatmap_timeout",
            f"collective_rpc({WORKER_METHOD}) exceeded the deadline: {text}",
            status=504,
            details={"guidance": (
                "the collective-RPC channel may be desynchronised after a "
                "timeout; heatmap sampling is disabled until the serve "
                "restarts")})
    if ("shutting down" in low or "sleeping" in low
            or "EngineDead" in name or "enginedead" in low):
        return AdminError("engine_unavailable",
                          f"the engine is not accepting utility calls: "
                          f"{text}", status=503)
    return AdminError("fq_heatmap_worker_error",
                      f"{name}: {text}", status=503)


# ------------------------------------------------------------------- router


def attach_router(app: Any, *, environ: Mapping[str, str] | None = None
                  ) -> bool:
    """Attach ``/fq/heatmap*``, if and only if both gates are on.

    Never raises: a read-only telemetry surface must not be able to stop
    a serve from booting.
    """
    if not heatmap_enabled(environ):
        logger.debug("FQ heatmap API not attached: %s", gate_reason(environ))
        return False
    try:
        app.include_router(build_router(environ=environ))
    except Exception:  # noqa: BLE001
        logger.exception(
            "FQ heatmap API failed to attach — continuing without it")
        return False
    logger.info(
        "FQ heatmap API enabled (%s=1): GET /fq/heatmap, GET "
        "/fq/heatmap/meta, POST /fq/heatmap/reset", HEATMAP_API_ENV)
    return True


def build_router(*, environ: Mapping[str, str] | None = None) -> Any:
    """The ``/fq/heatmap`` router. FastAPI is imported here, never at
    module scope (the package laziness contract).

    Two ``APIRouter``s may share the ``/fq`` prefix; FastAPI merges them,
    and the admin router's ``/fq/layer/{layer}`` cannot shadow
    ``/fq/heatmap`` because the literal segment differs.
    """
    import asyncio

    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    # ``from __future__ import annotations`` stringifies every annotation
    # and FastAPI resolves handler annotations against the MODULE
    # globals; fastapi is a local here, so publish the names or FastAPI
    # reads ``raw_request: "Request"`` as an unresolvable name, decides
    # it must be a query parameter, and answers 422 on every route.
    globals().setdefault("Request", Request)
    globals().setdefault("JSONResponse", JSONResponse)
    globals().setdefault("Response", Response)

    router = APIRouter(prefix="/fq", tags=["fungible-quant heatmap"])
    lock = asyncio.Lock()
    cache: dict[tuple, dict] = {}
    meta_cache: dict[str, Any] = {}
    seq = {"sample_id": 0}
    boot_id = uuid.uuid4().hex[:8]

    def engine_client(request: Request):
        return request.app.state.engine_client

    def _fail(exc: AdminError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    def _gates_and_token(request: Request) -> AdminError | None:
        """Gates + token only. No poison check, no DP check — the reset
        route must stay reachable WHILE poisoned."""
        if not heatmap_enabled(environ):
            return AdminError("fq_heatmap_disabled", gate_reason(environ),
                              status=404)
        token = str(_env(environ).get(HEATMAP_TOKEN_ENV, "")).strip()
        if token:
            import hmac

            sent = request.headers.get(TOKEN_HEADER, "")
            # Compare bytes: hmac.compare_digest refuses str operands with
            # non-ASCII code points and Starlette decodes headers as
            # latin-1, so a single 0x80..0xff byte would turn a 403 into
            # an unhandled 500.
            if not hmac.compare_digest(
                    sent.encode("utf-8", "surrogateescape"),
                    token.encode("utf-8", "surrogateescape")):
                return AdminError(
                    "fq_heatmap_forbidden",
                    f"{TOKEN_HEADER} missing or wrong "
                    f"({HEATMAP_TOKEN_ENV} is set on this serve)",
                    status=403)
        return None

    def _guard(request: Request, *, needs_rpc: bool = True
               ) -> AdminError | None:
        blocked = _gates_and_token(request)
        if blocked is not None:
            return blocked
        if needs_rpc and _POISON.poisoned:
            return AdminError(
                "fq_heatmap_poisoned",
                f"a previous heatmap sample timed out ({_POISON.reason}); "
                "no further collective RPC will be issued",
                status=503,
                details={"since_unix_ms": _POISON.at_ms, "guidance": (
                    "the collective-RPC channel may be desynchronised "
                    "after a timeout; heatmap sampling is disabled until "
                    "the serve restarts. POST /fq/heatmap/reset "
                    '{"scope":"poison"} to override.')})
        dp = _dp_size(request)
        if dp > 1:
            return AdminError(
                "dp_not_supported",
                "data_parallel_size > 1: DPLBAsyncMPClient."
                "call_utility_async gathers from every engine and returns "
                "only [0] (core_client.py:1449-1458), so this endpoint "
                "would report one replica's routing as the whole serve's",
                status=501, details={"data_parallel_size": dp})
        return None

    async def _collective(request: Request, payload: dict,
                          timeout: float | None = None) -> list[dict]:
        engine = engine_client(request)
        if timeout is None:
            timeout = _env_float(environ, HEATMAP_TIMEOUT_S_ENV,
                                 DEFAULT_TIMEOUT_S)
        try:
            raw = await engine.collective_rpc(
                WORKER_METHOD, timeout=timeout,
                args=(json.dumps(payload, sort_keys=True,
                                 separators=(",", ":"), default=str),))
        except AdminError:
            raise
        except BaseException as exc:  # noqa: BLE001
            raise _map_rpc_exception(exc) from None
        if raw is None:
            raise AdminError(
                "fq_heatmap_worker_error",
                f"{WORKER_METHOD} returned nothing", status=503)
        results = _rank_results(raw)
        err = _first_error(results)
        if err is not None:
            raise _map_worker_error(err)
        if not results:
            raise AdminError("fq_heatmap_worker_error",
                             "no rank answered", status=503)
        return results

    def _merge(results: list[dict], reduce: str) -> tuple[dict, list[str]]:
        """MERGE_RULE: rank 0's arrays, cross-checked by digest.

        Explicitly NOT a sum. See :data:`MERGE_RULE` — under TP the ranks
        are replicas of one gate, so summing would report 4x the traffic.
        """
        global _LAST_DIVERGENCE_LOG_MS
        canonical = next((r for r in results if r.get("canonical")),
                         results[0])
        agree = _agree(results, "digest")
        warns: list[str] = []
        if not agree:
            bad = [{"rank": r.get("rank", i), "digest": r.get("digest")}
                   for i, r in enumerate(results)
                   if r.get("digest") != canonical.get("digest")]
            warns.append(
                "cross-rank digests DISAGREE: canonical rank "
                f"{canonical.get('rank')} digest "
                f"{canonical.get('digest')} vs {bad}. Under tensor "
                "parallelism the gate is replicated, so the routing "
                "histograms should be identical; add ?reduce=all to diff "
                "the arrays. Served anyway (the picture is still useful).")
            now = time.time() * 1000
            if now - _LAST_DIVERGENCE_LOG_MS > 60_000:
                _LAST_DIVERGENCE_LOG_MS = now
                logger.warning("FQ heatmap: %s", warns[-1])
        return canonical, warns

    def _build_envelope(results: list[dict], *, include: set[str],
                        precision: str, reduce: str,
                        clamp_warning: str | None) -> dict:
        canonical, warns = _merge(results, reduce)
        seq["sample_id"] += 1
        cells = int(canonical.get("cells") or 0)
        envelope: dict[str, Any] = {
            "schema": SCHEMA,
            "sample_id": seq["sample_id"],
            "server_boot_id": boot_id,
            "sampled_at_unix_ms": int(time.time() * 1000),
            "cached": False,
            "cache_age_ms": 0,
            "step": canonical.get("step"),
            "real_steps": canonical.get("real_steps"),
            "interval": canonical.get("interval"),
            "window": canonical.get("window"),
            "layers": canonical.get("layers"),
            "num_layers": canonical.get("num_layers"),
            "num_experts": canonical.get("num_experts"),
            "cells": cells,
            "mass_is_real": bool(canonical.get("mass_is_real")),
            "ranks": {
                "count": len(results),
                "canonical": canonical.get("rank"),
                "agree": _agree(results, "digest"),
                "reduce": reduce,
                "merge_rule": MERGE_RULE,
                "digests": [{"rank": r.get("rank"), "digest": r.get("digest")}
                            for r in results],
            },
            "policy_sha": canonical.get("policy_sha"),
            "apply_mode": canonical.get("apply_mode"),
            "encoding": {
                "layout": "layer-major",
                "byte_order": "little",
                "count": precision,
                "mass": precision,
                "live_count": precision,
                "cum_count": "u32",
                "tier": "u8",
                "transfer": "base64",
            },
            "count": canonical.get("count"),
            "mass": canonical.get("mass"),
            "tier": canonical.get("tier"),
            "live_count": canonical.get("live_count"),
            "cum_count": canonical.get("cum_count"),
            "cum_since_step": canonical.get("cum_since_step"),
            "cum_lossy": bool(canonical.get("cum_lossy")),
            "cum_dropped_slots": int(canonical.get("cum_dropped_slots") or 0),
            "cum_overflow_rebased": bool(
                canonical.get("cum_overflow_rebased")),
            "warnings": list(canonical.get("warnings") or []) + warns,
            # Popped by _sample; the pieces the ETag basis needs that are
            # not otherwise on the envelope.
            "_basis": {"win_pos": canonical.get("win_pos", 0),
                       "tier_digest": str(canonical.get("tier_digest") or "")},
        }
        if clamp_warning:
            envelope["warnings"].append(clamp_warning)
        if reduce == "all":
            envelope["per_rank"] = results
        # Sanity: every array must decode to exactly ``cells``. Better to
        # refuse here than to let a client render a shifted heatmap.
        for name, wire in (("count", precision), ("mass", precision),
                           ("live_count", precision), ("tier", "u8"),
                           ("cum_count", "u32")):
            blob = envelope.get(name)
            if not blob:
                continue
            per_cell = {"bf16": 2, "f32": 4, "u8": 1, "u32": 4}[wire]
            got = len(_unb64(blob)) // per_cell
            if got != cells:
                raise AdminError(
                    "fq_heatmap_worker_error",
                    f"{name} decodes to {got} cells, expected {cells}",
                    status=503)
        return envelope

    async def _sample(request: Request, *, include: set[str], precision: str,
                      reduce: str, layers: list[int] | None,
                      max_age_ms: int) -> dict:
        floor = max(0, _env_int(environ, HEATMAP_MIN_PERIOD_MS_ENV,
                                DEFAULT_MIN_PERIOD_MS))
        clamp_warning = None
        effective = max_age_ms
        if effective < floor:
            clamp_warning = (
                f"max_age_ms={max_age_ms} was clamped up to the "
                f"{HEATMAP_MIN_PERIOD_MS_ENV} floor of {floor} ms; every "
                "sample stalls the engine step loop, so the floor caps how "
                "often that can happen")
            effective = floor
        key = (tuple(sorted(include)), precision, reduce,
               tuple(layers) if layers else None)
        now = time.time() * 1000

        hit = cache.get(key)
        if hit is not None and (now - hit["at_ms"]) <= effective:
            return _as_cached(hit, now)

        async with lock:
            # Re-check inside the lock: N concurrent polls cost ONE RPC.
            now = time.time() * 1000
            hit = cache.get(key)
            if hit is not None and (now - hit["at_ms"]) <= effective:
                return _as_cached(hit, now)
            payload = {"op": "sample", "include": sorted(include),
                       "precision": precision,
                       "all_ranks": reduce == "all"}
            if layers:
                payload["layers"] = layers
            results = await _collective(request, payload)
            envelope = _build_envelope(
                results, include=include, precision=precision,
                reduce=reduce, clamp_warning=clamp_warning)
            basis = envelope.pop("_basis")
            envelope["etag"] = compute_etag(
                rolled=(envelope["window"] or {}).get("rolled", 0),
                win_pos=basis["win_pos"],
                tier_digest=basis["tier_digest"],
                mass_is_real=envelope["mass_is_real"],
                layers=envelope["layers"] or [],
                num_experts=envelope["num_experts"] or 0,
                include=sorted(include), precision=precision, reduce=reduce,
                step=envelope["step"] if "live" in include else None)
            cache[key] = {"envelope": envelope,
                          "at_ms": time.time() * 1000}
            return _as_fresh(envelope)

    def _copy(envelope: dict) -> dict:
        """Shallow copy with the mutable bits detached, so a handler can
        annotate a response without editing the cached envelope."""
        out = dict(envelope)
        out["warnings"] = list(envelope.get("warnings") or [])
        return out

    def _as_fresh(envelope: dict) -> dict:
        out = _copy(envelope)
        out["cached"] = False
        out["cache_age_ms"] = 0
        return out

    def _as_cached(hit: dict, now: float) -> dict:
        env = _copy(hit["envelope"])
        env["cached"] = True
        env["cache_age_ms"] = int(now - hit["at_ms"])
        return env

    async def _respond(request: Request, envelope: dict):
        etag = envelope.pop("etag", None) or ""
        headers = {
            "Cache-Control": "no-cache, must-revalidate",
            "X-FQ-Sample-Id": str(envelope.get("sample_id")),
            "X-FQ-Step": str(envelope.get("step")),
            "X-FQ-Merge-Rule": MERGE_RULE,
        }
        if etag:
            headers["ETag"] = etag
        if etag and _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        gzip_ok = "gzip" in request.headers.get("accept-encoding", "").lower()
        if not gzip_ok:
            envelope["warnings"].append(
                "the client did not offer gzip; this body is ~2.4x the "
                "size of the compressed one")
        body = json.dumps(envelope, separators=(",", ":"),
                          default=str).encode()
        if gzip_ok:
            # vLLM's build_app installs no GZipMiddleware, so compress
            # here. Level 1, off the event loop: measured 2.20 ms vs
            # 5.23 ms at level 6 for 4 % fewer bytes.
            body = await run_in_threadpool(gzip_bytes, body, GZIP_LEVEL)
            headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(body))
        return Response(content=body, media_type="application/json",
                        headers=headers)

    # ------------------------------------------------------------- routes

    @router.get("/heatmap")
    async def fq_heatmap(raw_request: Request):
        blocked = _guard(raw_request)
        if blocked is not None:
            return _fail(blocked)
        q = raw_request.query_params
        try:
            include = _parse_include(q.get("include"))
            precision = str(q.get("precision") or "bf16")
            if precision not in PRECISIONS:
                raise AdminError(
                    "bad_precision",
                    f"precision={precision!r} is not one of "
                    f"{sorted(PRECISIONS)}", status=400)
            reduce = str(q.get("reduce") or "rank0")
            if reduce not in REDUCERS:
                raise AdminError(
                    "bad_reduce",
                    f"reduce={reduce!r} is not one of {sorted(REDUCERS)}",
                    status=400)
            layers = parse_layers(q.get("layers"))
            try:
                max_age_ms = int(q.get("max_age_ms", DEFAULT_MAX_AGE_MS))
            except (TypeError, ValueError):
                raise AdminError(
                    "bad_max_age",
                    f"max_age_ms={q.get('max_age_ms')!r} is not an int",
                    status=400) from None
            envelope = await _sample(
                raw_request, include=include, precision=precision,
                reduce=reduce, layers=layers,
                max_age_ms=max(0, max_age_ms))
        except AdminError as exc:
            return _fail(exc)
        return await _respond(raw_request, envelope)

    @router.get("/heatmap/meta")
    async def fq_heatmap_meta(raw_request: Request):
        # A cached meta answer costs no RPC, so it stays available even
        # when the sampling path is poisoned — a health check that goes
        # dark exactly when something broke is worse than useless.
        blocked = _guard(raw_request,
                         needs_rpc="shape" not in meta_cache)
        if blocked is not None:
            return _fail(blocked)
        token_required = bool(
            str(_env(environ).get(HEATMAP_TOKEN_ENV, "")).strip())
        base = {
            "schema": SCHEMA,
            "server_boot_id": boot_id,
            "merge_rule": MERGE_RULE,
            "encoding": {"layout": "layer-major", "byte_order": "little",
                         "count": "bf16", "mass": "bf16",
                         "live_count": "bf16", "cum_count": "u32",
                         "tier": "u8", "transfer": "base64"},
            "precisions": sorted(PRECISIONS),
            "include_tokens": sorted(INCLUDE_TOKENS),
            "reset_scopes": sorted(RESET_SCOPES),
            "min_period_ms": _env_int(environ, HEATMAP_MIN_PERIOD_MS_ENV,
                                      DEFAULT_MIN_PERIOD_MS),
            "timeout_s": _env_float(environ, HEATMAP_TIMEOUT_S_ENV,
                                    DEFAULT_TIMEOUT_S),
            "token_required": token_required,
            "collector_zero_allowed": collector_zero_allowed(environ),
            "notes": [
                ("include=live defeats the ETag by design: live_count "
                 "changes every engine step, so no two samples "
                 "revalidate."),
                ("mass is null iff mass_is_real is false; alias it to "
                 "count client-side and label the view accordingly."),
            ],
        }
        if "shape" in meta_cache:
            # Cached for the process lifetime: state.layers is fixed at
            # loop construction (loop.py:491) and cannot change at
            # runtime. Only fq_active is re-derived.
            return JSONResponse(content={**base, **meta_cache["shape"],
                                         "fq_active": True, "cached": True,
                                         "poisoned": _POISON.poisoned})
        try:
            results = await _collective(raw_request, {"op": "meta"})
        except AdminError as exc:
            if exc.code == "fq_not_active":
                # A health check that 404s is useless as a health check:
                # report the shape we do not have as a FIELD.
                return JSONResponse(content={
                    **base, "fq_active": False, "cached": False,
                    "layers": None, "num_layers": None, "num_experts": None,
                    "cells": None, "mass_is_real": None,
                    "reason": exc.body()["error"]})
            return _fail(exc)
        canonical = results[0]
        shape = {
            "layers": canonical.get("layers"),
            "num_layers": canonical.get("num_layers"),
            "num_experts": canonical.get("num_experts"),
            "cells": canonical.get("cells"),
            "mass_is_real": canonical.get("mass_is_real"),
            "has_loop": canonical.get("has_loop"),
            "window": canonical.get("window"),
            "ranks": len(results),
        }
        meta_cache["shape"] = shape
        return JSONResponse(content={**base, **shape, "fq_active": True,
                                     "cached": False,
                                     "poisoned": _POISON.poisoned})

    @router.post("/heatmap/reset")
    async def fq_heatmap_reset(raw_request: Request):
        # The poison scope must be reachable WHILE poisoned, so it is
        # handled before the guard's poison check.
        try:
            body: dict = {}
            if (await raw_request.body()).strip():
                try:
                    body = dict(await raw_request.json())
                except Exception:  # noqa: BLE001
                    raise AdminError("bad_json", "body is not valid JSON",
                                     status=400) from None
            scope = str(body.get("scope") or "client")
            if scope not in RESET_SCOPES:
                raise AdminError(
                    "bad_scope",
                    f"scope={scope!r} is not one of {sorted(RESET_SCOPES)}",
                    status=400)
        except AdminError as exc:
            return _fail(exc)

        if scope == "poison":
            blocked = _gates_and_token(raw_request)
            if blocked is not None:
                return _fail(blocked)
            was = _POISON.poisoned
            _POISON.clear()
            return JSONResponse(content={
                "scope": "poison", "applied": bool(was), "was_poisoned": was,
                "note": ("the poison flag is cleared; the collective-RPC "
                         "channel may still be desynchronised — this is an "
                         "operator override, not a repair")})

        # scope=client issues no RPC, so a poisoned channel does not stop
        # an operator marking a baseline.
        blocked = _guard(raw_request, needs_rpc=scope != "client")
        if blocked is not None:
            return _fail(blocked)

        if scope == "client":
            # The default, and what the page uses. The server does
            # nothing: the client snapshots the decoded count array and
            # subtracts. It cannot destroy data the policy loop and every
            # other viewer are reading, compare mode comes free, and it
            # is honest that the result is a difference of two DECAYED
            # WINDOWS, not "traffic since the mark".
            hit = max(cache.values(), key=lambda h: h["at_ms"],
                      default=None)
            env = hit["envelope"] if hit else {}
            return JSONResponse(content={
                "scope": "client", "applied": False,
                "baseline_sample_id": env.get("sample_id"),
                "baseline_step": env.get("step"),
                "reason": body.get("reason"),
                "note": ("no server state was changed; snapshot and "
                         "subtract client-side. The difference is a "
                         "difference of two decayed windows, not traffic "
                         'since the mark — label the view "change since '
                         'mark".')})

        if scope == "heatmap":
            try:
                results = await _collective(raw_request, {"op": "reset_cum"})
            except AdminError as exc:
                return _fail(exc)
            cache.clear()
            canonical = results[0]
            return JSONResponse(content={
                "scope": "heatmap", "applied": True, "ranks": len(results),
                "agree": _agree(results, "digest"),
                "cum_since_step": canonical.get("cum_since_step"),
                "previous_cum_since_step": canonical.get(
                    "previous_cum_since_step"),
                "reason": body.get("reason"),
                "note": ("cumulative accumulator rebased; the collector "
                         "window and the policy are untouched. Poll "
                         "?include=cum for exact hits since this point.")})

        # scope == "collector": the destructive one.
        if not collector_zero_allowed(environ):
            return _fail(AdminError(
                "collector_zero_not_allowed",
                f"zeroing the collector is refused unless "
                f"{HEATMAP_ALLOW_COLLECTOR_ZERO_ENV}=1; it blinds the "
                "policy loop for 1-2 intervals (loop.py:661-667). Use "
                'scope="heatmap" instead.', status=409))
        try:
            # All ranks, never rank 0 only: divergent policy inputs would
            # break T6 cross-rank decision agreement (loop.py:694-698).
            results = await _collective(raw_request,
                                        {"op": "zero_collector"})
        except AdminError as exc:
            return _fail(exc)
        cache.clear()
        logger.warning(
            "FQ heatmap: collector zeroed on %d rank(s) by an operator "
            "(reason=%r)", len(results), body.get("reason"))
        return JSONResponse(content={
            "scope": "collector", "applied": True, "ranks": len(results),
            "reason": body.get("reason"),
            "warnings": list(results[0].get("warnings") or [])})

    router.fq_heatmap_cache = cache          # type: ignore[attr-defined]
    router.fq_heatmap_seq = seq              # type: ignore[attr-defined]
    return router
