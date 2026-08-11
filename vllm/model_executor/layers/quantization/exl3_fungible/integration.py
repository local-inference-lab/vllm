# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant M1 integration hook.

Wires the :class:`FqStatsCollector` into the model runner: called from
``gpu_worker.initialize_from_config`` right after the (optional)
routed-experts capturer init and BEFORE CUDA-graph capture, so the
collector's capture fn is recorded into the graphs.

The routed-experts capturer is only bound when
``enable_return_routed_experts`` is set, so this hook must be its own
call site — never piggybacked on that flag. ``bind_router`` chains any
previously bound capture fn, so both orders (capturer bound or not)
are safe.

Laziness contract: this module imports nothing from vllm at module
level, and ``maybe_init_fq_collector`` returns before importing
anything when ``VLLM_FQ_ENABLE`` is off — zero import cost for the
default path. The vllm symbols are resolved through the two seam
functions ``_moe_module_types`` / ``_collector_cls`` so CPU tests can
monkeypatch fakes in without building ``vllm._C``.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )

FQ_ENABLE_ENV = "VLLM_FQ_ENABLE"
FQ_WINDOW_LEN_ENV = "VLLM_FQ_WINDOW_LEN"
FQ_WINDOW_STRIDE_ENV = "VLLM_FQ_WINDOW_STRIDE"
FQ_DECAY_ENV = "VLLM_FQ_DECAY"
# Opt-in: record REAL gate mass (sum of routing weights) as well as hit
# counts. Off by default because it adds a guarded scatter_add_ to the
# per-layer capture path; the count-only default is one histc
# (PERFORMANCE.md: M1 gate is <0.5% decode overhead). With it off, mass
# is aliased to count and the collector reports mass_is_real() == False.
FQ_GATE_MASS_ENV = "VLLM_FQ_GATE_MASS"


def _moe_module_types() -> tuple[type, type]:
    """Lazy isinstance targets ``(MoERunner, BaseRouter)``.

    Monkeypatch seam for CPU tests (inject fake classes); lazy so the
    env-off path never touches the fused_moe import graph.

    ``MoERunner`` is imported from its defining module first: upstream moved
    it to ``fused_moe.runner.moe_runner`` and left ``fused_moe.layer`` as a
    re-export, which is exactly the kind of compatibility shim that gets
    dropped in a later cleanup. Falling back the other way keeps us working on
    both old and new trees instead of binding to the shim.
    """
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            MoERunner,
        )
    except ImportError:  # older trees: only the re-export exists
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        BaseRouter,
    )
    return MoERunner, BaseRouter


def _collector_cls() -> "type[FqStatsCollector]":
    """Lazy :class:`FqStatsCollector`; monkeypatch seam for CPU tests."""
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
    return FqStatsCollector


def _window_kwargs_from_env() -> dict:
    """Window knobs from env; unset knobs fall through to the
    ``FqStatsCollector`` signature defaults (single source of truth)."""
    kwargs: dict = {}
    v = os.environ.get(FQ_WINDOW_LEN_ENV)
    if v is not None:
        kwargs["window_len"] = int(v)
    v = os.environ.get(FQ_WINDOW_STRIDE_ENV)
    if v is not None:
        kwargs["window_stride"] = int(v)
    v = os.environ.get(FQ_DECAY_ENV)
    if v is not None:
        kwargs["decay"] = float(v)
    if os.environ.get(FQ_GATE_MASS_ENV, "0") == "1":
        kwargs["record_mass"] = True
    return kwargs


def maybe_init_fq_state(runner):
    """Build the full M2 loop (collector + policy/decision state machine).

    The model runner's step sites call ``.step(is_dummy=...)`` on
    whatever this returns; :class:`FungibleQuantState` and
    :class:`FqStatsCollector` share that contract, so the M1 call sites
    need no change. Degrades stepwise: env off / no MoE -> None; loop
    boot failure (no policy source, bad store, ...) -> bare collector
    (M1 observability only), never a dead engine.
    """
    collector = maybe_init_fq_collector(runner)
    if collector is None:
        return None
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    try:
        from vllm.model_executor.layers.quantization.exl3_fungible import (
            loop as fq_loop,
        )
        state = fq_loop.build_from_env(collector, rank=rank)
        if state is not None:
            _bind_apply_fn(state, runner, rank)
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "FQ loop init failed — falling back to collector-only")
        return collector
    return state if state is not None else collector


#: Live apply is gated on the runtime the swap is proven safe in.
FQ_LIVE_APPLY_ENV = "VLLM_FQ_LIVE_APPLY"


def _graphs_are_live(runner) -> bool:
    """True when a captured CUDA graph may replay over the expert slabs."""
    try:
        cfg = runner.vllm_config
        if bool(getattr(cfg.model_config, "enforce_eager", False)):
            return False
        mode = getattr(cfg.compilation_config, "cudagraph_mode", None)
        return not (mode is None or str(mode).endswith("NONE"))
    except Exception:  # noqa: BLE001 — unknown runtime is the unsafe one
        return True


def _bind_apply_fn(state, runner, rank: int) -> None:
    """Give the loop the swap backend it has been deciding without.

    Until now ``build_from_env`` never passed ``apply_fn``, so every interval
    logged "recording proposals only (M4 live wiring pending)" and
    ``_maybe_apply`` returned False before touching a weight — 64 swaps
    decided across 39 layers, zero installed.

    DISABLED BY DEFAULT, because the first live attempt DEADLOCKED THE ENGINE.

        16:55:42  FQ decide step=100 interval=1 swaps=64   <- first live apply
        16:55:50  generation throughput 10.4 tokens/s
        16:56:00  0.0 tokens/s
        16:56:43  No available shared memory broadcast block found in 60s

    The reasoning that led here was about CUDA-graph replay, and that was the
    wrong hazard. ``engine.stage()`` reads hundreds of MB of fragments
    SYNCHRONOUSLY, inside the model runner's step, INDEPENDENTLY ON EVERY TP
    RANK. Four ranks doing unequal amounts of IO at an implicit collective
    boundary drift apart, and the shared-memory broadcast starves. Eager
    execution does not help: this is TP lockstep, not graph safety.

    The admin path does not have this problem because it is coordinated from
    OUTSIDE the step -- the router drains, then dispatches to every rank via
    collective_rpc, then gathers. That is the shape an automatic apply needs
    too: propose inside the step, apply outside it, together.

    Until that exists, ``VLLM_FQ_LIVE_APPLY=1`` binds this anyway for
    experiments on a serve nobody is depending on. It is not a supported mode.

    GATED ON EAGER EXECUTION as well, for when it is re-enabled. ``admin.apply_retier`` can pass
    ``quiesce=nullcontext()`` because the HTTP request drains the engine
    first (``drain_mode="wait"``). The loop has no such drain: it decides
    inside the runner's step. With ``enforce_eager`` nothing is replaying and
    device work ordered on the same stream is ordered after the previous
    forward, so a nullcontext is honest. With CUDA graphs captured, a replay
    holds device pointers into the very slabs being rewritten — that needs a
    real drain, which does not exist yet.

    So: bind under eager, refuse loudly under graphs, and let an operator
    override only on purpose.
    """
    import logging

    log = logging.getLogger(__name__)
    # DEFAULT OFF. "auto" (bind whenever eager) deadlocked a live TP4 serve on
    # its first apply -- see the block comment below. Inline apply is opt-in
    # until it is coordinated across ranks.
    mode = os.environ.get(FQ_LIVE_APPLY_ENV, "off").strip().lower()
    if mode in ("0", "off", "false"):
        log.info("FQ live apply disabled by %s=%s — proposals only",
                 FQ_LIVE_APPLY_ENV, mode)
        return
    if mode == "auto" and _graphs_are_live(runner):
        log.warning(
            "FQ live apply NOT bound: CUDA graphs are captured and the loop "
            "has no drain, so a replay could read slabs mid-rewrite. The loop "
            "will record proposals only. Run with --enforce-eager, or set "
            "%s=1 to override deliberately.", FQ_LIVE_APPLY_ENV)
        return

    from vllm.model_executor.layers.quantization.exl3_fungible import (
        admin as _admin,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        swap as _swap,
    )

    # Size the staging buffers to the LOOP's cap, not the admin API's.
    # build_swap_engine defaults max_pairs to DEFAULT_MAX_ITEMS (32), which is
    # an operator-facing batch limit for hand-driven POST /fq/retier. The loop
    # proposes up to max_swaps_total (64) in one interval, so every interval
    # died on:
    #     ValueError: plan has 64 pairs, staging holds 32
    # The loop degraded correctly -- "keeping the incumbent tiering and
    # retrying next interval" -- but it could never succeed, so the serve
    # would have retried forever while looking busy.
    _cap = int(getattr(getattr(state, "cfg", None), "max_swaps_total", 0) or 0)
    engine = _admin.build_swap_engine(
        runner, rank=rank, max_pairs=max(_cap, _admin.DEFAULT_MAX_ITEMS))
    if engine is None:
        log.warning("FQ live apply NOT bound: no mixed-trellis layers "
                    "registered — a uniform-K serve has nothing to swap")
        return

    def apply_fn(proposed_doc, swaps) -> bool:
        # The loop indexes layers by ROW; the engine keys by GLOBAL layer id.
        plan = _swap.SwapPlan([
            (int(state.layers[row]), int(e_out), int(e_in))
            for row, e_out, e_in in swaps])
        # fail_atomic stages the pre-swap rows so an abort before the
        # visibility flip restores inside the same window. drop lets a pair
        # whose fragments cannot be supplied pend instead of losing the
        # interval; cardinality is preserved either way.
        if len(plan.swaps) > int(engine.max_pairs):
            # Configuration error, not a runtime condition: say which two
            # knobs disagree rather than surfacing a buffer-size ValueError.
            log.error(
                "FQ live apply: plan has %d pairs but staging holds %d — "
                "raise VLLM_FQ_MAX_SWAPS_TOTAL/max_pairs together or lower "
                "the loop cap; skipping this interval",
                len(plan.swaps), int(engine.max_pairs))
            return False
        staged = engine.stage(plan, fail_atomic=True, on_unavailable="drop")
        dropped = tuple(getattr(staged, "dropped", ()) or ())
        if dropped:
            log.warning("FQ live apply: %d pair(s) dropped for missing "
                        "fragments, %d applied", len(dropped),
                        len(getattr(staged, "plan", plan).swaps))
        if not getattr(getattr(staged, "plan", plan), "swaps", ()):
            return False
        import contextlib

        report = engine.apply(
            staged=staged,
            quiesce=contextlib.nullcontext(),
            memo_hook=None,          # correct for the mixed runtime
            policy_doc=dict(proposed_doc),
        )
        ok = bool(getattr(report, "ok", True))
        log.info("FQ live apply: %d swap(s) %s", len(plan.swaps),
                 "installed" if ok else "REFUSED")
        return ok

    state.apply_fn = apply_fn
    log.info("FQ live apply BOUND: %d mixed layers, rank %d — decisions will "
             "now change weights", len(getattr(engine, "layers", {})), rank)


def maybe_init_fq_collector(runner) -> "FqStatsCollector | None":
    """Build and bind the FQ stats collector, if enabled and applicable.

    Returns None (binding nothing, importing nothing heavy) unless
    ``VLLM_FQ_ENABLE=1`` and the model contains at least one MoERunner
    with a BaseRouter router. Otherwise binds EVERY such router via
    ``collector.bind_router`` — which chains a previously installed
    capture fn (e.g. the routed-experts capturer), so it is safe to
    call whether or not ``enable_return_routed_experts`` already bound
    one. Must run before CUDA-graph capture.

    Args:
        runner: the GPU model runner (needs ``.model.modules()`` and
            ``.device``).
    """
    if os.environ.get(FQ_ENABLE_ENV, "0") != "1":
        return None

    moe_runner_cls, base_router_cls = _moe_module_types()

    routers: list[tuple[int, object]] = []
    for module in runner.model.modules():
        if (isinstance(module, moe_runner_cls)
                and isinstance(module.router, base_router_cls)):
            routers.append((module.layer_id, module.router))
    if not routers:
        return None

    collector = _collector_cls()(
        routers[0][1].global_num_experts,
        device=runner.device,
        **_window_kwargs_from_env(),
    )
    for layer_id, router in routers:
        collector.bind_router(layer_id, router)
    # State the resolved mass mode in the engine log: "mass" is present in
    # every stats artifact either way, and an operator must not have to
    # diff it against "count" to discover which one they got.
    import logging

    logging.getLogger(__name__).info(
        "FQ stats: bound %d MoE routers, %d experts, gate mass %s (%s=%s)",
        len(routers), collector.num_experts,
        "RECORDED" if collector.mass_is_real() else "ALIASED TO COUNT",
        FQ_GATE_MASS_ENV, os.environ.get(FQ_GATE_MASS_ENV, "0"))
    return collector
