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
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "FQ loop init failed — falling back to collector-only")
        return collector
    return state if state is not None else collector


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
