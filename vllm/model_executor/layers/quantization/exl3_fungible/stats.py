# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant stats collector (M1).

Per-MoE-layer routing statistics with a CUDA-graph-safe capture path:
the capture fn does pure tensor ops on persistent, pre-allocated device
buffers — no host reads, no allocation, no Python-object mutation — so it
can be captured into CUDA graphs and keeps recording during replay
(validated by T1 in tests/exl3_fungible/ and the research branch T1 rig).

Host-side windowing runs OFF the hot path in step(), mirroring EPLB's
step cadence: every ``window_stride`` engine steps the accumulators are
rolled into a circular window buffer and zeroed; the policy engine reads
an exponentially-decayed view of the window.

Binding: chain through ``BaseRouter.set_capture_fn`` — the slot is
single-occupancy and the routed-experts capturer may already hold it
(gpu_model_runner binds it at init), so ``bind_router`` always chains the
previous fn after recording.

Gate mass (``record_mass=True``, ``VLLM_FQ_GATE_MASS=1``) is OPT-IN. The
base capture contract passes only ``topk_ids``; the gate weights are a
local in ``BaseRouter._select_experts`` and are handed to capture fns
tagged ``wants_topk_weights``. Recording them costs a guarded
``scatter_add_`` on top of the count histogram, which is why it is not
on by default (PERFORMANCE.md: M1 gate is <0.5% decode overhead).
Without it, ``mass`` is aliased to ``count`` at read time and
``mass_is_real()`` reports False — the alias is never silent.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


class FqStatsCollector:
    """Routing-count and gate-mass accumulators for one model replica.

    Args:
        num_experts: logical expert count per MoE layer (EP=1 → logical
            == physical; D4 keeps the policy domain logical).
        window_len: number of window slots retained.
        window_stride: engine steps per window roll.
        decay: exponential decay factor applied per window step in the
            policy view (λ from Phase 0a).
        device: device for the persistent buffers.
        record_mass: record REAL gate mass (sum of routing weights) in
            addition to hit counts. Opt-in: it adds a guarded
            ``scatter_add_`` to the capture path (see
            :meth:`make_capture_fn`). When off, ``mass`` is aliased to
            ``count`` and :meth:`mass_is_real` reports False.
    """

    def __init__(
        self,
        num_experts: int,
        *,
        window_len: int = 64,
        window_stride: int = 32,
        decay: float = 0.95,
        device: torch.device | str = "cuda",
        record_mass: bool = False,
    ) -> None:
        self.num_experts = num_experts
        self.window_len = window_len
        self.window_stride = window_stride
        self.decay = decay
        self.device = torch.device(device)
        self.record_mass = bool(record_mass)
        # layer_id -> persistent accumulators (allocated at bind time)
        self.count_buf: dict[int, torch.Tensor] = {}
        self.mass_buf: dict[int, torch.Tensor] = {}
        # Persistent 0-dim int64 tensor holding ``num_experts`` — the
        # overflow-slot index used by the weighted path's OOR redirect.
        # Pre-allocated so the capture fn never allocates (graph safety)
        # and never pays a full_like fill per call.
        self._oor_slot = torch.tensor(
            num_experts, dtype=torch.int64, device=self.device)
        # Layers whose capture fn records no gate mass: mass is aliased
        # to count at read time (decayed/summary).
        self._mass_is_count: set[int] = set()
        # layer_id -> [window_len, E] rolled snapshots
        self._count_win: dict[int, torch.Tensor] = {}
        self._mass_win: dict[int, torch.Tensor] = {}
        self._win_pos = 0
        self._windows_rolled = 0
        self._step = 0

    # ------------------------------------------------------------------ bind

    def bind_router(self, layer_id: int, router,
                    record_mass: bool | None = None) -> None:
        """Allocate this layer's buffers and chain the capture fn.

        Must run BEFORE CUDA-graph capture (i.e. at model-runner init,
        alongside the routed-experts capturer) so the capture fn's ops are
        recorded into the graphs.

        The accumulators carry one extra OVERFLOW slot at index
        ``num_experts`` for the weighted (scatter) capture path: any id
        outside ``[0, num_experts)`` (padding sentinels, garbage in
        replayed capture batches) is redirected there instead of
        corrupting device memory via an out-of-range ``scatter_add_``.
        The default histc path drops OOR ids by construction. Reads
        ignore the slot either way. Counts are float32 (histc output;
        exact below 2^24 per window).

        Gate mass requires the ROUTER to hand us ``topk_weights``: the
        base contract passes only ids, and the weights are a local in
        ``BaseRouter._select_experts`` — nothing on the router object
        holds them at capture time. We tag the capture fn with
        ``wants_topk_weights`` and then VERIFY the router honoured it
        (``router.capture_fn_wants_weights``). A runtime whose
        ``BaseRouter`` predates that contract would otherwise hand us
        ids only and leave ``mass`` at zero while we claimed it was
        real, so an unsupported router downgrades to count-only and
        :meth:`mass_is_real` stays False.
        """
        if record_mass is None:
            record_mass = self.record_mass
        count = torch.zeros(
            self.num_experts + 1, dtype=torch.float32, device=self.device)
        mass = torch.zeros(
            self.num_experts + 1, dtype=torch.float32, device=self.device)
        self.count_buf[layer_id] = count
        self.mass_buf[layer_id] = mass
        self._count_win[layer_id] = torch.zeros(
            (self.window_len, self.num_experts),
            dtype=torch.int64, device=self.device)
        self._mass_win[layer_id] = torch.zeros(
            (self.window_len, self.num_experts),
            dtype=torch.float32, device=self.device)

        prev = getattr(router, "capture_fn", None)
        router.set_capture_fn(
            self.make_capture_fn(layer_id, prev_fn=prev,
                                 record_mass=record_mass))
        if record_mass and not getattr(
                router, "capture_fn_wants_weights", False):
            logger.warning(
                "FQ stats: router for layer %s does not pass topk_weights "
                "to capture fns (no capture_fn_wants_weights) — gate mass "
                "cannot be recorded; falling back to count-only for this "
                "layer (mass will be aliased to count).", layer_id)
            router.set_capture_fn(
                self.make_capture_fn(layer_id, prev_fn=prev,
                                     record_mass=False))

    def make_capture_fn(
        self,
        layer_id: int,
        prev_fn: Callable | None = None,
        topk_weights_getter: Callable[[], torch.Tensor] | None = None,
        record_mass: bool | None = None,
    ) -> Callable:
        """Build the graph-safe capture fn for one layer.

        The returned callable only touches pre-allocated buffers with
        tensor ops, and is sized for the M1 hot-path gate: the count-only
        path is a single ``histc`` histogram per layer (~3 small kernels:
        cast, histc, add) instead of a guarded ``scatter_add_`` chain
        (~10 kernels — measured 4-5% decode overhead on the Fruit proxy,
        PERFORMANCE.md gate is <0.5%).

        With ``record_mass`` the fn additionally accumulates REAL gate
        mass — the sum of the routing weights, so a confident route
        outranks a marginal one, which is what the swap policy is
        specified to score on. That costs 5 extra device kernels (int64
        cast, ``lt``, ``where``, ``clamp_``, ``scatter_add_``) — 8 vs 3,
        measured 1.9x the CPU wall time of the count-only path at
        E=256/topk=8 — so it is OPT-IN (``VLLM_FQ_GATE_MASS=1``), NOT the
        default: a ~10-kernel guarded scatter chain previously measured
        4-5% decode overhead against a 0.5% gate. Counts are computed by
        the SAME histc in both modes, so turning mass on cannot perturb
        the count signal.

        Out-of-range ids (padding sentinels, garbage in replayed capture
        batches) MUST never index device memory: an OOR ``scatter_add_``
        is an illegal memory access that kills the engine (observed live
        on the first M2 dryrun boot). ``histc`` drops values outside
        [0, num_experts) by construction; the weighted path redirects
        them to the explicit overflow slot at index ``num_experts``,
        which reads slice off.

        Weights reach the fn either as the second positional argument
        (the router hands them over — the fn is tagged
        ``wants_topk_weights`` so ``BaseRouter._select_experts`` calls
        it with both) or, for tests and non-BaseRouter call sites, from
        ``topk_weights_getter``. Passing a getter implies
        ``record_mass``.

        When mass is not recorded it is aliased to count at read time
        (``decayed``/``summary``) so the signal degrades gracefully;
        :meth:`mass_is_real` tells a consumer which of the two it is
        looking at.
        """
        count = self.count_buf[layer_id]
        mass = self.mass_buf[layer_id]
        num_experts = self.num_experts
        oor_slot = self._oor_slot
        if record_mass is None:
            record_mass = self.record_mass or topk_weights_getter is not None

        if not record_mass:
            self._mass_is_count.add(layer_id)

            def _capture(topk_ids: torch.Tensor,
                         topk_weights: torch.Tensor | None = None) -> None:
                # One EXTRA bin, then slice it off. torch.histc's last bin is
                # CLOSED at max, so binning [0, E) as bins=E/max=E does not
                # drop id == E — it folds the padding sentinel into the last
                # real expert, biasing the routing histogram toward exactly
                # one expert on every batch. That is memory-safe but it
                # silently poisons the signal the swap policy reads.
                # With bins=E+1/max=E+1 each bin is exactly 1 wide, so E (and
                # E+1, via the same closed-last-bin rule) land in the overflow
                # bin and are discarded by the slice; ids <0 or >E+1 still
                # fall outside the range and are dropped by histc itself.
                # Exact for id values < 2^24 in fp32.
                flat = topk_ids.flatten().to(torch.float32)
                hist = torch.histc(flat, bins=num_experts + 1,
                                   min=0, max=num_experts + 1)
                count[:num_experts].add_(hist[:num_experts])
                if prev_fn is not None:
                    prev_fn(topk_ids)

            _capture.wants_topk_weights = False
            return _capture

        self._mass_is_count.discard(layer_id)

        def _capture(topk_ids: torch.Tensor,
                     topk_weights: torch.Tensor | None = None) -> None:
            # Count: identical histc to the count-only path (same
            # closed-last-bin sentinel fix), so the count signal does not
            # depend on whether mass recording is enabled.
            flat = topk_ids.flatten()
            hist = torch.histc(flat.to(torch.float32), bins=num_experts + 1,
                               min=0, max=num_experts + 1)
            count[:num_experts].add_(hist[:num_experts])
            if topk_weights is None and topk_weights_getter is not None:
                topk_weights = topk_weights_getter()
            if topk_weights is not None:
                # Mass: guarded scatter. Every id outside [0, E) — the
                # padding sentinel E included — is redirected to the
                # overflow slot at E, so scatter_add_ can never address
                # past the buffer, and reads ([:E]) drop it exactly the
                # way histc drops it from the count.
                #
                # Two-op guard rather than the obvious
                # ``where(ids.ge(0) & ids.lt(E), ids, E)``: negatives are
                # sent to the slot first, THEN everything is clamped down
                # to it, which covers ids >= E (E itself, E+1, garbage)
                # without a second compare. One kernel cheaper on a path
                # whose whole cost is kernel launches. A bare
                # ``clamp(0, E)`` would be wrong — it maps negatives onto
                # real expert 0.
                ids = flat.to(torch.int64)
                idx = torch.where(ids.lt(0), oor_slot, ids)
                idx.clamp_(max=num_experts)
                mass.scatter_add_(
                    0, idx, topk_weights.flatten().to(torch.float32))
            if prev_fn is not None:
                prev_fn(topk_ids)

        _capture.wants_topk_weights = True
        return _capture

    # ------------------------------------------------------------------ step

    def step(self, *, is_dummy: bool = False) -> None:
        """Advance one engine step (host side, off the capture path).

        Dummy/profile steps zero the accumulators without recording
        (EPLB semantics) but still advance the counter for rank lockstep.
        """
        self._step += 1
        if is_dummy:
            for lid in self.count_buf:
                self.count_buf[lid].zero_()
                self.mass_buf[lid].zero_()
            return
        if self._step % self.window_stride:
            return
        pos = self._win_pos
        e = self.num_experts
        for lid in self.count_buf:
            # [:e] drops the overflow slot — it never reaches the window.
            self._count_win[lid][pos].copy_(self.count_buf[lid][:e])
            self._mass_win[lid][pos].copy_(self.mass_buf[lid][:e])
            self.count_buf[lid].zero_()
            self.mass_buf[lid].zero_()
        self._win_pos = (pos + 1) % self.window_len
        self._windows_rolled += 1

    # ------------------------------------------------------------------ read

    def mass_is_real(self, layer_id: int | None = None) -> bool:
        """Is ``mass`` REAL gate mass, or is it aliased to ``count``?

        The two are otherwise indistinguishable to a consumer only by
        the arrays happening to be equal, which is a trap: a uniform
        router produces equal arrays legitimately, and an unbound
        weights path produces them by aliasing. Ask here instead.

        Args:
            layer_id: one layer, or None for the model-wide answer
                (True only if EVERY bound layer records real mass; False
                when nothing is bound yet).
        """
        if layer_id is not None:
            return (layer_id in self.count_buf
                    and layer_id not in self._mass_is_count)
        return bool(self.count_buf) and not self._mass_is_count

    def decayed(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decayed window sums (count, mass): w_e = Σ_i λ^i · W[-1-i].

        For layers captured without gate weights, mass is the count
        (graceful degradation — the score becomes pure routing pressure).
        """
        n = min(self._windows_rolled, self.window_len)
        e = self.num_experts
        if n == 0:
            z = torch.zeros(e, dtype=torch.float64, device=self.device)
            return z, z.clone()
        idx = torch.arange(n, device=self.device)
        slots = (self._win_pos - 1 - idx) % self.window_len
        lam = torch.pow(
            torch.tensor(self.decay, dtype=torch.float64,
                         device=self.device), idx)
        cw = self._count_win[layer_id][slots].to(torch.float64)
        c = (lam[:, None] * cw).sum(0)
        if layer_id in self._mass_is_count:
            return c, c.clone()
        mw = self._mass_win[layer_id][slots].to(torch.float64)
        return c, (lam[:, None] * mw).sum(0)

    def summary(self) -> dict:
        """Small host-side dict for persistence (few MB model-wide)."""
        out: dict = {
            "step": self._step,
            "windows_rolled": self._windows_rolled,
            "num_experts": self.num_experts,
            "decay": self.decay,
            # False => every "mass" below is a copy of "count", not gate
            # mass. Explicit so a consumer never has to infer it from the
            # arrays being equal.
            "mass_is_real": self.mass_is_real(),
            "layers": {},
        }
        for lid in sorted(self.count_buf):
            c, m = self.decayed(lid)
            out["layers"][lid] = {
                "count": c.cpu().tolist(),
                "mass": m.cpu().tolist(),
                "mass_is_real": self.mass_is_real(lid),
            }
        return out
