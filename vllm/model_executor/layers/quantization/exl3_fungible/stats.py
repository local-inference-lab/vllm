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
"""
from __future__ import annotations

from collections.abc import Callable

import torch


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
    """

    def __init__(
        self,
        num_experts: int,
        *,
        window_len: int = 64,
        window_stride: int = 32,
        decay: float = 0.95,
        device: torch.device | str = "cuda",
    ) -> None:
        self.num_experts = num_experts
        self.window_len = window_len
        self.window_stride = window_stride
        self.decay = decay
        self.device = torch.device(device)
        # layer_id -> persistent accumulators (allocated at bind time)
        self.count_buf: dict[int, torch.Tensor] = {}
        self.mass_buf: dict[int, torch.Tensor] = {}
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

    def bind_router(self, layer_id: int, router) -> None:
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
        """
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
            self.make_capture_fn(layer_id, prev_fn=prev))

    def make_capture_fn(
        self,
        layer_id: int,
        prev_fn: Callable | None = None,
        topk_weights_getter: Callable[[], torch.Tensor] | None = None,
    ) -> Callable:
        """Build the graph-safe capture fn for one layer.

        The returned callable only touches pre-allocated buffers with
        tensor ops, and is sized for the M1 hot-path gate: without a
        weights getter it is a single ``histc`` histogram per layer
        (~3 small kernels: cast, histc, add) instead of a guarded
        ``scatter_add_`` chain (~10 kernels — measured 4-5% decode
        overhead on the Fruit proxy, PERFORMANCE.md gate is <0.5%).

        Out-of-range ids (padding sentinels, garbage in replayed capture
        batches) MUST never index device memory: an OOR ``scatter_add_``
        is an illegal memory access that kills the engine (observed live
        on the first M2 dryrun boot). ``histc`` drops values outside
        [0, num_experts) by construction; the weighted path keeps the
        explicit overflow-slot redirect.

        ``topk_weights_getter`` is optional plumbing for gate mass (the
        base hook only passes topk_ids); when absent, mass is aliased to
        count at read time (``decayed``/``summary``), so the signal
        degrades gracefully rather than silently vanishing.
        """
        count = self.count_buf[layer_id]
        mass = self.mass_buf[layer_id]
        num_experts = self.num_experts

        if topk_weights_getter is None:
            self._mass_is_count.add(layer_id)

            def _capture(topk_ids: torch.Tensor) -> None:
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

            return _capture

        self._mass_is_count.discard(layer_id)

        def _capture(topk_ids: torch.Tensor) -> None:
            flat = topk_ids.flatten().to(torch.int64)
            oob = (flat < 0) | (flat >= num_experts)
            idx = torch.where(oob, torch.full_like(flat, num_experts), flat)
            count.scatter_add_(
                0, idx, torch.ones_like(flat, dtype=count.dtype))
            w = topk_weights_getter()
            mass.scatter_add_(0, idx, w.flatten().to(torch.float32))
            if prev_fn is not None:
                prev_fn(topk_ids)

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
            "layers": {},
        }
        for lid in sorted(self.count_buf):
            c, m = self.decayed(lid)
            out["layers"][lid] = {
                "count": c.cpu().tolist(),
                "mass": m.cpu().tolist(),
            }
        return out
