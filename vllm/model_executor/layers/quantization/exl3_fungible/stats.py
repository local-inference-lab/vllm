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
        """
        count = torch.zeros(
            self.num_experts, dtype=torch.int32, device=self.device)
        mass = torch.zeros(
            self.num_experts, dtype=torch.float32, device=self.device)
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
        tensor ops. ``topk_weights_getter`` is optional plumbing for gate
        mass (the base hook only passes topk_ids); when absent, mass
        accumulation adds 1.0 per routing (count-equivalent) so the
        signal degrades gracefully rather than silently vanishing.
        """
        count = self.count_buf[layer_id]
        mass = self.mass_buf[layer_id]

        def _capture(topk_ids: torch.Tensor) -> None:
            flat = topk_ids.flatten().to(torch.int64)
            ones_i = torch.ones_like(flat, dtype=torch.int32)
            count.scatter_add_(0, flat, ones_i)
            if topk_weights_getter is not None:
                w = topk_weights_getter()
                mass.scatter_add_(0, flat, w.flatten().to(torch.float32))
            else:
                mass.scatter_add_(
                    0, flat, torch.ones_like(flat, dtype=torch.float32))
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
        for lid in self.count_buf:
            self._count_win[lid][pos].copy_(self.count_buf[lid])
            self._mass_win[lid][pos].copy_(self.mass_buf[lid])
            self.count_buf[lid].zero_()
            self.mass_buf[lid].zero_()
        self._win_pos = (pos + 1) % self.window_len
        self._windows_rolled += 1

    # ------------------------------------------------------------------ read

    def decayed(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decayed window sums (count, mass): w_e = Σ_i λ^i · W[-1-i]."""
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
        mw = self._mass_win[layer_id][slots].to(torch.float64)
        return (lam[:, None] * cw).sum(0), (lam[:, None] * mw).sum(0)

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
