# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity-planned GDN prefill over the vLLM recurrent-state pool."""

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.b12x import get_b12x_gdn_prefill, get_b12x_scratch_buffers
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource


@triton.jit
def _stage_metadata(
    query_start_loc,
    state_indices,
    has_initial_state,
    checkpoint_indices,
    checkpoint_offsets,
    live_counts,
    out_query_start_loc,
    out_initial_indices,
    out_final_indices,
    out_checkpoint_indices,
    out_checkpoint_offsets,
    out_num_seqs,
    out_num_tokens,
    state_stride: tl.constexpr,
    HAS_CHECKPOINT: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    num_seqs = tl.load(live_counts)
    num_tokens = tl.load(live_counts + 1)
    live = row < num_seqs
    boundary = tl.load(query_start_loc + row, row <= num_seqs, other=0)
    boundary = tl.where(row <= num_seqs, boundary, num_tokens)
    tl.store(out_query_start_loc + row, boundary, row <= MAX_SEQS)
    slot = tl.load(state_indices + row.to(tl.int64) * state_stride, live, other=0)
    initial = tl.load(has_initial_state + row, live, other=False)
    tl.store(out_initial_indices + row, tl.where(initial, slot, 0), row < MAX_SEQS)
    tl.store(out_final_indices + row, slot, row < MAX_SEQS)
    checkpoint = tl.full((BLOCK,), 0, tl.int32)
    offset = tl.full((BLOCK,), 0, tl.int32)
    if HAS_CHECKPOINT:
        checkpoint = tl.load(checkpoint_indices + row, live, other=0)
        offset = tl.load(checkpoint_offsets + row, live, other=0)
    tl.store(out_checkpoint_indices + row, checkpoint, row < MAX_SEQS)
    tl.store(out_checkpoint_offsets + row, offset, row < MAX_SEQS)
    tl.store(out_num_seqs, num_seqs)
    tl.store(out_num_tokens, num_tokens)


def prefill_capacities(max_tokens: int) -> tuple[int, ...]:
    if max_tokens < 1:
        raise ValueError("GDN prefill token capacity must be positive")
    capacities = []
    capacity = 16
    while capacity < max_tokens:
        capacities.append(capacity)
        capacity *= 2
    capacities.append(max_tokens)
    return tuple(capacities)


class B12xGdnPrefill:
    """Precompiled capacity family with fixed buffers and caller-owned state."""

    def __init__(
        self,
        *,
        recurrent_state: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        max_tokens: int,
        max_seqs: int,
        key_heads: int,
        value_heads: int,
        checkpoint_export: bool,
    ) -> None:
        api = get_b12x_gdn_prefill()
        if api is None:
            raise RuntimeError("b12x GDN prefill requires b12x.sequence.gdn_prefill")
        self.api = api
        self.max_seqs = max_seqs
        self.max_tokens = max_tokens
        self.key_heads = key_heads
        self.value_heads = value_heads
        device = recurrent_state.device
        self.capacities = prefill_capacities(max_tokens)
        self.plans = tuple(
            api.plan(
                api.Caps(
                    device=device,
                    max_tokens=capacity,
                    max_seqs=max_seqs,
                    max_state_slots=recurrent_state.shape[0],
                    key_heads=key_heads,
                    value_heads=value_heads,
                    state_dtype=recurrent_state.dtype,
                    checkpoint_export=checkpoint_export,
                    null_state_index=0,
                )
            )
            for capacity in self.capacities
        )
        largest = max(self.plans, key=lambda plan: plan.scratch_specs()[0].shape[0])
        (self.scratch,) = get_b12x_scratch_buffers(largest)
        self.mixed_qkv = torch.empty(
            (max_tokens, (2 * key_heads + value_heads) * 128),
            dtype=torch.bfloat16,
            device=device,
        )
        self.a = torch.empty(
            (max_tokens, value_heads), dtype=torch.bfloat16, device=device
        )
        self.b = torch.empty_like(self.a)
        self.output = torch.empty(
            (max_tokens, value_heads, 128), dtype=torch.bfloat16, device=device
        )
        self.query_start_loc = torch.zeros(
            max_seqs + 1, dtype=torch.int32, device=device
        )
        self.initial_indices = torch.zeros(max_seqs, dtype=torch.int32, device=device)
        self.final_indices = torch.zeros_like(self.initial_indices)
        self.checkpoint_indices = torch.zeros_like(self.initial_indices)
        self.checkpoint_offsets = torch.zeros_like(self.initial_indices)
        self.num_seqs = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_tokens = torch.zeros_like(self.num_seqs)
        q, k, v = self.mixed_qkv.split(
            (key_heads * 128, key_heads * 128, value_heads * 128), dim=-1
        )
        self.bindings = tuple(
            api.bind(
                plan,
                scratch=self.scratch[: plan.scratch_specs()[0].shape[0]],
                q=q[:capacity].view(capacity, key_heads, 128),
                k=k[:capacity].view(capacity, key_heads, 128),
                v=v[:capacity].view(capacity, value_heads, 128),
                a=self.a[:capacity],
                b=self.b[:capacity],
                A_log=A_log,
                dt_bias=dt_bias,
                recurrent_state=recurrent_state,
                cu_seqlens=self.query_start_loc,
                initial_state_indices=self.initial_indices,
                final_state_indices=self.final_indices,
                checkpoint_state_indices=self.checkpoint_indices,
                checkpoint_offsets=self.checkpoint_offsets,
                num_seqs=self.num_seqs,
                num_tokens=self.num_tokens,
                output=self.output[:capacity],
            )
            for capacity, plan in zip(self.capacities, self.plans)
        )
        for binding in self.bindings:
            api.prewarm(binding)

    def run(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        live_counts: torch.Tensor,
        output: torch.Tensor,
        checkpoint: Any = None,
        scale: float = 128**-0.5,
        eps: float = 1e-6,
    ) -> None:
        rows = mixed_qkv.shape[0]
        if rows > self.max_tokens:
            raise ValueError(
                f"GDN prefill rows {rows} exceed planned capacity {self.max_tokens}"
            )
        if live_counts.shape != (2,) or live_counts.dtype != torch.int32:
            raise ValueError(
                "GDN prefill requires int32 device [num_seqs, num_tokens] counts"
            )
        if live_counts.device != self.num_tokens.device:
            raise ValueError("GDN prefill counts must reside on the plan device")
        retain_cuda_graph_capture_resource(self)
        index = next(
            i for i, capacity in enumerate(self.capacities) if rows <= capacity
        )
        binding = self.bindings[index]
        self.mixed_qkv[:rows].copy_(mixed_qkv)
        self.a[:rows].copy_(a)
        self.b[:rows].copy_(b)
        _stage_metadata[(1,)](
            query_start_loc,
            state_indices,
            has_initial_state,
            state_indices if checkpoint is None else checkpoint.state_indices,
            query_start_loc if checkpoint is None else checkpoint.checkpoint_offsets,
            live_counts,
            self.query_start_loc,
            self.initial_indices,
            self.final_indices,
            self.checkpoint_indices,
            self.checkpoint_offsets,
            self.num_seqs,
            self.num_tokens,
            state_indices.stride(0),
            HAS_CHECKPOINT=checkpoint is not None,
            MAX_SEQS=self.max_seqs,
            BLOCK=triton.next_power_of_2(self.max_seqs + 1),
        )
        self.api.run(
            binding,
            scale=scale,
            eps=eps,
            max_live_tokens=self.capacities[index],
            max_live_seqs=self.max_seqs,
        )
        output[:rows].copy_(self.output[:rows])
