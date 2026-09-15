# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity-planned GDN prefill over the vLLM recurrent-state pool."""

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GdnPrefillStaging:
    """Reusable capacity buffers independent of a recurrent-state generation."""

    max_tokens: int
    max_seqs: int
    key_heads: int
    value_heads: int
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    output: torch.Tensor
    query_start_loc: torch.Tensor
    initial_indices: torch.Tensor
    final_indices: torch.Tensor
    checkpoint_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens: int,
        max_seqs: int,
        key_heads: int,
        value_heads: int,
        device: torch.device,
    ) -> "GdnPrefillStaging":
        mixed_qkv = torch.empty(
            (max_tokens, (2 * key_heads + value_heads) * 128),
            dtype=torch.bfloat16,
            device=device,
        )
        a = torch.empty((max_tokens, value_heads), dtype=torch.bfloat16, device=device)
        initial_indices = torch.zeros(max_seqs, dtype=torch.int32, device=device)
        return cls(
            max_tokens=max_tokens, max_seqs=max_seqs,
            key_heads=key_heads, value_heads=value_heads,
            mixed_qkv=mixed_qkv, a=a, b=torch.empty_like(a),
            output=torch.empty(
                (max_tokens, value_heads, 128), dtype=torch.bfloat16, device=device,
            ),
            query_start_loc=torch.zeros(max_seqs + 1, dtype=torch.int32, device=device),
            initial_indices=initial_indices,
            final_indices=torch.zeros_like(initial_indices),
            checkpoint_indices=torch.zeros_like(initial_indices),
            checkpoint_offsets=torch.zeros_like(initial_indices),
            num_seqs=torch.zeros(1, dtype=torch.int32, device=device),
            num_tokens=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def is_compatible(
        self, *, max_tokens: int, max_seqs: int, key_heads: int, value_heads: int,
        device: torch.device,
    ) -> bool:
        return (
            self.max_tokens >= max_tokens
            and self.max_seqs >= max_seqs
            and self.key_heads == key_heads
            and self.value_heads == value_heads
            and self.mixed_qkv.device == device
        )

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (
            self.mixed_qkv, self.a, self.b, self.output, self.query_start_loc,
            self.initial_indices, self.final_indices, self.checkpoint_indices,
            self.checkpoint_offsets, self.num_seqs, self.num_tokens,
        ))


class B12xGdnPrefill:
    """Layer-held GDN capacity family with reusable staging buffers.

    The layer supplies one declared ``Plan`` for every admitted capacity.
    This helper owns only capacity staging buffers; the recurrent-state pool
    remains a binding-time caller resource and can be invalidated without
    discarding the plans or these buffers.
    """

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
        plans: Mapping[int, object],
        staging: GdnPrefillStaging,
    ) -> None:
        del checkpoint_export
        api = get_b12x_gdn_prefill()
        if api is None:
            raise RuntimeError("b12x GDN prefill requires b12x.sequence.gdn_prefill")
        self.api = api
        self.max_seqs, self.max_tokens = max_seqs, max_tokens
        self.key_heads, self.value_heads = key_heads, value_heads
        self.capacities = prefill_capacities(max_tokens)
        if set(plans) != set(self.capacities):
            raise ValueError("declared GDN plans must cover exactly every capacity")
        self.plans = dict(plans)
        # The state pool is a generation-bound caller resource.  Capacity
        # staging and executable ownership survive its replacement, but every
        # invocation must bind the currently published pool.
        self.recurrent_state = recurrent_state
        self.A_log = A_log
        self.dt_bias = dt_bias
        device = recurrent_state.device
        if staging is None:
            raise RuntimeError(
                "GDN prefill staging must be materialized during preparation"
            )
        if not staging.is_compatible(
            max_tokens=max_tokens, max_seqs=max_seqs, key_heads=key_heads,
            value_heads=value_heads, device=device,
        ):
            raise ValueError("GDN staging buffers do not cover the prepared capacity")
        self.staging = staging
        self.mixed_qkv = staging.mixed_qkv
        self.a = staging.a
        self.b = staging.b
        self.output = staging.output
        self.query_start_loc = staging.query_start_loc
        self.initial_indices = staging.initial_indices
        self.final_indices = staging.final_indices
        self.checkpoint_indices = staging.checkpoint_indices
        self.checkpoint_offsets = staging.checkpoint_offsets
        self.num_seqs = staging.num_seqs
        self.num_tokens = staging.num_tokens
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
        capacity = next(capacity for capacity in self.capacities if rows <= capacity)
        plan = self.plans[capacity]
        (scratch,) = get_b12x_scratch_buffers(plan)
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
        q, k, v = self.mixed_qkv[:capacity].split(
            (self.key_heads * 128, self.key_heads * 128, self.value_heads * 128),
            dim=-1,
        )
        binding = self.api.bind(
            plan,
            scratch=scratch,
            q=q.view(capacity, self.key_heads, 128),
            k=k.view(capacity, self.key_heads, 128),
            v=v.view(capacity, self.value_heads, 128),
            a=self.a[:capacity],
            b=self.b[:capacity],
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            recurrent_state=self.recurrent_state,
            cu_seqlens=self.query_start_loc,
            initial_state_indices=self.initial_indices,
            final_state_indices=self.final_indices,
            checkpoint_state_indices=self.checkpoint_indices,
            checkpoint_offsets=self.checkpoint_offsets,
            num_seqs=self.num_seqs,
            num_tokens=self.num_tokens,
            output=self.output[:capacity],
        )
        self.api.run(
            binding,
            scale=scale,
            eps=eps,
            max_live_tokens=capacity,
            max_live_seqs=self.max_seqs,
        )
        output[:rows].copy_(self.output[:rows])
