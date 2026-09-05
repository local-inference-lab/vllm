# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-sharded self-attention for vision towers replicated across TP ranks.

When a vision tower's head count does not divide the tensor-parallel size,
vLLM replicates the tower on every rank and shards a batch by image
(``run_dp_sharded_mrope_vision_model``). A request with one large image then
runs the whole encoder on a single rank while the other ranks wait for the
gather; for a 40,000-patch image on the Kimi-K3 TP9 target that is 1.1 s of
which 0.9 s is the quadratic self-attention.

A :class:`QueryShardPlan` keeps the patch embedding, norms, projections and
MLPs replicated (every rank computes the identical full tensors) and splits
only the self-attention by query rows: rank ``r`` computes attention for its
share of each image's 128-row query tiles against the image's complete keys
and values, the shares are all-gathered in the tensor-parallel group and put
back in row order. FlashAttention computes every query row independently
(its own dot products, running maximum, running sum and rescaling, over the
key blocks of its sequence in a fixed order), so a row's output does not
depend on which other rows share its block or which GPU runs it: the
gathered output equals the unsharded computation bit for bit. Shard
boundaries are aligned to the kernel's ``tile``-row query blocks, counted
from each sequence's start, so the launch geometry of every shard matches
the unsharded run and no block is split across ranks.

The plan is activated for the duration of one encoder forward with
:func:`activate_query_shard_plan`; the encoder layers look it up with
:func:`active_query_shard_plan`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gatherv,
)

QUERY_TILE_ROWS = 128
"""Query rows per FlashAttention forward block for 128-wide heads."""

_active = threading.local()


@dataclass(frozen=True)
class QueryShardPlan:
    """Row assignment of one encoder forward across the tensor-parallel group.

    ``local_rows`` are the rows (in packed image order) this rank computes,
    ``cu_seqlens_q`` the per-image query boundaries of those rows,
    ``sizes`` the row count of every rank (for the all-gather), and
    ``gather_index`` maps each packed row to its position in the rank-major
    gathered tensor (``None`` when that order already is the packed order).
    """

    rank: int
    world_size: int
    total_rows: int
    local_start: int
    local_stop: int
    local_rows: torch.Tensor | None
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    sizes: tuple[int, ...]
    gather_index: torch.Tensor | None

    @property
    def num_local_rows(self) -> int:
        return self.sizes[self.rank]

    def select_queries(self, query: torch.Tensor) -> torch.Tensor:
        """Rows of ``query`` (packed, row-major) that this rank attends for."""
        if self.local_rows is None:
            return query.narrow(0, self.local_start, self.local_stop - self.local_start)
        return query.index_select(0, self.local_rows)

    def gather(self, local_output: torch.Tensor) -> torch.Tensor:
        """All-gather the per-rank attention outputs back into packed row order."""
        gathered = tensor_model_parallel_all_gatherv(
            local_output, dim=0, sizes=list(self.sizes)
        )
        if self.gather_index is None:
            return gathered
        return gathered.index_select(0, self.gather_index)


def partition_rows(
    seqlens: Sequence[int], world_size: int, tile: int = QUERY_TILE_ROWS
) -> list[list[tuple[int, int]]]:
    """Row ranges ``[start, stop)`` of every rank for every packed sequence.

    Each sequence's rows are cut into ``tile``-aligned query blocks that are
    dealt to ranks as evenly as possible (the first ``tiles % world_size``
    ranks get one extra block). The result is indexed ``[rank][sequence]``;
    a rank without rows in a sequence gets an empty range at the sequence's
    end.
    """
    ranges: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
    start = 0
    for length in seqlens:
        tiles = (length + tile - 1) // tile
        base, extra = divmod(tiles, world_size)
        first_tile = 0
        for rank in range(world_size):
            count = base + (1 if rank < extra else 0)
            row_start = start + min(first_tile * tile, length)
            row_stop = start + min((first_tile + count) * tile, length)
            ranges[rank].append((row_start, row_stop))
            first_tile += count
        start += length
    return ranges


def build_query_shard_plan(
    seqlens: Sequence[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    tile: int = QUERY_TILE_ROWS,
) -> QueryShardPlan:
    """Plan the query split of packed sequences of lengths ``seqlens``."""
    ranges = partition_rows(seqlens, world_size, tile)
    sizes = tuple(
        sum(stop - start for start, stop in ranges[r]) for r in range(world_size)
    )
    total_rows = sum(seqlens)

    # Position of every packed row inside the rank-major gathered tensor.
    packed_to_gathered = torch.empty(total_rows, dtype=torch.int64)
    position = 0
    contiguous = True
    for r in range(world_size):
        for start, stop in ranges[r]:
            if stop > start:
                packed_to_gathered[start:stop] = torch.arange(
                    position, position + stop - start
                )
                position += stop - start
    if total_rows and not torch.equal(packed_to_gathered, torch.arange(total_rows)):
        contiguous = False

    my_ranges = [(start, stop) for start, stop in ranges[rank] if stop > start]
    if my_ranges:
        local_start, local_stop = my_ranges[0][0], my_ranges[-1][1]
    else:
        local_start = local_stop = 0
    local_is_slice = (
        sum(stop - start for start, stop in my_ranges) == local_stop - local_start
    )
    local_rows = None
    if not local_is_slice:
        local_rows = torch.cat(
            [torch.arange(start, stop) for start, stop in my_ranges]
        ).to(device=device)

    # One query segment per packed sequence, empty where this rank has no
    # rows, so segment ``b`` of the local queries pairs with key segment
    # ``b`` of the full ``cu_seqlens_k``.
    local_lengths = [stop - start for start, stop in ranges[rank]]
    cu_seqlens_q = torch.zeros(len(local_lengths) + 1, dtype=torch.int32)
    if local_lengths:
        cu_seqlens_q[1:] = torch.cumsum(
            torch.tensor(local_lengths, dtype=torch.int32), 0
        )
    return QueryShardPlan(
        rank=rank,
        world_size=world_size,
        total_rows=total_rows,
        local_start=local_start,
        local_stop=local_stop,
        local_rows=local_rows,
        cu_seqlens_q=cu_seqlens_q.to(device=device),
        max_seqlen_q=max(local_lengths, default=0),
        max_seqlen_k=max(seqlens, default=0),
        sizes=sizes,
        gather_index=None if contiguous else packed_to_gathered.to(device=device),
    )


def query_shard_enabled(total_patches: int) -> bool:
    """Whether the replicated encoder should split attention for this batch."""
    return (
        envs.VLLM_VIT_QUERY_SHARD
        and get_tensor_model_parallel_world_size() > 1
        and total_patches >= envs.VLLM_VIT_QUERY_SHARD_MIN_PATCHES
    )


def maybe_build_query_shard_plan(
    seqlens: Sequence[int], device: torch.device
) -> QueryShardPlan | None:
    """A plan for the current tensor-parallel group, or ``None`` when disabled."""
    if not query_shard_enabled(sum(seqlens)):
        return None
    return build_query_shard_plan(
        seqlens,
        rank=get_tensor_model_parallel_rank(),
        world_size=get_tensor_model_parallel_world_size(),
        device=device,
    )


@contextmanager
def activate_query_shard_plan(plan: QueryShardPlan | None) -> Iterator[None]:
    """Make ``plan`` visible to encoder layers on this thread during the block."""
    previous = getattr(_active, "plan", None)
    _active.plan = plan
    try:
        yield
    finally:
        _active.plan = previous


def active_query_shard_plan() -> QueryShardPlan | None:
    return getattr(_active, "plan", None)


def sharded_varlen_attention(
    plan: QueryShardPlan,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    fa_version: int | None,
) -> torch.Tensor:
    """Non-causal varlen attention over ``(rows, heads, dim)``, split by query rows.

    ``key``/``value`` hold every packed row; ``cu_seqlens_k`` are the packed
    sequence boundaries. Returns the full ``(rows, heads, dim)`` output in
    packed order on every rank.
    """
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    local_query = plan.select_queries(query)
    if local_query.shape[0] > 0:
        kwargs = {} if fa_version is None else {"fa_version": fa_version}
        local_output = flash_attn_varlen_func(
            local_query,
            key,
            value,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=plan.max_seqlen_q,
            max_seqlen_k=plan.max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            **kwargs,
        )
    else:
        local_output = query.new_empty((0, *query.shape[1:]))
    return plan.gather(local_output)
