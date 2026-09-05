# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row partition and gather order of the query-sharded encoder attention.

The GPU equivalence of the sharded attention against the unsharded kernel is
checked by research/tp9-colocated-qsrt-20260905/vit_query_shard_check.py on a
nine-GPU tensor-parallel group; these tests pin the CPU-side plan.
"""

import pytest
import torch

from vllm.model_executor.models.vit_query_shard import (
    QUERY_TILE_ROWS,
    build_query_shard_plan,
    partition_rows,
)


@pytest.mark.parametrize("world_size", [1, 2, 9])
@pytest.mark.parametrize(
    "seqlens",
    [[40000], [16384], [1000], [130], [64], [4096, 100, 900], [256, 256, 256, 256]],
)
def test_partition_covers_every_row_once_with_aligned_boundaries(
    seqlens: list[int], world_size: int
) -> None:
    ranges = partition_rows(seqlens, world_size)
    assert len(ranges) == world_size
    covered = torch.zeros(sum(seqlens), dtype=torch.int64)
    for rank in range(world_size):
        assert len(ranges[rank]) == len(seqlens)
        for image, (row_start, row_stop) in enumerate(ranges[rank]):
            image_start = sum(seqlens[:image])
            image_stop = image_start + seqlens[image]
            assert image_start <= row_start <= row_stop <= image_stop
            # Boundaries inside an image sit on query-tile multiples.
            if row_start < image_stop:
                assert (row_start - image_start) % QUERY_TILE_ROWS == 0
            if row_stop < image_stop:
                assert (row_stop - image_start) % QUERY_TILE_ROWS == 0
            covered[row_start:row_stop] += 1
    assert torch.all(covered == 1)


def test_partition_balances_tiles_across_ranks() -> None:
    ranges = partition_rows([40000], 9)
    tiles = [
        (stop - start + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS
        for (start, stop) in (r[0] for r in ranges)
    ]
    assert sum(tiles) == (40000 + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS
    assert max(tiles) - min(tiles) <= 1


@pytest.mark.parametrize("world_size", [2, 9])
@pytest.mark.parametrize("seqlens", [[40000], [130], [4096, 100, 900]])
def test_plan_gathers_back_into_packed_order(
    seqlens: list[int], world_size: int
) -> None:
    total = sum(seqlens)
    rows = torch.arange(total)
    gathered_parts = []
    plans = []
    for rank in range(world_size):
        plan = build_query_shard_plan(
            seqlens, rank=rank, world_size=world_size, device=torch.device("cpu")
        )
        plans.append(plan)
        local = plan.select_queries(rows)
        assert local.shape[0] == plan.num_local_rows == plan.sizes[rank]
        # One query segment per sequence (empty where the rank has no rows),
        # so segment b pairs with key segment b of the packed sequences.
        assert plan.cu_seqlens_q.numel() == len(seqlens) + 1
        assert plan.cu_seqlens_q[-1].item() == plan.num_local_rows
        assert plan.max_seqlen_q == max(
            (plan.cu_seqlens_q[1:] - plan.cu_seqlens_q[:-1]).tolist(), default=0
        )
        gathered_parts.append(local)
    gathered = torch.cat(gathered_parts)
    for plan in plans:
        restored = (
            gathered
            if plan.gather_index is None
            else gathered.index_select(0, plan.gather_index)
        )
        assert torch.equal(restored, rows)
    assert all(p.max_seqlen_k == max(seqlens) for p in plans)
    if len(seqlens) == 1:
        # One image: every rank owns one contiguous range, no reordering.
        assert all(p.gather_index is None and p.local_rows is None for p in plans)


def test_plan_handles_ranks_without_rows() -> None:
    plans = [
        build_query_shard_plan(
            [64], rank=rank, world_size=9, device=torch.device("cpu")
        )
        for rank in range(9)
    ]
    assert plans[0].sizes == (64, 0, 0, 0, 0, 0, 0, 0, 0)
    assert plans[1].num_local_rows == 0
    assert plans[1].max_seqlen_q == 0
    assert plans[1].cu_seqlens_q.tolist() == [0, 0]
    assert plans[1].select_queries(torch.arange(64)).shape[0] == 0
