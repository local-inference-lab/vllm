# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import (
    _build_indexer_replica_group_ranks,
    _needs_indexer_replica_groups,
    _validate_indexer_shard_count,
)


def test_build_indexer_two_by_four_groups_for_tp8():
    dcp_groups, query_split_groups = _build_indexer_replica_group_ranks(
        [[0, 1, 2, 3, 4, 5, 6, 7]], 4
    )

    assert dcp_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert query_split_groups == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.parametrize(
    ("indexer_shards", "expected_dcp", "expected_query_split"),
    [
        (
            1,
            [[0], [1], [2], [3], [4], [5], [6], [7]],
            [list(range(8))],
        ),
        (
            2,
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[0, 2, 4, 6], [1, 3, 5, 7]],
        ),
        (
            8,
            [list(range(8))],
            [[0], [1], [2], [3], [4], [5], [6], [7]],
        ),
    ],
)
def test_build_indexer_groups_cover_dcp1_partial_and_full(
    indexer_shards,
    expected_dcp,
    expected_query_split,
):
    dcp_groups, query_split_groups = _build_indexer_replica_group_ranks(
        [list(range(8))], indexer_shards
    )

    assert dcp_groups == expected_dcp
    assert query_split_groups == expected_query_split


@pytest.mark.parametrize(
    ("indexer_shards", "expected"),
    [(0, False), (1, True), (2, True), (4, True), (8, False)],
)
def test_indexer_replica_group_gate_for_dcp8(indexer_shards, expected):
    assert _needs_indexer_replica_groups(indexer_shards, 8) is expected


def test_build_indexer_replica_groups_stay_inside_each_tp_group():
    dcp_groups, query_split_groups = _build_indexer_replica_group_ranks(
        [list(range(8)), list(range(8, 16))], 4
    )

    assert dcp_groups == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert query_split_groups == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [8, 12],
        [9, 13],
        [10, 14],
        [11, 15],
    ]


def test_build_indexer_replica_groups_rejects_non_divisor():
    with pytest.raises(ValueError, match="must divide"):
        _build_indexer_replica_group_ranks([list(range(8))], 3)


@pytest.mark.parametrize("indexer_shards", [0, 1, 2, 4])
def test_validate_indexer_shard_count_accepts_supported_dcp4_layouts(indexer_shards):
    _validate_indexer_shard_count(indexer_shards, 4)


@pytest.mark.parametrize("indexer_shards", [-1, 3, 5, 8])
def test_validate_indexer_shard_count_rejects_invalid_dcp4_layouts(indexer_shards):
    with pytest.raises(ValueError, match=r"shards=.*DCP=4"):
        _validate_indexer_shard_count(indexer_shards, 4)


def test_indexer_group_selector_supports_partial_target_and_full_draft(monkeypatch):
    partial_dcp = SimpleNamespace(world_size=2)
    full_dcp = SimpleNamespace(world_size=4)
    partial_query_split = SimpleNamespace(world_size=4)
    full_query_split = SimpleNamespace(world_size=2)
    monkeypatch.setattr(parallel_state, "_INDEXER_DCP", partial_dcp)
    monkeypatch.setattr(parallel_state, "_DCP", full_dcp)
    monkeypatch.setattr(parallel_state, "_INDEXER_QUERY_SPLIT", partial_query_split)
    monkeypatch.setattr(parallel_state, "_QUERY_SPLIT", full_query_split)

    assert parallel_state.get_indexer_dcp_group(2) is partial_dcp
    assert parallel_state.get_indexer_dcp_group(4) is full_dcp
    assert parallel_state.get_indexer_query_split_group(2) is partial_query_split
    assert parallel_state.get_indexer_query_split_group(4) is full_query_split


def test_indexer_group_selector_uses_tp_query_split_for_dcp1(monkeypatch):
    dcp = SimpleNamespace(world_size=1)
    query_split = SimpleNamespace(world_size=8)
    monkeypatch.setattr(parallel_state, "_INDEXER_DCP", None)
    monkeypatch.setattr(parallel_state, "_INDEXER_QUERY_SPLIT", None)
    monkeypatch.setattr(parallel_state, "_DCP", dcp)
    monkeypatch.setattr(parallel_state, "_QUERY_SPLIT", query_split)

    assert parallel_state.get_indexer_dcp_group(1) is dcp
    assert parallel_state.get_indexer_query_split_group(1) is query_split


def test_indexer_group_selector_rejects_unknown_shard_count(monkeypatch):
    monkeypatch.setattr(parallel_state, "_INDEXER_DCP", SimpleNamespace(world_size=2))
    monkeypatch.setattr(parallel_state, "_DCP", SimpleNamespace(world_size=4))

    with pytest.raises(RuntimeError, match="requested=3, partial=2, configured=4"):
        parallel_state.get_indexer_dcp_group(3)
    with pytest.raises(RuntimeError, match="requested=3, partial=2, configured=4"):
        parallel_state.get_indexer_query_split_group(3)
