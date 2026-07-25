# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.parallel_state import _build_indexer_replica_group_ranks


def test_build_indexer_two_by_four_groups_for_tp8():
    dcp_groups, query_split_groups = _build_indexer_replica_group_ranks(
        [[0, 1, 2, 3, 4, 5, 6, 7]], 4
    )

    assert dcp_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert query_split_groups == [[0, 4], [1, 5], [2, 6], [3, 7]]


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
