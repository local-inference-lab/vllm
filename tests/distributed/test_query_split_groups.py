# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.parallel_state import _get_query_split_group_ranks


@pytest.mark.parametrize(
    ("tp_groups", "dcp_size", "expected"),
    [
        (
            [list(range(8))],
            4,
            [[0, 4], [1, 5], [2, 6], [3, 7]],
        ),
        (
            [list(range(8))],
            2,
            [[0, 2, 4, 6], [1, 3, 5, 7]],
        ),
        (
            [list(range(8)), list(range(8, 16))],
            4,
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [8, 12],
                [9, 13],
                [10, 14],
                [11, 15],
            ],
        ),
        (
            [list(range(8))],
            8,
            [[0], [1], [2], [3], [4], [5], [6], [7]],
        ),
    ],
)
def test_get_query_split_group_ranks(tp_groups, dcp_size, expected):
    assert _get_query_split_group_ranks(tp_groups, dcp_size) == expected


@pytest.mark.parametrize("dcp_size", [0, -1])
def test_get_query_split_group_ranks_rejects_nonpositive_dcp(dcp_size):
    with pytest.raises(ValueError, match="must be positive"):
        _get_query_split_group_ranks([[0, 1]], dcp_size)


def test_get_query_split_group_ranks_rejects_partial_dcp_group():
    with pytest.raises(ValueError, match="must be divisible"):
        _get_query_split_group_ranks([[0, 1, 2]], 2)
