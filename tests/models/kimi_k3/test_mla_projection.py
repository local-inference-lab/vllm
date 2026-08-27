# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.mla import (
    _restore_merged_output_order,
    _shard_qkv_a_projection,
)


def test_restore_merged_output_order() -> None:
    rank_major = torch.tensor(
        [
            [0, 1, 100, 101, 102],
            [2, 3, 103, 104, 105],
            [4, 5, 106, 107, 108],
            [6, 7, 109, 110, 111],
        ]
    ).flatten()

    output = _restore_merged_output_order(rank_major, [8, 12], tp_size=4)

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ]
        ),
    )


def test_restore_merged_output_order_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="Unexpected gathered merged projection"):
        _restore_merged_output_order(torch.zeros(19), [8, 12], tp_size=4)


@pytest.mark.parametrize(
    ("additional_config", "tp_size", "expected"),
    [
        ({}, 16, False),
        ({"kimi_shard_qkv_a": True}, 1, False),
        ({"kimi_shard_qkv_a": True}, 16, True),
    ],
)
def test_shard_qkv_a_projection_selection(
    additional_config: dict[str, bool],
    tp_size: int,
    expected: bool,
) -> None:
    config = SimpleNamespace(additional_config=additional_config)

    assert _shard_qkv_a_projection(config, [1536, 576], tp_size) is expected


def test_shard_qkv_a_projection_rejects_fractional_shards() -> None:
    config = SimpleNamespace(additional_config={"kimi_shard_qkv_a": True})

    with pytest.raises(ValueError, match="divisible by TP=10"):
        _shard_qkv_a_projection(config, [1536, 576], tp_size=10)
