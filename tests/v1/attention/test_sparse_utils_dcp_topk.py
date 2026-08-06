# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_dcp_local_topk_to_global,
    triton_gather_topk_ids_by_position,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DCP top-k helpers require CUDA"
)


def test_convert_dcp_local_topk_to_global() -> None:
    token_indices = torch.tensor(
        [[0, 1, 3, 4, 7, 8, 15, -1]], dtype=torch.int32, device="cuda"
    )
    scores = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 8)

    triton_convert_dcp_local_topk_to_global(
        token_indices,
        scores,
        dcp_world_size=3,
        dcp_rank=1,
        cp_kv_cache_interleave_size=4,
        BLOCK_N=8,
    )

    expected = torch.tensor(
        [[4, 5, 7, 16, 19, 28, 43, -1]], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(token_indices, expected)
    torch.testing.assert_close(
        scores[:, :-1],
        torch.arange(7, dtype=torch.float32, device="cuda").reshape(1, 7),
    )
    assert torch.isneginf(scores[0, -1])


def test_gather_topk_ids_by_position_handles_invalid_positions() -> None:
    candidate_ids = torch.tensor(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
        dtype=torch.int32,
        device="cuda",
    )
    positions = torch.tensor(
        [[5, 0, -1, 6], [1, 4, 3, 0]], dtype=torch.int32, device="cuda"
    )
    out = torch.empty_like(positions)

    triton_gather_topk_ids_by_position(
        candidate_ids,
        positions,
        out,
        BLOCK_N=4,
    )

    expected = torch.tensor(
        [[15, 10, -1, -1], [21, 24, 23, 20]], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(out, expected)
