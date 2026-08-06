# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_dcp_global_topk_indices_and_lens,
    compute_global_topk_indices_and_lens,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


def _inputs() -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    topk_indices = torch.tensor(
        [[0, 1, 2, -1], [0, 1, 2, 3]], dtype=torch.int32, device=device
    )
    # The second row is graph padding. Its stale request index must never be
    # used to address the block table.
    token_to_req_indices = torch.tensor([0, 1 << 29], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
        device=device,
    )
    is_valid_token = torch.tensor([True, False], device=device)
    return topk_indices, token_to_req_indices, block_table, is_valid_token


def test_global_topk_ignores_stale_padding_request_index() -> None:
    topk_indices, token_to_req_indices, block_table, is_valid_token = _inputs()

    indices, lengths = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size=2,
        is_valid_token=is_valid_token,
    )
    torch.cuda.synchronize()

    # Padding-row indices are unspecified; the zero length makes them inert.
    assert indices.cpu().tolist()[0] == [20, 21, 22, -1]
    assert lengths.cpu().tolist() == [3, 0]


def test_dcp_global_topk_ignores_stale_padding_request_index() -> None:
    topk_indices, token_to_req_indices, block_table, is_valid_token = _inputs()

    indices, lengths = compute_dcp_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size=2,
        is_valid_token=is_valid_token,
        dcp_world_size=2,
        dcp_rank=0,
        cp_kv_cache_interleave_size=1,
    )
    torch.cuda.synchronize()

    assert indices.cpu().tolist() == [[20, 21, -1, -1], [-1, -1, -1, -1]]
    assert lengths.cpu().tolist() == [2, 0]
