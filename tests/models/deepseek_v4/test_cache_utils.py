# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    combine_topk_swa_indices,
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


def test_combine_topk_swa_indices_matches_reference_across_worker_tiles() -> None:
    """Sparse and sliding-window metadata cover every query token exactly."""
    device = torch.device("cuda")
    query_start = torch.tensor([0, 129, 300], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1024, 2048], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([512, 1024], dtype=torch.int32, device=device)
    topk = 8
    window_size = 8
    compress_ratio = 4
    req_stride = 4096
    swa_offset = 2048
    topk_indices = torch.arange(
        300 * topk,
        dtype=torch.int32,
        device=device,
    ).reshape(300, topk)

    actual_indices, actual_lens = combine_topk_swa_indices(
        topk_indices,
        query_start,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        req_stride,
        swa_offset,
    )

    expected_indices = torch.full_like(actual_indices, -1)
    expected_lens = torch.empty_like(actual_lens)
    query_start_cpu = query_start.cpu().tolist()
    for req_idx, (start, end) in enumerate(
        zip(query_start_cpu[:-1], query_start_cpu[1:], strict=True)
    ):
        query_len = end - start
        seq_len = int(seq_lens[req_idx].item())
        gather_start = seq_len - int(gather_lens[req_idx].item())
        start_pos = seq_len - query_len
        for token_idx in range(start, end):
            pos = start_pos + token_idx - start
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            expected_indices[token_idx, :topk_len] = (
                topk_indices[token_idx, :topk_len] + req_stride * req_idx
            )
            expected_indices[token_idx, topk_len : topk_len + swa_len] = (
                torch.arange(
                    swa_offset + pos - swa_len + 1 - gather_start,
                    swa_offset + pos + 1 - gather_start,
                    dtype=torch.int32,
                    device=device,
                )
                + req_stride * req_idx
            )
            expected_lens[token_idx] = topk_len + swa_len

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_lens, expected_lens)
