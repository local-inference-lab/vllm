# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 index-K side cache: top-k block selection must agree with bf16.

The MiniMax M3 lightning-indexer scores only feed a top-k block ranking (the
attention itself reads the main KV cache), so an fp8 (e4m3) side cache trades
~2% relative score error for half the per-step index-K read bandwidth -- the
dominant linear-in-context decode cost at long context. This test checks that
the decode top-k selection from an fp8 cache matches the bf16 reference,
including strongly-matching planted blocks.
"""
import pytest
import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_fp8_index_cache_topk_matches_bf16() -> None:
    torch.manual_seed(3)
    dev = "cuda"
    page, head_dim, heads, topk = SPARSE_BLOCK_SIZE, 128, 1, 16
    num_pages = 100
    seq_len = 96 * page + 37

    keys = torch.randn(num_pages * page, head_dim, dtype=torch.bfloat16, device=dev)
    # RMS-normalize like the real (post index_k_norm) keys.
    keys = keys * keys.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(
        torch.bfloat16
    )
    q = torch.randn(1, heads, head_dim, dtype=torch.bfloat16, device=dev)
    planted = (7, 40, 88)
    for b in planted:
        keys[b * page + 5] = (q[0, 0] * 3).to(torch.bfloat16)

    block_table = torch.arange(num_pages, dtype=torch.int32, device=dev).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        idx = minimax_m3_index_decode(
            q,
            cache.view(num_pages, page, head_dim).contiguous(),
            block_table,
            seq_lens,
            seq_len,
            topk,
            0,
            0,
            heads,
            1,
            1,
        )
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    top_bf16 = select(keys)
    top_fp8 = select(keys.to(torch.float8_e4m3fn))

    # Planted (clearly relevant) blocks must never be lost.
    assert all(b in top_fp8 for b in planted)
    # Selection may differ only at the noise floor (near-tied filler blocks).
    assert len(top_bf16 & top_fp8) >= topk - 2
