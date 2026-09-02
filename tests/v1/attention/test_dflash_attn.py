# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV draft attention against FlashAttention 2 at the DFlash draft shape."""

import os

import pytest
import torch

H, HKV, D, BLOCK, WINDOW = 8, 2, 128, 256, 2048
SCALE = D**-0.5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 split-KV draft attention",
)


def _module():
    from vllm.v1.attention.backends import dflash_attn

    if not dflash_attn.is_available(torch.device("cuda", 0)):
        pytest.skip("dflash_attn library could not be built or loaded")
    return dflash_attn


def _case(seqlens, qlens, strided, device):
    batch = len(seqlens)
    nblk = [(length + BLOCK - 1) // BLOCK for length in seqlens]
    total_blocks = sum(nblk) + 2
    if strided:  # backend layout: [num_blocks, hkv, block, 2D] -> transposed views
        kv = torch.randn(
            total_blocks, HKV, BLOCK, 2 * D, device=device, dtype=torch.bfloat16
        )
        k, v = kv.transpose(1, 2).split(D, dim=-1)
    else:
        k = torch.randn(
            total_blocks, BLOCK, HKV, D, device=device, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
    block_table = torch.zeros(batch, max(nblk), dtype=torch.int32, device=device)
    nxt = 1
    for b in range(batch):
        block_table[b, : nblk[b]] = torch.arange(
            nxt, nxt + nblk[b], dtype=torch.int32, device=device
        )
        nxt += nblk[b]
    tokens = sum(qlens)
    q = torch.randn(tokens, H, D, device=device, dtype=torch.bfloat16)
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(qlens), 0).tolist(),
        dtype=torch.int32,
        device=device,
    )
    seqused = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return q, k, v, block_table, cu, seqused


CASES = [
    ([100], [8]),
    ([2047], [8]),
    ([2048], [8]),
    ([2056], [8]),
    ([3000], [8]),
    ([9000], [8]),
    ([50, 2500, 4096, 7], [8, 8, 8, 8]),
    ([300, 6000], [5, 3]),
    ([8], [8]),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("strided", [True, False])
@pytest.mark.parametrize(("seqlens", "qlens"), CASES)
def test_split_kv_matches_flash_attn(seqlens, qlens, strided, causal):
    dfa = _module()
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = torch.device("cuda", 0)
    torch.manual_seed(1234 + sum(seqlens))
    q, k, v, block_table, cu, seqused = _case(seqlens, qlens, strided, device)
    window = (WINDOW - 1, 0) if causal else (WINDOW - 1, WINDOW - 1)
    ref = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        max_seqlen_q=max(qlens),
        seqused_k=seqused,
        max_seqlen_k=max(seqlens),
        softmax_scale=SCALE,
        causal=causal,
        window_size=window,
        block_table=block_table,
        fa_version=2,
    )
    op = dfa.DFlashDecodeAttention(device, HKV, max_batch=len(seqlens), window=WINDOW)
    out = torch.empty_like(q)
    op(q, k, v, block_table, seqused, cu, SCALE, out, causal=causal)
    torch.cuda.synchronize()
    assert not torch.isnan(out).any()
    err = (out.float() - ref.float()).abs().max().item()
    rel = err / max(1e-6, ref.float().abs().max().item())
    assert rel < 2e-2, f"relative error {rel:.3e} vs FlashAttention 2"
    again = torch.empty_like(q)
    op(q, k, v, block_table, seqused, cu, SCALE, again, causal=causal)
    torch.cuda.synchronize()
    assert torch.equal(out, again), "split-KV attention must be deterministic"


def test_graph_replay_matches_eager():
    dfa = _module()
    device = torch.device("cuda", 0)
    torch.manual_seed(7)
    q, k, v, block_table, cu, seqused = _case([2048, 777], [8, 8], True, device)
    op = dfa.DFlashDecodeAttention(device, HKV, max_batch=2, window=WINDOW)
    eager = torch.empty_like(q)
    op(q, k, v, block_table, seqused, cu, SCALE, eager, causal=False)
    torch.cuda.synchronize()
    captured = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(q, k, v, block_table, seqused, cu, SCALE, captured, causal=False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(q, k, v, block_table, seqused, cu, SCALE, captured, causal=False)
    for _ in range(3):
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)


def test_workspace_rejects_oversized_batch():
    dfa = _module()
    device = torch.device("cuda", 0)
    q, k, v, block_table, cu, seqused = _case([64, 64], [8, 8], True, device)
    op = dfa.DFlashDecodeAttention(device, HKV, max_batch=1, window=WINDOW)
    with pytest.raises(ValueError):
        op(q, k, v, block_table, seqused, cu, SCALE, torch.empty_like(q))
    assert os.getenv("VLLM_GLM53_DFLASH_ATTN", "0") in ("0", "1")
