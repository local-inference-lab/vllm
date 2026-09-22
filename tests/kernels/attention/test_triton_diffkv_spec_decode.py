# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short multi-token DiffKV attention against an independent PyTorch reference."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_case(
    q_lens, kv_lens, heads_kv=1, window=None, sinks=False, capacity=64, layout="HND"
):
    torch.manual_seed(71)
    device = "cuda"
    heads, hq, hv, block = 16, 192, 128, 16
    columns = (max(kv_lens) + block - 1) // block + 1
    blocks = columns * len(kv_lens)
    table = torch.randperm(blocks, device=device).reshape(len(kv_lens), columns).int()
    packed = (
        torch.randn(blocks, heads_kv, block, hq + hv, device=device) * 0.3
    ).bfloat16()
    if layout == "NHD":
        packed = packed.transpose(1, 2).contiguous().transpose(1, 2)
    q = torch.randn(sum(q_lens), heads, hq, device=device, dtype=torch.bfloat16)
    k, v = packed.transpose(1, 2).split([hq, hv], dim=-1)
    out = torch.empty(sum(q_lens), heads, hv, device=device, dtype=torch.bfloat16)
    partial = torch.full((capacity, heads, 16, hv), float("nan"), device=device)
    maximum = torch.full((capacity, heads, 16), float("nan"), device=device)
    expsum = torch.full_like(maximum, float("nan"))
    sink = torch.linspace(-1, 2, heads, device=device) if sinks else None
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    starts = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    args = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=starts,
        seqused_k=lengths,
        softmax_scale=hq**-0.5,
        causal=True,
        window_size=(-1, -1) if window is None else (window - 1, 0),
        block_table=table,
        softcap=0,
        sinks=sink,
        max_seqlen_q=max(q_lens),
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=partial,
        softmax_segm_max=maximum,
        softmax_segm_expsum=expsum,
    )
    return SimpleNamespace(**locals())


def reference(c):
    results, start = [], 0
    for i, q_len in enumerate(c.q_lens):
        length = int(c.lengths[i])
        positions = torch.arange(length, device="cuda")
        pages = c.table[i, positions // c.block].long()
        k = (
            c.k[pages, positions % c.block]
            .float()
            .repeat_interleave(c.heads // c.heads_kv, 1)
        )
        v = (
            c.v[pages, positions % c.block]
            .float()
            .repeat_interleave(c.heads // c.heads_kv, 1)
        )
        scores = (
            torch.einsum("qhd,khd->hqk", c.q[start : start + q_len].float(), k)
            * c.hq**-0.5
        )
        distance = (length - q_len + torch.arange(q_len, device="cuda"))[
            :, None
        ] - positions
        allowed = distance >= 0
        if c.window is not None:
            allowed &= distance < c.window
        scores.masked_fill_(~allowed, -torch.inf)
        if c.sink is not None:
            scores = torch.cat([scores, c.sink[:, None, None].expand(-1, q_len, 1)], -1)
        probs = scores.softmax(-1)[..., :length]
        results.append(torch.einsum("hqk,khd->qhd", probs, v))
        start += q_len
    return torch.cat(results).bfloat16()


@pytest.mark.parametrize(
    "q_lens,kv_lens", [([8], [2111]), ([1, 3, 8], [19, 319, 2111])]
)
@pytest.mark.parametrize("heads_kv", [1, 2])
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@torch.inference_mode()
def test_short_queries_use_split_kv(q_lens, kv_lens, heads_kv, window, sinks, layout):
    c = make_case(q_lens, kv_lens, heads_kv, window, sinks, layout=layout)
    unified_attention_diffkv(**c.args)
    torch.testing.assert_close(c.out, reference(c), atol=0.02, rtol=0.02)
    # A real split launch writes intermediate results for every query token.
    assert torch.isfinite(c.expsum[: sum(q_lens), :, 0]).all()


@pytest.mark.parametrize("capacity", [1, 4, 8])
@torch.inference_mode()
def test_small_scratch_falls_back_without_writes(capacity):
    c = make_case([8, 8], [319, 2111], capacity=capacity)
    unified_attention_diffkv(**c.args)
    torch.testing.assert_close(c.out, reference(c), atol=0.02, rtol=0.02)
    assert torch.isnan(c.partial).all()
    assert torch.isnan(c.maximum).all()
    assert torch.isnan(c.expsum).all()


@pytest.mark.parametrize("window", [None, 128])
@torch.inference_mode()
def test_short_query_graph_replay_changes_pages(window):
    c = make_case([8, 3, 1], [319, 511, 2111], window=window, sinks=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            unified_attention_diffkv(**c.args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unified_attention_diffkv(**c.args)
    for step in range(3):
        c.q.normal_()
        c.table.copy_(c.table.roll(1, dims=1))
        c.lengths.copy_(
            torch.tensor([300 + step, 490 + step, 2090 + step], device="cuda")
        )
        graph.replay()
        torch.testing.assert_close(c.out, reference(c), atol=0.02, rtol=0.02)
