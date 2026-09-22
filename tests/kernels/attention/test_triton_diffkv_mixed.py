# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed DiffKV attention against a PyTorch reference, without FA3/FA4.

Multi-token split-KV requires the short-query dispatcher correction (#839).
E4M3 is additionally exercised when the runtime advertises that cache mode.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
    TritonAttentionDiffKVMetadataBuilder,
)

pytestmark = pytest.mark.skip_global_cleanup
CACHE_DTYPES = [
    mode
    for mode in ("bfloat16", "fp8_e4m3")
    if mode in TritonAttentionDiffKVBackend.supported_kv_cache_dtypes
]


def make_case(cache_dtype, heads_kv, window, q_lens=(8, 3, 1, 257)):
    torch.manual_seed(72)
    heads, hk, hv, block = 16, 192, 128, 16
    kv_lens = [319, 511, 2111, 2311][: len(q_lens)]
    columns = (max(kv_lens) + block - 1) // block + 1
    table = (
        torch.randperm(columns * len(q_lens), device="cuda").reshape(-1, columns).int()
    )
    packed = (
        torch.randn(columns * len(q_lens), heads_kv, block, hk + hv, device="cuda")
        * 0.3
    ).bfloat16()
    k_scale, v_scale = (
        torch.tensor(0.037, device="cuda"),
        torch.tensor(0.061, device="cuda"),
    )
    if cache_dtype == "fp8_e4m3":
        k, v = packed.split([hk, hv], dim=-1)
        packed = (
            torch.cat((k / k_scale, v / v_scale), dim=-1)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    q = torch.randn(sum(q_lens), heads, hk, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(sum(q_lens), heads, hv, device="cuda", dtype=torch.bfloat16)
    starts_cpu = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts_cpu.to("cuda"),
        query_start_loc_cpu=starts_cpu,
        seq_lens=lengths,
        num_reqs=len(q_lens),
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
        block_table_tensor=table,
        slot_mapping=torch.arange(sum(q_lens), device="cuda"),
    )
    builder = object.__new__(TritonAttentionDiffKVMetadataBuilder)
    builder.seq_threshold_3D, builder.num_par_softmax_segments = 128, 16
    builder.softmax_segm_output = torch.full(
        (128, heads, 16, hv), float("nan"), device="cuda"
    )
    builder.softmax_segm_max = torch.full((128, heads, 16), float("nan"), device="cuda")
    builder.softmax_segm_expsum = torch.full_like(
        builder.softmax_segm_max, float("nan")
    )
    builder.reorder_batch_threshold, builder.rswa_window = 8, None
    builder.device = torch.device("cuda")
    metadata = builder.build(0, common)
    impl = TritonAttentionDiffKVImpl(
        num_heads=heads,
        head_size=hk,
        scale=hk**-0.5,
        num_kv_heads=heads_kv,
        alibi_slopes=None,
        sliding_window=window,
        kv_cache_dtype=cache_dtype,
    )
    sink = torch.linspace(-1, 2, heads, device="cuda")
    impl.sinks = sink
    layer = SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)
    return SimpleNamespace(**locals())


def reference(c):
    packed = c.packed
    if c.cache_dtype == "fp8_e4m3":
        packed = packed.view(torch.float8_e4m3fn)
    k, v = packed.transpose(1, 2).split([c.hk, c.hv], dim=-1)
    k, v = k.float(), v.float()
    if c.cache_dtype == "fp8_e4m3":
        # The reader casts dequantized operands to the BF16 query dtype.
        k = (k * c.k_scale).bfloat16().float()
        v = (v * c.v_scale).bfloat16().float()
    results, start = [], 0
    for i, q_len in enumerate(c.q_lens):
        length = int(c.lengths[i])
        pos = torch.arange(length, device="cuda")
        pages = c.table[i, pos // c.block].long()
        keys = k[pages, pos % c.block].repeat_interleave(c.heads // c.heads_kv, 1)
        values = v[pages, pos % c.block].repeat_interleave(c.heads // c.heads_kv, 1)
        scores = (
            torch.einsum("qhd,khd->hqk", c.q[start : start + q_len].float(), keys)
            * c.hk**-0.5
        )
        distance = (length - q_len + torch.arange(q_len, device="cuda"))[:, None] - pos
        allowed = distance >= 0
        if c.window is not None:
            allowed &= distance < c.window
        scores.masked_fill_(~allowed, -torch.inf)
        scores = torch.cat((scores, c.sink[:, None, None].expand(-1, q_len, 1)), dim=-1)
        probs = scores.softmax(-1)[..., :length]
        results.append(torch.einsum("hqk,khd->qhd", probs, values))
        start += q_len
    return torch.cat(results).bfloat16()


def execute(c):
    c.impl.forward(c.layer, c.q, None, None, c.packed, c.metadata, c.out)


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("heads_kv", [1, 2])
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("q_lens", [(1, 1, 1, 257), (8, 3, 1, 257)])
@torch.inference_mode()
def test_mixed_reference_and_page_reuse(cache_dtype, heads_kv, window, q_lens):
    c = make_case(cache_dtype, heads_kv, window, q_lens=q_lens)
    decode_rows = sum(q_lens[:-1])
    decode_metadata = c.metadata.partitions[0]
    # Compare dispatch with the same short queries in isolation. Single-token
    # decode is always split-KV; multi-token decode gains that path with #839.
    # Probe only inside the test, never during collection or module import.
    c.impl.forward(
        c.layer,
        c.q[:decode_rows],
        None,
        None,
        c.packed,
        decode_metadata,
        c.out[:decode_rows],
    )
    expected_split = bool(
        torch.isfinite(c.builder.softmax_segm_expsum[:decode_rows, :, 0]).all()
    )
    if max(q_lens[:-1]) == 1:
        assert expected_split
    c.builder.softmax_segm_output.fill_(float("nan"))
    c.builder.softmax_segm_max.fill_(float("nan"))
    c.builder.softmax_segm_expsum.fill_(float("nan"))
    for _ in range(2):
        execute(c)
        torch.testing.assert_close(c.out, reference(c), atol=0.02, rtol=0.02)
        assert (
            bool(
                torch.isfinite(c.builder.softmax_segm_expsum[:decode_rows, :, 0]).all()
            )
            == expected_split
        )
        # No prefill row may overwrite the decode workspace.
        assert torch.isnan(c.builder.softmax_segm_expsum[decode_rows:]).all()
        c.q.normal_()
        c.table.copy_(c.table.roll(1, dims=1))
        c.lengths.sub_(1)


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("window", [None, 128])
@torch.inference_mode()
def test_decode_graph_replay_still_uses_persistent_metadata(cache_dtype, window):
    c = make_case(cache_dtype, 1, window, q_lens=(8, 3, 1))
    assert not c.metadata.partitions
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            execute(c)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(c)
    for _ in range(2):
        c.q.normal_()
        c.table.copy_(c.table.roll(1, dims=1))
        c.lengths.sub_(1)
        graph.replay()
        torch.testing.assert_close(c.out, reference(c), atol=0.02, rtol=0.02)
