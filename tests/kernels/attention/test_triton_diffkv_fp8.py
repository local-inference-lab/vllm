# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed DiffKV write/read tests with a PyTorch reference, including SM120."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVImpl,
)

pytestmark = pytest.mark.skip_global_cleanup


def _case(cache_dtype, path, num_kv_heads, window, sinks, layout):
    device = "cuda"
    torch.manual_seed(31)
    q_lens = [1, 3, 17] if path == "mixed" else [1, 1, 1]
    kv_lens = [17, 129, 259]
    block_size, heads, hq, hv = 16, 16, 192, 128
    # The serving block table is padded to the configured context capacity.
    columns = 18
    blocks = columns * len(kv_lens)
    table = torch.randperm(blocks, device=device).view(3, columns).int()
    dtype = torch.uint8 if cache_dtype.startswith("fp8") else torch.bfloat16
    if layout == "HND":
        cache = torch.zeros(
            blocks, num_kv_heads, block_size, hq + hv, device=device, dtype=dtype
        )
    else:
        cache = torch.zeros(
            blocks, block_size, num_kv_heads, hq + hv, device=device, dtype=dtype
        ).transpose(1, 2)
    query = torch.randn(sum(q_lens), heads, hq, device=device, dtype=torch.bfloat16)
    # Two trailing source rows have invalid slots and must not enter the cache.
    key = (
        torch.randn(sum(kv_lens) + 2, num_kv_heads, hq, device=device) * 0.25
    ).bfloat16()
    value = (
        torch.randn(sum(kv_lens) + 2, num_kv_heads, hv, device=device) * 0.5
    ).bfloat16()
    slots = torch.cat(
        [
            table[i, torch.arange(length, device=device) // block_size].long()
            * block_size
            + torch.arange(length, device=device) % block_size
            for i, length in enumerate(kv_lens)
        ]
        + [torch.tensor([-1, -1], device=device)]
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor(0.037, device=device),
        _v_scale=torch.tensor(0.061, device=device),
    )
    sink = torch.linspace(-1, 2, heads, device=device) if sinks else None
    impl = TritonAttentionDiffKVImpl(
        num_heads=heads,
        head_size=hq,
        scale=hq**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=window,
        kv_cache_dtype=cache_dtype,
        sinks=sink,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=sum(q_lens),
        use_cascade=False,
        query_start_loc=torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_query_len=max(q_lens),
        block_table=table,
        seq_threshold_3D=8 if path == "decode3d" else 0,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(8, heads, 16, hv, device=device),
        softmax_segm_max=torch.empty(8, heads, 16, device=device),
        softmax_segm_expsum=torch.empty(8, heads, 16, device=device),
    )
    output = torch.empty(sum(q_lens), heads, hv, device=device, dtype=torch.bfloat16)
    return SimpleNamespace(**locals())


def _run(c):
    c.impl.do_kv_cache_update(c.layer, c.key, c.value, c.cache, c.slots)
    return c.impl.forward(
        c.layer, c.query, c.key, c.value, c.cache, c.metadata, c.output
    )


def _reference(c):
    """Independently quantize input, scatter by slot, then use FP32 attention."""
    if c.dtype == torch.uint8:
        fp8 = c.impl.fp8_dtype
        key = (c.key.float() / c.layer._k_scale).to(fp8).float()
        value = (c.value.float() / c.layer._v_scale).to(fp8).float()
    else:
        key, value = c.key.float(), c.value.float()
    packed = torch.zeros(
        c.blocks * c.block_size, c.num_kv_heads, c.hq + c.hv, device=c.device
    )
    valid = c.slots >= 0
    packed[c.slots[valid]] = torch.cat([key[valid], value[valid]], dim=-1)
    expected_cache = packed.view(c.blocks, c.block_size, c.num_kv_heads, -1).transpose(
        1, 2
    )
    actual = (
        c.cache.view(c.impl.fp8_dtype).float()
        if c.dtype == torch.uint8
        else c.cache.float()
    )
    torch.testing.assert_close(actual, expected_cache, rtol=0, atol=0)
    outputs = []
    start = 0
    for i, q_len in enumerate(c.q_lens):
        length = int(c.metadata.seq_lens[i])
        pos = torch.arange(length, device=c.device)
        physical = (
            c.table[i, pos // c.block_size].long() * c.block_size + pos % c.block_size
        )
        kv = packed[physical]
        k, v = kv[..., : c.hq], kv[..., c.hq :]
        if c.dtype == torch.uint8:
            k = (k * c.layer._k_scale).bfloat16().float()
            v = (v * c.layer._v_scale).bfloat16().float()
        groups = c.heads // c.num_kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        q = c.query[start : start + q_len].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * c.hq**-0.5
        delta = (length - q_len + torch.arange(q_len, device=c.device))[:, None] - pos
        allowed = delta >= 0
        if c.window is not None:
            allowed &= delta < c.window
        scores.masked_fill_(~allowed, -torch.inf)
        if c.sink is not None:
            scores = torch.cat([scores, c.sink[:, None, None].expand(-1, q_len, 1)], -1)
        prob = scores.softmax(-1)[..., :length]
        outputs.append(torch.einsum("hqk,khd->qhd", prob, v))
        start += q_len
    return torch.cat(outputs).bfloat16()


@pytest.mark.parametrize("cache_dtype", ["bfloat16", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize("path", ["mixed", "decode2d", "decode3d"])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@torch.inference_mode()
def test_diffkv_torch_reference(cache_dtype, path, num_kv_heads, window, sinks, layout):
    c = _case(cache_dtype, path, num_kv_heads, window, sinks, layout)
    if cache_dtype.startswith("fp8"):
        assert not c.impl.supports_quant_query_input
    _run(c)
    torch.testing.assert_close(c.output, _reference(c), atol=0.02, rtol=0.02)
    logical_elements = c.blocks * c.num_kv_heads * c.block_size * (c.hq + c.hv)
    assert c.cache.untyped_storage().nbytes() == logical_elements * (
        1 if c.dtype == torch.uint8 else 2
    )


@pytest.mark.parametrize("cache_dtype", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("path", ["mixed", "decode3d"])
@torch.inference_mode()
def test_diffkv_graph_replay_reuses_pages(cache_dtype, path):
    c = _case(cache_dtype, path, 2, 128, True, "HND")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(c)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(c)
    for step in range(3):
        c.query.normal_()
        c.key.normal_(std=0.25)
        c.value.normal_(std=0.5)
        # Same captured pointers, changed sequences and physical page ownership.
        c.metadata.seq_lens.copy_(
            torch.tensor([17, 125 + step, 255 + step], device="cuda")
        )
        c.table.copy_(c.table.roll(1, dims=1))
        graph.replay()
        torch.testing.assert_close(c.output, _reference(c), atol=0.02, rtol=0.02)
