# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-kernel BF16 regressions for DiffKV page-table and graph-padding bounds.

Normal pytest uses the installed vLLM module. Executing this file runs the same
pytest tests and adds --diffkv-source-module for before/after source selection.
No launch geometry is patched, and no FP8 or FlashAttention reference is used.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skip_global_cleanup

SENTINEL = 123.0


@pytest.fixture
def launch(request):
    import vllm.v1.attention.ops.triton_attention_helpers as helpers
    import vllm.v1.attention.ops.triton_unified_attention_diffkv as installed

    path = request.config.getoption("--diffkv-source-module", default=None)
    module = installed
    if path is not None:
        path = Path(path).resolve()
        helper_path = path.with_name("triton_attention_helpers.py")
        selected_helper = helper_path.read_bytes()
        installed_helper = Path(inspect.getfile(helpers)).read_bytes()
        assert selected_helper == installed_helper, (
            "Selected source and installed attention helpers must match"
        )
        spec = importlib.util.spec_from_file_location("diffkv_bounds_source", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    assert not module.is_batch_invariant, "These tests require split-KV dispatch"
    assert torch.cuda.is_available(), "These are GPU tests, not CPU substitutes"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield module.unified_attention_diffkv
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def make_case(q_lens, kv_lens, *, extra_tokens=0, split=False, sinks=False):
    torch.manual_seed(83)
    heads, qk_dim, v_dim, block = 16, 192, 128, 16
    columns = (max(kv_lens) + block - 1) // block
    count = len(q_lens) * columns
    table = torch.arange(count, device="cuda", dtype=torch.int32).reshape(
        len(q_lens), columns
    )
    cache = torch.randn(count, block, 1, qk_dim + v_dim, device="cuda").bfloat16()
    cache.mul_(0.25)
    real_tokens = sum(q_lens)
    capacity = real_tokens + extra_tokens
    q = torch.randn(capacity, heads, qk_dim, device="cuda", dtype=torch.bfloat16)
    out = torch.full(
        (capacity, heads, v_dim), SENTINEL, device="cuda", dtype=torch.bfloat16
    )
    prefix = [0]
    for length in q_lens:
        prefix.append(prefix[-1] + length)
    starts = torch.tensor(prefix, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    sink = torch.linspace(-1, 2, heads, device="cuda") if sinks else None
    args = dict(
        q=q,
        k=cache[..., :qk_dim],
        v=cache[..., qk_dim:],
        out=out,
        cu_seqlens_q=starts,
        seqused_k=lengths,
        softmax_scale=qk_dim**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        max_seqlen_q=max(q_lens),
        sinks=sink,
    )
    if split:
        segments = 16
        # Defined padded partials make the baseline sentinel failure independent
        # of allocator contents. Main attention never writes these padded rows.
        partial = torch.ones(capacity, heads, segments, v_dim, device="cuda")
        maximum = torch.zeros(capacity, heads, segments, device="cuda")
        expsum = torch.ones_like(maximum)
        args.update(
            seq_threshold_3D=max(capacity, len(q_lens)),
            num_par_softmax_segments=segments,
            softmax_segm_output=partial,
            softmax_segm_max=maximum,
            softmax_segm_expsum=expsum,
        )
    return SimpleNamespace(**locals())


def reference(case):
    """Independent small query-only PyTorch attention, including sink mass."""
    outputs = []
    start = 0
    for seq, (q_len, kv_len) in enumerate(zip(case.q_lens, case.kv_lens)):
        if q_len == 0:
            continue
        pos = torch.arange(kv_len, device="cuda")
        pages = case.table[seq, pos // case.block].long()
        kv = case.cache[pages, pos % case.block, 0].float()
        q = case.q[start : start + q_len].float()
        scores = torch.einsum("qhd,kd->qhk", q, kv[:, : case.qk_dim])
        scores *= case.qk_dim**-0.5
        query_pos = kv_len - q_len + torch.arange(q_len, device="cuda")
        scores.masked_fill_(pos[None, None, :] > query_pos[:, None, None], -torch.inf)
        if case.sink is not None:
            scores = torch.cat(
                [scores, case.sink[None, :, None].expand(q_len, -1, 1)], dim=-1
            )
        weights = scores.softmax(-1)[..., :kv_len]
        outputs.append(torch.einsum("qhk,kd->qhd", weights, kv[:, case.qk_dim :]))
        start += q_len
    return torch.cat(outputs)


def check_output(case):
    torch.testing.assert_close(
        case.out[: case.real_tokens].float(), reference(case), atol=0.003, rtol=0.02
    )


@pytest.mark.parametrize(
    "padding_kv_length", [0, 32], ids=["zero_length", "nonzero_length"]
)
@torch.inference_mode()
def test_split_kv_preserves_trailing_padding(launch, padding_kv_length):
    case = make_case(
        [1, 1, 1, 0],
        [17, 33, 65, padding_kv_length],
        extra_tokens=5,
        split=True,
        sinks=True,
    )
    launch(**case.args)
    torch.cuda.synchronize()
    # Check padding first: the nonzero-length baseline case deterministically
    # writes 1 from the defined partials over this sentinel.
    assert torch.all(case.out[case.real_tokens :] == SENTINEL)
    assert torch.all(case.partial[case.real_tokens :] == 1)
    assert not torch.all(case.partial[: case.real_tokens] == 1)
    check_output(case)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch(**case.args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(**case.args)
    for _ in range(2):
        case.q.normal_()
        case.table.copy_(case.table.roll(1, dims=1))
        case.out.fill_(SENTINEL)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(case.out[case.real_tokens :] == SENTINEL)
        check_output(case)


@pytest.mark.parametrize("kv_len,q_len", [(1, 1), (33, 3), (65, 8)])
@torch.inference_mode()
def test_minimal_width_2d_table(launch, kv_len, q_len):
    case = make_case([q_len], [kv_len])
    # The unchanged public wrapper chooses 2D TILE_SIZE=32 without split buffers.
    # Odd page counts with block16 require an extra table column in the old
    # unmasked load. Keep the logical table minimal and at the allocation end.
    # A 512-byte backing is already allocator-aligned; caching must be disabled
    # in the sanitizer process so a larger caching slab cannot hide the overread.
    backing = torch.empty(128, device="cuda", dtype=torch.int32)
    table = backing[-case.columns :].view(1, case.columns)
    table.copy_(case.table)
    case.table = table
    case.args["block_table"] = table
    assert table.shape[1] == (kv_len + case.block - 1) // case.block
    assert table.storage_offset() + table.numel() == backing.numel()
    launch(**case.args)
    torch.cuda.synchronize()
    check_output(case)
    # Numerical agreement alone cannot detect the unused table-lane read.
    # Run this exact test under compute-sanitizer memcheck as documented.


class _SourceOption:
    def pytest_addoption(self, parser):
        parser.addoption("--diffkv-source-module", type=Path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]], plugins=[_SourceOption()]))
