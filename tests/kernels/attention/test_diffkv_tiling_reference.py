# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 wrapper geometries against PyTorch, independent of FlashAttention.

On the standalone lab base, multi-query requests remain 2D: PR839 is not
included here. The same kwargs allow integration source to select 3D. Record
actual dispatch rather than claiming standalone multi-query split-KV coverage.
Run only in the separately authorized GPU validation stage.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import triton_unified_attention_diffkv as ops

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def fp32_oracle_without_tf32():
    """Keep the eager FP32 oracle independent of TF32 matmul precision."""
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32


class ForwardRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))
            return self.kernel[grid](**kwargs)

        return launch


def reference(q, k, v, table, query_lens, kv_lens, block):
    outputs, offset = [], 0
    for seq, (qlen, length) in enumerate(zip(query_lens, kv_lens)):
        positions = torch.arange(length, device=q.device)
        pages = table[seq, positions // block].long()
        keys = k[pages, positions % block, 0].float()
        values = v[pages, positions % block, 0].float()
        query = q[offset : offset + qlen].float()
        scores = torch.einsum("qhd,kd->hqk", query, keys) * 192**-0.5
        query_positions = length - qlen + torch.arange(qlen, device=q.device)
        causal = positions[None, :] <= query_positions[:, None]
        scores.masked_fill_(~causal[None, :, :], float("-inf"))
        outputs.append(torch.einsum("hqk,kd->qhd", scores.softmax(-1), values))
        offset += qlen
    return torch.cat(outputs)


@pytest.mark.parametrize(
    "query_lens",
    [[8], [8] * 8, [8] * 16, [1, 3, 8, 17], [64], [128], [256], [512], [2048]],
)
@pytest.mark.parametrize("context", [0, 2111, 2887])
@pytest.mark.parametrize("block", [16, 64])
@torch.inference_mode()
def test_bf16_prefill_and_multiquery_geometry(
    monkeypatch, record_property, query_lens, context, block
):
    if not current_platform.is_cuda():
        pytest.skip("CUDA numerical test")
    torch.manual_seed(71)
    tokens, sequences = sum(query_lens), len(query_lens)
    kv_lens = [max(q, context - i * 7) for i, q in enumerate(query_lens)]
    # Aligned table coverage isolates launch tiling from the separate PR854 mask.
    coverage = ((max(kv_lens) + 63) // 64) * 64
    columns = coverage // block
    blocks = sequences * columns
    table = torch.randperm(blocks, device="cuda").reshape(sequences, columns).int()
    packed = (torch.randn(blocks, block, 1, 320, device="cuda") * 0.3).bfloat16()
    k, v = packed[..., :192], packed[..., 192:]
    q = torch.randn(tokens, 16, 192, dtype=torch.bfloat16, device="cuda")
    # Poison token padding: accidental writes past valid queries are observable.
    output = torch.full(
        (tokens + 3, 16, 128), -17.0, dtype=torch.bfloat16, device="cuda"
    )
    partial = torch.full((tokens, 16, 16, 128), float("nan"), device="cuda")
    maximum = torch.full((tokens, 16, 16), float("nan"), device="cuda")
    expsum = torch.full_like(maximum, float("nan"))
    starts = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    recorder = ForwardRecorder(ops.kernel_unified_attention_diffkv)
    reducer = ForwardRecorder(ops.kernel_reduce_segments_diffkv)
    monkeypatch.setattr(ops, "kernel_unified_attention_diffkv", recorder)
    monkeypatch.setattr(ops, "kernel_reduce_segments_diffkv", reducer)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=output[:tokens],
        cu_seqlens_q=starts,
        seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        softmax_scale=192**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        max_seqlen_q=max(query_lens),
        seq_threshold_3D=tokens if max(query_lens) < 512 else 0,
        num_par_softmax_segments=16,
        softmax_segm_output=partial,
        softmax_segm_max=maximum,
        softmax_segm_expsum=expsum,
    )
    ops.unified_attention_diffkv(**kwargs)
    expected = reference(q, k, v, table, query_lens, kv_lens, block)
    torch.testing.assert_close(output[:tokens].float(), expected, atol=0.02, rtol=0.02)
    relative = (output[:tokens].float() - expected).norm(dim=-1) / expected.norm(
        dim=-1
    ).clamp_min(1e-6)
    assert relative.max().item() < 0.02
    assert (output[tokens:] == -17).all()
    assert len(recorder.calls) == 1
    _, launch = recorder.calls[0]
    eligible = (
        launch["IS_3D"]
        and tokens >= 128
        and max(query_lens) > 1
        and not ops.is_batch_invariant
        and current_platform.is_device_capability(120, q.device.index or 0)
    )
    record_property("diffkv_dispatch", "3d" if launch["IS_3D"] else "2d")
    record_property("integrated_query_reuse_exercised", bool(eligible))
    if launch["IS_3D"]:
        assert launch["NUM_SEGMENTS_PER_SEQ"] == 16
        assert torch.isfinite(expsum[:, :, 0]).all()
        assert (launch["BLOCK_M"], launch["TILE_SIZE"]) == (32 if eligible else 16, 16)
        assert len(reducer.calls) == 1
    else:
        assert (launch["BLOCK_M"], launch["TILE_SIZE"]) == (16, 32)
        assert torch.isnan(partial).all()
        assert torch.isnan(maximum).all()
        assert torch.isnan(expsum).all()
        assert not reducer.calls

    # Reuse the exact input tensors and actual dispatch. Only select the old
    # M16 geometry for the second real GPU launch; scratch/output are distinct.
    # Standalone multi-query dispatch stays 2D and still executes this equality
    # check. Eligible M32-vs-M16 coverage is recorded only on integrated 3D.
    baseline_output = torch.full_like(output, -17)
    baseline_kwargs = dict(
        kwargs,
        out=baseline_output[:tokens],
        softmax_segm_output=torch.full_like(partial, float("nan")),
        softmax_segm_max=torch.full_like(maximum, float("nan")),
        softmax_segm_expsum=torch.full_like(expsum, float("nan")),
    )

    def original_tiling(*args, **controls):
        return 16, 16 if controls["use_3d"] else 32

    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(ops, "_select_diffkv_tiling", original_tiling)
        ops.unified_attention_diffkv(**baseline_kwargs)
    assert len(recorder.calls) == 2
    baseline_launch = recorder.calls[1][1]
    assert baseline_launch["IS_3D"] == launch["IS_3D"]
    assert baseline_launch["BLOCK_M"] == 16
    assert baseline_launch["TILE_SIZE"] == launch["TILE_SIZE"]
    assert baseline_launch["NUM_SEGMENTS_PER_SEQ"] == launch["NUM_SEGMENTS_PER_SEQ"]
    for pointer in (
        "query_ptr",
        "key_cache_ptr",
        "value_cache_ptr",
        "block_tables_ptr",
        "seq_lens_ptr",
        "query_start_len_ptr",
    ):
        assert baseline_launch[pointer] is launch[pointer]
    if launch["IS_3D"]:
        assert len(reducer.calls) == 2
        assert reducer.calls[1][0] == reducer.calls[0][0] == (tokens, 16)
        assert reducer.calls[1][1]["BLOCK_Q"] == 1
        assert reducer.calls[1][1]["TILE_SIZE"] == 16
        assert reducer.calls[1][1]["NUM_SEGMENTS_PER_SEQ"] == 16
        assert torch.isfinite(baseline_kwargs["softmax_segm_expsum"][:, :, 0]).all()
        for actual, name in (
            (partial, "softmax_segm_output"),
            (maximum, "softmax_segm_max"),
            (expsum, "softmax_segm_expsum"),
        ):
            torch.testing.assert_close(
                actual, baseline_kwargs[name], rtol=0, atol=0, equal_nan=True
            )
    else:
        assert not reducer.calls
    assert torch.equal(output[:tokens], baseline_output[:tokens])
    assert (baseline_output[tokens:] == -17).all()
