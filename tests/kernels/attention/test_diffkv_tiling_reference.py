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
@pytest.mark.parametrize("context", [0, 2111])
@pytest.mark.parametrize("block", [16, 64])
@torch.inference_mode()
def test_bf16_prefill_and_multiquery_geometry(monkeypatch, query_lens, context, block):
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
    monkeypatch.setattr(ops, "kernel_unified_attention_diffkv", recorder)
    ops.unified_attention_diffkv(
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
    expected = reference(q, k, v, table, query_lens, kv_lens, block)
    torch.testing.assert_close(output[:tokens].float(), expected, atol=0.02, rtol=0.02)
    relative = (output[:tokens].float() - expected).norm(dim=-1) / expected.norm(
        dim=-1
    ).clamp_min(1e-6)
    assert relative.max().item() < 0.02
    assert (output[tokens:] == -17).all()
    assert len(recorder.calls) == 1
    _, launch = recorder.calls[0]
    if launch["IS_3D"]:
        # Integration-only on multi-query input. This never forces old dispatch.
        assert launch["NUM_SEGMENTS_PER_SEQ"] == 16
        assert torch.isfinite(expsum[:, :, 0]).all()
        if current_platform.is_device_capability(120, q.device.index or 0):
            assert (launch["BLOCK_M"], launch["TILE_SIZE"]) == (32, 64)
    else:
        expected_tile = (
            (32, 64)
            if (
                tokens >= 512
                and max(query_lens) >= 512
                and not ops.is_batch_invariant
                and current_platform.is_device_capability(120, q.device.index or 0)
            )
            else (16, 32)
        )
        assert (launch["BLOCK_M"], launch["TILE_SIZE"]) == expected_tile
        assert torch.isnan(partial).all()
        assert torch.isnan(maximum).all()
        assert torch.isnan(expsum).all()
