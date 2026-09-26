# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel draft sampling must reproduce the full-vocab path exactly.

TP shards are simulated in one process: each shard runs the local kernels and
the per-shard results are stacked where the all-gather would put them.
"""

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs
from vllm.v1.worker.gpu.spec_decode import vocab_parallel as vp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

V, TP, STEPS, MAX_REQS = 16384 * 3, 4, 7, 8
VL = V // TP


def _draft(trial: int, dtype: torch.dtype):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(trial)
    reqs = 6
    slots = torch.tensor([3, 0, 5, 1, 7, 2], device=dev, dtype=torch.int32)
    temps = torch.zeros(MAX_REQS, device=dev)
    temps[slots.long()] = torch.tensor([1.0, 0.7, 1.3, 0.0, 1.0, 0.6], device=dev)
    seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=dev)
    rows = reqs * STEPS + 5  # plus CUDA-graph padding rows
    idx = torch.full((rows,), -1, dtype=torch.int32, device=dev)
    idx[: reqs * STEPS] = slots.repeat_interleave(STEPS)
    cols = torch.arange(STEPS, device=dev, dtype=torch.int32).repeat(rows // STEPS + 1)
    cols = cols[:rows].contiguous()
    pos = torch.randint(10, 100000, (rows,), generator=g, device=dev)
    logits = torch.randn((rows, V), generator=g, device=dev) * 2.5
    logits[:, :64] += 6.0 * torch.rand((rows, 64), generator=g, device=dev)
    logits = logits.to(torch.bfloat16)

    full = torch.zeros((MAX_REQS, STEPS, V), dtype=dtype, device=dev)
    ref = gumbel_sample(
        logits,
        idx,
        temps,
        seeds,
        pos,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=full,
        logits_cache_col=cols,
    )
    caches = [
        vp.VPDraftCache(
            torch.zeros((MAX_REQS, STEPS, VL), dtype=dtype, device=dev), r * VL, V, None
        )
        for r in range(TP)
    ]
    stats = torch.stack(
        [
            vp.draft_local(
                logits[:, r * VL : (r + 1) * VL].contiguous(),
                idx,
                temps,
                seeds,
                pos,
                cols,
                caches[r],
            )
            for r in range(TP)
        ]
    )
    for c in caches:
        tokens, _, _ = vp.draft_combine(stats, idx, cols, c)
    return dict(
        slots=slots,
        temps=temps,
        seeds=seeds,
        idx=idx,
        cols=cols,
        logits=logits,
        full=full,
        ref=ref,
        caches=caches,
        tokens=tokens,
    )


@pytest.mark.parametrize("trial", range(4))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vocab_parallel_draft_matches_full_vocab(trial, dtype):
    d = _draft(trial, dtype)
    valid = d["idx"] >= 0
    assert torch.equal(d["tokens"][valid], d["ref"][valid])
    for r, c in enumerate(d["caches"]):
        assert torch.equal(c.logits, d["full"][:, :, r * VL : (r + 1) * VL])
    t = d["temps"][d["idx"][valid].long()]
    t = torch.where(t > 0, t, torch.ones_like(t)).double()
    lse_ref = torch.logsumexp(d["logits"][valid].double() / t[:, None], dim=-1)
    rows = (d["idx"][valid].long(), d["cols"][valid].long())
    torch.testing.assert_close(
        d["caches"][0].lse[rows].double(), lse_ref, atol=1e-5, rtol=0
    )
    tok_logit = (
        d["logits"][valid].float().gather(-1, d["ref"][valid][:, None]).squeeze(-1)
    )
    assert torch.equal(d["caches"][0].token_logit[rows], tok_logit)


@pytest.mark.parametrize("trial", range(4))
def test_vocab_parallel_rejection_matches_full_vocab(trial, monkeypatch):
    d = _draft(100 + trial, torch.float32)
    dev = d["logits"].device
    g = torch.Generator(device=dev).manual_seed(trial)
    reqs = d["slots"].numel()
    depths = [7, 3, 0, 5, 7, 2]
    draft_tok = d["tokens"][: reqs * STEPS].view(reqs, STEPS)
    cu = [0]
    for n in depths:
        cu.append(cu[-1] + n + 1)
    num_logits = cu[-1]
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=dev)
    exp_idx = torch.zeros(num_logits, dtype=torch.int32, device=dev)
    exp_local = torch.zeros(num_logits, dtype=torch.int32, device=dev)
    pos = torch.zeros(num_logits, dtype=torch.int64, device=dev)
    target = torch.randn((num_logits, V), generator=g, device=dev) * 2.5
    for q, n in enumerate(depths):
        s0 = cu[q]
        draft_sampled[s0 + 1 : s0 + 1 + n] = draft_tok[q, :n]
        exp_idx[s0 : s0 + n + 1] = d["slots"][q]
        exp_local[s0 : s0 + n + 1] = torch.arange(n + 1, dtype=torch.int32, device=dev)
        pos[s0 : s0 + n + 1] = torch.arange(n + 1, device=dev) + 5000 + 97 * q
        for i in range(n + 1):
            src = q * STEPS + min(i, STEPS - 1)
            target[s0 + i] = 0.7 * d["logits"][src].float() + 0.3 * target[s0 + i]
    cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
    args = (
        draft_sampled,
        cu_t,
        pos,
        d["slots"],
        exp_idx,
        exp_local,
        d["temps"],
        d["seeds"],
        STEPS,
    )
    ref_sampled, ref_n = rs.rejection_sample(target, d["full"], *args)

    caches = d["caches"]

    def resample(
        target_logits,
        lse,
        step,
        cu_num,
        eidx,
        dsamp,
        temp,
        seed,
        p,
        cache,
        use_fp64=False,
    ):
        stats = torch.stack(
            [
                vp.resample_local(
                    target_logits,
                    lse,
                    step,
                    cu_num,
                    eidx,
                    dsamp,
                    temp,
                    seed,
                    p,
                    c,
                    use_fp64,
                )
                for c in caches
            ]
        )
        return vp.resample_combine(stats, use_fp64)

    monkeypatch.setattr(vp, "resample", resample)
    for c in caches:
        sampled, n = rs.rejection_sample(target, c.logits, *args)
        assert torch.equal(n, ref_n)
        for q in range(reqs):
            k = int(ref_n[q])
            assert torch.equal(sampled[q, :k], ref_sampled[q, :k])
