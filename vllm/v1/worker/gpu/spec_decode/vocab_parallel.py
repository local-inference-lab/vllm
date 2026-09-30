# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel draft sampling for speculative decoding (no logits all-gather).

The drafter's lm_head is vocab-parallel: each TP rank holds logits for its
contiguous vocab shard.  Gathering them (``[rows, vocab]`` per step) was the
largest collective of a C32 step.  Sampling needs far less:

  * Gumbel-max draws key the noise by absolute token id (``gumbel.py``), so
    each rank draws over its shard with the same noise the full-vocab kernel
    uses, and the global argmax of the per-rank winners is the token the
    full-vocab draw returns (ties aside, which have probability zero).
  * Verification needs q(d) for the drawn token d and, on rejection, the
    residual max(p - q, 0) over the vocabulary.  q(d) comes from d's logit
    (known to the winning rank) and the global normalizer (combined from
    per-rank max / sum-exp); both are cached per (request, step).  The
    residual is formed and Gumbel-sampled per shard, then combined the same way.

Every rank ends with identical tokens and statistics.  Pre-temperature draft
logits stay cached per shard for the residual.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

BLOCK = 1024

# draft-logits cache (sharded) data_ptr -> VPDraftCache
_REGISTRY: dict[int, VPDraftCache] = {}


class VPDraftCache:
    """Side caches of a sharded draft-logits cache [reqs, steps, local_vocab]."""

    def __init__(
        self, logits: torch.Tensor, vocab_start: int, vocab_size: int, tp_group
    ):
        reqs, steps, _ = logits.shape
        self.logits = logits
        self.vocab_start = int(vocab_start)
        self.local_vocab = int(logits.shape[-1])
        self.vocab_size = int(vocab_size)
        self.tp_group = tp_group
        dev = logits.device
        # Pre-temperature logit of the drawn token, and logsumexp(logits / T).
        # Row ``reqs`` is a sink for CUDA-graph padding rows (slot -1).
        self.num_reqs = int(reqs)
        self.token_logit = torch.zeros(
            (reqs + 1, steps), dtype=torch.float32, device=dev
        )
        self.lse = torch.zeros((reqs + 1, steps), dtype=torch.float32, device=dev)
        _REGISTRY[logits.data_ptr()] = self


def lookup(draft_logits: torch.Tensor | None) -> VPDraftCache | None:
    if draft_logits is None:
        return None
    return _REGISTRY.get(draft_logits.data_ptr())


@triton.jit
def _vp_draft_block_kernel(
    blk_val_ptr,  # [rows, nblk] fp32/fp64: noisy max
    blk_idx_ptr,  # [rows, nblk] int64: absolute token id
    blk_max_ptr,  # [rows, nblk] fp32: max(logit / T)
    blk_sum_ptr,  # [rows, nblk] fp32: sum exp(logit / T - blk_max)
    nblk,
    cache_ptr,  # [reqs, steps, local_vocab]
    cache_stride_0,
    cache_stride_1,
    col_ptr,  # [rows] step column
    logits_ptr,  # [rows, local_vocab]
    logits_stride,
    idx_mapping_ptr,  # [rows] request slot (-1: padding)
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    local_vocab,
    vocab_start,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < local_vocab
    logits = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")
    )
    logits = logits.to(tl.float32)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    col = tl.load(col_ptr + row).to(tl.int64)
    # Cache the pre-temperature logits (the rejection sampler divides by the
    # same temperature on load, exactly as for the full-vocab cache).
    tl.store(
        cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs,
        logits,
        mask=mask & valid,
    )
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + row)
    keys = vocab_start + offs
    value, idx = gumbel_noised_argmax(
        logits,
        keys,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=True,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=True,
    )
    tl.store(blk_val_ptr + row * nblk + b, value)
    tl.store(
        blk_idx_ptr + row * nblk + b, (vocab_start + b * BLOCK_SIZE + idx).to(tl.int64)
    )
    t = tl.where(temp > 0.0, temp, 1.0)
    scaled = tl.where(mask, logits / t, float("-inf"))
    m = tl.max(scaled, axis=0)
    s = tl.where(m > float("-inf"), tl.sum(tl.exp(scaled - m)), 0.0)
    tl.store(blk_max_ptr + row * nblk + b, m)
    tl.store(blk_sum_ptr + row * nblk + b, s)


def draft_local(
    local_logits: torch.Tensor,  # [rows, local_vocab]
    idx_mapping: torch.Tensor,  # [rows]
    temperature: torch.Tensor,  # [max_reqs]
    seeds: torch.Tensor,  # [max_reqs]
    pos: torch.Tensor,  # [rows]
    col: torch.Tensor,  # [rows]
    cache: VPDraftCache,
    use_fp64: bool = False,
) -> torch.Tensor:
    """This shard's part: caches the shard's logits and returns per-row
    [noisy max, token id, its logit, max(logit/T), sumexp] as fp64 [rows, 5]."""
    rows, local_vocab = local_logits.shape
    nblk = triton.cdiv(local_vocab, BLOCK)
    dev = local_logits.device
    val_dtype = torch.float64 if use_fp64 else torch.float32
    blk_val = torch.empty((rows, nblk), dtype=val_dtype, device=dev)
    blk_idx = torch.empty((rows, nblk), dtype=torch.int64, device=dev)
    blk_max = torch.empty((rows, nblk), dtype=torch.float32, device=dev)
    blk_sum = torch.empty((rows, nblk), dtype=torch.float32, device=dev)
    _vp_draft_block_kernel[(rows, nblk)](
        blk_val,
        blk_idx,
        blk_max,
        blk_sum,
        nblk,
        cache.logits,
        cache.logits.stride(0),
        cache.logits.stride(1),
        col.contiguous(),
        local_logits,
        local_logits.stride(0),
        idx_mapping.contiguous(),
        seeds,
        pos.contiguous(),
        temperature,
        local_vocab,
        cache.vocab_start,
        BLOCK_SIZE=BLOCK,
        USE_FP64=use_fp64,
    )
    # Per-shard winner (first max, i.e. lowest token id among ties).
    best_blk = blk_val.argmax(dim=-1, keepdim=True)
    best_val = blk_val.gather(-1, best_blk).squeeze(-1).to(torch.float64)
    best_id = blk_idx.gather(-1, best_blk).squeeze(-1)
    best_logit = local_logits.gather(
        -1, (best_id - cache.vocab_start).unsqueeze(-1)
    ).squeeze(-1)
    row_max = blk_max.max(dim=-1).values
    row_sum = (
        (blk_sum * torch.exp(blk_max - row_max.unsqueeze(-1))).nan_to_num_(0.0).sum(-1)
    )
    # token ids < 2**53 are exact in fp64
    return torch.stack(
        (
            best_val,
            best_id.to(torch.float64),
            best_logit.to(torch.float64),
            row_max.to(torch.float64),
            row_sum.to(torch.float64),
        ),
        dim=-1,
    )


def draft_combine(
    gathered: torch.Tensor,  # [tp, rows, 5] in vocab-shard order
    idx_mapping: torch.Tensor,
    col: torch.Tensor,
    cache: VPDraftCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global winner and normalizer; records (logit, logsumexp) per slot.
    Returns (tokens, per-shard max [rows, tp], per-shard sumexp [rows, tp])."""
    tp, rows, _ = gathered.shape
    winner = gathered[:, :, 0].argmax(dim=0)  # first max = lowest vocab shard
    pick = gathered.gather(0, winner.view(1, rows, 1).expand(1, rows, 5)).squeeze(0)
    tokens = pick[:, 1].to(torch.int64)
    tok_logit = pick[:, 2].to(torch.float32)
    maxes = gathered[:, :, 3]  # [tp, rows]
    sums = gathered[:, :, 4]
    gmax = maxes.max(dim=0).values
    total = (sums * torch.exp(maxes - gmax)).nan_to_num_(0.0).sum(dim=0)
    lse = (gmax + torch.log(total)).to(torch.float32)
    req = idx_mapping.to(torch.int64)
    slot = torch.where(req >= 0, req, torch.full_like(req, cache.num_reqs))
    colv = col.to(torch.int64)
    cache.token_logit[slot, colv] = tok_logit
    cache.lse[slot, colv] = lse
    return (
        tokens,
        maxes.t().to(torch.float32).contiguous(),
        sums.t().to(torch.float32).contiguous(),
    )


def draft_sample(
    local_logits,
    idx_mapping,
    temperature,
    seeds,
    pos,
    col,
    cache,
    use_fp64: bool = False,
):
    """Sample one draft token per row over the TP-sharded vocabulary: see
    draft_local / draft_combine; one tiny all-gather instead of the logits'."""
    from vllm.distributed import tensor_model_parallel_all_gather

    stats = draft_local(
        local_logits, idx_mapping, temperature, seeds, pos, col, cache, use_fp64
    )
    gathered = tensor_model_parallel_all_gather(stats.unsqueeze(0), dim=0)
    return draft_combine(gathered, idx_mapping, col, cache)


@triton.jit
def _vp_resample_block_kernel(
    out_val_ptr,  # [reqs, nblk]
    out_idx_ptr,  # [reqs, nblk] absolute token id
    nblk,
    target_logits_ptr,  # [num_logits, V] full (processed)
    target_logits_stride,
    target_lse_ptr,  # [reqs] (rejected position), 0 if unused
    draft_cache_ptr,  # [max_reqs, steps, local_vocab] pre-temperature
    draft_stride_0,
    draft_stride_1,
    draft_lse_cache_ptr,  # [max_reqs, steps]
    draft_lse_stride,
    rejected_step_ptr,  # [reqs]
    cu_num_logits_ptr,  # [reqs + 1]
    expanded_idx_mapping_ptr,  # [num_logits]
    draft_sampled_ptr,  # [num_logits]
    temp_ptr,
    seed_ptr,
    pos_ptr,
    local_vocab,
    vocab_start,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """One shard's part of _resample_kernel (standard verification, draft
    logits present): the residual log(max(p - q, 0)) at the rejected position,
    or the target logits at the bonus position, Gumbel-sampled per block."""
    r = tl.program_id(0)
    b = tl.program_id(1)
    resample_idx = tl.load(rejected_step_ptr + r)
    start = tl.load(cu_num_logits_ptr + r).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + r + 1)
    tok_idx = start + resample_idx
    req = tl.load(expanded_idx_mapping_ptr + tok_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    is_bonus = tok_idx == end - 1
    offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < local_vocab
    keys = vocab_start + offs
    if temp == 0.0 and not is_bonus:
        tl.store(out_val_ptr + r * nblk + b, float("-inf"))
        tl.store(
            out_idx_ptr + r * nblk + b, (vocab_start + b * BLOCK_SIZE).to(tl.int64)
        )
        return
    rejected = tl.load(draft_sampled_ptr + tok_idx + 1, mask=not is_bonus, other=0)
    target = tl.load(
        target_logits_ptr + tok_idx * target_logits_stride + keys,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if is_bonus or rejected < 0:
        residual = target
    else:
        draft = (
            tl.load(
                draft_cache_ptr
                + req * draft_stride_0
                + resample_idx * draft_stride_1
                + offs,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        t_lse = tl.load(target_lse_ptr + r)
        d_lse = tl.load(draft_lse_cache_ptr + req * draft_lse_stride + resample_idx)
        t_lp = target - t_lse
        ratio = tl.exp((draft - d_lse) - t_lp)
        residual = tl.where(
            ratio < 1.0, t_lp + tldevice.log1p(-ratio), float("-inf")
        ).to(tl.float32)
        residual = tl.where(mask, residual, float("-inf"))
    seed = tl.load(seed_ptr + req)
    pos = tl.load(pos_ptr + tok_idx)
    value, idx = gumbel_noised_argmax(
        residual,
        keys,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    tl.store(out_val_ptr + r * nblk + b, value)
    tl.store(
        out_idx_ptr + r * nblk + b, (vocab_start + b * BLOCK_SIZE + idx).to(tl.int64)
    )


def resample_local(
    target_logits: torch.Tensor,
    target_rejected_lse: torch.Tensor,
    rejected_step: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    cache: VPDraftCache,
    use_fp64: bool = False,
) -> torch.Tensor:
    """This shard's resample winner per request: fp64 [reqs, 2] (value, id)."""
    reqs = int(cu_num_logits.shape[0]) - 1
    nblk = triton.cdiv(cache.local_vocab, BLOCK)
    dev = target_logits.device
    val_dtype = torch.float64 if use_fp64 else torch.float32
    val = torch.empty((reqs, nblk), dtype=val_dtype, device=dev)
    idx = torch.empty((reqs, nblk), dtype=torch.int64, device=dev)
    _vp_resample_block_kernel[(reqs, nblk)](
        val,
        idx,
        nblk,
        target_logits,
        target_logits.stride(0),
        target_rejected_lse,
        cache.logits,
        cache.logits.stride(0),
        cache.logits.stride(1),
        cache.lse,
        cache.lse.stride(0),
        rejected_step,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cache.local_vocab,
        cache.vocab_start,
        BLOCK_SIZE=BLOCK,
        USE_FP64=use_fp64,
    )
    best = val.argmax(dim=-1, keepdim=True)
    best_val = val.gather(-1, best).squeeze(-1).to(torch.float64)
    best_idx = idx.gather(-1, best).squeeze(-1)
    return torch.stack((best_val, best_idx.to(torch.float64)), dim=-1)


def resample_combine(gathered: torch.Tensor, use_fp64: bool = False):
    """[tp, reqs, 2] -> ([reqs, tp] values, [reqs, tp] ids), shard order kept
    (the insert kernel takes the first max)."""
    val_dtype = torch.float64 if use_fp64 else torch.float32
    return (
        gathered[:, :, 0].t().contiguous().to(val_dtype),
        gathered[:, :, 1].t().contiguous().to(torch.int64),
    )


def resample(
    target_logits,
    target_rejected_lse,
    rejected_step,
    cu_num_logits,
    expanded_idx_mapping,
    draft_sampled,
    temperature,
    seed,
    pos,
    cache,
    use_fp64: bool = False,
):
    """Per-request (value, token) candidates from every shard: ([reqs, tp],
    [reqs, tp]) handed to _insert_resampled_kernel as its vocab blocks."""
    from vllm.distributed import tensor_model_parallel_all_gather

    stats = resample_local(
        target_logits,
        target_rejected_lse,
        rejected_step,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cache,
        use_fp64,
    )
    gathered = tensor_model_parallel_all_gather(stats.unsqueeze(0), dim=0)
    return resample_combine(gathered, use_fp64)
