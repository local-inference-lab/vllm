# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requantise a draft model's lm_head to FP8.

Draft tokens are proposals the target verifies or replaces, so the served distribution
depends on the target alone: this can cost acceptance, never correctness.

Worth doing because an MTP head is read once per speculative step -- at k=3 a
154,880 x 6,144 bf16 head is 1.4 GB/rank/step at TP=4, a large share of a memory-bound
decode step. Draft model only; the target's head is untouched.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


class Fp8DraftHeadMethod:
    """Drop-in ``quant_method`` replacement: only the matmul changes.

    Swapped in after loading, so the checkpoint path, vocab-parallel sharding and the TP
    gather are untouched.
    """

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor):
        from vllm import _custom_ops as ops

        self.weight_fp8 = weight_fp8                              # [vocab_rank, hidden]
        # [1, N] to broadcast against cutlass_scaled_mm's [K, N] operand; bound once
        # rather than reshaped/imported per drafter step.
        self.weight_scale = weight_scale.reshape(1, -1).contiguous()
        self._scaled_mm = ops.cutlass_scaled_mm

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"FP8 draft head needs bf16/fp16 hidden states, got {x.dtype}."
            )
        flat = x.reshape(-1, x.shape[-1])
        # Per-token scale; decode drafts a handful of rows, so this is far cheaper than
        # the weight read it enables.
        x_amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_scale = (x_amax / FP8_MAX).to(torch.float32)
        x_fp8 = (flat / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        # cutlass_scaled_mm, not torch._scaled_mm: runs at M=1, where _scaled_mm's
        # alignment requirements are version-dependent.
        out = self._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])


ROWS_PER_CHUNK = 4096


def _quantize_rowwise(wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-scaled FP8 copy of ``wd``, built in chunks.

    Per-output-row rather than per-tensor: a few high-magnitude vocabulary rows would
    otherwise set one global scale and crush every other token's logit resolution.

    Chunked because promoting a 454 MB head to fp32 whole allocates ~908 MB and the clamp
    another -- unaffordable on unified memory. Caps the transient at ~100 MB.
    """
    out = torch.empty_like(wd, dtype=FP8_DTYPE)
    scale = torch.empty(wd.shape[0], 1, dtype=torch.float32, device=wd.device)
    for lo in range(0, wd.shape[0], ROWS_PER_CHUNK):
        hi = min(lo + ROWS_PER_CHUNK, wd.shape[0])
        block = wd[lo:hi].to(torch.float32)
        s = (block.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        out[lo:hi] = (block / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        scale[lo:hi] = s
        del block, s
    return out, scale


def quantize_draft_lm_head_fp8(model: torch.nn.Module) -> int:
    """Swap every ParallelLMHead in ``model`` to FP8. Returns bytes freed."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    # Tied embeddings share one storage between lm_head and embed_tokens. Quantising stays
    # safe (separate FP8 tensor, only the matmul swaps) but freeing the bf16 copy would
    # delete the input embedding. Detect by storage, not by tie_word_embeddings, so a model
    # that ties without advertising it is still handled.
    shared: dict[int, int] = {}
    for p in model.parameters():
        if p is not None and p.device.type != "meta":
            shared[p.data_ptr()] = shared.get(p.data_ptr(), 0) + 1

    freed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, ParallelLMHead):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dtype not in (torch.bfloat16, torch.float16):
            continue

        wd = w.data
        w_fp8, scale = _quantize_rowwise(wd)
        before = wd.numel() * wd.element_size()
        after = w_fp8.numel() * w_fp8.element_size() + scale.numel() * scale.element_size()
        mod.quant_method = Fp8DraftHeadMethod(w_fp8, scale)

        tied = shared.get(wd.data_ptr(), 1) > 1
        if tied:
            # Keep bf16: the input embedding still needs it. The traffic saving comes from
            # the matmul, not from freeing, so it still applies.
            logger.info(
                "Draft lm_head %s is tied: using FP8 for the matmul, keeping the bf16 copy "
                "(%.0f MB not freed)", name, before / 2**20
            )
        else:
            # Freeing makes this net-negative on memory. A later read of `.weight` now fails
            # loudly rather than silently falling back to bf16 and faking the speedup.
            mod.register_parameter("weight", None)
            freed += before - after
        del wd, w
        logger.info(
            "Draft lm_head %s requantised to FP8: %.0f MB -> %.0f MB",
            name, before / 2**20, after / 2**20,
        )

    if freed:
        torch.cuda.empty_cache()
    return freed
