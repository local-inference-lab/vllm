# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for replicated-embedding fused gather/norm kernels.

Enabling ``VLLM_REPLICATE_EMBED`` must preserve every gathered embedding and
match the unfused RMSNorm reference.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_embed_norm import (
    fused_embed_eh_norm,
    fused_embed_norm,
)
from vllm.models.deepseek_v32.common.kernels import fused_eh_norm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPE = torch.bfloat16
VOCAB, HIDDEN, NUM_TOKENS, EPS = 8192, 4096, 129, 1e-6

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused embed/norm Triton kernels require a CUDA/ROCm device",
)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Full-precision (fp32) reference RMSNorm.
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(var + eps) * w.float()


@requires_cuda
@torch.inference_mode()
def test_fused_embed_norm_matches_reference():
    """Main-model fusion: the residual is the exact gather and the second output
    is a correct RMSNorm. The norm matches a full-precision reference to ~2 bf16
    ulp (rtol 1e-2) -- that gap is bf16 rounding, not the kernel."""
    set_random_seed(13)
    table = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device="cuda")
    ids = torch.randint(0, VOCAB, (NUM_TOKENS,), dtype=torch.int32, device="cuda")
    weight = torch.empty(HIDDEN, dtype=DTYPE, device="cuda").normal_(1.0, 0.1)

    residual, normed = fused_embed_norm(ids, table, chain_weight=weight, eps=EPS)

    embeds = table[ids.long()]
    torch.testing.assert_close(residual, embeds, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        normed.float(), _rmsnorm(embeds, weight, EPS), atol=1e-3, rtol=1e-2
    )


@requires_cuda
@torch.inference_mode()
def test_fused_embed_eh_norm_matches_reference():
    """The fused MTP gather/norm must preserve every embedding row."""
    set_random_seed(13)
    table = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device="cuda")
    ids = torch.randint(0, VOCAB, (NUM_TOKENS,), dtype=torch.int32, device="cuda")
    prev = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device="cuda")
    positions = torch.arange(NUM_TOKENS, dtype=torch.int64, device="cuda")
    enorm_w = torch.randn(HIDDEN, dtype=DTYPE, device="cuda")
    hnorm_w = torch.randn(HIDDEN, dtype=DTYPE, device="cuda")

    fused = fused_embed_eh_norm(ids, table, prev, enorm_w, hnorm_w, EPS)
    local = fused_eh_norm(positions, table[ids.long()], prev, enorm_w, hnorm_w, EPS)
    ref = torch.cat(
        (
            _rmsnorm(table[ids.long()], enorm_w, EPS).to(DTYPE),
            _rmsnorm(prev, hnorm_w, EPS).to(DTYPE),
        ),
        dim=-1,
    )

    # Both Triton paths implement the same operation and must remain bit-exact.
    torch.testing.assert_close(fused, local, atol=0.0, rtol=0.0)
    # PyTorch may reduce RMSNorm inputs in a different order. Compare against
    # the independent reference at the precision promised by the BF16 output.
    torch.testing.assert_close(fused.float(), ref.float(), atol=1e-3, rtol=1e-2)
