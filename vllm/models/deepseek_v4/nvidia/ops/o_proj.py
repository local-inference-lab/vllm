# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    nope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Inverse interleaved RoPE followed by unquantized grouped BF16 projection."""
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    pairs = o[..., nope_dim:].float().unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(-1)
    rotated = (
        torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
        .flatten(-2)
        .to(o.dtype)
    )
    values = torch.cat((o[..., :nope_dim], rotated), dim=-1)
    grouped = values.reshape(o.shape[0], n_groups, -1).transpose(0, 1)
    weight = wo_a.weight.reshape(n_groups, o_lora_rank, -1)
    projected = torch.bmm(grouped, weight.transpose(1, 2)).transpose(0, 1)
    return wo_b(projected.flatten(1))


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection with BF16 or block-scaled FP8 weights.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    if wo_a.weight.dtype == torch.bfloat16:
        return bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            nope_dim=nope_dim,
            o_lora_rank=o_lora_rank,
        )
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
