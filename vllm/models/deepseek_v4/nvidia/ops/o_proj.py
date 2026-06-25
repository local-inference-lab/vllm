# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op


def _deepseek_v4_fp8_o_proj_einsum(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    z: torch.Tensor,
    recipe_0: int,
    recipe_1: int,
    recipe_2: int,
) -> None:
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (weight, weight_scale_inv),
        z,
        recipe=(recipe_0, recipe_1, recipe_2),
    )


def _deepseek_v4_fp8_o_proj_einsum_fake(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    z: torch.Tensor,
    recipe_0: int,
    recipe_1: int,
    recipe_2: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_o_proj_einsum",
    op_func=_deepseek_v4_fp8_o_proj_einsum,
    mutates_args=["z"],
    fake_impl=_deepseek_v4_fp8_o_proj_einsum_fake,
)


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
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
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
    torch.ops.vllm.deepseek_v4_fp8_o_proj_einsum(
        o_fp8,
        o_scale,
        wo_a.weight,
        wo_a.weight_scale_inv,
        z,
        einsum_recipe[0],
        einsum_recipe[1],
        einsum_recipe[2],
    )
    return wo_b(z.flatten(1))
