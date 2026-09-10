# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_TOPK = 6

# Adapted from:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/moe_fused_gate.py


def can_use_dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    indices_dtype: torch.dtype,
) -> bool:
    return (
        current_platform.is_cuda()
        and gating_output.dtype == torch.float32
        and gating_output.ndim == 2
        and gating_output.shape[1] in (256, 384)
        and gating_output.is_contiguous()
        and correction_bias is not None
        and correction_bias.dtype == torch.float32
        and correction_bias.shape == (gating_output.shape[1],)
        and correction_bias.is_contiguous()
        and topk == _TOPK
        and renormalize
        and indices_dtype in (torch.int32, torch.uint32, torch.int64)
    )


if current_platform.is_cuda():

    @triton.jit
    def _dsv4_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        input_ids_ptr,
        bias_vl_ptr,
        image_sentinel_lo,
        hash_indices_ptr,
        is_padding_ptr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        RENORMALIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_VL: tl.constexpr,
        HAS_HASH: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row = tl.program_id(0)
        topk_offsets = tl.arange(0, BLOCK_K)
        output_mask = topk_offsets < TOP_K
        output_offsets = row * TOP_K + topk_offsets
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        if HAS_PADDING and tl.load(is_padding_ptr + row):
            tl.store(topk_weights_ptr + output_offsets, 0.0, mask=output_mask)
            tl.store(topk_ids_ptr + output_offsets, -1, mask=output_mask)
            return

        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS
        bias = tl.zeros([BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            bias = tl.load(
                correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0
            ).to(tl.float32)
        is_image = False
        if HAS_VL or HAS_HASH:
            token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        if HAS_VL:
            # Image tokens carry five consecutive in-vocab sentinel ids
            # starting at image_sentinel_lo and use bias_vl for expert
            # selection instead of the regular correction bias. Ids above the
            # sentinel block are regular special tokens and must not match.
            bias_vl = tl.load(
                bias_vl_ptr + expert_offsets, mask=expert_mask, other=0.0
            ).to(tl.float32)
            is_image = (token_id >= image_sentinel_lo) & (
                token_id < image_sentinel_lo + 5
            )
            bias = tl.where(is_image, bias_vl, bias)

        if HAS_HASH and not is_image:
            selected_ids = tl.load(
                hash_indices_ptr + token_id * TOP_K + topk_offsets,
                mask=output_mask,
                other=0,
            ).to(tl.int32)
            hash_logits = tl.load(
                gating_output_ptr + row * NUM_EXPERTS + selected_ids,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            selected_weights = tl.sqrt(
                tl.where(
                    hash_logits > 20.0, hash_logits, tl.log(1.0 + tl.exp(hash_logits))
                )
            )
            selected_weights = tl.where(output_mask, selected_weights, 0.0)
        else:
            logits = tl.load(
                gating_output_ptr + row * NUM_EXPERTS + expert_offsets,
                mask=expert_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.sqrt(
                tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))
            )
            current = tl.where(expert_mask, weights + bias, -float("inf"))
            current = tl.where(current == current, current, -1e30)
            weights = tl.where(weights == weights, weights, 0.0)

            selected_weights = tl.zeros([BLOCK_K], dtype=tl.float32)
            selected_ids = tl.zeros([BLOCK_K], dtype=tl.int32)
            for slot in tl.static_range(TOP_K):
                max_value = tl.max(current, axis=0)
                candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
                expert_id = tl.min(candidate, axis=0).to(tl.int32)
                selected_weight = tl.sum(
                    tl.where(expert_offsets == expert_id, weights, 0.0), axis=0
                )
                is_slot = topk_offsets == slot
                selected_weights = tl.where(is_slot, selected_weight, selected_weights)
                selected_ids = tl.where(is_slot, expert_id, selected_ids)
                current = tl.where(expert_offsets == expert_id, -float("inf"), current)

        selected_weights = tl.where(
            selected_weights == selected_weights, selected_weights, 0.0
        )
        if RENORMALIZE:
            weight_sum = tl.sum(selected_weights, axis=0)
            selected_weights /= tl.where(weight_sum > 0.0, weight_sum, 1.0)
        selected_weights *= routed_scaling_factor

        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()

        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    input_ids: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    hash_indices_table: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
    topk: int = _TOPK,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gating_output.shape
    assert current_platform.is_cuda(), "DeepSeek V4 vision routing requires CUDA"
    assert gating_output.is_contiguous()
    assert 0 < topk <= num_experts
    has_vl = bias_vl is not None and image_sentinel_lo > 0
    if bias_vl is not None:
        assert input_ids is not None, "bias_vl routing requires input_ids"
        assert bias_vl.dtype == torch.float32 and bias_vl.is_contiguous()
        assert bias_vl.shape == (num_experts,)
        assert input_ids.is_contiguous()
    if hash_indices_table is not None:
        assert input_ids is not None and input_ids.is_contiguous()
        assert hash_indices_table.is_contiguous()
        assert hash_indices_table.shape[1] == topk
    shape = (num_tokens, topk)
    topk_weights = gating_output.new_empty(shape, dtype=torch.float32)
    topk_ids = gating_output.new_empty(shape, dtype=indices_dtype)
    if num_tokens > 0:
        _dsv4_topk_kernel[(num_tokens,)](
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            input_ids,
            bias_vl,
            image_sentinel_lo,
            hash_indices_table,
            is_padding,
            NUM_EXPERTS=num_experts,
            BLOCK_N=triton.next_power_of_2(num_experts),
            TOP_K=topk,
            BLOCK_K=triton.next_power_of_2(topk),
            RENORMALIZE=renormalize,
            HAS_BIAS=correction_bias is not None,
            HAS_VL=has_vl,
            HAS_HASH=hash_indices_table is not None,
            HAS_PADDING=is_padding is not None,
            num_warps=1,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    return topk_weights, topk_ids
