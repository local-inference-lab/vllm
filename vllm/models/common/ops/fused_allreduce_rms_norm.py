# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
    get_b12x_pcie_allreduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM,
    _can_use_flashinfer,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.layernorm import RMSNorm


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + standard RMSNorm with available fusion.

    An active B12X PCIe communicator gets the first opportunity to execute the
    fused operation; FlashInfer is used for shapes B12X rejects. The final
    fallback is an explicit all-reduce followed by RMSNorm.

    Args:
        hidden_states: Per-rank partial output of a row-parallel linear run
            with ``reduce_results=False``. The B12X and FlashInfer paths
            overwrite it with the new residual.
        residual: Residual stream added after the reduction.
        norm: RMSNorm applied to the reduced sum.

    Returns:
        ``(normed_output, new_residual)``, equivalent to
        ``norm(all_reduce(hidden_states), residual)``.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states, residual)

    b12x_communicator = get_b12x_pcie_allreduce()
    if b12x_communicator is not None and b12x_communicator.try_fused_add_rms_norm(
        hidden_states,
        residual,
        norm.weight,
        norm.variance_epsilon,
    ):
        return hidden_states, residual

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            norm_out = torch.empty_like(hidden_states)
            # With norm_out provided, the kernel writes the new residual
            # (all_reduce(hidden_states) + residual) into the hidden_states
            # buffer and the normalized result into norm_out.
            flashinfer_trtllm_fused_allreduce_norm(
                allreduce_in=hidden_states,
                residual=residual,
                rms_gamma=norm.weight,
                rms_eps=norm.variance_epsilon,
                world_size=tp_size,
                weight_bias=0.0,  # standard RMSNorm (Gemma would use 1.0)
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=max_token_num,
                pattern_code=_AR_RESIDUAL_RMS_NORM,
                norm_out=norm_out,
            )
            return norm_out, hidden_states

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
