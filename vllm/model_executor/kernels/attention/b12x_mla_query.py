# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X BF16 MLA query projection and assembly."""

import torch

from vllm.utils.b12x import b12x_layer, get_b12x_mla_query_projection
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)


def can_implement_bf16_mla_query(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    module = get_b12x_mla_query_projection()
    return bool(
        module is not None
        and module.can_implement(
            num_heads=num_heads,
            max_m=max_m,
            nope_dim=nope_dim,
            latent_dim=latent_dim,
            output_dtype=output_dtype,
            weight_format="bf16",
            device=device,
        )
    )


def _b12x_bf16_mla_query_impl(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    module = get_b12x_mla_query_projection()
    if module is None:
        raise ImportError("b12x.gemm.mla_query_projection is not available")
    layer = b12x_layer(_resolve_layer_name(layer_name))
    plan = layer.b12x_query_plan(int(q_nope.shape[1]))
    module.run(q_nope, weight, q_pe, output, plan=plan)


def _b12x_bf16_mla_query_fake(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    # The layer name is intentionally not resolved while tracing.
    del q_nope, weight, q_pe, output, layer_name


direct_register_custom_op(
    op_name="b12x_bf16_mla_query",
    op_func=_b12x_bf16_mla_query_impl,
    mutates_args=["output"],
    fake_impl=_b12x_bf16_mla_query_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_bf16_mla_query(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
    *,
    layer_name: LayerNameType,
) -> torch.Tensor:
    torch.ops.vllm.b12x_bf16_mla_query(q_nope, weight, q_pe, output, layer_name)
    return output


__all__ = ["can_implement_bf16_mla_query", "run_bf16_mla_query"]
