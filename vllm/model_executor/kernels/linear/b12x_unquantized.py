# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-sized unquantized linears through b12x ``gemm.bf16_gemv``.

Under ``--linear-backend b12x``, BF16 linear layers declare b12x projection
plans for the CUDA-graph token counts up to ``MAX_ROWS``.  Preparation
autotunes each plan across b12x's GEMV kernels and cuBLAS (b12x's ``torch``
backend), so a layer leaves cuBLAS only where a native kernel measures faster;
a layer whose plans all keep cuBLAS traces a plain ``F.linear``.  Larger
batches keep ``F.linear``.
"""

from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_bf16_gemv,
    register_b12x_layer,
    set_b12x_preparation_provider,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

MAX_ROWS = 8


def _aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % 16 == 0 and tensor.stride(0) % 8 == 0


def _plan(api, layer: torch.nn.Module, rows: int, bias: torch.Tensor | None):
    n, k = map(int, layer.weight.shape)
    return api.plan(
        api.GemvQuery(
            source_dtype="bfloat16",
            weight_dtype="bfloat16",
            max_rows=rows,
            in_features=k,
            out_features=n,
            source_contiguous=True,
            source_aligned=True,
            weight_contiguous=True,
            weight_aligned=True,
            bias_dtype=None if bias is None else "bfloat16",
        )
    )


def _b12x_bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    rows = int(x.shape[0])
    plan = min(
        (
            (capacity, plan)
            for capacity, plan in layer.b12x_bf16_gemv_plans.items()
            if capacity >= rows
        ),
        default=(None, None),
        key=lambda item: item[0],
    )[1]
    if plan is None or rows == 0 or not x.is_contiguous() or not _aligned(x):
        return torch.nn.functional.linear(x, weight, bias)
    api = get_b12x_bf16_gemv()
    assert api is not None
    return api.mm(x, weight, plan=plan, bias=bias)


def _b12x_bf16_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: LayerNameType,
) -> torch.Tensor:
    del bias, layer_name
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="b12x_bf16_linear",
    op_func=_b12x_bf16_linear,
    fake_impl=_b12x_bf16_linear_fake,
)


def b12x_bf16_gemv_enabled() -> bool:
    """Routing needs autotuning: only measured plans may leave cuBLAS."""
    config = get_current_vllm_config_or_none()
    return (
        envs.VLLM_B12X_BF16_GEMV
        and config is not None
        and config.kernel_config.enable_b12x_autotune
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and get_b12x_bf16_gemv() is not None
    )


def maybe_attach_b12x_bf16_gemv(layer: torch.nn.Module) -> bool:
    """Route the layer's decode-sized calls through b12x when its weight fits."""
    weight = getattr(layer, "weight", None)
    bias = getattr(layer, "bias", None)
    if (
        not b12x_bf16_gemv_enabled()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or not _aligned(weight)
        or (bias is not None and bias.dtype != torch.bfloat16)
        or getattr(layer, "b12x_preparation_suppressed", False)
    ):
        return False
    name = b12x_layer_prefix(layer)
    register_b12x_layer(name, layer)
    layer.b12x_layer_name = _encode_layer_name(name)
    # Plain-string copy for the trace-time routing check: torch.compile sees
    # the opaque LayerName as a FakeScriptObject.
    layer.b12x_bf16_gemv_name = name
    layer.b12x_bf16_gemv_plans = {}
    set_b12x_preparation_provider(layer, _B12xBf16GemvProvider())
    return True


@torch.compiler.assume_constant_result
def _selects_native_kernel(layer_name: str) -> bool:
    """Whether preparation picked a b12x kernel over cuBLAS for any row count.

    Weight-stage preparation completes before the model is first traced, so a
    layer whose plans all chose cuBLAS keeps a plain ``F.linear``.
    """
    plans = b12x_layer(layer_name).b12x_bf16_gemv_plans
    return any(
        plan.selection is not None and plan.selection.config.backend != "torch"
        for plan in plans.values()
    )


def b12x_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        getattr(layer, "b12x_bf16_gemv_plans", None) is None
        or x.ndim != 2
        or not _selects_native_kernel(layer.b12x_bf16_gemv_name)
    ):
        return torch.nn.functional.linear(x, weight, bias)
    return torch.ops.vllm.b12x_bf16_linear(x, weight, bias, layer.b12x_layer_name)


class _B12xBf16GemvProvider:
    """Declares one autotuned projection plan per decode token count."""

    def get_b12x_preparation_units(
        self,
        layer: torch.nn.Module,
        workload: B12xWorkload,
    ) -> Sequence[B12xPreparationUnit]:
        api = get_b12x_bf16_gemv()
        weight = layer.weight
        if api is None or weight.is_meta:
            return ()
        bias = getattr(layer, "bias", None)
        k = int(weight.shape[1])
        prefix = _resolve_layer_name(layer.b12x_layer_name)
        plans = layer.b12x_bf16_gemv_plans
        requests = []
        for rows in sorted({r for r in workload.token_counts if r <= MAX_ROWS}):
            plan = plans.get(rows)
            if plan is None:
                plan = plans[rows] = _plan(api, layer, rows, bias)

            def call(state, *, m=rows):
                from b12x.preparation import PreparedCall

                x = torch.empty((m, k), dtype=torch.bfloat16, device=weight.device)
                return PreparedCall(
                    run=lambda: state.run(x, weight, bias=bias),
                    produce=lambda: x.normal_(),
                    owners=(weight,) if bias is None else (weight, bias),
                )

            requests.append(
                plan.request(
                    name=f"linear.bf16_gemv.{prefix}.m{rows}",
                    prepare_call=call,
                    benchmark_call=call,
                )
            )
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="BF16_GEMV",
                key=(prefix, workload.token_counts),
                requests=tuple(requests),
                stage="weights",
            ),
        )
