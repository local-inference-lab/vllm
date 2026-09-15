# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_blockscaled as _import_b12x_blockscaled,
    get_b12x_intrinsics as _import_b12x_intrinsics,
    register_b12x_layer,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


def _plan(layer: torch.nn.Module, tokens: int):
    try:
        return layer.b12x_mxfp4_plans[tokens]
    except (AttributeError, KeyError):
        raise PreparationResourceUnavailableError(
            f"MXFP4 layer has no declared plan for exact M={tokens}") from None


def _b12x_mxfp4_linear(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    plan = _plan(layer, int(x_packed.shape[0]))
    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None
    output = blockscaled.mm_mxfp4(
        x_packed, x_scale_swizzled, layer.weight, layer.weight_scale,
        plan=plan, out_dtype=out_dtype,
    )
    if bias is not None:
        output = output + bias
    return output


def _b12x_mxfp4_linear_fake(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    del x_scale_swizzled, bias, layer_name
    return x_packed.new_empty((x_packed.shape[0], out_features), dtype=out_dtype)


direct_register_custom_op(
    op_name="b12x_mxfp4_linear",
    op_func=_b12x_mxfp4_linear,
    fake_impl=_b12x_mxfp4_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_mxfp4_linear(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.ops.vllm.b12x_mxfp4_linear(
        x_packed, x_scale_swizzled, bias, out_features, out_dtype, layer_name
    )


def _apply_b12x_mxfp4_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

    output_size = int(layer.weight.shape[0])
    output_shape = [*x.shape[:-1], output_size]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_packed, x_scale_swizzled = flashinfer_mxfp4_quantize(x_2d, backend="cute-dsl")
    output = run_b12x_mxfp4_linear(
        x_packed, x_scale_swizzled, bias, output_size, x.dtype, layer.b12x_layer_name,
    )
    return output.view(*output_shape)


class B12xMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 linear through prepared native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X MXFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X MXFP4 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None or _import_b12x_intrinsics() is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "b12x native MXFP4 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "B12X MXFP4 GEMM requires dynamic MXFP4 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        intrinsics = _import_b12x_intrinsics()
        assert intrinsics is not None
        replace_parameter(layer, "weight_scale", intrinsics.swizzle_block_scale(layer.weight_scale.data))
        name = b12x_layer_prefix(layer)
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        layer.b12x_mxfp4_plans = {}
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload,
    ) -> Sequence[B12xPreparationUnit]:
        weight, scales = layer.weight, layer.weight_scale
        if weight.is_meta or scales.is_meta:
            return ()
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        n, packed_k = map(int, weight.shape)
        prefix = _resolve_layer_name(layer.b12x_layer_name)
        plans = layer.b12x_mxfp4_plans
        requests = []
        for tokens in workload.token_counts:
            plan = plans.get(tokens)
            if plan is None:
                query = blockscaled.FixedBlockscaledQuery(
                    recipe="mxfp4", call_kind="serialized", max_rows=tokens,
                    in_features=packed_k * 2, padded_in_features=packed_k * 2,
                    out_features=n, input_dtype="uint8",
                    output_dtype=str(workload.output_dtype).removeprefix("torch."),
                    expected_m=tokens,
                )
                plan = blockscaled.plan(query)
                plans[tokens] = plan

            def call(state, *, rows=tokens):
                from b12x.preparation import PreparedCall
                from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

                # The trial owns its activation/quantization buffers; only the
                # loaded weights and their swizzled scales are borrowed.
                source = torch.empty(
                    (rows, packed_k * 2),
                    dtype=workload.output_dtype,
                    device=weight.device,
                )
                quantized: list[torch.Tensor] = []

                def produce() -> None:
                    indices = torch.arange(
                        source.numel(), device=source.device, dtype=torch.float32,
                    ).reshape_as(source)
                    source.copy_((indices.remainder(37).sub_(18)).mul_(1 / 32))
                    values, source_scales = flashinfer_mxfp4_quantize(
                        source, backend="cute-dsl",
                    )
                    quantized[:] = [values, source_scales]

                def run():
                    values, source_scales = quantized
                    return state.run_serialized(
                        values, source_scales, weight, scales, None,
                        ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu",
                        c_dtype=str(workload.output_dtype).removeprefix("torch."),
                        sf_vec_size=32, block_fp8=False, stream=None,
                    )

                return PreparedCall(
                    run=run, produce=produce, owners=(weight, scales),
                )
            requests.append(plan.request(
                name=f"linear.mxfp4.{prefix}.m{tokens}",
                prepare_call=call,
                benchmark_call=call,
            ))
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="MXFP4",
                key=(prefix, tuple(sorted(plans))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
        return _apply_b12x_mxfp4_linear(layer, x, bias)


__all__ = ["B12xMxFp4LinearKernel"]
