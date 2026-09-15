# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_blockscaled as _import_b12x_blockscaled,
    get_b12x_dense_activation_mode,
    get_b12x_intrinsics as _import_b12x_intrinsics,
    register_b12x_layer,
    run_b12x_blockscaled_linear,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


def _serialized_name(layer: torch.nn.Module, rows: int) -> str:
    prefix = _resolve_layer_name(layer.b12x_layer_name)
    return f"linear.nvfp4.{prefix}.serialized.m{rows}"


def _declare_serialized_plan(layer: torch.nn.Module, rows: int, out_dtype: torch.dtype):
    api = _import_b12x_blockscaled()
    assert api is not None
    packed = layer.b12x_nvfp4_packed_weight
    return api.plan(api.FixedBlockscaledQuery(
        recipe="nvfp4", call_kind="serialized", max_rows=rows,
        in_features=packed.in_features, padded_in_features=packed.padded_in_features,
        out_features=packed.out_features, input_dtype="uint8",
        output_dtype=str(out_dtype).removeprefix("torch."),
        expected_m=rows, alpha_mode="tensor",
    ))


def _serialized_call(layer: torch.nn.Module, rows: int, out_dtype: torch.dtype):
    packed = layer.b12x_nvfp4_packed_weight
    weight, scales = layer.weight, layer.weight_scale
    activation_scale = None if layer.b12x_weight_only else layer.input_global_scale_inv
    c_dtype = str(out_dtype).removeprefix("torch.")

    def call(state):
        from b12x.preparation import PreparedCall

        source = torch.empty((rows, packed.in_features), dtype=out_dtype, device=weight.device)
        quantized = []

        def produce():
            indices = torch.arange(source.numel(), device=source.device,
                                   dtype=torch.float32).reshape_as(source)
            source.copy_((indices.remainder(43).sub_(21)).mul_(1 / 32))
            quantized[:] = scaled_fp4_quant(
                source, activation_scale, is_sf_swizzled_layout=True,
            )

        def run():
            values, source_scales = quantized
            return state.run_serialized(
                values, source_scales, weight, scales, layer.alpha,
                ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                c_dtype=c_dtype, sf_vec_size=16, block_fp8=False, stream=None,
            )

        return PreparedCall(
            run=run, produce=produce,
            owners=(weight, scales, layer.alpha, activation_scale),
        )

    return call


def _serialized_plan_for(layer: torch.nn.Module, rows: int, out_dtype: torch.dtype):
    """The exact-M serialized plan for rows, declared on first use with its default configuration."""
    plans = layer.b12x_nvfp4_serialized_plans
    plan = plans.get(rows)
    if plan is None:
        plan = _declare_serialized_plan(layer, rows, out_dtype)
        plans[rows] = plan
    return plan


def _b12x_nvfp4_serialized_linear(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Serialized NVFP4 GEMM on pre-quantized activations.

    ``alpha`` overrides the layer's static alpha when the activation scale is
    computed per call (online draft heads); the plan is the layer's exact-M
    serialized declaration either way.
    """
    layer = b12x_layer(_resolve_layer_name(layer_name))
    plan = _serialized_plan_for(layer, int(x_packed.shape[0]), out_dtype)
    api = _import_b12x_blockscaled()
    assert api is not None
    output = api.mm_nvfp4(
        x_packed, x_scale_swizzled, layer.weight, layer.weight_scale,
        layer.alpha if alpha is None else alpha,
        plan=plan, out_dtype=out_dtype,
    )
    if bias is not None:
        output = output + bias
    return output


def _b12x_nvfp4_serialized_linear_fake(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    del x_scale_swizzled, bias, layer_name, alpha
    return x_packed.new_empty((x_packed.shape[0], out_features), dtype=out_dtype)


direct_register_custom_op(
    op_name="b12x_nvfp4_serialized_linear",
    op_func=_b12x_nvfp4_serialized_linear,
    fake_impl=_b12x_nvfp4_serialized_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_nvfp4_serialized_linear(
    x_packed: torch.Tensor,
    x_scale_swizzled: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.b12x_nvfp4_serialized_linear(
        x_packed, x_scale_swizzled, bias, out_features, out_dtype, layer_name, alpha
    )


def _apply_b12x_nvfp4_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output_size = int(layer.weight.shape[0])
    output_shape = [*x.shape[:-1], output_size]
    x_2d = x.reshape(-1, x.shape[-1])
    mode = layer.b12x_activation_mode
    if x.dtype == torch.bfloat16 and layer.b12x_bf16_input_supported:
        source = x_2d.contiguous()
        out_features = int(layer.b12x_nvfp4_packed_weight.out_features)
        output = run_b12x_blockscaled_linear(
            source, bias, out_features, layer.b12x_layer_name,
        )
        return output.view(*output_shape)
    if mode == "a16":
        raise ValueError("b12x NVFP4 A16 requires BF16 activations and N%8=0")
    x_packed, x_scale_swizzled = scaled_fp4_quant(
        x_2d,
        layer.input_global_scale_inv,
        is_sf_swizzled_layout=True,
    )
    output = run_b12x_nvfp4_serialized_linear(
        x_packed, x_scale_swizzled, bias, output_size, x.dtype, layer.b12x_layer_name,
    )
    return output.view(*output_shape)


class B12xNvFp4LinearKernel(NvFp4LinearKernel):
    """ModelOpt NVFP4 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X NVFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X NVFP4 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None or _import_b12x_intrinsics() is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "b12x native NVFP4 GEMM is not supported"
        if not hasattr(blockscaled, "w4a16"):
            return (
                False,
                "b12x NVFP4 requires a source build with dense precision selection",
            )
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # A method that pre-quantizes or keeps BF16 activations sets the mode
        # before weight processing; the configured default applies otherwise.
        mode = getattr(layer, "b12x_activation_mode", None) or get_b12x_dense_activation_mode("nvfp4")
        layer.b12x_weight_only = self.config.use_a16
        if self.config.use_a16:
            if mode == "quantized":
                raise ValueError("W4A16_NVFP4 checkpoint cannot use quantized activations")
            mode = "a16"
        layer.b12x_activation_mode = mode
        n, packed_k = layer.weight.shape
        logical_k = int(packed_k) * 2
        if mode == "a16" and logical_k % 32:
            stored_k = (logical_k + 31) // 32 * 32
            # Align serialized storage without changing the model's logical K
            # or splitting a 16-element quantization group.
            values = torch.zeros((n, stored_k // 2), dtype=torch.uint8, device=layer.weight.device)
            values[:, :packed_k].copy_(layer.weight.data)
            scales = torch.zeros((n, stored_k // 16), dtype=layer.weight_scale.dtype,
                                 device=layer.weight_scale.device)
            scales[:, :logical_k // 16].copy_(layer.weight_scale.data)
            replace_parameter(layer, "weight", values)
            replace_parameter(layer, "weight_scale", scales)
        intrinsics = _import_b12x_intrinsics()
        assert intrinsics is not None
        replace_parameter(
            layer, "weight_scale", intrinsics.swizzle_block_scale(layer.weight_scale.data),
        )
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        packed = blockscaled.pack_weight(
            layer.weight.data, layer.weight_scale.data, recipe="nvfp4",
            global_scale=layer.weight_global_scale,
        )
        layer.b12x_nvfp4_packed_weight = replace(packed, in_features=logical_k)
        layer.b12x_bf16_input_supported = (
            current_platform.is_device_capability_family(120)
            and (logical_k % 128 == 0 or mode == "a16")
            and n % 8 == 0
        )
        name = b12x_layer_prefix(layer)
        activation_scale = None if layer.b12x_weight_only else layer.input_global_scale_inv
        # A reload into the same packed storage keeps the holder and its
        # prepared plan; new storage declares anew.
        existing = getattr(layer, "b12x_linear", None)
        if existing is None or not existing.holds(layer.b12x_nvfp4_packed_weight):
            layer.b12x_linear = B12xBlockscaledLinear(
                layer.b12x_nvfp4_packed_weight,
                recipe="nvfp4",
                activation_mode=mode,
                layer_name=name,
                activation_scale=activation_scale,
            )
        else:
            layer.b12x_nvfp4_packed_weight = existing.packed
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        layer.b12x_nvfp4_serialized_plans = {}
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload,
    ) -> Sequence[B12xPreparationUnit]:
        packed = layer.b12x_nvfp4_packed_weight
        weight, scales = layer.weight, layer.weight_scale
        if packed.values.is_meta or weight.is_meta or scales.is_meta:
            return ()
        packed_input = (
            workload.output_dtype == torch.bfloat16
            and layer.b12x_bf16_input_supported
            and not getattr(layer, "b12x_nvfp4_serialized_activations", False)
        )
        if layer.b12x_activation_mode == "a16" and not packed_input:
            raise ValueError("b12x W4A16 preparation requires BF16 activations and N%8=0")
        if packed_input:
            linear = layer.b12x_linear
            return (linear.unit(workload, name=f"linear.nvfp4.{linear.layer_name}"),)

        prefix = _resolve_layer_name(layer.b12x_layer_name)
        plans = layer.b12x_nvfp4_serialized_plans
        requests = []
        for rows in workload.token_counts:
            plan = plans.get(rows)
            if plan is None:
                plan = _declare_serialized_plan(layer, rows, workload.output_dtype)
                plans[rows] = plan
            call = _serialized_call(layer, rows, workload.output_dtype)
            requests.append(plan.request(
                name=_serialized_name(layer, rows),
                prepare_call=call, benchmark_call=call,
            ))
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="NVFP4",
                key=(prefix, "serialized", tuple(sorted(plans))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def get_workspace_size(self, layer: torch.nn.Module, rows: int) -> int:
        linear = getattr(layer, "b12x_linear", None)
        return 0 if linear is None else linear.get_workspace_size(rows)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_b12x_nvfp4_linear(
            layer,
            x,
            bias,
        )


__all__ = ["B12xNvFp4LinearKernel"]
