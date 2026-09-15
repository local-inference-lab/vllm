# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_blockscaled as _import_b12x_blockscaled,
    get_b12x_tensor_fp8_linear as _import_b12x_tensor_fp8,
    register_b12x_layer,
    reuse_packed_weight_storage,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel


def _block_fp8_plan(layer: torch.nn.Module, rows: int, out_dtype: torch.dtype):
    plans = layer.b12x_block_fp8_plans
    plan = plans.get(rows)
    if plan is None:
        api = _import_b12x_blockscaled()
        assert api is not None
        n, k = map(int, layer.weight.shape)
        query = api.FixedBlockscaledQuery(
            recipe="block_fp8", call_kind="serialized", max_rows=rows,
            in_features=k, padded_in_features=k, out_features=n,
            input_dtype="float8_e4m3fn",
            output_dtype=str(out_dtype).removeprefix("torch."), expected_m=rows,
        )
        plan = api.plan(query)
        plans[rows] = plan
    return plan

def _b12x_block_fp8_linear(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    plan = _block_fp8_plan(layer, int(a.shape[0]), out_dtype)
    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None
    return blockscaled.mm_block_fp8(
        a, a_scale, weight, weight_scale, plan=plan, out_dtype=out_dtype,
    )


def _b12x_block_fp8_linear_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    del a_scale, weight_scale, layer_name
    return a.new_empty((a.shape[0], weight.shape[0]), dtype=out_dtype)


direct_register_custom_op(
    op_name="b12x_block_fp8_linear",
    op_func=_b12x_block_fp8_linear,
    fake_impl=_b12x_block_fp8_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_block_fp8_linear(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.ops.vllm.b12x_block_fp8_linear(
        a, a_scale, weight, weight_scale, out_dtype, layer_name
    )


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """K128 block-FP8 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X FP8 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "B12X regular block-FP8 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if config.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 input dtype"
        if config.input_dtype != config.out_dtype:
            return False, "Input and output dtype must match"

        act_group_shape = config.activation_quant_key.scale.group_shape
        if act_group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per-token group activation quantization "
                "with group_shape=(1,128)",
            )
        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"

        out_features, in_features = config.weight_shape
        if in_features <= 0 or in_features % 128 != 0:
            return False, "Input features must be a positive multiple of 128"
        if out_features <= 0 or out_features % 128 != 0:
            return False, "Output features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        if params.weight_scale_inv is not None:
            weight_scale = params.weight_scale_inv
            scale_attr = params.WEIGHT_SCALE_INV
        else:
            weight_scale = params.weight_scale
            scale_attr = params.WEIGHT_SCALE
        if weight_scale is not None and weight_scale.dtype in (
            torch.float8_e8m0fnu,
            torch.uint8,
        ):
            # TODO: Remove once B12X supports 128x128 UE8M0 block scales.
            replace_parameter(
                layer,
                scale_attr,
                _upcast_e8m0_to_fp32(weight_scale).contiguous(),
            )
        name = b12x_layer_prefix(layer)
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        layer.b12x_block_fp8_plans = {}
        self._b12x_block_fp8_owner = layer
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload,
    ) -> Sequence[B12xPreparationUnit]:
        weight = layer.weight
        weight_scale = getattr(layer, "weight_scale_inv", None)
        if weight_scale is None:
            weight_scale = layer.weight_scale
        if weight.is_meta or weight_scale.is_meta:
            return ()
        k = int(weight.shape[1])
        prefix = _resolve_layer_name(layer.b12x_layer_name)
        requests = []
        for rows in workload.token_counts:
            plan = _block_fp8_plan(layer, rows, workload.output_dtype)

            def call(state, *, m=rows):
                from b12x.preparation import PreparedCall

                values = torch.empty(
                    (m, k), dtype=torch.float8_e4m3fn, device=weight.device,
                )
                scales = torch.empty(
                    (m, k // 128), dtype=torch.float32, device=weight.device,
                )

                def produce() -> None:
                    indices = torch.arange(
                        values.numel(), device=values.device, dtype=torch.float32,
                    ).reshape_as(values)
                    values.copy_((indices.remainder(31).sub_(15)).mul_(1 / 32))
                    # Dynamic block-FP8 consumes a per-token, per-K-block scale.
                    scales.copy_(
                        torch.linspace(
                            0.5, 1.0, scales.numel(), device=scales.device,
                            dtype=scales.dtype,
                        ).reshape_as(scales),
                    )

                return PreparedCall(
                    run=lambda: state.run_serialized(
                        values, scales, weight, weight_scale, None,
                        ab_dtype="float8_e4m3fn", sf_dtype="float32",
                        c_dtype=str(workload.output_dtype).removeprefix("torch."),
                        sf_vec_size=128, block_fp8=True, stream=None,
                    ),
                    produce=produce,
                    owners=(weight, weight_scale),
                )
            requests.append(plan.request(
                name=f"linear.block_fp8.{prefix}.m{rows}",
                prepare_call=call,
                benchmark_call=call,
            ))
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="BLOCK_FP8",
                key=(prefix, tuple(sorted(plans))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return run_b12x_block_fp8_linear(
            A, As, B, Bs, self.config.out_dtype, self._b12x_block_fp8_owner.b12x_layer_name,
        )


def _tensor_fp8_plan(layer: torch.nn.Module, rows: int, out_dtype: torch.dtype):
    plans = layer.b12x_tensor_fp8_plans
    plan = plans.get(rows)
    if plan is None:
        api = _import_b12x_tensor_fp8()
        assert api is not None
        packed = layer.b12x_tensor_fp8_packed_weight
        n, k, padded_k = int(packed.out_features), int(packed.in_features), int(packed.padded_in_features)
        query = api.FixedBlockscaledQuery(
            recipe="tensor_fp8", call_kind="packed", max_rows=rows,
            in_features=k, padded_in_features=padded_k, out_features=n,
            input_dtype="float8_e4m3fn",
            output_dtype=str(out_dtype).removeprefix("torch."), expected_m=rows,
        )
        plan = api.plan(query)
        plans[rows] = plan
    return plan

def _b12x_tensor_fp8_linear(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    packed = layer.b12x_tensor_fp8_packed_weight
    plan = _tensor_fp8_plan(layer, int(source.shape[0]), out_dtype)
    tensor_fp8 = _import_b12x_tensor_fp8()
    assert tensor_fp8 is not None
    return tensor_fp8.mm(source, packed, plan=plan, bias=bias, out_dtype=out_dtype)


def _b12x_tensor_fp8_linear_fake(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    del bias, layer_name
    return source.new_empty((source.shape[0], out_features), dtype=out_dtype)


direct_register_custom_op(
    op_name="b12x_tensor_fp8_linear",
    op_func=_b12x_tensor_fp8_linear,
    fake_impl=_b12x_tensor_fp8_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_b12x_tensor_fp8_linear(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    out_dtype: torch.dtype,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.ops.vllm.b12x_tensor_fp8_linear(
        source, bias, out_features, out_dtype, layer_name
    )


def _apply_b12x_tensor_fp8_packed_linear(
    layer: torch.nn.Module,
    x_q: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed_weight = layer.b12x_tensor_fp8_packed_weight
    input_2d = x_q.reshape(-1, x_q.shape[-1]).contiguous()
    out_features = int(packed_weight.out_features)
    output_shape = [*x_q.shape[:-1], out_features]
    output = run_b12x_tensor_fp8_linear(
        input_2d, bias, out_features, out_dtype, layer.b12x_layer_name,
    )
    return output.view(*output_shape)


class B12xTensorFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Static per-tensor FP8 linear through the B12X SM12x dense GEMM."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x tensor FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x tensor FP8 kernels require a Blackwell 12x device"
        tensor_fp8 = _import_b12x_tensor_fp8()
        if tensor_fp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not tensor_fp8.is_supported():
            return False, "b12x.gemm.tensor_fp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        activation_scale = config.activation_quant_key.scale
        weight_scale = config.weight_quant_key.scale
        if (
            not activation_scale.static
            or not activation_scale.group_shape.is_per_tensor()
        ):
            return False, "requires static per-tensor activation scales"
        if not weight_scale.static or not weight_scale.group_shape.is_per_tensor():
            return False, "requires static per-tensor weight scales"
        if config.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "supports only bf16/fp16 input dtype"
        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "supports only bf16/fp16 output dtype"
        out_features, in_features = config.weight_shape
        if out_features <= 0 or in_features <= 0 or in_features % 32 != 0:
            return False, "weight dimensions must be positive with K divisible by 32"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale, input_scale, _ = self._get_layer_params(layer)
        assert weight.dtype == torch.float8_e4m3fn
        assert input_scale is not None
        assert weight_scale.numel() == input_scale.numel() == 1

        out_features, in_features = map(int, self.config.weight_shape)
        assert tuple(weight.shape) == (in_features, out_features)

        tensor_fp8 = _import_b12x_tensor_fp8()
        assert tensor_fp8 is not None
        output_scale = (
            input_scale.detach().to(torch.float32).reshape(1)
            * weight_scale.detach().to(torch.float32).reshape(1)
        ).contiguous()
        packed_weight = tensor_fp8.pack_weight(
            weight.detach().T.contiguous(),
            output_scale,
        )
        layer.b12x_tensor_fp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_tensor_fp8_packed_weight", None),
            packed_weight,
        )
        weight_name, weight_scale_name, _, _ = self.layer_param_names
        replace_parameter(layer, weight_name, weight.new_empty((0,)))
        replace_parameter(layer, weight_scale_name, weight_scale.new_empty((0,)))
        name = b12x_layer_prefix(layer)
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        layer.b12x_tensor_fp8_plans = {}
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
        self._b12x_tensor_fp8_owner = layer


    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload,
    ) -> Sequence[B12xPreparationUnit]:
        packed = layer.b12x_tensor_fp8_packed_weight
        if packed.values.is_meta:
            return ()
        prefix = _resolve_layer_name(layer.b12x_layer_name)
        requests = []
        for rows in workload.token_counts:
            plan = _tensor_fp8_plan(layer, rows, self.config.out_dtype)

            def call(state, *, m=rows):
                from b12x.preparation import PreparedCall

                source = torch.empty(
                    (m, int(packed.in_features)),
                    dtype=torch.float8_e4m3fn,
                    device=packed.values.device,
                )

                def produce() -> None:
                    indices = torch.arange(
                        source.numel(), device=source.device, dtype=torch.float32,
                    ).reshape_as(source)
                    source.copy_((indices.remainder(29).sub_(14)).mul_(1 / 32))

                return PreparedCall(
                    run=lambda: state.run_tensor_fp8(
                        source, packed.values, packed.scale_mma, packed.block_scale,
                        packed.output_scale, out_dtype=self.config.out_dtype,
                        stream=None,
                    ),
                    produce=produce,
                    owners=(
                        packed.values,
                        packed.scale_mma,
                        packed.block_scale,
                        packed.output_scale,
                    ),
                )
            requests.append(plan.request(
                name=f"linear.tensor_fp8.{prefix}.m{rows}",
                prepare_call=call,
                benchmark_call=call,
            ))
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="TENSOR_FP8",
                key=(prefix, tuple(sorted(plans))),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        _, _, input_scale, input_scale_ub = self._get_layer_params(layer)
        input_2d = x.reshape(-1, x.shape[-1])
        x_q, _ = self.quant_fp8(input_2d, input_scale, input_scale_ub)
        out_dtype = self.config.out_dtype
        output = _apply_b12x_tensor_fp8_packed_linear(
            layer,
            x_q,
            bias,
            out_dtype,
        )
        return output.view(*x.shape[:-1], output.shape[-1])

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        del A, B, out_dtype, As, Bs, bias, output_shape
        raise NotImplementedError("b12x tensor FP8 linear overrides apply_weights")


__all__ = [
    "B12xFp8BlockScaledMMKernel",
    "B12xTensorFP8ScaledMMLinearKernel",
]
