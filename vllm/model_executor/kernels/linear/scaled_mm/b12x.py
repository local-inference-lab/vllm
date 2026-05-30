# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Fp8BlockScaledMMLinearKernel,
)


def _import_b12x_block_fp8():
    try:
        return importlib.import_module("b12x.gemm.block_fp8_linear")
    except ImportError:
        return None


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


def _empty_plan_scratch(
    plan,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )


def _b12x_fp8_block_scaled_linear_op(
    input_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Stateless block-FP8 linear via the b12x SM120 MXFP8 GEMM.

    Takes the pre-packed weight tensors directly (no layer / prefix /
    no_compile_layers lookup) and RETURNS the [tokens, out_features] result. It is
    a functional op (no mutated args), like the other FP8 kernels and the WO op,
    so it composes with torch.compile / piecewise cudagraphs without per-layer
    registration and without an auto_functionalized re-inplacing pass.
    out_features / in_features are derived from the weight shape [N, K].
    """
    block_fp8 = _import_b12x_block_fp8()
    if block_fp8 is None:
        raise ImportError("b12x.gemm.block_fp8_linear is not importable")

    tokens = int(input_2d.shape[0])
    out_features = int(weight_values.shape[0])
    if tokens == 0:
        # Empty batch: return an empty result here, in the op body (runtime,
        # opaque to the tracer), instead of branching on the dynamic token dim in
        # apply_weights -- a traced `shape[0] == 0` guard would force the batch
        # symint to be a graph input and break the MTP/fullgraph compile.
        return input_2d.new_empty((0, out_features))
    in_features = int(weight_values.shape[1])
    packed_weight = block_fp8.BlockFP8LinearWeight(
        weight=block_fp8.MXFP8Rows(
            values=weight_values,
            scale_rows=weight_scale_rows,
            scale_mma=weight_scale_mma,
        ),
        in_features=in_features,
        out_features=out_features,
        block_size=(128, 128),
    )
    output = torch.empty(
        (tokens, out_features), dtype=input_2d.dtype, device=input_2d.device
    )
    plan = block_fp8.plan_block_fp8_linear_scratch(
        block_fp8.BlockFP8LinearScratchCaps(
            device=input_2d.device,
            max_tokens=tokens,
            in_features=in_features,
            out_features=out_features,
            output_dtype=input_2d.dtype,
        )
    )
    scratch = _empty_plan_scratch(plan, input_2d.device)
    # DeepGEMM-style regime hint: this binding is built per-forward / per-capture
    # with the live token count, so declare it as expected_m -- the b12x tile
    # selector then picks the decode tile (32x128) for wide-N (N>1536) FP8 linears
    # at small M and the prefill tile (64x128) at large M (e.g. dense-MLP
    # down-projections in DeepSeek-V4 decode). expected_m only selects the tile
    # (not a compile-cache key), and each cudagraph-captured batch size warms its
    # own tile, so this is safe (vLLM does not freeze b12x kernel resolution).
    binding = plan.bind(
        scratch=scratch,
        source=input_2d,
        packed_weight=packed_weight,
        output=output.view(tokens, out_features, 1),
        bias=bias,
        expected_m=tokens,
    )
    block_fp8.block_fp8_linear_mxfp8(binding=binding)
    return output


def _b12x_fp8_block_scaled_linear_op_fake(
    input_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_rows: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Use the symbolic shapes directly -- do NOT int(input_2d.shape[0]); that
    # specializes the dynamic token dim to the trace's example size (e.g. 2048)
    # and violates dynamic-shape constraints. weight_values.shape[0] is the
    # static out_features dim. (out_features, in_features) = weight_values.shape.
    return input_2d.new_empty((input_2d.shape[0], weight_values.shape[0]))


direct_register_custom_op(
    op_name="b12x_fp8_block_scaled_linear",
    op_func=_b12x_fp8_block_scaled_linear_op,
    fake_impl=_b12x_fp8_block_scaled_linear_op_fake,
)


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Block-FP8 linear through the native b12x SM120 MXFP8 GEMM path."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x FP8 kernels require a Blackwell 12x device"
        if _import_b12x_block_fp8() is None:
            return False, "b12x.gemm.block_fp8_linear is not importable"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if _current_linear_backend() != "b12x" and not envs.VLLM_USE_B12X_FP8_GEMM:
            return False, "b12x FP8 GEMM is not enabled"

        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 output dtype"

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per-token group activation "
                "quantization with group_shape=(1,128).",
            )

        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"

        out_features, in_features = config.weight_shape
        if in_features % 128 != 0 or out_features <= 0:
            return False, "Input features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            layer.b12x_packed_weight = None
            return

        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        if weight_scale is None:
            raise ValueError("b12x FP8 linear requires block weight scales")
        assert layer.weight_block_size is not None

        block_fp8 = _import_b12x_block_fp8()
        if block_fp8 is None:
            raise ImportError("b12x.gemm.block_fp8_linear is not importable")
        layer.b12x_packed_weight = block_fp8.pack_block_fp8_linear_weight_mxfp8(
            params.weight.detach(),
            weight_scale.detach(),
            block_size=tuple(layer.weight_block_size),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            raise RuntimeError(
                "b12x generic FP8 linear was called for a layer owned by a fused "
                "b12x path"
            )
        packed_weight = getattr(layer, "b12x_packed_weight", None)
        if packed_weight is None:
            raise RuntimeError(
                "b12x FP8 packed weights are missing; process_weights_after_loading "
                "did not run for this layer"
            )
        out_features = int(packed_weight.out_features)
        input_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output_shape = [*x.shape[:-1], out_features]
        # NB: do NOT branch on input_2d.shape[0] here (the dynamic token dim) -- a
        # traced `== 0` guard forces the batch symint to become a graph input and
        # breaks the MTP/fullgraph compile (copy_misaligned_inputs: got int). The
        # empty-batch case is handled inside the op body (runtime, untraced).
        if input_2d.dtype != self.config.out_dtype:
            raise RuntimeError(
                "b12x FP8 linear currently expects input and output dtype to "
                f"match, got input={input_2d.dtype}, output={self.config.out_dtype}"
            )
        # Stateless functional op: pass the pre-packed weight tensors directly (no
        # layer / prefix / no_compile_layers lookup) and get the result back,
        # matching the other FP8 kernels and the WO op. Composes with
        # torch.compile / piecewise cudagraphs as-is (no auto_functionalized).
        output = torch.ops.vllm.b12x_fp8_block_scaled_linear(
            input_2d,
            packed_weight.weight.values,
            packed_weight.weight.scale_rows,
            packed_weight.weight.scale_mma,
            bias,
        )
        return output.view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        del A, B, As, Bs
        raise NotImplementedError("b12x FP8 linear overrides apply_weights")
