# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    B12xNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import _Fp8OnlineLinearBase
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kNvfp4Dynamic,
    kNvfp4Static,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_blockscaled

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class Nvfp4OnlineLinearMethod(_Fp8OnlineLinearBase):
    """Online NVFP4 draft head with selectable activation precision."""

    def __init__(self, *, use_a16: bool = False):
        super().__init__()
        supported, reason = B12xNvFp4LinearKernel.is_supported()
        if not supported:
            raise ValueError(f"Online NVFP4 draft head requires b12x: {reason}")
        self.kernel = B12xNvFp4LinearKernel(NvFp4LinearLayerConfig())
        self.use_a16 = use_a16
        if use_a16 and self.input_dtype != torch.bfloat16:
            raise ValueError("A16 LM heads require BF16 activations")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight = layer.weight.contiguous()
        if weight.shape[1] % 16:
            raise ValueError("Online NVFP4 head requires K divisible by 16")
        amax = weight.abs().amax().float().clamp_min(1e-8)
        global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
        packed, scales = scaled_fp4_quant(
            weight, global_scale, is_sf_swizzled_layout=False
        )
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale", scales)
        replace_parameter(layer, "weight_global_scale", global_scale.reciprocal())
        replace_parameter(layer, "input_global_scale_inv", torch.ones_like(amax))
        replace_parameter(layer, "alpha", layer.weight_global_scale.clone())
        self.kernel.process_weights_after_loading(layer)
        layer.b12x_activation_mode = "a16" if self.use_a16 else "quantized"
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_a16:
            return self.kernel.apply_weights(layer, x, bias)
        amax = x.abs().amax().float().clamp_min(1e-8)
        input_scale = amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
        x_packed, x_scale = scaled_fp4_quant(
            x.reshape(-1, x.shape[-1]),
            input_scale.reciprocal(),
            is_sf_swizzled_layout=True,
        )
        blockscaled = get_b12x_blockscaled()
        assert blockscaled is not None
        output = blockscaled.mm_nvfp4(
            x_packed,
            x_scale,
            layer.weight,
            layer.weight_scale,
            input_scale * layer.weight_global_scale,
            out_dtype=x.dtype,
        )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[0])


def _quantize_moe_weight_to_nvfp4(
    weight: torch.Tensor,
    moe_tp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights ``(E, N, K)`` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales,
    matching the ModelOpt NVFP4 checkpoint layout. Returns packed FP4 weights
    ``(E, N, K // 2)``, block scales ``(E, N, K // 16)``, and the per-expert
    global scale ``(E,)`` stored as ``amax / (fp4_max * fp8_max)``.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    k = weight.shape[-1]
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight_amax(weight.flatten(1), dim=-1).to(torch.float32)
    amax = amax_for_moe_weight_quant(amax, moe_tp_size).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # Keep the original BF16/FP16 values as the quantizer input. Folding each
    # expert's FP32 global scale into the weight would add a BF16/FP16 rounding
    # before the group-16 scale and E2M1 values are selected.
    weight = weight.contiguous()
    quantized_experts = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_scale,
            strict=True,
        )
    ]
    qweight = torch.stack([quantized for quantized, _ in quantized_experts])
    block_scale = torch.stack([block_scale for _, block_scale in quantized_experts])
    return (
        qweight,
        block_scale,
        weight_scale_2,
    )


class Nvfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """Online NVFP4 MoE quantization with per-token activation scales.

    Quantizes fp16/bf16 expert weights to NVFP4 at load time; the FlashInfer
    TRTLLM kernel computes per-token activation scales at runtime. Blackwell
    (SM100) only.
    """

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        if not current_platform.is_device_capability_family(100):
            raise ValueError(
                "nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
            )
        super().__init__(layer.moe_config)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        self._setup_kernel(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(self, layer: Module) -> None:
        moe_tp_size = self.moe.tp_size
        w13, w13_scale, w13_scale_2 = _quantize_moe_weight_to_nvfp4(
            layer.w13_weight, moe_tp_size
        )
        w2, w2_scale, w2_scale_2 = _quantize_moe_weight_to_nvfp4(
            layer.w2_weight, moe_tp_size
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)

        # Neutral (1.0) activation global scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        ones = torch.ones(layer.num_experts, device=w13.device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=True,
            )

        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )
