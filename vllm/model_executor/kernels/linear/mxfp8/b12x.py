# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xWorkload,
    b12x_layer_prefix,
    get_b12x_dense_activation_mode,
    register_b12x_layer,
    reuse_packed_weight_storage,
    run_b12x_blockscaled_linear,
    set_b12x_preparation_provider,
)
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.torch_utils import _encode_layer_name

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class B12xMxfp8LinearKernel(Mxfp8LinearKernel):
    """ModelOpt MXFP8 linear through a layer-held prepared b12x plan."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x MXFP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x MXFP8 kernels require a Blackwell 12x device"
        api = _import_b12x_blockscaled()
        if api is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not api.is_supported():
            return False, "b12x.gemm.blockscaled is not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig):
        del config
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        scales = layer.weight_scale.data
        assert weight.dtype == MXFP8_VALUE_DTYPE and weight.ndim == 2
        assert scales.dtype == MXFP8_SCALE_DTYPE and scales.ndim == 2
        out_features, in_features = map(int, weight.shape)
        assert in_features % MXFP8_BLOCK_SIZE == 0
        api = _import_b12x_blockscaled()
        assert api is not None
        packed = api.pack_weight(
            weight.detach(),
            scales[:out_features, : in_features // MXFP8_BLOCK_SIZE].detach(),
        )
        # Both B12X activation-precision paths read the MMA-layout scales.
        # The row-layout copy is packaging metadata, not an execution input.
        packed = replace(packed, weight=replace(packed.weight, scale_rows=None))
        layer.b12x_mxfp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_mxfp8_packed_weight", None), packed
        )
        # A method that pre-quantizes or keeps BF16 activations sets the mode
        # before weight processing; the configured default applies otherwise.
        layer.b12x_activation_mode = getattr(
            layer, "b12x_activation_mode", None
        ) or get_b12x_dense_activation_mode("mxfp8")
        layer.b12x_bf16_input_supported = (
            in_features % 128 == 0 and out_features % 8 == 0
        )
        name = b12x_layer_prefix(layer)
        # A reload into the same packed storage keeps the holder and its
        # prepared plan; new storage declares anew.
        existing = getattr(layer, "b12x_linear", None)
        if existing is None or not existing.holds(layer.b12x_mxfp8_packed_weight):
            layer.b12x_linear = B12xBlockscaledLinear(
                layer.b12x_mxfp8_packed_weight,
                recipe="mxfp8",
                activation_mode=layer.b12x_activation_mode,
                layer_name=name,
            )
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        replace_parameter(layer, "weight", weight.new_empty((0,)))
        replace_parameter(layer, "weight_scale", scales.new_empty((0,)))
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[object]:
        packed = layer.b12x_mxfp8_packed_weight
        if packed.weight.values.is_meta:
            return ()
        linear = layer.b12x_linear
        return (linear.unit(workload, name=f"linear.mxfp8.{linear.layer_name}"),)

    def get_workspace_size(self, layer: torch.nn.Module, rows: int) -> int:
        linear = getattr(layer, "b12x_linear", None)
        return 0 if linear is None else linear.get_workspace_size(rows)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = x.reshape(-1, x.shape[-1]).contiguous()
        if source.dtype != torch.bfloat16:
            raise ValueError("prepared vLLM MXFP8 path requires BF16 activations")
        out_features = int(layer.b12x_mxfp8_packed_weight.out_features)
        output = run_b12x_blockscaled_linear(
            source, bias, out_features, layer.b12x_layer_name
        )
        return output.view(*x.shape[:-1], out_features)


__all__ = ["B12xMxfp8LinearKernel"]
