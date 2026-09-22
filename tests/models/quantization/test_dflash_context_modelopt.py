# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context fusion follows the generic ModelOpt format after its refactor."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.modelopt import (
    build_linear_method,
    kMxfp8Static,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


def model_and_layers(methods):
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = nn.LayerNorm(32)
    model._fused_kv_linear = nn.Module()
    model._fused_kv_quant_method = None
    layers = []
    for index, method in enumerate(methods):
        mxfp8 = method is not None and method.spec.weight == kMxfp8Static
        weight = (torch.arange(192).reshape(6, 32) / 100 + index).to(
            torch.float8_e4m3fn if mxfp8 else torch.float32
        )
        projection = SimpleNamespace(
            quant_method=method,
            weight=weight,
            q_size=2,
            weight_scale=torch.full((6, 1), 127 + index, dtype=torch.uint8),
            bias=torch.arange(6, dtype=torch.float32) + index,
        )
        layers.append(
            SimpleNamespace(qkv_proj=projection, q_size=2, k_norm=nn.LayerNorm(2))
        )
    model.layers = [SimpleNamespace(self_attn=layer) for layer in layers]
    return model, layers


def method(algo):
    return build_linear_method(SimpleNamespace(), algo, "draft")


@pytest.mark.parametrize("bias", [False, True])
def test_unquantized_context_fusion_and_postload(bias):
    model, layers = model_and_layers([None, None])
    model._build_context_kv_buffers(layers, bias)
    inputs = torch.arange(96, dtype=torch.float32).reshape(3, 32) / 10
    expected = torch.cat(
        [
            torch.nn.functional.linear(
                inputs, a.qkv_proj.weight[2:], a.qkv_proj.bias[2:] if bias else None
            )
            for a in layers
        ],
        dim=-1,
    )
    actual = torch.nn.functional.linear(
        inputs, model._fused_kv_weight, model._fused_kv_bias
    )
    torch.testing.assert_close(actual, expected)
    assert model._fused_kv_weight_scale is None
    model.process_weights_after_loading()
    assert model._fused_kv_quant_method is None


@pytest.mark.parametrize("mixed_index", [0, 1])
def test_mixed_mxfp8_is_rejected_before_reading_weights(mixed_index):
    methods = [None, None]
    methods[mixed_index] = method("MXFP8")
    model, layers = model_and_layers(methods)
    for layer in layers:
        del layer.qkv_proj.weight
    with pytest.raises(ValueError, match="Every DFlash attention layer"):
        model._build_context_kv_buffers(layers, False)


def test_mxfp8_scales_and_backend_are_preserved():
    quant = method("MXFP8")
    model, layers = model_and_layers([quant, quant])
    model._build_context_kv_buffers(layers, False)
    expected = torch.cat([a.qkv_proj.weight[2:].float() for a in layers])
    scales = torch.cat([a.qkv_proj.weight_scale[2:] for a in layers])
    observed = []
    quant.kernel = SimpleNamespace(
        process_weights_after_loading=lambda layer: observed.append(
            (layer.weight.float().clone(), layer.weight_scale.clone())
        )
    )
    model.process_weights_after_loading()
    torch.testing.assert_close(observed[0][0], expected)
    torch.testing.assert_close(observed[0][1], scales)
    assert model._fused_kv_quant_method is quant
    assert model._fused_kv_weight is None and model._fused_kv_weight_scale is None


def test_other_modelopt_format_is_not_mxfp8():
    model, layers = model_and_layers([method("FP8"), method("FP8")])
    model._build_context_kv_buffers(layers, False)
    assert model._fused_kv_weight_scale is None
    model.process_weights_after_loading()
    assert model._fused_kv_quant_method is None
