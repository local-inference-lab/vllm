# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import linear
from vllm.utils.b12x import b12x_layer


def make_layer(shape, prefix):
    layer = linear.LinearBase(4, 8, disable_tp=True, prefix=prefix)
    layer.weight = torch.nn.Parameter(torch.ones(shape, dtype=torch.bfloat16))
    layer.bias = None
    layer.quant_method._use_b12x = True
    return layer


@pytest.fixture
def b12x_platform(monkeypatch):
    from b12x.gemm import bf16_gemv

    monkeypatch.setattr(
        linear,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_cpu=lambda: False, is_xpu=lambda: False
        ),
    )
    monkeypatch.setattr(bf16_gemv, "is_supported", lambda device: True)


def test_convolution_weight_holder_does_not_declare_gemm(b12x_platform):
    layer = make_layer((8, 1, 4), "mamba.conv1d")
    layer.quant_method.process_weights_after_loading(layer)
    assert not hasattr(layer, "_b12x_unquantized")
    assert not hasattr(layer, "b12x_preparation_provider")


@pytest.mark.parametrize("prefix", ["", "same.linear"])
def test_projection_owners_have_distinct_names(b12x_platform, prefix):
    layers = [make_layer((8, 4), prefix), make_layer((8, 4), prefix)]
    for layer in layers:
        layer.quant_method.process_weights_after_loading(layer)
    first, second = (layer._b12x_unquantized for layer in layers)
    assert first.name != second.name
    assert b12x_layer(first.name) is layers[0]
    assert b12x_layer(second.name) is layers[1]
