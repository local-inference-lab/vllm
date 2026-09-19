# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source ownership and byte preservation for opt-in expert caching."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.b12x_cache import (
    ModelOptNvFp4CacheMoE,
    cache_provider,
)


@pytest.fixture(autouse=True)
def single_rank_parameters(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )


def method_and_layer():
    method = object.__new__(ModelOptNvFp4CacheMoE)
    method.quant_config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True, group_size=16
    )
    method.moe = SimpleNamespace(is_act_and_mul=True)
    method.use_global_sf = False
    method.provider = SimpleNamespace(model=Mock(checkpoint_id="a" * 64))
    method.prefix = "model.layers.0.mlp.experts"
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    layer.activation = SimpleNamespace(value="silu")
    return method, layer


def test_cpu_allocation_overrides_ambient_device_and_preserves_sources(monkeypatch):
    method, layer = method_and_layer()

    def forbidden(*args, **kwargs):
        raise AssertionError("source allocation initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.device("meta"):
        method.create_weights(layer, 4, 128, 128, torch.bfloat16)
    assert all(p.device.type == "cpu" for p in layer.parameters())
    method.provider.model.reserve_source.assert_called_once_with(
        4 * (3 * 128 * 128 // 2 + 3 * 128 * 128 // 16 + 24)
    )
    for parameter in layer.parameters():
        parameter.data.fill_(1)
    before = {name: value.clone() for name, value in layer.named_parameters()}
    method.process_weights_after_loading(layer)
    source = method.provider.model.add_source.call_args.args[0]
    assert source.weights.layer_name == method.prefix
    assert source.plan.source.w13_layout.value == "w31"
    assert source.plan.activation.mode.value == "a16"
    assert source.weights.w13.data_ptr() == layer.w13_weight.data_ptr()
    for name, value in layer.named_parameters():
        assert torch.equal(value.view(torch.uint8), before[name].view(torch.uint8))
    assert len(source.owners) == len(before)


def test_loader_rejects_global_scale_reconciliation():
    method, layer = method_and_layer()
    method.create_weights(layer, 4, 128, 128, torch.bfloat16)
    layer.w13_weight_scale_2.data.fill_(1)
    layer.w13_weight_scale_2.data[0, 1] = 2
    with pytest.raises(ValueError, match="never requantizes"):
        method.process_weights_after_loading(layer)
    method.provider.model.add_source.assert_not_called()


def test_custom_additional_config_does_not_enable_cache():
    config = SimpleNamespace(additional_config=object())
    assert cache_provider(config) is None
