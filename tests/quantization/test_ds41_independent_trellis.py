# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.models.deepseek_v4_1.trellis import (
    DeepseekV41TrellisConfig,
    IndependentTrellisMoEMethod,
)


def test_independent_config_preserves_native_dense_scale_contract():
    config = DeepseekV41TrellisConfig.from_config(
        {
            "format_version": 1,
            "manifest": "trellis-manifest.json",
            "bits": 2,
            "codebook": "lut_e4m3",
        }
    )
    assert config.is_scale_e8m0
    assert config.weight_block_size == [32, 32]
    assert config.is_checkpoint_fp8_serialized
    with pytest.raises(ValueError, match="independent K2"):
        DeepseekV41TrellisConfig.from_config({"bits": 3})


@pytest.mark.parametrize(
    "model_type,expected", [("deepseek_v41", "trellis_dense"), ("llama", None)]
)
def test_override_only_selects_independent_deepseek_checkpoint(model_type, expected):
    hf = SimpleNamespace(model_type=model_type)
    assert (
        DeepseekV41TrellisConfig.override_quantization_method(
            {"quant_method": "trellis_dense"}, None, hf
        )
        == expected
    )
    assert DeepseekV41TrellisConfig.override_quantization_method(None, None, hf) is None


def test_deferred_experts_do_not_register_missing_checkpoint_parameters():
    moe = FusedMoEConfig(
        num_experts=384,
        num_local_experts=384,
        num_logical_experts=384,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size=1152,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = IndependentTrellisMoEMethod(moe)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.ffn.experts"
    method.create_weights(layer, 384, 5120, 1152, torch.bfloat16)
    assert not list(layer.named_parameters())
    assert not layer.state_dict()
    assert layer.w13_weight.numel() == layer.w2_weight.numel() == 0
    assert method.layer_index == 17 and method.local_intermediate == 1152
    assert method.moe.swiglu_limit == 10.0
    assert (
        method.get_fused_moe_quant_config(layer).weight_quant_dtype == "trellis_dense"
    )
