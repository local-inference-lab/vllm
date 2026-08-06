# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config


def test_kimi_k3_nested_text_and_quantization_config() -> None:
    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
    }
    config = KimiK3Config(
        text_config={
            "model_type": "kimi_linear",
            "hidden_size": 7168,
            "routed_expert_hidden_size": 3584,
            "moe_intermediate_size": 3072,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "quantization_config": quantization_config,
        },
        quantization_config=None,
    )

    assert config.hidden_size == 7168
    assert config.text_config.routed_expert_hidden_size == 3584
    assert config.text_config.moe_intermediate_size == 3072
    assert config.text_config.hidden_act == "situ"
    assert config.quantization_config == quantization_config
