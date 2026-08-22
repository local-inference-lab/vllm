# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.interfaces import supports_lora
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    _make_deepseek_v4_weights_mapper,
)


def test_deepseek_v4_declares_lora_layout() -> None:
    assert supports_lora(DeepseekV4ForCausalLM)
    assert DeepseekV4ForCausalLM.is_3d_moe_weight
    assert DeepseekV4ForCausalLM.lora_skip_prefixes == ["mtp."]
    assert DeepseekV4ForCausalLM.packed_modules_mapping == {
        "fused_wqa_wkv": ["q_a_proj", "kv_proj"],
        "wq_b": ["q_b_proj"],
        "wo_b": ["o_b_proj"],
        "fused_wkv_wgate": ["kv_proj", "gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


def test_deepseek_v4_maps_transformers_lora_parents() -> None:
    mapper = _make_deepseek_v4_weights_mapper("fp4")

    assert mapper._map_name(
        "layers.0.self_attn.q_a_proj.lora_A.weight"
    ) == "model.layers.0.attn.q_a_proj.lora_A.weight"
    assert mapper._map_name(
        "layers.0.mlp.experts.base_layer.lora_A.weight"
    ) == "model.layers.0.ffn.experts.base_layer.lora_A.weight"
    assert mapper._map_name(
        "layers.0.mlp.shared_experts.gate_proj.lora_A.weight"
    ) == "model.layers.0.ffn.shared_experts.gate_proj.lora_A.weight"
