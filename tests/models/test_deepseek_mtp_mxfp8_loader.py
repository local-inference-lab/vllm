# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.models.deepseek_mtp import _try_load_fp8_linear_as_bf16


def _write_serialized_nextn_index(model_dir, layer: int) -> None:
    prefix = f"model.layers.{layer}.mlp.experts.0.down_proj"
    required = {
        f"{prefix}.weight": "model.safetensors",
        f"{prefix}.weight_scale": "model.safetensors",
        f"{prefix}.weight_scale_2": "model.safetensors",
        f"{prefix}.input_scale": "model.safetensors",
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": required})
    )


def test_glm_nextn_keeps_serialized_quantized_modules_targeted_by_config(
    tmp_path,
):
    _write_serialized_nextn_index(tmp_path, layer=2)
    hf_config = PretrainedConfig()
    hf_config.architectures = ["Glm4MoeForCausalLM"]
    hf_config.model_type = "glm_moe_dsa"
    hf_config.num_hidden_layers = 2
    hf_config.num_nextn_predict_layers = 1
    hf_config._name_or_path = str(tmp_path)
    hf_config.quantization_config = {
        "ignore": [],
        "quantized_layers": {
            "model.layers.2.self_attn.fused_qkv_a_proj": {},
            "model.layers.2.mlp.shared_experts.gate_up_proj": {},
        },
    }

    SpeculativeConfig.hf_config_override(hf_config)

    ignored = hf_config.quantization_config["ignore"]
    assert "model.layers.2.self_attn*" not in ignored
    assert "model.layers.2.mlp.shared_experts*" not in ignored
    assert "model.layers.2.eh_proj*" in ignored
    assert hf_config.model_type == "deepseek_mtp"


def test_mtp_fallback_loader_accepts_mxfp8_weight_scale():
    weight_bf16 = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    param = torch.nn.Parameter(torch.empty_like(weight_bf16, dtype=torch.bfloat16))
    params = {"model.layers.78.self_attn.fused_qkv_a_proj.weight": param}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()

    assert _try_load_fp8_linear_as_bf16(
        "model.layers.78.self_attn.fused_qkv_a_proj.weight",
        weight_fp8,
        pending,
        params,
        loaded,
    )
    assert _try_load_fp8_linear_as_bf16(
        "model.layers.78.self_attn.fused_qkv_a_proj.weight_scale",
        scales,
        pending,
        params,
        loaded,
    )

    expected = dequant_mxfp8_to_bf16(weight_fp8, scales)
    assert torch.equal(param.data, expected)
    assert "model.layers.78.self_attn.fused_qkv_a_proj.weight" in loaded
    assert pending == {}
