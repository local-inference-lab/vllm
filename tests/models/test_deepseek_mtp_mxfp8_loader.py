# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    _get_local_model_path,
    _try_load_fp8_linear_as_bf16,
)


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


def test_mtp_indexer_loader_falls_back_to_split_parameters(monkeypatch):
    prefix = "model.layers.78.mtp_block.self_attn.indexer"
    wk = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))
    weights_proj = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))

    def load_parameter(param, loaded_weight):
        param.data.copy_(loaded_weight)

    wk.weight_loader = load_parameter
    weights_proj.weight_loader = load_parameter
    params = {
        f"{prefix}.wk.weight": wk,
        f"{prefix}.weights_proj.weight": weights_proj,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=0, n_shared_experts=0),
        quant_config=None,
        model=SimpleNamespace(mtp_start_layer_idx=78, num_mtp_layers=1),
        named_parameters=lambda: params.items(),
        _rewrite_spec_layer_name=lambda _layer, name: name,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp."
        "rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp.fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp.get_pp_missing_layer_names",
        lambda _model: set(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp.get_spec_layer_idx_from_weight_name",
        lambda _config, name: 78 if ".layers.78." in name else None,
    )
    wk_weight = torch.full_like(wk, 1)
    weights_proj_weight = torch.full_like(weights_proj, 2)

    loaded = DeepSeekMTP.load_weights(
        model,
        [
            (f"{prefix}.wk.weight", wk_weight),
            (f"{prefix}.weights_proj.weight", weights_proj_weight),
        ],
    )

    assert torch.equal(wk, wk_weight)
    assert torch.equal(weights_proj, weights_proj_weight)
    assert loaded == set(params)


def test_mtp_serialized_probe_uses_target_revision_for_same_model(monkeypatch):
    calls: list[tuple[str | None, str | None]] = []

    def fake_resolve(model_path, revision=None):
        calls.append((model_path, revision))
        return "/cache/pinned-snapshot"

    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp._resolve_cached_hf_model_path",
        fake_resolve,
    )
    model = "org/model"
    config = PretrainedConfig()
    config._name_or_path = model
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            revision=None,
            draft_model_config=SimpleNamespace(
                model=model,
                model_path=None,
                model_weights=None,
                revision=None,
            ),
        ),
        model_config=SimpleNamespace(
            model=model,
            model_path=None,
            model_weights=None,
            revision="target-commit",
        ),
    )

    assert _get_local_model_path(config, vllm_config) == "/cache/pinned-snapshot"
    assert calls == [(model, "target-commit")]


def test_mtp_serialized_probe_prefers_explicit_draft_revision(monkeypatch):
    calls: list[tuple[str | None, str | None]] = []

    def fake_resolve(model_path, revision=None):
        calls.append((model_path, revision))
        return "/cache/draft-snapshot"

    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_mtp._resolve_cached_hf_model_path",
        fake_resolve,
    )
    model = "org/model"
    config = PretrainedConfig()
    config._name_or_path = model
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            revision="draft-commit",
            draft_model_config=SimpleNamespace(
                model=model,
                model_path=None,
                model_weights=None,
                revision="draft-commit",
            ),
        ),
        model_config=SimpleNamespace(
            model=model,
            model_path=None,
            model_weights=None,
            revision="target-commit",
        ),
    )

    assert _get_local_model_path(config, vllm_config) == "/cache/draft-snapshot"
    assert calls == [(model, "draft-commit")]
