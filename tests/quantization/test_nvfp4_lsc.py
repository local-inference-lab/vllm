# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed expert storage must retain nonexpert precision and tensors."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.nvfp4_lsc import (
    Nvfp4LscConfig,
    Nvfp4LscMoEMethod,
)
from vllm.model_executor.model_loader.nvfp4_lsc_loader import Nvfp4LscModelLoader


def test_lsc_config_preserves_source_recipes_and_avoids_expert_allocations():
    original = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.45.mlp.experts": {"quant_algo": "MXFP8"},
        },
    }
    owner = Nvfp4LscConfig.from_config(
        {
            "format_version": 1,
            "checkpoint_root": "/lsc",
            "source_quantization_config": original,
        }
    )
    assert owner._resolve_quant_algo("model.layers.3.mlp.experts") == "NVFP4"
    assert owner._resolve_quant_algo("model.layers.45.mlp.experts") == "MXFP8"
    moe = FusedMoEConfig(
        num_experts=288,
        num_local_experts=288,
        num_logical_experts=288,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size=1024,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = Nvfp4LscMoEMethod(moe, owner)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    method.create_weights(layer, 288, 4096, 1024, torch.bfloat16)
    assert not list(layer.parameters()) and not layer.state_dict()
    assert layer.w13_weight.numel() == layer.w2_weight.numel() == 0
    with pytest.raises(ValueError, match="source_quantization_config"):
        Nvfp4LscConfig.from_config({"format_version": 1, "checkpoint_root": "/lsc"})


def test_prepared_nvfp4_accepts_kernel_order_and_rejects_activation_change(monkeypatch):
    from b12x.moe import fused_moe

    from vllm.model_executor.layers.fused_moe import b12x

    moe = FusedMoEConfig(
        num_experts=8,
        num_local_experts=8,
        num_logical_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=128,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    one = torch.ones(8)
    quant = nvfp4_moe_quant_config(one, one, one, one, one, one, gemm1_clamp_limit=10.0)
    backend = b12x.B12xExperts(moe, quant)
    layer = torch.nn.Module()
    layer.activation, layer.apply_router_weight_on_input = moe.activation, False
    prepared = Mock(spec=fused_moe.PreparedExperts)
    activation = fused_moe.ActivationSpec(
        mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
    )
    prepared.plan = SimpleNamespace(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=activation,
    )
    prepared.num_experts, prepared.hidden_size, prepared.intermediate_size = 8, 256, 128
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *a, **k: None
    )
    backend.install_prepared_experts(layer, prepared)
    assert backend._prepared_experts is prepared
    prepared.plan.activation = replace(activation, mode="a16")
    with pytest.raises(ValueError, match="activation"):
        backend.install_prepared_experts(layer, prepared)


def test_loader_excludes_compressed_main_experts_and_retains_native_tensors(tmp_path):
    from b12x.moe.checkpoints.nvfp4_lsc import CODEC, SCHEMA

    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import get_model_loader

    directory = tmp_path / "tensors"
    directory.mkdir()
    filename = "weights.safetensors"
    tensors = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        (
            "model.language_model.layers.3.mlp.experts.0.gate_proj."
            "weight_scale.nvfp4_lsc_fixed"
        ): torch.ones(8, dtype=torch.uint8),
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight": torch.full(
            (8,), 42, dtype=torch.uint8
        ),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, directory / filename)
    common = {"schema": SCHEMA, "codec": CODEC, "family": "glm53_nvfp4"}
    (tmp_path / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": filename}]})
    )
    (tmp_path / "build-contract.json").write_text(
        json.dumps({**common, "source_names": {k: filename for k in tensors}})
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": "nvfp4_lsc",
                "checkpoint_root": str(tmp_path),
            }
        ),
        hf_text_config=SimpleNamespace(num_hidden_layers=45),
    )
    loader = get_model_loader(LoadConfig(load_format="nvfp4_lsc"))
    assert isinstance(loader, Nvfp4LscModelLoader)
    assert get_quantization_config("nvfp4_lsc") is Nvfp4LscConfig
    actual = dict(loader.get_all_weights(config, SimpleNamespace()))
    assert set(actual) == {name for name in tensors if ".layers.3." not in name}
    for name, tensor in actual.items():
        assert torch.equal(tensor, tensors[name])
