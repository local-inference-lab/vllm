# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.model_loader.exact_mxfp4_loader import ExactMXFP4ModelLoader
from vllm.models.deepseek_v4_1.exact_mxfp4 import (
    DeepseekV41ExactMXFP4Config,
    ExactMXFP4MoEMethod,
)


def test_exact_config_keeps_native_dense_precision_and_empty_expert_handles():
    owner = DeepseekV41ExactMXFP4Config.from_config(
        {"format_version": 1, "checkpoint_root": "/x4t"}
    )
    assert owner.is_scale_e8m0 and owner.weight_block_size == [32, 32]
    assert owner.is_checkpoint_fp8_serialized and owner.scale_scratch is None
    moe = FusedMoEConfig(
        num_experts=384,
        num_local_experts=384,
        num_logical_experts=384,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size=576,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = ExactMXFP4MoEMethod(moe, owner)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.ffn.experts"
    method.create_weights(layer, 384, 5120, 576, torch.bfloat16)
    assert not list(layer.named_parameters()) and not layer.state_dict()
    assert method.layer_index == 17 and method.local_intermediate == 576
    assert method.get_fused_moe_quant_config(layer).weight_quant_dtype == "mxfp4"
    with pytest.raises(ValueError, match="format_version"):
        DeepseekV41ExactMXFP4Config.from_config({})


def test_loader_preserves_file_backed_engram_and_native_draft(tmp_path):
    from b12x.moe.checkpoints.exact_mxfp4 import CODEC, SCHEMA

    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    name = "model-00001.safetensors"
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": torch.ones(8, dtype=torch.uint8),
        "layers.0.ffn.experts.0.w1.scale.exact_mxfp4_fixed": torch.ones(
            8, dtype=torch.uint8
        ),
        "layers.40.ffn.experts.0.w1.weight": torch.full((8,), 42, dtype=torch.uint8),
        "layers.1.engram.embedding.weight": torch.ones((8, 8), dtype=torch.uint8),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / name)
    common = {"schema": SCHEMA, "codec": CODEC, "family": "deepseek_v41"}
    (tmp_path / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": name}]})
    )
    (tmp_path / "build-contract.json").write_text(
        json.dumps({**common, "source_names": {k: name for k in tensors}})
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=40,
            quantization_config={
                "quant_method": "exact_mxfp4",
                "checkpoint_root": str(tmp_path),
            },
        )
    )
    model = SimpleNamespace(
        checkpoint_file_weight_filter=lambda n: ".engram.embedding." in n
    )
    loader = ExactMXFP4ModelLoader(LoadConfig(load_format="exact_mxfp4"))
    result = dict(loader.get_all_weights(config, model))
    assert len(result) == 3
    assert result["layers.1.engram.embedding.weight"].device.type == "meta"
    assert result[
        "layers.1.engram.embedding.weight"
    ]._vllm_file_tensor_source.shape == (8, 8)
    assert torch.equal(
        result["layers.40.ffn.experts.0.w1.weight"],
        tensors["layers.40.ffn.experts.0.w1.weight"],
    )
    model.checkpoint_weight_name_prefixes = ("layers.40.",)
    assert list(dict(loader.get_all_weights(config, model))) == [
        "layers.40.ffn.experts.0.w1.weight"
    ]
