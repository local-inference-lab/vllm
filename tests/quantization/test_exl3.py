# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3MoEParameter,
)
from vllm.model_executor.models import glm4_moe


def _rank_sliced_metadata(**overrides):
    metadata = {
        "format": "exl3-trellis",
        "bits": 3.0,
        "codebook": "mcg",
        "experts_per_layer": 256,
        "moe_layers": [3, 77],
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        ),
        "tp": 4,
    }
    metadata.update(overrides)
    return metadata


def test_rank_sliced_checkpoint_selects_exl3_override():
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())

    assert get_quantization_config("exl3") is Exl3Config
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, None, hf_config
        )
        == "exl3"
    )
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, "fp8", hf_config
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"codebook": "mul1"}, "MCG codebook"),
        ({"moe_layers": [77, 3]}, "moe_layers"),
        ({"tensor_schema": "unsupported"}, "tensor schema"),
    ],
)
def test_rank_sliced_metadata_fails_closed(overrides, message):
    config = Exl3Config()
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata(**overrides))

    with pytest.raises(ValueError, match=message):
        config.maybe_update_config("unused", hf_config)


def test_rank_sliced_metadata_admits_only_declared_moe_layers():
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )

    assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
    assert config._moe_prefix_is_exl3("model.layers.77.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.2.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.78.mlp.experts")
    assert (
        config.codebook_for_prefix("model.layers.10.mlp.experts.0.gate_proj") == "mcg"
    )


def test_rank_sliced_weight_name_keeps_only_local_tp_rank(monkeypatch):
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    prefix = "model.layers.3.mlp.experts.17.gate_proj"

    assert (
        config.normalize_rank_sliced_weight_name(f"{prefix}.rank2.trellis")
        == f"{prefix}.trellis"
    )
    assert config.normalize_rank_sliced_weight_name(f"{prefix}.rank1.trellis") is None
    assert (
        config.normalize_rank_sliced_weight_name("model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_glm_model_normalizes_rank_sliced_weights_before_auto_loading(monkeypatch):
    observed = []

    class RecordingLoader:
        def __init__(self, model):
            assert model is glm_model

        def load_weights(self, weights, *, mapper):
            observed.extend(weights)
            assert mapper is glm4_moe.Glm4MoeModel.hf_to_vllm_mapper
            return {name for name, _ in observed}

    def normalize(name: str) -> str | None:
        if ".rank1." in name:
            return None
        return name.replace(".rank0.", ".")

    monkeypatch.setattr(glm4_moe, "AutoWeightsLoader", RecordingLoader)
    monkeypatch.setattr(
        glm4_moe,
        "skip_spec_layers",
        lambda weights, config: weights,
    )
    monkeypatch.setattr(
        glm4_moe,
        "maybe_fuse_shared_experts",
        lambda weights, **kwargs: weights,
    )
    glm_model = object.__new__(glm4_moe.Glm4MoeModel)
    torch.nn.Module.__init__(glm_model)
    glm_model.quant_config = SimpleNamespace(
        normalize_rank_sliced_weight_name=normalize
    )
    glm_model.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
    local = torch.tensor(1)
    remote = torch.tensor(2)
    ordinary = torch.tensor(3)

    loaded = glm_model.load_weights(
        [
            ("layers.3.mlp.experts.0.gate_proj.rank0.trellis", local),
            ("layers.3.mlp.experts.0.gate_proj.rank1.trellis", remote),
            ("embed_tokens.weight", ordinary),
        ]
    )

    assert observed == [
        ("layers.3.mlp.experts.0.gate_proj.trellis", local),
        ("embed_tokens.weight", ordinary),
    ]
    assert loaded == {
        "layers.3.mlp.experts.0.gate_proj.trellis",
        "embed_tokens.weight",
    }


def test_rank_sliced_parameter_preallocates_projection_major_slab(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=3,
        shard_ids=("w1", "w3"),
        preallocate=True,
    )
    w1 = torch.arange(8, dtype=torch.int16).reshape(2, 2, 2)
    w3 = w1 + 20

    param.load_exl3_weight(w1, expert_id=1, shard_id="w1")
    param.load_exl3_weight(w3, expert_id=2, shard_id="w3")

    assert param.exl3_backing is not None
    assert tuple(param.exl3_backing.shape) == (2, 3, 2, 2, 2)
    assert (
        param.exl3_tensors[(1, "w1")].data_ptr() == param.exl3_backing[0, 1].data_ptr()
    )
    assert (
        param.exl3_tensors[(2, "w3")].data_ptr() == param.exl3_backing[1, 2].data_ptr()
    )
    torch.testing.assert_close(param.exl3_tensors[(1, "w1")], w1)
    torch.testing.assert_close(param.exl3_tensors[(2, "w3")], w3)
