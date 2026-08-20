# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod


COMPONENTS = ["q_proj", "k_proj", "v_proj"]
OUTPUT_SPLITS = [12288, 1024, 1024]


def _entry(key: str, width: int, scale: str = "always") -> dict:
    return {
        "quant_format": "exl3",
        "bits_per_weight": width,
        "codebook": "mcg",
        "scale": scale,
        "stored_tensors": {
            f"{key}.trellis": {},
            f"{key}.suh": {},
            f"{key}.svh": {},
            f"{key}.mcg": {},
        },
    }


def _projection(name: str, width: int, scale: str = "always") -> dict:
    return {"name": name, "K": width, "codebook": "mcg", "scale": scale}


def _mixed_checkpoint() -> tuple[dict, dict]:
    split = "model.language_model.layers.1.self_attn"
    fused = "model.language_model.layers.4.self_attn"
    topology = {
        "schema": "exl3_qkv_topology/1",
        "layers": [
            {
                "layer": split,
                "variant": "split",
                "components": COMPONENTS,
                "output_splits": OUTPUT_SPLITS,
                "projections": [
                    _projection("q_proj", 6, "always"),
                    _projection("k_proj", 5, "never"),
                    _projection("v_proj", 4, "auto"),
                ],
            },
            {
                "layer": fused,
                "variant": "fused_uniform",
                "components": COMPONENTS,
                "output_splits": OUTPUT_SPLITS,
                "projection": _projection("qkv_proj", 6),
            },
        ],
    }
    storage = {
        f"{split}.q_proj": _entry(f"{split}.q_proj", 6, "always"),
        f"{split}.k_proj": _entry(f"{split}.k_proj", 5, "never"),
        f"{split}.v_proj": _entry(f"{split}.v_proj", 4, "auto"),
        f"{fused}.qkv_proj": _entry(f"{fused}.qkv_proj", 6),
    }
    return topology, storage


def _configure(topology: dict, storage: dict) -> Exl3Config:
    config = Exl3Config(
        tensor_storage=storage,
        exl3_qkv_topology=topology,
    )
    config.maybe_update_config(
        "unused",
        SimpleNamespace(
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
        ),
    )
    return config


def test_checkpoint_mixes_explicit_split_and_fused_uniform_routes():
    topology, storage = _mixed_checkpoint()
    config = _configure(topology, storage)

    split = config.qkv_topology_for_prefix("model.layers.1.self_attn.qkv_proj")
    fused = config.qkv_topology_for_prefix("model.layers.4.self_attn.qkv_proj")
    assert split is not None and split["variant"] == "split"
    assert [item["K"] for item in split["projections"]] == [6, 5, 4]
    assert fused is not None and fused["variant"] == "fused_uniform"
    assert fused["output_splits"] == OUTPUT_SPLITS
    assert [(row["variant"], row["launches"]) for row in config.qkv_registry_rows] == [
        ("split", 3),
        ("fused_uniform", 1),
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda topology, storage: topology["layers"].pop(), "does not cover"),
        (
            lambda topology, storage: topology["layers"][1].update(
                variant="mixed_width"
            ),
            "future research",
        ),
        (
            lambda topology, storage: topology["layers"][1]["projection"].update(
                K=[5, 6, 6]
            ),
            "one integer",
        ),
        (
            lambda topology, storage: storage.update(
                {
                    f"{topology['layers'][1]['layer']}.q_proj": _entry(
                        f"{topology['layers'][1]['layer']}.q_proj", 6
                    )
                }
            ),
            "duplicate split\\+fused",
        ),
    ],
)
def test_topology_metadata_fails_closed(mutation, message):
    topology, storage = _mixed_checkpoint()
    mutation(topology, storage)

    with pytest.raises(ValueError, match=message):
        _configure(topology, storage)


def test_declared_topology_rejects_route_fallback():
    topology, storage = _mixed_checkpoint()
    config = _configure(topology, storage)

    with pytest.raises(ValueError, match="no explicit EXL3 QKV route"):
        config.qkv_topology_for_prefix("model.layers.3.self_attn.qkv_proj")


def test_fused_uniform_issues_one_apply_and_returns_views(monkeypatch):
    topology, storage = _mixed_checkpoint()
    config = _configure(topology, storage)
    method = Exl3LinearMethod(config)
    layer = SimpleNamespace(
        prefix="model.layers.4.self_attn.qkv_proj",
        exl3_qkv_variant="fused_uniform",
        exl3_output_partition_sizes=[6, 2, 2],
    )
    calls = []
    packed = torch.arange(20, dtype=torch.float16).reshape(2, 10)

    def apply_one(actual_layer, x, shard_id):
        calls.append((actual_layer, x, shard_id))
        return packed

    monkeypatch.setattr(method, "_apply_one", apply_one)
    q, k, v = method.apply_qkv_views(layer, torch.ones(2, 4, dtype=torch.float16))

    assert len(calls) == 1 and calls[0][2] is None
    assert q.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert k.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert (q.shape[-1], k.shape[-1], v.shape[-1]) == (6, 2, 2)
    assert config.qkv_route_counters == {"split": 0, "fused_uniform": 1}
    assert config.qkv_warmup_counters == {"split": 0, "fused_uniform": 1}


def test_split_route_retains_three_applies_and_cat(monkeypatch):
    topology, storage = _mixed_checkpoint()
    config = _configure(topology, storage)
    method = Exl3LinearMethod(config)
    layer = SimpleNamespace(
        prefix="model.layers.1.self_attn.qkv_proj",
        exl3_qkv_variant="split",
        exl3_shard_ids=["q", "k", "v"],
    )
    calls = []

    def apply_one(actual_layer, x, shard_id):
        calls.append(shard_id)
        return torch.full((x.shape[0], 2), len(calls), dtype=torch.float16)

    monkeypatch.setattr(method, "_apply_one", apply_one)
    output = method.apply(layer, torch.ones(2, 4, dtype=torch.float16))

    assert calls == ["q", "k", "v"]
    assert output.tolist()[0] == [1, 1, 2, 2, 3, 3]
    assert config.qkv_route_counters == {"split": 1, "fused_uniform": 0}
    assert config.qkv_warmup_counters == {"split": 1, "fused_uniform": 0}


def test_metadata_objects_are_not_aliased_to_caller():
    topology, storage = _mixed_checkpoint()
    original = deepcopy(topology)
    config = _configure(topology, storage)
    topology["layers"][0]["projections"][0]["K"] = 3

    assert config.qkv_topology_by_layer[1]["projections"][0]["K"] == 6
    assert original["layers"][0]["projections"][0]["K"] == 6


def test_fused_consumer_source_has_one_apply_and_view_split():
    source = inspect.getsource(Exl3LinearMethod.apply_qkv_views)
    fused_source = source.split(
        'record_qkv_route(layer, "fused_uniform")', maxsplit=1
    )[1]

    assert fused_source.count("self._apply_one(") == 1
    assert ".split(layer.exl3_output_partition_sizes" in fused_source
    assert "torch.cat" not in fused_source
