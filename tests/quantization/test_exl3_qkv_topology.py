# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod


COMPONENTS = ["q_proj", "k_proj", "v_proj"]
OUTPUT_SPLITS = [12288, 1024, 1024]


def _entry(
    key: str,
    width: int,
    scale: str = "always",
    codebook: str = "mcg",
) -> dict:
    return {
        "quant_format": "exl3",
        "bits_per_weight": width,
        "codebook": codebook,
        "scale": scale,
        "stored_tensors": {
            f"{key}.trellis": {},
            f"{key}.suh": {},
            f"{key}.svh": {},
            f"{key}.{codebook}": {},
        },
    }


def _projection(
    name: str,
    width: int,
    scale: str = "always",
    codebook: str = "mcg",
) -> dict:
    return {"name": name, "K": width, "codebook": codebook, "scale": scale}


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


def _set_split_q(
    topology: dict,
    storage: dict,
    width: int,
    codebook: str,
) -> None:
    layer = topology["layers"][0]["layer"]
    topology["layers"][0]["projections"][0].update(K=width, codebook=codebook)
    key = f"{layer}.q_proj"
    storage[key] = _entry(key, width, codebook=codebook)


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


def test_text_routes_ignore_visual_layer_index_collisions():
    topology, storage = _mixed_checkpoint()
    visual = "model.visual.layers.1.self_attn"
    for component, width in zip(COMPONENTS, (6, 5, 4), strict=True):
        key = f"{visual}.{component}"
        storage[key] = _entry(key, width)

    config = _configure(topology, storage)

    assert config.qkv_topology_for_prefix(
        "model.visual.layers.1.self_attn.qkv_proj"
    ) is None
    assert config.qkv_topology_for_prefix(
        "language_model.model.layers.1.self_attn.qkv_proj"
    )["variant"] == "split"




@pytest.mark.parametrize(("width", "codebook"), [(3, "mcg"), (8, "mul1")])
def test_projection_accepts_supported_k_boundaries_and_codebooks(width, codebook):
    topology, storage = _mixed_checkpoint()
    _set_split_q(topology, storage, width, codebook)

    config = _configure(topology, storage)

    projection = config.qkv_topology_by_layer[1]["projections"][0]
    assert (projection["K"], projection["codebook"]) == (width, codebook)


@pytest.mark.parametrize(
    "layer",
    [
        "model.language_model.mtp.layers.1.self_attn",
        "model.visual.layers.1.self_attn",
    ],
)
def test_topology_rejects_non_decoder_layer_identities(layer):
    topology, storage = _mixed_checkpoint()
    topology["layers"][0]["layer"] = layer

    with pytest.raises(ValueError, match="full text-decoder self_attn block"):
        _configure(topology, storage)


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
            lambda topology, storage: topology["layers"][0]["projections"][0].update(
                K=2
            ),
            "one integer in 3\\.\\.8",
        ),
        (
            lambda topology, storage: topology["layers"][0]["projections"][0].update(
                K=9
            ),
            "one integer in 3\\.\\.8",
        ),
        (
            lambda topology, storage: topology["layers"][0]["projections"][0].update(
                codebook="other"
            ),
            "mcg.*mul1",
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


def test_fused_uniform_rejects_tensor_parallel_runtime():
    topology, storage = _mixed_checkpoint()
    method = Exl3LinearMethod(_configure(topology, storage))
    layer = torch.nn.Module()
    layer.prefix = "model.layers.4.self_attn.qkv_proj"
    layer.tp_rank = 0
    layer.tp_size = 2

    with pytest.raises(NotImplementedError, match="requires TP=1"):
        method.create_weights(
            layer,
            input_size_per_partition=4096,
            output_partition_sizes=OUTPUT_SPLITS,
            input_size=4096,
            output_size=sum(OUTPUT_SPLITS),
            params_dtype=torch.float16,
        )


def test_fused_uniform_rejects_runtime_output_split_mismatch():
    topology, storage = _mixed_checkpoint()
    method = Exl3LinearMethod(_configure(topology, storage))
    layer = torch.nn.Module()
    layer.prefix = "model.layers.4.self_attn.qkv_proj"
    layer.tp_rank = 0
    layer.tp_size = 1
    runtime_splits = [OUTPUT_SPLITS[0], OUTPUT_SPLITS[1], OUTPUT_SPLITS[2] - 1]

    with pytest.raises(ValueError, match="output_splits disagree"):
        method.create_weights(
            layer,
            input_size_per_partition=4096,
            output_partition_sizes=runtime_splits,
            input_size=4096,
            output_size=sum(runtime_splits),
            params_dtype=torch.float16,
        )


def test_split_rejects_runtime_output_split_mismatch():
    topology, storage = _mixed_checkpoint()
    method = Exl3LinearMethod(_configure(topology, storage))
    layer = torch.nn.Module()
    layer.prefix = "model.layers.1.self_attn.qkv_proj"
    layer.tp_rank = 0
    layer.tp_size = 1
    runtime_splits = [OUTPUT_SPLITS[0] - 1, *OUTPUT_SPLITS[1:]]

    with pytest.raises(ValueError, match="output_splits disagree"):
        method.create_weights(
            layer,
            input_size_per_partition=4096,
            output_partition_sizes=runtime_splits,
            input_size=4096,
            output_size=sum(runtime_splits),
            params_dtype=torch.float16,
        )




def test_fused_uniform_rejects_packed_fallback():
    topology, storage = _mixed_checkpoint()
    method = Exl3LinearMethod(_configure(topology, storage))
    layer = SimpleNamespace(exl3_qkv_variant="fused_uniform")

    with pytest.raises(RuntimeError, match="cannot use the packed-tensor fallback"):
        method.apply(layer, torch.ones(2, 4, dtype=torch.float16))


def test_split_view_fallback_rejects_packed_width_mismatch(monkeypatch):
    topology, storage = _mixed_checkpoint()
    method = Exl3LinearMethod(_configure(topology, storage))
    layer = SimpleNamespace(
        exl3_qkv_variant="split",
        exl3_output_partition_sizes=[6, 2, 2],
    )
    monkeypatch.setattr(
        method,
        "apply",
        lambda actual_layer, x, bias: torch.zeros(
            *x.shape[:-1], 9, dtype=x.dtype
        ),
    )

    with pytest.raises(RuntimeError):
        method.apply_qkv_views(layer, torch.ones(2, 4, dtype=torch.float16))


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


def test_qkv_views_honor_return_bias_false():
    calls = []

    def apply_views(layer, x, bias):
        calls.append((layer, bias))
        return x[..., :2], x[..., 2:4], x[..., 4:6]

    layer = SimpleNamespace(
        bias=None,
        skip_bias_add=False,
        return_bias=False,
        quant_method=SimpleNamespace(apply_qkv_views=apply_views),
    )

    views = QKVParallelLinear.forward_qkv_views(
        layer, torch.ones(2, 6, dtype=torch.float16)
    )

    assert len(views) == 3
    assert calls == [(layer, None)]


def test_qkv_views_return_deferred_bias_when_requested():
    bias = torch.nn.Parameter(torch.ones(6))

    def apply_views(layer, x, actual_bias):
        assert actual_bias is None
        return x[..., :2], x[..., 2:4], x[..., 4:6]

    layer = SimpleNamespace(
        bias=bias,
        skip_bias_add=True,
        return_bias=True,
        quant_method=SimpleNamespace(apply_qkv_views=apply_views),
    )

    views, deferred_bias = QKVParallelLinear.forward_qkv_views(
        layer, torch.ones(2, 6, dtype=torch.float16)
    )

    assert len(views) == 3
    assert deferred_bias is bias


def test_metadata_objects_are_not_aliased_to_caller():
    topology, storage = _mixed_checkpoint()
    original = deepcopy(topology)
    config = _configure(topology, storage)
    topology["layers"][0]["projections"][0]["K"] = 3

    assert config.qkv_topology_by_layer[1]["projections"][0]["K"] == 6
    assert original["layers"][0]["projections"][0]["K"] == 6
