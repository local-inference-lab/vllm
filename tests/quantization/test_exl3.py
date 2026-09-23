# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.exl3 import plan_exl3_extent


def _kimi_manifest(**layout):
    return SimpleNamespace(
        geometry=SimpleNamespace(
            num_slots=96,
            slot_channels=32,
            moe_layer_indices=tuple(range(1, 93)),
        ),
        layout=SimpleNamespace(
            extent_alignment_slots=layout.get("alignment", 4),
            extent_barriers=layout.get("barriers", (48,)),
        ),
    )


@pytest.mark.parametrize("tp_size", range(2, 25))
def test_exl3_rank_extents_cover_complete_blocks_without_padding(tp_size):
    manifest = _kimi_manifest()
    for layer in range(1, 93):
        covered: list[int] = []
        for rank in range(tp_size):
            extent = plan_exl3_extent(manifest, layer, tp_size, rank)
            first, count = extent.first_slot, extent.slot_count
            assert count > 0 and count % 4 == first % 4 == 0
            assert first // 48 == (first + count - 1) // 48
            assert extent.intermediate_size == 32 * count
            covered.extend(range(first, first + count))
        assert sorted(covered) == list(range(96))


def test_exl3_tp10_balances_resident_expert_payload():
    manifest = _kimi_manifest()
    totals = [
        sum(
            plan_exl3_extent(manifest, layer, 10, rank).slot_count
            for layer in range(1, 93)
        )
        for rank in range(10)
    ]
    assert sorted(totals) == [880, 880, 884, 884, 884, 884, 884, 884, 884, 884]


@pytest.mark.parametrize(
    "layer,tp,rank", [(0, 10, 0), (93, 10, 0), (1, 1, 0), (1, 25, 0), (1, 10, 10)]
)
def test_exl3_rejects_invalid_layer_or_rank_identity(layer, tp, rank):
    with pytest.raises(ValueError):
        plan_exl3_extent(_kimi_manifest(), layer, tp, rank)


@pytest.mark.parametrize(
    "layout", [{"barriers": ()}, {"barriers": (32,)}, {"alignment": 5}]
)
def test_exl3_rejects_unplanned_container_layouts(layout):
    with pytest.raises(NotImplementedError):
        plan_exl3_extent(_kimi_manifest(**layout), 1, 9, 0)


@pytest.fixture
def moe_config():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    return FusedMoEConfig(
        num_experts=896,
        num_local_experts=896,
        num_logical_experts=896,
        experts_per_token=16,
        hidden_dim=3584,
        intermediate_size=384,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )


def test_exl3_defers_expert_storage_to_extent_preparation(moe_config):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    method = Exl3MoEMethod(moe_config)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.mlp.experts"
    method.create_weights(layer, 896, 3584, 384, torch.bfloat16)
    assert method.layer_index == 17
    assert set(dict(layer.named_parameters())) == {"w13_weight", "w2_weight"}
    assert all(weight.numel() == 0 for weight in layer.parameters())
    config = method.get_fused_moe_quant_config(layer)
    assert config.weight_quant_dtype == "exl3"
    assert config.quant_dtype is None


@pytest.mark.parametrize(
    "fields",
    [
        {"in_dtype": torch.float16},
        {"activation_situ_beta": 1.0},
        {"activation_situ_linear_beta": 0.0},
        {"has_bias": True},
    ],
)
def test_exl3_rejects_mismatched_expert_math(moe_config, fields):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    with pytest.raises(ValueError, match="BF16 SiTU"):
        Exl3MoEMethod(replace(moe_config, **fields))


@pytest.mark.parametrize(
    "fields", [{"use_ep": True, "ep_size": 2}, {"dp_size": 2}, {"enable_eplb": True}]
)
def test_exl3_rejects_non_tp_expert_partitioning(moe_config, fields):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    parallel = replace(moe_config.moe_parallel_config, **fields)
    with pytest.raises(NotImplementedError, match="without EP or DP"):
        Exl3MoEMethod(replace(moe_config, moe_parallel_config=parallel))


@pytest.fixture
def checkpoint_quant_config():
    return {
        "quant_method": "exl3",
        "dense_format": "mxfp8",
        "ignored_layers": ["kv_b_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj"],
        "exl3": {"manifest": "exl3-manifest.json"},
    }


def test_exl3_config_preserves_serialized_projection_formats(
    checkpoint_quant_config,
):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
    from vllm.models.kimi_k3.nvidia.kda import (
        use_split_mixed_precision_input_projection,
    )
    from vllm.models.kimi_k3.nvidia.model import KimiLinearModel

    cls = get_quantization_config("exl3")
    assert cls.override_quantization_method(checkpoint_quant_config, None) == "exl3"
    assert cls.override_quantization_method({"quant_method": "modelopt"}, None) is None
    config = cls.from_config(checkpoint_quant_config)
    config.packed_modules_mapping = KimiLinearModel.packed_modules_mapping
    layer = object.__new__(LinearBase)
    for name in ("in_proj_qkv", "fused_qkv_a_proj", "q_b_proj", "o_proj"):
        method = config.get_quant_method(layer, f"model.layers.3.self_attn.{name}")
        assert isinstance(method, ModelOptLinearMethod)
        assert method.spec.weight == kMxfp8Static
    for prefix in (
        "model.layers.3.self_attn.in_proj_gfab",
        "model.layers.3.self_attn.g_proj",
        "model.layers.3.self_attn.f_b_proj",
        "model.layers.3.self_attn.kv_b_proj",
        "vision_tower.blocks.0.attn.qkv_proj",
        "mm_projector.0",
        "lm_head",
    ):
        assert isinstance(
            config.get_quant_method(layer, prefix), UnquantizedLinearMethod
        )
    assert use_split_mixed_precision_input_projection(config)
    assert not use_split_mixed_precision_input_projection(None)
    assert config.separate_mla_output_gate
    with pytest.raises(ValueError, match="some but not all shards"):
        config.get_quant_method(layer, "model.layers.3.self_attn.fused_qkv_a_g_proj")


@pytest.mark.parametrize("defect", ["manifest", "dense", "gate", "qkv"])
def test_exl3_config_rejects_unsupported_wire_formats(checkpoint_quant_config, defect):
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    if defect == "manifest":
        checkpoint_quant_config["exl3"]["manifest"] = "other-manifest.json"
    elif defect == "dense":
        checkpoint_quant_config["dense_format"] = "bf16"
    elif defect == "gate":
        checkpoint_quant_config["ignored_layers"].remove("g_proj")
    else:
        checkpoint_quant_config["ignored_layers"].append("q_proj")
    with pytest.raises(ValueError):
        Exl3Config.from_config(checkpoint_quant_config)


@pytest.mark.parametrize("tp_size", [8, 9, 10, 12, 16])
def test_exl3_head_padding_preserves_checkpoint_geometry(tp_size):
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    text = SimpleNamespace(
        num_attention_heads=96,
        linear_attn_config={"num_heads": 96, "head_dim": 128},
        moe_intermediate_size=3072,
    )
    model = SimpleNamespace(
        quantization="exl3",
        hf_text_config=text,
        get_model_arch_config=lambda: (
            text.num_attention_heads,
            text.moe_intermediate_size,
        ),
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=tp_size, enable_expert_parallel=False
    )
    update = KimiK3ForConditionalGenerationConfig.update_model_config_for_parallelism
    update(model, parallel)
    update(model, parallel)
    expected = ((96 + tp_size - 1) // tp_size) * tp_size
    assert text.num_attention_heads == text.linear_attn_config["num_heads"] == expected
    assert (
        text.original_num_attention_heads
        == text.linear_attn_config["original_num_heads"]
        == 96
    )
    assert model.model_arch_config == (expected, 3072)
    # A reused config must derive another topology from the checkpoint, not its padding.
    parallel.tensor_parallel_size = 8
    update(model, parallel)
    assert model.model_arch_config == (96, 3072)
    assert text.linear_attn_config["num_heads"] == 96


def test_exl3_padding_does_not_mutate_official_mxfp4_config():
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    model = SimpleNamespace(quantization="mxfp4")
    KimiK3ForConditionalGenerationConfig.update_model_config_for_parallelism(
        model, SimpleNamespace(tensor_parallel_size=10)
    )
    assert vars(model) == {"quantization": "mxfp4"}
