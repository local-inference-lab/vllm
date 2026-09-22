# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import qsrt_k2
from vllm.model_executor.layers.quantization.utils.qsrt_k2 import plan_qsrt_k2_extent


@pytest.mark.parametrize("tp_size", range(2, 25))
def test_qsrt_k2_rank_extents_cover_complete_blocks_without_padding(tp_size):
    for layer in range(1, 93):
        covered = []
        for rank in range(tp_size):
            extent = plan_qsrt_k2_extent(layer, tp_size, rank)
            first, count = extent.first_slot, extent.slot_count
            assert count > 0 and count % 4 == first % 4 == 0
            assert first // 48 == (first + count - 1) // 48
            assert extent.intermediate_size == 32 * count
            covered.extend(range(first, first + count))
        assert sorted(covered) == list(range(96))


def test_qsrt_k2_tp10_balances_resident_expert_payload():
    totals = [
        sum(plan_qsrt_k2_extent(layer, 10, rank).slot_count for layer in range(1, 93))
        for rank in range(10)
    ]
    assert sorted(totals) == [880, 880, 884, 884, 884, 884, 884, 884, 884, 884]


@pytest.mark.parametrize(
    "layer,tp,rank", [(0, 10, 0), (93, 10, 0), (1, 1, 0), (1, 25, 0), (1, 10, 10)]
)
def test_qsrt_k2_rejects_invalid_source_or_rank_identity(layer, tp, rank):
    with pytest.raises(ValueError):
        plan_qsrt_k2_extent(layer, tp, rank)


@pytest.fixture
def source_header(tmp_path, monkeypatch):
    path = tmp_path / "qsrt-layer-00001.safetensors"
    path.write_bytes(b"header-fixture")
    manifest = {
        "codec": "QSRT",
        "complete": True,
        "storage_format": "qsrt_atoms_v2",
        "storage_schema": "kquant_kimi_k3_qsrt_atoms_v2",
        "profile": "k2_coupled_h512_h128",
        "all_experts_qsrt": True,
        "layers": {
            "1": {"qsrt_atoms": path.name, "atom_disk_bytes": path.stat().st_size}
        },
    }
    (tmp_path / "qsrt-manifest.json").write_text(json.dumps(manifest))
    metadata = {
        "format": "pt",
        "version": "2",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "profile": "k2_coupled_h512_h128",
        "profile_id": "3",
        "layer": "1",
        "experts": "896",
        "intermediate_channels": "3072",
        "latent_channels": "3584",
        "atom_channels": "32",
        "atom_slots": "96",
        "atom_slot_stride_bytes": "77242368",
        "p22_atom_bundle_bytes": "86208",
        "alignment_bytes": "4096",
        "residual_hadamard_block_size": "512",
        "preactivation_hadamard_block_size": "128",
        "postactivation_hadamard_block_size": "128",
        "intermediate_rotation_draws": "format_section[896:1792]",
        "schema": "kquant_kimi_k3_qsrt_atoms_v2",
    }
    formats = torch.zeros(4096, dtype=torch.uint8)
    formats[:896] = 0x44
    formats[896:1792:2] = 6
    scales = torch.zeros(24576, dtype=torch.uint8)
    scales[:21504].view(torch.float16).fill_(0.25)
    tensors = {"_qsrt_format_section": formats, "_qsrt_shared_scale_section": scales}

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def keys(self):
            return [*tensors, "qsrt_atoms"]

        def metadata(self):
            return metadata

        def get_tensor(self, name):
            # Reading the multi-GiB slab during metadata validation is a bug.
            return tensors[name]

        def get_slice(self, name):
            assert name == "qsrt_atoms"
            return SimpleNamespace(get_shape=lambda: [96, 77242368])

    monkeypatch.setattr(qsrt_k2, "safe_open", lambda *args, **kwargs: Handle())
    return tmp_path, metadata, formats, scales


def test_qsrt_k2_metadata_retains_rotation_draws_without_loading_atoms(source_header):
    root, _, formats, scales = source_header
    source = qsrt_k2.read_qsrt_k2_source(root, 1)
    assert source.scales.shape == (3, 3584)
    assert torch.equal(source.rotation_draws, formats[896:1792])
    assert source.scales.eq(0.25).all()
    formats.zero_()
    scales.zero_()
    assert source.rotation_draws[0] == 6 and source.scales.eq(0.25).all()


@pytest.mark.parametrize("defect", ["profile", "stride", "draw", "padding", "scale"])
def test_qsrt_k2_metadata_rejects_incompatible_or_corrupt_containers(
    source_header, defect
):
    root, metadata, formats, scales = source_header
    if defect == "profile":
        metadata["profile"] = "k3x22_k4x2"
    elif defect == "stride":
        metadata["atom_slot_stride_bytes"] = "1"
    elif defect == "draw":
        formats[896] = 8
    elif defect == "padding":
        formats[-1] = 1
    else:
        scales[:2].view(torch.float16).fill_(float("nan"))
    with pytest.raises(ValueError):
        qsrt_k2.read_qsrt_k2_source(root, 1)


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


def test_qsrt_k2_defers_expert_storage_to_source_extent_preparation(moe_config):
    from vllm.model_executor.layers.quantization.qsrt_k2 import QsrtK2MoEMethod

    method = QsrtK2MoEMethod(moe_config)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.mlp.experts"
    method.create_weights(layer, 896, 3584, 384, torch.bfloat16)
    assert method.layer_index == 17
    assert set(dict(layer.named_parameters())) == {"w13_weight", "w2_weight"}
    assert all(weight.numel() == 0 for weight in layer.parameters())
    config = method.get_fused_moe_quant_config(layer)
    assert config.weight_quant_dtype == "btx"
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
def test_qsrt_k2_rejects_mismatched_expert_math(moe_config, fields):
    from vllm.model_executor.layers.quantization.qsrt_k2 import QsrtK2MoEMethod

    with pytest.raises(ValueError, match="BF16 SiTU"):
        QsrtK2MoEMethod(replace(moe_config, **fields))


@pytest.mark.parametrize(
    "fields", [{"use_ep": True, "ep_size": 2}, {"dp_size": 2}, {"enable_eplb": True}]
)
def test_qsrt_k2_rejects_non_tp_expert_partitioning(moe_config, fields):
    from vllm.model_executor.layers.quantization.qsrt_k2 import QsrtK2MoEMethod

    parallel = replace(moe_config.moe_parallel_config, **fields)
    with pytest.raises(NotImplementedError, match="without EP or DP"):
        QsrtK2MoEMethod(replace(moe_config, moe_parallel_config=parallel))


@pytest.fixture
def checkpoint_quant_config():
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "dense_format": "mxfp8",
        "ignored_layers": ["kv_b_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj"],
        "qsrt": {
            "storage_format": "qsrt_atoms_v2",
            "profile": "k2_coupled_h512_h128",
            "schema": "kquant_kimi_k3_qsrt_atoms_v2",
            "encoding": "qsrt_sqg_e4m3",
            "codebook": "sqg_xor_cheb_t12",
            "artifact_manifest": "qsrt-manifest.json",
        },
    }


def test_qsrt_k2_config_preserves_serialized_projection_formats(
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

    cls = get_quantization_config("qsrt_k2")
    assert cls.override_quantization_method(checkpoint_quant_config, None) == "qsrt_k2"
    assert cls.override_quantization_method({"quant_algo": "NVFP4"}, None) is None
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


@pytest.mark.parametrize("defect", ["profile", "schema", "dense", "gate", "qkv"])
def test_qsrt_k2_config_rejects_unsupported_wire_formats(
    checkpoint_quant_config, defect
):
    from vllm.model_executor.layers.quantization.qsrt_k2 import QsrtK2Config

    if defect in ("profile", "schema"):
        checkpoint_quant_config["qsrt"][defect] = "unsupported"
    elif defect == "dense":
        checkpoint_quant_config["dense_format"] = "bf16"
    elif defect == "gate":
        checkpoint_quant_config["ignored_layers"].remove("g_proj")
    else:
        checkpoint_quant_config["ignored_layers"].append("q_proj")
    with pytest.raises(ValueError):
        QsrtK2Config.from_config(checkpoint_quant_config)


@pytest.mark.parametrize("tp_size", [8, 9, 10, 12, 16])
def test_qsrt_k2_head_padding_preserves_checkpoint_geometry(tp_size):
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    text = SimpleNamespace(
        num_attention_heads=96,
        linear_attn_config={"num_heads": 96, "head_dim": 128},
        moe_intermediate_size=3072,
    )
    model = SimpleNamespace(
        quantization="qsrt_k2",
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


def test_qsrt_k2_padding_does_not_mutate_official_mxfp4_config():
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    model = SimpleNamespace(quantization="mxfp4")
    KimiK3ForConditionalGenerationConfig.update_model_config_for_parallelism(
        model, SimpleNamespace(tensor_parallel_size=10)
    )
    assert vars(model) == {"quantization": "mxfp4"}
