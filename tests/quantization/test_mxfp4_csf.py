# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
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
from vllm.model_executor.model_loader.mxfp4_csf_loader import Mxfp4CsfModelLoader
from vllm.model_executor.model_loader.mxfp4_csf_loader import (
    _slice_scale_plane as slice_scale_plane,
)
from vllm.models.deepseek_v4_1.mxfp4_csf import (
    DeepseekV41Mxfp4CsfConfig,
    Mxfp4CsfMoEMethod,
)

from .csf_fixtures import decode_planes, matrix


def write_index(root, tensors, filename, family):
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": family,
                "quantization_config": {"weight_scale_encoding": "csf_32x2_v1"},
            }
        )
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {name: f"tensors/{filename}" for name in tensors},
            }
        )
    )


@pytest.fixture
def moe():
    return FusedMoEConfig(
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


@pytest.fixture
def owner():
    return DeepseekV41Mxfp4CsfConfig.from_config(
        {"format_version": 1, "checkpoint_root": "/csf"}
    )


def test_exact_config_keeps_native_dense_precision_and_empty_expert_handles(moe, owner):
    assert owner.is_scale_e8m0 and owner.weight_block_size == [32, 32]
    assert owner.is_checkpoint_fp8_serialized and owner.scale_scratch is None
    method = Mxfp4CsfMoEMethod(moe, owner)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.ffn.experts"
    method.create_weights(layer, 384, 5120, 576, torch.bfloat16)
    assert not list(layer.named_parameters()) and not layer.state_dict()
    assert method.layer_index == 17 and method.local_intermediate == 576
    assert method.get_fused_moe_quant_config(layer).weight_quant_dtype == "mxfp4"
    with pytest.raises(ValueError, match="format_version"):
        DeepseekV41Mxfp4CsfConfig.from_config({})


def test_exact_config_and_loader_are_registered():
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.mxfp4_csf import Mxfp4CsfConfig
    from vllm.model_executor.model_loader import get_model_loader

    assert get_quantization_config("mxfp4_csf") is Mxfp4CsfConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="mxfp4_csf")), Mxfp4CsfModelLoader
    )


@pytest.mark.parametrize(
    "activation_mode,width", [("a16", 288), ("a8", 288), ("a8", 576)]
)
def test_csf_cpu_shard_reader_sizes_native_scratch_and_restores_threads(
    monkeypatch, moe, owner, activation_mode, width
):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint
    from vllm.models.deepseek_v4_1 import mxfp4_csf
    from vllm.utils.torch_utils import set_default_torch_num_threads

    monkeypatch.setattr(mxfp4_csf, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(mxfp4_csf, "get_tensor_model_parallel_rank", lambda: 0)
    method = Mxfp4CsfMoEMethod(moe, owner, activation_mode=activation_mode)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.ffn.experts"
    method.create_weights(layer, 384, 5120, width, torch.bfloat16)

    def fail_reader(*args, **kwargs):
        assert torch.get_num_threads() == 1
        if activation_mode == "a8" and width == 288:
            assert kwargs["w13_scale_scratch"].shape == (384, 768, 160)
            assert kwargs["w2_scale_scratch"].shape == (384, 5120, 12)
        else:
            assert kwargs["w13_scale_scratch"].shape == (384, 160, 2 * width)
            assert kwargs["w2_scale_scratch"].shape == (384, width // 32, 5120)
        raise OSError("checkpoint read failed")

    monkeypatch.setattr(checkpoint, "read_mxfp4_csf_layer", fail_reader)
    with set_default_torch_num_threads(4):
        with pytest.raises(OSError, match="checkpoint read failed"):
            method.process_weights_after_loading(layer)
        assert torch.get_num_threads() == 4


@pytest.mark.parametrize("mode", ["normal", "pipeline", "ubatching", "wrong_loader"])
@pytest.mark.parametrize("force_a16", [False, True])
def test_target_uses_csf_while_dense_and_draft_keep_native_methods(
    monkeypatch, moe, owner, mode, force_a16
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.deepseek_v4_1 import mxfp4_csf

    monkeypatch.setattr(mxfp4_csf.envs, "VLLM_B12X_MOE_FP4_FORCE_A16", force_a16)

    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=40)),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2 if mode == "pipeline" else 1,
            use_ubatching=mode == "ubatching",
        ),
        load_config=SimpleNamespace(
            load_format="safetensors" if mode == "wrong_loader" else "mxfp4_csf"
        ),
    )
    monkeypatch.setattr(mxfp4_csf, "get_current_vllm_config", lambda: config)
    native = object()
    monkeypatch.setattr(
        mxfp4_csf.DeepseekV41FP8Config,
        "get_quant_method",
        lambda *args, **kwargs: native,
    )
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = moe
    if mode == "normal":
        method = owner.get_quant_method(layer, "model.layers.39.ffn.experts")
        assert isinstance(method, Mxfp4CsfMoEMethod)
        assert method.activation_mode == ("a16" if force_a16 else "a8")
        assert method.get_fused_moe_quant_config(layer).quant_dtype == (
            None if force_a16 else "mxfp8"
        )
    elif mode == "wrong_loader":
        with pytest.raises(ValueError, match="load-format"):
            owner.get_quant_method(layer, "model.layers.39.ffn.experts")
    else:
        with pytest.raises(NotImplementedError, match="PP1 without ubatching"):
            owner.get_quant_method(layer, "model.layers.39.ffn.experts")
    assert owner.get_quant_method(layer, "model.layers.40.ffn.experts") is native
    assert owner.get_quant_method(torch.nn.Module(), "model.layers.0.attn") is native


@pytest.mark.parametrize(
    "field,value", [("dp_size", 2), ("use_ep", True), ("enable_eplb", True)]
)
def test_exact_experts_reject_non_tp_execution(moe, owner, field, value):
    parallel = replace(moe.moe_parallel_config, **{field: value})
    with pytest.raises(NotImplementedError, match="TP without EP/DP"):
        Mxfp4CsfMoEMethod(replace(moe, moe_parallel_config=parallel), owner)


@pytest.mark.parametrize("activation_mode", ["a16", "a8"])
def test_prepared_native_experts_validate_source_shape_and_swiglu_limit(
    monkeypatch, moe, owner, activation_mode
):
    from b12x.moe import fused_moe

    from vllm.model_executor.layers.fused_moe import b12x

    layer = torch.nn.Module()
    layer.activation = moe.activation
    layer.apply_router_weight_on_input = False
    quant = Mxfp4CsfMoEMethod(
        moe, owner, activation_mode=activation_mode
    ).get_fused_moe_quant_config(layer)
    backend = b12x.B12xExperts(moe, quant)
    prepared = Mock(spec=fused_moe.PreparedExperts)
    activation = fused_moe.ActivationSpec(
        mode=activation_mode,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
        swiglu_limit=10.0,
    )
    prepared.plan = SimpleNamespace(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=activation,
    )
    prepared.num_experts = moe.num_experts
    prepared.hidden_size = moe.hidden_dim
    prepared.intermediate_size = moe.intermediate_size_per_partition
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *a, **k: None
    )
    backend.install_prepared_experts(layer, prepared)
    assert layer._b12x_prepared_experts is prepared
    assert backend._prepared_experts is prepared
    prepared.plan.activation = replace(
        activation, mode="a16" if activation_mode == "a8" else "a8"
    )
    with pytest.raises(ValueError, match="activation"):
        backend.install_prepared_experts(layer, prepared)
    prepared.plan.activation = replace(activation, swiglu_limit=None)
    with pytest.raises(ValueError, match="activation"):
        backend.install_prepared_experts(layer, prepared)
    prepared.plan.activation = activation
    prepared.hidden_size += 128
    with pytest.raises(ValueError, match="geometry"):
        backend.install_prepared_experts(layer, prepared)
    prepared.hidden_size = moe.hidden_dim
    prepared.plan.source = fused_moe.PackedSource(format="modelopt_nvfp4")
    with pytest.raises(TypeError, match="encoding"):
        backend.install_prepared_experts(layer, prepared)


def test_loader_preserves_file_backed_engram_and_native_draft(tmp_path):
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    name = "model-00001.safetensors"
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": torch.ones(8, dtype=torch.uint8),
        "layers.0.ffn.experts.0.w1.scale.csf32_codes": torch.ones(8, dtype=torch.uint8),
        "layers.40.ffn.experts.0.w1.weight": torch.full((8,), 42, dtype=torch.uint8),
        "layers.1.engram.embedding.weight": torch.ones((8, 8), dtype=torch.uint8),
        "norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / name)
    write_index(tmp_path, tensors, name, "deepseek_v41")
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=40,
            quantization_config={
                "quant_method": "mxfp4_csf",
                "checkpoint_root": str(tmp_path),
            },
        )
    )
    model = SimpleNamespace(
        checkpoint_file_weight_filter=lambda n: ".engram.embedding." in n
    )
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
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


def test_kimi_mxfp4_csf_keeps_bf16_dense_and_situ_experts(monkeypatch, moe):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import kimi_mxfp4_csf

    owner = kimi_mxfp4_csf.KimiMxfp4CsfConfig.from_config(
        {"format_version": 1, "checkpoint_root": "/csf"}
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, use_ubatching=False),
        load_config=SimpleNamespace(load_format="mxfp4_csf"),
    )
    monkeypatch.setattr(kimi_mxfp4_csf, "get_current_vllm_config", lambda: config)
    experts = Mock(spec=RoutedExperts)
    experts.moe_config = replace(
        moe,
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_limit=None,
    )
    method = owner.get_quant_method(experts, "model.layers.1.mlp.experts")
    assert isinstance(method, Mxfp4CsfMoEMethod)
    assert method.activation_mode == "a16"
    assert method.get_fused_moe_quant_config(experts).quant_dtype is None
    assert isinstance(
        owner.get_quant_method(Mock(spec=LinearBase), "model.layers.0.q_proj"),
        UnquantizedLinearMethod,
    )
    config.parallel_config.use_ubatching = True
    with pytest.raises(NotImplementedError, match="PP1 without ubatching"):
        owner.get_quant_method(experts, "model.layers.1.mlp.experts")


@pytest.mark.parametrize("quant_method", ["mxfp4_csf"])
@pytest.mark.parametrize("load_format", ["mxfp4_csf"])
def test_kimi_loader_skips_compressed_experts_with_nested_text_config(
    tmp_path, quant_method, load_format
):
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    filename = "model-00001.safetensors"
    expert = "language_model.model.layers.1.block_sparse_moe.experts.0.w1."
    dense = "language_model.model.layers.0.mlp.gate_proj.weight"
    tensors = {
        expert + "weight_packed": torch.ones(8, dtype=torch.uint8),
        expert + "weight_scale.csf32_codes": torch.ones(8, dtype=torch.uint8),
        dense: torch.ones((8, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / filename)
    write_index(tmp_path, tensors, filename, "kimi_k3")
    config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        hf_text_config=SimpleNamespace(
            num_hidden_layers=93,
            quantization_config={
                "quant_method": quant_method,
                "checkpoint_root": str(tmp_path),
            },
        ),
    )
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format=load_format))
    result = dict(loader.get_all_weights(config, SimpleNamespace()))
    assert list(result) == [dense]
    assert torch.equal(result[dense], tensors[dense])


@pytest.mark.parametrize("family", ["kimi_k3", "deepseek_v41"])
def test_mxfp4_csf_selects_retained_precision_and_rejects_other_configs(
    monkeypatch, moe, family
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import (
        get_quantization_config,
        kimi_mxfp4_csf,
    )
    from vllm.model_executor.layers.quantization.mxfp4_csf import Mxfp4CsfConfig
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint
    from vllm.models.deepseek_v4_1 import mxfp4_csf

    monkeypatch.setattr(checkpoint, "checkpoint_contract", lambda _: {"family": family})
    owner = get_quantization_config("mxfp4_csf").from_config(
        {"format_version": 1, "checkpoint_root": "/compressed-scales"}
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=40)),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, use_ubatching=False),
        load_config=SimpleNamespace(load_format="mxfp4_csf"),
    )
    for module in (kimi_mxfp4_csf, mxfp4_csf):
        monkeypatch.setattr(module, "get_current_vllm_config", lambda: config)
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = moe
    assert owner.get_name() == "mxfp4_csf"
    assert isinstance(
        owner.get_quant_method(layer, "model.layers.3.mlp.experts"), Mxfp4CsfMoEMethod
    )
    if family == "deepseek_v41":
        assert owner.is_checkpoint_fp8_serialized
        assert owner.weight_block_size == [32, 32] and owner.is_scale_e8m0
        assert isinstance(owner, DeepseekV41Mxfp4CsfConfig)
    else:
        assert isinstance(
            owner.get_quant_method(Mock(spec=LinearBase), "model.layers.0.q_proj"),
            UnquantizedLinearMethod,
        )
    assert (
        Mxfp4CsfConfig.override_quantization_method(
            {"quant_method": "mxfp4_csf"}, "mxfp4_csf"
        )
        == "mxfp4_csf"
    )
    for name in ("exact_mxfp4", "kimi_x4t"):
        assert (
            Mxfp4CsfConfig.override_quantization_method(
                {"quant_method": name}, "mxfp4_csf"
            )
            is None
        )


@pytest.mark.parametrize("fault", ["encoding", "family", "inventory", "path"])
def test_csf_contract_rejects_invalid_encoding_and_index(tmp_path, fault):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as reader

    write_index(tmp_path, {"weight": None}, "weights.safetensors", "deepseek_v41")
    config = json.loads((tmp_path / "config.json").read_text())
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    if fault == "encoding":
        config["quantization_config"]["weight_scale_encoding"] = "unknown"
    elif fault == "family":
        config["model_type"] = "unrecognized"
    elif fault == "inventory":
        index["weight_map"] = {}
    else:
        index["weight_map"]["weight"] = "../weights.safetensors"
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        reader.checkpoint_contract(str(tmp_path))


def test_csf_contract_takes_the_family_from_the_top_level_model_type(tmp_path):
    """DeepSeek's multimodal configs name the model at the top (deepseek_v41)
    and the text tower in text_config (deepseek_v41_text); the contract's
    family is the model's, whichever section carries quantization_config."""
    from vllm.model_executor.model_loader import mxfp4_csf_loader as reader

    write_index(tmp_path, {"weight": None}, "weights.safetensors", "deepseek_v41")
    config = json.loads((tmp_path / "config.json").read_text())
    config["text_config"] = {"model_type": "deepseek_v41_text"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert reader.checkpoint_contract(str(tmp_path))["family"] == "deepseek_v41"

    # checkpoint_contract caches by root; the nested variant gets its own.
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    write_index(nested_root, {"weight": None}, "weights.safetensors", "deepseek_v4")
    nested = {
        "model_type": "deepseek_v4",
        "text_config": {
            "model_type": "deepseek_v4_text",
            "quantization_config": config["quantization_config"],
        },
    }
    (nested_root / "config.json").write_text(json.dumps(nested))
    assert reader.checkpoint_contract(str(nested_root))["family"] == "deepseek_v4_flash"


def test_csf_tensor_reader_resolves_separate_shards_and_closes_after_failure(
    tmp_path, monkeypatch
):
    from vllm.model_executor.model_loader import csf_utils

    tensors = tmp_path / "tensors"
    tensors.mkdir()
    fixture, expected = matrix(64, 96, group_size=32, seed=9)
    weight = fixture.weight.tensor
    save_file({"arbitrary.packed": weight}, tensors / "weights.safetensors")
    components = {
        f"arbitrary.scales.csf32_{part}": torch.from_numpy(
            getattr(fixture.scales, part)
        )
        for part in ("bases", "codes", "page_bases", "offsets", "exceptions")
    }
    save_file(components, tensors / "scales.safetensors")
    index = {
        "arbitrary.packed": "tensors/weights.safetensors",
        **{name: "tensors/scales.safetensors" for name in components},
    }
    opened, closed = [], []
    original_open = csf_utils.safe_open

    class TrackedShard:
        def __init__(self, path, **kwargs):
            self.path = path
            self.shard = original_open(path, **kwargs)

        def __enter__(self):
            opened.append(self.path.name)
            return self.shard.__enter__()

        def __exit__(self, *exc):
            closed.append(self.path.name)
            return self.shard.__exit__(*exc)

    monkeypatch.setattr(csf_utils, "safe_open", TrackedShard)
    with (
        pytest.raises(RuntimeError, match="preparation failed"),
        csf_utils.CsfTensorReader(tmp_path, index, "mxfp4") as reader,
    ):
        for _ in range(2):
            source = reader.matrix("arbitrary.packed", "arbitrary.scales")
            assert source.weight.get_shape() == [64, 48]
            assert torch.equal(source.weight[8:16, 2:6], weight[8:16, 2:6])
            actual = decode_planes(SimpleNamespace(planes=(source.scales,)), 64, 3, 32)
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        raise RuntimeError("preparation failed")
    assert (
        sorted(opened)
        == sorted(closed)
        == ["scales.safetensors", "weights.safetensors"]
    )


def test_csf_tensor_reader_shapes_slab_scales_for_the_source(tmp_path):
    """A container stores one projection's slab scales flat; the reader must
    hand them to CsfSource as one segment of 16-row groups."""
    from vllm.model_executor.model_loader import csf_utils

    rows, columns = 64, 160
    rng = np.random.default_rng(5)
    bases = rng.integers(0, 254, (rows // 16, 16), dtype=np.uint8)
    bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
    fixed = np.concatenate(
        (bases, np.packbits(bits, axis=1, bitorder="little").reshape(rows // 16, -1)),
        1,
    ).reshape(-1)
    exceptions = (np.array([3, 700], dtype=np.uint32) | np.uint32(200) << 24).astype(
        np.uint32
    )
    weight = torch.zeros((rows, columns * 32 // 2), dtype=torch.int8)
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    save_file(
        {
            "w.weight": weight,
            "w.scale.mxfp4_csf_fixed": torch.from_numpy(fixed),
            "w.scale.mxfp4_csf_exceptions": torch.from_numpy(exceptions),
        },
        tensors / "shard.safetensors",
    )
    index = {
        name: "tensors/shard.safetensors"
        for name in ("w.weight", "w.scale.mxfp4_csf_fixed", "w.scale.mxfp4_csf_exceptions")
    }
    with csf_utils.CsfTensorReader(tmp_path, index, "mxfp4") as reader:
        source = reader.matrix("w.weight", "w.scale")
    plane = source.scales
    assert plane.encoding == "csf" and plane.segments == 1
    assert plane.components[0].shape == (1, rows // 16, 16 * (1 + (columns + 7) // 8))
    assert np.array_equal(plane.components[0].numpy().reshape(-1), fixed)
    assert np.array_equal(plane.components[1].numpy(), exceptions)
    assert torch.equal(plane.starts, torch.tensor([0, 2], dtype=torch.int64))
    assert plane.source_rows == rows and plane.source_columns == columns


def decode(fixed, exceptions, rows, columns):
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )[:, :columns]
    result = (stream[:, :16].reshape(rows, 1) + bits).astype(np.uint8)
    words = exceptions.numpy()
    result.flat[words & 0xFFFFFF] = words >> 24
    return result


@pytest.mark.parametrize("columns", [9, 18, 36, 72, 160])
@pytest.mark.parametrize("row_slice", [(0, 64), (0, 32), (32, 64)])
def test_slices_preserve_all_scale_bytes(columns, row_slice):
    rng = np.random.default_rng(78)
    rows = 64
    bits = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
    bases = rng.integers(0, 254, (rows // 16, 16), dtype=np.uint8)
    fixed = torch.from_numpy(
        np.concatenate(
            (
                bases,
                np.packbits(bits, axis=1, bitorder="little").reshape(rows // 16, -1),
            ),
            1,
        )
    )
    positions = np.unique(rng.integers(0, rows * columns, 50)).astype(np.uint32)
    values = rng.integers(0, 256, len(positions), dtype=np.uint32)
    exceptions = torch.from_numpy(positions | values << 24)
    from b12x.loader._csf_blocks import export_csf_blocks

    reference = decode(fixed, exceptions, rows, columns)
    plane = export_csf_blocks(
        fixed.numpy(),
        exceptions.numpy(),
        rows=rows,
        columns=columns,
        quantization="mxfp4",
    )
    for c0, c1 in ((0, columns), (1, columns), (columns // 2, columns)):
        sliced = slice_scale_plane(plane, rows, columns, row_slice, (c0, c1))
        actual = decode_planes(
            SimpleNamespace(planes=(sliced,)), row_slice[1] - row_slice[0], c1 - c0, 32
        )[0].numpy()
        assert np.array_equal(actual, reference[row_slice[0] : row_slice[1], c0:c1])


def test_rejects_unaligned_rows_and_out_of_block_records():
    source, _ = matrix(64, 288, group_size=32, seed=3)
    with pytest.raises(ValueError, match="32-row"):
        slice_scale_plane(source.scales, 64, 9, (1, 33), (0, 9))
    records = source.scales.exceptions.copy()
    records[0] |= 64
    invalid = replace(source.scales, exceptions=records)
    if not torch.cuda.is_available():
        pytest.skip("GPU payload validation requires CUDA")
    from b12x._lib.quant._csf_prepare import validate_source

    with pytest.raises(ValueError, match="positions"):
        validate_source(slice_scale_plane(invalid, 64, 9, (0, 64), (0, 9)).to("cuda"))


@pytest.mark.parametrize("tp_size,rank", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_tensor_sources_preserve_projection_order_and_tp_bytes(tp_size, rank):
    """An arbitrary geometry loads without any model identity or checkpoint path."""
    from vllm.model_executor.model_loader.mxfp4_csf_loader import (
        _load_mxfp4_csf_weights as load_mxfp4_csf_weights,
    )

    device = "cpu"
    experts, hidden, local = 3, 256, 96
    intermediate = local * tp_size
    sources, scales = [], []
    for expert in range(experts):
        pairs = [
            matrix(r, c, group_size=32, seed=expert * 17 + projection)
            for projection, (r, c) in enumerate(
                ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
            )
        ]
        sources.append(tuple(pair[0] for pair in pairs))
        scales.append(tuple(pair[1] for pair in pairs))
    scratch13 = torch.empty(
        (experts, hidden // 32, 2 * local), dtype=torch.uint8, device=device
    )
    scratch2 = torch.empty(
        (experts, local // 32, hidden), dtype=torch.uint8, device=device
    )
    weights = load_mxfp4_csf_weights(
        iter(sources),
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tp_rank=rank,
        tp_size=tp_size,
        device=device,
        w13_scale_scratch=scratch13,
        w2_scale_scratch=scratch2,
    )
    first, last = rank * local, (rank + 1) * local
    expected13 = torch.stack(
        [
            torch.cat((gate.weight[first:last, :], up.weight[first:last, :]))
            for gate, up, _ in sources
        ]
    )
    expected2 = torch.stack(
        [down.weight[:, first // 2 : last // 2] for _, _, down in sources]
    )
    assert torch.equal(weights.w13.cpu(), expected13)
    assert torch.equal(weights.w2.cpu(), expected2)
    assert (
        weights.w13_scale_scratch is scratch13 and weights.w2_scale_scratch is scratch2
    )
    for batch, expected in (
        (
            weights.w13_scales,
            torch.stack(
                [torch.cat((s[0][first:last], s[1][first:last])) for s in scales]
            ),
        ),
        (
            weights.w2_scales,
            torch.stack([s[2][:, first // 32 : last // 32] for s in scales]),
        ),
    ):
        output = decode_planes(
            batch, expected.shape[1], expected.shape[2], group_size=32
        )
        assert torch.equal(output, expected)


@pytest.mark.parametrize("count", [0, 2])
def test_expert_inventory_must_match_declared_count(count):
    from vllm.model_executor.model_loader.mxfp4_csf_loader import (
        _load_mxfp4_csf_weights as load_mxfp4_csf_weights,
    )

    source, _ = matrix(128, 128, group_size=32, seed=0)
    with pytest.raises(ValueError, match="zip"):
        load_mxfp4_csf_weights(
            [(source, source, source)] * count,
            num_experts=1,
            hidden_size=128,
            intermediate_size=128,
            tp_rank=0,
            tp_size=1,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )


# DeepSeek-V4-Flash (0731 and Vision-Exp) serving config: the source block-FP8
# fields plus the MXFP4-CSF method and container location.
DS4_FLASH_SERVING = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp4_csf",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
    "format_version": 1,
    "checkpoint_root": "/csf",
}


@pytest.fixture
def ds4_flash_moe():
    return FusedMoEConfig(
        num_experts=256,
        num_local_experts=256,
        num_logical_experts=256,
        experts_per_token=6,
        hidden_dim=4096,
        intermediate_size=1024,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )


def ds4_flash_vllm_config(
    *, pp=1, ubatching=False, load_format="mxfp4_csf", ep=False, expert_dtype="fp4"
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=43, expert_dtype=expert_dtype)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            use_ubatching=ubatching,
            enable_expert_parallel=ep,
        ),
        load_config=SimpleNamespace(load_format=load_format),
    )


def test_ds4_flash_reader_requests_native_target_expert_names(monkeypatch):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint

    requested = []

    class Reader:
        def __init__(self, root, source_names, codec):
            assert codec == "mxfp4"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def matrix(self, weight, scale):
            requested.append((weight, scale))
            return weight

    monkeypatch.setattr(
        checkpoint,
        "checkpoint_contract",
        lambda _: {
            "family": "deepseek_v4_flash",
            "source_names": {
                "layers.42.ffn.experts.0.w1.weight": "weights.safetensors"
            },
        },
    )
    monkeypatch.setattr(checkpoint, "CsfTensorReader", Reader)
    monkeypatch.setattr(
        checkpoint, "_load_mxfp4_csf_weights", lambda experts, **_: list(experts)
    )
    experts = checkpoint.read_mxfp4_csf_layer(
        "/csf",
        42,
        num_experts=256,
        hidden_size=4096,
        intermediate_size=2048,
        tp_rank=1,
        tp_size=2,
        device="cpu",
        w13_scale_scratch=None,
        w2_scale_scratch=None,
    )
    assert len(experts) == len(requested) // 3 == 256
    prefix = "layers.42.ffn.experts.255"
    assert requested[-3:] == [
        (f"{prefix}.{p}.weight", f"{prefix}.{p}.scale") for p in ("w1", "w3", "w2")
    ]
    assert experts[-1] == tuple(weight for weight, _ in requested[-3:])


@pytest.mark.parametrize(
    "layer,geometry,tp_size,error",
    [
        (43, (256, 4096, 2048), 2, "outside the compressed expert inventory"),
        (0, (256, 4096, 2048), 3, "aligned to 32"),
    ],
)
def test_ds4_flash_reader_rejects_missing_layers_and_unaligned_tp(
    monkeypatch, layer, geometry, tp_size, error
):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint

    monkeypatch.setattr(
        checkpoint,
        "checkpoint_contract",
        lambda _: {"family": "deepseek_v4_flash", "source_names": {}},
    )
    num_experts, hidden_size, intermediate_size = geometry
    with pytest.raises(ValueError, match=error):
        checkpoint.read_mxfp4_csf_layer(
            "/csf",
            layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=0,
            tp_size=tp_size,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )


@pytest.mark.parametrize(
    "family,geometry,tp_size,admitted",
    [
        ("deepseek_v41", (384, 5120, 2304), 3, True),
        ("deepseek_v41", (384, 5120, 2304), 6, True),
        ("deepseek_v4_flash", (256, 4096, 2048), 3, False),
        ("kimi_k3", (896, 3584, 3072), 3, True),
    ],
)
def test_reader_accepts_tp_from_geometry(
    monkeypatch, family, geometry, tp_size, admitted
):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint

    class PastTpGuard(Exception):
        pass

    def reader(*args, **kwargs):
        raise PastTpGuard

    monkeypatch.setattr(
        checkpoint,
        "checkpoint_contract",
        lambda _: {
            "family": family,
            "source_names": {
                checkpoint.expert_tensor_names(family, 1, 0)[0][
                    0
                ]: "weights.safetensors"
            },
        },
    )
    monkeypatch.setattr(checkpoint, "CsfTensorReader", reader)
    num_experts, hidden_size, intermediate_size = geometry
    expected = (
        pytest.raises(PastTpGuard)
        if admitted
        else pytest.raises(ValueError, match="aligned to 32")
    )
    with expected:
        checkpoint.read_mxfp4_csf_layer(
            "/csf",
            1,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_size - 1,
            tp_size=tp_size,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )


def test_mxfp4_csf_dispatches_ds4_flash_with_source_dense_config(monkeypatch):
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.mxfp4_csf import Mxfp4CsfConfig
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint
    from vllm.models.deepseek_v4.mxfp4_csf import DeepseekV4Mxfp4CsfConfig
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    monkeypatch.setattr(
        checkpoint, "checkpoint_contract", lambda _: {"family": "deepseek_v4_flash"}
    )
    owner = get_quantization_config("mxfp4_csf").from_config(DS4_FLASH_SERVING)
    source = {
        key: value
        for key, value in DS4_FLASH_SERVING.items()
        if key not in ("format_version", "checkpoint_root")
    }
    native = DeepseekV4FP8Config.from_config({**source, "quant_method": "fp8"})
    assert isinstance(owner, DeepseekV4Mxfp4CsfConfig)
    assert owner.get_name() == "mxfp4_csf" and owner.get_min_capability() == 120
    assert owner.checkpoint_root == "/csf" and owner.scale_scratch is None
    for field in (
        "is_checkpoint_fp8_serialized",
        "activation_scheme",
        "ignored_layers",
        "weight_block_size",
        "is_scale_e8m0",
    ):
        assert getattr(owner, field) == getattr(native, field)
    # The serving config must select the CSF reader, not the native FP8 path.
    hf_config = SimpleNamespace(model_type="deepseek_v4")
    assert (
        DeepseekV4FP8Config.override_quantization_method(
            DS4_FLASH_SERVING, "mxfp4_csf", hf_config=hf_config
        )
        is None
    )
    for method in (Mxfp4CsfConfig, DeepseekV4Mxfp4CsfConfig):
        assert (
            method.override_quantization_method(
                DS4_FLASH_SERVING, "mxfp4_csf", hf_config=hf_config
            )
            == "mxfp4_csf"
        )
    with pytest.raises(ValueError, match=r"\[128, 128\]"):
        DeepseekV4Mxfp4CsfConfig.from_config(
            {**DS4_FLASH_SERVING, "weight_block_size": [32, 32]}
        )
    with pytest.raises(ValueError, match="format_version"):
        DeepseekV4Mxfp4CsfConfig.from_config({**DS4_FLASH_SERVING, "format_version": 2})


@pytest.mark.parametrize("force_a16", [False, True])
def test_ds4_flash_target_experts_use_csf_while_dense_and_drafts_stay_native(
    monkeypatch, ds4_flash_moe, force_a16
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.deepseek_v4 import mxfp4_csf

    owner = mxfp4_csf.DeepseekV4Mxfp4CsfConfig.from_config(DS4_FLASH_SERVING)
    config = ds4_flash_vllm_config()
    monkeypatch.setattr(mxfp4_csf.envs, "VLLM_B12X_MOE_FP4_FORCE_A16", force_a16)
    monkeypatch.setattr(mxfp4_csf, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mxfp4_csf, "get_current_vllm_config_or_none", lambda: config)
    native = object()
    monkeypatch.setattr(
        mxfp4_csf.DeepseekV4FP8Config, "get_quant_method", lambda *a, **k: native
    )
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = ds4_flash_moe
    for prefix in (
        "model.layers.0.ffn.experts",
        "model.layers.42.ffn.experts",
        "language_model.model.layers.42.ffn.experts",
    ):
        method = owner.get_quant_method(layer, prefix)
        assert isinstance(method, Mxfp4CsfMoEMethod) and method.owner is owner
        assert method.activation_mode == ("a16" if force_a16 else "a8")
        assert method.get_fused_moe_quant_config(layer).quant_dtype == (
            None if force_a16 else "mxfp8"
        )
    # MTP and DSpark draft layers are numbered after the 43 target layers.
    for prefix in ("model.layers.43.ffn.experts", "model.layers.45.ffn.experts"):
        assert owner.get_quant_method(layer, prefix) is native
    assert owner.get_quant_method(torch.nn.Module(), "model.layers.0.attn.wq_a") is (
        native
    )


@pytest.mark.parametrize(
    "mode,error,match",
    [
        ("pipeline", NotImplementedError, "PP1 without ubatching"),
        ("ubatching", NotImplementedError, "PP1 without ubatching"),
        ("wrong_loader", ValueError, "load-format"),
        ("fp8_experts", ValueError, "MXFP4 routed experts"),
        ("expert_parallel", NotImplementedError, "expert parallelism"),
    ],
)
def test_ds4_flash_csf_rejects_unsupported_execution(
    monkeypatch, ds4_flash_moe, mode, error, match
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models import deepseek_v41
    from vllm.models.deepseek_v4 import mxfp4_csf

    owner = mxfp4_csf.DeepseekV4Mxfp4CsfConfig.from_config(DS4_FLASH_SERVING)
    config = ds4_flash_vllm_config(
        pp=2 if mode == "pipeline" else 1,
        ubatching=mode == "ubatching",
        load_format="safetensors" if mode == "wrong_loader" else "mxfp4_csf",
        ep=mode == "expert_parallel",
        expert_dtype="fp8" if mode == "fp8_experts" else "fp4",
    )
    for module in (mxfp4_csf, deepseek_v41.quant_config):
        monkeypatch.setattr(module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mxfp4_csf, "get_current_vllm_config_or_none", lambda: config)
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = ds4_flash_moe
    with pytest.raises(error, match=match):
        owner.get_quant_method(layer, "model.layers.42.ffn.experts")
    if mode == "expert_parallel":
        # MegaMoE experts never reach a quantization method.
        with pytest.raises(error, match=match):
            owner.get_quant_method(torch.nn.Module(), "model.layers.0.attn.wq_a")


def test_ds4_flash_loader_keeps_vision_draft_and_dense_tensors(tmp_path):
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    name = "model-00001-of-00001.safetensors"

    def byte():
        return torch.ones(8, dtype=torch.uint8)

    compressed = {
        "layers.0.ffn.experts.0.w1.weight": byte(),
        "layers.0.ffn.experts.0.w1.scale.csf32_codes": byte(),
        "layers.0.ffn.experts.0.w1.scale.csf32_exceptions": torch.zeros(
            1, dtype=torch.uint32
        ),
        "layers.42.ffn.experts.255.w2.weight": byte(),
    }
    draft = {
        "mtp.0.ffn.experts.0.w1.weight": torch.full((8,), 42, dtype=torch.uint8),
        "mtp.0.ffn.experts.0.w1.scale": byte(),
        "mtp.2.markov_head.markov_w1.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    retained = {
        **draft,
        "layers.0.attn.wq_a.scale": byte(),
        "layers.0.ffn.gate.bias_vl": torch.ones(8),
        "layers.0.ffn.gate.tid2eid": torch.ones(8, dtype=torch.int32),
        "layers.0.hc_attn_fn": torch.ones(8),
        "vision.blocks.0.attn.wqkv.weight": torch.ones(8, dtype=torch.bfloat16),
        "aligner.w1.weight": torch.ones(8, dtype=torch.bfloat16),
        "image_start": torch.ones(8),
    }
    tensors = {**compressed, **retained}
    save_file(tensors, tensor_dir / name)
    write_index(tmp_path, tensors, name, "deepseek_v4_flash")
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=43,
            quantization_config={**DS4_FLASH_SERVING, "checkpoint_root": str(tmp_path)},
        )
    )
    loader = Mxfp4CsfModelLoader(LoadConfig(load_format="mxfp4_csf"))
    result = dict(loader.get_all_weights(config, SimpleNamespace()))
    assert sorted(result) == sorted(retained)
    for key, value in retained.items():
        assert torch.equal(result[key], value)
    dspark = SimpleNamespace(checkpoint_weight_name_prefixes=("mtp.",))
    assert sorted(dict(loader.get_all_weights(config, dspark))) == sorted(draft)
