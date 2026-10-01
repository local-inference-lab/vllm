# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
)
from vllm.model_executor.model_loader.mxfp4_csf_loader import Mxfp4CsfModelLoader
from vllm.models.deepseek_v4_1.mxfp4_csf import (
    DeepseekV41Mxfp4CsfConfig,
    Mxfp4CsfMoEMethod,
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


def test_csf_cpu_shard_reader_bounds_threads_and_restores_on_failure(
    monkeypatch, moe, owner
):
    from vllm.model_executor.model_loader import mxfp4_csf_loader as checkpoint
    from vllm.models.deepseek_v4_1 import mxfp4_csf
    from vllm.utils.torch_utils import set_default_torch_num_threads

    monkeypatch.setattr(mxfp4_csf, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(mxfp4_csf, "get_tensor_model_parallel_rank", lambda: 0)
    method = Mxfp4CsfMoEMethod(moe, owner)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.ffn.experts"
    method.create_weights(layer, 384, 5120, 576, torch.bfloat16)

    def fail_reader(*args, **kwargs):
        assert torch.get_num_threads() == 1
        raise OSError("checkpoint read failed")

    monkeypatch.setattr(checkpoint, "read_mxfp4_csf_layer", fail_reader)
    with set_default_torch_num_threads(4):
        with pytest.raises(OSError, match="checkpoint read failed"):
            method.process_weights_after_loading(layer)
        assert torch.get_num_threads() == 4


@pytest.mark.parametrize("mode", ["normal", "pipeline", "ubatching", "wrong_loader"])
def test_target_uses_csf_while_dense_and_draft_keep_native_methods(
    monkeypatch, moe, owner, mode
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.deepseek_v4_1 import mxfp4_csf

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
        assert isinstance(
            owner.get_quant_method(layer, "model.layers.39.ffn.experts"),
            Mxfp4CsfMoEMethod,
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


def test_prepared_native_experts_validate_source_shape_and_swiglu_limit(
    monkeypatch, moe, owner
):
    from b12x.moe import fused_moe

    from vllm.model_executor.layers.fused_moe import b12x

    layer = torch.nn.Module()
    layer.activation = moe.activation
    layer.apply_router_weight_on_input = False
    quant = Mxfp4CsfMoEMethod(moe, owner).get_fused_moe_quant_config(layer)
    backend = b12x.B12xExperts(moe, quant)
    prepared = Mock(spec=fused_moe.PreparedExperts)
    activation = fused_moe.ActivationSpec(
        mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
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
    from vllm.model_executor.model_loader.mxfp4_csf_loader import CODEC, SCHEMA

    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    name = "model-00001.safetensors"
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": torch.ones(8, dtype=torch.uint8),
        "layers.0.ffn.experts.0.w1.scale.mxfp4_csf_fixed": torch.ones(
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
    assert isinstance(
        owner.get_quant_method(experts, "model.layers.1.mlp.experts"),
        Mxfp4CsfMoEMethod,
    )
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
    from vllm.model_executor.model_loader.mxfp4_csf_loader import CODEC, SCHEMA

    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    filename = "model-00001.safetensors"
    expert = "language_model.model.layers.1.block_sparse_moe.experts.0.w1."
    dense = "language_model.model.layers.0.mlp.gate_proj.weight"
    tensors = {
        expert + "weight_packed": torch.ones(8, dtype=torch.uint8),
        expert + "weight_scale.mxfp4_csf_fixed": torch.ones(8, dtype=torch.uint8),
        dense: torch.ones((8, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, tensor_dir / filename)
    common = {"schema": SCHEMA, "codec": CODEC, "family": "kimi_k3"}
    (tmp_path / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": filename}]})
    )
    (tmp_path / "build-contract.json").write_text(
        json.dumps({**common, "source_names": {k: filename for k in tensors}})
    )
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


@pytest.mark.parametrize("codec", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("fault", ["schema", "codec", "family", "inventory", "path"])
def test_csf_contract_rejects_mismatched_identity_and_shard_inventory(
    tmp_path, codec, fault
):
    from vllm.model_executor.model_loader import mxfp4_csf_loader, nvfp4_csf_loader

    reader = mxfp4_csf_loader if codec == "mxfp4" else nvfp4_csf_loader
    identity = {
        "schema": reader.SCHEMA,
        "codec": reader.CODEC,
        "family": next(iter(reader.FAMILIES)),
    }
    manifest = {**identity, "shards": [{"file": "weights.safetensors"}]}
    contract = {**identity, "source_names": {"weight": "weights.safetensors"}}
    if fault in ("schema", "codec", "family"):
        contract[fault] = "unrecognized"
    elif fault == "inventory":
        contract["source_names"]["weight"] = "missing.safetensors"
    else:
        contract["source_names"]["weight"] = "../weights.safetensors"
        manifest["shards"][0]["file"] = "../weights.safetensors"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "build-contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        reader.checkpoint_contract(str(tmp_path))


def test_csf_tensor_reader_resolves_separate_shards_and_closes_after_failure(
    tmp_path, monkeypatch
):
    from vllm.model_executor.model_loader import csf_utils

    tensors = tmp_path / "tensors"
    tensors.mkdir()
    weight = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
    fixed = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    exceptions = torch.tensor([0xFE000009], dtype=torch.uint32)
    save_file({"arbitrary.packed": weight}, tensors / "weights.safetensors")
    save_file(
        {
            "arbitrary.scales.mxfp4_csf_fixed": fixed,
            "arbitrary.scales.mxfp4_csf_exceptions": exceptions,
        },
        tensors / "scales.safetensors",
    )
    index = {
        "arbitrary.packed": "weights.safetensors",
        "arbitrary.scales": "scales.safetensors",
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
            assert source.weight.get_shape() == [16, 8]
            assert torch.equal(source.weight[8:16, 2:6], weight[8:16, 2:6])
            assert torch.equal(source.fixed, fixed)
            assert torch.equal(source.exceptions, exceptions)
        raise RuntimeError("preparation failed")
    assert (
        sorted(opened)
        == sorted(closed)
        == ["scales.safetensors", "weights.safetensors"]
    )
