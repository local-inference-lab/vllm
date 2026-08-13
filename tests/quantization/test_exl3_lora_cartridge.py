# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for CUDA-graph-safe EXL3 MSRT cartridges."""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file as _save_file

import vllm.model_executor.layers.quantization.exl3 as exl3_module
import vllm.model_executor.layers.quantization.exl3_lora_cartridge as cartridge_module
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.model_executor.layers.quantization.exl3_lora_cartridge import (
    Exl3CUDAGraphCartridgeRuntime,
    Exl3LoraCartridge,
    apply_exl3_cudagraph_cartridge,
    prepare_exl3_cudagraph_cartridge_runtime,
    stage_exl3_cartridge_adapter,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

CPU = torch.device("cpu")
BASE_COMPATIBILITY_SHA256 = "a" * 64


def _canonical_tensor_record(name, tensor):
    logical_name = re.sub(r"\.rank0(?=\.)", "", name)
    raw = tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    dtype = {
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int16: "I16",
        torch.int32: "I32",
    }[tensor.dtype]
    return {
        "name": logical_name,
        "dtype": dtype,
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rank_sliced_metadata(tensors_by_layer):
    compatibility_by_layer = {}
    for layer, tensors in tensors_by_layer.items():
        records = [
            _canonical_tensor_record(name, tensor) for name, tensor in tensors.items()
        ]
        payload = {
            "schema": "fq-msrt-base-layer-compatibility/1",
            "k": 2,
            "layer": layer,
            "tensors": sorted(records, key=lambda record: record["name"]),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        compatibility_by_layer[str(layer)] = hashlib.sha256(encoded).hexdigest()
    coverage = sorted(tensors_by_layer)
    root_payload = {
        "schema": "fq-msrt-base-compatibility/2",
        "k": 2,
        "layers": compatibility_by_layer,
    }
    encoded_root = json.dumps(
        root_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return {
        "format": "exl3-trellis",
        "runtime_profile": "exl3-msrt-base/1",
        "bits": 2,
        "codebook": "mcg",
        "experts_per_layer": 2,
        "moe_layers": [coverage[0], coverage[-1]],
        "moe_layer_coverage": coverage,
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        ),
        "tensor_parallel": dict(exl3_module._RANK_SLICED_FULL_TP),
        "mcg_multiplier": 3417055213,
        "compatibility_by_layer": compatibility_by_layer,
        "compatibility_sha256": hashlib.sha256(encoded_root).hexdigest(),
    }


def _producer_valid_base_layer(layer=3):
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.layers.{layer}.mlp.experts.0.{projection}.rank0"
        tensors[f"{prefix}.trellis"] = torch.zeros(8, 8, 32, dtype=torch.int16)
        tensors[f"{prefix}.suh"] = torch.zeros(128, dtype=torch.float16)
        tensors[f"{prefix}.svh"] = torch.zeros(128, dtype=torch.float16)
        tensors[f"{prefix}.mcg"] = torch.tensor(-877912083, dtype=torch.int32)
    return tensors


def save_file(tensors, path, *, layout=None):
    """Write one test shard and its required v3 adapter contract."""
    path = Path(path)
    _save_file(tensors, path)
    labels = {key.rsplit(".trellis_", 1)[1] for key in tensors if ".trellis_" in key}

    def stage_key(label):
        match = re.match(r"^(.*?)(\d+)$", label)
        return (match.group(1), int(match.group(2))) if match else (label, -1)

    ordered = sorted(labels, key=stage_key)
    ranks = {
        int(match.group(1))
        for key in tensors
        if (match := re.search(r"\.rank(\d+)\.", key))
    }
    rank_sharded = layout == "rank-sharded" or (
        layout is None and bool(ranks) and ranks != {0}
    )
    parent = "k2"
    chain = []
    for label in ordered:
        trellis = next(
            tensor
            for key, tensor in tensors.items()
            if key.endswith(f".trellis_{label}")
        )
        chain.append(
            {
                "label": label,
                "k": max(1, trellis.shape[-1] // 16),
                "parent": parent,
                "experts": "all",
            }
        )
        parent = label
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = {}
    selected_layers = set()
    selected_experts = set()
    for key in tensors:
        target = re.match(
            r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.",
            key,
        )
        if target is None or ".trellis_" not in key:
            continue
        layer_id, expert_id = map(int, target.groups())
        label = key.rsplit(".trellis_", 1)[1]
        selected_layers.add(layer_id)
        selected_experts.add(expert_id)
        by_layer = coverage.setdefault(label, {})
        by_layer.setdefault(str(layer_id), set()).add(expert_id)
    coverage = {
        label: {
            layer_id: sorted(experts) for layer_id, experts in sorted(by_layer.items())
        }
        for label, by_layer in sorted(
            coverage.items(), key=lambda item: stage_key(item[0])
        )
    }
    config = {
        "schema": "fq-cartridge-adapter/3",
        "assembly": "test",
        "format": "exl3-msrt-packed",
        "runtime_profile": "exl3-msrt-additive/1",
        "rotation_ownership": "base",
        "standard_lora_compatible": False,
        "codebook": "mcg",
        "mcg_ownership": "adapter-config",
        "runtime_operation": ("base_exl3_gemm + sum(stage_exl3_gemm / stage_scale)"),
        "mcg_multiplier": 3417055213,
        "scale_shape": [],
        "base": {
            "label": "k2",
            "k": 2,
            "compatibility_sha256": BASE_COMPATIBILITY_SHA256,
            "compatibility_by_layer": {
                str(layer): "b" * 64 for layer in sorted(selected_layers)
            },
        },
        "chain": chain,
        "tensor_parallel": {
            "layout": "rank-sharded" if rank_sharded else "full",
            "world_size": len(ranks) if rank_sharded else 1,
            "ranks": sorted(ranks) if rank_sharded else [0],
            "axis_by_projection": {
                "gate_proj": "output",
                "up_proj": "output",
                "down_proj": "input",
            },
        },
        "shards": [{"path": path.name, "size": path.stat().st_size, "sha256": digest}],
        "num_tensors": len(tensors),
        "selected_experts": sorted(selected_experts),
        "selected_layers": sorted(selected_layers),
        "coverage": coverage,
        "producer_verified_signer": None,
        "campaign": {
            "recipe_sha256": "e" * 64,
            "base_model": "test",
            "base_revision": "test",
            "encoder_sha256": None,
            "signer_pubkey": "f" * 64,
            "block_size": 128,
            "moe_layers": sorted(selected_layers),
        },
        "source_assembly": {
            "path": "assemblies/test/assembly.jsonl",
            "sha256": "d" * 64,
        },
        "tool_version": "test",
        "created_utc": "2026-01-01T00:00:00Z",
    }
    (path.parent / "adapter_config.json").write_text(json.dumps(config))


def load_cartridge_from_adapter(adapter_path, layer, num_experts, device):
    layer.exl3_base_compatibility_verified = True
    layer.exl3_base_layer_compatibility_sha256 = "b" * 64
    with stage_exl3_cartridge_adapter(
        SimpleNamespace(modules=lambda: iter((layer,))), adapter_path
    ) as staged:
        return cartridge_module._load_cartridge_from_staged_adapter(
            staged, adapter_path, layer, num_experts, device
        )


def prepare_exl3_cartridge_into_model(model, adapter_path):
    for layer in model.modules():
        layer.exl3_base_compatibility_verified = True
        layer.exl3_base_layer_compatibility_sha256 = "b" * 64
    with stage_exl3_cartridge_adapter(model, adapter_path) as staged:
        return cartridge_module.prepare_staged_exl3_cartridge_into_model(model, staged)


def _runtime_layer(device: torch.device = CPU):
    pointers = tuple(torch.zeros(2, dtype=torch.int64, device=device) for _ in range(9))
    return SimpleNamespace(
        local_num_experts=2,
        exl3_hidden_size=4,
        exl3_intermediate_size_per_partition=2,
        exl3_params_dtype=torch.float16,
        exl3_cartridge_capable=True,
        exl3_cartridge_enabled=False,
        exl3_tp_size=1,
        exl3_tp_rank=0,
        exl3_base_compatibility_sha256=BASE_COMPATIBILITY_SHA256,
        exl3_base_compatibility_by_layer={"3": "b" * 64, "4": "b" * 64},
        exl3_base_compatibility_verified=True,
        exl3_base_layer_compatibility_sha256="b" * 64,
        exl3_max_num_batched_tokens=16,
        exl3_layer_bitrates=(2, 2),
        exl3_pointer_tables=pointers,
        top_k=1,
        w13_trellis=SimpleNamespace(device=device),
        activation=SimpleNamespace(value="silu"),
        layer_name="model.layers.3.mlp.experts",
    )


def _loader_layer():
    layer = _runtime_layer()
    layer.exl3_hidden_size = 16
    layer.exl3_intermediate_size_per_partition = 16
    rotations = {
        (0, "w1"): torch.ones(128, dtype=torch.float16),
        (0, "w3"): torch.ones(128, dtype=torch.float16),
    }
    layer.w13_suh = SimpleNamespace(exl3_tensors=rotations)
    layer.w13_svh = SimpleNamespace(exl3_tensors=rotations)
    down_rotations = {(0, "w2"): torch.ones(128, dtype=torch.float16)}
    layer.w2_suh = SimpleNamespace(exl3_tensors=down_rotations)
    layer.w2_svh = SimpleNamespace(exl3_tensors=down_rotations)
    return layer


def _loader_tensors(*, ranks=(0,), label="res1", bits=2, layer=3):
    tensors = {}
    for rank in ranks:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.{layer}.mlp.experts.0.{projection}.rank{rank}"
            tensors[f"{prefix}.trellis_{label}"] = torch.zeros(
                8,
                8,
                bits * 16,
                dtype=torch.int16,
            )
            tensors[f"{prefix}.scale_{label}"] = torch.tensor(1.0)
    return tensors


def _cartridge(device: torch.device = CPU, num_stages: int = 1):
    cartridge = Exl3LoraCartridge(num_stages, 2, device)
    for stage_idx in range(num_stages):
        for expert_id in range(2):
            for shard_id in ("w1", "w3", "w2"):
                cartridge.set_stage_tensors(
                    stage_idx,
                    expert_id,
                    shard_id,
                    torch.zeros(1, 1, 16, dtype=torch.int16),
                    float(stage_idx + 1),
                )
    cartridge.active = True
    return cartridge


def test_checkpoint_metadata_cannot_enable_cartridge_runtime():
    config = Exl3Config.from_config({"cartridge_runtime": True})
    assert not hasattr(config, "cartridge_runtime")


def test_base_compatibility_identity_is_byte_exact_and_layer_scoped():
    tensors_by_layer = {
        3: _producer_valid_base_layer(3),
        78: _producer_valid_base_layer(78),
    }
    metadata = _rank_sliced_metadata(tensors_by_layer)
    assert (
        metadata["compatibility_by_layer"]["3"]
        == "3518095ac1ae5fd32f775862acce5daa5f0c8a4c96fb308b45df4e95da377d58"
    )
    single_layer_metadata = _rank_sliced_metadata({3: _producer_valid_base_layer(3)})
    assert (
        single_layer_metadata["compatibility_sha256"]
        == "cbb1f591900487ad16f33fd2d33174beb09d55d301f164aff37ed829f712ce01"
    )

    for layer_id in (3, 78):
        config = Exl3Config()
        config._configure_rank_sliced(metadata)
        for name, tensor in tensors_by_layer[layer_id].items():
            config.record_rank_sliced_compatibility_tensor(name, tensor)
        assert (
            config.verified_rank_sliced_layer_compatibility_sha256(
                f"wrapped.model.layers.{layer_id}.mlp.experts"
            )
            == metadata["compatibility_by_layer"][str(layer_id)]
        )

    changed = Exl3Config()
    changed._configure_rank_sliced(metadata)
    for index, (name, tensor) in enumerate(tensors_by_layer[3].items()):
        changed.record_rank_sliced_compatibility_tensor(
            name,
            tensor.clone().add_(1) if index == 0 else tensor,
        )
    with pytest.raises(ValueError, match="do not match compatibility digest"):
        changed.verified_rank_sliced_layer_compatibility_sha256(
            "model.layers.3.mlp.experts"
        )


def test_base_compatibility_root_rejects_map_mismatch():
    tensors = {
        3: {
            "model.layers.3.mlp.experts.0.gate_proj.rank0.mcg": torch.tensor(
                -877912083, dtype=torch.int32
            )
        }
    }
    metadata = _rank_sliced_metadata(tensors)
    metadata["compatibility_by_layer"]["3"] = "f" * 64

    with pytest.raises(ValueError, match="does not match its compatibility"):
        Exl3Config()._configure_rank_sliced(metadata)


def test_base_runtime_profile_is_closed():
    metadata = _rank_sliced_metadata({3: _producer_valid_base_layer(3)})
    metadata["unknown"] = True

    with pytest.raises(ValueError, match="closed exl3-msrt-base/1"):
        Exl3Config()._configure_rank_sliced(metadata)


def test_runtime_accepts_tensor_parallel_weights():
    layer = _runtime_layer()
    layer.exl3_tp_rank = 1
    layer.exl3_tp_size = 2

    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)

    assert runtime.tp_rank == 1
    assert runtime.tp_size == 2
    assert runtime.ig.shape[-1] == layer.exl3_intermediate_size_per_partition


def test_runtime_rejects_invalid_tensor_parallel_rank():
    layer = _runtime_layer()
    layer.exl3_tp_rank = 2
    layer.exl3_tp_size = 2

    with pytest.raises(ValueError, match="rank=2, size=2"):
        prepare_exl3_cudagraph_cartridge_runtime(layer)


def test_runtime_allocates_packed_kernel_workspaces_without_dense_weights():
    layer = _runtime_layer()
    layer.exl3_params_dtype = torch.bfloat16
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    assert runtime.dtype == torch.float16
    assert runtime._packed_tensors == ()
    assert not hasattr(runtime, "w13")
    assert not hasattr(runtime, "w2")
    assert runtime.xh.shape == (16, 4)
    assert runtime.ig.shape[-2:] == (16, 2)


def test_runtime_rejects_extension_without_additive_entrypoint():
    layer = _runtime_layer()
    layer.w13_trellis = SimpleNamespace(device=torch.device("cuda"))
    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "_load_exl3_ext",
            return_value=SimpleNamespace(EXL3_MOE_ADDITIVE_ABI_VERSION=1),
        ),
        pytest.raises(RuntimeError, match="exl3_moe_additive_fused"),
    ):
        Exl3CUDAGraphCartridgeRuntime(layer)


@pytest.mark.parametrize("abi", [None, 0, 2, "1"])
def test_runtime_rejects_missing_or_wrong_additive_abi(abi):
    extension = SimpleNamespace(exl3_moe_additive_fused=MagicMock())
    if abi is not None:
        extension.EXL3_MOE_ADDITIVE_ABI_VERSION = abi
    with (
        patch.object(exl3_module, "_EXL3_EXT", extension),
        patch.object(cartridge_module, "_load_exl3_ext", return_value=extension),
        pytest.raises(RuntimeError, match="EXL3_MOE_ADDITIVE_ABI_VERSION=1"),
    ):
        cartridge_module._load_additive_exl3_ext()


def test_runtime_rejects_activation_before_materialization():
    runtime = prepare_exl3_cudagraph_cartridge_runtime(_runtime_layer())
    with pytest.raises(RuntimeError, match="unmaterialized"):
        runtime.activate()


def test_cartridge_validates_indices_and_scale():
    cartridge = Exl3LoraCartridge(1, 2, torch.device("cpu"))
    tensors = (torch.zeros(1, 1, 16, dtype=torch.int16),)
    with pytest.raises(IndexError, match="stage index"):
        cartridge.set_stage_tensors(1, 0, "w1", *tensors, 1.0)
    with pytest.raises(IndexError, match="expert index"):
        cartridge.set_stage_tensors(0, 2, "w1", *tensors, 1.0)
    with pytest.raises(ValueError, match="unsupported EXL3 shard"):
        cartridge.set_stage_tensors(0, 0, "bad", *tensors, 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        cartridge.set_stage_tensors(0, 0, "w1", *tensors, 0.0)


def test_runtime_retains_packed_stages_and_builds_kernel_metadata():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    cartridge = _cartridge()

    runtime.materialize(layer, cartridge)

    assert len(runtime.pointer_args) == 9
    assert len(runtime._packed_tensors) == 15
    assert all(table.shape == (1, 2) for table in runtime.pointer_args[:6])
    assert all(table.shape == (1,) for table in runtime.pointer_args[6:])
    assert runtime.max_residual_bits == 1
    gate_scales, up_scales, down_scales = runtime.pointer_args[3:6]
    expected_scales = torch.ones(1, 2, dtype=torch.float32)
    assert torch.equal(gate_scales, expected_scales)
    assert torch.equal(up_scales, expected_scales)
    assert torch.equal(down_scales, expected_scales)
    assert runtime._active is False


def test_runtime_builds_multistage_extension_metadata():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    runtime.materialize(layer, _cartridge(num_stages=2))

    pointers = runtime.pointer_args[:3]
    scales = runtime.pointer_args[3:6]
    bitrates = runtime.pointer_args[6:]
    assert all(table.shape == (2, 2) for table in pointers)
    assert all(
        torch.equal(
            table,
            torch.tensor([[1.0, 1.0], [0.5, 0.5]], dtype=torch.float32),
        )
        for table in scales
    )
    assert all(
        torch.equal(table, torch.tensor([1, 1], dtype=torch.int32))
        for table in bitrates
    )


def test_runtime_zero_scales_sparse_stage_entries():
    layer = _runtime_layer()
    cartridge = _cartridge(num_stages=2)
    for shard_id in ("w1", "w3", "w2"):
        del cartridge.stages[1][(1, shard_id)]
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    runtime.materialize(layer, cartridge)

    for pointers, scales in zip(
        runtime.pointer_args[:3], runtime.pointer_args[3:6], strict=True
    ):
        assert pointers[1, 1].item() == pointers[1, 0].item()
        assert torch.equal(
            scales,
            torch.tensor([[1.0, 1.0], [0.5, 0.0]], dtype=torch.float32),
        )


def test_runtime_rejects_nonfinite_inverse_scale():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    cartridge = _cartridge()
    for stage in cartridge.stages:
        for tensors in stage.values():
            tensors["scale"] = 1e-40

    with pytest.raises(ValueError, match="inverse scale is not finite in FP32"):
        runtime.materialize(layer, cartridge)

    assert runtime._active is False


def test_create_weights_supports_full_rank_base_at_tp2(monkeypatch):
    identity_tensors = {
        3: {
            "model.layers.3.mlp.experts.0.gate_proj.rank0.mcg": torch.tensor(
                -877912083, dtype=torch.int32
            )
        }
    }
    quant_config = Exl3Config()
    quant_config._configure_rank_sliced(_rank_sliced_metadata(identity_tensors))
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False, tp_rank=1, tp_size=2),
        has_bias=False,
    )
    method = Exl3MoEMethod(quant_config, moe)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        use_v2_model_runner=False,
        lora_config=None,
        model_config=SimpleNamespace(runner_type="generate"),
    )
    monkeypatch.setattr(
        exl3_module,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(exl3_module.envs, "VLLM_ENABLE_EXL3_CARTRIDGE", True)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=128,
        params_dtype=torch.float16,
    )

    assert layer.exl3_tp_rank == 1
    assert layer.exl3_tp_size == 2
    assert layer.exl3_cartridge_capable is True
    assert layer.exl3_cartridge_enabled is False
    assert (
        layer.exl3_base_compatibility_sha256
        == quant_config.rank_sliced_metadata["compatibility_sha256"]
    )
    assert isinstance(layer.w13_trellis, exl3_module.Exl3MoEParameter)
    assert layer.w13_trellis.exl3_tp_slice == (1, 128, 128, True)
    assert layer.w2_trellis.exl3_tp_slice == (0, 128, 128, True)
    assert (
        quant_config.normalize_rank_sliced_weight_name(
            "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis"
        )
        == "model.layers.3.mlp.experts.0.gate_proj.trellis"
    )
    assert (
        quant_config.normalize_rank_sliced_weight_name(
            "model.layers.3.mlp.experts.0.gate_proj.rank1.trellis"
        )
        is None
    )

    full_trellis = torch.cat(
        (
            torch.ones(8, 8, 32, dtype=torch.int16),
            torch.full((8, 8, 32), 2, dtype=torch.int16),
        ),
        dim=1,
    )
    layer.w13_trellis.load_exl3_weight(
        full_trellis,
        expert_id=0,
        shard_id="w1",
    )
    assert torch.equal(
        layer.w13_trellis.exl3_tensors[(0, "w1")],
        torch.full((8, 8, 32), 2, dtype=torch.int16),
    )

    full_rotation = torch.cat(
        (
            torch.ones(128, dtype=torch.float16),
            torch.full((128,), 2, dtype=torch.float16),
        )
    )
    layer.w13_svh.load_exl3_weight(
        full_rotation,
        expert_id=0,
        shard_id="w1",
    )
    assert torch.equal(
        layer.w13_svh.exl3_tensors[(0, "w1")],
        torch.full((128,), 2, dtype=torch.float16),
    )


def test_rank_sliced_draft_layer_skips_cartridge_path():
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(
        rank_sliced_metadata={"tp": 1},
        rank_sliced_supports_dynamic_tp=MagicMock(return_value=False),
    )
    method._apply_rank_sliced = MagicMock(
        return_value=torch.ones(2, 4, dtype=torch.float16)
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        expert_map=None,
        apply_router_weight_on_input=False,
        exl3_cartridge_enabled=False,
    )
    x = torch.ones(2, 4, dtype=torch.float16)
    weights = torch.ones(2, 1, dtype=torch.float32)
    ids = torch.zeros(2, 1, dtype=torch.long)

    with patch(
        "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
        "apply_exl3_cudagraph_cartridge"
    ) as cartridge:
        output = method.apply(layer, x, weights, ids, None, None)

    assert output.shape == (2, 4)
    cartridge.assert_not_called()


def test_active_rank_sliced_layer_skips_compressed_base_path():
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(rank_sliced_metadata={"tp": 1})
    method._apply_rank_sliced = MagicMock()
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        expert_map=None,
        apply_router_weight_on_input=False,
        exl3_cartridge_enabled=True,
    )
    x = torch.ones(2, 4, dtype=torch.float16)
    weights = torch.ones(2, 1, dtype=torch.float32)
    ids = torch.zeros(2, 1, dtype=torch.long)

    with patch(
        "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
        "apply_exl3_cudagraph_cartridge",
        return_value=torch.ones(2, 4, dtype=torch.float16),
    ) as cartridge:
        output = method.apply(layer, x, weights, ids, None, None)

    assert output.shape == (2, 4)
    method._apply_rank_sliced.assert_not_called()
    cartridge.assert_called_once()
    call_x, call_weights, call_ids, call_layer = cartridge.call_args.args
    assert torch.equal(call_x, x)
    assert torch.equal(call_weights, weights)
    assert torch.equal(call_ids, ids)
    assert call_layer is layer


def test_graph_path_routes_original_ids_once():
    layer = _runtime_layer()
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    runtime.pointer_args = tuple(torch.zeros(2, dtype=torch.int64) for _ in range(9))
    runtime._materialized = True
    runtime.activate()
    inputs = torch.zeros(3, 4, dtype=torch.float16)
    weights = torch.ones(3, 1, dtype=torch.float32)
    ids = torch.tensor([[1], [0], [1]], dtype=torch.long)

    def run_additive(*args):
        args[1].fill_(torch.nan)

    additive = MagicMock(side_effect=run_additive)
    runtime.ext = SimpleNamespace(exl3_moe_additive_fused=additive)
    output = apply_exl3_cudagraph_cartridge(inputs, weights, ids, layer)

    assert torch.isnan(output).all()
    additive.assert_called_once()
    assert torch.equal(additive.call_args.args[2], ids)


def test_graph_path_accepts_empty_route_batch():
    layer = _runtime_layer()
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    runtime._materialized = True
    runtime.activate()
    runtime.ext = SimpleNamespace(exl3_moe_additive_fused=MagicMock())

    output = apply_exl3_cudagraph_cartridge(
        torch.empty(0, 4, dtype=torch.float16),
        torch.empty(0, 1, dtype=torch.float32),
        torch.empty(0, 1, dtype=torch.long),
        layer,
    )

    assert output.shape == (0, 4)
    runtime.ext.exl3_moe_additive_fused.assert_not_called()


def test_packed_runtime_retains_trellis_storage_after_cartridge_clear():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    cartridge = _cartridge()
    source_trellis = cartridge.get_stage_tensors(0, 0, "w1")["trellis"]
    assert isinstance(source_trellis, torch.Tensor)

    runtime.materialize(layer, cartridge)
    pointer = source_trellis.data_ptr()
    cartridge.clear()

    assert any(tensor.data_ptr() == pointer for tensor in runtime._packed_tensors)
    assert runtime.pointer_args[0][0, 0].item() == pointer


def test_loader_filters_layer_and_sorts_stage_numbers(tmp_path):
    path = tmp_path / "cartridge.safetensors"

    def tensors_for(layer: int, value: int):
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.{layer}.mlp.experts.0.{projection}.rank0"
            tensors.update(
                {
                    f"{prefix}.trellis_res{value}": torch.full(
                        (8, 8, 32), value, dtype=torch.int16
                    ),
                    f"{prefix}.scale_res{value}": torch.tensor(float(value)),
                }
            )
        return tensors

    tensors = {}
    tensors.update(tensors_for(3, 10))
    tensors.update(tensors_for(3, 2))
    tensors.update(tensors_for(4, 20))
    tensors.update(tensors_for(4, 10))
    tensors.update(tensors_for(4, 2))
    save_file(tensors, path)

    cartridge = load_cartridge_from_adapter(
        str(path), _loader_layer(), 2, torch.device("cpu")
    )

    assert cartridge is not None
    assert cartridge.num_stages == 2
    stage0 = cartridge.get_stage_tensors(0, 0, "w1")
    stage1 = cartridge.get_stage_tensors(1, 0, "w1")
    assert stage0 is not None and stage0["scale"] == 2.0
    assert stage1 is not None and stage1["scale"] == 10.0


def _tp_loader_layer(rank: int):
    layer = _loader_layer()
    layer.exl3_hidden_size = 128
    layer.exl3_intermediate_size_per_partition = 128
    layer.exl3_tp_rank = rank
    layer.exl3_tp_size = 2
    return layer


def test_loader_slices_legacy_full_rank_cartridge_for_each_tp_worker(tmp_path):
    path = tmp_path / "full-rank.safetensors"
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        dim = 0 if projection == "down_proj" else 1
        trellis = torch.cat(
            (
                torch.ones(8, 8, 16, dtype=torch.int16),
                torch.full((8, 8, 16), 2, dtype=torch.int16),
            ),
            dim=dim,
        )
        prefix = f"model.layers.3.mlp.experts.0.{projection}.rank0"
        tensors[f"{prefix}.trellis_res1"] = trellis
        tensors[f"{prefix}.scale_res1"] = torch.tensor(3.0)
    save_file(tensors, path)

    rank0 = load_cartridge_from_adapter(str(path), _tp_loader_layer(0), 2, CPU)
    rank1 = load_cartridge_from_adapter(str(path), _tp_loader_layer(1), 2, CPU)

    assert rank0 is not None and rank1 is not None
    for shard_id in ("w1", "w3", "w2"):
        rank0_tensors = rank0.get_stage_tensors(0, 0, shard_id)
        rank1_tensors = rank1.get_stage_tensors(0, 0, shard_id)
        assert rank0_tensors is not None and rank1_tensors is not None
        assert rank0_tensors["trellis"].shape == (8, 8, 16)
        assert rank1_tensors["trellis"].shape == (8, 8, 16)
        assert torch.equal(
            rank0_tensors["trellis"],
            torch.ones(8, 8, 16, dtype=torch.int16),
        )
        assert torch.equal(
            rank1_tensors["trellis"],
            torch.full((8, 8, 16), 2, dtype=torch.int16),
        )


def test_loader_selects_matching_rank_from_tp_sharded_cartridge(tmp_path):
    path = tmp_path / "rank-sharded.safetensors"
    tensors = {}
    for rank in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.3.mlp.experts.0.{projection}.rank{rank}"
            tensors[f"{prefix}.trellis_res1"] = torch.full(
                (8, 8, 16), rank + 1, dtype=torch.int16
            )
            tensors[f"{prefix}.scale_res1"] = torch.tensor(2.0)
    save_file(tensors, path)

    layer = _tp_loader_layer(1)
    cartridge = load_cartridge_from_adapter(str(path), layer, 2, CPU)

    assert cartridge is not None
    for shard_id in ("w1", "w3", "w2"):
        stage = cartridge.get_stage_tensors(0, 0, shard_id)
        assert stage is not None
        assert stage["scale"] == 2.0
        assert torch.equal(
            stage["trellis"],
            torch.full((8, 8, 16), 2, dtype=torch.int16),
        )

    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    runtime.materialize(layer, cartridge)
    assert runtime.pointer_args[0].shape == (1, 2)
    assert runtime.pointer_args[3].shape == (1, 2)
    assert runtime.pointer_args[6].shape == (1,)


def test_loader_honors_rank_sharded_world_size_one(tmp_path):
    path = tmp_path / "rank-sharded-tp1.safetensors"
    save_file(_loader_tensors(bits=1), path, layout="rank-sharded")

    cartridge = load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)

    assert cartridge is not None
    assert cartridge.get_stage_tensors(0, 0, "w1") is not None


def test_loader_rejects_divergent_replicated_rank_scales(tmp_path):
    path = tmp_path / "rank-scales.safetensors"
    tensors = _loader_tensors(ranks=(0, 1), bits=1)
    tensors["model.layers.3.mlp.experts.0.gate_proj.rank1.scale_res1"] = torch.tensor(
        2.0
    )
    save_file(tensors, path)

    with pytest.raises(ValueError, match="scales differ across TP ranks"):
        load_cartridge_from_adapter(str(path), _tp_loader_layer(1), 2, CPU)


def test_loader_rejects_inconsistent_tp_rank_topology(tmp_path):
    path = tmp_path / "inconsistent-ranks.safetensors"
    tensors = {}
    for rank in range(2):
        projections = (
            ("gate_proj", "up_proj", "down_proj")
            if rank == 0
            else ("gate_proj", "up_proj")
        )
        for projection in projections:
            prefix = f"model.layers.3.mlp.experts.0.{projection}.rank{rank}"
            tensors[f"{prefix}.trellis_res1"] = torch.zeros(8, 8, 16, dtype=torch.int16)
            tensors[f"{prefix}.scale_res1"] = torch.tensor(1.0)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="inconsistent stage topology"):
        load_cartridge_from_adapter(str(path), _tp_loader_layer(1), 2, CPU)


def test_loader_rejects_missing_scale_on_nonlocal_tp_rank(tmp_path):
    path = tmp_path / "missing-scale.safetensors"
    tensors = _loader_tensors(ranks=(0, 1), bits=1)
    del tensors["model.layers.3.mlp.experts.0.gate_proj.rank0.scale_res1"]
    save_file(tensors, path)

    with pytest.raises(ValueError, match="has no scalar scale companion"):
        load_cartridge_from_adapter(str(path), _tp_loader_layer(1), 2, CPU)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis_res1",
            "has no scalar scale companion",
        ),
        (
            "model.layers.3.mlp.experts.0.gate_proj.rank0.scale_res1",
            "Orphaned MSRT cartridge scale",
        ),
    ],
)
def test_cartridge_key_index_reports_orphaned_companions(key, message):
    with pytest.raises(ValueError, match=message):
        cartridge_module._index_cartridge_keys((key,))


def test_manifest_layer_names_remain_canonical():
    key = "wrapped.model.layers.3.mlp.experts.0.gate_proj.rank0.trellis_res1"
    match = cartridge_module._CARTRIDGE_KEY_RE.fullmatch(key)
    assert match is not None
    config = {
        "coverage": {"res1": {"3": [0]}},
        "selected_layers": [3],
        "selected_experts": [0],
        "chain": [{"label": "res1", "experts": "all"}],
    }

    with pytest.raises(ValueError, match="Invalid MSRT cartridge layer name"):
        cartridge_module._validate_manifest_tensor_coverage(
            config, {match.group("layer"): [(key, match)]}
        )


def test_loader_rejects_fp32_inverse_scale_overflow(tmp_path):
    path = tmp_path / "inverse-overflow.safetensors"
    tensors = _loader_tensors()
    for key in tuple(tensors):
        if ".scale_" in key:
            tensors[key] = torch.tensor(1e-40, dtype=torch.float32)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="inverse scale is not finite in FP32"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_validates_tp_sharded_trellis_against_local_shape(tmp_path):
    path = tmp_path / "wrong-local-shape.safetensors"
    tensors = {}
    for rank in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.3.mlp.experts.0.{projection}.rank{rank}"
            shape = (
                (8, 16, 16) if rank == 1 and projection == "gate_proj" else (8, 8, 16)
            )
            tensors[f"{prefix}.trellis_res1"] = torch.zeros(
                *shape,
                dtype=torch.int16,
            )
            tensors[f"{prefix}.scale_res1"] = torch.tensor(1.0)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="Invalid MSRT cartridge trellis"):
        load_cartridge_from_adapter(
            str(path),
            _tp_loader_layer(1),
            2,
            CPU,
        )


def test_loader_rejects_incomplete_projection(tmp_path):
    path = tmp_path / "broken.safetensors"
    save_file(
        {
            "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis_res1": (
                torch.zeros(8, 8, 32, dtype=torch.int16)
            ),
            "model.layers.3.mlp.experts.0.gate_proj.rank0.scale_res1": (
                torch.tensor(1.0)
            ),
        },
        path,
    )

    with pytest.raises(ValueError, match="Incomplete MSRT cartridge"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_invalid_trellis_shape(tmp_path):
    path = tmp_path / "invalid-shape.safetensors"
    prefix = "model.layers.3.mlp.experts.0"
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        tensor_prefix = f"{prefix}.{projection}.rank0"
        tensors[f"{tensor_prefix}.trellis_res1"] = torch.zeros(
            8,
            8,
            15,
            dtype=torch.int16,
        )
        tensors[f"{tensor_prefix}.scale_res1"] = torch.tensor(1.0)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="Invalid MSRT cartridge trellis"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_oversized_packed_dimensions(tmp_path):
    path = tmp_path / "oversized.safetensors"
    prefix = "model.layers.3.mlp.experts.0"
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        tensor_prefix = f"{prefix}.{projection}.rank0"
        tensors[f"{tensor_prefix}.trellis_res1"] = torch.zeros(
            16, 8, 32, dtype=torch.int16
        )
        tensors[f"{tensor_prefix}.scale_res1"] = torch.tensor(1.0)
    save_file(tensors, path)

    with pytest.raises(ValueError, match="128-aligned logical shape"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_malformed_target_key(tmp_path):
    path = tmp_path / "malformed.safetensors"
    save_file(
        {
            "model.layers.3.mlp.experts.0.gate_proj.rankx.trellis_res1": (
                torch.zeros(1, 1, 16, dtype=torch.int16)
            )
        },
        path,
    )

    with pytest.raises(ValueError, match="Malformed MSRT cartridge key"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_requires_versioned_adapter_manifest(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    (tmp_path / "adapter_config.json").unlink()

    with pytest.raises(ValueError, match="cannot open EXL3 cartridge adapter_config"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_operation", "base + residual", "runtime_operation"),
        ("codebook", "mul1", "codebook"),
        ("mcg_multiplier", 1, "mcg_multiplier"),
    ],
)
def test_loader_rejects_decode_contract_mismatch(
    tmp_path,
    field,
    value,
    message,
):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_incomplete_v3_manifest(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.pop("source_assembly")
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="manifest fields do not match"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


@pytest.mark.parametrize("label", ["-bad", "_bad", "bad.label", "a" * 33])
def test_loader_rejects_noncanonical_labels(tmp_path, label):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["assembly"] = label
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="invalid assembly label"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_accepts_32_character_label(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["assembly"] = "a" * 32
    config_path.write_text(json.dumps(config))

    assert load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_manifest_coverage_mismatch(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["coverage"]["res1"]["3"] = [1]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"do not match adapter_config\.json coverage"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_cartridge_bound_to_different_base(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base"]["compatibility_sha256"] = "b" * 64
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="different base checkpoint"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_base_bitrate_mismatch(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base"]["k"] = 3
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"base\.k does not match"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_shard_changed_after_manifest(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=r"does not match adapter_config\.json"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_stage_bitrate_mismatch(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(bits=2), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["chain"][0]["k"] = 1
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="declares K1"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_tensor_ranks_disagreeing_with_manifest(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(ranks=(0, 1), bits=1), path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["tensor_parallel"] = {
        "layout": "full",
        "world_size": 1,
        "ranks": [0],
        "axis_by_projection": {
            "gate_proj": "output",
            "up_proj": "output",
            "down_proj": "input",
        },
    }
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="ranks do not match adapter_config"):
        load_cartridge_from_adapter(
            str(path),
            _tp_loader_layer(1),
            2,
            CPU,
        )


def test_loader_rejects_ignored_codebook_companion(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    tensors = _loader_tensors()
    tensors["model.layers.3.mlp.experts.0.gate_proj.rank0.mul1_res1"] = torch.tensor(
        1, dtype=torch.int32
    )
    save_file(tensors, path)

    with pytest.raises(ValueError, match="Malformed MSRT cartridge key"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_loader_rejects_child_only_expert_coverage(tmp_path):
    path = tmp_path / "child-only.safetensors"
    tensors = {}
    for label, experts in (("res1", (0,)), ("res2", (0, 1))):
        for expert in experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.3.mlp.experts.{expert}.{projection}.rank0"
                tensors[f"{prefix}.trellis_{label}"] = torch.zeros(
                    8, 8, 16, dtype=torch.int16
                )
                tensors[f"{prefix}.scale_{label}"] = torch.tensor(1.0)
    save_file(tensors, path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["chain"][0]["experts"] = [0]
    config["chain"][1]["experts"] = "all"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="not parent-closed"):
        load_cartridge_from_adapter(str(path), _loader_layer(), 2, CPU)


def test_staging_uses_verified_private_bytes_after_source_replacement(tmp_path):
    path = tmp_path / "source.safetensors"
    save_file(_loader_tensors(bits=1), path)
    layer = _loader_layer()

    with stage_exl3_cartridge_adapter(
        SimpleNamespace(modules=lambda: iter((layer,))), str(path)
    ) as staged:
        replacement = tmp_path / "replacement.safetensors"
        _save_file(_loader_tensors(bits=2), replacement)
        os.replace(replacement, path)
        cartridge = cartridge_module._load_cartridge_from_staged_adapter(
            staged, str(path), layer, 2, CPU
        )

    assert cartridge is not None
    stage = cartridge.get_stage_tensors(0, 0, "w1")
    assert stage is not None and stage["trellis"].shape[-1] == 16


def test_staging_rejects_missing_and_non_regular_requested_paths(tmp_path):
    layer = _loader_layer()
    model = SimpleNamespace(modules=lambda: iter((layer,)))
    with pytest.raises(ValueError, match="must exist"):
        stage_exl3_cartridge_adapter(model, str(tmp_path / "typo.safetensors"))

    fifo = tmp_path / "adapter.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="directory or regular file"):
        stage_exl3_cartridge_adapter(model, str(fifo))

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    shard = adapter_dir / "cartridge.safetensors"
    save_file(_loader_tensors(), shard)
    shard.unlink()
    os.mkfifo(shard)
    with pytest.raises(ValueError, match="must be a regular file"):
        stage_exl3_cartridge_adapter(model, str(adapter_dir))


def test_model_loader_prepares_only_manifest_selected_layers(tmp_path, monkeypatch):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    layers = [_runtime_layer(), _runtime_layer()]
    layers[1].layer_name = "model.layers.4.mlp.experts"
    model = SimpleNamespace(modules=lambda: iter(layers))

    def mark_materialized(runtime, _layer, _cartridge):
        runtime._materialized = True

    monkeypatch.setattr(
        Exl3CUDAGraphCartridgeRuntime,
        "materialize",
        mark_materialized,
    )

    assert prepare_exl3_cartridge_into_model(model, str(path)) == 1
    assert isinstance(
        layers[0]._exl3_cartridge_runtime,
        Exl3CUDAGraphCartridgeRuntime,
    )
    assert not hasattr(layers[1], "_exl3_cartridge_runtime")


def test_model_loader_matches_wrapped_live_layer_by_numeric_index(
    tmp_path, monkeypatch
):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    layer = _loader_layer()
    layer.layer_name = "language_model.model.layers.3.mlp.experts"
    model = SimpleNamespace(modules=lambda: iter((layer,)))

    monkeypatch.setattr(
        Exl3CUDAGraphCartridgeRuntime,
        "materialize",
        lambda runtime, _layer, _cartridge: setattr(runtime, "_materialized", True),
    )

    with stage_exl3_cartridge_adapter(model, str(path)) as staged:
        assert staged.local_layer_names == ("model.layers.3.mlp.experts",)
        cartridge = cartridge_module._load_cartridge_from_staged_adapter(
            staged, str(path), layer, 2, CPU
        )
        assert cartridge is not None
        assert (
            cartridge_module.prepare_staged_exl3_cartridge_into_model(model, staged)
            == 1
        )


def test_model_loader_rejects_ambiguous_wrapped_live_layer(tmp_path):
    path = tmp_path / "cartridge.safetensors"
    save_file(_loader_tensors(), path)
    layer = _loader_layer()
    layer.layer_name = "model.layers.3.wrapper.layers.4.mlp.experts"
    model = SimpleNamespace(modules=lambda: iter((layer,)))

    with pytest.raises(ValueError, match=r"exactly one numeric layers\.<index>"):
        stage_exl3_cartridge_adapter(model, str(path))


def test_model_loader_accepts_authenticated_nonlocal_mtp_layer(tmp_path):
    path = tmp_path / "target-and-mtp.safetensors"
    tensors = _loader_tensors(layer=3)
    tensors.update(_loader_tensors(layer=78))
    save_file(tensors, path)
    target = _loader_layer()
    target.exl3_base_compatibility_by_layer["78"] = "b" * 64
    draft = _loader_layer()
    draft.layer_name = "model.layers.78.mlp.experts"
    draft.exl3_cartridge_capable = False
    draft.exl3_is_draft = True
    model = SimpleNamespace(modules=lambda: iter((target, draft)))

    with stage_exl3_cartridge_adapter(model, str(path)) as staged:
        assert staged.local_layer_names == ("model.layers.3.mlp.experts",)
        assert set(staged.state.by_layer) == {
            "model.layers.3.mlp.experts",
            "model.layers.78.mlp.experts",
        }
        assert not hasattr(draft, "_exl3_cartridge_runtime")


def test_model_swap_to_narrower_cartridge_and_shared_workspace(tmp_path, monkeypatch):
    wide_dir = tmp_path / "wide"
    narrow_dir = tmp_path / "narrow"
    wide_dir.mkdir()
    narrow_dir.mkdir()
    wide_path = wide_dir / "cartridge.safetensors"
    narrow_path = narrow_dir / "cartridge.safetensors"
    wide_tensors = _loader_tensors(layer=3)
    wide_tensors.update(_loader_tensors(layer=4))
    save_file(wide_tensors, wide_path)
    save_file(_loader_tensors(layer=3), narrow_path)

    layers = [_loader_layer(), _loader_layer()]
    layers[1].layer_name = "model.layers.4.mlp.experts"
    model = SimpleNamespace(modules=lambda: iter(layers))

    def mark_materialized(runtime, _layer, _cartridge):
        runtime._materialized = True

    monkeypatch.setattr(
        Exl3CUDAGraphCartridgeRuntime,
        "materialize",
        mark_materialized,
    )

    assert prepare_exl3_cartridge_into_model(model, str(wide_path)) == 2
    first_runtime = layers[0]._exl3_cartridge_runtime
    second_runtime = layers[1]._exl3_cartridge_runtime
    assert first_runtime.xh is second_runtime.xh

    assert prepare_exl3_cartridge_into_model(model, str(narrow_path)) == 1
    assert layers[0]._exl3_cartridge_runtime is first_runtime
    assert not hasattr(layers[1], "_exl3_cartridge_runtime")
    assert cartridge_module.deactivate_exl3_cartridge(model) == 1
    assert not hasattr(model, "_exl3_cartridge_workspaces")


def test_worker_cartridge_graph_operations_target_model():
    model = SimpleNamespace()
    compilation_config = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE,
        compile_sizes=[32, 16],
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[16],
        get_compile_ranges=lambda: [],
    )
    model_runner = SimpleNamespace(
        model=model,
        clear_cudagraphs=MagicMock(),
        capture_model=MagicMock(return_value=123),
        _dummy_run=MagicMock(),
    )
    worker = object.__new__(Worker)
    worker.model_runner = model_runner
    worker.vllm_config = SimpleNamespace(
        compilation_config=compilation_config,
    )
    worker._warmup_kernels_once = MagicMock()
    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "has_exl3_cartridge",
            return_value=True,
        ) as has_cartridge,
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "activate_exl3_cartridge",
            return_value=10,
        ) as activate,
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "deactivate_exl3_cartridge",
            return_value=10,
        ) as deactivate,
        patch("torch.accelerator.synchronize") as synchronize,
        patch("torch.accelerator.empty_cache") as empty_cache,
    ):
        Worker.clear_exl3_cartridge_cudagraphs(worker)
        assert Worker.has_exl3_cartridge(worker) is True
        assert Worker.activate_exl3_cartridge(worker) == 10
        assert Worker.deactivate_exl3_cartridge(worker) == 10
        assert Worker.capture_exl3_cartridge_cudagraphs(worker) == 123

    model_runner.clear_cudagraphs.assert_called_once_with()
    model_runner.capture_model.assert_called_once_with()
    model_runner._dummy_run.assert_called_once_with(
        32, skip_eplb=True, remove_lora=False
    )
    worker._warmup_kernels_once.assert_called_once_with()
    has_cartridge.assert_called_once_with(model)
    activate.assert_called_once_with(model)
    deactivate.assert_called_once_with(model)
    synchronize.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_worker_relocks_workspace_after_recapture_failure():
    model_runner = SimpleNamespace(
        capture_model=MagicMock(side_effect=RuntimeError("capture failed"))
    )
    worker = object.__new__(Worker)
    worker.model_runner = model_runner
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    worker._warmup_kernels_once = MagicMock()
    with (
        patch("vllm.v1.worker.gpu_worker.lock_workspace") as lock,
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        Worker.capture_exl3_cartridge_cudagraphs(worker)

    lock.assert_called_once_with()


def test_worker_stages_only_after_abi_preflight():
    staged = SimpleNamespace(
        state=SimpleNamespace(by_layer={"model.layers.3.mlp.experts": []}),
        local_layer_names=("model.layers.3.mlp.experts",),
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace()),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            use_v2_model_runner=False,
            lora_config=None,
        ),
    )
    order = []
    with (
        patch.object(
            cartridge_module,
            "_load_additive_exl3_ext",
            side_effect=lambda: order.append("abi"),
        ),
        patch.object(
            cartridge_module,
            "stage_exl3_cartridge_adapter",
            side_effect=lambda *_args: (order.append("stage"), staged)[1],
        ),
    ):
        assert Worker.stage_exl3_cartridge(worker, "/adapter", "id") == 1
    assert order == ["abi", "stage"]


def test_worker_does_not_register_stage_on_abi_failure():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace()),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            use_v2_model_runner=False,
            lora_config=None,
        ),
    )
    with (
        patch.object(
            cartridge_module,
            "_load_additive_exl3_ext",
            side_effect=RuntimeError("ABI mismatch"),
        ),
        patch.object(cartridge_module, "stage_exl3_cartridge_adapter") as stage,
        pytest.raises(RuntimeError, match="ABI mismatch"),
    ):
        Worker.stage_exl3_cartridge(worker, "/adapter", "id")
    stage.assert_not_called()
    assert not hasattr(worker, "_exl3_cartridge_stages")


def test_gpu_runner_clear_cudagraphs_releases_every_graph_owner():
    runner = object.__new__(GPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    runner.encoder_cudagraph_manager = MagicMock()
    with (
        patch(
            "vllm.v1.worker.gpu_model_runner.CUDAGraphWrapper.clear_all_graphs"
        ) as clear_graphs,
        patch(
            "vllm.v1.worker.gpu_model_runner.BreakableCUDAGraphWrapper.clear_all_graphs"
        ) as clear_breakable,
        patch("vllm.v1.worker.gpu_model_runner.unlock_workspace") as unlock,
        patch("torch.accelerator.synchronize") as synchronize,
        patch("torch.accelerator.empty_cache") as empty_cache,
    ):
        GPUModelRunner.clear_cudagraphs(runner)

    clear_graphs.assert_called_once_with()
    clear_breakable.assert_called_once_with()
    runner.encoder_cudagraph_manager.clear.assert_called_once_with()
    unlock.assert_called_once_with()
    synchronize.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_sync_engine_rejects_cartridge_graph_switches():
    engine = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="only by AsyncLLM"):
        LLMEngine.load_exl3_cartridge(engine, "cartridge.safetensors")
    with pytest.raises(NotImplementedError, match="only by AsyncLLM"):
        LLMEngine.deactivate_exl3_cartridge(engine)


def test_async_load_stages_before_pause_and_discards_after_success():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        methods = []

        async def rpc(method, *args, **kwargs):
            del args, kwargs
            methods.append(method)
            if method == "stage_exl3_cartridge":
                engine._pause_generation.assert_not_awaited()
                return [2, 2]
            if method in {
                "prepare_staged_exl3_cartridge",
                "activate_exl3_cartridge",
            }:
                return [2, 2]
            return None

        engine.collective_rpc = AsyncMock(side_effect=rpc)

        assert await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors") == [
            2,
            2,
        ]
        assert methods == [
            "stage_exl3_cartridge",
            "clear_exl3_cartridge_cudagraphs",
            "prepare_staged_exl3_cartridge",
            "activate_exl3_cartridge",
            "capture_exl3_cartridge_cudagraphs",
            "discard_staged_exl3_cartridge",
        ]
        engine._pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)
        engine._resume_generation.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.parametrize(
    "stage_result",
    [RuntimeError("worker preflight failed"), [2, 0]],
)
def test_async_stage_failure_discards_without_pausing(stage_result):
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        methods = []

        async def rpc(method, *args, **kwargs):
            del args, kwargs
            methods.append(method)
            if method == "stage_exl3_cartridge":
                if isinstance(stage_result, BaseException):
                    raise stage_result
                return stage_result
            return None

        engine.collective_rpc = AsyncMock(side_effect=rpc)
        with pytest.raises(RuntimeError):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")
        assert methods == [
            "stage_exl3_cartridge",
            "discard_staged_exl3_cartridge",
        ]
        engine._pause_generation.assert_not_awaited()
        engine._resume_generation.assert_not_awaited()

    asyncio.run(run())


def test_async_load_rolls_back_after_post_pause_cancellation():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        methods = []
        first_clear = True

        async def rpc(method, *args, **kwargs):
            nonlocal first_clear
            del args, kwargs
            methods.append(method)
            if method == "stage_exl3_cartridge":
                return [2, 2]
            if method == "clear_exl3_cartridge_cudagraphs" and first_clear:
                first_clear = False
                raise asyncio.CancelledError()
            if method == "deactivate_exl3_cartridge":
                return [2, 2]
            return None

        engine.collective_rpc = AsyncMock(side_effect=rpc)
        with pytest.raises(asyncio.CancelledError):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")
        assert methods == [
            "stage_exl3_cartridge",
            "clear_exl3_cartridge_cudagraphs",
            "clear_exl3_cartridge_cudagraphs",
            "deactivate_exl3_cartridge",
            "capture_exl3_cartridge_cudagraphs",
            "discard_staged_exl3_cartridge",
        ]
        engine._resume_generation.assert_awaited_once_with()

    asyncio.run(run())


def test_async_discard_failure_still_resumes():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()

        async def rpc(method, *args, **kwargs):
            del args, kwargs
            if method == "stage_exl3_cartridge":
                return [1]
            if method in {
                "prepare_staged_exl3_cartridge",
                "activate_exl3_cartridge",
            }:
                return [1]
            if method == "discard_staged_exl3_cartridge":
                raise RuntimeError("discard failed")
            return None

        engine.collective_rpc = AsyncMock(side_effect=rpc)
        with pytest.raises(RuntimeError, match="discard failed"):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")
        engine._resume_generation.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_phase", ["discard", "resume"])
def test_async_cleanup_cancellation_is_not_hidden_by_primary_error(cancel_phase):
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        phase_started = asyncio.Event()
        finish_phase = asyncio.Event()
        first_clear = True

        async def rpc(method, *args, **kwargs):
            nonlocal first_clear
            del args, kwargs
            if method == "stage_exl3_cartridge":
                return [1]
            if method == "clear_exl3_cartridge_cudagraphs" and first_clear:
                first_clear = False
                raise RuntimeError("primary failed")
            if method == "deactivate_exl3_cartridge":
                return [1]
            if method == "discard_staged_exl3_cartridge" and cancel_phase == "discard":
                phase_started.set()
                await finish_phase.wait()
            return None

        async def resume():
            if cancel_phase == "resume":
                phase_started.set()
                await finish_phase.wait()

        engine.collective_rpc = AsyncMock(side_effect=rpc)
        engine._resume_generation = AsyncMock(side_effect=resume)

        with patch("vllm.v1.engine.async_llm.logger.error") as log_error:
            load_task = asyncio.create_task(
                AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")
            )
            await phase_started.wait()
            load_task.cancel()
            await asyncio.sleep(0)
            finish_phase.set()
            with pytest.raises(asyncio.CancelledError):
                await load_task

        engine._resume_generation.assert_awaited_once_with()
        log_error.assert_not_called()

    asyncio.run(run())


def test_async_primary_error_survives_non_cancellation_cleanup_failure():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        first_clear = True

        async def rpc(method, *args, **kwargs):
            nonlocal first_clear
            del args, kwargs
            if method == "stage_exl3_cartridge":
                return [1]
            if method == "clear_exl3_cartridge_cudagraphs" and first_clear:
                first_clear = False
                raise RuntimeError("primary failed")
            if method == "deactivate_exl3_cartridge":
                return [1]
            if method == "discard_staged_exl3_cartridge":
                raise RuntimeError("discard failed")
            return None

        engine.collective_rpc = AsyncMock(side_effect=rpc)
        with (
            patch("vllm.v1.engine.async_llm.logger.error") as log_error,
            pytest.raises(RuntimeError, match="primary failed"),
        ):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")

        engine._resume_generation.assert_awaited_once_with()
        log_error.assert_called_once()
        assert log_error.call_args.args[0].startswith("EXL3 cartridge cleanup failed")

    asyncio.run(run())


def test_public_resume_waits_for_cartridge_transaction_lock():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine._engine_mutation_lock = asyncio.Lock()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def load(_path):
            entered.set()
            await release.wait()
            return [1]

        engine._load_exl3_cartridge = AsyncMock(side_effect=load)
        engine._resume_generation = AsyncMock()
        load_task = asyncio.create_task(
            AsyncLLM.load_exl3_cartridge(engine, "cartridge.safetensors")
        )
        await entered.wait()
        resume_task = asyncio.create_task(AsyncLLM.resume_generation(engine))
        await asyncio.sleep(0)
        engine._resume_generation.assert_not_awaited()
        release.set()
        assert await load_task == [1]
        await resume_task
        engine._resume_generation.assert_awaited_once_with()

    asyncio.run(run())


def test_async_deactivate_recaptures_before_resuming():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        engine.collective_rpc = AsyncMock(side_effect=[[True], None, [2], 123])

        assert await AsyncLLM._deactivate_exl3_cartridge(engine) == [2]
        assert [call.args[0] for call in engine.collective_rpc.await_args_list] == [
            "has_exl3_cartridge",
            "clear_exl3_cartridge_cudagraphs",
            "deactivate_exl3_cartridge",
            "capture_exl3_cartridge_cudagraphs",
        ]
        engine._pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)
        engine._resume_generation.assert_awaited_once_with()

    asyncio.run(run())


def test_async_deactivate_is_noop_without_runtime():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock()
        engine._pause_generation = AsyncMock()
        engine.collective_rpc = AsyncMock(return_value=[False, False])

        assert await AsyncLLM._deactivate_exl3_cartridge(engine) == [0, 0]
        engine.collective_rpc.assert_awaited_once_with("has_exl3_cartridge")
        engine.is_paused.assert_not_awaited()
        engine._pause_generation.assert_not_awaited()

    asyncio.run(run())


def test_async_load_shuts_down_when_base_graph_restore_fails():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine._pause_generation = AsyncMock()
        engine._resume_generation = AsyncMock()
        engine.shutdown = MagicMock()
        engine.collective_rpc = AsyncMock(
            side_effect=[
                [1],
                RuntimeError("load failed"),
                RuntimeError("restore failed"),
                None,
            ]
        )

        with pytest.raises(EngineDeadError):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")

        engine.shutdown.assert_called_once_with()
        engine._resume_generation.assert_not_awaited()

    asyncio.run(run())
