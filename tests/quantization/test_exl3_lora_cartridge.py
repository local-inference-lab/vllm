# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for CUDA-graph-safe EXL3 MSRT cartridges."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.layers.quantization.exl3 as exl3_module
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.model_executor.layers.quantization.exl3_lora_cartridge import (
    Exl3CUDAGraphCartridgeRuntime,
    Exl3LoraCartridge,
    apply_exl3_cartridge,
    apply_exl3_cudagraph_cartridge,
    load_cartridge_from_adapter,
    load_exl3_cartridge_into_model,
    prepare_exl3_cudagraph_cartridge_runtime,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.worker.gpu_worker import Worker

CPU = torch.device("cpu")


def _runtime_layer(device: torch.device = CPU):
    return SimpleNamespace(
        local_num_experts=2,
        exl3_hidden_size=4,
        exl3_intermediate_size_per_partition=2,
        exl3_params_dtype=torch.float16,
        w13_trellis=SimpleNamespace(device=device),
        activation=SimpleNamespace(value="silu"),
        layer_name="model.layers.3.mlp.experts",
    )


def _loader_layer():
    layer = _runtime_layer()
    layer.exl3_hidden_size = 16
    layer.exl3_intermediate_size_per_partition = 16
    return layer


def _cartridge(device: torch.device = CPU):
    cartridge = Exl3LoraCartridge(1, 2, device)
    for expert_id in range(2):
        for shard_id in ("w1", "w3", "w2"):
            cartridge.set_stage_tensors(
                0,
                expert_id,
                shard_id,
                torch.zeros(1, 1, 16, dtype=torch.int16),
                torch.ones(1, dtype=torch.float16),
                torch.ones(1, dtype=torch.float16),
                1.0,
            )
    cartridge.active = True
    return cartridge


def test_config_requires_explicit_cartridge_runtime():
    assert not Exl3Config.from_config({}).cartridge_runtime
    assert Exl3Config.from_config({"cartridge_runtime": True}).cartridge_runtime
    with pytest.raises(TypeError, match="must be a boolean"):
        Exl3Config.from_config({"cartridge_runtime": 1})


def test_runtime_rejects_tensor_parallel_weights():
    layer = _runtime_layer()
    layer.exl3_tp_size = 2
    with pytest.raises(NotImplementedError, match="tensor_parallel_size=1"):
        prepare_exl3_cudagraph_cartridge_runtime(layer)


def test_runtime_preserves_fp16_dense_kernel_contract():
    layer = _runtime_layer()
    layer.exl3_params_dtype = torch.bfloat16
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    assert runtime.dtype == torch.float16
    assert runtime.w13.dtype == torch.float16
    assert runtime.w2.dtype == torch.float16


def test_runtime_rejects_activation_before_materialization():
    runtime = prepare_exl3_cudagraph_cartridge_runtime(_runtime_layer())
    with pytest.raises(RuntimeError, match="unmaterialized"):
        runtime.activate()


def test_cartridge_validates_indices_and_scale():
    cartridge = Exl3LoraCartridge(1, 2, torch.device("cpu"))
    tensors = (
        torch.zeros(1, 1, 16, dtype=torch.int16),
        torch.ones(1, dtype=torch.float16),
        torch.ones(1, dtype=torch.float16),
    )
    with pytest.raises(IndexError, match="stage index"):
        cartridge.set_stage_tensors(1, 0, "w1", *tensors, 1.0)
    with pytest.raises(IndexError, match="expert index"):
        cartridge.set_stage_tensors(0, 2, "w1", *tensors, 1.0)
    with pytest.raises(ValueError, match="unsupported EXL3 shard"):
        cartridge.set_stage_tensors(0, 0, "bad", *tensors, 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        cartridge.set_stage_tensors(0, 0, "w1", *tensors, 0.0)


def test_apply_cartridge_sums_stages_with_scale():
    base_output = torch.zeros(4, 2, dtype=torch.float16)
    inputs = torch.randn(4, 4, dtype=torch.float16)
    cartridge = Exl3LoraCartridge(2, 1, torch.device("cpu"))
    for stage, scale in enumerate((2.0, 4.0)):
        cartridge.set_stage_tensors(
            stage,
            0,
            "w1",
            torch.zeros(1, 1, 16, dtype=torch.int16),
            torch.ones(4, dtype=torch.float16),
            torch.ones(2, dtype=torch.float16),
            scale,
        )
    cartridge.active = True

    with patch(
        "vllm.model_executor.layers.quantization.exl3_lora_cartridge._exl3_gemm",
        return_value=torch.full((4, 16), 8.0, dtype=torch.float16),
    ) as gemm:
        result = apply_exl3_cartridge(
            base_output, inputs, None, "w13", 0, "w1", cartridge
        )

    assert torch.equal(result, torch.full((4, 2), 6.0, dtype=torch.float32))
    assert gemm.call_count == 2
    assert gemm.call_args.args[0].shape == (4, 16)


def test_runtime_materializes_exact_combined_projection_weights():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    cartridge = _cartridge()

    def base_projection(_layer, group, inputs, _expert_id, shard_id):
        value = {"w1": 1.0, "w3": 2.0, "w2": 3.0}[shard_id]
        width = 4 if group == "w2" else 2
        return torch.full((inputs.shape[0], width), value, dtype=torch.float16)

    def residual_projection(inputs, trellis, *_args, **_kwargs):
        return torch.ones((inputs.shape[0], trellis.shape[1] * 16), dtype=torch.float16)

    with (
        patch.object(
            __import__(
                "vllm.model_executor.layers.quantization.exl3_lora_cartridge",
                fromlist=["Exl3MoEMethod"],
            ).Exl3MoEMethod,
            "_apply_expert",
            side_effect=base_projection,
        ),
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge._exl3_gemm",
            side_effect=residual_projection,
        ),
    ):
        runtime.materialize(layer, cartridge)

    assert torch.equal(runtime.w13[:, :2], torch.full((2, 2, 4), 2.0))
    assert torch.equal(runtime.w13[:, 2:], torch.full((2, 2, 4), 3.0))
    assert torch.equal(runtime.w2, torch.full((2, 4, 2), 4.0))
    assert runtime.active.item() == 0.0


def test_runtime_rejects_nonfinite_materialized_weights():
    layer = _runtime_layer()
    runtime = Exl3CUDAGraphCartridgeRuntime(layer)
    cartridge = _cartridge()
    for stage in cartridge.stages:
        for tensors in stage.values():
            tensors["scale"] = 1e-30

    def base_projection(_layer, group, inputs, _expert_id, _shard_id):
        width = 4 if group == "w2" else 2
        return torch.zeros((inputs.shape[0], width), dtype=torch.float16)

    with (
        patch.object(
            __import__(
                "vllm.model_executor.layers.quantization.exl3_lora_cartridge",
                fromlist=["Exl3MoEMethod"],
            ).Exl3MoEMethod,
            "_apply_expert",
            side_effect=base_projection,
        ),
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge._exl3_gemm",
            side_effect=lambda inputs, *_args, **_kwargs: torch.ones(
                (inputs.shape[0], 16), dtype=torch.float16
            ),
        ),
        pytest.raises(ValueError, match="non-finite weights"),
    ):
        runtime.materialize(layer, cartridge)

    assert runtime.active.item() is False
    assert not runtime._materialized


def test_cartridge_runtime_rejects_data_parallel_creation(monkeypatch):
    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False, tp_rank=0, tp_size=1),
        has_bias=False,
    )
    method.quant_config = SimpleNamespace(
        rank_sliced_metadata={"tp": 1, "experts_per_layer": 2},
        cartridge_runtime=True,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(runner_type="generate"),
    )
    monkeypatch.setattr(exl3_module, "get_current_vllm_config_or_none", lambda: config)

    with pytest.raises(NotImplementedError, match="data_parallel_size=1"):
        method.create_weights(
            SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
            num_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )


def test_create_weights_stamps_draft_layer_cartridge_disabled(monkeypatch):
    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False, tp_rank=0, tp_size=1),
        has_bias=False,
    )
    method.quant_config = SimpleNamespace(
        rank_sliced_metadata={"tp": 1, "experts_per_layer": 2},
        cartridge_runtime=True,
        rank_sliced_layer_bitrates=MagicMock(return_value=(2, 2)),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(runner_type="draft"),
    )
    monkeypatch.setattr(exl3_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(exl3_module, "Exl3MoEParameter", MagicMock())
    layer = SimpleNamespace(
        layer_name="model.layers.3.mlp.experts",
        register_parameter=MagicMock(),
    )

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=128,
        params_dtype=torch.float16,
    )

    assert layer.exl3_is_draft is True
    assert layer.exl3_cartridge_enabled is False


def test_rank_sliced_draft_layer_skips_cartridge_path():
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(
        rank_sliced_metadata={"tp": 1},
        cartridge_runtime=True,
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


def test_graph_path_uses_device_scalar_without_host_branch():
    layer = _runtime_layer()
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    base = torch.full((3, 4), 2.0, dtype=torch.float16)
    dense = torch.full((3, 4), torch.nan, dtype=torch.float16)
    inputs = torch.zeros(3, 4, dtype=torch.float16)
    weights = torch.ones(3, 1, dtype=torch.float32)
    ids = torch.tensor([[1], [0], [1]], dtype=torch.long)

    with patch(
        "vllm.model_executor.layers.quantization.exl3_lora_cartridge.fused_experts",
        return_value=dense,
    ) as fused:
        runtime.deactivate()
        inactive = apply_exl3_cudagraph_cartridge(base, inputs, weights, ids, layer)
        runtime.active.fill_(1)
        active = apply_exl3_cudagraph_cartridge(base, inputs, weights, ids, layer)

    assert torch.equal(inactive, base)
    assert torch.isnan(active).all()
    assert fused.call_count == 2
    assert torch.equal(fused.call_args_list[0].args[4], torch.zeros_like(ids))
    assert torch.equal(fused.call_args_list[1].args[4], ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_activation_replays_real_fused_moe_in_one_cuda_graph():
    device = torch.device("cuda")
    layer = _loader_layer()
    layer.w13_trellis.device = device
    runtime = prepare_exl3_cudagraph_cartridge_runtime(layer)
    base = torch.full((3, 16), 2.0, dtype=torch.float16, device=device)
    inputs = torch.ones(3, 16, dtype=torch.float16, device=device)
    weights = torch.ones(3, 1, dtype=torch.float32, device=device)
    ids = torch.zeros(3, 1, dtype=torch.long, device=device)

    runtime.w13.fill_(0.1)
    runtime.w2.fill_(0.1)
    apply_exl3_cudagraph_cartridge(base, inputs, weights, ids, layer)
    w13_pointer = runtime.w13.data_ptr()
    w2_pointer = runtime.w2.data_ptr()

    graph = torch.cuda.CUDAGraph()
    runtime.deactivate()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = apply_exl3_cudagraph_cartridge(base, inputs, weights, ids, layer)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, base)

    runtime.w13.fill_(0.2)
    runtime._materialized = True
    runtime.w2.fill_(0.2)
    runtime.activate()
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(output, base)
    assert runtime.w13.data_ptr() == w13_pointer
    assert runtime.w2.data_ptr() == w2_pointer

    runtime.w13.fill_(torch.nan)
    runtime.deactivate()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, base)


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
                    f"{prefix}.suh_res{value}": torch.ones(128, dtype=torch.float16),
                    f"{prefix}.svh_res{value}": torch.ones(128, dtype=torch.float16),
                    f"{prefix}.scale_res{value}": torch.tensor(float(value)),
                }
            )
        return tensors

    tensors = {}
    tensors.update(tensors_for(3, 10))
    tensors.update(tensors_for(3, 2))
    tensors.update(tensors_for(4, 1))
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


def test_loader_rejects_incomplete_projection(tmp_path):
    path = tmp_path / "broken.safetensors"
    save_file(
        {
            "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis_res1": (
                torch.zeros(8, 8, 32, dtype=torch.int16)
            ),
            "model.layers.3.mlp.experts.0.gate_proj.rank0.suh_res1": (
                torch.ones(128, dtype=torch.float16)
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
            8, 8, 15 if projection == "gate_proj" else 32, dtype=torch.int16
        )
        tensors[f"{tensor_prefix}.suh_res1"] = torch.ones(128, dtype=torch.float16)
        tensors[f"{tensor_prefix}.svh_res1"] = torch.ones(128, dtype=torch.float16)
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
        tensors[f"{tensor_prefix}.suh_res1"] = torch.ones(256, dtype=torch.float16)
        tensors[f"{tensor_prefix}.svh_res1"] = torch.ones(128, dtype=torch.float16)
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


def test_model_loader_preflights_every_layer_before_materializing():
    layers = [_runtime_layer(), _runtime_layer()]
    layers[1].layer_name = "model.layers.4.mlp.experts"
    for layer in layers:
        prepare_exl3_cudagraph_cartridge_runtime(layer)
    model = SimpleNamespace(modules=lambda: iter(layers))

    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "load_cartridge_from_adapter",
            side_effect=[_cartridge(), None],
        ),
        patch.object(layers[0]._exl3_cartridge_runtime, "materialize") as materialize,
        pytest.raises(ValueError, match="has no tensors"),
    ):
        load_exl3_cartridge_into_model(model, "cartridge.safetensors")

    materialize.assert_not_called()


def test_model_loader_deactivates_every_layer_after_materialization_failure():
    layers = [_runtime_layer(), _runtime_layer()]
    layers[1].layer_name = "model.layers.4.mlp.experts"
    runtimes = [prepare_exl3_cudagraph_cartridge_runtime(layer) for layer in layers]
    model = SimpleNamespace(modules=lambda: iter(layers))

    def activate(_layer, _cartridge):
        runtimes[0].active.fill_(1)

    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "load_cartridge_from_adapter",
            side_effect=[_cartridge(), _cartridge()],
        ),
        patch.object(runtimes[0], "materialize", side_effect=activate),
        patch.object(
            runtimes[1],
            "materialize",
            MagicMock(side_effect=RuntimeError("decode failed")),
        ),
        pytest.raises(RuntimeError, match="decode failed"),
    ):
        load_exl3_cartridge_into_model(model, "cartridge.safetensors")

    assert all(runtime.active.item() == 0.0 for runtime in runtimes)


def test_model_loader_activates_every_layer_only_after_materialization():
    layers = [_runtime_layer(), _runtime_layer()]
    layers[1].layer_name = "model.layers.4.mlp.experts"
    runtimes = [prepare_exl3_cudagraph_cartridge_runtime(layer) for layer in layers]
    model = SimpleNamespace(modules=lambda: iter(layers))

    def mark_materialized(_layer, _cartridge):
        for runtime in runtimes:
            if not runtime._materialized:
                runtime._materialized = True
                return

    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "load_cartridge_from_adapter",
            side_effect=[_cartridge(), _cartridge()],
        ),
        patch.object(runtimes[0], "materialize", side_effect=mark_materialized),
        patch.object(runtimes[1], "materialize", side_effect=mark_materialized),
    ):
        assert load_exl3_cartridge_into_model(model, "cartridge.safetensors") == 2

    assert all(runtime.active.item() == 1.0 for runtime in runtimes)


def test_worker_cartridge_operations_use_collective_rpc_target_model():
    model = SimpleNamespace()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    with (
        patch(
            "vllm.model_executor.layers.quantization.exl3_lora_cartridge."
            "prepare_exl3_cartridge_into_model",
            return_value=10,
        ) as prepare,
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
    ):
        assert Worker.prepare_exl3_cartridge(worker, "cartridge.safetensors") == 10
        assert Worker.activate_exl3_cartridge(worker) == 10
        assert Worker.deactivate_exl3_cartridge(worker) == 10

    prepare.assert_called_once_with(model, "cartridge.safetensors")
    activate.assert_called_once_with(model)
    deactivate.assert_called_once_with(model)


def test_engine_cartridge_operations_dispatch_to_every_worker():
    engine = SimpleNamespace(
        _prepare_for_exl3_weight_switch=MagicMock(),
        collective_rpc=MagicMock(side_effect=[[10], [10], [10]]),
    )

    assert LLMEngine.load_exl3_cartridge(engine, "cartridge.safetensors") == [10]
    assert LLMEngine.deactivate_exl3_cartridge(engine) == [10]
    assert engine.collective_rpc.call_args_list == [
        (("prepare_exl3_cartridge",), {"args": ("cartridge.safetensors",)}),
        (("activate_exl3_cartridge",), {}),
        (("deactivate_exl3_cartridge",), {}),
    ]


def test_engine_cartridge_load_rolls_back_every_worker_on_commit_failure():
    engine = SimpleNamespace(
        _prepare_for_exl3_weight_switch=MagicMock(),
        collective_rpc=MagicMock(
            side_effect=[[10, 10], RuntimeError("commit failed"), [10, 10]]
        ),
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        LLMEngine.load_exl3_cartridge(engine, "cartridge.safetensors")

    assert engine.collective_rpc.call_args_list[-1] == (
        ("deactivate_exl3_cartridge",),
        {},
    )


def test_sync_engine_weight_switch_requires_cache_invalidation():
    engine = SimpleNamespace(
        has_unfinished_requests=MagicMock(return_value=False),
        reset_prefix_cache=MagicMock(return_value=True),
        reset_mm_cache=MagicMock(),
        reset_encoder_cache=MagicMock(),
    )

    LLMEngine._prepare_for_exl3_weight_switch(engine)

    engine.reset_prefix_cache.assert_called_once_with(
        reset_running_requests=True, reset_connector=True
    )
    engine.reset_mm_cache.assert_called_once_with()
    engine.reset_encoder_cache.assert_called_once_with()


def test_sync_engine_weight_switch_refuses_active_requests():
    engine = SimpleNamespace(has_unfinished_requests=MagicMock(return_value=True))

    with pytest.raises(RuntimeError, match="quiescent"):
        LLMEngine._prepare_for_exl3_weight_switch(engine)


def test_sync_engine_weight_switch_refuses_failed_cache_reset():
    engine = SimpleNamespace(
        has_unfinished_requests=MagicMock(return_value=False),
        reset_prefix_cache=MagicMock(return_value=False),
    )

    with pytest.raises(RuntimeError, match="Unable to clear prefix cache"):
        LLMEngine._prepare_for_exl3_weight_switch(engine)


def test_async_load_clears_caches_and_rolls_back_on_cancellation():
    async def run():
        engine = object.__new__(AsyncLLM)
        engine.is_paused = AsyncMock(return_value=False)
        engine.pause_generation = AsyncMock()
        engine.resume_generation = AsyncMock()
        engine.collective_rpc = AsyncMock(side_effect=[asyncio.CancelledError(), [2]])

        with pytest.raises(asyncio.CancelledError):
            await AsyncLLM._load_exl3_cartridge(engine, "cartridge.safetensors")

        engine.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)
        assert engine.collective_rpc.await_args_list[-1].args == (
            "deactivate_exl3_cartridge",
        )
        engine.resume_generation.assert_awaited_once_with()

    asyncio.run(run())
