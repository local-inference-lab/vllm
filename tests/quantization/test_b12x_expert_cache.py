# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source ownership and byte preservation for opt-in expert caching."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.b12x_cache import (
    ModelOptNvFp4CacheMoE,
    cache_provider,
)


@pytest.fixture(autouse=True)
def single_rank_parameters(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )


def method_and_layer():
    method = object.__new__(ModelOptNvFp4CacheMoE)
    method.quant_config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True, group_size=16
    )
    method.moe = SimpleNamespace(is_act_and_mul=True)
    method.use_global_sf = False
    method.provider = SimpleNamespace(model=Mock(checkpoint_id="a" * 64))
    method.prefix = "model.layers.0.mlp.experts"
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    layer.activation = SimpleNamespace(value="silu")
    return method, layer


def test_cpu_allocation_overrides_ambient_device_and_preserves_sources(monkeypatch):
    method, layer = method_and_layer()

    def forbidden(*args, **kwargs):
        raise AssertionError("source allocation initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.device("meta"):
        method.create_weights(layer, 4, 128, 128, torch.bfloat16)
    assert all(p.device.type == "cpu" for p in layer.parameters())
    method.provider.model.reserve_source.assert_called_once_with(
        4 * (3 * 128 * 128 // 2 + 3 * 128 * 128 // 16 + 24)
    )
    for parameter in layer.parameters():
        parameter.data.fill_(1)
    before = {name: value.clone() for name, value in layer.named_parameters()}
    method.process_weights_after_loading(layer)
    source = method.provider.model.add_source.call_args.args[0]
    assert source.weights.layer_name == method.prefix
    assert source.plan.source.w13_layout.value == "w31"
    assert source.plan.activation.mode.value == "a16"
    assert source.weights.w13.data_ptr() == layer.w13_weight.data_ptr()
    for name, value in layer.named_parameters():
        assert torch.equal(value.view(torch.uint8), before[name].view(torch.uint8))
    assert len(source.owners) == len(before)


def test_loader_rejects_global_scale_reconciliation():
    method, layer = method_and_layer()
    method.create_weights(layer, 4, 128, 128, torch.bfloat16)
    layer.w13_weight_scale_2.data.fill_(1)
    layer.w13_weight_scale_2.data[0, 1] = 2
    with pytest.raises(ValueError, match="never requantizes"):
        method.process_weights_after_loading(layer)
    method.provider.model.add_source.assert_not_called()


def test_custom_additional_config_does_not_enable_cache():
    config = SimpleNamespace(additional_config=object())
    assert cache_provider(config) is None


@pytest.mark.parametrize(
    "quantization,dtype,backend",
    [
        ("awq", torch.bfloat16, "b12x"),
        ("modelopt_fp4", torch.float16, "b12x"),
        ("modelopt_fp4", torch.bfloat16, "flashinfer_cutlass"),
    ],
)
def test_incompatible_recipe_is_rejected_before_checkpoint_access(
    quantization, dtype, backend
):
    config = SimpleNamespace(
        additional_config={"b12x_expert_cache": {}},
        model_config=SimpleNamespace(quantization=quantization, dtype=dtype),
        kernel_config=SimpleNamespace(moe_backend=backend),
    )
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        cache_provider(config, create=True)


@pytest.mark.parametrize("overlapped", [False, True])
def test_shared_wrapper_executes_once_and_cache_observes_only_routed_ids(
    monkeypatch, overlapped
):
    """The runner retains shared gating and ownership outside the routed method."""
    from b12x.moe import fused_moe as moe

    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )

    method, layer = method_and_layer()
    method.moe_kernel = None
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1], [0]])
    weights = torch.tensor([[0.25], [0.75]])
    routed = x * 2
    model = method.provider.model
    model.plans = {method.prefix: object()}
    bind = Mock(return_value=object())
    monkeypatch.setattr(moe, "bind", bind)
    monkeypatch.setattr(moe, "run", lambda **kwargs: routed)

    class GatedShared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            return inputs * torch.sigmoid(inputs.sum(-1, keepdim=True))

    shared = GatedShared()
    wrapper = object.__new__(SharedExperts)
    torch.nn.Module.__init__(wrapper)
    wrapper._layer = shared
    wrapper._workspace_layers = ()
    wrapper._output = [shared(x) if overlapped else None, None]
    wrapper.enable_dbo = False
    wrapper._determine_shared_experts_order = lambda _: (
        SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        if overlapped
        else SharedExpertsOrder.NO_OVERLAP
    )
    wrapper.wait = Mock()
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner._shared_experts = wrapper
    runner.router = SimpleNamespace(select_experts=lambda **kwargs: (weights, ids))
    runner.routed_experts = SimpleNamespace(
        quant_method=method,
        forward_modular=lambda **kwargs: method.apply(layer, **kwargs),
    )
    shared_output, routed_output = runner._apply_quant_method(
        x, x, x, shared_experts_overlapping=overlapped
    )
    assert shared.calls == 1
    assert torch.equal(shared_output, x * torch.sigmoid(x.sum(-1, keepdim=True)))
    assert routed_output is routed
    assert wrapper._output == [None, None]
    assert wrapper.wait.call_count == int(overlapped)
    assert not method.mk_can_overlap_shared_experts
    bind.assert_called_once_with(
        model.plans[method.prefix], a=x, topk_ids=ids, topk_weights=weights
    )
    model.observe.assert_called_once_with(method.prefix, ids)


def test_cache_still_rejects_unowned_workspace():
    method, layer = method_and_layer()
    with pytest.raises(ValueError, match="caller-owned scratch"):
        method.apply(layer, None, None, None, workspace=(object(),))
    method.provider.model.observe.assert_not_called()


def test_next_host_lower_bound_rejects_before_fingerprint_or_model_allocation(
    monkeypatch,
):
    from b12x.integration.vllm import checkpoint, expert_cache
    from b12x.moe.fused_moe import cache_source

    from vllm.model_executor.layers.fused_moe.b12x_cache import _CacheProvider

    settings = dict(
        mode="static",
        activation="w4a16",
        profile_path="profile.json",
        workload="mixed",
        expert_device_bytes=1 << 30,
        host_bytes=40 << 30,
        kv_reserved_bytes=2 << 30,
        graph_reserved_bytes=512 << 20,
        device_safety_bytes=1 << 30,
        host_safety_bytes=1 << 30,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization="modelopt_fp4",
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(model_type="qwen3_next"),
            model="checkpoint",
        ),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_dbo=False,
        ),
        speculative_config=None,
        lora_config=None,
        additional_config={"b12x_expert_cache": settings},
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    audit = Mock(return_value={"host_expert_lower_bound_bytes": 81 << 30})
    monkeypatch.setattr(checkpoint, "audit", audit)
    fingerprint = Mock(
        side_effect=AssertionError("full weight hash preceded admission")
    )
    model = Mock(side_effect=AssertionError("model allocation preceded admission"))
    monkeypatch.setattr(cache_source, "checkpoint_fingerprint", fingerprint)
    monkeypatch.setattr(expert_cache, "ExpertCacheModel", model)
    with pytest.raises(ValueError, match="mapped backing exceed"):
        _CacheProvider(config)
    audit.assert_called_once_with("checkpoint", check_values=True)
    fingerprint.assert_not_called()
    model.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
@pytest.mark.parametrize("side_stream", [False, True])
def test_shared_cache_composition_replays_after_promotion(
    tmp_path, monkeypatch, side_stream
):
    """Exercise real prepared routed execution plus wrapper-owned gated shared work."""
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    torch.manual_seed(45)
    method, layer = method_and_layer()
    method.moe_kernel = None
    method.create_weights(layer, 4, 128, 128, torch.bfloat16)
    for name, parameter in layer.named_parameters():
        if parameter.dtype == torch.uint8:
            parameter.data.random_(0, 256)
        else:
            parameter.data.fill_(0.25)
    method.process_weights_after_loading(layer)
    source = method.provider.model.add_source.call_args.args[0]
    placement = moe.ExpertResidencyPlan(
        total_experts=4,
        hbm_expert_ids=(0, 1),
        grace_expert_ids=(2, 3),
        layer=method.prefix,
        model_fingerprint="a" * 64,
        workload="synthetic shared-composition correctness",
        provenance="seed 45",
    )
    plan = moe.plan_execution(
        experts=source,
        capacity=moe.ExecutionCapacity(max_tokens=4, top_k=2),
        placement=placement,
        memory_budget=moe.ExpertMemoryBudget(hbm_bytes=1 << 30, grace_bytes=1 << 30),
        updates=moe.ResidencyUpdateCapacity(max_pairs=2),
    )
    method.provider.model.plans = {method.prefix: plan}
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 0.125
    ids = torch.tensor([[0, 2], [3, 1], [2, 2], [1, 0]], device="cuda")
    weights = torch.rand(4, 2, device="cuda")

    class GatedShared(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.silu(value) * torch.sigmoid(
                value.sum(-1, keepdim=True)
            )

    monkeypatch.setenv(
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0" if side_stream else "1"
    )
    parallel = SimpleNamespace(enable_eplb=False, use_fi_nvl_two_sided_kernels=False)
    wrapper = SharedExperts(
        GatedShared(),
        SimpleNamespace(moe_parallel_config=parallel),
        False,
        lambda: method.mk_can_overlap_shared_experts,
    )
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner._shared_experts = wrapper
    runner.router = SimpleNamespace(select_experts=lambda **kwargs: (weights, ids))
    runner.routed_experts = SimpleNamespace(
        quant_method=method,
        forward_modular=lambda **kwargs: method.apply(layer, **kwargs),
    )

    def prepare(state):
        binding = state.bind(a=x, topk_ids=ids, topk_weights=weights)
        return PreparedCall(
            run=binding.run, output=binding.output, owners=(binding,), close=state.close
        )

    def run():
        overlapping = wrapper.maybe_forward_async(x)
        assert overlapping == side_stream
        shared, routed = runner._apply_quant_method(
            x, x, x, shared_experts_overlapping=overlapping
        )
        return shared + routed

    with PreparationSession(
        autotune=False, compile_workers=0, cache_dir=tmp_path
    ) as session:
        session.prepare((plan.request(name="routed", prepare_call=prepare),))
        state = plan.prepared.state
        pointers = state.pointers()
        for _ in range(3):
            expected = run().clone()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = run()
        session.freeze()
        torch.accelerator.synchronize()
        state.updates.apply(
            ((2, 0), (3, 1)), expected=state.updates.snapshot(), quiescent=True
        )
        for _ in range(5):
            before = torch.accelerator.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                torch.accelerator.memory_stats()["allocation.all.allocated"] == before
            )
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            assert torch.isfinite(captured).all() and torch.count_nonzero(captured)
            assert state.pointers() == pointers
        graph.reset()
    assert plan.prepared is None
