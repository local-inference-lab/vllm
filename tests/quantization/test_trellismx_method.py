# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of actual Jovian method selection and the carrier loader ABI.

The runtime launch is replaced by a recorder: these tests prove ownership and
dispatch, not CUDA correctness. No weights or models are downloaded.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import trellismx
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)


@pytest.fixture
def method_inputs(monkeypatch):
    parallel = SimpleNamespace(tp_size=4, ep_size=1, tp_rank=0)
    moe = SimpleNamespace(
        moe_parallel_config=parallel,
        hidden_dim=4096,
        intermediate_size_per_partition=512,
        num_experts=288,
        experts_per_token=8,
        has_bias=False,
        is_lora_enabled=False,
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        swiglu_limit=10.0,
    )
    config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    monkeypatch.setattr(trellismx, "load_overlay", lambda _: object())
    monkeypatch.setattr(
        trellismx,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="glm5_next_text")
            ),
        ),
    )
    return config, moe


def test_opt_in_selects_native_method_before_stock_oracle(method_inputs, monkeypatch):
    config, moe = method_inputs
    monkeypatch.setenv("VLLM_TRELLISMX_CHECKPOINT", "/fixture")
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = moe
    method = config.get_quant_method(layer, "model.layers.3.mlp.experts")
    assert isinstance(method, trellismx.TrellisMXMoEMethod)
    assert not method.is_monolithic
    assert not method.supports_eplb
    assert not method.mk_can_overlap_shared_experts


def test_opt_out_does_not_read_overlay(method_inputs, monkeypatch):
    config, moe = method_inputs
    monkeypatch.delenv("VLLM_TRELLISMX_CHECKPOINT", raising=False)
    assert (
        trellismx.maybe_trellismx_method(
            config, SimpleNamespace(moe_config=moe), "model.layers.3.mlp.experts"
        )
        is None
    )


@pytest.mark.parametrize("field,value", [("tp_size", 2), ("ep_size", 4)])
def test_rejects_unsupported_sharding(method_inputs, field, value):
    config, moe = method_inputs
    setattr(moe.moe_parallel_config, field, value)
    with pytest.raises(ValueError, match="TP4"):
        trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)


def test_inherits_real_modelopt_carrier_weight_loader_shapes(
    method_inputs, monkeypatch
):
    from vllm.model_executor import parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    layer = torch.nn.Module()
    # Exercise the real loader with tiny tensors instead of 288 experts.
    method.create_weights(layer, 2, 32, 16, torch.bfloat16)
    assert layer.w13_weight.shape == (2, 32, 16)
    assert layer.w2_weight.shape == (2, 32, 8)
    assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert method.uses_weight_scale_2_pattern()


def test_subclass_retains_exact_modelopt_scale_loader_semantics(method_inputs):
    """A non-``ModelOpt`` class name must not lose carrier-loader dispatch."""
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    assert method.uses_modelopt_carrier_weight_loader()

    routed = RoutedExperts.__new__(RoutedExperts)
    routed.quant_config = None
    routed.quant_method = method
    routed.moe_config = moe
    routed.expert_map_manager = SimpleNamespace(
        map_global_to_local=lambda expert_id: expert_id
    )

    per_tensor = torch.full((1, 2), float("nan"))
    assert routed.weight_loader(
        per_tensor,
        torch.tensor(7.0),
        "w1_weight_scale_2",
        "w1",
        0,
        return_success=True,
    )
    assert per_tensor[0, 0] == 7.0
    assert torch.isnan(per_tensor[0, 1])

    assert routed.weight_loader(
        per_tensor,
        torch.tensor(11.0),
        "w3_weight_scale_2",
        "w3",
        0,
        return_success=True,
    )
    assert per_tensor[0, 1] == 11.0

    input_scale = torch.full((1, 2), float("nan"))
    assert routed.weight_loader(
        input_scale,
        torch.tensor(13.0),
        "w1_input_scale",
        "w1",
        0,
        return_success=True,
    )
    assert input_scale[0, 0] == 13.0
    assert torch.isnan(input_scale[0, 1])
    assert routed.weight_loader(
        input_scale,
        torch.tensor(17.0),
        "w3_input_scale",
        "w3",
        0,
        return_success=True,
    )
    assert input_scale[0, 1] == 17.0

    # ModelOpt's combined w13 block-scale layout is TP-sliced across the fused
    # intermediate dimension. Force rank 1/2 and verify it receives the second
    # half rather than broadcasting a scalar or overwriting the whole tensor.
    moe.tp_rank = 1
    moe.tp_size = 2
    block_param = torch.nn.Parameter(
        torch.full((1, 4, 2), float("nan"), dtype=torch.float32)
    )
    block_param.quant_method = "block"
    combined = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    assert routed.weight_loader(
        block_param,
        combined,
        "w13_weight_scale",
        "w1",
        0,
        return_success=True,
    )
    torch.testing.assert_close(block_param.data[0], combined[0, 4:8])


def test_process_weights_after_loading_builds_native_sidecar(
    method_inputs, monkeypatch
):
    p8_native_kernel = pytest.importorskip(
        "b12x.moe._shared.trellismx.p8_native_kernel"
    )

    real_empty = torch.empty
    calls = []
    sidecar = object()
    runtime = SimpleNamespace(device=object())

    class Recorder:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return runtime

    monkeypatch.setattr(p8_native_kernel, "P8NativeTPMoE", Recorder())
    monkeypatch.setattr(
        trellismx.torch.cuda, "get_device_capability", lambda _: (12, 0)
    )
    monkeypatch.setattr(
        trellismx.torch,
        "empty",
        lambda *args, **kwargs: real_empty(0),
    )

    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    method.overlay = SimpleNamespace(
        sidecar=lambda layer, rank: sidecar,
        records={(3, 0): {"source_design_sha256": "design", "bits": 5}},
        transform_hash="transform",
    )
    fake_cuda = SimpleNamespace(type="cuda")
    layer = SimpleNamespace(
        **{
            name: SimpleNamespace(
                device=fake_cuda,
                dtype=torch.float32,
                numel=lambda: 2,
                element_size=lambda: 4,
            )
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            )
        }
    )
    method.process_weights_after_loading(layer)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (sidecar,)
    assert kwargs["tp_rank"] == 0
    assert kwargs["layer"] == 3
    assert kwargs["expected_design_sha256"] == "design"
    assert kwargs["expected_transform_sha256"] == "transform"
    assert method.runtime is runtime
    assert layer.b12x_warmup_provider is method
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_2",
        "w2_weight_scale_2",
        "w13_input_scale",
        "w2_input_scale",
    ):
        value = getattr(layer, name)
        assert isinstance(value, torch.nn.Parameter)
        assert value.numel() == 0


def test_dispatch_passes_only_routed_input_and_routes(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    x = torch.empty((1, 4096), dtype=torch.bfloat16)
    weights, ids, shared = object(), object(), object()
    seen = []

    def record(*args):
        seen.append(args)
        return "routed result"

    method.runtime = record
    result = method.apply(None, x, weights, ids, shared, shared)
    assert seen == [(x, weights, ids)]
    assert result == "routed result"
    with pytest.raises(RuntimeError, match="routing belongs"):
        method.apply_monolithic()


def test_empty_dispatch_does_not_launch(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    method.runtime = lambda *args: pytest.fail("empty batch launched")
    result = method.apply(None, torch.empty((0, 4096)), None, None, None, None)
    assert result.shape == (0, 4096)


def test_missing_runtime_cannot_serve_carrier(method_inputs):
    config, moe = method_inputs
    method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", 3)
    with pytest.raises(RuntimeError, match="not loaded"):
        method.apply(None, None, None, None, None, None)


def test_native_runtime_operator_namespace_matches_calls():
    p8_native_kernel = pytest.importorskip(
        "b12x.moe._shared.trellismx.p8_native_kernel"
    )

    assert p8_native_kernel.P8NativeTPMoE is not None
    assert hasattr(torch.ops.b12x, "dense_gemm_launch")
    assert hasattr(torch.ops.b12x, "tp_moe_dynamic_launch")


def test_warmup_covers_decode_verification_and_prefill_without_dedup(method_inputs):
    config, moe = method_inputs
    calls = []

    class Recorder:
        device = "cpu"

        def __call__(self, x, weights, ids):
            calls.append((tuple(x.shape), tuple(weights.shape), tuple(ids.shape)))

    keys = []
    for index in (3, 4):
        method = trellismx.TrellisMXMoEMethod(config, moe, "/fixture", index)
        method.runtime = Recorder()
        unit = method.get_b12x_warmup_unit(None, (1, 4, 17), torch.bfloat16)
        keys.append(unit.key)
        unit.compile()
    assert keys[0] != keys[1]
    assert [call[0][0] for call in calls] == [1, 4, 17, 1, 4, 17]
    assert all(call[1][1] == call[2][1] == 8 for call in calls)


def test_mtp_alias_is_opt_in_and_does_not_select_p8(method_inputs, monkeypatch):
    config, moe = method_inputs
    prefix = "model.layers.45.mtp_block.mlp.experts"
    canonical = "model.language_model.layers.45.mlp.experts"
    monkeypatch.delenv("VLLM_TRELLISMX_CHECKPOINT", raising=False)
    assert (
        canonical
        not in ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates(prefix)
    )
    monkeypatch.setenv("VLLM_TRELLISMX_CHECKPOINT", "/fixture")
    assert canonical in ModelOptMixedPrecisionConfig._quantized_layer_prefix_candidates(
        prefix
    )
    assert (
        trellismx.maybe_trellismx_method(
            config, SimpleNamespace(moe_config=moe), prefix
        )
        is None
    )
