# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed expert storage must retain nonexpert precision and tensors."""

import json
from contextlib import ExitStack, contextmanager
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
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.layers.quantization.nvfp4_csf import (
    Nvfp4CsfMoEMethod,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .csf_fixtures import decode_planes, matrix


@pytest.mark.parametrize("selected", [None, "modelopt_mixed"])
def test_model_config_detects_csf(selected):
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

    config = SimpleNamespace(
        quantization=selected,
        model_arch_config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
            }
        ),
        hf_config=PretrainedConfig(),
    )
    ModelConfig._verify_quantization(config)
    assert config.quantization == "modelopt_mixed"


def test_csf_config_preserves_recipes_and_avoids_expert_allocations():
    original = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.3.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "weight_scale_encoding": "csf",
            },
            "model.layers.45.mlp.experts": {"quant_algo": "MXFP8"},
        },
    }
    owner = ModelOptMixedPrecisionConfig.from_config(original)
    assert owner._resolve_quant_algo("model.layers.3.mlp.experts") == "NVFP4"
    assert owner._resolve_quant_algo("model.layers.45.mlp.experts") == "MXFP8"
    moe = FusedMoEConfig(
        num_experts=288,
        num_local_experts=288,
        num_logical_experts=288,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size=1024,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = Nvfp4CsfMoEMethod(moe, owner.csf_state)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    method.create_weights(layer, 288, 4096, 1024, torch.bfloat16)
    assert all(p.numel() == 0 for p in layer.parameters())
    assert owner._resolve_weight_scale_encoding("model.layers.3.mlp.experts") == "csf"
    assert owner._resolve_weight_scale_encoding("model.layers.45.mlp.experts") is None


def test_prepared_nvfp4_accepts_kernel_order_and_rejects_activation_change(monkeypatch):
    from b12x.moe import fused_moe

    from vllm.model_executor.layers.fused_moe import b12x

    moe = FusedMoEConfig(
        num_experts=8,
        num_local_experts=8,
        num_logical_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=128,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        swiglu_limit=10.0,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    one = torch.ones(8)
    quant = nvfp4_moe_quant_config(one, one, one, one, one, one, gemm1_clamp_limit=10.0)
    backend = b12x.B12xExperts(moe, quant)
    layer = torch.nn.Module()
    layer.activation, layer.apply_router_weight_on_input = moe.activation, False
    prepared = Mock(spec=fused_moe.PreparedExperts)
    activation = fused_moe.ActivationSpec(
        mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
    )
    prepared.plan = SimpleNamespace(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=activation,
    )
    prepared.num_experts, prepared.hidden_size, prepared.intermediate_size = 8, 256, 128
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *a, **k: None
    )
    backend.install_prepared_experts(layer, prepared)
    assert backend._prepared_experts is prepared
    prepared.plan.activation = replace(activation, mode="a16")
    with pytest.raises(ValueError, match="activation"):
        backend.install_prepared_experts(layer, prepared)


@pytest.mark.parametrize("use_a16", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
def test_csf_preparation_preserves_phase_policy_or_a16_cutoff(
    monkeypatch, use_a16, hybrid
):
    """Semantic hybrid precision is independent of the ordinary A16 cutoff."""
    from b12x.moe import fused_moe

    monkeypatch.setenv("VLLM_B12X_ACTIVATION_MODE_A16_M", "32")
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", "1" if hybrid else "0")
    moe = FusedMoEConfig(
        num_experts=1,
        num_local_experts=1,
        num_logical_experts=1,
        experts_per_token=1,
        hidden_dim=256,
        intermediate_size=128,
        in_dtype=torch.bfloat16,
        device="cpu",
        activation=MoEActivation.SILU,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )
    method = Nvfp4CsfMoEMethod(
        moe, SimpleNamespace(scale_scratch=None), use_a16=use_a16
    )
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    method.create_weights(layer, 1, 256, 128, torch.bfloat16)
    projections = []
    for rows, channels in ((128, 256), (128, 256), (256, 128)):
        source, _ = matrix(rows, channels, group_size=16, seed=7)
        projections.append(
            replace(
                source,
                global_scale=torch.tensor(1.0),
                input_scale=torch.tensor(0.5),
            )
        )
    monkeypatch.setattr(method, "_expert_tensors", lambda: iter([tuple(projections)]))
    plans = []

    def stop_before_cuda_preparation(*, plan, weights):
        plans.append(plan)
        raise StopIteration

    monkeypatch.setattr(fused_moe, "prepare_weights", stop_before_cuda_preparation)
    with pytest.raises(StopIteration):
        method.process_weights_after_loading(layer)
    assert plans[0].activation.a16_max_tokens == (32 if use_a16 and not hybrid else 0)
    assert plans[0].activation.mode == ("a16" if use_a16 else "a4")


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_standard_loader_yields_all_csf_components_and_native_tensors(
    tmp_path, load_format
):
    tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        (
            "model.layers.3.mlp.experts.0.gate_proj.weight_scale.nvfp4_csf_fixed"
        ): torch.ones(8, dtype=torch.uint8),
        (
            "model.layers.3.mlp.experts.0.gate_proj.weight_scale.nvfp4_csf_exceptions"
        ): torch.tensor([42], dtype=torch.uint32),
        "model.layers.45.mlp.experts.0.gate_proj.weight": torch.full(
            (8,), 42, dtype=torch.uint8
        ),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
    }
    directory = tmp_path / "tensors"
    directory.mkdir()
    mapping = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        filename = f"tensors/model-{i:05}.safetensors"
        save_file({name: tensor}, tmp_path / filename)
        mapping[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    loader = DefaultModelLoader(LoadConfig(load_format=load_format))
    actual = dict(
        loader._get_weights_iterator(
            DefaultModelLoader.Source(str(tmp_path), revision=None)
        )
    )
    assert actual.keys() == tensors.keys()
    for name, tensor in actual.items():
        assert tensor.dtype == tensors[name].dtype
        assert torch.equal(tensor, tensors[name])


@pytest.mark.parametrize("algo,encoding", [("MXFP8", "csf"), ("NVFP4", "unknown")])
def test_unsupported_scale_encoding_is_rejected(algo, encoding):
    with pytest.raises(ValueError, match="weight_scale_encoding"):
        ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.experts": {
                        "quant_algo": algo,
                        "weight_scale_encoding": encoding,
                    }
                },
            }
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_sources_preserve_tp_bytes_and_fp32_with_bf16_default(rank):
    from vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils import (
        prepare_nvfp4_csf_weights as load_nvfp4_csf_weights,
    )

    device = "cpu"
    experts, hidden, intermediate, local = 3, 256, 256, 128
    calibration = torch.linspace(0.013123, 0.056789, experts, dtype=torch.float32)
    gate_input = calibration * 19.321
    up_input = calibration * 20.123
    down_input = calibration * 31.345
    sources, scales = [], []
    for expert in range(experts):
        pairs = [
            matrix(r, c, group_size=16, seed=expert * 17 + projection)
            for projection, (r, c) in enumerate(
                ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
            )
        ]
        sources.append(
            tuple(
                replace(p[0], global_scale=calibration[expert], input_scale=a[expert])
                for p, a in zip(pairs, (up_input, gate_input, down_input), strict=True)
            )
        )
        scales.append(tuple(pair[1] for pair in pairs))
    scratch13 = torch.empty(
        (experts, 2 * local, hidden // 16), device=device, dtype=torch.float8_e4m3fn
    )
    scratch2 = torch.empty(
        (experts, hidden, local // 16), device=device, dtype=torch.float8_e4m3fn
    )
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        weights = load_nvfp4_csf_weights(
            iter(sources),
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tp_rank=rank,
            tp_size=2,
            device=device,
            w13_scale_scratch=scratch13,
            w2_scale_scratch=scratch2,
        )
    finally:
        torch.set_default_dtype(default)
    for name, expected in (
        ("w13_global_scales", calibration.to(device)),
        ("w2_global_scales", calibration.to(device)),
        ("input_scale", up_input.to(device).reciprocal()),
        ("intermediate_scale", down_input.to(device).reciprocal()),
    ):
        actual = getattr(weights.packed, name)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    first, last = rank * local, (rank + 1) * local
    assert torch.equal(
        weights.packed.w13.cpu(),
        torch.stack(
            [
                torch.cat((up.weight[first:last, :], gate.weight[first:last, :]))
                for up, gate, _ in sources
            ]
        ),
    )
    assert torch.equal(
        weights.packed.w2.cpu(),
        torch.stack([down.weight[:, first // 2 : last // 2] for _, _, down in sources]),
    )
    assert weights.packed.w13_block_scales is scratch13
    assert weights.packed.w2_block_scales is scratch2
    expected13 = torch.stack(
        [torch.cat((s[0][first:last], s[1][first:last])) for s in scales]
    )
    expected2 = torch.stack([s[2][:, first // 16 : last // 16] for s in scales])
    for planes, expected in (
        (weights.w13_scales, expected13),
        (weights.w2_scales, expected2),
    ):
        actual = decode_planes(
            planes, expected.shape[1], expected.shape[2], group_size=16
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_padded_shards_hold_whole_tiles_and_zero_the_tail(rank):
    """TP-padded experts (GLM-5.3 at TP6): rank r holds checkpoint channels
    [r * local, (r + 1) * local) and zero weights past the checkpoint width,
    here 320 channels over three 128-channel shards."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils import (
        prepare_nvfp4_csf_weights as load_nvfp4_csf_weights,
    )

    experts, hidden, intermediate, local = 2, 256, 320, 128
    one = torch.tensor(1.0)
    sources, scales = [], []
    for expert in range(experts):
        pairs = [
            matrix(r, c, group_size=16, seed=expert * 17 + projection)
            for projection, (r, c) in enumerate(
                ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
            )
        ]
        sources.append(
            tuple(replace(p[0], global_scale=one, input_scale=one) for p in pairs)
        )
        scales.append(tuple(pair[1] for pair in pairs))
    weights = load_nvfp4_csf_weights(
        iter(sources),
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tp_rank=rank,
        tp_size=3,
        device="cpu",
        w13_scale_scratch=torch.empty(
            (experts, 2 * local, hidden // 16), dtype=torch.float8_e4m3fn
        ),
        w2_scale_scratch=torch.empty(
            (experts, hidden, local // 16), dtype=torch.float8_e4m3fn
        ),
        local_size=local,
    )
    first = rank * local
    real = min(intermediate - first, local)
    w13, w2 = weights.packed.w13, weights.packed.w2
    assert w13.shape == (experts, 2 * local, hidden // 2)
    assert w2.shape == (experts, hidden, local // 2)
    for expert, (up, gate, down) in enumerate(sources):
        assert torch.equal(w13[expert, :real], up.weight[first : first + real, :])
        assert torch.equal(
            w13[expert, local : local + real], gate.weight[first : first + real, :]
        )
        assert torch.equal(
            w2[expert, :, : real // 2], down.weight[:, first // 2 : (first + real) // 2]
        )
    assert not w13[:, real:local].any() and not w13[:, local + real :].any()
    assert not w2[:, :, real // 2 :].any()
    decoded13 = decode_planes(weights.w13_scales, 2 * local, hidden // 16, 16)
    decoded2 = decode_planes(weights.w2_scales, hidden, local // 16, 16)
    for expert, (up, gate, down) in enumerate(scales):
        assert torch.equal(decoded13[expert, :real], up[first : first + real])
        assert torch.equal(
            decoded13[expert, local : local + real], gate[first : first + real]
        )
        assert torch.equal(
            decoded2[expert, :, : real // 16],
            down[:, first // 16 : (first + real) // 16],
        )
    # Padded rows scale zero weights by 1.0 (E4M3 0x38), which packs without
    # replacement words.
    assert (decoded13[:, real:local] == 0x38).all()
    assert (decoded13[:, local + real :] == 0x38).all()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.tensor(1.0, dtype=torch.bfloat16),
        torch.tensor(float("nan")),
        torch.tensor(0.0),
    ],
)
def test_missing_or_invalid_calibration_is_rejected(bad):
    from vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils import (
        prepare_nvfp4_csf_weights as load_nvfp4_csf_weights,
    )

    source, _ = matrix(128, 128, group_size=16, seed=0)
    source = replace(source, global_scale=bad, input_scale=torch.tensor(1.0))
    with pytest.raises(ValueError, match="global_scale"):
        load_nvfp4_csf_weights(
            [(source, source, source)],
            num_experts=1,
            hidden_size=128,
            intermediate_size=128,
            tp_rank=0,
            tp_size=1,
            device="cpu",
            w13_scale_scratch=None,
            w2_scale_scratch=None,
        )


def test_scale_prefetch_expands_the_next_layer_for_this_forward_only(monkeypatch):
    """Layer N expands layer N+1's scales on the side stream after its own MoE;
    layer N+1 skips its expansion in the same forward pass only."""
    pytest.importorskip("b12x")
    import b12x.moe.fused_moe as fused_moe

    import vllm.forward_context as forward_context
    import vllm.model_executor.layers.quantization.nvfp4_csf as csf

    main_stream, side_stream = Mock(), Mock()
    active_stream = main_stream

    @contextmanager
    def use_stream(stream):
        nonlocal active_stream
        previous, active_stream = active_stream, stream
        try:
            yield
        finally:
            active_stream = previous

    monkeypatch.setattr(csf, "_is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active_stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: side_stream)
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    expanded = []
    monkeypatch.setattr(
        fused_moe,
        "expand_scales",
        lambda prepared: expanded.append((prepared, torch.cuda.current_stream())),
    )

    def step(tokens):
        context = SimpleNamespace(attn_metadata={}, additional_kwargs={})
        monkeypatch.setattr(
            forward_context, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
        return torch.zeros(tokens, 8)

    owner = SimpleNamespace(
        scale_layers={}, scale_stream=torch.cuda.Stream(), scale_prefetch=None
    )
    calls = []

    def method(index):
        m = object.__new__(Nvfp4CsfMoEMethod)
        m.owner, m.layer_index = owner, index
        m.backend = SimpleNamespace(scales_expanded=False, _a4_prefill_enabled=False)
        m.prepared = f"layer-{index}"
        m.moe_done, m.scales_ready = torch.cuda.Event(), torch.cuda.Event()
        m.moe_kernel = SimpleNamespace(
            apply=lambda **_: calls.append((index, m.backend.scales_expanded))
        )
        owner.scale_layers[index] = m
        return m

    first, second = method(3), method(4)
    layer = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        activation=MoEActivation.SILU,
        global_num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    def forward(x):
        ids = torch.zeros(x.shape[0], 1, dtype=torch.int64)
        for m in (first, second):
            m.apply(layer, x, None, ids, None, None)

    forward(step(2000))  # above the stage-read limit: the calls expand
    assert calls == [(3, False), (4, True)]
    assert [p for p, _ in expanded] == ["layer-4"]
    assert expanded[0][1] == owner.scale_stream
    assert owner.scale_prefetch is None
    # A prefetch left from another forward pass is waited for, never reused.
    owner.scale_prefetch = (4, -1, second.scales_ready)
    calls.clear()
    second.apply(layer, step(2000), None, None, None, None)
    assert calls == [(4, False)]
    main_stream.wait_event.assert_called_with(second.scales_ready)
    # Decode-sized calls read compressed scales per stage: nothing to expand.
    expanded.clear()
    calls.clear()
    forward(step(8))
    assert calls == [(3, False), (4, False)] and expanded == []
    # The following layer's selected consumer governs prefetch independently.
    first.backend._a4_prefill_enabled = second.backend._a4_prefill_enabled = True
    first.backend.uses_expanded_nvfp4_scales = Mock(return_value=True)
    second.backend.uses_expanded_nvfp4_scales = Mock(return_value=False)
    forward(step(2000))
    first.backend.uses_expanded_nvfp4_scales.assert_not_called()
    second.backend.uses_expanded_nvfp4_scales.assert_called_once_with(2000, torch.int64)
    assert expanded == []
    second.backend.uses_expanded_nvfp4_scales.return_value = True
    forward(step(2000))
    assert [p for p, _ in expanded] == ["layer-4"]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_normal_expert_hooks_preserve_tp_weights_scales_and_calibration(
    tmp_path, rank, reverse, direct
):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils import (
        prepare_nvfp4_csf_weights,
    )

    if direct and not torch.cuda.is_available():
        pytest.skip("B12X direct checkpoint transfers require CUDA")
    device = "cuda" if direct else "cpu"
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(), tp_size=2, tp_rank=rank
    )
    moe = FusedMoEConfig(
        num_experts=2,
        num_local_experts=2,
        num_logical_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        in_dtype=torch.bfloat16,
        device=device,
        activation=MoEActivation.SILU,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel,
    )
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.moe_config = moe
    layer.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
    layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    layer.lora_base_layer_prefix = ""
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                layer.layer_name: {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "weight_scale_encoding": "csf",
                }
            },
        }
    )
    method = config.get_quant_method(layer, layer.layer_name)
    assert isinstance(method, Nvfp4CsfMoEMethod)
    with torch.device(device):
        method.create_weights(layer, 2, 128, 64, torch.bfloat16)
    tensors, sources = {}, []
    for expert in range(2):
        projections = []
        for projection, proj_name in enumerate(("up_proj", "gate_proj", "down_proj")):
            source, _ = matrix(128, 128, group_size=16, seed=expert * 3 + projection)
            source = replace(
                source,
                global_scale=torch.tensor(0.031234 + expert),
                input_scale=torch.tensor(0.234567 + projection),
            )
            projections.append(source)
            prefix = f"{expert}.{proj_name}."
            tensors.update(
                {
                    prefix + suffix: tensor
                    for suffix, tensor in (
                        ("weight", source.weight.tensor),
                        ("weight_scale.nvfp4_csf_fixed", source.fixed),
                        ("weight_scale.nvfp4_csf_exceptions", source.exceptions),
                        ("weight_scale_2", source.global_scale),
                        ("input_scale", source.input_scale),
                    )
                }
            )
        sources.append(tuple(projections))
    items = list(tensors.items())
    if reverse:
        items.reverse()
    # Each component in a separate shard exercises arrival-order independence.
    mapping = {}
    for i, (name, tensor) in enumerate(items):
        filename = f"model-{i:03d}.safetensors"
        save_file({name: tensor}, tmp_path / filename)
        mapping[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with ExitStack() as stack:
        weights = loader._get_weights_iterator(
            DefaultModelLoader.Source(str(tmp_path), revision=None)
        )
        if direct:
            from b12x.loader._checkpoint import DirectWeightSession

            from vllm.model_executor.weight_transfer import weight_transfer

            session = stack.enter_context(DirectWeightSession(read_mode="bounce"))
            stack.enter_context(weight_transfer(session))
            weights = session.weights([tmp_path / name for name in mapping.values()])
        loaded = set(layer.load_weights(weights))
    assert loaded == set(dict(layer.named_parameters()))
    kwargs = dict(
        num_experts=2,
        hidden_size=128,
        device="cpu",
        w13_scale_scratch=torch.empty((2, 128, 8), dtype=torch.float8_e4m3fn),
        w2_scale_scratch=torch.empty((2, 128, 4), dtype=torch.float8_e4m3fn),
    )
    actual = prepare_nvfp4_csf_weights(
        method._expert_tensors(), intermediate_size=64, tp_rank=0, tp_size=1, **kwargs
    )
    expected = prepare_nvfp4_csf_weights(
        sources, intermediate_size=128, tp_rank=rank, tp_size=2, **kwargs
    )
    for name in (
        "w13",
        "w2",
        "w13_global_scales",
        "w2_global_scales",
        "input_scale",
        "intermediate_scale",
    ):
        torch.testing.assert_close(
            getattr(actual.packed, name), getattr(expected.packed, name), rtol=0, atol=0
        )
    for name, rows, columns in (("w13_scales", 128, 8), ("w2_scales", 128, 4)):
        assert torch.equal(
            decode_planes(getattr(actual, name), rows, columns, 16),
            decode_planes(getattr(expected, name), rows, columns, 16),
        )
    missing = method._components[(0, "w1")].pop("fixed")
    with pytest.raises(ValueError, match="Missing CSF tensors"):
        list(method._expert_tensors())
    method._components[(0, "w1")]["fixed"] = missing
