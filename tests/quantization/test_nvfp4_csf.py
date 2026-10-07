# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed expert storage must retain nonexpert precision and tensors."""

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
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.nvfp4_csf import (
    Nvfp4CsfConfig,
    Nvfp4CsfMoEMethod,
)
from vllm.model_executor.model_loader.nvfp4_csf_loader import Nvfp4CsfModelLoader

from .csf_fixtures import decode_planes, matrix


@pytest.mark.parametrize("config_cls", [Nvfp4CsfConfig])
def test_csf_config_preserves_source_recipes_and_avoids_expert_allocations(config_cls):
    original = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.45.mlp.experts": {"quant_algo": "MXFP8"},
        },
    }
    owner = config_cls.from_config(
        {
            "format_version": 1,
            "checkpoint_root": "/lsc",
            "source_quantization_config": original,
        }
    )
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
    method = Nvfp4CsfMoEMethod(moe, owner)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    method.create_weights(layer, 288, 4096, 1024, torch.bfloat16)
    assert not list(layer.parameters()) and not layer.state_dict()
    assert layer.w13_weight.numel() == layer.w2_weight.numel() == 0
    with pytest.raises(ValueError, match="source_quantization_config"):
        Nvfp4CsfConfig.from_config({"format_version": 1, "checkpoint_root": "/lsc"})


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


def _write_checkpoint(root, family, tensors):
    from vllm.model_executor.model_loader.nvfp4_csf_loader import CODEC, SCHEMA

    directory = root / "tensors"
    directory.mkdir()
    filename = "weights.safetensors"
    save_file(tensors, directory / filename)
    common = {"schema": SCHEMA, "codec": CODEC, "family": family}
    (root / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": filename}]})
    )
    (root / "build-contract.json").write_text(
        json.dumps({**common, "source_names": {k: filename for k in tensors}})
    )


@pytest.mark.parametrize("quant_method", ["nvfp4_csf"])
@pytest.mark.parametrize("load_format", ["nvfp4_csf"])
@pytest.mark.parametrize(
    "family,layer_list,num_layers",
    [
        ("glm53_nvfp4", "model.language_model.layers", 45),
        # GLM-5.3 (744B): the MTP layer 78 stores BF16 experts.
        ("glm53_744b_nvfp4", "model.layers", 78),
    ],
)
def test_loader_excludes_compressed_main_experts_and_retains_native_tensors(
    tmp_path, quant_method, load_format, family, layer_list, num_layers
):
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import get_model_loader

    tensors = {
        f"{layer_list}.3.mlp.experts.0.gate_proj.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        (
            f"{layer_list}.3.mlp.experts.0.gate_proj.weight_scale.nvfp4_csf_fixed"
        ): torch.ones(8, dtype=torch.uint8),
        f"{layer_list}.{num_layers}.mlp.experts.0.gate_proj.weight": torch.full(
            (8,), 42, dtype=torch.uint8
        ),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    _write_checkpoint(tmp_path, family, tensors)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": quant_method,
                "checkpoint_root": str(tmp_path),
            }
        ),
        hf_text_config=SimpleNamespace(num_hidden_layers=num_layers),
    )
    loader = get_model_loader(LoadConfig(load_format=load_format))
    assert isinstance(loader, Nvfp4CsfModelLoader)
    assert get_quantization_config("nvfp4_csf") is Nvfp4CsfConfig
    actual = dict(loader.get_all_weights(config, SimpleNamespace()))
    assert set(actual) == {name for name in tensors if ".layers.3." not in name}
    for name, tensor in actual.items():
        assert torch.equal(tensor, tensors[name])


def test_single_nvfp4_source_quantizes_the_modules_that_store_scales(tmp_path):
    """GLM-5.3 (744B) declares one NVFP4 recipe whose ignore list misses its
    BF16 MTP layer; the checkpoint's weight scales decide instead."""
    names = (
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight_scale",
        "model.layers.3.mlp.experts.1.down_proj.weight_scale",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.78.mlp.experts.0.up_proj.weight",
        "model.layers.78.self_attn.o_proj.weight",
    )
    _write_checkpoint(
        tmp_path,
        "glm53_744b_nvfp4",
        {name: torch.ones(1, dtype=torch.uint8) for name in names},
    )
    owner = Nvfp4CsfConfig.from_config(
        {
            "format_version": 1,
            "checkpoint_root": str(tmp_path),
            "source_quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "ignore": ["lm_head", "model.layers.3.self_attn*"],
            },
        }
    )
    assert owner.quantized_layers == {
        "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
    }
    assert owner._resolve_quant_algo("model.layers.3.mlp.experts") == "NVFP4"
    assert owner._resolve_quant_algo("model.layers.78.mlp.experts") is None
    assert owner._resolve_quant_algo("model.layers.78.self_attn.o_proj") is None
    assert owner.is_layer_excluded("model.layers.3.self_attn.o_proj")


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_sources_preserve_tp_bytes_and_fp32_with_bf16_default(rank):
    from vllm.model_executor.model_loader.nvfp4_csf_loader import (
        _load_nvfp4_csf_weights as load_nvfp4_csf_weights,
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
    from vllm.model_executor.model_loader.nvfp4_csf_loader import (
        _load_nvfp4_csf_weights as load_nvfp4_csf_weights,
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
    from vllm.model_executor.model_loader.nvfp4_csf_loader import (
        _load_nvfp4_csf_weights as load_nvfp4_csf_weights,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_scale_prefetch_expands_the_next_layer_for_this_forward_only(monkeypatch):
    """Layer N expands layer N+1's scales on the side stream after its own MoE;
    layer N+1 skips its expansion in the same forward pass only."""
    pytest.importorskip("b12x")
    import b12x.moe.fused_moe as fused_moe

    import vllm.forward_context as forward_context

    expanded = []
    monkeypatch.setattr(
        fused_moe,
        "expand_scales",
        lambda prepared: expanded.append((prepared, torch.cuda.current_stream())),
    )
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")

    def step(starts, gdn_prefills=None):
        metadata = {
            "attn": SimpleNamespace(
                query_start_loc=torch.tensor(starts, dtype=torch.int32),
                num_actual_tokens=starts[-1],
            )
        }
        if gdn_prefills is not None:
            metadata["kda"] = SimpleNamespace(
                num_prefills=gdn_prefills, num_spec_decode_tokens=starts[-1]
            )
        context = SimpleNamespace(attn_metadata=metadata)
        monkeypatch.setattr(
            forward_context, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
        return torch.zeros(starts[-1], 8)

    owner = SimpleNamespace(
        scale_layers={}, scale_stream=torch.cuda.Stream(), scale_prefetch=None
    )
    calls = []

    def method(index):
        m = object.__new__(Nvfp4CsfMoEMethod)
        m.owner, m.layer_index, m.a4_prefill = owner, index, True
        m.backend = SimpleNamespace(scales_expanded=False)
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
        for m in (first, second):
            m.apply(layer, x, None, None, None, None)

    forward(step([0, 4, 300]))  # decode rows, then prefill rows
    assert calls == [(3, False), (4, True)]
    assert [p for p, _ in expanded] == ["layer-4"]
    assert expanded[0][1] == owner.scale_stream
    assert owner.scale_prefetch is None
    # A prefetch left from another forward pass is waited for, never reused.
    owner.scale_prefetch = (4, -1, second.scales_ready)
    calls.clear()
    second.apply(layer, step([0, 300]), None, None, None, None)
    assert calls == [(4, False)]
    # Decode-only steps (a GDN layer counts no prefills) expand nothing.
    expanded.clear()
    calls.clear()
    forward(step([0, 4, 8], gdn_prefills=0))
    assert calls == [(3, False), (4, False)] and expanded == []


def _write_hf_layout_checkpoint(root, retained):
    """A Hugging Face-layout GLM-5.3-Flash checkpoint: the index names every
    routed-expert scale stream; one shard holds the tensors under test."""
    layers = "model.language_model.layers"
    shard = "tensors/model-00001-of-00001.safetensors"
    (root / "tensors").mkdir()
    save_file(retained, root / shard)
    weight_map = dict.fromkeys(retained, shard)
    for layer in range(3, 45):
        for expert in range(288):
            for projection in ("up_proj", "gate_proj", "down_proj"):
                scale = f"{layers}.{layer}.mlp.experts.{expert}.{projection}"
                for stream in ("fixed", "exceptions"):
                    weight_map[f"{scale}.weight_scale.nvfp4_csf_{stream}"] = shard
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


_CSF_RECIPES = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "model.language_model.layers.3.mlp.experts": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "weight_scale_encoding": "csf",
        },
        "model.language_model.layers.3.self_attn.o_proj": {
            "quant_algo": "MXFP8",
            "group_size": 32,
        },
        "model.visual.blocks.0.attn.qkv": {"quant_algo": "MXFP8", "group_size": 32},
        "model.visual.blocks.0.mlp.gate_proj": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
        "model.visual.blocks.0.mlp.up_proj": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
    },
}


def test_csf_recipes_in_a_modelopt_config_select_the_csf_reader():
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    plain = {
        **_CSF_RECIPES,
        "quantized_layers": {
            name: {k: v for k, v in recipe.items() if k != "weight_scale_encoding"}
            for name, recipe in _CSF_RECIPES["quantized_layers"].items()
        },
    }
    assert Nvfp4CsfConfig.override_quantization_method(_CSF_RECIPES, None) == (
        "nvfp4_csf"
    )
    assert (
        ModelOptMixedPrecisionConfig.override_quantization_method(_CSF_RECIPES, None)
        is None
    )
    assert Nvfp4CsfConfig.override_quantization_method(plain, None) is None
    assert ModelOptMixedPrecisionConfig.override_quantization_method(plain, None) == (
        "modelopt_mixed"
    )


def test_csf_modelopt_config_keeps_recipes_including_the_vision_tower():
    owner = Nvfp4CsfConfig.from_config(_CSF_RECIPES)
    assert owner.checkpoint_root is None
    assert owner._resolve_quant_algo("language_model.model.layers.3.mlp.experts") == (
        "NVFP4"
    )
    assert owner._resolve_quant_algo("visual.blocks.0.attn.qkv") == "MXFP8"
    owner.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert owner._resolve_quant_algo("visual.blocks.0.mlp.gate_up_proj") == (
        "W4A16_NVFP4"
    )


def test_hf_layout_loader_reads_retained_tensors_and_roots_the_experts(tmp_path):
    from vllm.model_executor.model_loader import get_model_loader

    layers = "model.language_model.layers"
    retained = {
        f"{layers}.3.mlp.experts.0.gate_proj.weight": torch.ones(8, dtype=torch.uint8),
        f"{layers}.45.mlp.experts.0.gate_proj.weight": torch.full(
            (8,), 42, dtype=torch.uint8
        ),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    _write_hf_layout_checkpoint(tmp_path, retained)
    config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        hf_config=SimpleNamespace(quantization_config=_CSF_RECIPES),
        hf_text_config=SimpleNamespace(num_hidden_layers=45),
    )
    owner = SimpleNamespace(checkpoint_root=None)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.quant_method = SimpleNamespace(owner=owner)

    loader = get_model_loader(LoadConfig(load_format="nvfp4_csf"))
    actual = dict(loader.get_all_weights(config, model))

    assert set(actual) == {name for name in retained if ".layers.3." not in name}
    assert owner.checkpoint_root == str(tmp_path.resolve())
