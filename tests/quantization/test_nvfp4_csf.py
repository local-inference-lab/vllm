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


@pytest.mark.parametrize("quant_method", ["nvfp4_csf"])
@pytest.mark.parametrize("load_format", ["nvfp4_csf"])
def test_loader_excludes_compressed_main_experts_and_retains_native_tensors(
    tmp_path, quant_method, load_format
):
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.nvfp4_csf_loader import CODEC, SCHEMA

    directory = tmp_path / "tensors"
    directory.mkdir()
    filename = "weights.safetensors"
    tensors = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones(
            8, dtype=torch.uint8
        ),
        (
            "model.language_model.layers.3.mlp.experts.0.gate_proj."
            "weight_scale.nvfp4_csf_fixed"
        ): torch.ones(8, dtype=torch.uint8),
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight": torch.full(
            (8,), 42, dtype=torch.uint8
        ),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, directory / filename)
    common = {"schema": SCHEMA, "codec": CODEC, "family": "glm53_nvfp4"}
    (tmp_path / "manifest.json").write_text(
        json.dumps({**common, "shards": [{"file": filename}]})
    )
    (tmp_path / "build-contract.json").write_text(
        json.dumps({**common, "source_names": {k: filename for k in tensors}})
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "quant_method": quant_method,
                "checkpoint_root": str(tmp_path),
            }
        ),
        hf_text_config=SimpleNamespace(num_hidden_layers=45),
    )
    loader = get_model_loader(LoadConfig(load_format=load_format))
    assert isinstance(loader, Nvfp4CsfModelLoader)
    assert get_quantization_config("nvfp4_csf") is Nvfp4CsfConfig
    actual = dict(loader.get_all_weights(config, SimpleNamespace()))
    assert set(actual) == {name for name in tensors if ".layers.3." not in name}
    for name, tensor in actual.items():
        assert torch.equal(tensor, tensors[name])


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
