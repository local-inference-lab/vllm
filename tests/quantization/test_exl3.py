# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.exl3 import plan_exl3_extent


def _kimi_manifest(**layout):
    return SimpleNamespace(
        geometry=SimpleNamespace(
            num_slots=96,
            slot_channels=32,
            moe_layer_indices=tuple(range(1, 93)),
        ),
        layout=SimpleNamespace(
            extent_alignment_slots=layout.get("alignment", 4),
            extent_barriers=layout.get("barriers", (48,)),
        ),
    )


@pytest.mark.parametrize("tp_size", range(2, 25))
def test_exl3_rank_extents_cover_complete_blocks_without_padding(tp_size):
    manifest = _kimi_manifest()
    for layer in range(1, 93):
        covered: list[int] = []
        for rank in range(tp_size):
            extent = plan_exl3_extent(manifest, layer, tp_size, rank)
            first, count = extent.first_slot, extent.slot_count
            assert count > 0 and count % 4 == first % 4 == 0
            assert first // 48 == (first + count - 1) // 48
            assert extent.intermediate_size == 32 * count
            covered.extend(range(first, first + count))
        assert sorted(covered) == list(range(96))


def test_exl3_tp10_balances_resident_expert_payload():
    manifest = _kimi_manifest()
    totals = [
        sum(
            plan_exl3_extent(manifest, layer, 10, rank).slot_count
            for layer in range(1, 93)
        )
        for rank in range(10)
    ]
    assert sorted(totals) == [880, 880, 884, 884, 884, 884, 884, 884, 884, 884]


@pytest.mark.parametrize(
    "layer,tp,rank", [(0, 10, 0), (93, 10, 0), (1, 1, 0), (1, 25, 0), (1, 10, 10)]
)
def test_exl3_rejects_invalid_layer_or_rank_identity(layer, tp, rank):
    with pytest.raises(ValueError):
        plan_exl3_extent(_kimi_manifest(), layer, tp, rank)


@pytest.mark.parametrize(
    "layout", [{"barriers": ()}, {"barriers": (32,)}, {"alignment": 5}]
)
def test_exl3_rejects_unplanned_container_layouts(layout):
    with pytest.raises(NotImplementedError):
        plan_exl3_extent(_kimi_manifest(**layout), 1, 9, 0)


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


def test_exl3_defers_expert_storage_to_extent_preparation(moe_config):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    method = Exl3MoEMethod(moe_config)
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.17.mlp.experts"
    method.create_weights(layer, 896, 3584, 384, torch.bfloat16)
    assert method.layer_index == 17
    assert set(dict(layer.named_parameters())) == {"w13_weight", "w2_weight"}
    assert all(weight.numel() == 0 for weight in layer.parameters())
    config = method.get_fused_moe_quant_config(layer)
    assert config.weight_quant_dtype == "exl3"
    assert config.quant_dtype is None


def test_exl3_installs_common_trellis_package_and_checks_local_width(
    moe_config, monkeypatch
):
    from b12x.moe import fused_moe

    from vllm.model_executor.layers.fused_moe import b12x
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    layer = torch.nn.Module()
    layer.activation = moe_config.activation
    layer.apply_router_weight_on_input = False
    quant = Exl3MoEMethod(moe_config).get_fused_moe_quant_config(layer)
    backend = b12x.B12xExperts(moe_config, quant)
    # Weight preparation is GPU-only; exercise the installation boundary on CPU.
    prepared = Mock(spec=fused_moe.PreparedExperts)
    prepared.plan = SimpleNamespace(
        source=Mock(spec=fused_moe.TrellisSource),
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ",
            io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
    )
    prepared.num_experts = moe_config.num_experts
    prepared.hidden_size = moe_config.hidden_dim
    prepared.intermediate_size = moe_config.intermediate_size_per_partition
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *a, **k: None
    )
    backend.install_prepared_experts(layer, prepared)
    assert layer._b12x_prepared_experts is prepared
    assert backend._prepared_experts is prepared
    prepared.intermediate_size -= 128
    with pytest.raises(ValueError, match="geometry"):
        backend.install_prepared_experts(layer, prepared)


@pytest.mark.parametrize("tp_size", [9, 10, 16])
def test_exl3_moe_uses_manifest_width_without_changing_shared_experts(
    moe_config, monkeypatch, tp_size
):
    from vllm.config import ParallelConfig
    from vllm.model_executor.layers.quantization.utils import exl3
    from vllm.models.kimi_k3.nvidia import model as kimi
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    config = KimiLinearConfig(
        hidden_size=3584,
        moe_intermediate_size=3072,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=1,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    manifest = _kimi_manifest()
    monkeypatch.setattr(exl3, "load_exl3_manifest", lambda root: manifest)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(kimi, "GateLinear", lambda **kwargs: torch.nn.Module())
    shared_widths = []

    def shared_mlp(**kwargs):
        shared_widths.append(kwargs["intermediate_size"])
        return torch.nn.Module()

    def experts(**kwargs):
        layer = torch.nn.Module()
        layer.moe_config = replace(
            moe_config,
            intermediate_size=kwargs["intermediate_size"],
            moe_parallel_config=replace(
                moe_config.moe_parallel_config, tp_size=tp_size
            ),
        )
        return layer

    monkeypatch.setattr(kimi, "KimiMLP", shared_mlp)
    monkeypatch.setattr(kimi, "FusedMoEFactory", experts)
    runtime = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        parallel_config=ParallelConfig(),
        model_config=SimpleNamespace(model="checkpoint", quantization="exl3"),
    )
    quant = SimpleNamespace(get_name=lambda: "exl3")
    for rank in range(tp_size):
        monkeypatch.setattr(
            kimi, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        for layer in (1, 2):
            moe = kimi.KimiMoE(config, runtime, quant, layer_idx=layer)
            extent = plan_exl3_extent(manifest, layer, tp_size, rank)
            assert (
                moe.experts.moe_config.intermediate_size_per_partition
                == extent.intermediate_size
            )
    assert set(shared_widths) == {3072}
    assert config.moe_intermediate_size == 3072


@pytest.mark.parametrize(
    "fields",
    [
        {"in_dtype": torch.float16},
        {"activation_situ_beta": 1.0},
        {"activation_situ_linear_beta": 0.0},
        {"has_bias": True},
    ],
)
def test_exl3_rejects_mismatched_expert_math(moe_config, fields):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    with pytest.raises(ValueError, match="BF16 SiTU"):
        Exl3MoEMethod(replace(moe_config, **fields))


def test_exl3_shared_mlp_loads_tp9_tail_without_changing_checkpoint_width(
    monkeypatch, default_vllm_config
):
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config
    from vllm.models.kimi_k3.nvidia import model as kimi

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=9, rank_in_group=8)
    )
    quant = Exl3Config(["gate_up_proj", "down_proj"])
    model = kimi.KimiLinearModel.__new__(kimi.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(linear_attn_config={}, is_moe=False)
    model.mlp = kimi.KimiMLP(32, 6144, "silu", quant, reduce_results=False)
    gate = torch.randn(6144, 32)
    up = torch.randn_like(gate)
    down = torch.randn(32, 6144)
    model.load_weights(
        [
            (f"mlp.{name}.weight", weight)
            for name, weight in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down),
            )
        ]
    )
    local = model.mlp.down_proj.weight.shape[1]
    assert local == 704
    start, valid = 8 * local, 6144 - 8 * local
    for shard, weight in enumerate((gate, up)):
        actual = model.mlp.gate_up_proj.weight[shard * local : (shard + 1) * local]
        torch.testing.assert_close(actual[:valid], weight[start:])
        assert torch.count_nonzero(actual[valid:]) == 0
    torch.testing.assert_close(model.mlp.down_proj.weight[:, :valid], down[:, start:])
    assert torch.count_nonzero(model.mlp.down_proj.weight[:, valid:]) == 0


@pytest.mark.parametrize(
    "fields", [{"use_ep": True, "ep_size": 2}, {"dp_size": 2}, {"enable_eplb": True}]
)
def test_exl3_rejects_non_tp_expert_partitioning(moe_config, fields):
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

    parallel = replace(moe_config.moe_parallel_config, **fields)
    with pytest.raises(NotImplementedError, match="without EP or DP"):
        Exl3MoEMethod(replace(moe_config, moe_parallel_config=parallel))


@pytest.fixture
def checkpoint_quant_config():
    return {
        "quant_method": "exl3",
        "dense_format": "mxfp8",
        "ignored_layers": ["kv_b_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj"],
        "exl3": {"manifest": "exl3-manifest.json"},
    }


def test_exl3_detection_preserves_explicit_quantization(checkpoint_quant_config):
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    for selected in (None, "exl3"):
        assert (
            Exl3Config.override_quantization_method(checkpoint_quant_config, selected)
            == "exl3"
        )
    assert (
        Exl3Config.override_quantization_method(checkpoint_quant_config, "mxfp4")
        is None
    )


@pytest.mark.parametrize("selected", [None, "exl3", "mxfp4"])
def test_model_config_detects_exl3_without_overriding_explicit_method(
    checkpoint_quant_config, selected
):
    from transformers import PreTrainedConfig

    from vllm.config import ModelConfig

    config = SimpleNamespace(
        quantization=selected,
        model_arch_config=SimpleNamespace(quantization_config=checkpoint_quant_config),
        hf_config=PreTrainedConfig(),
    )
    if selected == "mxfp4":
        with pytest.raises(ValueError, match="does not match"):
            ModelConfig._verify_quantization(config)
    else:
        ModelConfig._verify_quantization(config)
        assert config.quantization == "exl3"


def test_exl3_config_preserves_serialized_projection_formats(
    checkpoint_quant_config,
):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
    from vllm.models.kimi_k3.nvidia.kda import (
        use_split_mixed_precision_input_projection,
    )
    from vllm.models.kimi_k3.nvidia.model import KimiLinearModel

    cls = get_quantization_config("exl3")
    assert cls.override_quantization_method(checkpoint_quant_config, None) == "exl3"
    assert cls.override_quantization_method({"quant_method": "modelopt"}, None) is None
    config = cls.from_config(checkpoint_quant_config)
    assert isinstance(config, Exl3Config)
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


@pytest.mark.parametrize("tp_size", [9, 10, 16])
@pytest.mark.parametrize("channels", [1, 2])
def test_exl3_kda_state_loader_zero_fills_only_absent_heads(
    monkeypatch, tp_size, channels
):
    from vllm.models.kimi_k3.nvidia import kda
    from vllm.models.kimi_k3.nvidia.tp_projection import (
        enable_kimi_projection_tail_padding,
    )

    monkeypatch.setattr(kda, "get_tensor_model_parallel_rank", lambda: tp_size - 1)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.full((2 * channels,), float("nan")))
    checkpoint = torch.arange((2 * tp_size - 1) * channels, dtype=torch.float32)
    loader = kda.a_log_weight_loader(0)
    with pytest.raises(RuntimeError):
        loader(layer.weight, checkpoint)
    enable_kimi_projection_tail_padding(layer)
    loader(layer.weight, checkpoint)
    expected = torch.cat((checkpoint[-channels:], torch.zeros(channels)))
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [9, 10, 16])
def test_exl3_kda_convolution_padding_updates_decode_storage(tp_size):
    from vllm.models.kimi_k3.nvidia.kda import _make_decode_conv1d_weight_loader

    parameter = torch.nn.Parameter(torch.full((12, 1, 3), float("nan")))
    parameter.allow_tp_padding = True
    decode = torch.full((3, 3, 4), float("nan"))
    checkpoint = torch.arange((tp_size * 4 - 2) * 3).reshape(-1, 3).float()
    loader = _make_decode_conv1d_weight_loader(
        [tp_size * 4] * 3, tp_size, tp_size - 1, decode
    )
    expected = torch.cat((checkpoint[-2:], torch.zeros(2, 3)))
    for shard in range(3):
        loader(parameter, checkpoint + shard, shard)
        shard_expected = expected.clone()
        shard_expected[:2] += shard
        torch.testing.assert_close(
            parameter[shard * 4 : (shard + 1) * 4, 0],
            shard_expected,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(decode[shard], shard_expected.T, rtol=0, atol=0)


def test_exl3_split_kda_projection_keeps_gate_and_beta_order():
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    qkv = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    gfab = torch.arange(24, dtype=torch.float32).reshape(6, 4) + 1
    fb = torch.tensor([[2.0], [3.0]])
    layer = SimpleNamespace(
        local_projection_size=2,
        head_dim=1,
        local_num_heads=1,
        in_proj_padding=2,
        _split_projection_overlap_max_tokens=0,
        _projection_aux_stream=None,
        _projection_events=None,
        in_proj_qkv=lambda value: (value @ qkv.T, None),
        in_proj_gfab=lambda value: (value @ gfab.T, None),
        f_b_proj=lambda value: (value @ fb.T, None),
    )
    actual = KimiK3DeltaAttention._project_split_input(layer, x)
    expected = (x @ qkv.T, x @ gfab[:2].T, (x @ gfab[2:3].T) @ fb.T, x @ gfab[3:4].T)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "original_heads,checkpoint_heads,padding_value",
    [(70, 69, 0), (70, 70, 0), (70, 72, 0), (96, 128, 0), (96, 128, 1)],
)
@pytest.mark.parametrize("legacy_shape", [False, True])
def test_kimi_weight_loading_checks_original_heads_before_tail_padding(
    monkeypatch, original_heads, checkpoint_heads, padding_value, legacy_shape
):
    from vllm.models.kimi_k3.nvidia import kda
    from vllm.models.kimi_k3.nvidia.model import KimiLinearModel

    model = KimiLinearModel.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(linear_attn_config={}, is_moe=False)
    model.attention = torch.nn.Module()
    model.attention._tp_checkpoint_dimensions = {"A_log": (None, original_heads)}
    local_heads = (original_heads + 8) // 9
    parameter = torch.nn.Parameter(torch.full((local_heads,), -1.0))
    parameter.allow_tp_padding = True
    parameter.weight_loader = kda.a_log_weight_loader(0)
    model.attention.register_parameter("A_log", parameter)
    monkeypatch.setattr(kda, "get_tensor_model_parallel_rank", lambda: 8)
    checkpoint = torch.arange(checkpoint_heads, dtype=torch.float32)
    if original_heads == 96:
        checkpoint[96:] = padding_value
    if legacy_shape:
        checkpoint = checkpoint.view(1, 1, -1, 1)
    weights = [("attention.A_log", checkpoint)]
    valid = checkpoint_heads == original_heads or (
        original_heads == 96 and padding_value == 0 and not legacy_shape
    )
    if not valid:
        with pytest.raises(ValueError, match="original head count"):
            model.load_weights(weights)
        assert torch.all(parameter == -1)
    else:
        assert model.load_weights(weights) == {"attention.A_log"}
        expected = torch.zeros(local_heads)
        values = torch.arange(8 * local_heads, original_heads)
        expected[: len(values)] = values
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("suffix", ["weight", "weight_scale"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_kimi_padded_projection_rejects_truncated_weights_and_scales(
    axis, suffix, delta
):
    from vllm.models.kimi_k3.nvidia.tp_projection import (
        projection_checkpoint_dimensions,
        validate_checkpoint_tensor,
    )

    dimensions = projection_checkpoint_dimensions({"projection": (axis, 70 * 128)})
    name = f"projection.{suffix}"
    size = 70 * 128 // (32 if suffix == "weight_scale" and axis == 1 else 1)
    shape = [2, 2]
    shape[axis] = size + delta
    value = torch.empty(shape)
    if delta:
        with pytest.raises(ValueError, match="original head count"):
            validate_checkpoint_tensor(name, value, dimensions)
    else:
        validate_checkpoint_tensor(name, value, dimensions)


@pytest.mark.parametrize("defect", ["manifest", "dense", "gate", "qkv"])
def test_exl3_config_rejects_unsupported_wire_formats(checkpoint_quant_config, defect):
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    if defect == "manifest":
        checkpoint_quant_config["exl3"]["manifest"] = "other-manifest.json"
    elif defect == "dense":
        checkpoint_quant_config["dense_format"] = "bf16"
    elif defect == "gate":
        checkpoint_quant_config["ignored_layers"].remove("g_proj")
    else:
        checkpoint_quant_config["ignored_layers"].append("q_proj")
    with pytest.raises(ValueError):
        Exl3Config.from_config(checkpoint_quant_config)


@pytest.mark.parametrize("tp_size", [8, 9, 10, 12, 16])
def test_exl3_head_padding_preserves_checkpoint_geometry(tp_size):
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    text = SimpleNamespace(
        num_attention_heads=96,
        linear_attn_config={"num_heads": 96, "head_dim": 128},
        moe_intermediate_size=3072,
    )
    model = SimpleNamespace(
        quantization="exl3",
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


def test_exl3_padding_does_not_mutate_official_mxfp4_config():
    from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

    model = SimpleNamespace(quantization="mxfp4")
    KimiK3ForConditionalGenerationConfig.update_model_config_for_parallelism(
        model, SimpleNamespace(tensor_parallel_size=10)
    )
    assert vars(model) == {"quantization": "mxfp4"}


@pytest.mark.parametrize("backend", ["nvidia", "amd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_latent_runner_preserves_routed_output_dtype(backend, dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
    from vllm.models.kimi_k3.nvidia import latent_moe_runner

    cls = (
        latent_moe_runner.LatentMoERunner
        if backend == "nvidia"
        else ROCmLatentMoERunner
    )
    runner = SimpleNamespace(
        layer_name="latent_dtype_test",
        _quant_method=SimpleNamespace(has_unpadded_output=False, output_dtype=dtype),
        moe_config=SimpleNamespace(should_defer_moe_finalize=lambda _: False),
        _maybe_pad_hidden_states=lambda shared, routed: (routed, None, None),
        _forward_entry=torch.ops.vllm.moe_forward_shared,
        _select_tail_tier=lambda *_: latent_moe_runner.LatentTailTier.COLUMN_PARALLEL,
        _shard_up_proj_tail=lambda routed, shared, _: routed,
        _maybe_add_zero_expert_output=lambda result: result,
    )
    runner._encode_layer_name = lambda: MoERunner._encode_layer_name(runner)
    with FakeTensorMode():
        hidden = torch.empty(2, 8, dtype=torch.bfloat16)
        shared = torch.empty(2, 16, dtype=torch.bfloat16)
        result = cls._fused_forward(runner, hidden, torch.empty(2, 4), None, shared)
    assert result.shape == hidden.shape
    assert result.dtype == dtype


@pytest.mark.parametrize("tp_size", [8, 9])
def test_sharded_latent_tail_covers_every_output_column(monkeypatch, tp_size):
    from vllm.models.kimi_k3.nvidia import latent_moe_runner

    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(4096, 8, generator=generator)
    partials = torch.randn(tp_size, 2, 8, generator=generator)
    shared = torch.randn(tp_size, 2, 4096, generator=generator)
    reduced_latent = partials.sum(0)
    monkeypatch.setattr(
        latent_moe_runner, "tensor_model_parallel_all_reduce", lambda _: reduced_latent
    )
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=tp_size),
        routed_output_transform=SimpleNamespace(
            norm=None, up_proj=SimpleNamespace(weight=weight)
        ),
        _maybe_reduce_final_output=lambda output, *args, **kwargs: output,
    )
    outputs = []
    for rank in range(tp_size):
        monkeypatch.setattr(
            latent_moe_runner, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        outputs.append(
            latent_moe_runner.LatentMoERunner._shard_up_proj_tail(
                runner, partials[rank], shared[rank].clone(), None
            )
        )
    torch.testing.assert_close(
        torch.stack(outputs).sum(0),
        torch.nn.functional.linear(reduced_latent, weight) + shared.sum(0),
    )
