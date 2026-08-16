# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
import vllm.model_executor.layers.quantization.online.fp8 as fp8_module
import vllm.model_executor.parameter as parameter_module
from vllm.config import CompilationMode
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3LinearMethod,
    Exl3MoEMethod,
    Exl3MoEParameter,
    Exl3OnlineLinearMethod,
    Exl3Parameter,
)
from vllm.model_executor.layers.quantization.exl3_online_cache import (
    Exl3OnlineCacheResult,
)
from vllm.model_executor.models import glm4_moe


def _rank_sliced_metadata(**overrides):
    metadata = {
        "format": "exl3-trellis",
        "bits": 3.0,
        "codebook": "mcg",
        "experts_per_layer": 256,
        "moe_layers": [3, 77],
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        ),
        "tp": 4,
    }
    metadata.update(overrides)
    return metadata


def _set_online_overlay(monkeypatch, args: QuantizationConfigArgs) -> object:
    current = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=args,
            enforce_eager=True,
        )
    )
    sentinel = object()
    monkeypatch.setattr(exl3_module, "get_current_vllm_config_or_none", lambda: current)
    monkeypatch.setattr(exl3_module, "Mxfp8OnlineLinearMethod", lambda: sentinel)
    return sentinel


def _mock_linear() -> Mock:
    return Mock(spec=LinearBase)


def test_exl3_online_overlay_quantizes_only_bf16_dense_and_shared(monkeypatch):
    config = Exl3Config(
        tensor_storage={"model.layers.3.self_attn.q_b_proj": {"quant_format": "exl3"}}
    )
    sentinel = _set_online_overlay(
        monkeypatch,
        QuantizationConfigArgs(linear="mxfp8", shared_experts="mxfp8"),
    )

    serialized = config.get_quant_method(
        _mock_linear(), "model.layers.3.self_attn.q_b_proj"
    )
    dense_bf16 = config.get_quant_method(
        _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
    )
    shared_bf16 = config.get_quant_method(
        _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
    )

    assert isinstance(serialized, Exl3LinearMethod)
    assert dense_bf16 is sentinel
    assert shared_bf16 is sentinel


def test_exl3_linear_overlay_does_not_select_shared_experts(monkeypatch):
    config = Exl3Config()
    sentinel = _set_online_overlay(monkeypatch, QuantizationConfigArgs(linear="mxfp8"))

    dense = config.get_quant_method(
        _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
    )
    shared = config.get_quant_method(
        _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
    )

    assert dense is sentinel
    assert isinstance(shared, UnquantizedLinearMethod)


def test_exl3_online_overlay_honors_unfused_ignore_names(monkeypatch):
    config = Exl3Config()
    config.packed_modules_mapping = {
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    }
    sentinel = _set_online_overlay(
        monkeypatch,
        QuantizationConfigArgs(
            linear="mxfp8",
            ignore=["re:.*\\.q_a_proj$", "re:.*kv_a_proj_with_mqa"],
        ),
    )

    ignored = config.get_quant_method(
        _mock_linear(), "model.layers.3.self_attn.fused_qkv_a_proj"
    )
    kept = config.get_quant_method(_mock_linear(), "model.layers.3.self_attn.kv_b_proj")

    assert isinstance(ignored, UnquantizedLinearMethod)
    assert kept is sentinel


def test_exl3_online_overlay_rejects_split_packed_quantization(monkeypatch):
    config = Exl3Config()
    config.packed_modules_mapping = {
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    }
    _set_online_overlay(
        monkeypatch,
        QuantizationConfigArgs(linear="mxfp8", ignore=["re:.*\\.q_a_proj$"]),
    )

    with pytest.raises(ValueError, match="different quantization schemes"):
        config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.fused_qkv_a_proj"
        )


def test_exl3_online_overlay_rejects_mixed_packed_storage(monkeypatch):
    config = Exl3Config(
        tensor_storage={"model.layers.3.self_attn.q_a_proj": {"quant_format": "exl3"}}
    )
    config.packed_modules_mapping = {
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    }
    _set_online_overlay(monkeypatch, QuantizationConfigArgs(linear="mxfp8"))

    with pytest.raises(ValueError, match="mixes EXL3 and BF16 source shards"):
        config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.fused_qkv_a_proj"
        )


def test_exl3_online_overlay_never_quantizes_bf16_lm_head(monkeypatch):
    config = Exl3Config()
    _set_online_overlay(monkeypatch, QuantizationConfigArgs(linear="mxfp8"))
    lm_head = type("ParallelLMHead", (torch.nn.Module,), {})()

    method = config.get_quant_method(lm_head, "lm_head")

    assert isinstance(method, UnquantizedLinearMethod)


def test_exl3_online_overlay_preserves_rank_sliced_routed_experts(monkeypatch):
    config = Exl3Config()
    config.rank_sliced_metadata = _rank_sliced_metadata()
    _set_online_overlay(
        monkeypatch,
        QuantizationConfigArgs(linear="mxfp8", shared_experts="mxfp8"),
    )
    routed = Mock(spec=RoutedExperts)
    routed.moe_config = object()

    method = config.get_quant_method(routed, "model.layers.3.mlp.experts")

    assert isinstance(method, Exl3MoEMethod)


def test_exl3_online_trellis_selects_cached_k6_method(monkeypatch):
    config = Exl3Config()
    config._online_model_identity = "model"  # noqa: SLF001
    config._online_encoder_identity = "encoder"  # noqa: SLF001
    _set_online_overlay(monkeypatch, QuantizationConfigArgs(linear="mxfp8"))
    monkeypatch.setattr(
        fp8_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    monkeypatch.setenv("VLLM_EXL3_ONLINE_TRELLIS_BITS", "6")

    method = config.get_quant_method(_mock_linear(), "model.layers.3.self_attn.o_proj")

    assert isinstance(method, Exl3OnlineLinearMethod)
    assert method.bits == 6
    assert method.model_identity == "model"
    assert method.encoder_identity == "encoder"


def test_exl3_online_trellis_logs_one_stable_summary(monkeypatch):
    config = Exl3Config()
    config._online_model_identity = "model"  # noqa: SLF001
    config._online_encoder_identity = "encoder"  # noqa: SLF001
    _set_online_overlay(monkeypatch, QuantizationConfigArgs(linear="mxfp8"))
    monkeypatch.setattr(
        fp8_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    monkeypatch.setenv("VLLM_EXL3_ONLINE_TRELLIS_BITS", "6")
    warning_keys = set()
    monkeypatch.setattr(
        exl3_module.logger,
        "warning_once",
        lambda message, *args: warning_keys.add((message, args)),
    )

    config.get_quant_method(_mock_linear(), "model.layers.3.self_attn.o_proj")
    config.get_quant_method(_mock_linear(), "model.layers.4.self_attn.o_proj")

    assert len(warning_keys) == 1


def test_online_trellis_cache_off_does_not_require_hub_commit(tmp_path, monkeypatch):
    config = Exl3Config()
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "off")
    monkeypatch.setenv("VLLM_EXL3_ENCODER_SOURCE", str(tmp_path))

    config._configure_online_cache_identity(  # noqa: SLF001
        "org/model", hf_config=None, revision=None
    )

    assert config._online_model_identity == "cache-disabled"  # noqa: SLF001
    assert config._online_encoder_identity == "cache-disabled"  # noqa: SLF001


def test_online_trellis_encoder_requires_quantize_entrypoint(tmp_path, monkeypatch):
    encoder = tmp_path / "encoder"
    quantizer = encoder / "modules" / "quant" / "exl3_lib"
    quantizer.mkdir(parents=True)
    (quantizer / "quantize.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("VLLM_EXL3_ENCODER_SOURCE", str(encoder))
    monkeypatch.setattr(exl3_module, "_EXL3_ONLINE_QUANTIZER", None)
    monkeypatch.setattr(
        exl3_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    try:
        with pytest.raises(RuntimeError, match="has no quantize_exl3"):
            exl3_module._load_exl3_online_quantizer()
    finally:
        for name in tuple(exl3_module.sys.modules):
            if name == "_vllm_exl3_encoder" or name.startswith("_vllm_exl3_encoder."):
                exl3_module.sys.modules.pop(name, None)


def test_exl3_online_trellis_cache_hit_skips_encoder(monkeypatch):
    monkeypatch.setattr(
        fp8_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    method = Exl3OnlineLinearMethod(
        bits=6,
        prefix="model.layers.3.self_attn.o_proj",
        model_identity="model",
        encoder_identity="encoder",
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros((256, 128), dtype=torch.float16)),
    )
    layer.exl3_online_input_size = 128
    layer.exl3_online_output_size = 256
    tensors = {
        "trellis": torch.zeros((8, 16, 96), dtype=torch.int16),
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(256, dtype=torch.float16),
    }
    captured = {}

    def cache_hit(key, *, device, quantize):
        del quantize
        captured["key"] = key
        captured["device"] = device
        return Exl3OnlineCacheResult(tensors, 0.25, True, None)

    monkeypatch.setattr(exl3_module, "load_or_quantize", cache_hit)
    monkeypatch.setattr(
        exl3_module,
        "_load_exl3_online_quantizer",
        lambda: pytest.fail("cache hit must not import the encoder"),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(method, "_warm_decode_shapes", lambda layer: None)

    method.process_weights_after_loading(layer)

    assert captured["key"].tp_world_size == 4
    assert captured["key"].tp_rank == 2
    assert captured["device"] == torch.device("cpu")
    assert layer.weight.numel() == 0
    assert layer.exl3_online_trellis_weight is tensors["trellis"]


def test_rank_sliced_checkpoint_selects_exl3_override():
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())

    assert get_quantization_config("exl3") is Exl3Config
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, None, hf_config
        )
        == "exl3"
    )
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, "fp8", hf_config
        )
        is None
    )


def test_glm_model_retains_quant_config_for_weight_loading(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(glm4_moe, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        glm4_moe,
        "make_layers",
        lambda *args, **kwargs: (0, 0, torch.nn.ModuleList()),
    )
    monkeypatch.setattr(
        glm4_moe,
        "make_empty_intermediate_tensors_factory",
        lambda *args, **kwargs: object(),
    )
    quant_config = object()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=1,
                hidden_size=8,
                num_hidden_layers=0,
            )
        ),
        cache_config=object(),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(enable_eplb=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    model = glm4_moe.Glm4MoeModel(vllm_config=vllm_config)

    assert model.quant_config is quant_config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"codebook": "mul1"}, "MCG codebook"),
        ({"moe_layers": [77, 3]}, "moe_layers"),
        ({"tensor_schema": "unsupported"}, "tensor schema"),
        ({"rotation_layout": "implicit_magic"}, "rotation_layout"),
        ({"rotation_layout": "shared_h_v1"}, "shared_h_tensor_schema"),
    ],
)
def test_rank_sliced_metadata_fails_closed(overrides, message):
    config = Exl3Config()
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata(**overrides))

    with pytest.raises(ValueError, match=message):
        config.maybe_update_config("unused", hf_config)


def test_rank_sliced_metadata_admits_only_declared_moe_layers():
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )

    assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
    assert config._moe_prefix_is_exl3("model.layers.77.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.2.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.78.mlp.experts")
    assert (
        config.codebook_for_prefix("model.layers.10.mlp.experts.0.gate_proj") == "mcg"
    )


def test_rank_sliced_shared_h_metadata_is_explicit_and_legacy_defaults_unchanged():
    legacy = Exl3Config()
    legacy.maybe_update_config(
        "unused", SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())
    )
    shared = Exl3Config()
    shared.maybe_update_config(
        "unused",
        SimpleNamespace(
            hybrid_tr3_tail=_rank_sliced_metadata(
                rotation_layout="shared_h_v1",
                shared_h_tensor_schema=(
                    "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
                ),
            )
        ),
    )

    assert legacy.rank_sliced_rotation_layout == "per_expert_v1"
    assert shared.rank_sliced_rotation_layout == "shared_h_v1"


def test_mixed_rank_sliced_metadata_hydrates_per_layer_bitrates(monkeypatch):
    metadata = _rank_sliced_metadata(
        bits="mixed",
        bits_per_expert="tier_bitmap.json:k",
        k_values=[3, 4],
        experts_per_layer=4,
        moe_layers=[77, 78],
    )
    payload = {
        "77": {"k": [3, 4, 3, 4]},
        # The checkpoint's MTP overlay records a uniform K3 tail this way.
        "78": {"tail_tr3": [0, 1, 2, 3]},
    }
    monkeypatch.setattr(
        exl3_module,
        "get_hf_file_to_dict",
        lambda filename, model_name, revision=None: payload,
    )
    config = Exl3Config()

    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=metadata, _commit_hash="revision"),
    )

    assert config.bits is None
    assert config.rank_sliced_k_values == (3, 4)
    assert config.rank_sliced_layer_bitrates("model.layers.77.mlp.experts") == (
        3,
        4,
        3,
        4,
    )
    assert config.rank_sliced_layer_bitrates("model.layers.78.mlp.experts") == (
        3,
        3,
        3,
        3,
    )


def test_rank_sliced_weight_name_keeps_only_local_tp_rank(monkeypatch):
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    prefix = "model.layers.3.mlp.experts.17.gate_proj"

    assert (
        config.normalize_rank_sliced_weight_name(f"{prefix}.rank2.trellis")
        == f"{prefix}.trellis"
    )
    assert config.normalize_rank_sliced_weight_name(f"{prefix}.rank1.trellis") is None
    assert (
        config.normalize_rank_sliced_weight_name("model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_glm_model_normalizes_rank_sliced_weights_before_auto_loading(monkeypatch):
    observed = []

    class RecordingLoader:
        def __init__(self, model):
            assert model is glm_model

        def load_weights(self, weights, *, mapper):
            observed.extend(weights)
            assert mapper is glm4_moe.Glm4MoeModel.hf_to_vllm_mapper
            return {name for name, _ in observed}

    def normalize(name: str) -> str | None:
        if ".rank1." in name:
            return None
        return name.replace(".rank0.", ".")

    monkeypatch.setattr(glm4_moe, "AutoWeightsLoader", RecordingLoader)
    monkeypatch.setattr(
        glm4_moe,
        "skip_spec_layers",
        lambda weights, config: weights,
    )
    monkeypatch.setattr(
        glm4_moe,
        "maybe_fuse_shared_experts",
        lambda weights, **kwargs: weights,
    )
    glm_model = object.__new__(glm4_moe.Glm4MoeModel)
    torch.nn.Module.__init__(glm_model)
    glm_model.quant_config = SimpleNamespace(
        normalize_rank_sliced_weight_name=normalize
    )
    glm_model.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
    local = torch.tensor(1)
    remote = torch.tensor(2)
    ordinary = torch.tensor(3)

    loaded = glm_model.load_weights(
        [
            ("layers.3.mlp.experts.0.gate_proj.rank0.trellis", local),
            ("layers.3.mlp.experts.0.gate_proj.rank1.trellis", remote),
            ("embed_tokens.weight", ordinary),
        ]
    )

    assert observed == [
        ("layers.3.mlp.experts.0.gate_proj.trellis", local),
        ("embed_tokens.weight", ordinary),
    ]
    assert loaded == {
        "layers.3.mlp.experts.0.gate_proj.trellis",
        "embed_tokens.weight",
    }


def test_rank_sliced_shared_h_names_map_once_and_fail_closed(monkeypatch):
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(
            hybrid_tr3_tail=_rank_sliced_metadata(
                rotation_layout="shared_h_v1",
                shared_h_tensor_schema=(
                    "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
                ),
            )
        ),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    prefix = "model.layers.3.mlp.experts"

    assert (
        config.normalize_rank_sliced_weight_name(
            f"{prefix}.shared_h.gate_proj.rank2.suh"
        )
        == f"{prefix}.0.gate_proj.suh"
    )
    assert (
        config.normalize_rank_sliced_weight_name(
            f"{prefix}.shared_h.down_proj.rank1.svh"
        )
        is None
    )
    with pytest.raises(ValueError, match="requires suh"):
        config.normalize_rank_sliced_weight_name(
            f"{prefix}.shared_h.gate_proj.rank2.svh"
        )
    with pytest.raises(ValueError, match="must store H-side rotations"):
        config.normalize_rank_sliced_weight_name(f"{prefix}.7.down_proj.rank2.svh")

    legacy = Exl3Config()
    legacy.maybe_update_config(
        "unused", SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())
    )
    with pytest.raises(ValueError, match="does not declare"):
        legacy.normalize_rank_sliced_weight_name(f"{prefix}.shared_h.up_proj.rank2.suh")


def test_rank_sliced_parameter_preallocates_projection_major_slab(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=3,
        shard_ids=("w1", "w3"),
        preallocate=True,
    )
    w1 = torch.arange(8, dtype=torch.int16).reshape(2, 2, 2)
    w3 = w1 + 20

    param.load_exl3_weight(w1, expert_id=1, shard_id="w1")
    param.load_exl3_weight(w3, expert_id=2, shard_id="w3")

    assert param.exl3_backing is not None
    assert tuple(param.exl3_backing.shape) == (2, 3, 2, 2, 2)
    assert (
        param.exl3_tensors[(1, "w1")].data_ptr() == param.exl3_backing[0, 1].data_ptr()
    )
    assert (
        param.exl3_tensors[(2, "w3")].data_ptr() == param.exl3_backing[1, 2].data_ptr()
    )
    torch.testing.assert_close(param.exl3_tensors[(1, "w1")], w1)
    torch.testing.assert_close(param.exl3_tensors[(2, "w3")], w3)


@pytest.mark.parametrize(
    ("shape", "tp_slice", "expected"),
    [
        ((12,), (0, 4, 4, 1), torch.arange(4, 8)),
        (
            (4, 6, 2),
            (1, 32, 32, 16),
            torch.tensor(
                [
                    [[4, 5], [6, 7]],
                    [[16, 17], [18, 19]],
                    [[28, 29], [30, 31]],
                    [[40, 41], [42, 43]],
                ]
            ),
        ),
        ((6, 4, 2), (0, 32, 32, 16), torch.arange(16, 32).reshape(2, 4, 2)),
    ],
)
def test_r7_parameter_streams_owned_tp_slice(monkeypatch, shape, tp_slice, expected):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    source = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int16).reshape(
        shape
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=1,
        shard_ids=("w1",),
        tp_slice=tp_slice,
    )

    param.load_exl3_weight(source, expert_id=0, shard_id="w1")
    loaded = param.exl3_tensors[(0, "w1")]

    torch.testing.assert_close(loaded, expected.to(torch.int16))
    assert loaded.is_contiguous()
    assert loaded.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert loaded.untyped_storage().nbytes() == loaded.numel() * loaded.element_size()


def test_r7_parameter_streaming_tp_slice_fails_closed(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=1,
        shard_ids=("w1",),
        tp_slice=(1, 24, 32, 16),
    )

    with pytest.raises(ValueError, match="not quantum-aligned"):
        param.load_exl3_weight(
            torch.zeros((4, 8, 2), dtype=torch.int16),
            expert_id=0,
            shard_id="w1",
        )


@pytest.mark.parametrize(
    ("parameter_cls", "load_kwargs"),
    [
        (Exl3Parameter, {"shard_id": "w1"}),
        (Exl3MoEParameter, {"expert_id": 0, "shard_id": "w1"}),
    ],
)
def test_exl3_parameters_own_borrowed_instanttensor_storage(
    monkeypatch, parameter_cls, load_kwargs
):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    source = torch.arange(8, dtype=torch.int16)
    source._vllm_instanttensor_borrowed = True
    kwargs = {"weight_loader": lambda *args, **kwargs: None}
    if parameter_cls is Exl3MoEParameter:
        kwargs.update(num_experts=1, shard_ids=("w1",))
    param = parameter_cls(**kwargs)

    param.load_exl3_weight(source, **load_kwargs)
    loaded = next(iter(param.exl3_tensors.values()))
    source.zero_()

    torch.testing.assert_close(loaded, torch.arange(8, dtype=torch.int16))
    assert loaded.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_rank_sliced_broadcast_pointer_table_repeats_one_physical_row():
    slab = torch.ones((1, 128), dtype=torch.float16)

    table = Exl3MoEMethod._pointer_table(slab, num_experts=4)

    assert table.tolist() == [slab.data_ptr()] * 4
    with pytest.raises(RuntimeError, match="per-expert or broadcast"):
        Exl3MoEMethod._pointer_table(
            torch.ones((2, 128), dtype=torch.float16), num_experts=4
        )


def test_rank_sliced_shared_h_create_weights_allocates_one_physical_row(
    monkeypatch,
):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 4
    )
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(
            hybrid_tr3_tail=_rank_sliced_metadata(
                rotation_layout="shared_h_v1",
                shared_h_tensor_schema=(
                    "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
                ),
            )
        ),
    )
    monkeypatch.setattr(
        exl3_module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
            model_config=SimpleNamespace(runner_type="generate"),
        ),
    )
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.3.mlp.experts"
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False, tp_rank=0, tp_size=4),
        has_bias=False,
    )
    method = Exl3MoEMethod(config, moe)

    method.create_weights(
        layer,
        num_experts=256,
        hidden_size=6144,
        intermediate_size_per_partition=512,
        params_dtype=torch.bfloat16,
    )

    assert layer.exl3_shared_h_rotations
    assert layer.w13_suh.exl3_num_experts == 1
    assert layer.w2_svh.exl3_num_experts == 1
    assert layer.w13_svh.exl3_num_experts == 256
    assert layer.w2_suh.exl3_num_experts == 256
    assert layer.w13_trellis.exl3_num_experts == 256
    assert layer.w2_trellis.exl3_num_experts == 256


def test_rank_sliced_weights_use_unified_fused_moe_contract(monkeypatch):
    experts = 2
    hidden = intermediate = 128
    bits = 3
    slabs = {
        "w13_trellis": torch.zeros(
            (2, experts, hidden // 16, intermediate // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w2_trellis": torch.zeros(
            (experts, intermediate // 16, hidden // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w13_suh": torch.ones((2, experts, hidden), dtype=torch.float16),
        "w13_svh": torch.ones((2, experts, intermediate), dtype=torch.float16),
        "w2_suh": torch.ones((experts, intermediate), dtype=torch.float16),
        "w2_svh": torch.ones((experts, hidden), dtype=torch.float16),
    }

    class FakeFusedMoe:
        def __init__(self):
            self.plan_kwargs = None
            self.prepare_kwargs = None

        def plan_weights(self, **kwargs):
            self.plan_kwargs = kwargs
            return SimpleNamespace(source_format=kwargs["source_format"])

        def prepare_weights(self, **kwargs):
            self.prepare_kwargs = kwargs
            return SimpleNamespace(plan=kwargs["plan"])

    api = FakeFusedMoe()
    monkeypatch.setattr(exl3_module, "_load_b12x_fused_moe", lambda: api)
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(bits=float(bits))
    method._rank_sliced_backing = lambda _layer, name: slabs[name]
    marker = torch.tensor(0xCBAC1FED - (1 << 32), dtype=torch.int32)
    layer = SimpleNamespace(
        local_num_experts=experts,
        exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate,
        exl3_params_dtype=torch.float16,
        exl3_layer_bitrates=(bits,) * experts,
        exl3_mixed_bitrate=False,
        activation=MoEActivation.SILU,
        w13_mcg=SimpleNamespace(exl3_tensors={(0, "w1"): marker}),
    )

    method._prepare_rank_sliced_weights(layer)

    assert api.plan_kwargs == {
        "quant_modes": "w4a16",
        "source_format": "exl3_trellis_mcg",
        "activation": "silu",
        "params_dtype": torch.float16,
        "num_experts": experts,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "w13_layout": "w13",
        "trellis_bits": bits,
        "trellis_tile_config": (64, 128, 64, 128),
    }
    assert api.prepare_kwargs is not None
    assert api.prepare_kwargs["plan"] is layer.exl3_trellis_weights.plan
    assert api.prepare_kwargs["params_dtype"] == torch.float16
    assert api.prepare_kwargs["w1_fp4"] is slabs["w13_trellis"]
    assert api.prepare_kwargs["w2_fp4"] is slabs["w2_trellis"]
    assert api.prepare_kwargs["trellis_mcg"] is marker


def test_rank_sliced_weights_pass_shared_h_rows_without_expansion(monkeypatch):
    experts = 3
    hidden = intermediate = 128
    bits = 3
    slabs = {
        "w13_trellis": torch.zeros(
            (2, experts, hidden // 16, intermediate // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w2_trellis": torch.zeros(
            (experts, intermediate // 16, hidden // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w13_suh": torch.ones((2, 1, hidden), dtype=torch.float16),
        "w13_svh": torch.ones((2, experts, intermediate), dtype=torch.float16),
        "w2_suh": torch.ones((experts, intermediate), dtype=torch.float16),
        "w2_svh": torch.ones((1, hidden), dtype=torch.float16),
    }

    class FakeFusedMoe:
        @staticmethod
        def plan_weights(**kwargs):
            return SimpleNamespace(source_format=kwargs["source_format"])

        @staticmethod
        def prepare_weights(**kwargs):
            return SimpleNamespace(**kwargs)

    monkeypatch.setattr(exl3_module, "_load_b12x_fused_moe", lambda: FakeFusedMoe())
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(bits=float(bits))
    method._rank_sliced_backing = lambda _layer, name: slabs[name]
    marker = torch.tensor(0xCBAC1FED - (1 << 32), dtype=torch.int32)
    layer = SimpleNamespace(
        local_num_experts=experts,
        exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate,
        exl3_params_dtype=torch.float16,
        exl3_layer_bitrates=(bits,) * experts,
        exl3_mixed_bitrate=False,
        exl3_shared_h_rotations=True,
        activation=MoEActivation.SILU,
        w13_mcg=SimpleNamespace(exl3_tensors={(0, "w1"): marker}),
    )

    method._prepare_rank_sliced_weights(layer)

    prepared = layer.exl3_trellis_weights
    assert tuple(prepared.gate_suh.shape) == (1, hidden)
    assert tuple(prepared.up_suh.shape) == (1, hidden)
    assert tuple(prepared.down_svh.shape) == (1, hidden)
    assert all(tuple(table.shape) == (experts,) for table in layer.exl3_pointer_tables)


@pytest.mark.parametrize("shared_h", [False, True])
def test_mixed_rank_sliced_weights_are_partitioned_by_declared_bitrate(
    monkeypatch, shared_h
):
    experts = 4
    hidden = intermediate = 128
    bitrates = (3, 4, 3, 4)

    def parameter(shard_ids=(), tensors=None, backing=None):
        return SimpleNamespace(
            exl3_shard_ids=list(shard_ids),
            exl3_tensors=dict(tensors or {}),
            exl3_backing=backing,
        )

    w13_tensors = {}
    w2_tensors = {}
    for expert, bits in enumerate(bitrates):
        for shard in ("w1", "w3"):
            w13_tensors[(expert, shard)] = torch.zeros(
                (hidden // 16, intermediate // 16, 16 * bits),
                dtype=torch.int16,
            )
        w2_tensors[(expert, "w2")] = torch.zeros(
            (intermediate // 16, hidden // 16, 16 * bits),
            dtype=torch.int16,
        )

    slabs = {
        "w13_suh": torch.ones(
            (2, 1 if shared_h else experts, hidden), dtype=torch.float16
        ),
        "w13_svh": torch.ones((2, experts, intermediate), dtype=torch.float16),
        "w2_suh": torch.ones((experts, intermediate), dtype=torch.float16),
        "w2_svh": torch.ones((1 if shared_h else experts, hidden), dtype=torch.float16),
    }
    layer = SimpleNamespace(
        local_num_experts=experts,
        exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate,
        exl3_params_dtype=torch.float16,
        exl3_layer_bitrates=bitrates,
        activation=MoEActivation.SILU,
        layer_name="model.layers.3.mlp.experts",
        w13_trellis=parameter(("w1", "w3"), w13_tensors),
        w2_trellis=parameter(("w2",), w2_tensors),
    )
    for prefix, shards in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        for suffix in ("suh", "svh", "mcg", "mul1"):
            name = f"{prefix}_{suffix}"
            backing = slabs.get(name)
            setattr(layer, name, parameter(shards, backing=backing))

    class FakeMixedApi:
        def __init__(self):
            self.prepared = []

        def prepare_weights(self, **kwargs):
            self.prepared.append(kwargs)
            return SimpleNamespace(**kwargs)

        @staticmethod
        def build_tiered_maps(tier0, tier1, *, device):
            assert tuple(tier0) == (0, 2)
            assert tuple(tier1) == (1, 3)
            return (
                torch.tensor([0, 2, 1, 3], dtype=torch.int32, device=device),
                torch.tensor([0, 1, 256, 257], dtype=torch.int32, device=device),
            )

        @staticmethod
        def combine_trellis_rotations(tier0, tier1):
            return tier0, tier1

    api = FakeMixedApi()
    monkeypatch.setattr(exl3_module, "_load_b12x_mixed_trellis", lambda: api)
    method = object.__new__(Exl3MoEMethod)
    method._rank_sliced_backing = lambda _layer, name: slabs[name]

    method._prepare_mixed_rank_sliced_weights(layer)

    assert [entry["trellis_bits"] for entry in api.prepared] == [3, 3, 4, 4]
    assert [entry["num_experts"] for entry in api.prepared] == [2] * 4
    for entry in api.prepared:
        bits = entry["trellis_bits"]
        assert tuple(entry["w13"].shape) == (
            2,
            2,
            hidden // 16,
            intermediate // 16,
            16 * bits,
        )
        assert tuple(entry["w2"].shape) == (
            2,
            intermediate // 16,
            hidden // 16,
            16 * bits,
        )
    assert [entry["tile_config"] for entry in api.prepared] == [
        (128, 128, 128, 128),
        (128, 128, 128, 128),
        (128, 128, 128, 128),
        (128, 128, 128, 128),
    ]
    assert layer.exl3_mixed_trellis["tier_ids"] == ((0, 2), (1, 3))
    assert layer.exl3_mixed_trellis["tier_bits"] == (3, 4)
    assert len(layer.exl3_mixed_trellis["tiers"]) == 2
    assert len(layer.exl3_mixed_trellis["prefill_tiers"]) == 2
    rotations = layer.exl3_mixed_trellis["rotations"]
    expected_h_rows = 1 if shared_h else experts
    assert rotations.gate_suh.shape[0] == expected_h_rows
    assert rotations.up_suh.shape[0] == expected_h_rows
    assert rotations.down_svh.shape[0] == expected_h_rows
    assert layer.exl3_mixed_trellis["broadcast_suh"] is shared_h
    assert layer.exl3_mixed_trellis["broadcast_svh"] is shared_h
    for entry in api.prepared:
        expected_tier_rows = 1 if shared_h else 2
        assert entry["gate_suh"].shape[0] == expected_tier_rows
        assert entry["up_suh"].shape[0] == expected_tier_rows
        assert entry["down_svh"].shape[0] == expected_tier_rows
        assert (
            entry["gate_suh"].untyped_storage().data_ptr()
            == rotations.gate_suh.untyped_storage().data_ptr()
        )
    assert layer.w13_trellis.exl3_tensors == {}
    assert layer.w2_trellis.exl3_tensors == {}
    assert layer.w13_suh.exl3_backing is None
    assert layer.w2_svh.exl3_backing is None


def test_mixed_trellis_buffer_accounting_ignores_metadata() -> None:
    shared = torch.empty(8, dtype=torch.uint8)
    first = SimpleNamespace(tensor=shared, metadata=None, block_size=8)
    second = SimpleNamespace(alias=shared.view(2, 4), label="prefill")

    assert exl3_module._unique_tensor_storage_bytes(first, second) == 8


def test_prepared_dense_weight_is_owned_by_source_tensor(monkeypatch) -> None:
    class FakeApi:
        def __init__(self):
            self.calls = 0

        def prepare_weight(self, trellis, suh, svh, **kwargs):
            del kwargs
            self.calls += 1
            return SimpleNamespace(trellis=trellis, suh=suh, svh=svh)

    api = FakeApi()
    monkeypatch.setattr(exl3_module, "_load_b12x_trellis_linear", lambda: api)
    trellis = torch.empty((8, 8, 96), dtype=torch.int16)
    suh = torch.empty(128, dtype=torch.float16)
    svh = torch.empty(128, dtype=torch.float16)

    first = exl3_module._b12x_trellis_weight(trellis, suh, svh, torch.float16)
    second = exl3_module._b12x_trellis_weight(trellis, suh, svh, torch.float16)
    assert first is second
    assert api.calls == 1

    source_ref = weakref.ref(trellis)
    del first, second, trellis
    gc.collect()
    assert source_ref() is None


def test_b12x_trellis_scratch_uses_explicit_small_m_contract(monkeypatch) -> None:
    calls = []
    api = SimpleNamespace(
        k6_mcg_small_m_scratch_elements=lambda size_k, size_n: (
            calls.append((size_k, size_n)) or 131072
        )
    )
    monkeypatch.setattr(exl3_module, "_load_b12x_trellis_linear", lambda: api)

    assert exl3_module._b12x_trellis_c_tmp_elements(8, 2048, 4096) == 131072
    assert calls == [(2048, 4096)]


def test_b12x_trellis_scratch_uses_generic_contract_above_small_m(
    monkeypatch,
) -> None:
    api = SimpleNamespace(k6_mcg_small_m_scratch_elements=Mock())
    monkeypatch.setattr(exl3_module, "_load_b12x_trellis_linear", lambda: api)

    assert exl3_module._b12x_trellis_c_tmp_elements(17, 2048, 4096) == 262144
    api.k6_mcg_small_m_scratch_elements.assert_not_called()


def test_b12x_trellis_scratch_without_query_uses_allocation_free_abi(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        exl3_module,
        "_load_b12x_trellis_linear",
        lambda: SimpleNamespace(),
    )

    assert exl3_module._b12x_trellis_c_tmp_elements(8, 2048, 4096) == 1


def test_mixed_trellis_dispatches_decode_and_one_grid_prefill(monkeypatch):
    class FakeMixedApi:
        def __init__(self):
            self.calls = []

        def run_mixed_trellis(self, x, *args):
            launch = args[7]
            self.calls.append((int(x.shape[0]), args[0], args[1], launch))
            return torch.full_like(x, launch.value)

    mixed_api = FakeMixedApi()
    decode_tiers = (object(), object())
    prefill_tiers = (object(), object())
    runtime = {
        "mixed_api": mixed_api,
        "decode": {
            "launch": SimpleNamespace(value=1),
            "buffers": object(),
        },
        "prefill": {
            "launch": SimpleNamespace(value=3),
            "buffers": object(),
        },
        "max_decode_m": 8,
        "max_batched_tokens": 16,
        "prefill_capacity": 8,
    }
    layer = SimpleNamespace(
        exl3_mixed_trellis={
            "tiers": decode_tiers,
            "prefill_tiers": prefill_tiers,
            "global_to_combined": object(),
            "descriptor_map": object(),
            "rotations": object(),
        }
    )
    method = object.__new__(Exl3MoEMethod)
    monkeypatch.setattr(
        method, "_mixed_rank_sliced_runtime", lambda layer, x, ids: runtime
    )
    weights = torch.ones((16, 2), dtype=torch.float32)
    ids = torch.zeros((16, 2), dtype=torch.int64)

    decode = method._apply_mixed_rank_sliced(
        layer, torch.zeros((4, 4)), weights[:4], ids[:4]
    )
    prefill = method._apply_mixed_rank_sliced(layer, torch.zeros((16, 4)), weights, ids)

    torch.testing.assert_close(decode, torch.ones_like(decode))
    torch.testing.assert_close(prefill, torch.full_like(prefill, 3))
    assert [call[0] for call in mixed_api.calls] == [4, 8, 8]
    assert mixed_api.calls[0][1:3] == decode_tiers
    assert all(call[1:3] == prefill_tiers for call in mixed_api.calls[1:])


@pytest.mark.parametrize(
    "tier_signature",
    [
        ((3, 192), (4, 64)),
        ((3, 206), (4, 50)),
        ((3, 148), (4, 108)),
    ],
)
def test_mixed_trellis_prefill_block_policy_qualified_partitions(
    tier_signature,
) -> None:
    common = {
        "configured_block_m": 64,
        "explicit_override": False,
        "hidden_size": 6144,
        "intermediate_size": 512,
        "tier_signature": tier_signature,
        "topk": 8,
        "device_major": 12,
        "prefill_tile_config": (128, 128, 32, 512),
    }

    assert exl3_module._resolve_mixed_trellis_prefill_block_m(**common) == 32
    assert (
        exl3_module._resolve_mixed_trellis_prefill_block_m(
            **{**common, "explicit_override": True}
        )
        == 64
    )


@pytest.mark.parametrize(
    "tier_signature",
    [
        ((3, 171), (4, 86), (5, 1)),
        ((3, 197), (4, 58), (5, 2)),
    ],
)
def test_mixed_trellis_prefill_block_policy_qualifies_native_k5(
    tier_signature,
) -> None:
    assert (
        exl3_module._resolve_mixed_trellis_prefill_block_m(
            configured_block_m=64,
            explicit_override=False,
            hidden_size=6144,
            intermediate_size=512,
            tier_signature=tier_signature,
            topk=8,
            device_major=12,
            prefill_tile_config=(128, 128, 32, 512),
        )
        == 32
    )


def test_r7_native_layer_budget_is_unlimited_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_EXL3_R7_FUSED_LAYERS", raising=False)
    assert exl3_module._r7_fused_layer_budget() is None
    monkeypatch.setenv("VLLM_EXL3_R7_FUSED_LAYERS", "48")
    assert exl3_module._r7_fused_layer_budget() == 48


def test_mixed_trellis_prefill_block_policy_rejects_unqualified_partition() -> None:
    common = {
        "configured_block_m": 64,
        "explicit_override": False,
        "hidden_size": 6144,
        "intermediate_size": 512,
        "tier_signature": ((3, 192), (4, 64)),
        "topk": 8,
        "device_major": 12,
        "prefill_tile_config": (128, 128, 32, 512),
    }
    assert (
        exl3_module._resolve_mixed_trellis_prefill_block_m(
            **{**common, "tier_signature": ((3, 128), (4, 128))}
        )
        == 64
    )
    assert (
        exl3_module._resolve_mixed_trellis_prefill_block_m(
            **{**common, "tier_signature": ((4, 256),)}
        )
        == 64
    )
    assert (
        exl3_module._resolve_mixed_trellis_prefill_block_m(
            **{**common, "device_major": 11}
        )
        == 64
    )


@pytest.mark.parametrize(
    ("hidden", "intermediate", "expected"),
    [
        (6144, 512, (128, 128, 32, 512)),
        (256, 128, (128, 128, 64, 256)),
        (128, 128, (128, 128, 128, 128)),
    ],
)
def test_mixed_trellis_uses_large_m_safe_tile_geometry(hidden, intermediate, expected):
    assert Exl3MoEMethod._mixed_trellis_tile_config(hidden, intermediate) == expected


def test_r7_projection_tiers_accept_native_k3_k4_k5() -> None:
    projection_bits = {
        ("w13", "w1"): (3, 4, 5, 3),
        ("w13", "w3"): (4, 3, 5, 4),
        ("w2", "w2"): (5, 4, 3, 5),
    }

    def parameter(group: str, shard: str):
        return SimpleNamespace(
            exl3_tensors={
                (expert, shard): torch.empty((1, 1, 16 * bits), dtype=torch.int16)
                for expert, bits in enumerate(projection_bits[(group, shard)])
            }
        )

    layer = SimpleNamespace(
        local_num_experts=4,
        w13_trellis=SimpleNamespace(exl3_tensors={}),
        w2_trellis=SimpleNamespace(exl3_tensors={}),
    )
    layer.w13_trellis.exl3_tensors.update(parameter("w13", "w1").exl3_tensors)
    layer.w13_trellis.exl3_tensors.update(parameter("w13", "w3").exl3_tensors)
    layer.w2_trellis.exl3_tensors.update(parameter("w2", "w2").exl3_tensors)
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(r7_routed_experts={"k_values": (3, 4, 5)})

    bits, tiers = method._r7_projection_tiers(layer)

    assert bits == (3, 4, 5)
    assert tiers == {
        "gate": (0, 1, 2, 0),
        "up": (1, 0, 2, 1),
        "down": (2, 1, 0, 2),
    }


def test_rank_sliced_runtime_scope_is_per_owning_model():
    """Target and rank-sliced MTP draft layers must not share a cached runtime.

    The rank-sliced runtime cache stores mutable Trellis/prefill scratch and
    parity staging buffers. A target MoE layer and an MTP draft layer of the same
    model have identical shapes, topk and planner settings, so a shape-only key
    would hand the draft the target's scratch and break the target/draft
    isolation their independently captured CUDA graphs depend on.
    """
    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_scope = exl3_module._runtime_scope_id(target_config)
    draft_scope = exl3_module._runtime_scope_id(draft_config)

    # Distinct owning configs must never collide...
    assert target_scope != draft_scope
    # ...and the scope must be stable, so every layer of one model keeps sharing
    # a single runtime (the prefill arena is ~1 GiB; per-layer runtimes would not
    # fit on a 75+ layer model).
    assert exl3_module._runtime_scope_id(target_config) == target_scope
    assert exl3_module._runtime_scope_id(draft_config) == draft_scope


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("moe_layers", [3.5, 77]),
        ("moe_layers", [False, 77]),
        ("k_values", [3.5, 4]),
        ("k_values", [True, 4]),
    ],
)
def test_r7_schema_rejects_lossy_integer_coercion(field, value):
    r7 = {
        "schema": "r7-complete-v2-checkpoint-v1",
        "codebook": "mcg",
        "bits": "mixed_tensor",
        "moe_layers": [3, 77],
        "k_values": [3, 4, 5],
    }
    r7[field] = value
    with pytest.raises(ValueError):
        Exl3Config.from_config({"r7_routed_experts": r7})


def test_r7_schema_normalizes_declared_integer_contract():
    config = Exl3Config.from_config(
        {
            "r7_routed_experts": {
                "schema": "r7-complete-v2-checkpoint-v1",
                "codebook": "mcg",
                "bits": "mixed_tensor",
                "moe_layers": [3, 77],
                "k_values": [5, 3, 4, 3],
            }
        }
    )
    assert config.r7_routed_experts["moe_layers"] == (3, 77)
    assert config.r7_routed_experts["k_values"] == (3, 4, 5)


def test_rank_sliced_runtime_key_differs_across_models_with_same_shape():
    """Two same-shape layers owned by different models get different cache keys."""

    def _key(quant_config):
        # Mirrors the scope-prefixed key built in Exl3MoEMethod._rank_sliced_runtime
        # for two layers whose shape/planner components are byte-for-byte equal.
        return (
            exl3_module._runtime_scope_id(quant_config),
            0,  # device index
            torch.bfloat16,
            5120,  # hidden size
            768,  # intermediate size per partition
            64,  # local experts
            8,  # topk
            3072,  # max batched tokens
        )

    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_key = _key(target_config)
    draft_key = _key(draft_config)

    assert target_key != draft_key
    # Everything except the leading scope is identical, proving the scope is the
    # only thing preventing the collision.
    assert target_key[1:] == draft_key[1:]
    # Same owner -> same key, so target layers still share one runtime.
    assert _key(target_config) == target_key


def test_rank_sliced_window_defaults_to_min_capturable_m(monkeypatch) -> None:
    """Every rank-sliced layer must cover small rows without an env workaround.

    Regression test for the boot failure reported in vLLM #183: with the Trellis
    window left at its historical default of 4, CUDA-graph capture of an EXL3
    rank-sliced MTP draft reaches the eager parity path at m=1,2,3 and the engine
    cannot start:

        RuntimeError: EXL3 eager parity path entered during CUDA graph capture
        (m=3); capture sizes must lie inside the Trellis window [4, 32]

    It was invariant to num_speculative_tokens and to cudagraph_capture_sizes,
    because m here is the draft's row count per step, not a target batch size.
    Target profiling can also produce m=3, so role-dependent defaults merely
    move the same failure from the draft to MTP0. The backend declares one
    capability floor and uses it for both roles.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    monkeypatch.delenv("VLLM_EXL3_TRELLIS_MIN_M", raising=False)

    # The GLM-5.2 MTP head is named exactly like a target layer, so the role
    # comes from the exl3_is_draft stamp applied by load_eagle_model -- name
    # inspection alone cannot classify it.
    draft = SimpleNamespace(
        layer_name="model.layers.78.mlp.experts", exl3_is_draft=True
    )
    target = SimpleNamespace(layer_name="model.layers.30.mlp.experts")

    assert exl3_mod._is_draft_layer(draft)
    assert not exl3_mod._is_draft_layer(target)
    # Unstamped draft with a distinctive prefix still classifies via fallback.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    # A stamp always wins over the name, in both directions.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )

    def resolved():
        return exl3_mod._positive_env_int(
            "VLLM_EXL3_TRELLIS_MIN_M", exl3_mod._DEFAULT_TRELLIS_MIN_M
        )

    assert resolved() == exl3_mod._DEFAULT_TRELLIS_MIN_M
    assert resolved() == exl3_mod.MIN_CAPTURABLE_TRELLIS_M == 1

    # An explicit value remains authoritative as a diagnostic kill switch.
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_MIN_M", "4")
    assert resolved() == 4


def test_draft_role_stamp_wins_over_name() -> None:
    """The exl3_is_draft stamp set in create_weights is authoritative.

    Forward/plan/capture time has no current vllm config, so the role cannot be
    inferred there; create_weights stamps it from runner_type while the
    construction context is live. Stamped values must win over any name
    heuristic in both directions.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    # GLM-5.2 MTP head: named like a target, stamped draft.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.78.mlp.experts", exl3_is_draft=True)
    )
    # Target stamped False keeps its role even with a suspicious name.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )
    # Unstamped layers fall back to the name heuristic.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.30.mlp.experts")
    )
