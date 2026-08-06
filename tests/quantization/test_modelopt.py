# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test ModelOpt quantization method setup and weight loading.

Run `pytest tests/quantization/test_modelopt.py`.
"""

import os
from types import SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _skip(msg: str) -> NoReturn:
    pytest.skip(msg)
    raise RuntimeError(msg)


def _snapshot_download_or_skip(model_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        _skip(f"huggingface_hub is required to download {model_id}: {e}")

    try:
        return snapshot_download(
            repo_id=model_id,
            repo_type="model",
            # These checkpoints are already small; download full repo for simplicity.
            allow_patterns=["*"],
        )
    except Exception as e:
        _skip(f"Failed to download {model_id} from the HF Hub: {e}")


def _mock_lm_head() -> Mock:
    lm_head = Mock(spec=ParallelLMHead)
    lm_head.__class__ = ParallelLMHead
    return lm_head


def _mock_linear() -> Mock:
    return Mock(spec=LinearBase)


def _mixed_precision_config(quantized_layers: dict) -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers=quantized_layers,
        fp8_config=ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        nvfp4_config=ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        w4a16_nvfp4_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        mxfp8_config=ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
    )


def test_modelopt_nvfp4_quantizes_parallel_lm_head():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptNvFp4LinearMethod)


def test_modelopt_nvfp4_online_quantizes_bf16_dense_linears():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[
            "model.layers.*.self_attn*",
            "*shared_experts*",
            "lm_head",
        ],
    )
    current_config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=QuantizationConfigArgs(linear="mxfp8")
        )
    )
    sentinel = MagicMock()

    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "get_current_vllm_config_or_none",
            return_value=current_config,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt.Mxfp8OnlineLinearMethod",
            return_value=sentinel,
        ),
    ):
        attn_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
        )
        indexer_method = config.get_quant_method(
            _mock_linear(), "model.layers.0.self_attn.indexer.wq_b"
        )
        # shared experts are governed solely by the shared_experts spec
        shared_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
        )
        lm_head_method = config.get_quant_method(_mock_lm_head(), "lm_head")

    assert attn_method is sentinel
    assert indexer_method is sentinel
    assert isinstance(shared_method, UnquantizedLinearMethod)
    assert isinstance(lm_head_method, UnquantizedLinearMethod)


def test_modelopt_nvfp4_dense_overlay_honors_ignore():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["model.layers.*.self_attn*"],
    )
    current_config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=QuantizationConfigArgs(
                linear="mxfp8",
                ignore=["re:.*o_proj"],
            )
        )
    )
    sentinel = MagicMock()

    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "get_current_vllm_config_or_none",
            return_value=current_config,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt.Mxfp8OnlineLinearMethod",
            return_value=sentinel,
        ),
    ):
        kv_b_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
        )
        o_proj_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.o_proj"
        )

    assert kv_b_method is sentinel
    assert isinstance(o_proj_method, UnquantizedLinearMethod)


def test_modelopt_nvfp4_dense_overlay_composes_with_shared_experts():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["model.layers.*.self_attn*", "*shared_experts*"],
    )
    current_config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=QuantizationConfigArgs(
                linear="mxfp8",
                shared_experts="mxfp8",
            )
        )
    )
    dense_sentinel = MagicMock()

    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "get_current_vllm_config_or_none",
            return_value=current_config,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt.Mxfp8OnlineLinearMethod",
            return_value=dense_sentinel,
        ),
    ):
        attn_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.self_attn.kv_b_proj"
        )
        shared_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.gate_up_proj"
        )

    assert attn_method is dense_sentinel
    assert shared_method is dense_sentinel


def test_modelopt_nvfp4_online_quantizes_bf16_shared_experts():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["*shared_experts*", "lm_head"],
    )
    current_config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization_config=QuantizationConfigArgs(shared_experts="mxfp8")
        )
    )
    sentinel = MagicMock()

    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "get_current_vllm_config_or_none",
            return_value=current_config,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt.Mxfp8OnlineLinearMethod",
            return_value=sentinel,
        ),
    ):
        gate_up_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.gate_up_proj"
        )
        down_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.down_proj"
        )
        router_method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.router"
        )
        lm_head_method = config.get_quant_method(_mock_lm_head(), "lm_head")

    assert gate_up_method is sentinel
    assert down_method is sentinel
    assert isinstance(router_method, UnquantizedLinearMethod)
    assert isinstance(lm_head_method, UnquantizedLinearMethod)


def test_modelopt_nvfp4_preserves_serialized_shared_experts():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(
            _mock_linear(), "model.layers.3.mlp.shared_experts.gate_up_proj"
        )

    assert isinstance(method, ModelOptNvFp4LinearMethod)


def test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["lm_head"],
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, UnquantizedLinearMethod)


def test_modelopt_mixed_precision_quantizes_parallel_lm_head():
    config = _mixed_precision_config(
        {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
    )

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptNvFp4LinearMethod)


def test_modelopt_mixed_precision_resolves_declared_packed_projection():
    config = _mixed_precision_config(
        {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.k_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert config._resolve_quant_algo("model.layers.0.self_attn.qkv_proj") == "MXFP8"


def test_modelopt_mixed_precision_does_not_quantize_unlisted_fused_sibling():
    config = _mixed_precision_config(
        {
            "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.in_proj_z": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        }
    )
    config.packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    assert (
        config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_qkvz") == "FP8"
    )
    assert config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_ba") is None


def test_modelopt_mixed_precision_infers_fused_gate_up_projection():
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"},
            "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(fake_layer, "model.layers.0.mlp.gate_up_proj")

    assert isinstance(method, ModelOptNvFp4LinearMethod)


@pytest.mark.parametrize(
    ("quantized_prefix", "missing_prefix"),
    [
        ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj"),
        ("model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.qkv_proj"),
    ],
)
def test_modelopt_mixed_precision_does_not_infer_missing_sibling_linear(
    quantized_prefix, missing_prefix
):
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            quantized_prefix: {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, missing_prefix)

    assert isinstance(method, UnquantizedLinearMethod)


def test_vocab_parallel_embedding_weight_loader_accepts_scalar_scale():
    holder = Mock()
    scale = torch.nn.Parameter(torch.empty(1))
    loaded_scale = torch.tensor(2.0)

    VocabParallelEmbedding.weight_loader(holder, scale, loaded_scale)

    assert torch.equal(scale, loaded_scale.reshape(1))


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8 checkpoint loading and structure validation."""
    # TODO: provide a small publicly available test checkpoint
    model_path = (
        "/home/scratch.omniml_data_1/zhiyu/ckpts/test_ckpts/"
        "TinyLlama-1.1B-Chat-v1.0-fp8-0710"
    )

    # Skip test if checkpoint doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(
            f"Test checkpoint not found at {model_path}. "
            "This test requires a local ModelOpt FP8 checkpoint."
        )

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # Check that ModelOpt quantization method is properly applied
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8LinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8LinearMethod)

            # Check weight dtype is FP8
            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Check scales are present and have correct dtype
            assert hasattr(qkv_proj, "weight_scale")
            assert hasattr(qkv_proj, "input_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.input_scale.dtype == torch.float32

            assert hasattr(o_proj, "weight_scale")
            assert hasattr(o_proj, "input_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.input_scale.dtype == torch.float32

            assert hasattr(gate_up_proj, "weight_scale")
            assert hasattr(gate_up_proj, "input_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.input_scale.dtype == torch.float32

            assert hasattr(down_proj, "weight_scale")
            assert hasattr(down_proj, "input_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.input_scale.dtype == torch.float32

        llm.apply_model(check_model)

        # Run a simple generation test to ensure the model works
        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8 output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pc_pt_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8_PER_CHANNEL_PER_TOKEN checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pc-pt"
    model_path = _snapshot_download_or_skip(model_id)

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8PcPtLinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8PcPtLinearMethod)

            fp8_dtype = current_platform.fp8_dtype()
            assert qkv_proj.weight.dtype == fp8_dtype
            assert o_proj.weight.dtype == fp8_dtype
            assert gate_up_proj.weight.dtype == fp8_dtype
            assert down_proj.weight.dtype == fp8_dtype

            # Per-channel scales; activations are dynamically scaled per token.
            assert hasattr(qkv_proj, "weight_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.weight_scale.dim() == 1
            assert not hasattr(qkv_proj, "input_scale")

            assert hasattr(o_proj, "weight_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.weight_scale.dim() == 1
            assert not hasattr(o_proj, "input_scale")

            assert hasattr(gate_up_proj, "weight_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.weight_scale.dim() == 1
            assert not hasattr(gate_up_proj, "input_scale")

            assert hasattr(down_proj, "weight_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.weight_scale.dim() == 1
            assert not hasattr(down_proj, "input_scale")

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8_PER_CHANNEL_PER_TOKEN output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pb_wo_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8_PB_WO checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pb-wo"
    model_path = _snapshot_download_or_skip(model_id)

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8PbWoLinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8PbWoLinearMethod)

            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Block scales; should be materialized as a 2D [out_blk, in_blk] tensor.
            assert hasattr(qkv_proj, "weight_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.weight_scale.dim() == 2

            assert hasattr(o_proj, "weight_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.weight_scale.dim() == 2

            assert hasattr(gate_up_proj, "weight_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.weight_scale.dim() == 2

            assert hasattr(down_proj, "weight_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.weight_scale.dim() == 2

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8_PB_WO output: {output}")


def test_modelopt_nvfp4_config_dispatches_w4a4_method():
    """``quant_method="NVFP4"`` (W4A4 default) routes to the existing
    ``ModelOptNvFp4LinearMethod``."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.LinearMethodCls is ModelOptNvFp4LinearMethod
    assert config.quant_method == "NVFP4"


def test_modelopt_nvfp4_config_dispatches_w4a16_method():
    """``quant_method="W4A16_NVFP4"`` routes to the new
    ``ModelOptNvFp4W4A16LinearMethod`` instead of the W4A4 sibling.

    Mirrors the FP8 dispatch precedent (``ModelOptFp8Config`` selects
    one of three FP8 LinearMethods on ``quant_method``); a regression
    here would mean a W4A16 NVFP4 checkpoint silently loaded under the
    W4A4 method, which would try to register an ``input_scale`` runtime
    parameter and (more importantly) call the cutlass W4A4 NVFP4 GEMM
    instead of FP4 Marlin.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
        ModelOptNvFp4W4A16LinearMethod,
    )

    config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod
    assert config.LinearMethodCls is not ModelOptNvFp4LinearMethod
    assert config.quant_method == "W4A16_NVFP4"


@pytest.mark.parametrize(
    "quant_method, expected_use_a16, act_key_is_none",
    [
        ("NVFP4", False, False),  # W4A4 default
        ("W4A16_NVFP4", True, True),  # native W4A16 ckpt
    ],
)
def test_modelopt_nvfp4_moe_dispatches_to_marlin_when_w4a16(
    quant_method, expected_use_a16, act_key_is_none
):
    """``ModelOptNvFp4FusedMoE``: when the ckpt's ``quant_method`` is
    ``W4A16_NVFP4``, the MoE class must pass ``activation_key=None`` to
    ``select_nvfp4_moe_backend``. That filters out every W4A4 backend
    (their ``_supports_quant_scheme`` requires
    ``(kNvfp4Static, kNvfp4Dynamic)`` exactly); Marlin survives because
    it only checks ``weight_key``. A regression here would mean a W4A16
    ckpt silently went to the cutlass W4A4 path.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    config = ModelOptNvFp4Config(
        quant_method=quant_method,
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )

    mock_select = MagicMock(return_value=(MagicMock(), MagicMock()))
    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt.select_nvfp4_moe_backend",
            mock_select,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "is_global_sf_supported_for_nvfp4_backend",
            return_value=False,
        ),
    ):
        moe = ModelOptNvFp4FusedMoE(config, MagicMock())

    assert moe.use_a16 is expected_use_a16
    _, kwargs = mock_select.call_args
    assert kwargs["weight_key"] is kNvfp4Static
    if act_key_is_none:
        assert kwargs["activation_key"] is None
    else:
        assert kwargs["activation_key"] is kNvfp4Dynamic


@pytest.mark.parametrize(
    "per_layer_algo, expected_linear_cls_name",
    [
        ("NVFP4", "ModelOptNvFp4LinearMethod"),
        ("W4A16_NVFP4", "ModelOptNvFp4W4A16LinearMethod"),
        ("MXFP8", "ModelOptMxFp8LinearMethod"),
    ],
)
def test_modelopt_mixed_precision_dispatches_layer_quant_algo(
    per_layer_algo, expected_linear_cls_name
):
    """``ModelOptMixedPrecisionConfig.get_quant_method`` must route a Linear
    layer to the right LinearMethod based on its per-layer ``quant_algo``
    entry in ``quantized_layers``. Verifies W4A16 and MXFP8 branches coexist
    with the existing ``NVFP4`` branch without regression. A regression here
    would mean a quantized layer in a mixed-precision ckpt
    silently fell through to ``UnquantizedLinearMethod``.

    NOTE: FP8 dispatch is not
    covered here because ``ModelOptFp8LinearMethod.__init__`` reads
    ``get_current_vllm_config().model_config.dtype``, which requires a
    fully constructed ``ModelConfig`` (real model path). FP8 routing in
    mixed-precision is exercised by the existing integration tests
    above that use the ``vllm_runner`` fixture (e.g.
    ``test_modelopt_fp8_checkpoint_setup``). Our PR doesn't change the
    FP8 branch, so this isn't a coverage gap.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import modelopt as m

    if (
        expected_linear_cls_name == "ModelOptNvFp4W4A16LinearMethod"
        and current_platform.is_rocm()
    ):
        pytest.skip("ModelOptNvFp4W4A16LinearMethod is not supported with rocm")

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.fake_proj": {"quant_algo": per_layer_algo},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, "model.layers.0.fake_proj")

    expected_cls = getattr(m, expected_linear_cls_name)
    assert isinstance(method, expected_cls), (
        f"Expected {expected_linear_cls_name}, got {type(method).__name__}"
    )


def test_modelopt_mixed_precision_prefers_explicit_w4a4_scheme() -> None:
    """Static FP4 activation metadata overrides a coarse W4A16 layer label."""
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts
    from vllm.model_executor.layers.quantization import modelopt as m

    w4a4_targets = [
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
    ]
    quantized_layers = {
        target: {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        for target in [*w4a4_targets, "model.layers.1.fake_proj"]
    }
    hf_quant_config: dict[str, Any] = {
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": w4a4_targets,
            }
        },
        "quantized_layers": quantized_layers,
    }

    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    assert config._resolve_quant_algo("model.layers.0.mlp.experts") == "NVFP4"
    assert (
        config._resolve_quant_algo("model.layers.0.mlp.shared_expert.gate_up_proj")
        == "NVFP4"
    )
    assert (
        config._resolve_quant_algo("model.layers.0.mlp.shared_expert.down_proj")
        == "NVFP4"
    )
    assert config._resolve_quant_algo("model.layers.1.fake_proj") == "W4A16_NVFP4"

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel",
        return_value=MagicMock(),
    ):
        method = config.get_quant_method(
            MagicMock(spec=LinearBase),
            "model.layers.0.mlp.shared_expert.down_proj",
        )
    assert isinstance(method, m.ModelOptNvFp4LinearMethod)

    fake_experts = MagicMock(spec=RoutedExperts)
    fake_experts.moe_config = MagicMock()
    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt.select_nvfp4_moe_backend",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "is_global_sf_supported_for_nvfp4_backend",
            return_value=False,
        ),
    ):
        moe_method = config.get_quant_method(fake_experts, "model.layers.0.mlp.experts")
    assert isinstance(moe_method, m.ModelOptNvFp4FusedMoE)
    assert not moe_method.use_a16


def test_modelopt_mixed_precision_resolves_minimax_qkv_from_shards() -> None:
    config = _mixed_precision_config(
        {
            "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.3.self_attn.k_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.3.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )

    assert (
        config._resolve_quant_algo("language_model.model.layers.3.self_attn.qkv_proj")
        == "MXFP8"
    )
    assert (
        config._resolve_quant_algo(
            "language_model.model.layers.3.self_attn.indexer.q_proj"
        )
        is None
    )
    assert (
        config._resolve_quant_algo(
            "language_model.model.layers.3.self_attn.indexer.k_proj"
        )
        is None
    )


def test_modelopt_mixed_precision_drives_minimax_indexer_split() -> None:
    from vllm.models.minimax_m3.nvidia.model import (
        _should_split_mxfp8_indexer_projection,
    )

    config = _mixed_precision_config(
        {
            "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.3.self_attn.k_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.3.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )

    assert _should_split_mxfp8_indexer_projection(
        config, "language_model.model.layers.3.self_attn"
    )


def test_modelopt_mixed_precision_resolves_minimax_dense_gate_up_shards() -> None:
    config = _mixed_precision_config(
        {
            "model.language_model.layers.0.mlp.gate_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.0.mlp.up_proj": {"quant_algo": "MXFP8"},
        }
    )

    assert (
        config._resolve_quant_algo("language_model.model.layers.0.mlp.gate_up_proj")
        == "MXFP8"
    )


def test_modelopt_mixed_precision_resolves_minimax_block_sparse_mlp_alias() -> None:
    config = _mixed_precision_config(
        {
            "model.language_model.layers.10.mlp.experts.0.gate_proj": {
                "quant_algo": "NVFP4"
            },
            "model.language_model.layers.10.mlp.experts.0.up_proj": {
                "quant_algo": "NVFP4"
            },
            "model.language_model.layers.10.mlp.experts.0.down_proj": {
                "quant_algo": "NVFP4"
            },
            "model.language_model.layers.10.mlp.shared_experts.gate_up_proj": {
                "quant_algo": "MXFP8"
            },
            "model.language_model.layers.10.mlp.shared_experts.down_proj": {
                "quant_algo": "MXFP8"
            },
        }
    )

    assert (
        config._resolve_quant_algo(
            "language_model.model.layers.10.block_sparse_moe.experts"
        )
        == "NVFP4"
    )
    assert (
        config._resolve_quant_algo(
            "language_model.model.layers.10."
            "block_sparse_moe.shared_experts.gate_up_proj"
        )
        == "MXFP8"
    )
    assert (
        config._resolve_quant_algo(
            "language_model.model.layers.10.block_sparse_moe.shared_experts.down_proj"
        )
        == "MXFP8"
    )


def test_modelopt_mixed_precision_rejects_mixed_fused_shards() -> None:
    config = _mixed_precision_config(
        {
            "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.language_model.layers.3.self_attn.k_proj": {"quant_algo": "NVFP4"},
            "model.language_model.layers.3.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )

    with pytest.raises(ValueError, match="Mixed quant_algo within fused layer"):
        config._resolve_quant_algo("language_model.model.layers.3.self_attn.qkv_proj")


def test_modelopt_mixed_precision_builds_w4a16_sibling_config():
    """Sanity: ``ModelOptMixedPrecisionConfig._from_config`` builds **two**
    NVFP4 sub-configs — one for W4A4 (default) and one tagged
    ``quant_method='W4A16_NVFP4'`` — so per-layer dispatch can hand
    Marlin-bound layers the right config without re-instantiating it on
    every call.
    """
    from vllm.model_executor.layers.quantization import modelopt as m

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.a": {"quant_algo": "NVFP4"},
                "model.layers.0.b": {"quant_algo": "W4A16_NVFP4"},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    assert config.nvfp4_config.quant_method == "NVFP4"
    assert config.nvfp4_config.LinearMethodCls is m.ModelOptNvFp4LinearMethod
    assert config.w4a16_nvfp4_config.quant_method == "W4A16_NVFP4"
    assert config.w4a16_nvfp4_config.LinearMethodCls is m.ModelOptNvFp4W4A16LinearMethod
