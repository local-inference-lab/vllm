# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.config.virtual_tp import (
    VIRTUAL_TP_PLAN_ATTR,
    apply_b12x_virtual_tp_padding_to_model_config,
    maybe_apply_b12x_virtual_tp_padding,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.fixture(autouse=True)
def virtual_tp_shape_test_device_count(monkeypatch):
    # These tests inspect configuration and CPU weight loading, independently
    # of the number of CUDA devices exposed to the test process.
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_count", lambda: 16)


class FakeKimiK3ModelConfig:
    def __init__(self, *, mm_encoder_tp_mode: str = "weights"):
        self.hf_text_config = SimpleNamespace(
            model_type="kimi_linear",
            num_attention_heads=96,
            intermediate_size=33792,
            moe_intermediate_size=3072,
            vocab_size=163840,
            linear_attn_config={"num_heads": 96},
            quantization_config={
                "dense_format": "mxfp8",
                "qsrt": {"profile": "k2_coupled_h512_h128"},
            },
        )
        self.vision_config = SimpleNamespace(
            hidden_size=1024,
            intermediate_size=4096,
            vt_intermediate_size=4096,
            merge_kernel_size=(2, 2),
        )
        self.hf_config = SimpleNamespace(
            model_type="kimi_k3",
            text_config=self.hf_text_config,
            vision_config=self.vision_config,
        )
        self.multimodal_config = SimpleNamespace(
            mm_encoder_tp_mode=mm_encoder_tp_mode,
        )
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeKimiK3DSparkModelConfig:
    def __init__(self):
        self.hf_config = SimpleNamespace(
            model_type="k3_dspark",
            architectures=["K3DSparkModel"],
            num_attention_heads=64,
            intermediate_size=14336,
            vocab_size=163840,
        )
        self.hf_text_config = self.hf_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


@pytest.mark.parametrize("tp_size", [6, 12])
def test_b12x_virtual_tp_padding_kimi_k3_vocab(tp_size: int):
    model_config = FakeKimiK3ModelConfig()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 3072
    assert text_config.vocab_size == 163840
    assert plan["attention_heads"]["padded_size"] == 96
    assert plan["moe_intermediate_size"]["padded_size"] == 3072
    assert plan["vocab_size"] == {
        "original_size": 163840,
        "padded_size": 163968,
        "tp_size": tp_size,
        "local_size": 163968 // tp_size,
        "padding_size": 192,
    }


@pytest.mark.parametrize(
    ("mm_encoder_tp_mode", "expected_intermediate_size", "expected_local_size"),
    [
        ("weights", 4224, 352),
        ("data", 4096, 4096),
    ],
)
def test_b12x_virtual_tp_kimi_k3_vision_sharding(
    mm_encoder_tp_mode: str,
    expected_intermediate_size: int,
    expected_local_size: int,
):
    model_config = FakeKimiK3ModelConfig(
        mm_encoder_tp_mode=mm_encoder_tp_mode,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=12),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    vision_config = model_config.vision_config
    plan = getattr(vision_config, VIRTUAL_TP_PLAN_ATTR)
    assert vision_config.original_intermediate_size == 4096
    assert vision_config.original_vt_intermediate_size == 4096
    assert vision_config.intermediate_size == expected_intermediate_size
    assert vision_config.vt_intermediate_size == expected_intermediate_size
    expected_axis = {
        "original_size": 4096,
        "padded_size": expected_intermediate_size,
        "tp_size": 12 if mm_encoder_tp_mode == "weights" else 1,
        "local_size": expected_local_size,
    }
    assert plan["vision_intermediate_size"] == expected_axis
    assert plan["vision_projector_hidden_size"] == expected_axis


def test_b12x_virtual_tp_kimi_k3_pads_linear_attention_heads():
    model_config = FakeKimiK3ModelConfig()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=10),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    linear_attn_config = model_config.hf_text_config.linear_attn_config
    assert linear_attn_config["original_num_heads"] == 96
    assert linear_attn_config["num_heads"] == 100
    assert model_config.hf_text_config.moe_intermediate_size == 3840


def test_b12x_virtual_tp_kimi_k3_pads_dense_intermediate_size():
    model_config = FakeKimiK3ModelConfig()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=10),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.original_intermediate_size == 33792
    assert text_config.intermediate_size == 33920
    assert plan["dense_intermediate_size"] == {
        "original_size": 33792,
        "padded_size": 33920,
        "tp_size": 10,
        "local_size": 3392,
    }


def test_b12x_virtual_tp_kimi_k3_dspark_tp12_zero_tail_plan():
    model_config = FakeKimiK3DSparkModelConfig()
    parallel_config = ParallelConfig(tensor_parallel_size=12)

    apply_b12x_virtual_tp_padding_to_model_config(
        cast(Any, model_config), parallel_config
    )

    config = model_config.hf_config
    plan = getattr(config, VIRTUAL_TP_PLAN_ATTR)
    assert config.original_num_attention_heads == 64
    assert config.num_attention_heads == 72
    assert plan["attention_heads"] == {
        "original_size": 64,
        "padded_size": 72,
        "tp_size": 12,
        "local_size": 6,
    }
    assert config.original_intermediate_size == 14336
    assert config.intermediate_size == 14340
    assert plan["dense_intermediate_size"] == {
        "original_size": 14336,
        "padded_size": 14340,
        "tp_size": 12,
        "local_size": 1195,
    }


def test_kimi_kda_conv_loader_zero_fills_virtual_heads():
    from vllm.models.kimi_k3.nvidia.kda import (
        _make_decode_conv1d_weight_loader,
    )

    model_config = FakeKimiK3ModelConfig()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=ParallelConfig(tensor_parallel_size=10),
        kernel_config=SimpleNamespace(moe_backend="b12x"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA_SPARSE,
        ),
    )
    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    local_projection = 10 * 128
    source_projection = 96 * 128
    param = torch.full((3 * local_projection, 1, 4), -1.0)
    loaded = torch.arange(source_projection * 4, dtype=torch.float32).reshape(
        source_projection, 1, 4
    )
    loader = _make_decode_conv1d_weight_loader(
        [100 * 128] * 3,
        tp_size=10,
        tp_rank=9,
        decode_conv1d_weight=None,
    )

    with set_current_vllm_config(cast(Any, vllm_config)):
        loader(param, loaded, 0)

    assert torch.equal(param[: 6 * 128], loaded[90 * 128 :])
    assert torch.count_nonzero(param[6 * 128 : local_projection]) == 0
    assert torch.all(param[local_projection:] == -1)
