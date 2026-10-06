# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 (744B) tensor-parallel padding for TP sizes that do not divide it.

The config hook pads the 64 MLA heads and the 2048-channel experts at TP6;
the loaders place the checkpoint at the head of the padded width and zero the
tail, so padded heads and channels contribute nothing.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaMTPConfig,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.b12x_mla_sparse import _round_up_heads


def _model_config(architecture="GlmMoeDsaForCausalLM", model_type="glm_moe_dsa"):
    text = SimpleNamespace(
        model_type=model_type,
        num_attention_heads=64,
        num_key_value_heads=64,
        moe_intermediate_size=2048,
        intermediate_size=12288,
    )
    model_config = SimpleNamespace(
        architecture=architecture,
        hf_config=text,
        hf_text_config=text,
        model_arch_config=None,
    )
    model_config.get_model_arch_config = lambda: SimpleNamespace(
        total_num_attention_heads=text.num_attention_heads
    )
    return model_config


def _parallel_config(tp_size, enable_expert_parallel=False):
    return SimpleNamespace(
        tensor_parallel_size=tp_size, enable_expert_parallel=enable_expert_parallel
    )


@pytest.fixture
def cuda_platform(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)


def test_target_and_mtp_draft_use_the_padding_hook() -> None:
    assert MODELS_CONFIG_MAP["GlmMoeDsaForCausalLM"] is GlmMoeDsaForCausalLM
    assert MODELS_CONFIG_MAP["DeepseekV32MTPModel"] is GlmMoeDsaMTPConfig


@pytest.mark.parametrize(
    "architecture,model_type",
    [("GlmMoeDsaForCausalLM", "glm_moe_dsa"), ("DeepseekV32MTPModel", "deepseek_mtp")],
)
def test_tp6_pads_heads_and_expert_width(cuda_platform, architecture, model_type):
    model_config = _model_config(architecture, model_type)
    # The MTP draft config keeps its target's type (SpeculativeConfig).
    model_config.hf_text_config.mtp_target_model_type = "glm_moe_dsa"
    parallel_config = _parallel_config(6)
    # Idempotent: the hook runs from both verify paths.
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)

    text = model_config.hf_text_config
    assert (text.num_attention_heads, text.original_num_attention_heads) == (66, 64)
    assert (text.num_key_value_heads, text.original_num_key_value_heads) == (66, 64)
    # 352 channels per rank: whole 32-channel tiles for B12X W4A16.
    assert (text.moe_intermediate_size, text.original_moe_intermediate_size) == (
        2112,
        2048,
    )
    assert text.intermediate_size == 12288
    assert model_config.model_arch_config.total_num_attention_heads == 66


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_divisible_tp_sizes_are_noops(cuda_platform, tp_size) -> None:
    model_config = _model_config()
    ModelConfig._update_model_config_for_parallelism(
        model_config, _parallel_config(tp_size)
    )
    text = model_config.hf_text_config
    assert (text.num_attention_heads, text.moe_intermediate_size) == (64, 2048)
    assert not hasattr(text, "original_num_attention_heads")
    assert model_config.model_arch_config is None


def test_deepseek_v32_mtp_drafts_are_not_padded(cuda_platform) -> None:
    model_config = _model_config("DeepseekV32MTPModel", model_type="deepseek_mtp")
    model_config.hf_text_config.mtp_target_model_type = "deepseek_v32"
    ModelConfig._update_model_config_for_parallelism(model_config, _parallel_config(6))
    assert model_config.hf_text_config.num_attention_heads == 64


def test_padded_experts_reject_expert_parallelism(cuda_platform) -> None:
    with pytest.raises(ValueError, match="expert parallelism"):
        ModelConfig._update_model_config_for_parallelism(
            _model_config(), _parallel_config(6, enable_expert_parallel=True)
        )


def _experts(tp_size):
    experts = RoutedExperts.__new__(RoutedExperts)
    experts.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=tp_size),
    )
    return experts


@pytest.mark.parametrize("rank", range(6))
def test_padded_expert_shards_take_whole_tiles_and_zero_the_tail(rank) -> None:
    """Rank r holds checkpoint channels [r * 352, (r + 1) * 352); the last
    rank's 64 channels past the checkpoint's 2048 are zero."""
    experts = _experts(6)
    width, local, hidden = 2048, 352, 8
    gate = torch.arange(width * hidden, dtype=torch.float32).view(width, hidden) + 1
    up = -gate
    down = gate.t().contiguous()
    w13 = torch.full((2 * local, hidden), float("nan"))
    w2 = torch.full((hidden, local), float("nan"))
    experts._load_w13(w13, 0, "w1", gate, rank)
    experts._load_w13(w13, 0, "w3", up, rank)
    experts._load_w2(w2, 1, down, rank)

    first = rank * local
    real = min(width - first, local)
    assert torch.equal(w13[:real], gate[first : first + real])
    assert torch.equal(w13[local : local + real], up[first : first + real])
    assert torch.equal(w2[:, :real], down[:, first : first + real])
    assert not w13[real:local].any() and not w13[local + real :].any()
    assert not w2[:, real:].any()


def test_evenly_split_expert_shards_keep_their_share() -> None:
    experts = _experts(8)
    gate = torch.arange(2048 * 4, dtype=torch.float32).view(2048, 4)
    w13 = torch.empty((2 * 256, 4))
    experts._load_w13(w13, 0, "w1", gate, 3)
    assert torch.equal(w13[:256], gate[768:1024])


@pytest.mark.parametrize(
    "heads,expected", [(8, 8), (11, 16), (16, 16), (22, 24), (33, 40), (64, 64)]
)
def test_sparse_mla_runs_whole_groups_of_eight_heads(heads, expected) -> None:
    assert _round_up_heads(heads) == expected
