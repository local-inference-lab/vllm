# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for draft config overrides used by SpeculativeConfig.

Callable ``hf_overrides`` on the target model config (e.g. the
``dummy_hf_overrides`` shrink used by ``tests/models/test_initialization.py``)
must also be applied when building the draft ``ModelConfig``. Otherwise a
draft belonging to a large target model is instantiated at full size even
when the target itself is shrunk — which is what kept spec-decode archs like
``EagleMistralLarge3ForCausalLM`` stuck at ``is_available_online=False``
("TODO: revert once figuring out OOM in CI").
"""

import functools
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_hf_config(**kwargs) -> PretrainedConfig:
    defaults = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return PretrainedConfig(**defaults)


@pytest.mark.cpu_test
def test_dict_overrides_are_not_forwarded_to_draft():
    """Dict overrides are target-specific key patches; the draft must get
    only the architecture-mapping override."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"max_position_embeddings": 1234}
    )
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["mtp", "draft_model"])
def test_mtp_draft_receives_target_dict_overrides(method: str):
    """An in-model MTP draft shares the target's positional geometry."""
    target_hf_overrides = {
        "architectures": ["Qwen3_8FlashNextForCausalLM"],
        "model_type": "qwen3_8_flash_next",
        "text_config": {
            "max_position_embeddings": 1048576,
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
        },
    }
    override = SpeculativeConfig.get_draft_hf_overrides(method, target_hf_overrides)
    assert callable(override)
    text_config = _make_hf_config(
        max_position_embeddings=262144,
        rope_parameters={"rope_type": "default"},
        hc_count=4,
        mtp_num_hidden_layers=2,
        num_attention_heads=4,
    )
    source = _make_hf_config(
        architectures=["Qwen3_8FlashNextForCausalLM"],
        model_type="qwen3_8_flash_next",
        text_config=text_config,
    )

    out = override(source)

    assert out.model_type == "qwen4_exp_mtp"
    assert out.architectures == ["Qwen4ExpMTP"]
    assert out.text_config.max_position_embeddings == 1048576
    assert out.text_config.rope_parameters == {"rope_type": "yarn", "factor": 4.0}


@pytest.mark.cpu_test
def test_none_overrides_fall_back_to_arch_mapping():
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
def test_callable_overrides_reach_the_draft_config():
    """A callable override (config-to-config transform) composes with the
    architecture-mapping override and is applied to the draft config."""

    def shrink(hf_config: PretrainedConfig) -> PretrainedConfig:
        hf_config.num_hidden_layers = 1
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(shrink)
    assert composed is not SpeculativeConfig.hf_config_override

    out = composed(_make_hf_config())
    # The shrink transform must have been applied to the draft config.
    assert out.num_hidden_layers == 1


@pytest.mark.cpu_test
def test_arch_mapping_applies_before_callable_override():
    """The static arch-mapping override runs first, so the user callable
    observes (and may adjust) the post-mapping config."""
    seen_architectures: list[str] = []

    def record(hf_config: PretrainedConfig) -> PretrainedConfig:
        seen_architectures.append(hf_config.architectures[0])
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(record)

    # MiMo is one of the arch-mapped model types: hf_config_override
    # rewrites architectures to ["MiMoMTPModel"].
    mimo = _make_hf_config(
        architectures=["MiMoForCausalLM"],
        model_type="mimo",
        num_nextn_predict_layers=1,
    )
    composed(mimo)
    assert seen_architectures == ["MiMoMTPModel"]


@pytest.mark.cpu_test
def test_inkling_override_exposes_all_mtp_depths():
    text_config = _make_hf_config(
        architectures=["InklingForCausalLM"],
        model_type="inkling_model",
        local_layer_ids=[1, 3],
    )
    config = _make_hf_config(
        architectures=["InklingForConditionalGeneration"],
        model_type="inkling_mm_model",
        text_config=text_config,
        mtp_config={
            "num_nextn_predict_layers": 8,
            "local_layer_ids": [0, 2, 4],
        },
    )

    out = SpeculativeConfig.hf_config_override(config)

    assert out is text_config
    assert out.model_type == "inkling_mtp"
    assert out.architectures == ["InklingMTPModel"]
    # Multi-module MTP: every checkpoint depth is exposed (module i drafts
    # speculative token i), no longer clamped to the first depth.
    assert out.n_predict == 8
    assert out.num_nextn_predict_layers == 8
    assert out.chain_hidden_post_norm is False
    assert out.local_layer_ids == [0, 2, 4]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "method,depth,valid",
    [("dspark", 5, True), ("dspark", 7, True), ("mtp", 6, True), ("mtp", 7, False)],
)
def test_block_diffusion_depth_does_not_follow_mtp_layer_count(method, depth, valid):
    """DSpark predicts a block independently of its three draft transformer layers."""
    model = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    revision = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
    target = ModelConfig(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        max_model_len=32768,
    )
    expected = (
        nullcontext()
        if valid
        else pytest.raises(ValueError, match="must be divisible by n_predict=3")
    )
    with expected:
        config = SpeculativeConfig(
            target_model_config=target,
            target_parallel_config=ParallelConfig(tensor_parallel_size=2),
            model=model,
            revision=revision,
            method=method,
            num_speculative_tokens=depth,
            draft_sample_method="probabilistic",
            enable_adaptive_verification=method == "dspark",
        )
        assert config.num_speculative_tokens == depth
        assert config.draft_model_config.hf_config.num_nextn_predict_layers == 3
        assert config.parallel_drafting == (method == "dspark")


def _module_level_shrink(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config.num_hidden_layers = 1
    return hf_config


@pytest.mark.cpu_test
def test_composed_override_is_picklable():
    """The draft ``ModelConfig`` is sent to spawned engine-core processes, so
    the composed override must be picklable. A nested local closure is not
    (it raised ``Can't get local object`` on DFlashDraftModel); a
    ``functools.partial`` over a module-referenceable static method is.
    Guard against regressing to a closure."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(_module_level_shrink)

    assert isinstance(composed, functools.partial)
    assert composed.func is SpeculativeConfig._apply_composed_hf_override

    out = composed(_make_hf_config())
    assert out.num_hidden_layers == 1


def _make_mtp_speculative_config(
    override: bool | None,
    checkpoint_value: bool,
) -> SpeculativeConfig:
    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=1,
        index_share_for_mtp_iteration=checkpoint_value,
    )
    draft_model_config = MagicMock(
        model="draft",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        return SpeculativeConfig(
            model="draft",
            method="mtp",
            num_speculative_tokens=1,
            index_share_for_mtp_iteration=override,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("override", "checkpoint_value", "expected"),
    [(None, True, True), (False, True, False), (True, False, True)],
)
def test_mtp_index_share_override(
    override: bool | None, checkpoint_value: bool, expected: bool
):
    speculative_config = _make_mtp_speculative_config(override, checkpoint_value)
    assert (
        speculative_config.draft_model_config.hf_config.index_share_for_mtp_iteration
        is expected
    )
