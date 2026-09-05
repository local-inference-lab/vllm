# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.8-Flash-Next configuration plumbing."""

from types import SimpleNamespace

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
)
from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)

_TEXT_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "layer_types": ["full_attention", "linear_attention"],
    "hc_count": 4,
    "mtp_num_hidden_layers": 2,
}


@pytest.mark.parametrize(
    ("model_type", "config_cls"),
    [
        ("qwen3_8_flash_next", Qwen3_8FlashNextConfig),
        ("qwen3_8_flash_next_text", Qwen3_8FlashNextTextConfig),
        ("qwen4_exp", Qwen4ExpConfig),
        ("qwen4_exp_text", Qwen4ExpTextConfig),
    ],
)
def test_config_registry(model_type, config_cls) -> None:
    assert _CONFIG_REGISTRY[model_type] is config_cls


def test_qwen4_alias_uses_qwen4_text_config() -> None:
    config = Qwen4ExpConfig(text_config=_TEXT_CONFIG)

    assert config.model_type == "qwen4_exp"
    assert isinstance(config.text_config, Qwen4ExpTextConfig)
    assert config.text_config.model_type == "qwen4_exp_text"


@pytest.mark.parametrize(
    ("configured_dtype", "expected_dtype"),
    [(None, "bfloat16"), ("bfloat16", "bfloat16"), ("float8_e4m3fn", "float8_e4m3fn")],
)
def test_ple_embedding_storage_dtype_is_preserved(
    configured_dtype: str | None, expected_dtype: str
) -> None:
    config = Qwen3_8FlashNextTextConfig(
        **_TEXT_CONFIG, ple_embedding_dtype=configured_dtype
    )

    assert config.ple_embedding_dtype == expected_dtype


@pytest.mark.parametrize("enabled", [True, False])
def test_mtp_selection_sharing_uses_serialized_text_configuration(enabled) -> None:
    config = Qwen3_8FlashNextConfig(
        text_config={
            **_TEXT_CONFIG,
            "index_share_for_mtp_iteration": enabled,
        }
    )
    restored = Qwen3_8FlashNextConfig.from_dict(config.to_dict())
    assert restored.text_config.index_share_for_mtp_iteration is enabled
    assert Qwen3_8FlashNextTextConfig(**_TEXT_CONFIG).index_share_for_mtp_iteration


def test_model_registry_aliases() -> None:
    assert _TEXT_GENERATION_MODELS["Qwen4ExpForCausalLM"] == (
        "vllm.models.qwen3_8_flash_next",
        "Qwen3_8FlashNextForCausalLM",
    )
    assert _MULTIMODAL_MODELS["Qwen4ExpForConditionalGeneration"] == (
        "vllm.models.qwen3_8_flash_next",
        "Qwen3_8FlashNextForConditionalGeneration",
    )
    assert _SPECULATIVE_DECODING_MODELS["Qwen3_8FlashNextMTP"] == (
        "vllm.models.qwen3_8_flash_next",
        "Qwen3_8FlashNextMTP",
    )


def test_model_registry_architectures_default_to_v2() -> None:
    assert {
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    } <= DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES


@pytest.mark.parametrize(
    ("config_cls", "architecture", "is_outer_config"),
    [
        (
            Qwen3_8FlashNextConfig,
            "Qwen3_8FlashNextForConditionalGeneration",
            True,
        ),
        (Qwen3_8FlashNextTextConfig, "Qwen3_8FlashNextForCausalLM", False),
        (Qwen4ExpConfig, "Qwen4ExpForConditionalGeneration", True),
        (Qwen4ExpTextConfig, "Qwen4ExpForCausalLM", False),
    ],
)
def test_mtp_override_recognizes_outer_and_text_types(
    config_cls, architecture: str, is_outer_config: bool
) -> None:
    if is_outer_config:
        config = config_cls(
            text_config=_TEXT_CONFIG,
            architectures=[architecture],
        )
    else:
        config = config_cls(**_TEXT_CONFIG, architectures=[architecture])

    config = SpeculativeConfig.hf_config_override(config)

    assert config.model_type == "qwen3_8_flash_next_mtp"
    assert config.architectures == ["Qwen3_8FlashNextMTP"]
    assert config.n_predict == 2
    assert config.hc_mult == 4


def test_mtp_arch_config_uses_native_layer_count() -> None:
    config = Qwen3_8FlashNextTextConfig(**_TEXT_CONFIG)
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS["qwen3_8_flash_next_mtp"]

    assert convertor_cls(config, config).get_num_hidden_layers() == 2


def _vllm_config(*, enable_dbo: bool = False, ple_layer_ids=None):
    rope_parameters = {
        "rope_type": "default",
        "mrope_section": [8, 4, 4],
        "mrope_interleaved": True,
    }
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=[] if ple_layer_ids is None else ple_layer_ids,
        indexer_n_heads=None,
        mamba_ssm_dtype=None,
        rope_parameters=rope_parameters,
    )
    model_config = SimpleNamespace(
        hf_config=text_config,
        hf_text_config=text_config,
        multimodal_config=None,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(enable_dbo=enable_dbo, ubatch_size=0),
        speculative_config=None,
    )


def test_qwen4_causal_alias_applies_text_config_hook() -> None:
    config_hook = MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"]
    vllm_config = _vllm_config()

    config_hook.verify_and_update_config(vllm_config)

    assert (
        "mrope_section" not in vllm_config.model_config.hf_text_config.rope_parameters
    )
    assert (
        "mrope_interleaved"
        not in vllm_config.model_config.hf_text_config.rope_parameters
    )


def test_qsa_and_ple_reject_dual_batch_overlap() -> None:
    config_hook = MODELS_CONFIG_MAP["Qwen4ExpForConditionalGeneration"]
    vllm_config = _vllm_config(enable_dbo=True, ple_layer_ids=[1])

    with pytest.raises(NotImplementedError, match="dual-batch overlap"):
        config_hook.verify_and_update_config(vllm_config)


def test_language_model_only_target_strips_mrope_from_native_draft() -> None:
    vllm_config = _vllm_config()
    vllm_config.model_config.multimodal_config = SimpleNamespace(
        language_model_only=True
    )
    draft_text_config = SimpleNamespace(
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [8, 4, 4],
            "mrope_interleaved": True,
        }
    )
    draft_outer_config = SimpleNamespace(
        vision_config=SimpleNamespace(),
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [8, 4, 4],
            "mrope_interleaved": True,
        },
    )
    draft_model_config = SimpleNamespace(
        hf_config=draft_outer_config,
        hf_text_config=draft_text_config,
        model_arch_config="stale",
        get_model_arch_config=lambda: "rebuilt",
    )
    vllm_config.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=draft_model_config,
    )

    config_hook = MODELS_CONFIG_MAP["Qwen3_8FlashNextForConditionalGeneration"]
    config_hook.verify_and_update_config(vllm_config)

    assert "mrope_section" not in (
        vllm_config.model_config.hf_text_config.rope_parameters
    )
    assert "mrope_section" not in draft_outer_config.rope_parameters
    assert "mrope_section" not in draft_text_config.rope_parameters
    assert draft_model_config.model_arch_config == "rebuilt"
