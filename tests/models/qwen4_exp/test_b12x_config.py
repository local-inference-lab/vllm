# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.8-Flash-Next configuration plumbing."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config import AttentionConfig, VllmConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
)
from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)
from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY, get_config
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
    [
        (None, "bfloat16"),
        ("bfloat16", "bfloat16"),
        ("float8_e4m3fn", "float8_e4m3fn"),
        ("nvfp4", "nvfp4"),
    ],
)
@pytest.mark.parametrize("config_cls", [Qwen3_8FlashNextTextConfig, Qwen4ExpTextConfig])
def test_ple_embedding_storage_dtype_is_preserved(
    configured_dtype: str | None, expected_dtype: str, config_cls
) -> None:
    config = config_cls(**_TEXT_CONFIG, ple_embedding_dtype=configured_dtype)

    assert config.ple_embedding_dtype == expected_dtype


@pytest.mark.parametrize(
    "config_cls",
    [
        Qwen4ExpConfig,
        Qwen4ExpTextConfig,
        Qwen3_8FlashNextConfig,
        Qwen3_8FlashNextTextConfig,
    ],
)
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.bfloat16, "bfloat16"),
        (torch.float8_e4m3fn, "float8_e4m3fn"),
        (torch.uint8, "nvfp4"),
    ],
)
def test_omitted_ple_dtype_is_resolved_from_checkpoint_headers(
    tmp_path, config_cls, dtype, expected
) -> None:
    config_dict = config_cls(**_TEXT_CONFIG, ple_layer_ids=[2]).to_dict()
    config_dict.get("text_config", config_dict).pop("ple_embedding_dtype")
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    save_file(
        {
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight": torch.zeros((2, 8), dtype=dtype),
        },
        tmp_path / "model.safetensors",
    )

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert config.get_text_config().ple_embedding_dtype == expected


def test_explicit_ple_dtype_does_not_probe_checkpoint(tmp_path, monkeypatch) -> None:
    config = Qwen4ExpConfig(
        **_TEXT_CONFIG, ple_layer_ids=[2], ple_embedding_dtype="bfloat16"
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_safetensors_params_metadata",
        lambda *args, **kwargs: pytest.fail("explicit PLE dtype must take precedence"),
    )

    restored = get_config(str(tmp_path), trust_remote_code=False)

    assert restored.get_text_config().ple_embedding_dtype == "bfloat16"


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


def test_model_registry_packages() -> None:
    assert _TEXT_GENERATION_MODELS["Qwen4ExpForCausalLM"] == (
        "vllm.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
    )
    assert _MULTIMODAL_MODELS["Qwen4ExpForConditionalGeneration"] == (
        "vllm.models.qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
    )
    assert _SPECULATIVE_DECODING_MODELS["Qwen3_8FlashNextMTP"] == (
        "vllm.models.qwen4_exp",
        "Qwen4ExpMTP",
    )
    for registry, suffix in (
        (_TEXT_GENERATION_MODELS, "ForCausalLM"),
        (_MULTIMODAL_MODELS, "ForConditionalGeneration"),
        (_SPECULATIVE_DECODING_MODELS, "MTP"),
    ):
        assert registry[f"Qwen3_8FlashNext{suffix}"] == registry[f"Qwen4Exp{suffix}"]


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    ],
)
def test_model_registry_architectures_default_to_v2(monkeypatch, architecture) -> None:
    import vllm.config.vllm as config_module
    from vllm.platforms import current_platform

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(config_module, "HAS_TRITON", True)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        attention_config=AttentionConfig(),
        _get_v2_model_runner_unsupported_features=lambda: [],
    )
    assert VllmConfig.use_v2_model_runner.fget(config)


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

    assert config.model_type == "qwen4_exp_mtp"
    assert config.architectures == ["Qwen4ExpMTP"]
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
