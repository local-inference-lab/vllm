# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpeculativeConfig.compose_draft_hf_overrides.

Callable ``hf_overrides`` on the target model config (e.g. the
``dummy_hf_overrides`` shrink used by ``tests/models/test_initialization.py``)
must also be applied when building the draft ``ModelConfig``. Otherwise a
draft belonging to a large target model is instantiated at full size even
when the target itself is shrunk — which is what kept spec-decode archs like
``EagleMistralLarge3ForCausalLM`` stuck at ``is_available_online=False``
("TODO: revert once figuring out OOM in CI").
"""

import functools
from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


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
def test_inkling_override_exposes_only_first_mtp_depth():
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
    assert out.n_predict == 1
    assert out.num_nextn_predict_layers == 8
    assert out.chain_hidden_post_norm is False
    assert out.local_layer_ids == [0, 2, 4]


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


@pytest.mark.cpu_test
def test_mtp_same_model_inherits_target_revisions():
    spec = SimpleNamespace(
        method="mtp",
        model="org/model",
        revision=None,
        code_revision=None,
        target_model_config=SimpleNamespace(
            model="org/model",
            revision="weights-commit",
            code_revision="code-commit",
        ),
    )

    SpeculativeConfig._inherit_target_revision_for_mtp(spec)

    assert spec.revision == "weights-commit"
    assert spec.code_revision == "code-commit"


@pytest.mark.cpu_test
def test_mtp_explicit_draft_revisions_are_preserved():
    spec = SimpleNamespace(
        method="mtp",
        model="org/model",
        revision="draft-weights",
        code_revision="draft-code",
        target_model_config=SimpleNamespace(
            model="org/model",
            revision="target-weights",
            code_revision="target-code",
        ),
    )

    SpeculativeConfig._inherit_target_revision_for_mtp(spec)

    assert spec.revision == "draft-weights"
    assert spec.code_revision == "draft-code"


@pytest.mark.cpu_test
def test_same_model_draft_inherits_smaller_target_position_limit():
    target_text = SimpleNamespace(max_position_embeddings=600_000)
    draft_text = SimpleNamespace(max_position_embeddings=1_000_000)
    spec = SimpleNamespace(
        model="org/model",
        target_model_config=SimpleNamespace(
            model="org/model",
            hf_config=SimpleNamespace(),
            hf_text_config=target_text,
        ),
        draft_model_config=SimpleNamespace(
            model="org/model",
            hf_config=SimpleNamespace(),
            hf_text_config=draft_text,
        ),
    )

    changed = SpeculativeConfig._cap_same_model_draft_position_embeddings(spec)

    assert changed is True
    assert draft_text.max_position_embeddings == 600_000


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("draft_model", "draft_max"),
    [
        ("org/other-draft", 1_000_000),
        ("org/model", 500_000),
    ],
)
def test_draft_position_limit_is_not_increased_or_applied_cross_model(
    draft_model: str,
    draft_max: int,
):
    draft_text = SimpleNamespace(max_position_embeddings=draft_max)
    spec = SimpleNamespace(
        model=draft_model,
        target_model_config=SimpleNamespace(
            model="org/model",
            hf_text_config=SimpleNamespace(max_position_embeddings=600_000),
        ),
        draft_model_config=SimpleNamespace(
            model=draft_model,
            hf_text_config=draft_text,
        ),
    )

    changed = SpeculativeConfig._cap_same_model_draft_position_embeddings(spec)

    assert changed is False
    assert draft_text.max_position_embeddings == draft_max
