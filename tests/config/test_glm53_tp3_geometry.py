# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.config import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.glm53_tp3 import (
    apply_glm53_tp3_draft_geometry,
    apply_glm53_tp3_target_geometry,
)


class FakeGlm53ModelConfig:
    """Device-free stand-in for the released GLM-5.3 checkpoint config."""

    def __init__(
        self,
        *,
        architecture: str = "Glm5NextForConditionalGeneration",
        model_type: str = "glm5_next",
        mm_encoder_tp_mode: str = "weights",
    ) -> None:
        self.hf_text_config = SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
            hidden_size=4096,
            num_attention_heads=64,
            num_key_value_heads=64,
            linear_num_heads=64,
            linear_attn_config={"num_heads": 64},
            moe_intermediate_size=2048,
            n_routed_experts=288,
            n_shared_experts=1,
            vocab_size=154880,
        )
        self.hf_config = SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
            text_config=self.hf_text_config,
            vision_config=SimpleNamespace(
                hidden_size=1024,
                num_heads=16,
                intermediate_size=4096,
                projection_intermediate_size=10240,
            ),
        )
        self.multimodal_config = SimpleNamespace(
            mm_encoder_tp_mode=mm_encoder_tp_mode
        )
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
            total_num_kv_heads=self.hf_text_config.num_key_value_heads,
            vocab_size=self.hf_text_config.vocab_size,
        )


class FakeDFlashModelConfig:
    """Device-free stand-in for the released dense DFlash checkpoint config."""

    def __init__(self) -> None:
        self.hf_text_config = SimpleNamespace(
            model_type="qwen3",
            architectures=["DFlash2DraftModel"],
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=154880,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
            total_num_kv_heads=self.hf_text_config.num_key_value_heads,
            vocab_size=self.hf_text_config.vocab_size,
        )


@pytest.fixture
def glm53_model_config() -> FakeGlm53ModelConfig:
    return FakeGlm53ModelConfig()


@pytest.fixture
def tp3_ep_parallel_config() -> ParallelConfig:
    return ParallelConfig(tensor_parallel_size=3, enable_expert_parallel=True)


def _snapshot(model_config: object) -> dict[str, Any]:
    return deepcopy(vars(model_config))


def test_glm53_tp3_target_geometry_uses_parallel_config_and_preserves_logical_axes(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
) -> None:
    applied = apply_glm53_tp3_target_geometry(
        cast(Any, glm53_model_config), tp3_ep_parallel_config
    )

    assert applied
    text_config = glm53_model_config.hf_text_config
    assert (
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.linear_num_heads,
        text_config.linear_attn_config["num_heads"],
    ) == (72, 72, 66, 66)
    assert (
        text_config.original_num_attention_heads,
        text_config.original_num_key_value_heads,
        text_config.original_linear_num_heads,
    ) == (64, 64, 64)

    # Routed experts keep the checkpoint width under EP. Only the replicated
    # shared expert gets a physical TP3 storage width.
    assert tp3_ep_parallel_config.enable_expert_parallel
    assert text_config.moe_intermediate_size == 2048
    assert text_config.n_routed_experts == 288
    assert text_config.n_shared_experts == 1
    assert text_config.glm53_tp3_shared_expert_intermediate_size == 2112

    # The tokenizer/logits contract remains the checkpoint vocabulary while
    # parameters are allocated using the explicitly recorded storage size.
    assert text_config.vocab_size == 154880
    assert text_config.glm53_tp3_vocab_padding_size == 192
    assert text_config.glm53_tp3_vocab_storage_size == 154944
    assert text_config.glm53_tp3_mtp_projection_size == 4098
    assert text_config.glm53_tp3_padding is True

    wrapper_config = glm53_model_config.hf_config
    assert wrapper_config.glm53_tp3_shared_expert_intermediate_size == 2112
    assert wrapper_config.glm53_tp3_vocab_storage_size == 154944
    assert wrapper_config.glm53_tp3_padding is True
    assert glm53_model_config.model_arch_config.total_num_attention_heads == 72
    assert glm53_model_config.model_arch_config.total_num_kv_heads == 72
    assert glm53_model_config.model_arch_config.vocab_size == 154880


def test_glm53_tp3_weights_mode_records_exact_vision_storage_geometry(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
) -> None:
    assert apply_glm53_tp3_target_geometry(
        cast(Any, glm53_model_config), tp3_ep_parallel_config
    )

    vision_config = glm53_model_config.hf_config.vision_config
    assert (
        vision_config.num_heads,
        vision_config.original_num_heads,
    ) == (18, 16)
    assert vision_config.hidden_size == 1024
    assert vision_config.glm53_tp3_attention_projection_size == 1152
    assert (
        vision_config.intermediate_size,
        vision_config.original_intermediate_size,
    ) == (4098, 4096)
    assert (
        vision_config.projection_intermediate_size,
        vision_config.original_projection_intermediate_size,
    ) == (10242, 10240)
    assert vision_config.glm53_tp3_padding is True


def test_glm53_tp3_target_geometry_is_idempotent(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
) -> None:
    assert apply_glm53_tp3_target_geometry(
        cast(Any, glm53_model_config), tp3_ep_parallel_config
    )
    once = _snapshot(glm53_model_config)

    assert apply_glm53_tp3_target_geometry(
        cast(Any, glm53_model_config), tp3_ep_parallel_config
    )

    assert _snapshot(glm53_model_config) == once


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("num_attention_heads", 63),
        ("num_key_value_heads", 63),
        ("linear_num_heads", 63),
        ("moe_intermediate_size", 2047),
        ("hidden_size", 4095),
        ("vocab_size", 154879),
    ],
)
def test_glm53_tp3_invalid_target_checkpoint_geometry_fails_closed(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
    attribute: str,
    invalid_value: int,
) -> None:
    setattr(glm53_model_config.hf_text_config, attribute, invalid_value)
    before = _snapshot(glm53_model_config)

    with pytest.raises(ValueError, match=rf"expected {attribute}="):
        apply_glm53_tp3_target_geometry(
            cast(Any, glm53_model_config), tp3_ep_parallel_config
        )

    assert _snapshot(glm53_model_config) == before


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("num_heads", 15),
        ("hidden_size", 1023),
        ("intermediate_size", 4095),
        ("projection_intermediate_size", 10239),
    ],
)
def test_glm53_tp3_invalid_vision_geometry_is_transactional(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
    attribute: str,
    invalid_value: int,
) -> None:
    vision_config = glm53_model_config.hf_config.vision_config
    setattr(vision_config, attribute, invalid_value)
    before = _snapshot(glm53_model_config)

    with pytest.raises(ValueError, match=rf"expected {attribute}="):
        apply_glm53_tp3_target_geometry(
            cast(Any, glm53_model_config), tp3_ep_parallel_config
        )

    assert _snapshot(glm53_model_config) == before
    assert not hasattr(glm53_model_config.hf_text_config, "glm53_tp3_padding")
    assert not hasattr(glm53_model_config.hf_config, "glm53_tp3_padding")
    assert not hasattr(vision_config, "glm53_tp3_padding")


def test_glm53_tp3_invalid_wrapper_geometry_is_transactional(
    glm53_model_config: FakeGlm53ModelConfig,
    tp3_ep_parallel_config: ParallelConfig,
) -> None:
    glm53_model_config.hf_config.hidden_size = 4095
    before = _snapshot(glm53_model_config)

    with pytest.raises(ValueError, match=r"expected hidden_size=4096"):
        apply_glm53_tp3_target_geometry(
            cast(Any, glm53_model_config), tp3_ep_parallel_config
        )

    assert _snapshot(glm53_model_config) == before
    assert not hasattr(glm53_model_config.hf_text_config, "glm53_tp3_padding")
    assert not hasattr(glm53_model_config.hf_config, "glm53_tp3_padding")
    assert not hasattr(
        glm53_model_config.hf_config.vision_config, "glm53_tp3_padding"
    )


def test_glm53_tp4_parallel_config_is_an_exact_attribute_noop(
    glm53_model_config: FakeGlm53ModelConfig,
) -> None:
    parallel_config = ParallelConfig(tensor_parallel_size=4)
    before = _snapshot(glm53_model_config)

    assert not apply_glm53_tp3_target_geometry(
        cast(Any, glm53_model_config), parallel_config
    )

    assert parallel_config.tensor_parallel_size == 4
    assert _snapshot(glm53_model_config) == before
    assert not hasattr(glm53_model_config.hf_text_config, "glm53_tp3_padding")


def test_unrelated_tp3_model_is_an_exact_noop() -> None:
    model_config = FakeGlm53ModelConfig(
        architecture="LlamaForCausalLM", model_type="llama"
    )
    parallel_config = ParallelConfig(tensor_parallel_size=3)
    before = _snapshot(model_config)

    assert not apply_glm53_tp3_target_geometry(
        cast(Any, model_config), parallel_config
    )

    assert _snapshot(model_config) == before
    assert not hasattr(model_config.hf_text_config, "glm53_tp3_padding")


def test_glm53_tp3_mtp_draft_preserves_expert_parallel_topology() -> None:
    target_model_config = FakeGlm53ModelConfig()
    draft_model_config = FakeGlm53ModelConfig(
        architecture="Glm5NextMTPModel", model_type="glm5_next_mtp"
    )
    target_parallel_config = ParallelConfig(
        tensor_parallel_size=3,
        data_parallel_size=2,
        data_parallel_size_local=2,
        enable_expert_parallel=True,
    )
    draft_parallel_config = ParallelConfig(tensor_parallel_size=3)
    speculative_config = SimpleNamespace(
        method="mtp",
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
        draft_model_config=draft_model_config,
        draft_parallel_config=draft_parallel_config,
    )

    SpeculativeConfig._apply_glm53_tp3_draft_geometry(
        cast(Any, speculative_config)
    )

    assert draft_parallel_config.tensor_parallel_size == 3
    assert draft_parallel_config.data_parallel_size == 2
    assert draft_parallel_config.data_parallel_size_local == 2
    assert draft_parallel_config.enable_expert_parallel
    draft_text_config = draft_model_config.hf_text_config
    assert draft_text_config.num_attention_heads == 72
    assert draft_text_config.linear_num_heads == 66
    assert draft_text_config.moe_intermediate_size == 2048
    assert draft_text_config.glm53_tp3_shared_expert_intermediate_size == 2112


def test_glm53_tp3_dflash_drops_ep_and_couples_heads_with_vocab_storage() -> None:
    target_model_config = FakeGlm53ModelConfig()
    draft_model_config = FakeDFlashModelConfig()
    target_parallel_config = ParallelConfig(
        tensor_parallel_size=3,
        data_parallel_size=2,
        data_parallel_size_local=2,
        enable_expert_parallel=True,
    )
    draft_parallel_config = ParallelConfig(
        tensor_parallel_size=3, enable_expert_parallel=True
    )
    speculative_config = SimpleNamespace(
        method="dflash",
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
        draft_model_config=draft_model_config,
        draft_parallel_config=draft_parallel_config,
    )

    SpeculativeConfig._apply_glm53_tp3_draft_geometry(
        cast(Any, speculative_config)
    )

    assert draft_parallel_config.tensor_parallel_size == 3
    assert draft_parallel_config.data_parallel_size == 2
    assert draft_parallel_config.data_parallel_size_local == 2
    assert not draft_parallel_config.enable_expert_parallel
    draft_config = draft_model_config.hf_text_config
    assert (
        draft_config.num_attention_heads,
        draft_config.num_key_value_heads,
    ) == (36, 9)
    assert (
        draft_config.original_num_attention_heads,
        draft_config.original_num_key_value_heads,
    ) == (32, 8)
    assert draft_config.vocab_size == 154880
    assert draft_config.original_vocab_size == 154880
    assert draft_config.draft_vocab_size == 154880
    assert draft_config.glm53_tp3_vocab_padding_size == 192
    assert draft_config.glm53_tp3_vocab_storage_size == 154944
    assert draft_model_config.model_arch_config.total_num_attention_heads == 36
    assert draft_model_config.model_arch_config.total_num_kv_heads == 9
    assert draft_model_config.model_arch_config.vocab_size == 154880


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("num_attention_heads", 31),
        ("num_key_value_heads", 7),
        ("vocab_size", 154879),
    ],
)
def test_glm53_tp3_invalid_dflash_checkpoint_geometry_fails_closed(
    attribute: str,
    invalid_value: int,
) -> None:
    target_model_config = FakeGlm53ModelConfig()
    draft_model_config = FakeDFlashModelConfig()
    target_parallel_config = ParallelConfig(tensor_parallel_size=3)
    draft_parallel_config = ParallelConfig(tensor_parallel_size=3)
    setattr(draft_model_config.hf_text_config, attribute, invalid_value)
    before = _snapshot(draft_model_config)

    with pytest.raises(ValueError, match=rf"expected {attribute}="):
        apply_glm53_tp3_draft_geometry(
            cast(Any, target_model_config),
            target_parallel_config,
            cast(Any, draft_model_config),
            draft_parallel_config,
        )

    assert _snapshot(draft_model_config) == before


def test_glm53_dflash_tp4_parallel_configs_are_an_exact_noop() -> None:
    target_model_config = FakeGlm53ModelConfig()
    draft_model_config = FakeDFlashModelConfig()
    target_parallel_config = ParallelConfig(tensor_parallel_size=4)
    draft_parallel_config = ParallelConfig(tensor_parallel_size=4)
    before = _snapshot(draft_model_config)

    assert not apply_glm53_tp3_draft_geometry(
        cast(Any, target_model_config),
        target_parallel_config,
        cast(Any, draft_model_config),
        draft_parallel_config,
    )

    assert target_parallel_config.tensor_parallel_size == 4
    assert draft_parallel_config.tensor_parallel_size == 4
    assert _snapshot(draft_model_config) == before
    assert not hasattr(draft_model_config.hf_text_config, "glm53_tp3_padding")
