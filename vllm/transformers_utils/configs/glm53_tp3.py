# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical TP3 geometry for the GLM-5.3 target and its draft models.

The checkpoint dimensions remain available as ``original_*`` attributes.  The
model implementations use the physical dimensions to allocate TP-sharded
parameters and the original dimensions while loading and producing logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

_GLM53_ARCHITECTURES = {
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTPModel",
}
_DFLASH_ARCHITECTURES = {"DFlashDraftModel", "DFlash2DraftModel"}


def _iter_hf_configs(model_config: ModelConfig):
    seen: set[int] = set()
    hf_config = model_config.hf_config
    for config in (
        hf_config,
        getattr(model_config, "hf_text_config", None),
        getattr(hf_config, "text_config", None),
    ):
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            yield config


def _has_architecture(model_config: ModelConfig, names: set[str]) -> bool:
    for config in _iter_hf_configs(model_config):
        if names.intersection(getattr(config, "architectures", None) or ()):
            return True
    return False


def is_glm53_config(model_config: ModelConfig | None) -> bool:
    if model_config is None:
        return False
    if _has_architecture(model_config, _GLM53_ARCHITECTURES):
        return True
    return any(
        getattr(config, "model_type", None)
        in {"glm5_next", "glm5_next_text", "glm5_next_mtp"}
        for config in _iter_hf_configs(model_config)
    )


def _logical_value(config: Any, name: str) -> int:
    original = getattr(config, f"original_{name}", None)
    return int(original if original is not None else getattr(config, name))


def _require_shape(config: Any, name: str, expected: int) -> None:
    value = _logical_value(config, name)
    if value != expected:
        raise ValueError(
            "GLM-5.3 TP3 padding only supports the released checkpoint "
            f"geometry: expected {name}={expected}, got {value}."
        )


def apply_glm53_tp3_target_geometry(
    model_config: ModelConfig | None,
    parallel_config: ParallelConfig | None,
) -> bool:
    """Apply GLM-5.3's physical TP3 axes using the actual parallel config.

    Returns whether the target is using the TP3 layout.  TP1/2/4 and unrelated
    configurations are exact no-ops.
    """
    if (
        model_config is None
        or parallel_config is None
        or parallel_config.tensor_parallel_size != 3
        or not is_glm53_config(model_config)
    ):
        return False

    text_config = model_config.hf_text_config
    target_shapes = (
        ("num_attention_heads", 64),
        ("num_key_value_heads", 64),
        ("linear_num_heads", 64),
        ("moe_intermediate_size", 2048),
        ("hidden_size", 4096),
        ("vocab_size", 154880),
    )
    for name, expected in target_shapes:
        _require_shape(text_config, name, expected)

    for config in _iter_hf_configs(model_config):
        if config is text_config:
            continue
        for name, expected in target_shapes:
            if hasattr(config, name):
                _require_shape(config, name, expected)

    vision_config = getattr(model_config.hf_config, "vision_config", None)
    multimodal_config = getattr(model_config, "multimodal_config", None)
    uses_weights_mode_vision = (
        vision_config is not None
        and multimodal_config is not None
        and multimodal_config.mm_encoder_tp_mode == "weights"
    )
    if uses_weights_mode_vision:
        for name, expected in (
            ("num_heads", 16),
            ("hidden_size", 1024),
            ("intermediate_size", 4096),
            ("projection_intermediate_size", 10240),
        ):
            _require_shape(vision_config, name, expected)
    for config in _iter_hf_configs(model_config):
        if hasattr(config, "num_attention_heads"):
            config.original_num_attention_heads = 64
            config.num_attention_heads = 72
        if hasattr(config, "num_key_value_heads"):
            config.original_num_key_value_heads = 64
            config.num_key_value_heads = 72
        if hasattr(config, "linear_num_heads"):
            config.original_linear_num_heads = 64
            config.linear_num_heads = 66
        linear_config = getattr(config, "linear_attn_config", None)
        if isinstance(linear_config, dict) and "num_heads" in linear_config:
            config.linear_attn_config = {**linear_config, "num_heads": 66}
        config.glm53_tp3_padding = True
        config.glm53_tp3_shared_expert_intermediate_size = 2112
        config.glm53_tp3_mtp_projection_size = 4098
        config.glm53_tp3_vocab_padding_size = 192
        config.glm53_tp3_vocab_storage_size = 154944

    if uses_weights_mode_vision:
        vision_config.original_num_heads = 16
        vision_config.num_heads = 18
        vision_config.original_intermediate_size = 4096
        vision_config.intermediate_size = 4098
        vision_config.original_projection_intermediate_size = 10240
        vision_config.projection_intermediate_size = 10242
        vision_config.glm53_tp3_attention_projection_size = 1152
        vision_config.glm53_tp3_padding = True

    model_config.model_arch_config = model_config.get_model_arch_config()
    return True


def apply_glm53_tp3_draft_geometry(
    target_model_config: ModelConfig | None,
    target_parallel_config: ParallelConfig | None,
    draft_model_config: ModelConfig | None,
    draft_parallel_config: ParallelConfig | None,
) -> bool:
    """Apply the target's TP3 contract to an MTP or DFlash draft config."""
    if (
        target_model_config is None
        or target_parallel_config is None
        or draft_model_config is None
        or draft_parallel_config is None
        or target_parallel_config.tensor_parallel_size != 3
        or draft_parallel_config.tensor_parallel_size != 3
        or not is_glm53_config(target_model_config)
    ):
        return False

    if is_glm53_config(draft_model_config):
        return apply_glm53_tp3_target_geometry(
            draft_model_config, draft_parallel_config
        )
    if not _has_architecture(draft_model_config, _DFLASH_ARCHITECTURES):
        return False

    text_config = draft_model_config.hf_text_config
    _require_shape(text_config, "num_attention_heads", 32)
    _require_shape(text_config, "num_key_value_heads", 8)
    _require_shape(text_config, "vocab_size", 154880)
    for config in _iter_hf_configs(draft_model_config):
        if hasattr(config, "num_attention_heads"):
            config.original_num_attention_heads = 32
            config.num_attention_heads = 36
        if hasattr(config, "num_key_value_heads"):
            config.original_num_key_value_heads = 8
            config.num_key_value_heads = 9
        if hasattr(config, "vocab_size"):
            config.original_vocab_size = 154880
            config.draft_vocab_size = 154880
        config.glm53_tp3_padding = True
        config.glm53_tp3_vocab_padding_size = 192
        config.glm53_tp3_vocab_storage_size = 154944

    draft_model_config.model_arch_config = draft_model_config.get_model_arch_config()
    return True
