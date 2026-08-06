# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig

logger = init_logger(__name__)

VIRTUAL_TP_PLAN_ATTR = "vllm_virtual_tp_plan"
VIRTUAL_TP_PROFILE_ATTR = "vllm_virtual_tp_profile"
_VIRTUAL_TP_PLAN_KIND_B12X_PADDED = "b12x-padded"
_GQA_GDN_MOE_PROFILE = "gqa-gdn-moe"
_DENSE_GQA_PROFILE = "dense-gqa"
_ATTENTION_HEAD_LOCAL_ALIGNMENT = 8
_MOE_INTERMEDIATE_LOCAL_ALIGNMENT = 32
_DENSE_GQA_INTERMEDIATE_LOCAL_ALIGNMENT = 32
_NVFP4_LOCAL_ALIGNMENT = 16
_SHARED_EXPERT_FP8_LOCAL_ALIGNMENT = 128
_VOCAB_GLOBAL_ALIGNMENT = 64
_MINIMAX_M3_VIRTUAL_TP_SIZE = 3
_MINIMAX_M3_QK_PER_KV = 16


def maybe_apply_b12x_virtual_tp_padding(vllm_config: VllmConfig) -> None:
    """Automatically pad config dimensions for B12X virtual TP sharding.

    Some B12X target models have dimensions that are not divisible by an
    otherwise useful TP size.  Native B12X kernels can run a larger logical
    per-rank shape as long as checkpoint tails are zero-filled during loading.
    This mutates the HuggingFace configs before vLLM's normal parallel-config
    verification and stores the original sizes in ``VIRTUAL_TP_PLAN_ATTR`` for
    weight loaders.
    """
    model_config = vllm_config.model_config
    if model_config is None:
        return

    plan_config = _get_plan_config(model_config)
    if getattr(plan_config, VIRTUAL_TP_PLAN_ATTR, None) is not None:
        return

    if not _is_supported_b12x_virtual_tp_config(model_config):
        return
    if not (_uses_b12x_attention(vllm_config) or _uses_native_b12x_moe(vllm_config)):
        return

    plan = _build_b12x_virtual_tp_plan(model_config, vllm_config.parallel_config)
    if not _plan_requires_padding(plan):
        return

    _validate_b12x_virtual_tp_config(vllm_config)

    _apply_b12x_virtual_tp_plan(model_config, plan)


def apply_b12x_virtual_tp_padding_to_model_config(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> None:
    """Pad model dimensions when B12X virtual TP alignment requires it."""
    plan_config = _get_plan_config(model_config)
    if getattr(plan_config, VIRTUAL_TP_PLAN_ATTR, None) is not None:
        return

    if not _is_supported_b12x_virtual_tp_config(model_config):
        return

    plan = _build_b12x_virtual_tp_plan(model_config, parallel_config)
    if not _plan_requires_padding(plan):
        return

    _apply_b12x_virtual_tp_plan(model_config, plan)


def _build_b12x_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    if _is_minimax_m3_config(model_config):
        return _build_minimax_m3_virtual_tp_plan(model_config, parallel_config)
    if _get_virtual_tp_profile(model_config) == _GQA_GDN_MOE_PROFILE:
        return _build_gqa_gdn_moe_virtual_tp_plan(model_config, parallel_config)
    if _get_virtual_tp_profile(model_config) == _DENSE_GQA_PROFILE:
        return _build_dense_gqa_virtual_tp_plan(model_config, parallel_config)

    attention_tp_size = parallel_config.tensor_parallel_size
    moe_tp_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    text_config = model_config.hf_text_config
    is_deepseek_v4 = _is_deepseek_v4_config(model_config)

    original_attention_heads = _require_int_attr(text_config, "num_attention_heads")
    # DeepSeek V4 carries output-group constraints tied to padded head count.
    # GLM/DSA only needs divisibility by TP at the model-config level. B12X
    # sparse MLA handles partial local head blocks by padding/slicing inside the
    # backend, so GLM TP6 can stay 64->66 instead of inflating to 96 heads.
    attention_head_alignment = _ATTENTION_HEAD_LOCAL_ALIGNMENT if is_deepseek_v4 else 1
    attention_axis = _make_virtual_axis(
        original_attention_heads,
        attention_tp_size,
        attention_head_alignment,
    )
    if is_deepseek_v4:
        original_output_groups = _require_int_attr(text_config, "o_groups")
        output_group_axis = _make_virtual_output_group_axis(
            original_output_groups,
            original_attention_heads,
            attention_axis["padded_size"],
            attention_tp_size,
        )
    else:
        output_group_axis = None

    moe_original_size = _require_int_attr(text_config, "moe_intermediate_size")
    moe_axis = _make_virtual_axis(
        moe_original_size,
        moe_tp_size,
        _get_moe_intermediate_local_alignment(model_config),
    )
    shared_expert_axis = None
    n_shared_experts = getattr(text_config, "n_shared_experts", None)
    if is_deepseek_v4 and n_shared_experts is not None:
        shared_expert_axis = _make_virtual_axis(
            moe_original_size * int(n_shared_experts),
            attention_tp_size,
            _SHARED_EXPERT_FP8_LOCAL_ALIGNMENT,
        )
    vocab_axis = _make_virtual_vocab_axis(
        _require_int_attr(text_config, "vocab_size"),
        attention_tp_size,
    )

    plan: dict[str, dict[str, int] | str] = {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "attention_heads": attention_axis,
        "moe_intermediate_size": moe_axis,
        "vocab_size": vocab_axis,
    }
    if output_group_axis is not None:
        plan["output_groups"] = output_group_axis
    if shared_expert_axis is not None:
        plan["shared_expert_intermediate_size"] = shared_expert_axis
    return plan


def _build_gqa_gdn_moe_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    """Build a plan for models combining GQA, GDN, and routed experts.

    Models opt into this reusable shape profile through
    ``VIRTUAL_TP_PROFILE_ATTR``. The profile is expressed in terms of common
    config attributes rather than architecture names, so compatible model
    families can reuse the same planner and loading paths.
    """
    text_config = model_config.hf_text_config
    tp_size = parallel_config.tensor_parallel_size
    moe_tp_size = (
        tp_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    attention_axis, kv_axis = _make_coupled_virtual_axes(
        _require_int_attr(text_config, "num_attention_heads"),
        _require_int_attr(text_config, "num_key_value_heads"),
        tp_size,
        ratio_key="q_heads_per_kv",
        allow_secondary_replication=True,
    )
    linear_value_axis, linear_key_axis = _make_coupled_virtual_axes(
        _require_int_attr(text_config, "linear_num_value_heads"),
        _require_int_attr(text_config, "linear_num_key_heads"),
        tp_size,
        ratio_key="value_heads_per_key",
    )
    moe_axis = _make_virtual_axis(
        _require_int_attr(text_config, "moe_intermediate_size"),
        moe_tp_size,
        _get_moe_intermediate_local_alignment(model_config),
    )
    dense_linear_alignment = _get_dense_linear_local_alignment(model_config)
    shared_expert_axis = _make_virtual_axis(
        _require_int_attr(text_config, "shared_expert_intermediate_size"),
        tp_size,
        dense_linear_alignment,
    )
    dense_intermediate_axis = None
    if _positive_int_attr(text_config, "intermediate_size"):
        dense_intermediate_axis = _make_virtual_axis(
            _require_int_attr(text_config, "intermediate_size"),
            tp_size,
            dense_linear_alignment,
        )
    mtp_projection_axis = _make_virtual_axis(
        _require_int_attr(text_config, "hidden_size"),
        tp_size,
    )

    plan: dict[str, dict[str, int] | str] = {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "model_type": _GQA_GDN_MOE_PROFILE,
        "attention_heads": attention_axis,
        "kv_heads": kv_axis,
        "linear_attention_key_heads": linear_key_axis,
        "linear_attention_value_heads": linear_value_axis,
        "moe_intermediate_size": moe_axis,
        "shared_expert_intermediate_size": shared_expert_axis,
        "mtp_projection_size": mtp_projection_axis,
        "vocab_size": _make_virtual_vocab_axis(
            _require_int_attr(text_config, "vocab_size"), tp_size
        ),
    }
    if dense_intermediate_axis is not None:
        plan["dense_intermediate_size"] = dense_intermediate_axis

    vision_config = getattr(model_config.hf_config, "vision_config", None)
    multimodal_config = getattr(model_config, "multimodal_config", None)
    vision_tp_size = (
        1
        if multimodal_config is not None
        and getattr(multimodal_config, "mm_encoder_tp_mode", None) == "data"
        else tp_size
    )
    if vision_config is not None:
        vision_heads = _require_int_attr(vision_config, "num_heads")
        vision_hidden_size = _require_int_attr(vision_config, "hidden_size")
        if vision_hidden_size % vision_heads != 0:
            raise ValueError(
                "B12X virtual TP padding requires the vision hidden size to "
                "be divisible by its attention head count."
            )
        vision_head_axis = _make_virtual_axis(vision_heads, vision_tp_size)
        vision_projection_axis = _make_scaled_virtual_axis(
            vision_hidden_size,
            vision_head_axis,
            vision_hidden_size // vision_heads,
            "head_size",
        )
        plan["vision_attention_heads"] = vision_head_axis
        plan["vision_attention_projection_size"] = vision_projection_axis
        plan["vision_intermediate_size"] = _make_virtual_axis(
            _require_int_attr(vision_config, "intermediate_size"),
            vision_tp_size,
            dense_linear_alignment,
        )

    return plan


def _build_dense_gqa_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    """Build zero-tail padding for a dense GQA transformer draft."""
    text_config = model_config.hf_text_config
    tp_size = parallel_config.tensor_parallel_size
    attention_axis, kv_axis = _make_coupled_virtual_axes(
        _require_int_attr(text_config, "num_attention_heads"),
        _require_int_attr(text_config, "num_key_value_heads"),
        tp_size,
        ratio_key="q_heads_per_kv",
        allow_secondary_replication=True,
    )
    intermediate_axis = _make_virtual_axis(
        _require_int_attr(text_config, "intermediate_size"),
        tp_size,
        _DENSE_GQA_INTERMEDIATE_LOCAL_ALIGNMENT,
    )
    return {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "model_type": _DENSE_GQA_PROFILE,
        "attention_heads": attention_axis,
        "kv_heads": kv_axis,
        "dense_intermediate_size": intermediate_axis,
        "vocab_size": _make_virtual_vocab_axis(
            _require_int_attr(text_config, "vocab_size"), tp_size
        ),
    }


def _build_minimax_m3_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    text_config = model_config.hf_text_config
    tp_size = parallel_config.tensor_parallel_size

    original_attention_heads = _require_int_attr(text_config, "num_attention_heads")
    original_kv_heads = _require_int_attr(text_config, "num_key_value_heads")
    sparse_config = getattr(text_config, "sparse_attention_config", None) or {}
    original_index_heads = int(
        sparse_config.get("sparse_num_index_heads", original_kv_heads)
    )
    intermediate_size = _require_int_attr(text_config, "intermediate_size")
    dense_intermediate_size = _require_int_attr(text_config, "dense_intermediate_size")
    n_shared_experts = int(getattr(text_config, "n_shared_experts", 1) or 1)
    vocab_size = _require_int_attr(text_config, "vocab_size")

    if tp_size != _MINIMAX_M3_VIRTUAL_TP_SIZE:
        q_heads_per_kv = (
            original_attention_heads // original_kv_heads
            if original_kv_heads > 0
            and original_attention_heads % original_kv_heads == 0
            else 0
        )
        return {
            "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
            "model_type": "minimax_m3",
            "attention_heads": _make_noop_axis(original_attention_heads, tp_size),
            "kv_heads": {
                **_make_noop_axis(original_kv_heads, tp_size),
                "q_heads_per_kv": q_heads_per_kv,
            },
            "index_heads": _make_noop_axis(original_index_heads, tp_size),
            "moe_intermediate_size": _make_noop_axis(intermediate_size, tp_size),
            "dense_intermediate_size": _make_noop_axis(
                dense_intermediate_size, tp_size
            ),
            "shared_expert_intermediate_size": _make_noop_axis(
                intermediate_size * n_shared_experts, tp_size
            ),
            "vocab_size": _make_noop_vocab_axis(vocab_size, tp_size),
        }

    if original_attention_heads % original_kv_heads != 0:
        raise ValueError(
            "MiniMax M3 virtual TP padding requires num_attention_heads to "
            "be divisible by num_key_value_heads."
        )
    q_heads_per_kv = original_attention_heads // original_kv_heads
    if q_heads_per_kv != _MINIMAX_M3_QK_PER_KV:
        raise ValueError(
            "MiniMax M3 virtual TP padding currently expects 16 query heads "
            f"per KV head, got {q_heads_per_kv}."
        )

    attention_axis, kv_axis = _make_coupled_virtual_axes(
        original_attention_heads,
        original_kv_heads,
        tp_size,
        primary_local_alignment=_ATTENTION_HEAD_LOCAL_ALIGNMENT,
        ratio_key="q_heads_per_kv",
    )
    index_axis = {
        "original_size": original_index_heads,
        "padded_size": kv_axis["padded_size"],
        "tp_size": tp_size,
        "local_size": kv_axis["local_size"],
    }

    moe_tp_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    moe_axis = _make_virtual_axis(
        intermediate_size,
        moe_tp_size,
        _get_moe_intermediate_local_alignment(model_config),
    )
    dense_axis = _make_virtual_axis(
        dense_intermediate_size,
        tp_size,
        _SHARED_EXPERT_FP8_LOCAL_ALIGNMENT,
    )
    shared_expert_axis = _make_virtual_axis(
        intermediate_size * n_shared_experts,
        tp_size,
        _SHARED_EXPERT_FP8_LOCAL_ALIGNMENT,
    )
    vocab_axis = _make_virtual_vocab_axis(
        vocab_size,
        tp_size,
    )

    return {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "model_type": "minimax_m3",
        "attention_heads": attention_axis,
        "kv_heads": kv_axis,
        "index_heads": index_axis,
        "moe_intermediate_size": moe_axis,
        "dense_intermediate_size": dense_axis,
        "shared_expert_intermediate_size": shared_expert_axis,
        "vocab_size": vocab_axis,
    }


def _plan_requires_padding(plan: dict[str, dict[str, int] | str]) -> bool:
    for axis in plan.values():
        if not isinstance(axis, dict):
            continue
        original_size = axis.get("original_size")
        padded_size = axis.get("padded_size")
        if (
            original_size is not None
            and padded_size is not None
            and int(original_size) != int(padded_size)
        ):
            return True
    return False


def _apply_b12x_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    if plan.get("model_type") == "minimax_m3":
        _apply_minimax_m3_virtual_tp_plan(model_config, plan)
        return
    if plan.get("model_type") == _GQA_GDN_MOE_PROFILE:
        _apply_gqa_gdn_moe_virtual_tp_plan(model_config, plan)
        return
    if plan.get("model_type") == _DENSE_GQA_PROFILE:
        _apply_dense_gqa_virtual_tp_plan(model_config, plan)
        return

    configs = tuple(_iter_virtual_tp_configs(model_config))
    attention_axis = _require_axis(plan, "attention_heads")
    moe_axis = _require_axis(plan, "moe_intermediate_size")
    vocab_axis = _require_axis(plan, "vocab_size")
    output_group_axis = _optional_axis(plan, "output_groups")
    shared_expert_axis = _optional_axis(plan, "shared_expert_intermediate_size")

    _set_all_config_attr(
        configs, "original_num_attention_heads", attention_axis["original_size"]
    )
    _set_existing_config_attr(
        configs, "num_attention_heads", attention_axis["padded_size"]
    )

    if output_group_axis is not None:
        _set_all_config_attr(
            configs, "original_o_groups", output_group_axis["original_size"]
        )
        _set_existing_config_attr(configs, "o_groups", output_group_axis["padded_size"])
    _set_all_config_attr(
        configs, "original_moe_intermediate_size", moe_axis["original_size"]
    )
    _set_existing_config_attr(configs, "moe_intermediate_size", moe_axis["padded_size"])

    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    if output_group_axis is None:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for B12X kernel "
            "compatibility: attention heads %d -> %d, MoE intermediate "
            "size %d -> %d, vocab size %d -> %d.",
            attention_axis["original_size"],
            attention_axis["padded_size"],
            moe_axis["original_size"],
            moe_axis["padded_size"],
            vocab_axis["original_size"],
            vocab_axis["padded_size"],
        )
    else:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for B12X kernel "
            "compatibility: attention heads %d -> %d, output groups %d -> %d, "
            "MoE intermediate size %d -> %d, vocab size %d -> %d.",
            attention_axis["original_size"],
            attention_axis["padded_size"],
            output_group_axis["original_size"],
            output_group_axis["padded_size"],
            moe_axis["original_size"],
            moe_axis["padded_size"],
            vocab_axis["original_size"],
            vocab_axis["padded_size"],
        )
    if shared_expert_axis is not None:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for shared experts: "
            "intermediate size %d -> %d.",
            shared_expert_axis["original_size"],
            shared_expert_axis["padded_size"],
        )


def _apply_gqa_gdn_moe_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    configs = tuple(_iter_virtual_tp_configs(model_config))
    config_axes = (
        ("attention_heads", "num_attention_heads"),
        ("kv_heads", "num_key_value_heads"),
        ("linear_attention_key_heads", "linear_num_key_heads"),
        ("linear_attention_value_heads", "linear_num_value_heads"),
        ("moe_intermediate_size", "moe_intermediate_size"),
        ("shared_expert_intermediate_size", "shared_expert_intermediate_size"),
    )
    for axis_name, attr in config_axes:
        _apply_virtual_axis_to_config_attr(configs, plan, axis_name, attr)
    if _optional_axis(plan, "dense_intermediate_size") is not None:
        _apply_virtual_axis_to_config_attr(
            configs, plan, "dense_intermediate_size", "intermediate_size"
        )

    vision_config = getattr(model_config.hf_config, "vision_config", None)
    if vision_config is not None:
        _apply_virtual_axis_to_config_attr(
            (vision_config,), plan, "vision_attention_heads", "num_heads"
        )
        _apply_virtual_axis_to_config_attr(
            (vision_config,), plan, "vision_intermediate_size", "intermediate_size"
        )
        setattr(vision_config, VIRTUAL_TP_PLAN_ATTR, plan)

    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    changes = []
    for axis_name, label in (
        ("attention_heads", "attention heads"),
        ("kv_heads", "KV heads"),
        ("linear_attention_key_heads", "GDN key heads"),
        ("linear_attention_value_heads", "GDN value heads"),
        ("moe_intermediate_size", "MoE intermediate size"),
        ("shared_expert_intermediate_size", "shared intermediate size"),
        ("dense_intermediate_size", "dense intermediate size"),
        ("mtp_projection_size", "MTP projection size"),
        ("vocab_size", "vocab storage size"),
        ("vision_attention_heads", "vision attention heads"),
        ("vision_intermediate_size", "vision intermediate size"),
    ):
        axis = _optional_axis(plan, axis_name)
        if axis is not None and axis["original_size"] != axis["padded_size"]:
            changes.append(f"{label} {axis['original_size']} -> {axis['padded_size']}")
    logger.warning(
        "Automatically enabled B12X virtual TP padding for the %s profile: %s.",
        _GQA_GDN_MOE_PROFILE,
        ", ".join(changes),
    )


def _apply_dense_gqa_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    configs = tuple(_iter_virtual_tp_configs(model_config))
    for axis_name, attr in (
        ("attention_heads", "num_attention_heads"),
        ("kv_heads", "num_key_value_heads"),
        ("dense_intermediate_size", "intermediate_size"),
    ):
        _apply_virtual_axis_to_config_attr(configs, plan, axis_name, attr)
    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()
    attention_axis = _require_axis(plan, "attention_heads")
    kv_axis = _require_axis(plan, "kv_heads")
    intermediate_axis = _require_axis(plan, "dense_intermediate_size")
    vocab_axis = _require_axis(plan, "vocab_size")
    logger.warning(
        "Enabled virtual TP padding for the %s profile: attention heads "
        "%d -> %d, KV heads %d -> %d, intermediate size %d -> %d, "
        "vocab storage size %d -> %d.",
        _DENSE_GQA_PROFILE,
        attention_axis["original_size"],
        attention_axis["padded_size"],
        kv_axis["original_size"],
        kv_axis["padded_size"],
        intermediate_axis["original_size"],
        intermediate_axis["padded_size"],
        vocab_axis["original_size"],
        vocab_axis["padded_size"],
    )


def _apply_minimax_m3_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    configs = tuple(_iter_virtual_tp_configs(model_config))
    attention_axis = _require_axis(plan, "attention_heads")
    kv_axis = _require_axis(plan, "kv_heads")
    moe_axis = _require_axis(plan, "moe_intermediate_size")
    dense_axis = _require_axis(plan, "dense_intermediate_size")
    vocab_axis = _require_axis(plan, "vocab_size")
    shared_expert_axis = _optional_axis(plan, "shared_expert_intermediate_size")

    _set_all_config_attr(
        configs, "original_num_attention_heads", attention_axis["original_size"]
    )
    _set_existing_config_attr(
        configs, "num_attention_heads", attention_axis["padded_size"]
    )
    _set_all_config_attr(
        configs, "original_num_key_value_heads", kv_axis["original_size"]
    )
    _set_all_config_attr(
        configs, "original_intermediate_size", moe_axis["original_size"]
    )
    _set_existing_config_attr(configs, "intermediate_size", moe_axis["padded_size"])
    _set_all_config_attr(
        configs, "original_dense_intermediate_size", dense_axis["original_size"]
    )
    _set_existing_config_attr(
        configs, "dense_intermediate_size", dense_axis["padded_size"]
    )

    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    logger.warning(
        "Automatically enabled B12X virtual TP padding for MiniMax M3 TP=3: "
        "attention heads %d -> %d, logical KV heads %d -> %d, MoE "
        "intermediate size %d -> %d, dense intermediate size %d -> %d, "
        "vocab size %d -> %d.",
        attention_axis["original_size"],
        attention_axis["padded_size"],
        kv_axis["original_size"],
        kv_axis["padded_size"],
        moe_axis["original_size"],
        moe_axis["padded_size"],
        dense_axis["original_size"],
        dense_axis["padded_size"],
        vocab_axis["original_size"],
        vocab_axis["padded_size"],
    )
    if shared_expert_axis is not None:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for MiniMax M3 "
            "shared experts: intermediate size %d -> %d.",
            shared_expert_axis["original_size"],
            shared_expert_axis["padded_size"],
        )


def _require_axis(plan: dict[str, dict[str, int] | str], name: str) -> dict[str, int]:
    axis = plan.get(name)
    if not isinstance(axis, dict):
        raise ValueError(f"B12X virtual TP plan missing axis {name!r}.")
    return axis


def _optional_axis(
    plan: dict[str, dict[str, int] | str], name: str
) -> dict[str, int] | None:
    axis = plan.get(name)
    if axis is None:
        return None
    if not isinstance(axis, dict):
        raise ValueError(f"B12X virtual TP plan axis {name!r} is invalid.")
    return axis


def _validate_b12x_virtual_tp_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    assert model_config is not None

    if not _is_supported_b12x_virtual_tp_config(model_config):
        raise ValueError(
            "B12X virtual TP padding is currently supported only for "
            "declared shape profiles, DeepSeek V4, sparse MLA/DSA models, "
            "and MiniMax M3 TP=3."
        )

    if parallel_config.enable_expert_parallel:
        raise ValueError(
            "B12X virtual TP padding is incompatible with expert "
            "parallelism. Use tensor parallelism for the B12X padded path."
        )

    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        raise ValueError(
            "B12X virtual TP padding is incompatible with DeepGEMM MegaMoE."
        )

    if not _uses_native_b12x_moe(vllm_config):
        raise ValueError(
            "B12X virtual TP padding requires the native B12X MoE "
            "backend. Pass --moe-backend b12x or set VLLM_USE_B12X_MOE=1."
        )

    if _get_virtual_tp_profile(model_config) == _GQA_GDN_MOE_PROFILE:
        return

    if _is_minimax_m3_config(model_config):
        if not _uses_minimax_m3_b12x_attention(vllm_config):
            raise ValueError(
                "MiniMax M3 virtual TP padding requires the B12X MiniMax M3 "
                "attention path. Pass --attention-backend B12X_ATTN or set "
                "VLLM_USE_B12X_MINIMAX_M3_MSA=1."
            )
        return

    if not _uses_b12x_attention(vllm_config):
        raise ValueError(
            "B12X virtual TP padding requires the B12X MLA sparse attention backend."
        )


def _is_deepseek_v4_config(model_config: ModelConfig) -> bool:
    for config in _iter_virtual_tp_configs(model_config):
        if getattr(config, "model_type", None) == "deepseek_v4":
            return True
        architectures = getattr(config, "architectures", None) or ()
        if (
            "DeepseekV4ForCausalLM" in architectures
            or "DeepseekV4ForCausalLMNextN" in architectures
            or "DeepSeekV4MTPModel" in architectures
        ):
            return True

    text_config = model_config.hf_text_config
    return (
        hasattr(text_config, "o_groups")
        and hasattr(text_config, "moe_intermediate_size")
        and hasattr(text_config, "n_routed_experts")
    )


def _is_sparse_mla_config(model_config: ModelConfig) -> bool:
    for config in _iter_virtual_tp_configs(model_config):
        if (
            getattr(config, "kv_lora_rank", None) is not None
            and getattr(config, "qk_rope_head_dim", None) is not None
            and _positive_int_attr(config, "index_topk")
        ):
            return True
    return False


def _is_minimax_m3_config(model_config: ModelConfig) -> bool:
    for config in _iter_virtual_tp_configs(model_config):
        model_type = getattr(config, "model_type", None)
        if model_type in {"minimax_m3_text", "minimax_m3_vl", "minimax_m3_mtp"}:
            return True
        architectures = getattr(config, "architectures", None) or ()
        if any(
            architecture
            in {
                "MiniMaxM3SparseForCausalLM",
                "MiniMaxM3SparseForConditionalGeneration",
                "MiniMaxM3MTP",
            }
            for architecture in architectures
        ):
            return True
    return False


def _get_virtual_tp_profile(model_config: ModelConfig) -> str | None:
    profiles = {
        str(profile)
        for config in _iter_virtual_tp_configs(model_config)
        if (profile := getattr(config, VIRTUAL_TP_PROFILE_ATTR, None)) is not None
    }
    if len(profiles) > 1:
        raise ValueError(
            "B12X virtual TP padding found conflicting model shape profiles: "
            f"{sorted(profiles)}."
        )
    return next(iter(profiles), None)


def _is_supported_b12x_virtual_tp_config(model_config: ModelConfig) -> bool:
    return (
        _get_virtual_tp_profile(model_config)
        in {_GQA_GDN_MOE_PROFILE, _DENSE_GQA_PROFILE}
        or _is_deepseek_v4_config(model_config)
        or _is_sparse_mla_config(model_config)
        or _is_minimax_m3_config(model_config)
        or _is_kimi_k3_config(model_config)
    )


def _is_kimi_k3_config(model_config: ModelConfig) -> bool:
    # Kimi K3 (KDA + MLA hybrid, 896-expert latent MoE). At TP6/TP12 only the
    # vocab axis (163840) needs virtual-TP padding; heads (96) and
    # moe_intermediate (3072) divide evenly, so the generic plan applies.
    for config in _iter_virtual_tp_configs(model_config):
        if getattr(config, "model_type", None) in {"kimi_k3", "kimi_linear"}:
            return True
    return False


def _uses_native_b12x_moe(vllm_config: VllmConfig) -> bool:
    moe_backend = vllm_config.kernel_config.moe_backend
    return moe_backend == "b12x" or (moe_backend == "auto" and envs.VLLM_USE_B12X_MOE)


def _uses_b12x_attention(vllm_config: VllmConfig) -> bool:
    backend = getattr(vllm_config.attention_config, "backend", None)
    if backend == AttentionBackendEnum.B12X_MLA_SPARSE:
        return True

    model_config = vllm_config.model_config
    if model_config is not None and _is_minimax_m3_config(model_config):
        return (
            backend == AttentionBackendEnum.B12X_ATTN
            or envs.VLLM_USE_B12X_MINIMAX_M3_MSA
        )

    return (
        model_config is not None
        and _is_supported_b12x_virtual_tp_config(model_config)
        and _get_virtual_tp_profile(model_config) is None
        and current_platform.is_cuda()
        and current_platform.has_device_capability(120)
    )


def _uses_minimax_m3_b12x_attention(vllm_config: VllmConfig) -> bool:
    backend = getattr(vllm_config.attention_config, "backend", None)
    return backend == AttentionBackendEnum.B12X_ATTN or (
        vllm_config.model_config is not None
        and _is_minimax_m3_config(vllm_config.model_config)
        and envs.VLLM_USE_B12X_MINIMAX_M3_MSA
    )


def _get_moe_intermediate_local_alignment(model_config: ModelConfig) -> int:
    force_a8 = _environment_flag("B12X_MOE_FORCE_A8") or _environment_flag(
        "B12X_FORCE_MOE_A8"
    )
    is_nvfp4_quantized = getattr(model_config, "is_nvfp4_quantized", None)
    if callable(is_nvfp4_quantized) and is_nvfp4_quantized() and not force_a8:
        return _NVFP4_LOCAL_ALIGNMENT
    return _MOE_INTERMEDIATE_LOCAL_ALIGNMENT


def _get_dense_linear_local_alignment(model_config: ModelConfig) -> int:
    is_nvfp4_quantized = getattr(model_config, "is_nvfp4_quantized", None)
    if callable(is_nvfp4_quantized) and is_nvfp4_quantized():
        return _NVFP4_LOCAL_ALIGNMENT
    return 1


def _environment_flag(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _make_virtual_axis(
    original_size: int,
    tp_size: int,
    local_alignment: int = 1,
) -> dict[str, int]:
    local_size = math.ceil(original_size / tp_size)
    local_size = math.ceil(local_size / local_alignment) * local_alignment
    return {
        "original_size": original_size,
        "padded_size": local_size * tp_size,
        "tp_size": tp_size,
        "local_size": local_size,
    }


def _make_coupled_virtual_axes(
    primary_size: int,
    secondary_size: int,
    tp_size: int,
    *,
    primary_local_alignment: int = 1,
    secondary_local_alignment: int = 1,
    ratio_key: str | None = None,
    allow_secondary_replication: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Pad two axes while preserving their integer primary/secondary ratio."""
    if secondary_size <= 0 or primary_size % secondary_size != 0:
        raise ValueError(
            "B12X virtual TP padding requires coupled dimensions to have an "
            f"integer ratio, got {primary_size} and {secondary_size}."
        )
    ratio = primary_size // secondary_size
    if (
        allow_secondary_replication
        and primary_size % tp_size == 0
        and secondary_size < tp_size
        and tp_size % secondary_size == 0
    ):
        primary_axis = _make_noop_axis(primary_size, tp_size)
        secondary_axis = {
            "original_size": secondary_size,
            "padded_size": secondary_size,
            "tp_size": tp_size,
            "local_size": 1,
        }
        if ratio_key is not None:
            secondary_axis[ratio_key] = ratio
        return primary_axis, secondary_axis

    secondary_local_size = math.ceil(secondary_size / tp_size)
    while (
        secondary_local_size % secondary_local_alignment != 0
        or secondary_local_size * ratio % primary_local_alignment != 0
    ):
        secondary_local_size += 1

    primary_local_size = secondary_local_size * ratio
    primary_axis = {
        "original_size": primary_size,
        "padded_size": primary_local_size * tp_size,
        "tp_size": tp_size,
        "local_size": primary_local_size,
    }
    secondary_axis = {
        "original_size": secondary_size,
        "padded_size": secondary_local_size * tp_size,
        "tp_size": tp_size,
        "local_size": secondary_local_size,
    }
    if ratio_key is not None:
        secondary_axis[ratio_key] = ratio
    return primary_axis, secondary_axis


def _make_scaled_virtual_axis(
    original_size: int,
    base_axis: dict[str, int],
    scale: int,
    scale_key: str,
) -> dict[str, int]:
    if base_axis["original_size"] * scale != original_size:
        raise ValueError(
            "B12X virtual TP padding cannot preserve a scaled dimension with "
            f"sizes {base_axis['original_size']} and {original_size}."
        )
    return {
        "original_size": original_size,
        "padded_size": base_axis["padded_size"] * scale,
        "tp_size": base_axis["tp_size"],
        "local_size": base_axis["local_size"] * scale,
        scale_key: scale,
    }


def _make_noop_axis(original_size: int, tp_size: int) -> dict[str, int]:
    return {
        "original_size": original_size,
        "padded_size": original_size,
        "tp_size": tp_size,
        "local_size": (
            original_size // tp_size
            if tp_size > 0 and original_size % tp_size == 0
            else 0
        ),
    }


def _make_noop_vocab_axis(original_size: int, tp_size: int) -> dict[str, int]:
    axis = _make_noop_axis(original_size, tp_size)
    axis["padding_size"] = math.lcm(_VOCAB_GLOBAL_ALIGNMENT, tp_size)
    return axis


def _make_virtual_vocab_axis(
    original_size: int,
    tp_size: int,
) -> dict[str, int]:
    padding_size = math.lcm(_VOCAB_GLOBAL_ALIGNMENT, tp_size)
    padded_size = math.ceil(original_size / padding_size) * padding_size
    assert padded_size % tp_size == 0
    return {
        "original_size": original_size,
        "padded_size": padded_size,
        "tp_size": tp_size,
        "local_size": padded_size // tp_size,
        "padding_size": padding_size,
    }


def _make_virtual_output_group_axis(
    original_size: int,
    original_attention_heads: int,
    padded_attention_heads: int,
    tp_size: int,
) -> dict[str, int]:
    if original_attention_heads % original_size != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding requires num_attention_heads to "
            "be divisible by o_groups."
        )

    heads_per_group = original_attention_heads // original_size
    if padded_attention_heads % heads_per_group != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding produced attention heads that do "
            "not preserve the original heads-per-output-group ratio."
        )

    padded_size = padded_attention_heads // heads_per_group
    if padded_size % tp_size != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding produced output groups that are "
            "not divisible by tensor parallel size."
        )

    return {
        "original_size": original_size,
        "padded_size": padded_size,
        "tp_size": tp_size,
        "local_size": padded_size // tp_size,
        "heads_per_group": heads_per_group,
    }


def _require_int_attr(config: Any, attr: str) -> int:
    value = getattr(config, attr, None)
    if value is None:
        raise ValueError(f"B12X virtual TP padding requires config attribute {attr!r}.")
    return int(value)


def _positive_int_attr(config: Any, attr: str) -> bool:
    value = getattr(config, attr, None)
    if value is None:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _get_plan_config(model_config: ModelConfig) -> Any:
    return model_config.hf_text_config or model_config.hf_config


def _iter_virtual_tp_configs(model_config: ModelConfig) -> Iterable[Any]:
    hf_config = model_config.hf_config
    yield from _unique_configs(
        (
            hf_config,
            model_config.hf_text_config,
            getattr(hf_config, "text_config", None),
        )
    )


def _unique_configs(configs: Iterable[Any]) -> Iterable[Any]:
    seen: set[int] = set()
    for config in configs:
        if config is None:
            continue
        config_id = id(config)
        if config_id in seen:
            continue
        seen.add(config_id)
        yield config


def _set_existing_config_attr(configs: Iterable[Any], attr: str, value: int) -> None:
    for config in configs:
        if hasattr(config, attr):
            setattr(config, attr, value)


def _set_all_config_attr(configs: Iterable[Any], attr: str, value: int) -> None:
    for config in configs:
        setattr(config, attr, value)


def _apply_virtual_axis_to_config_attr(
    configs: Iterable[Any],
    plan: dict[str, dict[str, int] | str],
    axis_name: str,
    attr: str,
) -> None:
    axis = _require_axis(plan, axis_name)
    for config in configs:
        if not hasattr(config, attr):
            continue
        setattr(config, f"original_{attr}", axis["original_size"])
        setattr(config, attr, axis["padded_size"])
