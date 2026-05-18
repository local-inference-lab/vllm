# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared guards for the GLM-5.1 B12X sparse MLA integration."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from importlib.util import find_spec

from vllm.platforms.interface import DeviceCapability

GLM51_MODEL_TYPE = "glm_moe_dsa"
GLM51_DRAFT_WRAPPER_MODEL_TYPES = frozenset({"deepseek_mtp"})
GLM51_QK_NOPE_HEAD_DIM = 192
GLM51_QK_ROPE_HEAD_DIM = 64
GLM51_KV_LORA_RANK = 512
GLM51_INDEX_N_HEADS = 32
GLM51_INDEX_TOPK = 2048
GLM51_BLOCK_SIZE = 64
GLM51_MLA_HEAD_SIZE = 576
GLM51_B12X_WORKSPACE_V_HEAD_DIM = 512
B12X_KV_CACHE_DTYPES = frozenset({"fp8_ds_mla", "fp8_e4m3", "fp8"})


@cache
def is_b12x_sparse_mla_available() -> bool:
    try:
        return find_spec("b12x.integration.mla") is not None
    except (ImportError, ValueError):
        return False


def _iter_config_candidates(config: object | None) -> Iterable[object]:
    seen: set[int] = set()
    stack = [config]
    while stack:
        cfg = stack.pop(0)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        yield cfg
        for attr in ("hf_text_config", "text_config", "hf_config"):
            child = getattr(cfg, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def get_b12x_config_attr(
    config: object | None,
    names: str | tuple[str, ...],
    default: object | None = None,
) -> object | None:
    if isinstance(names, str):
        names = (names,)
    for cfg in _iter_config_candidates(config):
        for name in names:
            if hasattr(cfg, name):
                return getattr(cfg, name)
    return default


def get_b12x_backend_name(vllm_config: object | None) -> str:
    attention_config = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention_config, "backend", None)
    return getattr(backend, "name", str(backend) if backend is not None else "").upper()


def b12x_attention_backend_overridden(vllm_config: object | None) -> bool:
    attention_config = getattr(vllm_config, "attention_config", None)
    return getattr(attention_config, "backend", None) is not None


def validate_b12x_sm120(capability: DeviceCapability) -> str | None:
    if capability.major != 12 or capability.minor != 0:
        return (
            "B12X sparse MLA requires an SM120 Blackwell 12x device, "
            f"got {capability}"
        )
    return None


def get_glm51_b12x_workspace_v_head_dim(hf_config: object | None) -> int | None:
    model_v_head_dim = get_b12x_config_attr(
        hf_config,
        ("v_head_dim", "mla_v_head_dim"),
        GLM51_QK_NOPE_HEAD_DIM + GLM51_QK_ROPE_HEAD_DIM,
    )
    if model_v_head_dim is None:
        return None
    try:
        return max(int(model_v_head_dim), GLM51_KV_LORA_RANK)
    except (TypeError, ValueError):
        return None


def validate_glm51_b12x_sparse_mla_config(
    hf_config: object | None,
    *,
    allow_draft_wrapper: bool = False,
) -> str | None:
    allowed_model_types = {GLM51_MODEL_TYPE}
    if allow_draft_wrapper:
        allowed_model_types.update(GLM51_DRAFT_WRAPPER_MODEL_TYPES)

    model_type = get_b12x_config_attr(hf_config, "model_type")
    if model_type not in allowed_model_types:
        if allow_draft_wrapper:
            return (
                "B12X sparse MLA currently requires model_type=glm_moe_dsa "
                "or its vLLM deepseek_mtp draft wrapper"
            )
        return "B12X sparse MLA currently requires model_type=glm_moe_dsa"

    required = (
        ("index_topk", GLM51_INDEX_TOPK),
        ("index_n_heads", GLM51_INDEX_N_HEADS),
        ("kv_lora_rank", GLM51_KV_LORA_RANK),
        ("qk_rope_head_dim", GLM51_QK_ROPE_HEAD_DIM),
        ("qk_nope_head_dim", GLM51_QK_NOPE_HEAD_DIM),
    )
    for name, expected in required:
        actual = get_b12x_config_attr(hf_config, name)
        if actual != expected:
            return f"B12X sparse MLA currently requires {name}={expected}"

    workspace_v_head_dim = get_glm51_b12x_workspace_v_head_dim(hf_config)
    if workspace_v_head_dim != GLM51_B12X_WORKSPACE_V_HEAD_DIM:
        return (
            "B12X sparse MLA currently requires workspace_v_head_dim="
            f"{GLM51_B12X_WORKSPACE_V_HEAD_DIM}"
        )
    return None


def is_glm51_b12x_sparse_mla_config(
    vllm_config: object | None,
    *,
    allow_draft_wrapper: bool = False,
) -> bool:
    if vllm_config is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_text_config", None)
    if (
        validate_glm51_b12x_sparse_mla_config(
            hf_config,
            allow_draft_wrapper=allow_draft_wrapper,
        )
        is not None
    ):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    cache_dtype = getattr(cache_config, "cache_dtype", None)
    return cache_dtype in B12X_KV_CACHE_DTYPES


def b12x_sparse_mla_active_for_config(
    vllm_config: object | None,
    *,
    allow_draft_wrapper: bool = True,
) -> bool:
    if vllm_config is None:
        return False
    backend_name = get_b12x_backend_name(vllm_config)
    if backend_name == "B12X_MLA_SPARSE":
        return True
    if b12x_attention_backend_overridden(vllm_config):
        return False
    return is_glm51_b12x_sparse_mla_config(
        vllm_config,
        allow_draft_wrapper=allow_draft_wrapper,
    )


def b12x_backend_active_for_config(
    vllm_config: object | None,
    *,
    allow_draft_wrapper: bool = True,
) -> bool:
    if b12x_sparse_mla_active_for_config(
        vllm_config,
        allow_draft_wrapper=allow_draft_wrapper,
    ):
        return True
    kernel_config = getattr(vllm_config, "kernel_config", None)
    moe_backend = getattr(kernel_config, "moe_backend", None)
    return str(moe_backend).lower() == "b12x"


def b12x_sparse_indexer_active_for_config(
    vllm_config: object | None,
    *,
    allow_draft_wrapper: bool = True,
) -> bool:
    return b12x_sparse_mla_active_for_config(
        vllm_config,
        allow_draft_wrapper=allow_draft_wrapper,
    )
