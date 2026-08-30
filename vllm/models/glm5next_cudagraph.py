# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph qualification contract for GLM-5.3-Flash."""

from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig

MAX_FULL_GRAPH_SEQS = 4
MAX_FULL_GRAPH_TOKENS = 32768
_RECIPE = {
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "index_kpool": 4,
}


def _backend_name(backend: Any) -> str | None:
    name = getattr(backend, "name", backend)
    return name.upper() if isinstance(name, str) else None


def is_glm53_full_graph_path(vllm_config: VllmConfig | None) -> bool:
    """Return whether a config selects the qualified mixed FULL graph path."""
    if vllm_config is None:
        return False
    hf_config = vllm_config.model_config.hf_text_config
    if getattr(hf_config, "model_type", None) not in {
        "glm5_next",
        "glm5_next_text",
    }:
        return False
    if any(getattr(hf_config, name, None) != value for name, value in _RECIPE.items()):
        return False
    if _backend_name(vllm_config.attention_config.backend) != "B12X":
        return False
    parallel = vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 4
        or parallel.decode_context_parallel_size != 4
        or parallel.cp_kv_cache_interleave_size != 4
    ):
        return False
    speculative = vllm_config.speculative_config
    return speculative is not None and speculative.num_speculative_tokens == 3


def require_glm53_full_graph_capacity(vllm_config: VllmConfig) -> None:
    """Fail closed when scheduler bounds exceed qualified static arenas."""
    scheduler = vllm_config.scheduler_config
    if not 0 < scheduler.max_num_seqs <= MAX_FULL_GRAPH_SEQS:
        raise ValueError(
            "GLM-5.3 mixed FULL max_num_seqs exceeds the qualified "
            f"capacity: {scheduler.max_num_seqs} > {MAX_FULL_GRAPH_SEQS}"
        )
    if not 0 < scheduler.max_num_batched_tokens <= MAX_FULL_GRAPH_TOKENS:
        raise ValueError(
            "GLM-5.3 mixed FULL max_num_batched_tokens exceeds the qualified "
            f"capacity: {scheduler.max_num_batched_tokens} > "
            f"{MAX_FULL_GRAPH_TOKENS}"
        )
