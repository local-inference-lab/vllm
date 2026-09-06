# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved LMCache geometry for vLLM DCP and hybrid KV cache groups."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

_DCP_LAYOUT_NAMESPACE = "##lmcache-dcp-layout-v1-"
_MAX_LCM_EXPANSION_FACTOR = 16


def _leaf_specs(spec: Any) -> list[Any]:
    """Return leaf specs from vLLM's optional uniform-type wrapper."""
    per_layer_specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(per_layer_specs, dict):
        return list(per_layer_specs.values())
    return [spec]


def _iter_kv_cache_specs(kv_cache_config: Any | None) -> Iterable[Any]:
    """Yield resolved leaf specs without depending on a vLLM helper API."""
    if kv_cache_config is None:
        return
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        yield from _leaf_specs(group.kv_cache_spec)


def _is_spec_kind(spec: Any, class_name: str) -> bool:
    return any(cls.__name__ == class_name for cls in type(spec).__mro__)


def _is_attention_spec(spec: Any) -> bool:
    return _is_spec_kind(spec, "AttentionSpec")


def _is_mamba_spec(spec: Any) -> bool:
    return _is_spec_kind(spec, "MambaSpec")


def get_tokens_per_block(spec: Any, dcp_size: int) -> int:
    """Resolve one engine group's span in global scheduler tokens.

    Attention block IDs address rank-local DCP shards and therefore span the
    physical attention block multiplied by DCP. Recurrent blocks are
    replicated and retain their physical span.
    """
    leaves = _leaf_specs(spec)
    if not leaves:
        raise ValueError("KV cache group has no leaf specs")
    block_sizes = {int(leaf.block_size) for leaf in leaves}
    if len(block_sizes) != 1:
        raise ValueError(
            "All leaf KV cache specs in one group must share a block size; "
            f"got {sorted(block_sizes)}"
        )
    block_size = block_sizes.pop()
    if block_size <= 0:
        raise ValueError(f"KV cache block size must be positive, got {block_size}")
    if any(_is_attention_spec(leaf) for leaf in leaves):
        return block_size * dcp_size
    return block_size


def get_group_tokens_per_block(
    vllm_config: Any,
    kv_cache_config: Any | None,
) -> list[int]:
    """Return all engine-group spans in global scheduler coordinates."""
    dcp_size = int(
        getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    )
    groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    return [
        get_tokens_per_block(group.kv_cache_spec, dcp_size) for group in groups
    ] or [int(vllm_config.cache_config.block_size) * dcp_size]


def get_lmcache_scheduler_block_size(
    vllm_config: Any,
    kv_cache_config: Any | None,
) -> int:
    """Resolve the least scheduler span aligned to every physical group."""
    group_spans = get_group_tokens_per_block(vllm_config, kv_cache_config)
    scheduler_block_size = math.lcm(*group_spans)
    largest_group_span = max(group_spans)
    if scheduler_block_size > largest_group_span * _MAX_LCM_EXPANSION_FACTOR:
        logger.warning(
            "LMCache resolved a scheduler block of %d tokens from group spans "
            "%s (%.1fx the largest span); near-coprime geometry can make "
            "cache chunks too coarse",
            scheduler_block_size,
            group_spans,
            scheduler_block_size / largest_group_span,
        )
    return scheduler_block_size


def get_lmcache_model_name(vllm_config: Any) -> str:
    """Decorate cache identity when DCP interleave changes the byte layout."""
    model_name = str(vllm_config.model_config.model)
    parallel_config = vllm_config.parallel_config
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1))
    interleave = int(getattr(parallel_config, "cp_kv_cache_interleave_size", 1))
    if dcp_size <= 1:
        return model_name
    return f"{model_name}{_DCP_LAYOUT_NAMESPACE}d{dcp_size}-interleave{interleave}"


def get_lmcache_base_model_name(model_name: str) -> str:
    """Recover the served model name from an LMCache cache identity."""
    return model_name.partition(_DCP_LAYOUT_NAMESPACE)[0]


def get_resolved_attention_block_sizes(
    vllm_config: Any,
    kv_cache_config: Any | None,
) -> set[int]:
    """Return all resolved physical attention block sizes."""
    block_sizes = {
        int(spec.block_size)
        for spec in _iter_kv_cache_specs(kv_cache_config)
        if _is_attention_spec(spec)
    }
    return block_sizes or {int(vllm_config.cache_config.block_size)}


def validate_mamba_step_alignment(
    vllm_config: Any,
    kv_cache_config: Any | None = None,
) -> None:
    """Require each scheduler step to advance a full resolved Mamba block."""
    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    mamba_block_sizes = {
        int(spec.block_size)
        for spec in _iter_kv_cache_specs(kv_cache_config)
        if _is_mamba_spec(spec)
    }
    if len(mamba_block_sizes) > 1:
        raise ValueError(
            "All Mamba KV cache groups must use one physical block size; got "
            f"{sorted(mamba_block_sizes)}."
        )
    block_size = (
        next(iter(mamba_block_sizes))
        if mamba_block_sizes
        else int(vllm_config.cache_config.block_size)
    )
    max_batched = int(vllm_config.scheduler_config.max_num_batched_tokens)
    if max_batched < block_size:
        raise ValueError(
            "Mamba-hybrid models with LMCache require "
            "max_num_batched_tokens >= block_size so every prefill step "
            "advances at least one full block; got "
            f"max_num_batched_tokens={max_batched}, block_size={block_size}. "
            f"Set --max-num-batched-tokens to at least {block_size}."
        )


def validate_dcp_support(
    vllm_config: Any,
    n_servers: int,
    kv_cache_config: Any | None = None,
) -> None:
    """Fail closed on incomplete DCP ownership or incompatible interleave."""
    parallel_config = vllm_config.parallel_config
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1))
    if dcp_size <= 1:
        return

    pcp_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1))
    if pcp_size > 1:
        raise ValueError(
            "LMCacheMPConnector does not support prefill-context parallelism "
            f"together with DCP (got pcp={pcp_size}, dcp={dcp_size})."
        )

    interleave = int(getattr(parallel_config, "cp_kv_cache_interleave_size", 1))
    if interleave < 1:
        raise ValueError(
            "LMCacheMPConnector requires cp_kv_cache_interleave_size >= 1 "
            f"under DCP (got {interleave})."
        )
    incompatible = sorted(
        block_size
        for block_size in get_resolved_attention_block_sizes(
            vllm_config, kv_cache_config
        )
        if interleave > block_size or block_size % interleave != 0
    )
    if incompatible:
        raise ValueError(
            f"cp_kv_cache_interleave_size ({interleave}) must be no greater "
            "than and evenly divide every resolved attention block size; "
            f"incompatible sizes: {incompatible}."
        )

    world_size = int(parallel_config.world_size)
    if n_servers <= 0 or world_size % n_servers:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by n_servers ({n_servers})"
        )
    ranks_per_server = world_size // n_servers
    if ranks_per_server < dcp_size or ranks_per_server % dcp_size:
        raise ValueError(
            "Each LMCache server needs a whole-number set of "
            f"decode_context_parallel_size ({dcp_size}) shards, but "
            f"{n_servers} server(s) leave only {ranks_per_server} rank(s) each."
        )
