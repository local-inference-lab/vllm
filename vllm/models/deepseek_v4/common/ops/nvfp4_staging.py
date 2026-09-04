# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent workspaces for DeepSeek-V4 native NVFP4 cache writes."""

from dataclasses import dataclass

import torch

import vllm.envs as envs


@dataclass(frozen=True)
class DeepseekV4NVFP4Staging:
    """A graph-stable BF16 row cache and its contiguous slot mapping."""

    cache: torch.Tensor  # [1, max_num_tokens, 512] BF16
    slot_mapping: torch.Tensor  # [max_num_tokens] int64: 0, 1, ...
    scale: torch.Tensor  # Scalar consumed by the cache-op API.


_STAGING: dict[tuple[str, int | None, str], DeepseekV4NVFP4Staging] = {}


def use_deepseek_v4_nvfp4_direct_write() -> bool:
    """Return whether the opt-in direct writer is available and requested."""
    if not envs.VLLM_DSV4_NVFP4_DIRECT_WRITE:
        return False
    if not hasattr(torch.ops._C, "fused_deepseek_v4_qnorm_rope_nvfp4_mla"):
        raise RuntimeError(
            "VLLM_DSV4_NVFP4_DIRECT_WRITE requires a vLLM extension built "
            "with fused_deepseek_v4_qnorm_rope_nvfp4_mla"
        )
    return True


def get_deepseek_v4_nvfp4_staging(
    *,
    device: torch.device | str,
    max_num_tokens: int,
    producer: str,
) -> DeepseekV4NVFP4Staging:
    """Return one persistent workspace per CUDA device and producer.

    Allocation occurs during model construction, before CUDA graph capture. If
    a later layer needs a larger construction-time capacity, replace the cached
    entry with a larger workspace. Layers that already hold the smaller object
    retain its graph-stable address and only use their own declared capacity.
    """
    resolved = torch.device(device)
    key = (resolved.type, resolved.index, producer)
    existing = _STAGING.get(key)
    if existing is not None and existing.cache.shape[1] >= max_num_tokens:
        return existing

    if producer not in {"swa", "compressor"}:
        raise ValueError(f"Unknown DeepSeek-V4 NVFP4 producer: {producer}")
    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive for NVFP4 staging")

    staging = DeepseekV4NVFP4Staging(
        cache=torch.empty(
            (1, max_num_tokens, 512), dtype=torch.bfloat16, device=resolved
        ),
        slot_mapping=torch.arange(max_num_tokens, dtype=torch.int64, device=resolved),
        scale=torch.ones(1, dtype=torch.float32, device=resolved),
    )
    _STAGING[key] = staging
    return staging
