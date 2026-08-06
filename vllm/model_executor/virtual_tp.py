# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import torch

from vllm.config.virtual_tp import VIRTUAL_TP_PLAN_ATTR


def get_current_virtual_tp_plan(config: Any | None = None) -> dict[str, Any] | None:
    if config is not None:
        return getattr(config, VIRTUAL_TP_PLAN_ATTR, None)

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None

    return getattr(vllm_config.model_config.hf_text_config, VIRTUAL_TP_PLAN_ATTR, None)


def is_virtual_tp_padded_enabled() -> bool:
    plan = get_current_virtual_tp_plan()
    return plan is not None


def get_virtual_tp_axis_local_size(axis_name: str, default: int) -> int:
    plan = get_current_virtual_tp_plan()
    if plan is None:
        return default

    axis = plan.get(axis_name)
    if not isinstance(axis, dict):
        return default

    local_size = axis.get("local_size")
    if local_size is None:
        return default
    return int(local_size)


def get_virtual_tp_axis_original_size(
    axis_name: str,
    default: int,
    *,
    config: Any | None = None,
) -> int:
    plan = get_current_virtual_tp_plan(config)
    if plan is None:
        return default

    axis = plan.get(axis_name)
    if not isinstance(axis, dict):
        return default

    original_size = axis.get("original_size")
    if original_size is None:
        return default
    return int(original_size)


def get_virtual_tp_axis_padded_size(
    axis_name: str,
    default: int,
    *,
    config: Any | None = None,
) -> int:
    plan = get_current_virtual_tp_plan(config)
    if plan is None:
        return default

    axis = plan.get(axis_name)
    if not isinstance(axis, dict):
        return default

    padded_size = axis.get("padded_size")
    if padded_size is None:
        return default
    return int(padded_size)


def get_virtual_tp_axis_shard_size(axis_name: str, param_axis_size: int) -> int:
    """Return the checkpoint slice size in the parameter's stored units."""
    virtual_local_size = get_virtual_tp_axis_local_size(axis_name, param_axis_size)
    return min(virtual_local_size, param_axis_size)


def get_virtual_tp_vocab_padding_size(default: int) -> int:
    plan = get_current_virtual_tp_plan()
    if plan is None:
        return default

    axis = plan.get("vocab_size")
    if not isinstance(axis, dict):
        return default

    padding_size = axis.get("padding_size")
    if padding_size is None:
        return default
    return int(padding_size)


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor,
    dim: int,
    start_idx: int,
    shard_size: int,
) -> torch.Tensor:
    """Return a strict shard or a zero-padded virtual-TP shard.

    Without an active virtual TP plan this intentionally keeps the previous
    ``narrow`` behavior so shape mistakes still fail immediately.  With virtual
    TP enabled, out-of-range tails are materialized as zeros on the same device
    and dtype as the checkpoint tensor.
    """
    if not is_virtual_tp_padded_enabled():
        return loaded_weight.narrow(dim, start_idx, shard_size)

    if loaded_weight.ndim == 0:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    dim = dim if dim >= 0 else loaded_weight.ndim + dim
    if dim < 0 or dim >= loaded_weight.ndim:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    available = loaded_weight.shape[dim] - start_idx
    if available >= shard_size:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    shape = list(loaded_weight.shape)
    shape[dim] = shard_size
    padded_weight = torch.zeros(
        shape,
        dtype=loaded_weight.dtype,
        device=loaded_weight.device,
    )

    if available > 0:
        valid_weight = loaded_weight.narrow(dim, start_idx, available)
        padded_weight.narrow(dim, 0, available).copy_(valid_weight)

    return padded_weight
