# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel collectives for Kimi-K3 projection outputs."""

import torch

from vllm.distributed import (
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.v1.attention.ops.b12x_dcp import (
    B12X_DECODE_TOKEN_CAP,
    try_b12x_projection_pair_gather,
    try_b12x_query_gather,
)


def _get_kimi_projection_group():
    """Return the coordinator spanning every projection weight shard."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError(
            "Kimi projection group does not span tensor-parallel ranks: "
            f"group={tp_group.world_size}, TP={tp_size}"
        )
    dcp_group = get_dcp_group()
    if dcp_group.world_size == tp_size and list(dcp_group.ranks) == list(
        tp_group.ranks
    ):
        return dcp_group
    return tp_group


def _try_b12x_kimi_projection_gather(
    output_parallel: torch.Tensor,
) -> torch.Tensor | None:
    """Gather one decode projection over the lossless B12X copy channel."""
    if (
        output_parallel.ndim != 2
        or not 0 < output_parallel.shape[0] <= B12X_DECODE_TOKEN_CAP
        or not output_parallel.is_cuda
        or not output_parallel.is_contiguous()
    ):
        return None

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None

    local_width = output_parallel.shape[1]
    restore_dtype: torch.dtype | None = None
    strip_local_width: int | None = None
    if output_parallel.dtype in (torch.float16, torch.bfloat16):
        padded_width = (local_width + 7) // 8 * 8
        if padded_width != local_width:
            output_parallel = torch.nn.functional.pad(
                output_parallel, (0, padded_width - local_width)
            )
            strip_local_width = local_width
        transport = output_parallel.view(output_parallel.shape[0], 1, padded_width)
    elif output_parallel.dtype == torch.float32:
        raw_width = local_width * output_parallel.element_size()
        if raw_width % 16:
            return None
        transport = output_parallel.view(torch.float8_e4m3fn).view(
            output_parallel.shape[0], 1, raw_width
        )
        restore_dtype = torch.float32
    elif output_parallel.dtype == torch.float8_e4m3fn:
        if local_width % 16:
            return None
        transport = output_parallel.view(output_parallel.shape[0], 1, local_width)
    else:
        return None

    gathered = try_b12x_query_gather(
        transport,
        _get_kimi_projection_group(),
        output_head_dim=transport.shape[-1],
    )
    if gathered is None:
        return None
    if strip_local_width is not None:
        gathered = gathered.narrow(-1, 0, strip_local_width).contiguous()
    gathered = gathered.flatten(1)
    if restore_dtype is not None:
        gathered = gathered.view(restore_dtype)
    return gathered


def gather_kimi_sharded_projection(
    output_parallel: torch.Tensor,
    *,
    use_b12x: bool,
) -> torch.Tensor:
    """Gather a rank-major Kimi projection through a lossless fast path."""
    if get_tensor_model_parallel_world_size() <= 1:
        return output_parallel
    if use_b12x:
        gathered = _try_b12x_kimi_projection_gather(output_parallel)
        if gathered is not None:
            return gathered
    return tensor_model_parallel_all_gather(output_parallel, dim=-1)


def gather_kimi_sharded_projection_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
    *,
    use_b12x: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather two decode projections behind one B12X barrier when enabled."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return local_first, local_second
    if use_b12x:
        projection_group = _get_kimi_projection_group()
        if projection_group.world_size == tp_size:
            gathered = try_b12x_projection_pair_gather(
                local_first,
                local_second,
                projection_group,
            )
            if gathered is not None:
                return gathered
    return (
        tensor_model_parallel_all_gather(local_first, dim=-1),
        tensor_model_parallel_all_gather(local_second, dim=-1),
    )
