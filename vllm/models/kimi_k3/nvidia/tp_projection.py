# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel collectives for Kimi-K3 projection outputs."""

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_in_place,
)
from vllm.v1.attention.ops import dcp_alltoall
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_b12x_all_gather_heads,
    dcp_b12x_all_gather_pair,
)

_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS = 8
_KIMI_INPLACE_REDUCTION_MIN_TOKENS = 1024


def reduce_kimi_full_width_projection(
    output_parallel: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Reduce a Kimi full-width row-parallel projection.

    A projection with at least 1,024 rows is a prefill intermediate whose
    rank-local value is dead after reduction. NCCL may therefore overwrite
    that storage and avoid an equally sized output allocation. Smaller
    projections retain the ordinary functional collective used by decode and
    CUDA Graph capture.
    """
    if tp_size <= 1:
        return output_parallel
    if (
        output_parallel.ndim == 2
        and output_parallel.shape[0] >= _KIMI_INPLACE_REDUCTION_MIN_TOKENS
        and output_parallel.is_contiguous()
    ):
        return tensor_model_parallel_all_reduce_in_place(output_parallel)
    return tensor_model_parallel_all_reduce(output_parallel)


def _get_kimi_projection_group():
    """Return the coordinator that spans every projection weight shard.

    Projection weights are sharded across the full tensor-parallel group. The
    DCP coordinator is valid only when its ordered rank list matches the
    tensor-parallel coordinator.
    """
    tp_size = get_tensor_model_parallel_world_size()
    dcp_group = get_dcp_group()
    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError(
            "Kimi projection group does not span tensor-parallel ranks: "
            f"group={tp_group.world_size}, TP={tp_size}"
        )
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
        not envs.VLLM_USE_B12X_DCP_A2A
        or output_parallel.ndim != 2
        or output_parallel.shape[0] != 1
        or not output_parallel.is_cuda
        or not output_parallel.is_contiguous()
    ):
        return None

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None
    projection_group = _get_kimi_projection_group()

    local_width = output_parallel.shape[1]
    restore_dtype: torch.dtype | None = None
    strip_local_width: int | None = None
    if output_parallel.dtype in (torch.float16, torch.bfloat16):
        if local_width % 8 == 0:
            transport = output_parallel.view(1, 1, local_width)
        else:
            padded_width = (local_width + 7) // 8 * 8
            transport = torch.nn.functional.pad(
                output_parallel, (0, padded_width - local_width)
            ).view(1, 1, padded_width)
            strip_local_width = local_width
    elif output_parallel.dtype == torch.float32:
        raw_width = local_width * output_parallel.element_size()
        if raw_width % 8 != 0:
            return None
        # The FP8 view exposes one-byte transport lanes without converting the
        # FP32 payload. The gathered result is restored to the original dtype.
        transport = output_parallel.view(torch.float8_e4m3fn).view(1, 1, raw_width)
        restore_dtype = torch.float32
    elif output_parallel.dtype == torch.float8_e4m3fn:
        if local_width % 16 != 0:
            return None
        transport = output_parallel.view(1, 1, local_width)
    else:
        return None

    gathered = dcp_b12x_all_gather_heads(
        transport,
        projection_group,
        max_batch_size=1,
    )
    if strip_local_width is not None:
        gathered = gathered.narrow(-1, 0, strip_local_width).contiguous()
    gathered = gathered.flatten(1)
    if restore_dtype is not None:
        gathered = gathered.view(restore_dtype)
    return gathered


def gather_kimi_sharded_projection(output_parallel: torch.Tensor) -> torch.Tensor:
    """Gather a rank-major Kimi-K3 projection through a lossless fast path."""
    if get_tensor_model_parallel_world_size() <= 1:
        return output_parallel
    gathered = _try_b12x_kimi_projection_gather(output_parallel)
    if gathered is not None:
        return gathered
    return tensor_model_parallel_all_gather(output_parallel, dim=-1)


def gather_kimi_sharded_projection_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather two decode projections behind one lossless B12X barrier."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return local_first, local_second
    if (
        local_first.ndim == local_second.ndim == 2
        and local_first.shape[0] == local_second.shape[0]
        and 0 < local_first.shape[0] <= _KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS
        and local_first.is_cuda
        and local_second.is_cuda
        and local_first.is_contiguous()
        and local_second.is_contiguous()
    ):
        projection_group = _get_kimi_projection_group()
        if projection_group.world_size == tp_size:
            return dcp_b12x_all_gather_pair(
                local_first,
                local_second,
                projection_group,
                max_batch_size=_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS,
            )
    return (
        gather_kimi_sharded_projection(local_first),
        gather_kimi_sharded_projection(local_second),
    )


def try_gather_kimi_sharded_projection_pair_topk(
    local_down: torch.Tensor,
    local_router: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use B12X Kimi projection transport and precomputed expert selection.

    A missing or ineligible binding returns ``None``. The model caller must
    then use the exact paired gather and ordinary router operations.
    """
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    pair_topk = getattr(
        dcp_alltoall,
        "try_dcp_b12x_all_gather_pair_kimi_topk",
        None,
    )
    if pair_topk is None:
        return None
    return pair_topk(
        local_down,
        local_router,
        correction_bias,
        _get_kimi_projection_group(),
    )


def try_select_kimi_routed_experts(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use B12X CuTeDSL expert selection for assembled Kimi router logits."""
    select_topk = getattr(dcp_alltoall, "try_b12x_kimi_topk16", None)
    if select_topk is None:
        return None
    return select_topk(
        router_logits,
        correction_bias,
    )
