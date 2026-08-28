# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X PCIe collectives used by dense MLA decode and Kimi projections.

The module owns B12X CUDA-IPC pools separately from vLLM process-group
coordinators. Eager execution and each CUDA-graph owner use distinct semantic
channels so signal state cannot leak between target, draft, and disposable
profiling graphs.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

B12X_DECODE_TOKEN_CAP = 8
_SUPPORTED_WORLD_SIZES = (2, 4, 8, 16)
_EAGER_CHANNEL_ID = "vllm:eager:dcp"
_MAX_CONCURRENT_CHANNELS = 2

_PoolKey = tuple[int, int, int, int, int, int]
_POOLS: dict[_PoolKey, Any] = {}
_DISABLED: set[_PoolKey] = set()
_ACTIVE_CAPTURE: dict[int, tuple[str, Any, ExitStack]] = {}


@lru_cache(maxsize=1)
def _load_pool_type() -> Any | None:
    try:
        from b12x.comm.pcie import DcpAllToAllPool
    except (AttributeError, ImportError):
        return None
    return DcpAllToAllPool


def _channel_id(group: GroupCoordinator) -> str:
    active = _ACTIVE_CAPTURE.get(id(group.device_group))
    return _EAGER_CHANNEL_ID if active is None else active[0]


def _pool_init_failed(
    group: GroupCoordinator,
    device: torch.device,
    error: Exception | None,
) -> bool:
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group.device_group)
    return bool(failed.item())


def _get_pool(
    group: GroupCoordinator,
    *,
    device: torch.device,
    total_heads: int,
    output_head_dim: int,
    query_head_dim: int,
    max_batch_size: int,
) -> Any | None:
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key: _PoolKey = (
        id(group.device_group),
        int(device_index),
        int(total_heads),
        int(output_head_dim),
        int(query_head_dim),
        int(max_batch_size),
    )
    if key in _DISABLED:
        return None
    if (pool := _POOLS.get(key)) is not None:
        return pool
    if torch.cuda.is_current_stream_capturing():
        return None
    pool_type = _load_pool_type()
    if pool_type is None:
        _DISABLED.add(key)
        return None

    pool = None
    init_error: Exception | None = None
    try:
        pool = pool_type.from_exchange_group(
            exchange_group=group.device_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=output_head_dim,
            query_head_dim=query_head_dim,
            single_channel=False,
            max_concurrent_channels=_MAX_CONCURRENT_CHANNELS,
        )
        active = _ACTIVE_CAPTURE.get(id(group.device_group))
        if active is None:
            pool.prepare_channels((_EAGER_CHANNEL_ID,))
            pool.for_stream(channel_id=_EAGER_CHANNEL_ID)
        else:
            active_channel, active_stream, active_stack = active
            pool.prepare_channels((_EAGER_CHANNEL_ID, active_channel))
            active_stack.enter_context(
                pool.capture(stream=active_stream, channel_id=active_channel)
            )
    except Exception as exc:
        init_error = exc

    if _pool_init_failed(group, device, init_error):
        if pool is not None:
            pool.close()
        _DISABLED.add(key)
        if init_error is not None:
            logger.warning(
                "B12X PCIe collective initialization failed; using the "
                "configured vLLM collective instead: %s",
                init_error,
            )
        return None

    assert pool is not None
    _POOLS[key] = pool
    logger.info(
        "Using B12X PCIe collectives "
        "(world_size=%d, batch_capacity=%d, heads=%d, "
        "query_head_dim=%d, output_head_dim=%d).",
        group.world_size,
        max_batch_size,
        total_heads,
        query_head_dim,
        output_head_dim,
    )
    return pool


@contextmanager
def capture_b12x_dcp_pools(
    group: GroupCoordinator,
    stream: torch.cuda.Stream | None,
    *,
    channel_id: str,
):
    """Bind every pool for ``group`` to one rank-stable graph owner."""
    group_id = id(group.device_group)
    active = _ACTIVE_CAPTURE.get(group_id)
    if active is not None:
        active_channel, active_stream, _ = active
        if active_channel != channel_id or active_stream is not stream:
            raise RuntimeError(
                "nested B12X capture must preserve its channel identity and stream"
            )
        yield
        return

    matching = sorted(
        ((key, pool) for key, pool in _POOLS.items() if key[0] == group_id),
        key=lambda item: item[0][1:],
    )
    try:
        with ExitStack() as stack:
            _ACTIVE_CAPTURE[group_id] = (channel_id, stream, stack)
            for _, pool in matching:
                stack.enter_context(pool.capture(stream=stream, channel_id=channel_id))
            yield
    finally:
        _ACTIVE_CAPTURE.pop(group_id, None)


def checkpoint_b12x_dcp_channels(
    group: GroupCoordinator,
) -> tuple[int, dict[_PoolKey, tuple[Any, Any]]]:
    """Snapshot pool channels before disposable CUDA-graph profiling."""
    group_id = id(group.device_group)
    checkpoints = {
        key: (pool, pool.checkpoint_channels())
        for key, pool in _POOLS.items()
        if key[0] == group_id
    }
    return group_id, checkpoints


def rollback_b12x_dcp_channels(
    checkpoint: tuple[int, dict[_PoolKey, tuple[Any, Any]]],
) -> None:
    """Release profiling channels and restore the preceding pool state."""
    group_id, checkpoints = checkpoint
    for key, pool in list(_POOLS.items()):
        if key[0] != group_id:
            continue
        saved = checkpoints.get(key)
        if saved is None:
            pool.close()
            del _POOLS[key]
            continue
        saved_pool, channel_checkpoint = saved
        if pool is not saved_pool:
            pool.close()
            _POOLS[key] = saved_pool
        saved_pool.rollback_channels(channel_checkpoint)


def _supported_output_layout(value: torch.Tensor) -> bool:
    if value.ndim != 3 or int(value.stride(2)) != 1:
        return False
    batch, heads, head_dim = (int(x) for x in value.shape)
    stride_batch, stride_head, _ = (int(x) for x in value.stride())
    return (stride_batch == heads * head_dim and stride_head == head_dim) or (
        stride_batch == head_dim and stride_head >= batch * head_dim
    )


def try_b12x_lse_reduce_scatter(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    group: GroupCoordinator,
    *,
    is_lse_base_on_e: bool,
    max_batch_size: int = B12X_DECODE_TOKEN_CAP,
    query_head_dim: int | None = None,
) -> torch.Tensor | None:
    """Combine dense MLA shard outputs when the B12X contract is satisfied."""
    world_size = group.world_size
    if (
        world_size not in _SUPPORTED_WORLD_SIZES
        or not partial_output.is_cuda
        or partial_output.dtype not in (torch.float16, torch.bfloat16)
        or partial_lse.dtype != torch.float32
        or partial_output.ndim != 3
        or partial_lse.shape != partial_output.shape[:2]
    ):
        return None
    batch, total_heads, output_head_dim = partial_output.shape
    if (
        batch < 1
        or batch > max_batch_size
        or total_heads % world_size
        or output_head_dim % 8
    ):
        return None
    query_head_dim = output_head_dim if query_head_dim is None else query_head_dim
    if query_head_dim <= 0 or query_head_dim % 8:
        return None

    pool = _get_pool(
        group,
        device=partial_output.device,
        total_heads=total_heads,
        output_head_dim=output_head_dim,
        query_head_dim=query_head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    if not _supported_output_layout(partial_output):
        partial_output = partial_output.contiguous()
    if not partial_lse.is_contiguous():
        partial_lse = partial_lse.contiguous()

    output_storage = torch.empty(
        (total_heads // world_size, batch, output_head_dim),
        device=partial_output.device,
        dtype=partial_output.dtype,
    )
    output = output_storage.transpose(0, 1)
    return pool.lse_reduce_scatter(
        partial_output,
        partial_lse,
        out=output,
        is_lse_base_on_e=is_lse_base_on_e,
        channel_id=_channel_id(group),
    )


def try_b12x_query_gather(
    local_query: torch.Tensor,
    group: GroupCoordinator,
    *,
    max_batch_size: int = B12X_DECODE_TOKEN_CAP,
    output_head_dim: int | None = None,
) -> torch.Tensor | None:
    """Gather rank-local dense MLA query heads through CUDA IPC."""
    world_size = group.world_size
    if (
        world_size not in _SUPPORTED_WORLD_SIZES
        or not local_query.is_cuda
        or local_query.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or local_query.ndim != 3
    ):
        return None
    local_query = local_query.contiguous()
    batch, local_heads, query_head_dim = local_query.shape
    alignment = 16 if local_query.dtype == torch.float8_e4m3fn else 8
    if (
        batch < 1
        or batch > max_batch_size
        or local_heads <= 0
        or query_head_dim % alignment
    ):
        return None
    output_head_dim = query_head_dim if output_head_dim is None else output_head_dim
    if output_head_dim <= 0 or output_head_dim % 8:
        return None

    pool = _get_pool(
        group,
        device=local_query.device,
        total_heads=local_heads * world_size,
        output_head_dim=output_head_dim,
        query_head_dim=query_head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    return pool.all_gather_heads(local_query, channel_id=_channel_id(group))


def try_b12x_projection_pair_gather(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
    group: GroupCoordinator,
    *,
    max_batch_size: int = B12X_DECODE_TOKEN_CAP,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather two rank-local projection rows behind one PCIe barrier."""
    supported_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
    )
    if (
        group.world_size not in _SUPPORTED_WORLD_SIZES
        or local_first.dtype not in supported_dtypes
        or local_second.dtype not in supported_dtypes
        or not local_first.is_cuda
        or not local_second.is_cuda
        or local_first.device != local_second.device
        or local_first.ndim != 2
        or local_second.ndim != 2
        or local_first.shape[0] != local_second.shape[0]
    ):
        return None
    local_first = local_first.contiguous()
    local_second = local_second.contiguous()
    batch = int(local_first.shape[0])
    first_row_bytes = int(local_first.shape[1]) * local_first.element_size()
    second_row_bytes = int(local_second.shape[1]) * local_second.element_size()
    if (
        batch < 1
        or batch > max_batch_size
        or first_row_bytes % 16
        or second_row_bytes % 16
    ):
        return None

    combined_row_bytes = first_row_bytes + second_row_bytes
    pool = _get_pool(
        group,
        device=local_first.device,
        total_heads=group.world_size,
        output_head_dim=combined_row_bytes,
        query_head_dim=combined_row_bytes,
        max_batch_size=max_batch_size,
    )
    if pool is None or not hasattr(pool, "all_gather_pair"):
        return None
    return pool.all_gather_pair(
        local_first,
        local_second,
        channel_id=_channel_id(group),
    )
