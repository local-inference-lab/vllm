# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared symmetric-memory infrastructure for context-parallel attention."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.device_communicators.all_reduce_utils import (
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False


@functools.cache
def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        device = torch.device("cuda", torch.accelerator.current_device_index())
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
            return False
        probe = symm_mem.empty(8, dtype=torch.uint8, device=device)
        probe.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(probe, group.device_group.group_name)
        spans = handle is not None and handle.multicast_ptr != 0
    except Exception as error:
        logger.debug("Direct CP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct CP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


@functools.cache
def _cuda_p2p_spans_group(group: GroupCoordinator) -> bool:
    """Check whether every rank can directly access every peer GPU.

    Args:
        group: Process group whose local CUDA devices should be checked.

    Returns:
        Whether all directed device pairs in the group support CUDA P2P access.
    """
    if not current_platform.is_cuda():
        return False
    if group.world_size <= 1:
        return True

    try:
        local_ranks: list[int | None] = [None] * group.world_size
        torch.distributed.all_gather_object(
            local_ranks,
            group.local_rank,
            group=group.cpu_group,
        )
        if any(rank is None for rank in local_ranks):
            return False
        spans = all(
            src == dst or gpu_p2p_access_check(src, dst)
            for src in local_ranks
            for dst in local_ranks
            if src is not None and dst is not None
        )
    except Exception as error:
        logger.debug("Direct CP CUDA P2P probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct CP CUDA P2P across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def direct_cp_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if use_direct is not None:
        return use_direct
    return (
        symm_mem_available
        and current_platform.is_cuda()
        and (supported_dtypes is None or dtype in supported_dtypes)
        and (
            all(in_the_same_node_as(group.cpu_group, source_rank=0))
            or _symm_mem_spans_group(group)
        )
    )


def direct_cp_peer_access_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    """Check eligibility for direct CP operations that access peer pointers.

    Args:
        group: Process group used by the direct CP operation.
        dtype: Data type transferred by the operation.
        use_direct: Explicit direct-path override, or ``None`` for auto selection.
        supported_dtypes: Data types supported by the direct implementation.

    Returns:
        Whether the direct peer-access path should be used.
    """
    if use_direct is not None:
        return use_direct
    enabled = direct_cp_enabled(group, dtype, None, supported_dtypes)
    if enabled and not _cuda_p2p_spans_group(group):
        logger.info_once(
            "Direct CP peer access is unavailable; falling back to collectives."
        )
        return False
    return enabled


def direct_cp_multicast_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    return direct_cp_enabled(
        group, dtype, use_direct, supported_dtypes
    ) and _symm_mem_spans_group(group)


class DirectCPWorkspace:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.device = torch.device(device)
        self.num_ubatches = num_ubatches
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=self.device)
        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []

    def _allocate(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = symm_mem.empty(shape, device=self.device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "CP symmetric memory rendezvous returned None"
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        self.device = storage.device
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
        disabled = [0] * self.num_ubatches
        for allocated, handle, _ in self._allocations:
            if allocated is storage:
                break
        else:
            return disabled
        try:
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, storage.device.index
            ):
                return disabled
            multicast_base = handle.multicast_ptr
        except Exception:
            return disabled
        if not multicast_base:
            return disabled
        storage_base = storage.data_ptr()
        return [
            multicast_base + (storage[ubatch].data_ptr() - storage_base)
            for ubatch in range(self.num_ubatches)
        ]
