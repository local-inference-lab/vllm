# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DCP collective selection and direct symmetric-memory implementations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.ops.common import cp_lse_ag_out_ar, cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

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
        logger.debug("Direct DCP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct DCP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def _direct_dcp_enabled(
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


def _direct_dcp_multicast_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    return _direct_dcp_enabled(
        group, dtype, use_direct, supported_dtypes
    ) and _symm_mem_spans_group(group)


def get_dcp_workspace_max_num_tokens(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    speculative_tokens = vllm_config.num_speculative_tokens
    tokens_per_seq = (
        1
        + (
            2
            if speculative_config is not None and speculative_config.parallel_drafting
            else 1
        )
        * speculative_tokens
    )
    return min(
        scheduler_config.max_num_batched_tokens,
        max(
            scheduler_config.max_num_seqs * tokens_per_seq,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )


class _DirectDCPWorkspace:
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
        assert handle is not None, "DCP symmetric memory rendezvous returned None"
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


def reserve_query_head_storage(
    query: torch.Tensor, padded_num_heads: int
) -> torch.Tensor:
    """Reserve backing storage for fixed-head decode kernels."""
    assert query.ndim == 3
    assert query.shape[1] <= padded_num_heads
    padded = query.new_empty((query.shape[0], padded_num_heads, query.shape[2]))
    padded.resize_(query.shape)
    padded.copy_(query)
    return padded


_A2A_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


class DirectDCPA2AWorkspace(_DirectDCPWorkspace):
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _A2A_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                f"Direct DCP A2A requires at least one ubatch slot, got {num_ubatches}"
            )
        super().__init__(group, device, num_ubatches)
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            output_shape, dtype
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(lse_shape, torch.float32)
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        output = partial_output.new_empty(
            (num_tokens, self.heads_per_rank, self.head_dim)
        )
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.peer_output_ptrs[ubatch],
            self.peer_lse_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_output[ubatch],
            self.received_lse[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output,
            self.world_size,
            self.rank,
            self.max_num_tokens,
            is_lse_base_on_e,
        )
        return output


@functools.cache
def get_direct_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace | None:
    if not _direct_dcp_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_A2A, _A2A_SUPPORTED_DTYPES
    ):
        return None
    return DirectDCPA2AWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )


def _q_gather_layout_supported(
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    padded_num_heads: int | None,
) -> bool:
    element_size = torch.empty((), dtype=dtype).element_size()
    gathered_num_heads = world_size * heads_per_rank
    storage_num_heads = (
        gathered_num_heads if padded_num_heads is None else padded_num_heads
    )
    return (
        heads_per_rank * head_dim * element_size % 16 == 0
        and storage_num_heads * head_dim * element_size % 16 == 0
    )


class DirectDCPQGatherWorkspace(_DirectDCPWorkspace):
    """Publish query shards directly into the consumer-final symmetric buffer.

    The final buffer is reusable after the downstream DCP output combine. That
    combine orders all ranks after attention has consumed the gathered query.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
        padded_num_heads: int | None = None,
    ) -> None:
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP q-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_num_tokens < 1 or heads_per_rank < 1 or head_dim < 1:
            raise ValueError(
                "Direct DCP q-gather dimensions must be positive, got "
                f"T={max_num_tokens}, H={heads_per_rank}, D={head_dim}"
            )
        gathered_num_heads = group.size() * heads_per_rank
        if not _q_gather_layout_supported(
            group.size(), heads_per_rank, head_dim, dtype, padded_num_heads
        ):
            raise ValueError("Direct DCP q-gather requires 16-byte-aligned query rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP q-gather requires at least two ranks")
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.gathered_num_heads = gathered_num_heads
        self.padded_num_heads = (
            self.gathered_num_heads if padded_num_heads is None else padded_num_heads
        )
        if self.padded_num_heads < self.gathered_num_heads:
            raise ValueError(
                "Direct DCP q-gather padded heads must cover gathered heads: "
                f"{self.padded_num_heads} < {self.gathered_num_heads}"
            )
        self.head_dim = head_dim

        query_shape = (
            num_ubatches,
            max_num_tokens,
            self.padded_num_heads,
            head_dim,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.final_query, _ = self._allocate(query_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        query_multicast_ptrs = self._multicast_ptrs(self.final_query)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(query_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(
            query_ptr and signal_ptr for query_ptr, signal_ptr in self.multicast_ptrs
        ):
            raise RuntimeError(
                "Direct DCP q-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 1))
        torch.accelerator.synchronize()

    def gather(self, local_query: torch.Tensor) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP q-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        if local_query.ndim == 3 and local_query.shape[1] != self.heads_per_rank:
            raise ValueError(
                f"DCP q-gather expected {self.heads_per_rank} local query heads, "
                f"got {local_query.shape[1]}"
            )

        num_tokens = local_query.shape[0]
        output = torch.as_strided(
            self.final_query[ubatch],
            size=(num_tokens, self.gathered_num_heads, self.head_dim),
            stride=(
                self.gathered_num_heads * self.head_dim,
                self.head_dim,
                1,
            ),
        )
        query_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_q_gather(
            local_query,
            output,
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            self.world_size,
            self.rank,
            self.max_num_tokens,
            self.padded_num_heads,
            query_multicast_ptr,
            signal_multicast_ptr,
        )
        return output


@functools.cache
def get_direct_dcp_q_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
    padded_num_heads: int | None = None,
) -> DirectDCPQGatherWorkspace | None:
    if not _direct_dcp_multicast_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_Q_GATHER
    ):
        return None
    if not _q_gather_layout_supported(
        group.world_size, heads_per_rank, head_dim, dtype, padded_num_heads
    ):
        return None
    return DirectDCPQGatherWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
        padded_num_heads,
    )


def kv_gather_slots() -> int:
    """Number of symmetric KV-gather slots per ubatch.

    A slot holds one gathered context window. The serial chunked-context loop
    alternates two; the Kimi-K3 pipelined loop publishes one window while the
    previous one is consumed and needs three (``VLLM_DCP_KV_GATHER_SLOTS``).
    """
    slots = int(envs.VLLM_DCP_KV_GATHER_SLOTS)
    if slots < 2:
        raise ValueError(f"VLLM_DCP_KV_GATHER_SLOTS must be at least 2, got {slots}")
    return slots


_KV_GATHER_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _kv_gather_layout_supported(
    token_dim: int,
    plane_split_dim: int,
    dtype: torch.dtype,
) -> bool:
    """Whether both packed output planes support 16-byte multicast stores."""
    if not 0 < plane_split_dim < token_dim:
        return False
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        plane_split_dim * element_size % 16 == 0
        and (token_dim - plane_split_dim) * element_size % 16 == 0
    )


class DirectDCPKVGatherWorkspace(_DirectDCPWorkspace):
    """Persistent symmetric buffers for direct DCP KV gather.

    Storage is owned by ``(DBO ubatch, buffer slot)``. Different ubatches have
    disjoint buffers and may run independently. Within one ubatch, publishing
    and consumption must remain stream ordered. Before reusing a slot, every
    rank must have consumed it and reached either a gather on another slot or
    another all-rank rendezvous: with S slots a gather into slot s may be
    issued once the rank has consumed the window S-2 gathers back and the
    gather before it has completed. Concurrent same-ubatch gathers from
    multiple streams are not supported.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_gathered_tokens: int,
        token_dim: int,
        plane_split_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _KV_GATHER_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP kv-gather does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP kv-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_gathered_tokens < 1 or token_dim < 1:
            raise ValueError(
                "Direct DCP kv-gather dimensions must be positive, got "
                f"T={max_gathered_tokens}, D={token_dim}"
            )
        if not _kv_gather_layout_supported(token_dim, plane_split_dim, dtype):
            raise ValueError(
                "Direct DCP kv-gather requires two nonempty 16-byte-aligned KV planes."
            )
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP kv-gather requires at least two ranks")
        if max_gathered_tokens % self.world_size != 0:
            raise ValueError(
                "Direct DCP kv-gather capacity must divide evenly across "
                f"ranks: {max_gathered_tokens} % {self.world_size} != 0"
            )
        self.max_gathered_tokens = max_gathered_tokens
        self.token_dim = token_dim
        self.plane_split_dim = plane_split_dim

        num_slots = kv_gather_slots()
        self.num_slots = num_slots
        kv_shape = (num_ubatches, num_slots, max_gathered_tokens, token_dim)
        signal_shape = (num_ubatches, num_slots, self.world_size)
        self.received_kv, self.peer_kv_ptrs = self._allocate(kv_shape, dtype)
        # Host copy for the copy-engine publisher, whose memcpys are issued
        # from the host.
        self.peer_kv_ptrs_host = self.peer_kv_ptrs.cpu()
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )
        kv_multicast_ptrs = self._multicast_ptrs(self.received_kv)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(kv_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        self.uses_multicast = all(
            kv_ptr and signal_ptr for kv_ptr, signal_ptr in self.multicast_ptrs
        )
        self.completion = self.received_signal.new_zeros((num_ubatches, num_slots))
        torch.accelerator.synchronize()

    def gather(
        self,
        local_kv: torch.Tensor,
        dst_rows: torch.Tensor,
        output_tokens: int,
        buffer_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish valid rows into compact request-major KV planes."""
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        # The custom op validates the dynamic tensor geometry, dtype, device,
        # capacity, and slot before launching. Avoid duplicating those checks on
        # this latency-sensitive host path.
        plane_split_dim = self.plane_split_dim
        kv_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            dst_rows,
            self.peer_kv_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output_tokens,
            plane_split_dim,
            buffer_slot,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
            kv_multicast_ptr,
            signal_multicast_ptr,
        )
        return self._slot_planes(ubatch, buffer_slot, output_tokens)

    def gather_dma(
        self,
        local_kv_c: torch.Tensor,
        local_k_pe: torch.Tensor,
        runs: torch.Tensor,
        output_tokens: int,
        buffer_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish plane-separated local rows with the copy engines.

        ``local_kv_c`` [T, plane_split_dim] and ``local_k_pe``
        [T, token_dim - plane_split_dim] hold the padded local rows;
        ``runs`` (CPU int64 [n, 3]: source row, destination row, rows) maps
        each request's valid rows to the compact request-major layout. The
        payload moves with cudaMemcpyAsync, so no SM memory pipeline is busy
        while it is in flight; two single-block kernels release this rank's
        epoch to every peer and wait for every source. Same layout, epoch and
        signal protocol as :meth:`gather`.
        """
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        _dma_kv_gather_op()(
            local_kv_c,
            local_k_pe,
            runs,
            self.peer_kv_ptrs_host[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output_tokens,
            self.plane_split_dim,
            buffer_slot,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
        )
        return self._slot_planes(ubatch, buffer_slot, output_tokens)

    def _slot_planes(
        self, ubatch: int, buffer_slot: int, output_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_dim = self.token_dim
        plane_split_dim = self.plane_split_dim
        slot = self.received_kv[ubatch, buffer_slot].view(-1)
        kv_c_capacity = self.max_gathered_tokens * plane_split_dim
        kv_c = slot[:kv_c_capacity].view(self.max_gathered_tokens, 1, plane_split_dim)
        k_pe = slot[kv_c_capacity:].view(
            self.max_gathered_tokens, 1, token_dim - plane_split_dim
        )
        return kv_c[:output_tokens], k_pe[:output_tokens]


@functools.cache
def _dma_kv_gather_op():
    """The copy-engine final-layout publisher: ``_C.direct_dcp_kv_gather_dma``
    when the built extension provides it, else the same op from the side
    extension named by ``VLLM_K3_DCP_GATHER_ROTATE_LIB``."""
    if hasattr(torch.ops._C, "direct_dcp_kv_gather_dma"):
        return torch.ops._C.direct_dcp_kv_gather_dma
    import os

    path = os.getenv("VLLM_K3_DCP_GATHER_ROTATE_LIB", "")
    if not path:
        raise RuntimeError(
            "The copy-engine DCP KV gather needs direct_dcp_kv_gather_dma "
            "(rebuild the _C extension or set VLLM_K3_DCP_GATHER_ROTATE_LIB)."
        )
    torch.ops.load_library(path)
    op = torch.ops._C_k3ext.direct_dcp_kv_gather_dma
    logger.info_once("DCP KV gather: copy-engine publisher from %s", path)
    return op


def build_dcp_kv_final_layout_runs(
    padded_local_seq_lens: list[int],
    local_context_lens_allranks: list[list[int]],
    local_starts: list[int],
    dcp_rank: int,
) -> torch.Tensor:
    """Per-request runs (source row, destination row, rows) of this rank's
    valid rows in the compact request-major layout, as a CPU int64 [n, 3]
    tensor. The same layout as build_dcp_kv_final_layout_dst_rows: requests
    outermost, and within a request the ranks' valid rows in rank order."""
    runs: list[tuple[int, int, int]] = []
    src_start = 0
    dst_start = 0
    for padded_len, context_lens, local_start in zip(
        padded_local_seq_lens, local_context_lens_allranks, local_starts, strict=True
    ):
        valid = [
            min(max(0, context_len - local_start), padded_len)
            for context_len in context_lens
        ]
        runs.append((src_start, dst_start + sum(valid[:dcp_rank]), valid[dcp_rank]))
        src_start += padded_len
        dst_start += sum(valid)
    return torch.tensor(runs, dtype=torch.int64)


class DCPKVGatherPipeline:
    """Publish gathered context windows on a side stream, one window ahead of
    their consumption on the compute stream.

    Windows are numbered globally across chunks, layers and forwards; window w
    is published into slot ``w % S`` of the direct DCP KV-gather workspace
    (S slots). Before the side stream publishes window w it waits for the
    compute stream's release of window ``w - (S - 1)``, and by stream order
    for the publication of window ``w - 1``. Releases are recorded in window
    order, so the release of ``w - (S - 1)`` also covers ``w - S``, the
    slot's previous occupant, on this rank.

    Peers are covered by the rendezvous inside every gather: a rank's
    publication of window w starts only after its gather of ``w - 1``
    completed, which requires every rank to have published ``w - 1``, and a
    rank publishes ``w - 1`` only after releasing ``w - 1 - (S - 1) = w - S``.
    No rank can therefore overwrite a peer's slot before the peer released
    it. With S = 3 the gather of window w runs while windows ``w - 2`` and
    ``w - 1`` are still being projected and attended.

    The side stream never allocates: the gather kernels read the chunk
    metadata and the paged cache and write persistent workspaces, and the
    published planes are views of the symmetric buffer.
    """

    MIN_SLOTS = 3

    def __init__(self, device: torch.device, num_slots: int) -> None:
        if num_slots < self.MIN_SLOTS:
            raise ValueError(
                "The DCP KV-gather pipeline needs at least "
                f"{self.MIN_SLOTS} gather slots, got {num_slots} "
                "(VLLM_DCP_KV_GATHER_SLOTS)."
            )
        self.stream = torch.cuda.Stream(device=device)
        self.num_slots = num_slots
        self.window = 0
        self._published = [torch.cuda.Event() for _ in range(num_slots)]
        self._released = [torch.cuda.Event() for _ in range(num_slots)]

    def begin(self) -> None:
        """Order the side stream after the compute stream's work so far: the
        chunk metadata, the paged-cache writes and the workspace reads of the
        previous layer."""
        self.stream.wait_stream(torch.cuda.current_stream())

    def publish(self, gather) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        """Run ``gather(slot)`` on the side stream once window
        ``w - (S - 1)`` has been released; return the slot and the result."""
        window = self.window
        self.window = window + 1
        slot = window % self.num_slots
        # Window w - (S - 1) occupies slot (w + 1) % S; its release is the
        # latest record on that slot's event when this window is published
        # (the compute stream releases window w - 1 only afterwards).
        guard = (slot + 1) % self.num_slots
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self._released[guard])
            result = gather(slot)
            self._published[slot].record(self.stream)
        return slot, result

    def acquire(self, slot: int) -> None:
        """Make the compute stream wait for the slot's published window."""
        torch.cuda.current_stream().wait_event(self._published[slot])

    def release(self, slot: int) -> None:
        """Record that the compute stream has finished reading the slot."""
        self._released[slot].record(torch.cuda.current_stream())


_kv_gather_pipelines: dict[tuple[torch.device, int], DCPKVGatherPipeline] = {}


def get_dcp_kv_gather_pipeline(
    device: torch.device, num_slots: int
) -> DCPKVGatherPipeline:
    """One pipeline per device and DBO ubatch: the layers of a ubatch share
    its gather buffers, so they share the window counter and the slot
    events."""
    key = (device, dbo_current_ubatch_id())
    pipeline = _kv_gather_pipelines.get(key)
    if pipeline is None:
        pipeline = DCPKVGatherPipeline(device, num_slots)
        _kv_gather_pipelines[key] = pipeline
        logger.info_once(
            "Direct DCP KV gather: publishing context windows one ahead on a "
            "side stream (%d slots).",
            num_slots,
        )
    elif pipeline.num_slots != num_slots:
        raise RuntimeError(
            "DCP KV-gather slot count changed between layers: "
            f"{pipeline.num_slots} vs {num_slots}"
        )
    return pipeline


@functools.cache
def get_direct_dcp_kv_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    token_dim: int,
    plane_split_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPKVGatherWorkspace | None:
    if not _direct_dcp_enabled(
        group,
        dtype,
        envs.VLLM_USE_DIRECT_DCP_KV_GATHER,
        _KV_GATHER_SUPPORTED_DTYPES,
    ):
        return None
    if not _kv_gather_layout_supported(token_dim, plane_split_dim, dtype):
        return None
    return DirectDCPKVGatherWorkspace(
        group.device_group,
        device,
        max_gathered_tokens,
        token_dim,
        plane_split_dim,
        dtype,
        num_ubatches,
    )


class MLADCPKVGather:
    """Own the exact DCP context-KV collective independently of decode DCP.

    Backends such as B12X own their decode query/output exchange, but chunked
    prefill still needs to exchange compressed KV. Keeping that collective in
    a small standalone object lets those backends use the final-layout
    symmetric publisher without duplicating the decode collectives.
    """

    def __init__(
        self,
        group: GroupCoordinator,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.device = torch.device(device)
        self.num_ubatches = max(num_ubatches, 1)
        self._direct_kv_gather_workspace: DirectDCPKVGatherWorkspace | None = None
        self._kv_gather: Callable[[torch.Tensor, torch.Tensor], object] | None = None

    def init_kv_gather(
        self,
        max_gathered_tokens: int,
        token_dim: int,
        plane_split_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        """Select the KV collective before allocating its local scratch.

        Returns whether the direct final-layout publisher was selected. That
        path only needs one rank's local rows; the fallback additionally needs
        a rank-major all-gather destination.
        """
        world_size = self.group.world_size
        if max_gathered_tokens <= 0 or max_gathered_tokens % world_size != 0:
            raise ValueError(
                "DCP KV gather capacity must be positive and divide evenly "
                f"across {world_size} ranks, got {max_gathered_tokens}"
            )
        if token_dim <= 0:
            raise ValueError(
                f"DCP KV gather token dimension must be positive: {token_dim}"
            )

        direct_workspace = get_direct_dcp_kv_gather_workspace(
            self.group,
            self.device,
            max_gathered_tokens,
            token_dim,
            plane_split_dim,
            dtype,
            self.num_ubatches,
        )
        self._direct_kv_gather_workspace = direct_workspace
        if direct_workspace is not None:
            transport = (
                "NVLS multicast" if direct_workspace.uses_multicast else "PCIe peer"
            )
            logger.info_once(
                "Using direct symmetric-memory DCP final-layout KV %s publisher "
                "for MLA.",
                transport,
            )
            self._kv_gather = None
        else:
            self._kv_gather = functools.partial(
                torch.distributed.all_gather_into_tensor,
                group=self.group.device_group,
            )
        return direct_workspace is not None

    @property
    def use_direct_kv_gather(self) -> bool:
        return self._direct_kv_gather_workspace is not None

    @property
    def kv_gather_slots(self) -> int:
        """Symmetric slots per ubatch of the direct publisher (0 without it)."""
        workspace = self._direct_kv_gather_workspace
        return 0 if workspace is None else workspace.num_slots

    def direct_kv_gather_dma(
        self,
        local_kv_c: torch.Tensor,
        local_k_pe: torch.Tensor,
        runs: torch.Tensor,
        output_tokens: int,
        buffer_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self._direct_kv_gather_workspace
        if workspace is None:
            raise RuntimeError("direct DCP KV gather is not enabled")
        return workspace.gather_dma(
            local_kv_c, local_k_pe, runs, output_tokens, buffer_slot
        )

    def kv_gather(
        self,
        gathered_kv: torch.Tensor,
        local_kv: torch.Tensor,
    ) -> object:
        kv_gather = self._kv_gather
        if kv_gather is None:
            raise RuntimeError("NCCL DCP KV gather is not selected")
        return kv_gather(gathered_kv, local_kv)

    def direct_kv_gather(
        self,
        local_kv: torch.Tensor,
        dst_rows: torch.Tensor,
        output_tokens: int,
        buffer_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self._direct_kv_gather_workspace
        if workspace is None:
            raise RuntimeError("direct DCP KV gather is not enabled")
        return workspace.gather(
            local_kv,
            dst_rows,
            output_tokens,
            buffer_slot,
        )


class DCPCombine(Protocol):
    def __call__(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor: ...


class MLADCPManager(MLADCPKVGather):
    """Select and own layer-level collective implementations for MLA DCP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        num_heads: int,
        query_head_dim: int,
        output_head_dim: int,
        query_dtype: torch.dtype,
        output_dtype: torch.dtype,
        padded_num_heads: int | None,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        super().__init__(
            get_dcp_group(),
            device,
            parallel_config.num_ubatches,
        )
        self.max_num_tokens = get_dcp_workspace_max_num_tokens(vllm_config)
        self.use_a2a = parallel_config.dcp_comm_backend == "a2a"
        self.padded_num_heads = padded_num_heads
        self.combine = self._init_combine(
            num_heads,
            output_head_dim,
            output_dtype,
            is_lse_base_on_e,
            use_pcp,
        )
        self.query_gather = (
            None
            if use_pcp
            else self._init_query_gather(
                num_heads,
                query_head_dim,
                query_dtype,
            )
        )

    def _init_combine(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> DCPCombine:
        direct_workspace = None
        if self.use_a2a:
            direct_workspace = get_direct_dcp_a2a_workspace(
                self.group,
                self.device,
                self.max_num_tokens,
                num_heads,
                head_dim,
                dtype,
                self.num_ubatches,
            )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP A2A for MLA.")
            return functools.partial(
                direct_workspace.lse_reduce,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
        return functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _init_query_gather(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        direct_workspace = get_direct_dcp_q_gather_workspace(
            self.group,
            self.device,
            self.max_num_tokens,
            num_heads,
            head_dim,
            dtype,
            self.num_ubatches,
            self.padded_num_heads,
        )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP query gather for MLA.")
            return direct_workspace.gather
        return self._gather_query

    def _gather_query(self, query: torch.Tensor) -> torch.Tensor:
        query = self.group.all_gather(query, dim=1)
        if self.padded_num_heads is not None:
            query = reserve_query_head_storage(query, self.padded_num_heads)
        return query
