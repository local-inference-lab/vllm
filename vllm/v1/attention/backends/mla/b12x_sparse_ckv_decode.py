# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning and deterministic union construction for sparse DCP CKV decode.

The B12X transport only exchanges fixed-width records.  This module owns the
vLLM policy which decides which records each request needs and how MTP rows map
into one per-request, densely packed sparse workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from vllm.distributed.parallel_state import register_model_parallel_cleanup_hook


@dataclass(frozen=True)
class SparseCKVDecodeLayout:
    dcp_world_size: int
    topk: int
    rows_per_request: int
    max_requests: int
    per_request_capacity: int
    pool_records: int
    max_fast_requests: int
    record_bytes: int
    prefetch_depth: int
    workspace_slots: int

    @property
    def workspace_bytes(self) -> int:
        return self.workspace_slots * self.pool_records * self.record_bytes

    def active_records(self, num_requests: int) -> int:
        if not 1 <= num_requests <= self.max_fast_requests:
            raise ValueError(
                f"request count {num_requests} is outside sparse capacity "
                f"[1, {self.max_fast_requests}]"
            )
        return num_requests * self.per_request_capacity


def plan_sparse_ckv_decode(
    *,
    dcp_world_size: int,
    topk: int,
    rows_per_request: int,
    max_requests: int,
    pool_records: int,
    record_bytes: int,
    prefetch_depth: int,
) -> SparseCKVDecodeLayout:
    """Return the static pooled-workspace layout for selected-record decode."""
    if not 2 <= dcp_world_size <= 8:
        raise ValueError("selected-record CKV decode supports DCP world sizes 2-8")
    if min(topk, rows_per_request, max_requests, record_bytes) <= 0:
        raise ValueError("topk, rows, requests, and record width must be positive")
    if prefetch_depth < 0:
        raise ValueError("prefetch depth cannot be negative")

    per_request_capacity = topk * rows_per_request
    requested_pool = int(pool_records)
    if requested_pool <= 0:
        requested_pool = max_requests * per_request_capacity
    max_fast_requests = min(max_requests, requested_pool // per_request_capacity)
    if max_fast_requests <= 0:
        raise ValueError(
            "selected-record pool cannot hold one request's worst-case MTP union"
        )
    # Capacity beyond the largest schedulable batch only wastes VRAM.
    effective_pool = min(
        requested_pool,
        max_requests * per_request_capacity,
    )
    return SparseCKVDecodeLayout(
        dcp_world_size=dcp_world_size,
        topk=topk,
        rows_per_request=rows_per_request,
        max_requests=max_requests,
        per_request_capacity=per_request_capacity,
        pool_records=effective_pool,
        max_fast_requests=max_fast_requests,
        record_bytes=record_bytes,
        prefetch_depth=prefetch_depth,
        workspace_slots=prefetch_depth + 1,
    )


def sparse_decode_batch_eligible(
    layout: SparseCKVDecodeLayout,
    *,
    num_requests: int,
    num_rows: int,
    max_query_len: int,
    num_actual_tokens: int,
    has_required_metadata: bool,
) -> bool:
    """Pure policy gate used before entering the selected-record fast path."""
    if not has_required_metadata or not 1 <= num_requests <= layout.max_fast_requests:
        return False
    if not 1 <= num_rows <= num_requests * layout.rows_per_request:
        return False
    if num_rows % num_requests:
        return False
    rows_per_request = num_rows // num_requests
    return (
        rows_per_request <= layout.rows_per_request
        and max_query_len == rows_per_request
        and num_actual_tokens == num_rows
    )


def sparse_decode_prefetch_targets(
    layer_index: int,
    depth: int,
    emits_topk_by_layer: list[bool | None],
) -> list[int]:
    """Return consecutive Shared layers following one Full/indexer layer."""
    targets: list[int] = []
    for distance in range(1, max(0, depth) + 1):
        target = layer_index + distance
        if target >= len(emits_topk_by_layer):
            break
        emits_topk = emits_topk_by_layer[target]
        if emits_topk is None or emits_topk:
            break
        targets.append(target)
    return targets


def owner_and_local_ordinal(
    logical_token: int,
    *,
    dcp_world_size: int,
    interleave: int,
) -> tuple[int, int]:
    """Map one global logical token to its DCP owner and owner-local ordinal."""
    if logical_token < 0:
        return -1, -1
    if dcp_world_size <= 0 or interleave <= 0:
        raise ValueError("DCP world size and interleave must be positive")
    owner = (logical_token // interleave) % dcp_world_size
    local_ordinal = (
        logical_token // (dcp_world_size * interleave) * interleave
        + logical_token % interleave
    )
    return owner, local_ordinal


def dense_union_remap_reference(
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU oracle: stable dense union and exact remap for every group.

    ``indices`` has shape ``[..., rows, topk]``.  Every leading group receives
    an independent stable union in flattened row order.
    """
    if indices.device.type != "cpu" or indices.dtype != torch.int32:
        raise ValueError("reference union requires a CPU int32 tensor")
    if indices.ndim < 2:
        raise ValueError("indices must include row and top-k dimensions")
    rows, topk = indices.shape[-2:]
    capacity = rows * topk
    groups = indices.reshape(-1, capacity)
    union = torch.full_like(groups, -1)
    remap = torch.full_like(groups, -1)
    counts = torch.zeros(groups.shape[0], dtype=torch.int32)
    for group_index, row in enumerate(groups.tolist()):
        slots: dict[int, int] = {}
        for input_index, token in enumerate(row):
            if token < 0:
                continue
            slot = slots.get(token)
            if slot is None:
                slot = len(slots)
                slots[token] = slot
                union[group_index, slot] = token
            remap[group_index, input_index] = slot
        counts[group_index] = len(slots)
    leading = indices.shape[:-2]
    return (
        union.reshape(*leading, capacity),
        remap.reshape(indices.shape),
        counts.reshape(leading),
    )


class SparseCKVDecodeState:
    """Persistent graph-safe union/remap state for one execution lane."""

    def __init__(
        self,
        *,
        layout: SparseCKVDecodeLayout,
        device: torch.device,
        exchange,
    ) -> None:
        self.layout = layout
        self.exchange = exchange
        capacity = layout.per_request_capacity
        hash_capacity = 1 << (2 * capacity - 1).bit_length()
        max_rows = layout.max_requests * layout.rows_per_request
        state_shape = (layout.dcp_world_size, layout.max_requests)
        self.all_rank_topk = torch.empty(
            (layout.dcp_world_size * max_rows, layout.topk),
            dtype=torch.int32,
            device=device,
        )
        self.union_indices = torch.empty(
            (*state_shape, capacity), dtype=torch.int32, device=device
        )
        self.remap = torch.empty(
            (
                *state_shape,
                layout.rows_per_request,
                layout.topk,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.union_counts = torch.empty(state_shape, dtype=torch.int32, device=device)
        self.hash_keys = torch.empty(
            (*state_shape, hash_capacity), dtype=torch.int32, device=device
        )
        self.hash_first_positions = torch.empty_like(self.hash_keys)
        self.first_to_dense = torch.empty_like(self.union_indices)
        self.local_slots = torch.empty_like(self.union_indices)
        # PCIeSelectedRecordExchange requires one contiguous destination-major
        # index plane.  A [:num_requests] view of local_slots retains the
        # max-request row stride and is therefore non-contiguous.  Preallocate
        # the eight small exact-shape planes once so decode never allocates or
        # copies through a temporary contiguous() tensor during graph replay.
        self.transport_local_slots = {
            request_count: torch.empty(
                (
                    layout.dcp_world_size,
                    request_count * layout.per_request_capacity,
                ),
                dtype=torch.int32,
                device=device,
            )
            for request_count in range(1, layout.max_fast_requests + 1)
        }
        self.selected_indices = torch.empty(
            (max_rows, layout.topk), dtype=torch.int32, device=device
        )
        # Cross-layer prefetch outlives a WorkspaceManager borrow. Keep the
        # payload in lane-owned storage so unrelated scratch users cannot
        # overwrite S1/S2/S3 records before attention consumes them.
        self.payload_workspace = torch.empty(
            (
                layout.workspace_slots,
                layout.pool_records,
                layout.record_bytes,
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.request_ids = torch.arange(
            layout.max_requests, dtype=torch.int32, device=device
        )
        self.patch_slots = torch.empty(max_rows, dtype=torch.int64, device=device)
        self.complete_events = [
            torch.cuda.Event(blocking=False) for _ in range(layout.workspace_slots)
        ]


_SELECTED_RECORD_EXCHANGES: dict[tuple[object, ...], object] = {}
_SELECTED_RECORD_STREAMS: dict[int, torch.cuda.Stream] = {}
_UNION_PREFLIGHT_RESULTS: dict[tuple[int, str, int | None], tuple[bool, str]] = {}


def close_selected_record_exchanges() -> None:
    """Release IPC slabs before their DCP process groups are destroyed."""
    exchanges = list(
        {id(value): value for value in _SELECTED_RECORD_EXCHANGES.values()}.values()
    )
    _SELECTED_RECORD_EXCHANGES.clear()
    _SELECTED_RECORD_STREAMS.clear()
    _UNION_PREFLIGHT_RESULTS.clear()

    first_error: Exception | None = None
    for exchange in exchanges:
        try:
            exchange.close()
        except Exception as exc:  # pragma: no cover - defensive shutdown path
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError(
            "failed to close a selected-record exchange"
        ) from first_error


register_model_parallel_cleanup_hook(close_selected_record_exchanges)


def get_selected_record_exchange(
    *,
    process_group: ProcessGroup,
    device: torch.device,
    layout: SparseCKVDecodeLayout,
    lane_key: tuple[str, int],
):
    """Return one generic B12X exchange per target/draft execution lane."""
    key = (
        id(process_group),
        device.type,
        device.index,
        layout.dcp_world_size,
        layout.pool_records,
        layout.record_bytes,
        lane_key,
    )
    exchange = _SELECTED_RECORD_EXCHANGES.get(key)
    if exchange is None:
        from sparkinfer.comm.pcie import SelectedRecordExchange

        exchange = SelectedRecordExchange.from_process_group(
            process_group=process_group,
            device=device,
            max_records=layout.pool_records,
            record_bytes=layout.record_bytes,
        )
        _SELECTED_RECORD_EXCHANGES[key] = exchange
    if id(exchange) not in _SELECTED_RECORD_STREAMS:
        _SELECTED_RECORD_STREAMS[id(exchange)] = torch.cuda.Stream(device=device)
    return exchange


def get_selected_record_stream(exchange) -> torch.cuda.Stream:
    """Return the dedicated stream created with one stream-affine exchange."""
    stream = _SELECTED_RECORD_STREAMS.get(id(exchange))
    if stream is None:
        raise RuntimeError("selected-record exchange has no execution stream")
    return stream


@lru_cache(maxsize=1)
def _load_union_extension():
    source = Path(__file__).with_name("b12x_sparse_ckv_union.cu")
    return load(
        name="vllm_b12x_sparse_ckv_union_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def preload_dense_union_extension() -> None:
    """Compile the union helper before CUDA graph warmup or capture."""
    _load_union_extension()


def preload_dense_union_extension_consistently(
    process_group: ProcessGroup,
    device: torch.device,
) -> None:
    """Preload on every DCP rank and fail or proceed as one group."""
    key = (id(process_group), device.type, device.index)
    cached = _UNION_PREFLIGHT_RESULTS.get(key)
    if cached is not None:
        ok, message = cached
        if not ok:
            raise RuntimeError(message)
        return

    local_error: Exception | None = None
    try:
        preload_dense_union_extension()
    except Exception as exc:  # pragma: no cover - exercised by fault injection
        local_error = exc

    status = torch.tensor(
        [0 if local_error is not None else 1],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=process_group)
    globally_ok = bool(status.item())
    if globally_ok:
        _UNION_PREFLIGHT_RESULTS[key] = (True, "")
        return

    message = (
        f"selected-record CKV union extension failed on this DCP rank: {local_error}"
        if local_error is not None
        else "selected-record CKV union extension failed on another DCP rank"
    )
    _UNION_PREFLIGHT_RESULTS[key] = (False, message)
    raise RuntimeError(message)


def build_dense_union_remap(
    indices: torch.Tensor,
    union_indices: torch.Tensor,
    remap: torch.Tensor,
    union_counts: torch.Tensor,
    hash_keys: torch.Tensor,
    hash_first_positions: torch.Tensor,
    first_to_dense: torch.Tensor,
    *,
    num_requests: int,
) -> None:
    """Build stable dense per-destination/per-request unions on CUDA."""
    _load_union_extension().build_dense_union_remap(
        indices,
        union_indices,
        remap,
        union_counts,
        hash_keys,
        hash_first_positions,
        first_to_dense,
        int(num_requests),
    )
