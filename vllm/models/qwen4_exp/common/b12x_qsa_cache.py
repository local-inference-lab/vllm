# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed QSA selector-cache views and request metadata helpers."""

from __future__ import annotations

import math

import torch

from vllm.utils.b12x import get_b12x_qsa
from vllm.v1.kv_cache_interface import AttentionSpec


def qsa_padded_page_size_bytes(
    spec: AttentionSpec,
    *,
    compress_ratio: int,
    index_head_dim: int,
) -> int:
    """Return one native B12x main page plus its packed selector tail."""
    main_page_size = int(spec.block_size)
    if main_page_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA page size and compression ratio must be positive")

    qsa = get_b12x_qsa()
    if qsa is None:
        raise RuntimeError("b12x QSA is required to size the selector cache")

    # Hybrid-cache alignment probes backend storage with a one-token page. Ask
    # b12x for the smallest ratio-aligned equivalent and scale its reported
    # compressed storage back to the probe width. Real manager pages are already
    # ratio-aligned and therefore consume the returned byte count directly.
    geometry_page_size = math.lcm(main_page_size, int(compress_ratio))
    kv_dtype = torch.float8_e4m3fn if spec.dtype == torch.uint8 else spec.dtype
    requirements = qsa.cache_requirements(
        main_page_size=geometry_page_size,
        kv_heads=int(spec.num_kv_heads),
        head_dim=int(spec.head_size),
        compress_ratio=compress_ratio,
        index_head_dim=index_head_dim,
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
    )
    scaled_bytes, remainder = divmod(
        int(requirements.compressed_page_nbytes) * main_page_size,
        geometry_page_size,
    )
    if remainder:
        raise ValueError("QSA selector bytes per manager page must be integral")
    return int(spec.unpadded_page_size_bytes) + scaled_bytes


def qsa_compressed_cache_view(
    kv_cache: torch.Tensor,
    *,
    compress_ratio: int,
    index_head_dim: int,
) -> torch.Tensor:
    """Return the zero-copy compressed-key view in a padded BLHNC page.

    ``kv_cache`` is the native B12x logical main-cache view
    ``[pages, 2, page_size, packed_kv_width]``. Its logical shape excludes page
    padding. The returned view starts immediately after each page's native K/V
    payload and advances by the input's physical page stride, which also works
    when multiple layers are interleaved inside the BLHNC allocation.
    """
    if kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8):
        raise TypeError("QSA main cache must be BF16 or FP8 E4M3FN storage")
    if kv_cache.ndim != 4 or int(kv_cache.shape[1]) != 2:
        raise ValueError(
            "QSA main cache must have logical shape "
            "[pages, 2, page_size, packed_kv_width]"
        )
    packed_width = int(kv_cache.shape[3])
    main_page_size = int(kv_cache.shape[2])
    expected_inner_strides = (
        main_page_size * packed_width,
        packed_width,
        1,
    )
    if tuple(map(int, kv_cache.stride()[1:])) != expected_inner_strides:
        raise ValueError("QSA main cache must have contiguous logical page payloads")
    if compress_ratio <= 0 or index_head_dim <= 0:
        raise ValueError("QSA compression ratio and index dim must be positive")

    num_pages = int(kv_cache.shape[0])
    if num_pages <= 0 or main_page_size <= 0:
        raise ValueError("QSA cache must contain positive page capacity")
    if main_page_size % compress_ratio:
        raise ValueError("QSA main page size must be divisible by compress_ratio")

    compressed_page_size = main_page_size // compress_ratio
    main_page_elements = math.prod(map(int, kv_cache.shape[1:]))
    main_page_nbytes = main_page_elements * int(kv_cache.element_size())
    tail_elements = compressed_page_size * int(index_head_dim)
    tail_nbytes = tail_elements * torch.bfloat16.itemsize
    physical_page_stride_nbytes = int(kv_cache.stride(0)) * int(kv_cache.element_size())
    if physical_page_stride_nbytes < main_page_nbytes + tail_nbytes:
        raise ValueError("QSA physical page stride does not preserve the selector tail")
    # For BLHNC with multiple layers, stride(0) spans every interleaved layer
    # page and is larger than one padded page. Storage-capacity validation below
    # is therefore the reliable check; equality is neither required nor wanted.
    tail_offset_nbytes = (
        int(kv_cache.storage_offset()) * int(kv_cache.element_size()) + main_page_nbytes
    )
    tail_end_nbytes = (
        tail_offset_nbytes + (num_pages - 1) * physical_page_stride_nbytes + tail_nbytes
    )
    storage_nbytes = int(kv_cache.untyped_storage().nbytes())
    if tail_end_nbytes > storage_nbytes:
        raise ValueError(
            "QSA main cache storage does not include the required selector tail"
        )
    if tail_offset_nbytes % torch.bfloat16.itemsize or (
        physical_page_stride_nbytes % torch.bfloat16.itemsize
    ):
        raise ValueError("QSA selector tail is not BF16 aligned")
    if storage_nbytes % torch.bfloat16.itemsize:
        raise ValueError("QSA cache storage is not BF16 aligned")
    bf16_storage = torch.empty(0, dtype=torch.bfloat16, device=kv_cache.device).set_(
        kv_cache.untyped_storage(),
        0,
        (storage_nbytes // torch.bfloat16.itemsize,),
        (1,),
    )
    return bf16_storage.as_strided(
        (num_pages, compressed_page_size, int(index_head_dim)),
        (
            physical_page_stride_nbytes // torch.bfloat16.itemsize,
            int(index_head_dim),
            1,
        ),
        storage_offset=tail_offset_nbytes // torch.bfloat16.itemsize,
    )


def canonical_qsa_rope_positions(
    positions: torch.Tensor,
    *,
    num_rows: int,
    position_axes: int,
) -> torch.Tensor:
    """Return scalar or MRoPE coordinates as ``[rows, axes]`` int64."""
    if position_axes not in (1, 3):
        raise ValueError("QSA position_axes must be 1 or 3")
    if positions.ndim == 1:
        if position_axes != 1:
            raise ValueError("three-axis QSA requires [3, rows] MRoPE positions")
        result = positions[:num_rows].unsqueeze(1)
    elif positions.ndim == 2:
        if int(positions.shape[0]) != position_axes:
            raise ValueError(
                f"QSA positions must have {position_axes} axes, got "
                f"{tuple(positions.shape)}"
            )
        result = positions[:, :num_rows].transpose(0, 1)
    else:
        raise ValueError("QSA positions must have shape [rows] or [axes, rows]")
    if tuple(result.shape) != (num_rows, position_axes):
        raise ValueError("QSA positions do not cover every active row")
    return result if result.dtype == torch.int64 else result.to(torch.int64)


def qsa_logical_positions(
    *,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Derive request-relative logical positions for packed query rows."""
    rows = int(request_ids.numel())
    if rows == 0:
        return sequence_lengths.new_empty((0,), dtype=torch.int64)
    if sequence_lengths.ndim != 1 or request_ids.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("QSA request IDs and query starts must be one-dimensional")
    num_requests = int(sequence_lengths.shape[0])
    if int(query_start_loc.numel()) != num_requests + 1:
        raise ValueError("QSA query starts must contain one terminal per request")
    if num_requests == 0:
        return request_ids.new_full((rows,), -1, dtype=torch.int64)
    request_lens = query_start_loc[1:] - query_start_loc[:-1]
    requests = request_ids.to(torch.long)
    valid = (requests >= 0) & (requests < num_requests)
    safe_requests = requests.clamp(0, num_requests - 1)
    row_ids = torch.arange(rows, dtype=torch.int64, device=request_ids.device)
    positions = (
        sequence_lengths.index_select(0, safe_requests).to(torch.int64)
        - request_lens.index_select(0, safe_requests).to(torch.int64)
        + row_ids
        - query_start_loc.index_select(0, safe_requests).to(torch.int64)
    )
    return torch.where(valid, positions, positions.new_full((), -1))


def qsa_compressed_slot_mapping(
    *,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    logical_positions: torch.Tensor,
    main_page_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Map completed-group rows into the tail sharing the main block table."""
    if block_table.ndim != 2 or request_ids.shape != logical_positions.shape:
        raise ValueError("QSA compressed-slot metadata has incompatible shapes")
    if main_page_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA page size and compression ratio must be positive")
    if main_page_size % compress_ratio:
        raise ValueError("QSA page size must be positive and divisible by ratio")
    if not int(block_table.shape[0]) or not int(block_table.shape[1]):
        return logical_positions.new_full(logical_positions.shape, -1)
    compressed_page_size = main_page_size // compress_ratio
    positions = logical_positions.to(torch.long)
    requests = request_ids.to(torch.long)
    group_ids = torch.div(positions.clamp_min(0), compress_ratio, rounding_mode="floor")
    table_columns = torch.div(group_ids, compressed_page_size, rounding_mode="floor")
    valid = (
        (positions >= 0)
        & ((positions + 1).remainder(compress_ratio) == 0)
        & (requests >= 0)
        & (requests < int(block_table.shape[0]))
        & (table_columns < int(block_table.shape[1]))
    )
    safe_requests = requests.clamp(0, max(int(block_table.shape[0]) - 1, 0))
    safe_columns = table_columns.clamp(0, max(int(block_table.shape[1]) - 1, 0))
    physical_pages = block_table[safe_requests, safe_columns].to(torch.long)
    valid &= physical_pages >= 0
    slots = physical_pages * compressed_page_size + group_ids.remainder(
        compressed_page_size
    )
    return torch.where(valid, slots, slots.new_full((), -1))


def qsa_raw_slot_mapping(
    *,
    state_slot_ids: torch.Tensor,
    request_ids: torch.Tensor,
    logical_positions: torch.Tensor,
    raw_ring_capacity: int,
) -> torch.Tensor:
    """Map packed rows into each request's persistent circular raw ring."""
    if raw_ring_capacity <= 0:
        raise ValueError("QSA raw ring capacity must be positive")
    if (
        state_slot_ids.ndim != 1
        or request_ids.ndim != 1
        or request_ids.shape != logical_positions.shape
    ):
        raise ValueError("QSA raw-slot metadata has incompatible shapes")
    requests = request_ids.to(torch.long)
    num_requests = int(state_slot_ids.shape[0])
    if num_requests == 0:
        return logical_positions.new_full(logical_positions.shape, -1)
    valid = (requests >= 0) & (requests < num_requests) & (logical_positions >= 0)
    safe_requests = requests.clamp(0, num_requests - 1)
    state_slots = state_slot_ids.index_select(0, safe_requests).to(torch.long)
    valid &= state_slots >= 0
    slots = state_slots * raw_ring_capacity + logical_positions.remainder(
        raw_ring_capacity
    )
    return torch.where(valid, slots, slots.new_full((), -1))


__all__ = [
    "canonical_qsa_rope_positions",
    "qsa_compressed_cache_view",
    "qsa_compressed_slot_mapping",
    "qsa_logical_positions",
    "qsa_padded_page_size_bytes",
    "qsa_raw_slot_mapping",
]
