# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 four-token pool formation and expansion kernels."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_HEAD_DIM = 128
_POOL_SIZE = 4
_FP8_MAX = 448.0


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _fwht128(x):
    x = _fwht_stage(x, 128, 64, 1)
    x = _fwht_stage(x, 128, 32, 2)
    x = _fwht_stage(x, 128, 16, 4)
    x = _fwht_stage(x, 128, 8, 8)
    x = _fwht_stage(x, 128, 4, 16)
    x = _fwht_stage(x, 128, 2, 32)
    x = _fwht_stage(x, 128, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _fwht_quant_kernel(
    q,
    q_out,
    scale_out,
    rows,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = row < rows
    dim = tl.arange(0, HEAD_DIM)
    x = tl.load(
        q + row[:, None] * HEAD_DIM + dim[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    n: tl.constexpr = BLOCK_R * HEAD_DIM
    x = tl.reshape(x, (n,))
    x = _fwht_stage(x, n, BLOCK_R * 64, 1)
    x = _fwht_stage(x, n, BLOCK_R * 32, 2)
    x = _fwht_stage(x, n, BLOCK_R * 16, 4)
    x = _fwht_stage(x, n, BLOCK_R * 8, 8)
    x = _fwht_stage(x, n, BLOCK_R * 4, 16)
    x = _fwht_stage(x, n, BLOCK_R * 2, 32)
    x = _fwht_stage(x, n, BLOCK_R, 64)
    x = (x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, HEAD_DIM))
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    quantized = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX)
    tl.store(
        q_out + row[:, None] * HEAD_DIM + dim[None, :],
        quantized,
        mask=row_mask[:, None],
    )
    tl.store(scale_out + row, scale, mask=row_mask)


def fwht128_quant_fp8(
    q: torch.Tensor,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    """Rotate and per-vector quantize contiguous BF16 rows into caller storage."""
    if q.ndim != 2 or q.shape[1] != _HEAD_DIM or q.dtype != torch.bfloat16:
        raise ValueError("GLM index queries must be contiguous BF16 [rows, 128]")
    if not q.is_contiguous():
        raise ValueError("GLM index queries must be contiguous")
    rows = int(q.shape[0])
    if q_out.shape != q.shape or q_out.dtype != torch.float8_e4m3fn:
        raise ValueError("GLM FP8 query output must match the query shape")
    if scale_out.shape != (rows,) or scale_out.dtype != torch.float32:
        raise ValueError("GLM query scales must be FP32 [rows]")
    if rows:
        block_r = 32
        _fwht_quant_kernel[(triton.cdiv(rows, block_r),)](
            q,
            q_out,
            scale_out,
            rows,
            HEAD_DIM=_HEAD_DIM,
            FP8_MAX=_FP8_MAX,
            BLOCK_R=block_r,
            num_warps=2,
        )


@triton.jit
def _write_pool(
    cache_fp8,
    cache_fp32,
    cache_location,
    x,
    dim,
    dim_mask,
    write_mask,
    page_stride_bytes,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    x = _fwht128(x.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    location = cache_location.to(tl.int64)
    page = location // PAGE_SIZE
    offset = location - page * PAGE_SIZE
    page_bytes = page * page_stride_bytes.to(tl.int64)
    k_offsets = page_bytes + offset * HEAD_DIM + dim.to(tl.int64)
    scale_byte_offset = page_bytes + PAGE_SIZE * HEAD_DIM + offset * 4
    tl.store(cache_fp8 + k_offsets, quantized, mask=dim_mask & write_mask)
    tl.store(cache_fp32 + scale_byte_offset // 4, scale, mask=write_mask)


@triton.jit
def _decode_update_kernel(
    cache_fp8,
    cache_fp32,
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    ape,
    slot_mapping,
    positions,
    page_stride_bytes,
    model_block_size,
    parent_stride_pages,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PACKED_MAIN_SLOTS: tl.constexpr,
):
    request = tl.program_id(0)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request)
    end = tl.load(query_start_loc + request + 1)
    dim = tl.arange(0, BLOCK_D)
    dim_mask = (dim < HEAD_DIM) & (state_slot >= 0)
    for row in tl.range(start, end):
        position = tl.load(positions + row).to(tl.int64)
        physical_slot = position % POOL_SIZE
        current_key = tl.load(
            key + row * key_stride_0 + dim, mask=dim_mask, other=0.0
        ).to(tl.float32)
        current_gate = tl.load(
            gate + row * gate_stride_0 + dim, mask=dim_mask, other=0.0
        ).to(tl.float32)
        cache_location = tl.load(slot_mapping + row).to(tl.int64)
        if PACKED_MAIN_SLOTS:
            parent_page = cache_location // model_block_size
            model_page_offset = cache_location - parent_page * model_block_size
            pool_offset = model_page_offset // POOL_SIZE
            child_page = pool_offset // PAGE_SIZE
            child_offset = pool_offset - child_page * PAGE_SIZE
            cache_location = (
                parent_page * parent_stride_pages + child_page
            ) * PAGE_SIZE + child_offset
            write_pool = (
                (position % POOL_SIZE == POOL_SIZE - 1)
                & (cache_location >= 0)
                & (state_slot >= 0)
            )
        else:
            write_pool = (cache_location >= 0) & (state_slot >= 0)
        pool_mask = (dim < HEAD_DIM) & write_pool
        maximum = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = current_gate
            if slot != POOL_SIZE - 1:
                tail_offset = (
                    state_slot * tail_stride_0
                    + tail_stride_1
                    + slot * tail_stride_2
                    + dim
                )
                score = tl.load(tail + tail_offset, mask=pool_mask, other=0.0).to(
                    tl.float32
                )
            score += tl.load(
                ape + slot * ape_stride_0 + dim,
                mask=pool_mask,
                other=0.0,
            ).to(tl.float32)
            maximum = tl.maximum(maximum, score)
        numerator = tl.zeros((BLOCK_D,), tl.float32)
        denominator = tl.zeros((BLOCK_D,), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            value = current_key
            score = current_gate
            if slot != POOL_SIZE - 1:
                base = state_slot * tail_stride_0 + slot * tail_stride_2 + dim
                value = tl.load(tail + base, mask=pool_mask, other=0.0).to(tl.float32)
                score = tl.load(
                    tail + base + tail_stride_1, mask=pool_mask, other=0.0
                ).to(tl.float32)
            score += tl.load(
                ape + slot * ape_stride_0 + dim,
                mask=pool_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.exp(score - maximum)
            numerator += value * weight
            denominator += weight
        pooled = numerator / tl.maximum(denominator, 1e-20)
        _write_pool(
            cache_fp8,
            cache_fp32,
            cache_location,
            pooled,
            dim,
            dim < HEAD_DIM,
            write_pool,
            page_stride_bytes,
            PAGE_SIZE,
            HEAD_DIM,
            FP8_MAX,
        )
        base = state_slot * tail_stride_0 + physical_slot * tail_stride_2
        tl.store(tail + base + dim, current_key, mask=dim_mask)
        tl.store(tail + base + tail_stride_1 + dim, current_gate, mask=dim_mask)


@triton.jit
def _prefill_pool_kernel(
    cache_fp8,
    cache_fp32,
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    ape,
    slot_mapping,
    positions,
    page_stride_bytes,
    model_block_size,
    parent_stride_pages,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    ape_stride_0,
    request_offset,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = request_offset + tl.program_id(0)
    pool_ordinal = tl.program_id(1)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    has_rows = end > start
    first_position = tl.load(positions + start, mask=has_rows, other=0).to(tl.int64)
    first_physical_slot = first_position % POOL_SIZE
    first_completion = start + (POOL_SIZE - 1 - first_physical_slot)
    completion_row = first_completion + pool_ordinal * POOL_SIZE
    write_pool = has_rows & (completion_row < end) & (state_slot >= 0)
    completion_position = tl.load(
        positions + completion_row, mask=write_pool, other=-1
    ).to(tl.int64)
    write_pool &= completion_position % POOL_SIZE == POOL_SIZE - 1
    main_cache_location = tl.load(
        slot_mapping + completion_row, mask=write_pool, other=-1
    ).to(tl.int64)
    write_pool &= main_cache_location >= 0
    if write_pool == 0:
        return
    parent_page = main_cache_location // model_block_size
    model_page_offset = main_cache_location - parent_page * model_block_size
    pool_offset = model_page_offset // POOL_SIZE
    child_page = pool_offset // PAGE_SIZE
    child_offset = pool_offset - child_page * PAGE_SIZE
    cache_location = (
        parent_page * parent_stride_pages + child_page
    ) * PAGE_SIZE + child_offset

    dim = tl.arange(0, BLOCK_D)
    dim_mask = dim < HEAD_DIM
    maximum = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        source_row = completion_row - (POOL_SIZE - 1 - slot)
        from_current_chunk = source_row >= start
        tail_offset = (
            state_slot * tail_stride_0 + tail_stride_1 + slot * tail_stride_2 + dim
        )
        current_score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_score = tl.load(
            tail + tail_offset,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        score = tl.where(from_current_chunk, current_score, previous_score)
        score += tl.load(
            ape + slot * ape_stride_0 + dim,
            mask=dim_mask & write_pool,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(maximum, score)

    numerator = tl.zeros((BLOCK_D,), tl.float32)
    denominator = tl.zeros((BLOCK_D,), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        source_row = completion_row - (POOL_SIZE - 1 - slot)
        from_current_chunk = source_row >= start
        tail_offset = state_slot * tail_stride_0 + slot * tail_stride_2 + dim
        current_value = tl.load(
            key + source_row * key_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_value = tl.load(
            tail + tail_offset,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        value = tl.where(from_current_chunk, current_value, previous_value)
        current_score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=dim_mask & write_pool & from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        previous_score = tl.load(
            tail + tail_offset + tail_stride_1,
            mask=dim_mask & write_pool & ~from_current_chunk,
            other=0.0,
        ).to(tl.float32)
        score = tl.where(from_current_chunk, current_score, previous_score)
        score += tl.load(
            ape + slot * ape_stride_0 + dim,
            mask=dim_mask & write_pool,
            other=0.0,
        ).to(tl.float32)
        weight = tl.exp(score - maximum)
        numerator += value * weight
        denominator += weight

    pooled = numerator / tl.maximum(denominator, 1e-20)
    _write_pool(
        cache_fp8,
        cache_fp32,
        cache_location,
        pooled,
        dim,
        dim_mask,
        write_pool,
        page_stride_bytes,
        PAGE_SIZE,
        HEAD_DIM,
        FP8_MAX,
    )


@triton.jit
def _prefill_tail_kernel(
    tail,
    state_slots,
    query_start_loc,
    key,
    gate,
    positions,
    request_offset,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    key_stride_0,
    gate_stride_0,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = request_offset + tl.program_id(0)
    state_slot = tl.load(state_slots + request).to(tl.int64)
    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    last_row = end - 1
    has_rows = end > start
    last_position = tl.load(positions + last_row, mask=has_rows, other=0).to(tl.int64)
    last_physical_slot = last_position % POOL_SIZE
    dim = tl.arange(0, BLOCK_D)
    dim_mask = (dim < HEAD_DIM) & has_rows & (state_slot >= 0)
    for slot in tl.static_range(0, POOL_SIZE):
        distance = (last_physical_slot - slot + POOL_SIZE) % POOL_SIZE
        source_row = last_row - distance
        source_in_chunk = source_row >= start
        source_position = tl.load(
            positions + source_row,
            mask=has_rows & source_in_chunk,
            other=-1,
        ).to(tl.int64)
        write_slot = dim_mask & source_in_chunk & (source_position % POOL_SIZE == slot)
        value = tl.load(
            key + source_row * key_stride_0 + dim,
            mask=write_slot,
            other=0.0,
        )
        score = tl.load(
            gate + source_row * gate_stride_0 + dim,
            mask=write_slot,
            other=0.0,
        )
        tail_offset = state_slot * tail_stride_0 + slot * tail_stride_2 + dim
        tl.store(tail + tail_offset, value, mask=write_slot)
        tl.store(tail + tail_offset + tail_stride_1, score, mask=write_slot)


def update_decode_pools(
    kv_cache: torch.Tensor,
    tail: torch.Tensor,
    state_slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    num_requests: int,
    *,
    num_decode_requests: int | None = None,
    max_query_len: int | None = None,
    model_block_size: int | None = None,
    parent_stride_pages: int | None = None,
) -> None:
    """Update request tails and write every newly completed FP8 C4 pool.

    Decode and speculative-decode requests retain ordered row processing. Prefill
    requests use one program per completed pool and a separate final-tail update;
    their positions must be consecutive within each request, as required by the
    packed GLM MLA cache contract.
    """
    if num_requests <= 0:
        return
    page_size = int(kv_cache.shape[1])
    if page_size != 64:
        raise ValueError("GLM C4 decode requires a 64-entry index-cache page")
    packed_main_slots = model_block_size is not None
    if packed_main_slots != (parent_stride_pages is not None):
        raise ValueError(
            "model_block_size and parent_stride_pages must be provided together"
        )
    if model_block_size is None:
        model_block_size = page_size * _POOL_SIZE
        parent_stride_pages = 1
    if model_block_size <= 0 or model_block_size % (page_size * _POOL_SIZE):
        raise ValueError(
            "GLM model pages must contain a whole number of 64-entry C4 pages"
        )
    assert parent_stride_pages is not None
    if parent_stride_pages <= 0:
        raise ValueError("GLM parent_stride_pages must be positive")
    if num_decode_requests is None:
        num_decode_requests = num_requests
    if not 0 <= num_decode_requests <= num_requests:
        raise ValueError("GLM decode request count exceeds the request batch")
    num_prefill_requests = num_requests - num_decode_requests
    if num_prefill_requests and not packed_main_slots:
        raise ValueError("parallel GLM prefill requires packed main-cache slots")
    if num_prefill_requests and (max_query_len is None or max_query_len <= 0):
        raise ValueError("parallel GLM prefill requires a positive max_query_len")

    common_args = (
        kv_cache.view(torch.float8_e4m3fn),
        kv_cache.view(torch.float32),
        tail,
        state_slots,
        query_start_loc,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        int(kv_cache.stride(0)),
        int(model_block_size),
        int(parent_stride_pages),
        int(tail.stride(0)),
        int(tail.stride(1)),
        int(tail.stride(2)),
        int(key.stride(0)),
        int(gate.stride(0)),
        int(ape.stride(0)),
    )
    common_meta = dict(
        PAGE_SIZE=page_size,
        HEAD_DIM=_HEAD_DIM,
        POOL_SIZE=_POOL_SIZE,
        FP8_MAX=_FP8_MAX,
        BLOCK_D=128,
        num_warps=4,
    )
    if num_decode_requests:
        _decode_update_kernel[(num_decode_requests,)](
            *common_args,
            PACKED_MAIN_SLOTS=packed_main_slots,
            **common_meta,
        )
    if num_prefill_requests:
        assert max_query_len is not None
        _prefill_pool_kernel[
            (num_prefill_requests, triton.cdiv(max_query_len, _POOL_SIZE))
        ](
            *common_args,
            num_decode_requests,
            **common_meta,
        )
        _prefill_tail_kernel[(num_prefill_requests,)](
            tail,
            state_slots,
            query_start_loc,
            key,
            gate,
            positions,
            num_decode_requests,
            int(tail.stride(0)),
            int(tail.stride(1)),
            int(tail.stride(2)),
            int(key.stride(0)),
            int(gate.stride(0)),
            HEAD_DIM=_HEAD_DIM,
            POOL_SIZE=_POOL_SIZE,
            BLOCK_D=128,
            num_warps=4,
        )


@triton.jit
def _expand_c4_block_table_kernel(
    source,
    output,
    rows,
    source_width,
    output_width,
    source_stride,
    output_stride,
    parent_stride_pages,
    SUBPAGES_PER_PARENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = rows * output_width
    mask = linear < total
    row = linear // output_width
    output_col = linear - row * output_width
    source_col = output_col // SUBPAGES_PER_PARENT
    child_page = output_col - source_col * SUBPAGES_PER_PARENT
    parent_page = tl.load(
        source + row * source_stride + source_col,
        mask=mask & (source_col < source_width),
        other=-1,
    ).to(tl.int64)
    child_page_id = parent_page * parent_stride_pages.to(tl.int64) + child_page
    child_page_id = tl.where(parent_page >= 0, child_page_id, -1)
    tl.store(output + row * output_stride + output_col, child_page_id, mask=mask)


def expand_c4_block_table(
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    rows: int,
    subpages_per_parent: int,
    parent_stride_pages: int,
) -> None:
    """Expand main-cache page IDs into the virtual 64-pool C4 page space."""
    if source.dtype != torch.int32 or output.dtype != torch.int32:
        raise TypeError("GLM block tables must use int32")
    if source.ndim != 2 or output.ndim != 2:
        raise ValueError("GLM block tables must be rank two")
    if not 0 <= rows <= min(int(source.shape[0]), int(output.shape[0])):
        raise ValueError("GLM block-table row count exceeds buffer capacity")
    expected_width = int(source.shape[1]) * subpages_per_parent
    if int(output.shape[1]) != expected_width:
        raise ValueError(
            f"GLM expanded block-table width must be {expected_width}, "
            f"got {int(output.shape[1])}"
        )
    if rows:
        block = 256
        total = rows * expected_width
        _expand_c4_block_table_kernel[(triton.cdiv(total, block),)](
            source,
            output,
            rows,
            int(source.shape[1]),
            expected_width,
            int(source.stride(0)),
            int(output.stride(0)),
            parent_stride_pages,
            SUBPAGES_PER_PARENT=subpages_per_parent,
            BLOCK=block,
            num_warps=4,
        )


@triton.jit
def _gather_c4_block_table_rows_kernel(
    source,
    request_ids,
    output,
    rows,
    width,
    source_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = rows * width
    mask = linear < total
    row = linear // width
    col = linear - row * width
    request = tl.load(request_ids + row, mask=mask, other=0).to(tl.int64)
    value = tl.load(source + request * source_stride + col, mask=mask, other=-1)
    tl.store(output + row * output_stride + col, value, mask=mask)


def gather_c4_block_table_rows(
    source: torch.Tensor,
    request_ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Gather request page tables into a fixed decode-row buffer."""
    rows = int(request_ids.shape[0])
    if source.dtype != torch.int32 or request_ids.dtype != torch.int32:
        raise TypeError("GLM block tables and request IDs must use int32")
    if output.dtype != torch.int32 or output.shape != (rows, source.shape[1]):
        raise ValueError("GLM decode block-table output has the wrong contract")
    if rows:
        block = 256
        total = rows * int(source.shape[1])
        _gather_c4_block_table_rows_kernel[(triton.cdiv(total, block),)](
            source,
            request_ids,
            output,
            rows,
            int(source.shape[1]),
            int(source.stride(0)),
            int(output.stride(0)),
            BLOCK=block,
            num_warps=4,
        )


@triton.jit
def _pool_seq_lens_kernel(
    positions,
    output,
    rows,
    dcp_size,
    dcp_rank,
    pool_interleave,
    POOL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < rows
    position = tl.load(positions + row, mask=mask, other=-1).to(tl.int64)
    global_pool_len = (position + 1) // POOL_SIZE
    rounds = global_pool_len // (dcp_size * pool_interleave)
    remainder = global_pool_len % (dcp_size * pool_interleave)
    remainder = tl.maximum(remainder - dcp_rank * pool_interleave, 0)
    remainder = tl.minimum(remainder, pool_interleave)
    local_pool_len = rounds * pool_interleave + remainder
    tl.store(output + row, local_pool_len, mask=mask)


def pool_seq_lens(
    positions: torch.Tensor,
    output: torch.Tensor,
    *,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    pool_interleave: int = 1,
) -> None:
    """Write the rank-local number of complete four-token pools per row."""
    rows = int(positions.shape[0])
    if positions.dtype != torch.int64 or positions.ndim != 1:
        raise TypeError("GLM positions must be int64 [rows]")
    if output.dtype != torch.int32 or output.shape != (rows,):
        raise ValueError("GLM pool sequence-length output must be int32 [rows]")
    if dcp_size < 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("GLM pool DCP rank must be within the DCP world")
    if pool_interleave < 1:
        raise ValueError("GLM pool interleave must be positive")
    if rows:
        block = 256
        _pool_seq_lens_kernel[(triton.cdiv(rows, block),)](
            positions,
            output,
            rows,
            dcp_size,
            dcp_rank,
            pool_interleave,
            POOL_SIZE=_POOL_SIZE,
            BLOCK=block,
            num_warps=4,
        )


@triton.jit
def _expand_pool_ids_kernel(
    pool_ids,
    positions,
    output,
    pool_stride,
    output_stride,
    HISTORY_TOKENS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = column < OUTPUT_WIDTH
    sequence_length = tl.load(positions + row).to(tl.int64) + 1
    complete_pools = sequence_length // POOL_SIZE
    tail_start = complete_pools * POOL_SIZE
    history = column < HISTORY_TOKENS
    pool_column = column // POOL_SIZE
    pool_offset = column % POOL_SIZE
    pool_id = tl.load(
        pool_ids + row * pool_stride + pool_column,
        mask=mask & history,
        other=-1,
    ).to(tl.int64)
    history_value = tl.where(pool_id >= 0, pool_id * POOL_SIZE + pool_offset, -1)
    tail_offset = column - HISTORY_TOKENS
    tail_count = sequence_length - tail_start
    in_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_value = tl.where(in_tail, tail_start + tail_offset, -1)
    value = tl.where(history, history_value, tail_value).to(tl.int32)
    tl.store(output + row * output_stride + column, value, mask=mask)


def expand_pool_ids(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Expand 512 selected pools to 2048 tokens and append up to three tail tokens."""
    if pool_ids.ndim != 2 or int(pool_ids.shape[1]) != 512:
        raise ValueError("GLM pool selection must have shape [rows, 512]")
    rows = int(pool_ids.shape[0])
    if positions.shape != (rows,) or positions.dtype != torch.int64:
        raise ValueError("GLM expansion positions must be int64 [rows]")
    if output.shape != (rows, 2051) or output.dtype != torch.int32:
        raise ValueError("GLM token selection output must be int32 [rows, 2051]")
    if rows:
        block_cols = 128
        _expand_pool_ids_kernel[(rows, triton.cdiv(2051, block_cols))](
            pool_ids,
            positions,
            output,
            int(pool_ids.stride(0)),
            int(output.stride(0)),
            HISTORY_TOKENS=2048,
            OUTPUT_WIDTH=2051,
            POOL_SIZE=_POOL_SIZE,
            BLOCK_COLS=block_cols,
            num_warps=4,
        )


__all__ = [
    "expand_c4_block_table",
    "expand_pool_ids",
    "fwht128_quant_fp8",
    "gather_c4_block_table_rows",
    "pool_seq_lens",
    "update_decode_pools",
]
