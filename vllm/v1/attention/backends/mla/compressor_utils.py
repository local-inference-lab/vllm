# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2

_C128A_TOPK_ALIGNMENT = 128
_COMPRESSED_MLA_SPLIT_ALIGNMENT = 64
_DSPARK_SWA_INDEX_ALIGNMENT = 512


def get_c128a_topk_width(max_model_len: int, compress_ratio: int) -> int:
    """Return C128 indexed width padded for FlashMLA B_TOPK divisibility.

    Args:
        max_model_len: Maximum model context length in tokens.
        compress_ratio: Ratio used to compress the indexed KV cache.

    Returns:
        The aligned compressed top-k width.
    """
    compressed_width = cdiv(max_model_len, compress_ratio)
    return cdiv(compressed_width, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT


def get_c128a_active_topk_width(
    max_seq_len: int,
    compress_ratio: int,
    capacity: int,
) -> int:
    """Return the active C128 index width for one scheduler step.

    Only complete compression groups produce selectable records. The metadata
    kernel uses a power-of-two width, with the FlashMLA alignment as its lower
    bound and the preallocated metadata buffer width as its upper bound.
    """
    complete_records = max(int(max_seq_len), 0) // compress_ratio
    return min(
        max(
            next_power_of_2(max(complete_records, 1)),
            _C128A_TOPK_ALIGNMENT,
        ),
        capacity,
    )


def get_dspark_swa_index_width(
    window_size: int,
    num_speculative_tokens: int,
) -> int:
    """Return the padded width of non-causal DSpark SWA indices.

    Args:
        window_size: Sliding-window attention width.
        num_speculative_tokens: Number of DSpark draft tokens.

    Returns:
        The aligned non-causal index width.
    """
    width = max(int(window_size), 0) + max(int(num_speculative_tokens), 0)
    return cdiv(width, _DSPARK_SWA_INDEX_ALIGNMENT) * _DSPARK_SWA_INDEX_ALIGNMENT


def get_compressed_mla_split_cap(width: int) -> int:
    """Return the maximum compressed-MLA split count for ``width``.

    Args:
        width: Combined SWA and indexed width.

    Returns:
        The split count capped by the kernel tile width.
    """
    return max(1, cdiv(max(int(width), 1), _COMPRESSED_MLA_SPLIT_ALIGNMENT))


@cache
def get_compressed_mla_max_q_chunks(
    max_rows: int,
    width: int,
    max_chunks: int,
    split_chunks_for_contract: Callable[..., int],
    decode_row_capacity: int | None = None,
) -> int:
    """Return the q-chunk cap over every reachable row count.

    Args:
        max_rows: Maximum number of query rows in one scheduler step.
        width: Combined SWA and indexed width.
        max_chunks: Maximum chunks allowed for each row.
        split_chunks_for_contract: Kernel contract split-count function.
        decode_row_capacity: Integration-declared decode/verifier row capacity.

    Returns:
        The largest reachable product of rows and chunks per row.
    """
    max_rows = max(int(max_rows), 1)
    width = max(int(width), 1)
    max_chunks = max(int(max_chunks), 1)
    return max(
        rows
        * split_chunks_for_contract(
            rows=rows,
            width=width,
            max_chunks=max_chunks,
            decode_row_capacity=decode_row_capacity,
        )
        for rows in range(1, max_rows + 1)
    )


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        pos_after_compress = pos // COMPRESS_RATIO

        if DCP_WORLD_SIZE == 1:
            block_ids = pos_after_compress // block_size
            block_offsets = pos_after_compress % block_size
            is_local = True
        else:
            virtual_block_size = block_size * DCP_WORLD_SIZE
            block_ids = pos_after_compress // virtual_block_size
            virtual_block_offsets = pos_after_compress - block_ids * virtual_block_size
            is_local = (
                virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            ) % DCP_WORLD_SIZE == DCP_RANK
            block_offsets = (
                virtual_block_offsets // (DCP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
                virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
            )

        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid & is_local,
        ).to(tl.int64)
        slot_ids = block_numbers * block_size + block_offsets

        # NOTE
        slot_ids = tl.where(is_valid & is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
