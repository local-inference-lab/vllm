# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _convert_dcp_local_topk_to_global_kernel(
    token_indices_ptr,
    scores_ptr,
    ti_stride0,
    ti_stride1,
    scores_stride0,
    scores_stride1,
    width: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < width
    idx_ptrs = token_indices_ptr + row * ti_stride0 + offs * ti_stride1
    local_idx = tl.load(idx_ptrs, mask=mask, other=-1)
    valid = local_idx >= 0

    interleave_block = local_idx // CP_KV_CACHE_INTERLEAVE_SIZE
    interleave_offset = local_idx % CP_KV_CACHE_INTERLEAVE_SIZE
    global_idx = (
        interleave_block * DCP_WORLD_SIZE + DCP_RANK
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + interleave_offset
    tl.store(idx_ptrs, tl.where(valid, global_idx, -1), mask=mask)

    score_ptrs = scores_ptr + row * scores_stride0 + offs * scores_stride1
    scores = tl.load(score_ptrs, mask=mask, other=-float("inf"))
    tl.store(score_ptrs, tl.where(valid, scores, -float("inf")), mask=mask)


def triton_convert_dcp_local_topk_to_global(
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    BLOCK_N: int = 128,
) -> None:
    """Convert local DCP top-k ids in-place to global logical token ids."""
    assert token_indices.dtype == torch.int32
    assert scores.dtype == torch.float32
    assert token_indices.shape == scores.shape
    assert token_indices.is_contiguous()
    assert scores.is_contiguous()
    width = token_indices.shape[1]
    assert width % BLOCK_N == 0, (
        f"top-k width ({width}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    grid = (token_indices.shape[0], width // BLOCK_N)
    _convert_dcp_local_topk_to_global_kernel[grid](
        token_indices,
        scores,
        token_indices.stride(0),
        token_indices.stride(1),
        scores.stride(0),
        scores.stride(1),
        width,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _gather_topk_ids_by_position_kernel(
    candidate_ids_ptr,
    positions_ptr,
    out_ptr,
    cand_stride0,
    cand_stride1,
    pos_stride0,
    pos_stride1,
    out_stride0,
    out_stride1,
    topk: tl.constexpr,
    candidate_width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < topk
    pos = tl.load(
        positions_ptr + row * pos_stride0 + offs * pos_stride1,
        mask=mask,
        other=-1,
    )
    valid = (pos >= 0) & (pos < candidate_width)
    gathered = tl.load(
        candidate_ids_ptr + row * cand_stride0 + pos * cand_stride1,
        mask=mask & valid,
        other=-1,
    )
    tl.store(
        out_ptr + row * out_stride0 + offs * out_stride1,
        tl.where(valid, gathered, -1),
        mask=mask,
    )


def triton_gather_topk_ids_by_position(
    candidate_ids: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    *,
    BLOCK_N: int = 128,
) -> None:
    """Gather final ids from flattened candidate ids using int32 top-k positions."""
    assert candidate_ids.dtype == torch.int32
    assert positions.dtype == torch.int32
    assert out.dtype == torch.int32
    assert candidate_ids.ndim == 2
    assert positions.ndim == 2
    assert out.shape == positions.shape
    assert candidate_ids.shape[0] == positions.shape[0]
    assert positions.shape[1] % BLOCK_N == 0, (
        f"top-k width ({positions.shape[1]}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    grid = (positions.shape[0], positions.shape[1] // BLOCK_N)
    _gather_topk_ids_by_position_kernel[grid](
        candidate_ids,
        positions,
        out,
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        positions.stride(0),
        positions.stride(1),
        out.stride(0),
        out.stride(1),
        positions.shape[1],
        candidate_ids.shape[1],
        BLOCK_N=BLOCK_N,
    )


# Kernel with prefill workspace support and valid count tracking
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # When set, scatter valid slots to a contiguous prefix [0, valid_count) using
    # valid_count_ptr as an atomic slot allocator (DCP filtering leaves interior
    # -1 gaps; the trtllm-gen sparse kernel reads the first valid_count entries).
    # Requires COUNT_VALID and an out buffer pre-filled with -1. Order within the
    # prefix is unspecified (only the selected set matters).
    COMPACT_TO_FRONT: tl.constexpr,
    # DCP de-interleave: with DCP_SIZE == 1 these are an exact no-op
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    # DCP de-interleave the global token id into this rank's local slot.
    # Tokens are interleaved in groups of DCP_INTERLEAVE across ranks. With
    # DCP_SIZE == 1 (and any interleave) owning_rank == 0 == DCP_RANK (never
    # remote) and local_idx == tok, so this reduces to the non-DCP path; with
    # DCP_INTERLEAVE == 1 it reduces to plain round-robin (tok % / // DCP_SIZE).
    owning_rank = (tok // DCP_INTERLEAVE) % DCP_SIZE
    is_remote = owning_rank != DCP_RANK
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE

    # Compute block id and in-block offset
    block_id = local_idx // BLOCK_SIZE
    inblock_off = local_idx % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block | is_remote
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill & ~is_remote, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    if COMPACT_TO_FRONT:
        # Scatter valid slots to a contiguous prefix. A per-tile exclusive prefix
        # sum gives each valid lane a distinct local offset; one atomic add of the
        # tile's valid count reserves a contiguous base across racing tiles. The
        # out buffer is pre-filled with -1, so unwritten tail slots stay -1.
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
        dest = base + local_offset
        out_ptr_dest = out_ptr + token_id * out_stride0 + dest * out_stride1
        tl.store(out_ptr_dest, out_val, mask=is_valid == 1)
    else:
        # Store results in place (input column == output column).
        out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
        tl.store(out_ptr_ij, out_val)

        # Count valid indices in this tile and atomically add to row total
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert req_id.shape[0] == token_indices.shape[0], (
        f"req_id ({req_id.shape[0]}) and token_indices ({token_indices.shape[0]}) "
        "must cover the same tokens; the grid is sized by req_id but the output "
        "is allocated like token_indices, so a longer req_id writes out of bounds"
    )
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        # DCP disabled (no-op de-interleave)
        1,
        0,
        1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


def triton_filter_and_convert_dcp_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int = 1,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = True,
    out: torch.Tensor | None = None,
    valid_counts: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Filter global per-request indices to this DCP rank's local slots.

    With ``compact_valid_to_front`` (default), the conversion kernel scatters
    this rank's owned slots to a contiguous prefix ``[0, valid_count)`` and
    leaves the rest ``-1``. DCP filtering marks non-owned slots ``-1`` and so
    creates interior gaps; the trtllm-gen sparse kernel reads the first
    ``valid_count`` entries of each row, so they must be a contiguous prefix.
    Compaction is fused into the kernel (atomic slot allocator) rather than a
    separate sort/gather pass. Prefix order is unspecified (only the set matters).
    """
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size
    # Interleave groups must align to KV blocks (globally enforced by
    # VllmConfig: block_size % cp_kv_cache_interleave_size == 0); assert the
    # local invariant so local_idx // BLOCK_SIZE never straddles a group.
    assert BLOCK_SIZE % cp_kv_cache_interleave_size == 0, (
        f"BLOCK_SIZE ({BLOCK_SIZE}) must be divisible by "
        f"cp_kv_cache_interleave_size ({cp_kv_cache_interleave_size})."
    )
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    if dcp_size == 1:
        assert out is None and valid_counts is None, (
            "preallocated out/valid_counts are only supported on the DCP path"
        )
        return triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
            BLOCK_N=BLOCK_N,
            return_valid_counts=return_valid_counts,
        )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    # The compaction uses the valid-count buffer as an atomic slot allocator, so
    # it requires counting. Pre-fill out with -1 so the unwritten tail stays -1.
    # Callers on the decode hot path (e.g. the b12x sparse-MLA backend) pass
    # preallocated CUDA-graph-stable buffers to keep the conversion zero-copy.
    count_valid = return_valid_counts or compact_valid_to_front
    if out is None:
        if compact_valid_to_front:
            out = torch.full_like(token_indices_c, -1)
        else:
            out = torch.empty_like(token_indices_c)
    else:
        assert out.dtype == torch.int32 and out.device == token_indices.device
        assert out.shape == token_indices_c.shape
        if compact_valid_to_front:
            out.fill_(-1)

    if count_valid:
        if valid_counts is None:
            valid_counts = torch.zeros(
                num_tokens, dtype=torch.int32, device=token_indices.device
            )
        else:
            assert valid_counts.dtype == torch.int32
            assert valid_counts.shape[0] == num_tokens
            valid_counts.zero_()
    else:
        valid_counts = None

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    _convert_req_index_to_global_index_kernel[(num_tokens, tiles_per_row)](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        # No prefill workspace on the DCP decode path.
        None,
        None,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        False,  # HAS_PREFILL
        count_valid,
        compact_valid_to_front,
        dcp_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
