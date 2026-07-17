# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_B12X_PAGED_INDEX_PAGE_SIZE = 64
_B12X_PAGED_INDEX_HEAD_DIM = 128
_B12X_PAGED_INDEX_SCALE_BYTES = 4
_B12X_PAGED_INDEX_PAGE_WIDTH = _B12X_PAGED_INDEX_PAGE_SIZE * (
    _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES
)
_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_B12X_PAGED_INDEX_TILE_BLOCK_Q = 32
_B12X_PAGED_INDEX_TILE_BLOCK_K = 512
_B12X_CONTIGUOUS_PREFILL_BLOCK_K = 256
_B12X_CONTIGUOUS_PREFILL512_BLOCK_K = 512
_B12X_CONTIGUOUS_PREFILL512_MIN_Q_ROWS = 1024
_B12X_CONTIGUOUS_PREFILL512_MIN_K_ROWS = 4096
_B12X_CONTIGUOUS_PREFILL512_SUPPORTED_HEADS = (32, 64)
# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32
_B12X_PREFILL_PAGED_ROUTE = "packed_contiguous"


def _dcp_global_topk_requested() -> bool:
    return envs.VLLM_DCP_GLOBAL_TOPK


def _use_persistent_topk_decode(topk_tokens: int) -> bool:
    return current_platform.is_cuda() and topk_tokens in (512, 1024, 2048)


def _local_to_global_position(
    local_idx: torch.Tensor, rank: int, world_size: int, interleave: int
) -> torch.Tensor:
    return (
        (local_idx // interleave) * (world_size * interleave)
        + rank * interleave
        + (local_idx % interleave)
    )


def _global_to_local_position(
    global_idx: torch.Tensor, interleave: int, world_size: int
) -> torch.Tensor:
    big = interleave * world_size
    return (global_idx // big) * interleave + (global_idx % interleave)


@triton.jit
def _dcp_pack_topk_candidates_kernel(
    topk_indices_ptr,
    logits_ptr,
    row_starts_ptr,
    packed_ptr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    logits_stride_0: tl.constexpr,
    logits_stride_1: tl.constexpr,
    packed_stride_0: tl.constexpr,
    packed_stride_1: tl.constexpr,
    packed_stride_2: tl.constexpr,
    logits_width,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    HAS_LOGITS: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    idx = tl.load(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        mask=mask,
        other=-1,
    )
    invalid = idx < 0
    idx_safe = tl.maximum(idx, 0)

    scores = tl.full((BLOCK_K,), -float("inf"), tl.float32)
    if HAS_LOGITS:
        score_idx = idx_safe
        if HAS_ROW_STARTS:
            row_start = tl.load(row_starts_ptr + row)
            score_idx += row_start
        score_idx = tl.minimum(score_idx, logits_width - 1)
        scores = tl.load(
            logits_ptr + row * logits_stride_0 + score_idx * logits_stride_1,
            mask=mask,
            other=-float("inf"),
        )
        scores = scores.to(tl.float32)
        scores = tl.where(invalid, -float("inf"), scores)

    global_pos = (
        (idx_safe // INTERLEAVE) * (WORLD_SIZE * INTERLEAVE)
        + RANK * INTERLEAVE
        + (idx_safe % INTERLEAVE)
    )
    global_pos = tl.where(invalid, -1, global_pos)

    packed_base = packed_ptr + row * packed_stride_0 + offsets * packed_stride_2
    tl.store(packed_base, global_pos, mask=mask)
    tl.store(
        packed_base + packed_stride_1,
        scores.to(tl.int32, bitcast=True),
        mask=mask,
    )


@triton.jit
def _dcp_finalize_topk_remap_kernel(
    all_candidates_ptr,
    selected_ptr,
    topk_indices_ptr,
    candidates_stride_0: tl.constexpr,
    candidates_stride_1: tl.constexpr,
    candidates_stride_2: tl.constexpr,
    selected_stride_0: tl.constexpr,
    selected_stride_1: tl.constexpr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOPK_TOKENS

    selected = tl.load(
        selected_ptr + row * selected_stride_0 + offsets * selected_stride_1,
        mask=mask,
        other=0,
    )
    global_pos = tl.load(
        all_candidates_ptr + row * candidates_stride_0 + selected * candidates_stride_2,
        mask=mask,
        other=-1,
    )

    owner = (global_pos // INTERLEAVE) % WORLD_SIZE
    big = INTERLEAVE * WORLD_SIZE
    local_pos = (global_pos // big) * INTERLEAVE + (global_pos % INTERLEAVE)
    mine = (owner == RANK) & (global_pos >= 0)
    final = tl.where(mine, local_pos, -1)

    tl.store(
        topk_indices_ptr + row * topk_stride_0 + offsets * topk_stride_1,
        final,
        mask=mask,
    )


@triton.jit
def _b12x_dcp_unpack_gathered_candidates_kernel(
    gathered_ptr,
    candidate_indices_ptr,
    candidate_score_bits_ptr,
    ROWS,
    TOPK_TOKENS: tl.constexpr,
    CANDIDATE_WIDTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < CANDIDATE_WIDTH

    rank = offsets // TOPK_TOKENS
    local_offset = offsets - rank * TOPK_TOKENS
    raw_row = rank * ROWS + row
    raw_base = (raw_row * 2) * TOPK_TOKENS + local_offset

    candidate_ids = tl.load(gathered_ptr + raw_base, mask=mask, other=-1)
    candidate_scores = tl.load(
        gathered_ptr + raw_base + TOPK_TOKENS,
        mask=mask,
        other=0,
    )
    out_base = row * CANDIDATE_WIDTH + offsets
    tl.store(candidate_indices_ptr + out_base, candidate_ids, mask=mask)
    tl.store(candidate_score_bits_ptr + out_base, candidate_scores, mask=mask)


def _use_triton_dcp_remap(topk_indices: torch.Tensor) -> bool:
    return HAS_TRITON and current_platform.is_cuda() and topk_indices.is_cuda


def _dcp_pack_topk_candidates(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack local candidate global positions and score bits for one all-gather."""
    packed = torch.empty(
        (topk_indices.shape[0], 2, topk_tokens),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if topk_indices.numel() == 0:
        return packed

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        logits_arg = logits if logits is not None else topk_indices
        row_starts_arg = row_starts if row_starts is not None else topk_indices
        _dcp_pack_topk_candidates_kernel[(topk_indices.shape[0],)](
            topk_indices,
            logits_arg,
            row_starts_arg,
            packed,
            topk_indices.stride(0),
            topk_indices.stride(1),
            logits.stride(0) if logits is not None else 0,
            logits.stride(1) if logits is not None else 0,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            logits.shape[1] if logits is not None else 1,
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            HAS_LOGITS=logits is not None,
            HAS_ROW_STARTS=row_starts is not None,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return packed

    invalid = topk_indices < 0
    idx_safe = torch.clamp(topk_indices, min=0)
    if logits is None:
        local_scores = torch.full(
            topk_indices.shape,
            float("-inf"),
            dtype=torch.float32,
            device=topk_indices.device,
        )
    else:
        score_idx = idx_safe.to(torch.int64)
        if row_starts is not None:
            score_idx = score_idx + row_starts.to(
                device=score_idx.device, dtype=score_idx.dtype
            ).view(-1, 1)
        score_idx = torch.clamp(score_idx, min=0, max=logits.shape[1] - 1)
        local_scores = torch.gather(logits, 1, score_idx).to(torch.float32)
    local_scores = local_scores.masked_fill(invalid, float("-inf"))

    global_pos = _local_to_global_position(idx_safe, rank, world_size, interleave)
    global_pos = torch.where(invalid, global_pos.new_full((), -1), global_pos)

    packed[:, 0, :].copy_(global_pos.to(torch.int32))
    packed[:, 1, :].copy_(local_scores.contiguous().view(torch.int32))
    return packed


def _dcp_finalize_topk_remap(
    all_candidates: torch.Tensor,
    selected: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
) -> None:
    """Finalize selected global candidates into rank-local top-k indices."""
    if topk_indices.numel() == 0:
        return

    if _use_triton_dcp_remap(topk_indices):
        block_k = triton.next_power_of_2(topk_tokens)
        num_warps = 4 if block_k <= 256 else 8
        _dcp_finalize_topk_remap_kernel[(topk_indices.shape[0],)](
            all_candidates,
            selected,
            topk_indices,
            all_candidates.stride(0),
            all_candidates.stride(1),
            all_candidates.stride(2),
            selected.stride(0),
            selected.stride(1),
            topk_indices.stride(0),
            topk_indices.stride(1),
            RANK=rank,
            WORLD_SIZE=world_size,
            INTERLEAVE=interleave,
            TOPK_TOKENS=topk_tokens,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
        return

    sel_global = torch.gather(all_candidates[:, 0, :], 1, selected.to(torch.int64))
    owner = (sel_global // interleave) % world_size
    local_of_g = _global_to_local_position(sel_global, interleave, world_size)
    mine = (owner == rank) & (sel_global >= 0)
    final = torch.where(mine, local_of_g, local_of_g.new_full((), -1))
    topk_indices.copy_(final.to(topk_indices.dtype))


def _dcp_all_gather_first_dim_into(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    """All-gather into caller-owned output, concatenating along dim 0."""
    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()
    assert output_tensor.shape[0] == input_tensor.shape[0] * group.world_size
    assert output_tensor.shape[1:] == input_tensor.shape[1:]

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        import torch.distributed as dist

        dist.all_gather_into_tensor(output_tensor, input_tensor, group=device_group)
        return

    gathered = group.all_gather(input_tensor, dim=0)
    output_tensor.copy_(gathered)


def _unpack_b12x_dcp_gathered_candidates(
    gathered_candidates: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_score_bits: torch.Tensor,
    *,
    dcp_world_size: int,
    topk_tokens: int,
) -> None:
    rows = int(candidate_indices.shape[0])
    candidate_width = int(candidate_indices.shape[1])
    if _use_triton_dcp_remap(candidate_indices):
        block_k = 1024
        grid = (rows, triton.cdiv(candidate_width, block_k))
        _b12x_dcp_unpack_gathered_candidates_kernel[grid](
            gathered_candidates,
            candidate_indices,
            candidate_score_bits,
            rows,
            topk_tokens,
            candidate_width,
            BLOCK_K=block_k,
        )
        return

    gathered_view = gathered_candidates.view(dcp_world_size, rows, 2, topk_tokens)
    for rank in range(dcp_world_size):
        start = rank * topk_tokens
        end = start + topk_tokens
        candidate_indices[:, start:end].copy_(gathered_view[rank, :, 0, :])
        candidate_score_bits[:, start:end].copy_(gathered_view[rank, :, 1, :])


def _get_dcp_warmup_params() -> tuple[int, int, int]:
    dcp_world_size = 1
    dcp_rank = 0
    cp_kv_cache_interleave_size = 1
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            parallel_config = vllm_config.parallel_config
            dcp_world_size = int(parallel_config.decode_context_parallel_size)
            cp_kv_cache_interleave_size = int(
                parallel_config.cp_kv_cache_interleave_size
            )
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        if int(dcp_group.world_size) > 1:
            dcp_world_size = int(dcp_group.world_size)
            dcp_rank = int(dcp_group.rank_in_group)
    except Exception:
        pass

    return dcp_world_size, dcp_rank, cp_kv_cache_interleave_size


def _sync_dcp_warmup() -> None:
    """Finish one-time DCP warmup on every rank before graph warmup proceeds."""
    if current_platform.is_cuda():
        torch.accelerator.synchronize()

    try:
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        if int(dcp_group.world_size) <= 1:
            return
        dcp_group.barrier()
    except Exception:
        return
    finally:
        if current_platform.is_cuda():
            torch.accelerator.synchronize()


def _prewarm_b12x_dcp_topk_merge(
    *,
    q_rows: int,
    topk_tokens: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    device: torch.device,
) -> None:
    if dcp_world_size <= 1 or topk_tokens <= 0:
        return

    topk = int(topk_tokens)
    # Graph warmup later exercises both small decode rows and prefill chunks.
    # Use the real runtime merge path here, including DCP all-gather, so Triton
    # and b12x row-topk kernels are loaded before rank-sensitive attention
    # collectives begin.
    warmup_rows = (1, 2, 4, max(1, int(q_rows)))
    seen_rows: set[int] = set()
    base_ids = torch.arange(topk, dtype=torch.int32, device=device)
    for rows in warmup_rows:
        rows = int(rows)
        if rows in seen_rows:
            continue
        seen_rows.add(rows)
        topk_indices = base_ids.expand(rows, topk).clone()
        topk_scores = torch.zeros((rows, topk), dtype=torch.float32, device=device)
        _merge_b12x_dcp_topk(
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            topk_tokens=topk,
            dcp_world_size=int(dcp_world_size),
            dcp_rank=int(dcp_rank),
            cp_kv_cache_interleave_size=int(cp_kv_cache_interleave_size),
        )
        _sync_dcp_warmup()


def _dcp_global_topk_remap(
    topk_indices: torch.Tensor,
    logits: torch.Tensor | None,
    topk_tokens: int,
    rank: int,
    world_size: int,
    interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Convert rank-local DCP top-k into one shared global top-k in place."""
    if world_size <= 1 or topk_indices.numel() == 0:
        return

    from vllm.distributed.parallel_state import get_dcp_group

    candidates = _dcp_pack_topk_candidates(
        topk_indices,
        logits,
        topk_tokens,
        rank,
        world_size,
        interleave,
        row_starts=row_starts,
    )
    all_candidates = get_dcp_group().all_gather(candidates.contiguous(), dim=2)
    all_scores = all_candidates[:, 1, :].view(torch.float32)
    _, selected = torch.topk(all_scores, topk_tokens, dim=1)
    _dcp_finalize_topk_remap(
        all_candidates,
        selected,
        topk_indices,
        topk_tokens,
        rank,
        world_size,
        interleave,
    )


def _assert_b12x_prefill_paged_route(obj: object, *, owner: str) -> None:
    route = getattr(obj, "route", None)
    if route is None:
        route = getattr(getattr(obj, "layout", None), "route", None)
    if route != _B12X_PREFILL_PAGED_ROUTE:
        raise RuntimeError(
            "B12X sparse prefill expected the b12x planner to resolve the paged "
            f"source to {_B12X_PREFILL_PAGED_ROUTE!r}, got {route!r} from {owner}."
        )


def _get_b12x_indexer_paged_supertile_k() -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K")
    tokens = _B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT if raw is None else int(raw)
    tokens = max(tokens, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )


def _get_b12x_paged_indexer_profile_q_rows(q_rows: int) -> int:
    """Return the largest q chunk the real prefill chunker can hand to b12x."""
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    tile_k = _get_b12x_indexer_paged_supertile_k()
    max_q_rows = max(1, max_logits_elems // max(1, tile_k))
    return min(max(1, int(q_rows)), max_q_rows)


def _get_b12x_paged_indexer_profile_k_rows(
    max_model_len: int,
    total_seq_lens: int,
) -> int:
    # The page table is per request, while total_seq_lens is the capacity of
    # the flattened prefill workspace across requests (currently up to
    # 40 * max_model_len).  Feeding that aggregate capacity to the B12X planner
    # produces a kernel schedule for an impossible per-row context.
    if int(max_model_len) > 0:
        return int(max_model_len)
    if int(total_seq_lens) > 0:
        return int(total_seq_lens)
    return _get_b12x_indexer_paged_supertile_k()


def _get_b12x_paged_indexer_profile_warmup_k_rows(profile_k_rows: int) -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_PROFILE_WARMUP_K")
    cap = _get_b12x_indexer_paged_supertile_k() if raw is None else int(raw)
    cap = max(cap, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    cap = (
        (cap + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )
    return min(max(1, int(profile_k_rows)), cap)


def _b12x_profile_rows_or_empty(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    rows = max(1, int(rows))
    if int(tensor.shape[0]) >= rows:
        return tensor[:rows].contiguous()
    return torch.empty(
        (rows, *tuple(tensor.shape[1:])),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _b12x_profile_weights_2d(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_q_heads: int,
    device: torch.device,
) -> torch.Tensor:
    q_rows = max(1, int(q_rows))
    num_q_heads = int(num_q_heads)
    if (
        weights.ndim == 2
        and int(weights.shape[0]) >= q_rows
        and int(weights.shape[1]) == num_q_heads
        and weights.dtype == torch.float32
        and weights.device == device
    ):
        return weights[:q_rows].contiguous()
    return torch.empty(
        (q_rows, num_q_heads),
        dtype=torch.float32,
        device=device,
    )


def _b12x_sparse_indexer_requested(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)

    if envs.VLLM_USE_B12X_SPARSE_INDEXER:
        return True

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False

    backend = vllm_config.attention_config.backend
    if isinstance(backend, str):
        return backend == "B12X_MLA_SPARSE"
    return getattr(backend, "name", None) == "B12X_MLA_SPARSE"


def _ensure_b12x_sparse_indexer_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("B12X sparse indexer/top-k requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError("B12X sparse indexer/top-k currently requires an SM120 GPU.")


def _use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    if not _b12x_sparse_indexer_requested(enabled):
        return False
    _ensure_b12x_sparse_indexer_supported()
    return True


def use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    return _use_b12x_sparse_indexer(enabled)


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _flatten_b12x_paged_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_shape_tail = (
        _B12X_PAGED_INDEX_PAGE_SIZE,
        _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
    )

    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise RuntimeError(
            "b12x paged indexer cache must be rank-3 uint8 with "
            f"shape [num_blocks, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}], got shape={tuple(kv_cache.shape)} "
            f"dtype={kv_cache.dtype}."
        )
    if tuple(kv_cache.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported shape, "
            f"got {tuple(kv_cache.shape)}; expected tail {expected_shape_tail}."
        )
    if kv_cache.stride(1) != expected_shape_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported layout, "
            f"shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}; "
            f"expected inner strides ({expected_shape_tail[1]}, 1)."
        )

    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _B12X_PAGED_INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_b12x_paged_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    topk_scores: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    shared_page_table: bool = False,
    output_physical_slots: bool = False,
) -> torch.Tensor:
    """Run b12x paged indexer top-k with caller-owned scratch.

    b12x sizes scratch from indexer K-cache rows/pages. For compressed sources
    such as C4, the metadata builder has already converted model context tokens
    to indexer K rows before this call. ``active_width`` is the builder-computed
    live K-row window (a metadata tensor, not an in-kernel reduction); when
    None, b12x falls back to the capacity cap.
    """
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        index_topk_fp8,
        plan_indexer_scratch,
    )

    if int(PAGED_INDEX_PAGE_SIZE) != _B12X_PAGED_INDEX_PAGE_SIZE:
        raise RuntimeError(
            "b12x paged indexer page-size contract changed, got "
            f"{PAGED_INDEX_PAGE_SIZE}; expected "
            f"{_B12X_PAGED_INDEX_PAGE_SIZE}."
        )

    index_k_cache = _flatten_b12x_paged_index_cache(kv_cache)
    expected_num_q_heads = int(q_fp8.shape[1])
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=q_fp8.device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=expected_num_q_heads,
            max_q_rows=int(q_fp8.shape[0]),
            max_page_table_width=int(block_table.shape[1]),
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch plan")
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=shared_page_table,
        output_physical_slots=output_physical_slots,
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(binding, owner="binding")
    return index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
        page_size=PAGED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        out_indices=topk_indices,
        out_scores=topk_scores,
    )


def _merge_b12x_dcp_topk(
    *,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor | None,
    topk_tokens: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    if dcp_world_size <= 1 or topk_indices.numel() == 0:
        return
    if topk_scores is None:
        raise RuntimeError(
            "B12X sparse indexer DCP requires a topk_scores_buffer for "
            "cross-rank candidate merge."
        )

    from b12x.attention.indexer.tiled_topk import run_row_topk

    from vllm.distributed.parallel_state import get_dcp_group
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_dcp_local_topk_to_global,
        triton_gather_topk_ids_by_position,
    )

    triton_convert_dcp_local_topk_to_global(
        topk_indices,
        topk_scores,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )

    if not topk_scores.is_contiguous():
        raise RuntimeError("B12X sparse indexer DCP requires contiguous topk scores.")

    rows = int(topk_indices.shape[0])
    candidate_width = int(dcp_world_size * topk_tokens)
    (
        candidates,
        gathered_candidates,
        candidate_indices,
        candidate_score_bits,
        candidate_lengths,
    ) = current_workspace_manager().get_simultaneous(
        ((rows, 2, int(topk_tokens)), torch.int32),
        ((rows * int(dcp_world_size), 2, int(topk_tokens)), torch.int32),
        ((rows, candidate_width), torch.int32),
        ((rows, candidate_width), torch.int32),
        ((rows,), torch.int32),
    )
    candidates[:, 0, :].copy_(topk_indices)
    candidates[:, 1, :].copy_(topk_scores.view(torch.int32))

    dcp_group = get_dcp_group()
    _dcp_all_gather_first_dim_into(
        dcp_group,
        candidates,
        gathered_candidates,
    )
    _unpack_b12x_dcp_gathered_candidates(
        gathered_candidates,
        candidate_indices,
        candidate_score_bits,
        dcp_world_size=dcp_world_size,
        topk_tokens=int(topk_tokens),
    )
    candidate_scores = candidate_score_bits.view(torch.float32)
    candidate_lengths.fill_(candidate_width)
    run_row_topk(
        row_logits=candidate_scores,
        lengths=candidate_lengths,
        topk=int(topk_tokens),
        output_values=topk_scores,
        output_indices=topk_indices,
    )
    triton_gather_topk_ids_by_position(
        candidate_indices,
        topk_indices,
        topk_indices,
    )


def _convert_b12x_dcp_local_topk_to_global(
    *,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor | None,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    if dcp_world_size <= 1 or topk_indices.numel() == 0:
        return
    if topk_scores is None:
        raise RuntimeError(
            "B12X sparse indexer DCP requires a topk_scores_buffer for "
            "local-to-global candidate conversion."
        )

    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_dcp_local_topk_to_global,
    )

    triton_convert_dcp_local_topk_to_global(
        topk_indices,
        topk_scores,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )


def _prewarm_b12x_paged_indexer_prefill(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    profile_q_rows: int,
    profile_k_rows: int,
    page_table_k_rows: int | None = None,
    output_physical_slots: bool = False,
) -> None:
    q_rows = max(1, int(profile_q_rows))
    k_rows = max(1, int(profile_k_rows))
    page_table_rows = (
        k_rows if page_table_k_rows is None else max(1, int(page_table_k_rows))
    )
    num_q_heads = int(q_quant.shape[1])
    page_table_width = max(
        1,
        (page_table_rows + _B12X_PAGED_INDEX_PAGE_SIZE - 1)
        // _B12X_PAGED_INDEX_PAGE_SIZE,
    )
    q_warm = _b12x_profile_rows_or_empty(q_quant, q_rows)
    weights_warm = _b12x_profile_weights_2d(
        weights,
        q_rows=q_rows,
        num_q_heads=num_q_heads,
        device=q_quant.device,
    )
    seq_lens = torch.full(
        (q_rows,),
        k_rows,
        dtype=torch.int32,
        device=q_quant.device,
    )
    block_table = torch.zeros(
        (1, page_table_width),
        dtype=torch.int32,
        device=q_quant.device,
    ).expand(q_rows, page_table_width)
    kv_cache_warm = kv_cache
    if int(kv_cache_warm.shape[0]) <= 0:
        kv_cache_warm = torch.empty(
            (
                1,
                _B12X_PAGED_INDEX_PAGE_SIZE,
                _B12X_PAGED_INDEX_HEAD_DIM + _B12X_PAGED_INDEX_SCALE_BYTES,
            ),
            dtype=torch.uint8,
            device=q_quant.device,
        )
    topk_indices = torch.empty(
        (q_rows, int(topk_tokens)),
        dtype=torch.int32,
        device=q_quant.device,
    )
    _run_b12x_paged_topk(
        q_fp8=q_warm,
        weights=weights_warm,
        kv_cache=kv_cache_warm,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=int(topk_tokens),
        shared_page_table=True,
        output_physical_slots=output_physical_slots,
    )


def _prewarm_b12x_contiguous_prefill_variants(
    *,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    profile_q_rows: int,
) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or q_quant.dtype != fp8_dtype:
        return
    if q_quant.device.type != "cuda":
        return
    if q_quant.ndim != 3 or int(q_quant.shape[2]) != _B12X_PAGED_INDEX_HEAD_DIM:
        return

    try:
        from b12x.attention.indexer.contiguous_kernel import (
            run_contiguous_logits_kernel,
        )
        from b12x.attention.indexer.tiled_topk import run_tiled_topk
    except (AttributeError, ImportError, ModuleNotFoundError):
        return

    q_rows = max(1, int(profile_q_rows))
    num_q_heads = int(q_quant.shape[1])
    topk = int(topk_tokens)
    q_warm = _b12x_profile_rows_or_empty(q_quant, q_rows)
    weights_warm = _b12x_profile_weights_2d(
        weights,
        q_rows=q_rows,
        num_q_heads=num_q_heads,
        device=q_quant.device,
    )

    for block_k in (
        _B12X_CONTIGUOUS_PREFILL_BLOCK_K,
        _B12X_CONTIGUOUS_PREFILL512_BLOCK_K,
    ):
        if block_k == _B12X_CONTIGUOUS_PREFILL512_BLOCK_K and (
            q_rows < _B12X_CONTIGUOUS_PREFILL512_MIN_Q_ROWS
            or num_q_heads not in _B12X_CONTIGUOUS_PREFILL512_SUPPORTED_HEADS
        ):
            continue
        k_rows = max(block_k, topk, 1)
        if block_k == _B12X_CONTIGUOUS_PREFILL512_BLOCK_K:
            k_rows = max(k_rows, _B12X_CONTIGUOUS_PREFILL512_MIN_K_ROWS)
        k_rows = (k_rows + block_k - 1) // block_k * block_k
        k_quant = torch.empty(
            (k_rows, _B12X_PAGED_INDEX_HEAD_DIM),
            dtype=fp8_dtype,
            device=q_quant.device,
        )
        k_scale = torch.empty((k_rows,), dtype=torch.float32, device=q_quant.device)
        k_start = torch.zeros((q_rows,), dtype=torch.int32, device=q_quant.device)
        k_end = torch.full(
            (q_rows,),
            k_rows,
            dtype=torch.int32,
            device=q_quant.device,
        )
        num_q_tiles = (
            q_rows + _B12X_PAGED_INDEX_TILE_BLOCK_Q - 1
        ) // _B12X_PAGED_INDEX_TILE_BLOCK_Q
        num_k_tiles = (k_rows + block_k - 1) // block_k
        tile_logits = torch.empty(
            (num_q_tiles * num_k_tiles * _B12X_PAGED_INDEX_TILE_BLOCK_Q * block_k,),
            dtype=torch.float32,
            device=q_quant.device,
        )
        output_values = torch.empty(
            (q_rows, topk),
            dtype=torch.float32,
            device=q_quant.device,
        )
        output_indices = torch.empty(
            (q_rows, topk),
            dtype=torch.int32,
            device=q_quant.device,
        )
        run_contiguous_logits_kernel(
            q_fp8=q_warm,
            weights=weights_warm,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=0,
            tile_num_k_tiles=num_k_tiles,
            prefill_block_k=block_k,
        )
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=None,
            lengths=k_end,
            topk=topk,
            block_q=_B12X_PAGED_INDEX_TILE_BLOCK_Q,
            block_k=block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
            input_extent=k_rows,
            zero_row_start=True,
        )


def _reserve_b12x_paged_indexer_scratch(
    *,
    q_rows: int,
    num_q_heads: int,
    topk_tokens: int,
    total_k_rows: int,
    device: torch.device,
    shared_page_table: bool = False,
) -> None:
    from b12x.attention.indexer import (
        INDEXER_SOURCE_LAYOUT_PAGED,
        PAGED_INDEX_PAGE_SIZE,
        B12XIndexerScratchCaps,
        plan_indexer_scratch,
    )

    page_table_width = max(
        1,
        (max(1, int(total_k_rows)) + int(PAGED_INDEX_PAGE_SIZE) - 1)
        // int(PAGED_INDEX_PAGE_SIZE),
    )
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=int(num_q_heads),
            max_q_rows=max(1, int(q_rows)),
            max_page_table_width=page_table_width,
            topk=int(topk_tokens),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=bool(shared_page_table),
        )
    )
    if shared_page_table:
        _assert_b12x_prefill_paged_route(plan, owner="scratch reservation plan")
    current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    use_b12x_sparse_indexer: bool = False,
    output_physical_slots: bool = False,
    topk_scores_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        run_profile_work = (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and forward_context.batch_descriptor is not None
        )
        if not run_profile_work:
            return sparse_attn_indexer_fake(
                hidden_states,
                k_cache_prefix,
                kv_cache,
                q_quant,
                q_scale,
                k,
                weights,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                total_seq_lens,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_fp4_cache,
                use_b12x_sparse_indexer,
                output_physical_slots,
            )

        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        if _b12x_sparse_indexer_requested(use_b12x_sparse_indexer):
            _ensure_b12x_sparse_indexer_supported()
            profile_q_rows = _get_b12x_paged_indexer_profile_q_rows(
                int(q_quant.shape[0])
            )
            profile_k_rows = _get_b12x_paged_indexer_profile_k_rows(
                max_model_len=max_model_len,
                total_seq_lens=total_seq_lens,
            )
            warmup_k_rows = _get_b12x_paged_indexer_profile_warmup_k_rows(
                profile_k_rows
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=False,
            )
            _reserve_b12x_paged_indexer_scratch(
                q_rows=profile_q_rows,
                num_q_heads=int(q_quant.shape[1]),
                topk_tokens=int(topk_tokens),
                total_k_rows=profile_k_rows,
                device=q_quant.device,
                shared_page_table=True,
            )
            (
                dcp_world_size,
                dcp_rank,
                cp_kv_cache_interleave_size,
            ) = _get_dcp_warmup_params()
            _prewarm_b12x_paged_indexer_prefill(
                q_quant=q_quant,
                weights=weights,
                kv_cache=kv_cache,
                topk_tokens=int(topk_tokens),
                profile_q_rows=profile_q_rows,
                profile_k_rows=warmup_k_rows,
                page_table_k_rows=profile_k_rows,
                output_physical_slots=output_physical_slots,
            )
            _prewarm_b12x_dcp_topk_merge(
                q_rows=profile_q_rows,
                topk_tokens=int(topk_tokens),
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                device=q_quant.device,
            )
            _prewarm_b12x_contiguous_prefill_variants(
                q_quant=q_quant,
                weights=weights,
                topk_tokens=int(topk_tokens),
                profile_q_rows=profile_q_rows,
            )
        else:
            # Reserve workspace for indexer during profiling run.
            current_workspace_manager().get_simultaneous(
                values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
            )

            # Dummy allocation to simulate peak logits tensor memory during
            # inference. The B12X path above streams one supertile at a time and
            # has already reserved its fixed scratch via the workspace manager.
            # FP8 elements so elements == bytes.
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            dcp_rank,
            dcp_world_size,
            cp_kv_cache_interleave_size,
            skip_topk_buffer_clear,
            use_b12x_sparse_indexer,
            output_physical_slots,
            topk_scores_buffer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    # DCP scalars arrive as op arguments (resolved once at layer construction),
    # not through per-step metadata.
    dcp_global_topk = _dcp_global_topk_requested() and dcp_world_size > 1
    if output_physical_slots:
        if not _use_b12x_sparse_indexer(use_b12x_sparse_indexer):
            raise RuntimeError(
                "physical sparse-indexer output requires the b12x paged indexer"
            )
        if dcp_world_size != 1:
            raise RuntimeError(
                "physical sparse-indexer output does not support decode context "
                "parallelism"
            )

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the
    # redundant fill. The b12x indexer path clears/fills the buffer itself.
    if not skip_topk_buffer_clear and not _use_b12x_sparse_indexer(
        use_b12x_sparse_indexer
    ):
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )
        if not use_b12x_indexer:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        for chunk in prefill_metadata.chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            weights_slice = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if use_b12x_indexer and chunk.total_seq_lens <= 0:
                topk_indices.fill_(-1)
                if dcp_global_topk:
                    if topk_scores_buffer is None:
                        raise RuntimeError(
                            "B12X sparse indexer DCP requires topk_scores_buffer."
                        )
                    topk_scores = topk_scores_buffer[
                        chunk.token_start : chunk.token_end, :topk_tokens
                    ]
                    topk_scores.fill_(-float("inf"))
                    _merge_b12x_dcp_topk(
                        topk_indices=topk_indices,
                        topk_scores=topk_scores,
                        topk_tokens=topk_tokens,
                        dcp_world_size=dcp_world_size,
                        dcp_rank=dcp_rank,
                        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    )
                    topk_indices.masked_fill_(topk_scores == -float("inf"), -1)
                continue

            if use_b12x_indexer:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12X sparse prefill requires single-request chunks so "
                        "the page table can be row-shared without packing."
                    )
                row_has_no_kv = chunk.cu_seqlen_ke <= chunk.cu_seqlen_ks
                seq_lens = torch.where(
                    row_has_no_kv,
                    torch.zeros_like(chunk.cu_seqlen_ks),
                    chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
                )
                block_table = chunk.block_table[:1].expand(
                    int(q_slice.shape[0]),
                    int(chunk.block_table.shape[1]),
                )
                topk_scores = None
                if dcp_world_size > 1:
                    if topk_scores_buffer is None:
                        raise RuntimeError(
                            "B12X sparse indexer DCP requires topk_scores_buffer."
                        )
                    topk_scores = topk_scores_buffer[
                        chunk.token_start : chunk.token_end, :topk_tokens
                    ]
                _run_b12x_paged_topk(
                    q_fp8=q_slice.contiguous(),
                    weights=weights_slice.contiguous(),
                    kv_cache=kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
                    topk_indices=topk_indices,
                    topk_tokens=topk_tokens,
                    topk_scores=topk_scores,
                    shared_page_table=True,
                    output_physical_slots=output_physical_slots,
                )
                if dcp_global_topk:
                    _merge_b12x_dcp_topk(
                        topk_indices=topk_indices,
                        topk_scores=topk_scores,
                        topk_tokens=topk_tokens,
                        dcp_world_size=dcp_world_size,
                        dcp_rank=dcp_rank,
                        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    )
                else:
                    _convert_b12x_dcp_local_topk_to_global(
                        topk_indices=topk_indices,
                        topk_scores=topk_scores,
                        dcp_world_size=dcp_world_size,
                        dcp_rank=dcp_rank,
                        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    )
                continue

            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )
            if dcp_global_topk:
                _dcp_global_topk_remap(
                    topk_indices,
                    logits,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=chunk.cu_seqlen_ks,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )

        b12x_seq_lens = decode_metadata.seq_lens
        b12x_block_table = decode_metadata.block_table
        if b12x_seq_lens.dim() == 2:
            b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
            if num_decode_tokens == b12x_batch_size * b12x_next_n:
                b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                b12x_block_table = b12x_block_table.repeat_interleave(
                    b12x_next_n, dim=0
                ).contiguous()
        b12x_decode_supported = (
            use_b12x_indexer
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if use_b12x_indexer and (
            decode_metadata.requires_padding or b12x_seq_lens.dim() != 1
        ):
            raise RuntimeError(
                "b12x sparse indexer decode requires an unpadded rank-1 "
                "seq_lens contract after native-spec normalization; refusing "
                "to fall back to DeepGEMM. "
                f"requires_padding={decode_metadata.requires_padding}, "
                f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}, "
                f"num_decode_tokens={num_decode_tokens}."
            )

        if b12x_decode_supported:
            # Prefix slice of an already-contiguous buffer stays contiguous
            # (b12x_seq_lens/b12x_block_table are normalized contiguous upstream),
            # so .contiguous() here was a guaranteed no-op per decoded token.
            seq_lens = b12x_seq_lens[:num_decode_tokens]
            block_table = b12x_block_table[:num_decode_tokens]
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            topk_scores = None
            if dcp_world_size > 1:
                if topk_scores_buffer is None:
                    raise RuntimeError(
                        "B12X sparse indexer DCP requires topk_scores_buffer."
                    )
                topk_scores = topk_scores_buffer[:num_decode_tokens, :topk_tokens]
            # b12x consumes indexer K-row metadata. DSV4/C4 seq_lens and
            # active_width have already been compressed by the metadata builder;
            # GLM passes one K row per context token.
            _run_b12x_paged_topk(
                q_fp8=q_quant[:num_decode_tokens].contiguous(),
                weights=weights[:num_decode_tokens].contiguous(),
                kv_cache=kv_cache,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=decode_metadata.schedule_metadata,
                active_width=decode_metadata.active_width,
                topk_indices=topk_indices,
                topk_tokens=topk_tokens,
                topk_scores=topk_scores,
                output_physical_slots=output_physical_slots,
            )
            if dcp_global_topk:
                _merge_b12x_dcp_topk(
                    topk_indices=topk_indices,
                    topk_scores=topk_scores,
                    topk_tokens=topk_tokens,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
            else:
                _convert_b12x_dcp_local_topk_to_global(
                    topk_indices=topk_indices,
                    topk_scores=topk_scores,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
            return topk_indices_buffer

        schedule_metadata = decode_metadata.schedule_metadata
        if schedule_metadata is None:
            raise RuntimeError(
                "DeepGEMM/XPU sparse indexer decode requires schedule metadata; "
                "enable VLLM_USE_B12X_SPARSE_INDEXER for the b12x path or check "
                "the indexer metadata builder."
            )

        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        paged_weights = weights[:num_padded_tokens]
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                paged_weights,
                seq_lens_xpu,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len,
            )
        else:
            from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                paged_weights,
                seq_lens,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                # SM100 varlen path when set; None keeps the non-varlen path.
                indices=decode_metadata.indices,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                decode_metadata.max_seq_len
                if decode_metadata.max_seq_len is not None
                else logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    use_b12x_sparse_indexer: bool = False,
    output_physical_slots: bool = False,
    topk_scores_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    del topk_scores_buffer, output_physical_slots
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "topk_scores_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        topk_scores_buffer: torch.Tensor | None = None,
        output_physical_slots: bool = False,
        num_q_heads: int | None = None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_scores_buffer = topk_scores_buffer
        self.output_physical_slots = bool(output_physical_slots)
        self.num_q_heads = num_q_heads
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_b12x_sparse_indexer = use_b12x_sparse_indexer()
        if self.output_physical_slots and not self.use_b12x_sparse_indexer:
            raise RuntimeError(
                "physical sparse-indexer output requires the b12x paged indexer"
            )
        if self.use_b12x_sparse_indexer:
            if self.use_fp4_cache:
                raise RuntimeError(
                    "B12X sparse indexer/top-k requires the FP8 paged index "
                    "cache; disable use_fp4_indexer_cache."
                )
        elif current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            False,  # skip_topk_buffer_clear: cleared by the fused-norm path only
            self.use_b12x_sparse_indexer,
            self.output_physical_slots,
            self.topk_scores_buffer,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
