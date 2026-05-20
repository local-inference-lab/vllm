# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
import json
import os

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.b12x_contract import (
    b12x_sparse_indexer_active_for_config,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)
_SINGLE_REQ_CONTEXT_CACHE: dict[
    tuple[str, int], tuple[torch.Tensor, torch.Tensor]
] = {}
_USE_SGL_KERNEL_FAST_TOPK_TRANSFORM = bool(
    int(os.getenv("VLLM_USE_SGL_KERNEL_FAST_TOPK_TRANSFORM", "0"))
)
_DEBUG_INDEXER_BLOCK_WIDTH = os.getenv("VLLM_DEBUG_INDEXER_BLOCK_WIDTH", "0") == "1"
_DEBUG_INDEXER_BLOCK_WIDTH_FILE = os.getenv(
    "VLLM_DEBUG_INDEXER_BLOCK_WIDTH_FILE", "/tmp/vllm_indexer_block_width.jsonl"
)
_DEBUG_INDEXER_BLOCK_WIDTH_MAX = int(
    os.getenv("VLLM_DEBUG_INDEXER_BLOCK_WIDTH_MAX", "128")
)
_DCP_FULL_INDEXER_STATIC_BLOCK_TABLE_MODE = (
    os.getenv("VLLM_DCP_FULL_INDEXER_STATIC_BLOCK_TABLE", "auto").strip().lower()
)


def _use_b12x_sparse_indexer_for_config(vllm_config: VllmConfig | None) -> bool:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ):
        return False
    return (
        envs.VLLM_USE_B12X_SPARSE_INDEXER
        or b12x_sparse_indexer_active_for_config(vllm_config)
    )


def _align_block_table_max_num_blocks(max_num_blocks: int, block_size: int) -> int:
    # Keep this in sync with MultiGroupBlockTable's row-width padding.
    if block_size <= 128:
        alignment = 128 // block_size
        return cdiv(max_num_blocks, alignment) * alignment
    return max_num_blocks


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # PDL: signal we're done with launch-prep, ready to read producer output.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)

    # PDL: signal that dependents may now begin.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _expand_block_table_to_page_table_1_kernel(
    block_table_ptr,
    out_ptr,
    block_table_stride,
    out_stride,
    block_table_width: tl.constexpr,
    total_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)

    # PDL: wait for producer kernel writes to complete.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    mask = offs < total_width
    block_id = offs // BLOCK_SIZE
    inblock = offs - block_id * BLOCK_SIZE
    physical_block = tl.load(
        block_table_ptr + row * block_table_stride + block_id,
        mask=mask & (block_id < block_table_width),
        other=-1,
    )
    physical_token = physical_block * BLOCK_SIZE + inblock
    physical_token = tl.where(physical_block >= 0, physical_token, -1)
    tl.store(out_ptr + row * out_stride + offs, physical_token, mask=mask)

    # PDL: signal dependents.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1 if current_platform.is_rocm() else 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    b12x_workspace: object | None = None
    page_table_1: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


# TODO (zyongye) optimize this, this is now vibe coded
def kv_spans_from_batches(
    start_seq_loc: torch.Tensor,
    seq_len_per_batch: torch.Tensor,
    device: torch.device,
    query_slice: slice | None = None,
    *,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    global_seq_len_per_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of
                     selected tokens per batch.
            Example: [0, 2, 4, 7] ->
                     batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B],
                         full sequence length (KV length) of each batch.
                         Example: [5, 9, 4].

    Returns:
      start_tensor: 1D long tensor [N], start offset in the
                    concatenated KV cache for each token's batch.
      end_location: 1D long tensor [N],
                    **exclusive** end = start + token's local position.
                    (So the attended KV slice is kv[start:end].)

    Assumes each batch contributes its full `seq_len_per_batch[i]`
    keys to the KV cache, andthe selected tokens within a batch
    are the **last** `counts[i]` positions of that sequence.
    """
    q = start_seq_loc.to(dtype=torch.long)
    # In DCP, the flattened K workspace is local to each DCP rank while query
    # positions are global. L is the local final KV length per request; global_L
    # is used only to reconstruct each query token's global end position.
    L = seq_len_per_batch.to(dtype=torch.long)
    global_L = (
        L
        if global_seq_len_per_batch is None
        else global_seq_len_per_batch.to(dtype=torch.long)
    )
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"
    assert global_L.dim() == 1 and global_L.numel() == L.numel()

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]  # [B]
    N = int(q[-1].item())  # total selected tokens
    B = L.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    if query_slice is None:
        slice_start, slice_stop = 0, N
    else:
        slice_start = 0 if query_slice.start is None else query_slice.start
        slice_stop = N if query_slice.stop is None else query_slice.stop
        if slice_start < 0 or slice_stop < slice_start or slice_stop > N:
            raise ValueError(
                f"Invalid query_slice={query_slice} for {N} selected tokens"
            )

    slice_len = slice_stop - slice_start
    if slice_len == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    if B == 1 and dcp_world_size <= 1:
        # Common long-prefill path: avoid building CPU vectors and copying them
        # to GPU for every chunk. With one request, all starts are zero and ends
        # are a contiguous range in that request's KV span.
        base = int((L[0] - counts[0]).item()) + slice_start + 1
        return (
            torch.zeros(slice_len, dtype=torch.int32, device=device),
            torch.arange(base, base + slice_len, dtype=torch.int32, device=device),
        )

    # KV start offsets per batch in the concatenated local KV workspace.
    kv_starts_per_batch = torch.cumsum(L, dim=0) - L  # [B]

    if dcp_world_size > 1:
        if slice_start != 0 or slice_stop != N:
            first_batch = int(
                torch.searchsorted(
                    q, torch.tensor(slice_start, dtype=q.dtype), right=True
                ).item()
                - 1
            )
            last_batch = int(
                torch.searchsorted(
                    q, torch.tensor(slice_stop - 1, dtype=q.dtype), right=True
                ).item()
                - 1
            )
            first_batch = max(0, min(first_batch, B - 1))
            last_batch = max(first_batch, min(last_batch, B - 1))

            batch_ids_compact = torch.arange(first_batch, last_batch + 1)
            overlap_start = torch.maximum(
                q[batch_ids_compact],
                torch.tensor(slice_start, dtype=q.dtype),
            )
            overlap_stop = torch.minimum(
                q[batch_ids_compact + 1],
                torch.tensor(slice_stop, dtype=q.dtype),
            )
            overlap_counts = overlap_stop - overlap_start
            valid = overlap_counts > 0
            batch_ids_compact = batch_ids_compact[valid]
            overlap_start = overlap_start[valid]
            overlap_counts = overlap_counts[valid]

            batch_id = torch.repeat_interleave(
                batch_ids_compact, overlap_counts, output_size=slice_len
            )
            segment_offsets = torch.cumsum(overlap_counts, dim=0) - overlap_counts
            token_pos_in_slice = torch.arange(slice_len, dtype=torch.long)
            global_token_pos = torch.repeat_interleave(
                overlap_start, overlap_counts, output_size=slice_len
            ) + (
                token_pos_in_slice
                - torch.repeat_interleave(
                    segment_offsets, overlap_counts, output_size=slice_len
                )
            )
        else:
            batch_id = torch.repeat_interleave(
                torch.arange(B), counts, output_size=N
            )
            global_token_pos = torch.arange(N, dtype=torch.long)

        pos_within_req = global_token_pos - q[batch_id] + 1
        global_end = global_L[batch_id] - counts[batch_id] + pos_within_req
        local_pos = get_dcp_local_seq_lens(
            global_end.to(torch.int32),
            dcp_world_size,
            dcp_rank,
            cp_kv_cache_interleave_size,
        ).to(torch.long)
        start_tensor = kv_starts_per_batch[batch_id]
        end_location = start_tensor + local_pos
        return start_tensor.int().to(device), end_location.int().to(device)

    if slice_start != 0 or slice_stop != N:
        # Build only the requested query sub-slice. The caller may split a long
        # request into many query chunks; constructing the full [N] vectors for
        # every sub-chunk creates large repeated CPU work and stalls prefill.
        first_batch = int(
            torch.searchsorted(
                q, torch.tensor(slice_start, dtype=q.dtype), right=True
            ).item()
            - 1
        )
        last_batch = int(
            torch.searchsorted(
                q, torch.tensor(slice_stop - 1, dtype=q.dtype), right=True
            ).item()
            - 1
        )
        first_batch = max(0, min(first_batch, B - 1))
        last_batch = max(first_batch, min(last_batch, B - 1))

        batch_ids_compact = torch.arange(first_batch, last_batch + 1)
        overlap_start = torch.maximum(
            q[batch_ids_compact],
            torch.tensor(slice_start, dtype=q.dtype),
        )
        overlap_stop = torch.minimum(
            q[batch_ids_compact + 1],
            torch.tensor(slice_stop, dtype=q.dtype),
        )
        overlap_counts = overlap_stop - overlap_start
        valid = overlap_counts > 0
        batch_ids_compact = batch_ids_compact[valid]
        overlap_start = overlap_start[valid]
        overlap_counts = overlap_counts[valid]

        batch_id = torch.repeat_interleave(
            batch_ids_compact, overlap_counts, output_size=slice_len
        )
        segment_offsets = torch.cumsum(overlap_counts, dim=0) - overlap_counts
        token_pos_in_slice = torch.arange(slice_len, dtype=torch.long)
        global_token_pos = torch.repeat_interleave(
            overlap_start, overlap_counts, output_size=slice_len
        ) + (
            token_pos_in_slice
            - torch.repeat_interleave(
                segment_offsets, overlap_counts, output_size=slice_len
            )
        )

        start_tensor = kv_starts_per_batch[batch_id]
        local_pos = L[batch_id] - counts[batch_id] + (global_token_pos - q[batch_id] + 1)
        end_location = start_tensor + local_pos

        return start_tensor.int().to(device), end_location.int().to(device)

    # For each selected token, which batch does it belong to?
    batch_id = torch.repeat_interleave(torch.arange(B), counts, output_size=N)  # [N]

    # Map batch KV start to each token
    start_tensor = kv_starts_per_batch[batch_id]  # [N]

    # End-align local positions inside each batch:
    # local_pos = L[b] - counts[b] + (1..counts[b])  for each batch b
    L_expand = torch.repeat_interleave(L, counts, output_size=N)  # [N]
    m_expand = torch.repeat_interleave(counts, counts, output_size=N)  # [N]
    # position within the selected block: 1..counts[b]
    pos_within = (
        torch.arange(N, dtype=torch.long)
        - torch.repeat_interleave(q[:-1], counts, output_size=N)
        + 1
    )

    local_pos = L_expand - m_expand + pos_within  # [N], 1-based
    end_location = start_tensor + local_pos  # exclusive end

    return start_tensor.int().to(device), end_location.int().to(device)


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1
    natively_supported_next_n: list[int] = [1, 2]
    # TODO (matt): integrate kernel with next_n = 4 support

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.index_topk = int(
            getattr(self.vllm_config.model_config.hf_config, "index_topk", 2048)
        )
        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens
        self.use_flattening = next_n not in self.natively_supported_next_n
        self.dcp_world_size = self.vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        has_full_cudagraphs = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )
        if _DCP_FULL_INDEXER_STATIC_BLOCK_TABLE_MODE in {"1", "true", "yes", "on"}:
            self.dcp_full_indexer_static_block_table = True
        elif _DCP_FULL_INDEXER_STATIC_BLOCK_TABLE_MODE in {
            "0",
            "false",
            "no",
            "off",
        }:
            self.dcp_full_indexer_static_block_table = False
        else:
            # DCP decode under full CUDA graph must keep graph-stable block-table
            # width. Dynamically narrowing it to the local decode length can make
            # the sparse MLA indexer feed stale/incorrect page metadata after a
            # long prefill, corrupting generation.
            self.dcp_full_indexer_static_block_table = (
                self.dcp_world_size > 1 and has_full_cudagraphs
            )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        if not self.use_flattening and next_n > 1:
            # Native MTP: 2D buffer for per-token seq_lens.
            # Flattening path is never used, so no expanded_seq_lens_buffer.
            self.decode_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_seqs, next_n),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Flattening or no MTP: 1D buffer for expanded per-token seq_lens.
            self.decode_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
        self.arange_buffer = torch.arange(
            scheduler_config.max_num_seqs * next_n,
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        max_num_blocks_per_req = _align_block_table_max_num_blocks(
            max_num_blocks_per_req, self.kv_cache_spec.block_size
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.page_table_1_buffer = torch.empty(
            (
                scheduler_config.max_num_seqs,
                max_num_blocks_per_req * self.kv_cache_spec.block_size,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.cu_seqlens_q_buffer = torch.arange(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )
        self._single_req_context_cache: tuple[
            int, torch.Tensor, torch.Tensor
        ] | None = None
        self._debug_indexer_block_width_count = 0

    def build_one_prefill_chunk(
        self,
        req_slice: slice,
        query_slice: slice,
        query_start_loc_cpu,
        seq_lens_cpu,
        global_seq_lens_cpu,
        block_table,
        skip_kv_gather: bool = False,
        context_cache: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, int | torch.Tensor]
        ]
        | None = None,
    ) -> DeepseekV32IndexerPrefillChunkMetadata:
        prefill_query_start_loc = (
            query_start_loc_cpu[req_slice.start : req_slice.stop + 1]
            - query_start_loc_cpu[req_slice.start]
        )
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            prefill_query_start_loc,
            seq_lens_cpu[req_slice],
            self.device,
            query_slice=query_slice,
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            global_seq_len_per_batch=(
                global_seq_lens_cpu[req_slice]
                if global_seq_lens_cpu is not None
                else None
            ),
        )
        token_start = query_start_loc_cpu[req_slice.start].item()
        num_reqs = req_slice.stop - req_slice.start
        cache_key = (req_slice.start, req_slice.stop)
        cached_context = context_cache.get(cache_key) if context_cache is not None else None
        if cached_context is not None:
            token_to_seq, cu_seq_lens, total_seq_lens = cached_context
        else:
            if num_reqs == 1:
                total_seq_lens = int(seq_lens_cpu[req_slice.start].item())
                single_req_cache = self._single_req_context_cache
                if (
                    single_req_cache is not None
                    and single_req_cache[0] == total_seq_lens
                ):
                    _, token_to_seq, cu_seq_lens = single_req_cache
                else:
                    global_cache_key = (str(self.device), total_seq_lens)
                    global_cached = _SINGLE_REQ_CONTEXT_CACHE.get(global_cache_key)
                    if global_cached is not None:
                        token_to_seq, cu_seq_lens = global_cached
                    else:
                        token_to_seq = torch.zeros(
                            total_seq_lens, dtype=torch.int32, device=self.device
                        )
                        cu_seq_lens = torch.tensor(
                            [0, total_seq_lens], dtype=torch.int32, device=self.device
                        )
                        if len(_SINGLE_REQ_CONTEXT_CACHE) > 64:
                            _SINGLE_REQ_CONTEXT_CACHE.clear()
                        _SINGLE_REQ_CONTEXT_CACHE[global_cache_key] = (
                            token_to_seq,
                            cu_seq_lens,
                        )
                    self._single_req_context_cache = (
                        total_seq_lens,
                        token_to_seq,
                        cu_seq_lens,
                    )
            else:
                total_seq_lens = int(seq_lens_cpu[req_slice].sum().item())
                seq_idx = torch.arange(0, num_reqs, dtype=torch.int32)
                token_to_seq = torch.repeat_interleave(
                    seq_idx, seq_lens_cpu[req_slice]
                ).to(self.device)
                cu_seq_lens = (
                    torch.cat(
                        [
                            torch.zeros(1, dtype=torch.int32),
                            seq_lens_cpu[req_slice].cumsum(dim=0),
                        ]
                    )
                    .to(torch.int32)
                    .to(self.device)
                )
            if context_cache is not None:
                context_cache[cache_key] = (token_to_seq, cu_seq_lens, total_seq_lens)
        assert total_seq_lens <= self.max_prefill_buffer_size

        return DeepseekV32IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            total_seq_lens=total_seq_lens,
            block_table=block_table[req_slice],
            token_start=token_start + query_slice.start,
            token_end=token_start + query_slice.stop,
            num_reqs=num_reqs,
            skip_kv_gather=skip_kv_gather,
        )

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
        global_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, max_decode_len)
        for native MTP.
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _pdl_kwargs = (
                    {"USE_GDC": True, "launch_pdl": True}
                    if (
                        current_platform.is_cuda()
                        and current_platform.has_device_capability(90)
                    )
                    else {}
                )
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                    **_pdl_kwargs,
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_global = (
                        global_seq_lens[:num_decodes].unsqueeze(1)
                        - max_decode_len
                        + 1
                        + self.offsets_buffer[:max_decode_len]
                    ).reshape(-1)
                    self.decode_seq_lens_buffer[:num_decode_tokens].copy_(
                        get_dcp_local_seq_lens(
                            expanded_global,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        ),
                        non_blocking=True,
                    )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_offsets = torch.repeat_interleave(
                    seq_lens_for_expansion - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                expanded_seq_lens = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                self.decode_seq_lens_buffer[:actual_expanded] = expanded_seq_lens
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 2
                # (B, next_n): token j attends to L - next_n + j + 1 KV tokens
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_seq_lens = (
                    seq_lens_for_expansion.unsqueeze(1)
                    - next_n
                    + 1
                    + self.offsets_buffer
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens.reshape(-1),
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).reshape(num_decodes, next_n)
                self.decode_seq_lens_buffer[:num_decodes] = expanded_seq_lens
                seq_lens = self.decode_seq_lens_buffer[:num_decodes]
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            global_seq_lens_cpu = (
                common_attn_metadata._seq_lens_cpu
                if common_attn_metadata._seq_lens_cpu is not None
                else common_attn_metadata.seq_lens_cpu_upper_bound
            )
            if global_seq_lens_cpu is None:
                raise RuntimeError(
                    "B12X MLA prefill metadata requires CPU seq_lens shadow "
                    "or seq_lens_cpu_upper_bound to avoid D2H sync."
                )
            indexer_seq_lens_cpu = (
                common_attn_metadata.dcp_local_seq_lens_cpu
                if common_attn_metadata.dcp_local_seq_lens_cpu is not None
                else global_seq_lens_cpu
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            chunk_specs = split_indexer_prefill_chunks(
                indexer_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )
            context_cache: dict[
                tuple[int, int], tuple[torch.Tensor, torch.Tensor, int | torch.Tensor]
            ] = {}
            chunks = [
                self.build_one_prefill_chunk(
                    req_slice,
                    query_slice,
                    query_start_loc_cpu,
                    indexer_seq_lens_cpu,
                    global_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    skip_kv_gather=query_slice.start > 0,
                    context_cache=context_cache,
                )
                for req_slice, query_slice in chunk_specs
            ]
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks=chunks,
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            seq_lens = (
                common_attn_metadata.dcp_local_seq_lens[:num_decodes]
                if common_attn_metadata.dcp_local_seq_lens is not None
                else common_attn_metadata.seq_lens[:num_decodes]
            )
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            seq_lens_cpu_hint = (
                common_attn_metadata._seq_lens_cpu
                if common_attn_metadata._seq_lens_cpu is not None
                else common_attn_metadata.seq_lens_cpu_upper_bound
            )
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len == next_n

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                    global_seq_lens=common_attn_metadata.seq_lens[:num_decodes]
                    if common_attn_metadata.dcp_local_seq_lens is not None
                    else None,
                )
            )
            if common_attn_metadata.dcp_local_seq_lens_cpu is not None:
                max_decode_seq_len = max(
                    1,
                    int(
                        common_attn_metadata.dcp_local_seq_lens_cpu[
                            :num_decodes
                        ].max()
                    ),
                )
            else:
                max_decode_seq_len = max(1, int(common_attn_metadata.max_seq_len))
            min_topk_blocks = cdiv(self.index_topk, self.kv_cache_spec.block_size)
            max_decode_blocks = max(
                cdiv(max_decode_seq_len, self.kv_cache_spec.block_size),
                min_topk_blocks,
            )
            block_table_width_before = (
                int(block_table.shape[1]) if block_table.dim() == 2 else None
            )
            keep_full_block_table = (
                self.dcp_full_indexer_static_block_table
                and self.dcp_world_size > 1
                and block_table.dim() == 2
            )
            if (
                not keep_full_block_table
                and block_table.dim() == 2
                and block_table.shape[1] > max_decode_blocks
            ):
                block_table = block_table[:, :max_decode_blocks]
            block_table_width_after = (
                int(block_table.shape[1]) if block_table.dim() == 2 else None
            )

            if (
                _DEBUG_INDEXER_BLOCK_WIDTH
                and self._debug_indexer_block_width_count
                < _DEBUG_INDEXER_BLOCK_WIDTH_MAX
            ):
                try:
                    dcp_local_cpu = common_attn_metadata.dcp_local_seq_lens_cpu
                    dcp_local = (
                        dcp_local_cpu[:num_decodes].tolist()
                        if dcp_local_cpu is not None
                        else None
                    )
                    seq_lens_cpu_debug = (
                        seq_lens_cpu_hint[:num_decodes].tolist()
                        if seq_lens_cpu_hint is not None
                        else None
                    )
                    payload = {
                        "count": self._debug_indexer_block_width_count,
                        "builder": type(self).__name__,
                        "dcp_world_size": int(self.dcp_world_size),
                        "dcp_rank": int(self.dcp_rank),
                        "num_decodes": int(num_decodes),
                        "num_decode_tokens": int(num_decode_tokens),
                        "batch_size": int(batch_size),
                        "max_decode_len": int(max_decode_len),
                        "max_query_len": int(common_attn_metadata.max_query_len),
                        "max_seq_len": int(common_attn_metadata.max_seq_len),
                        "max_model_len": int(self.vllm_config.model_config.max_model_len),
                        "max_decode_seq_len": int(max_decode_seq_len),
                        "min_topk_blocks": int(min_topk_blocks),
                        "max_decode_blocks": int(max_decode_blocks),
                        "block_table_width_before": block_table_width_before,
                        "block_table_width_after": block_table_width_after,
                        "keep_full_block_table": bool(keep_full_block_table),
                        "seq_lens_cpu": seq_lens_cpu_debug,
                        "dcp_local_seq_lens_cpu": dcp_local,
                        "requires_padding": bool(requires_padding),
                        "use_native": bool(use_native),
                    }
                    with open(
                        _DEBUG_INDEXER_BLOCK_WIDTH_FILE,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(json.dumps(payload, sort_keys=True) + "\n")
                    logger.warning("INDEXER_BLOCK_WIDTH_DEBUG %s", payload)
                except Exception:
                    logger.exception("INDEXER_BLOCK_WIDTH_DEBUG failed")
                self._debug_indexer_block_width_count += 1

            page_table_1 = None
            cu_seqlens_q = None
            if (
                _USE_SGL_KERNEL_FAST_TOPK_TRANSFORM
                and current_platform.is_cuda()
                and block_table.dim() == 2
                and seq_lens.dim() == 1
                and batch_size <= self.page_table_1_buffer.shape[0]
            ):
                page_width = block_table.shape[1] * self.kv_cache_spec.block_size
                page_table_1 = self.page_table_1_buffer[:batch_size, :page_width]
                _pdl_kwargs = (
                    {"USE_GDC": True, "launch_pdl": True}
                    if (
                        current_platform.is_cuda()
                        and current_platform.has_device_capability(90)
                    )
                    else {}
                )
                _expand_block_table_to_page_table_1_kernel[
                    (batch_size, cdiv(page_width, 256))
                ](
                    block_table,
                    page_table_1,
                    block_table.stride(0),
                    page_table_1.stride(0),
                    block_table.shape[1],
                    page_width,
                    self.kv_cache_spec.block_size,
                    BLOCK_N=256,
                    **_pdl_kwargs,
                )
                cu_seqlens_q = self.cu_seqlens_q_buffer[: batch_size + 1]

            # DeepGEMM is the default paged-MQA scheduler; b12x exposes a
            # compatible scheduler for the active b12x sparse indexer path.
            if _use_b12x_sparse_indexer_for_config(self.vllm_config):
                try:
                    from b12x.integration.nsa_indexer import (
                        get_paged_mqa_logits_metadata as b12x_get_metadata,
                    )

                    b12x_sched_seq_lens = (
                        seq_lens.reshape(-1) if seq_lens.dim() == 2 else seq_lens
                    )
                    b12x_get_metadata(
                        b12x_sched_seq_lens.contiguous(),
                        self.kv_cache_spec.block_size,
                        self.num_sms,
                        out=self.scheduler_metadata_buffer,
                    )
                except ImportError:
                    pass
            elif current_platform.is_cuda() and has_deep_gemm():
                # DeepGEMM 2.5+sm120 hard-requires context_lens to be 2D
                # `(batch, next_n)` per `csrc/apis/attention.hpp:195`. Older
                # comments in `sparse_attn_indexer.py:688` claimed the kernel
                # accepted both shapes; that's no longer true. Unsqueeze to
                # `(batch, 1)` here — the scheduler treats each cycle as a
                # single decode step which is correct for both nomtp and the
                # spec-decode propose loop (each draft step calls the indexer
                # independently with a fresh metadata build).
                sched_seq_lens = (
                    seq_lens if seq_lens.dim() == 2 else seq_lens.unsqueeze(-1)
                )
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    sched_seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
                page_table_1=page_table_1,
                cu_seqlens_q=cu_seqlens_q,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            head_dim=128,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata
