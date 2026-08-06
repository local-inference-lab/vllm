# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
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
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


def _align_block_table_width(num_blocks: int, block_size: int) -> int:
    """Match the model runner's backend-facing block-table row alignment."""
    if block_size > 128:
        return num_blocks
    alignment = 128 // block_size
    return cdiv(num_blocks, alignment) * alignment


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
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

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
        return [1, 64] if current_platform.is_rocm() else [64]

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


class B12xNonCompressedIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "B12X_NON_COMPRESSED_INDEXER"


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
    # Under DCP (dcp_world_size > 1) these hold this rank's local row bounds;
    # otherwise they hold the global bounds.
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False
    local_cu_seq_lens: torch.Tensor | None = None
    local_total_seq_lens: int = 0
    max_local_total_seq_lens: int = 0


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_B12X_PAGED_INDEX_TILE_BLOCK_K = 512


def _get_b12x_paged_indexer_supertile_k() -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K")
    tokens = _B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT if raw is None else int(raw)
    tokens = max(tokens, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor | None
    # Exact host-side decode max seq-len (compressed units) when derivable
    # from CPU shadows without a device sync; None => callers fall back to the
    # padded logits width.
    max_seq_len: int | None = None
    global_seq_lens: torch.Tensor | None = None
    # Live scorer window (max compressed context across the batch) in cache
    # tokens, computed host-side in build(); a metadata tensor read by the
    # captured indexer kernel, never an in-kernel reduction. None => b12x uses
    # the capacity cap.
    active_width: torch.Tensor | None = None
    # Per-flattened-row request id for the SM100 varlen paged kernel; None
    # selects the non-varlen paged path.
    indices: torch.Tensor | None = None


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


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


def _supports_varlen_paged_mqa_logits() -> bool:
    if (
        envs.VLLM_USE_B12X_SPARSE_INDEXER
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        # B12X consumes the already-flattened rank-1 seq_lens and repeated
        # block-table rows directly, so it does not need DeepGEMM's indices.
        return True
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_deep_gemm()
    )


def _uses_varlen_dspark_capacity(vllm_config: VllmConfig) -> bool:
    spec_config = vllm_config.speculative_config
    return bool(
        spec_config is not None
        and spec_config.use_dspark()
        and spec_config.dspark_capacity_verification_mode == "varlen"
        and (
            spec_config.dspark_confidence_threshold > 0.0
            or spec_config.dspark_budget_frac < 1.0
            or spec_config.dspark_sps_curve is not None
        )
    )


def _needs_varlen_decode(
    use_varlen_decode: bool,
    all_uniform_width: bool,
    max_decode_len: int,
    max_query_len: int,
) -> bool:
    """Use the compact scorer only when verification is actually ragged.

    Args:
        use_varlen_decode: Whether the active backend supports varlen decode.
        all_uniform_width: Whether every request has the same query width.
        max_decode_len: Largest decode width in the batch.
        max_query_len: Largest query width in the batch.

    Returns:
        Whether this batch requires the varlen decode path.
    """
    return (
        use_varlen_decode
        and not all_uniform_width
        and (max_decode_len > 1 or max_query_len > 1)
    )


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    # The indexer opts out of the shared reorder-threshold vote (see __init__),
    # so this is None; its own split uses self.decode_threshold.
    reorder_batch_threshold: int | None = None

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if _supports_varlen_paged_mqa_logits() and _uses_varlen_dspark_capacity(
            vllm_config
        ):
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        # The DCP sparse-indexer code is parameterized by interleave size, but
        # interleave > 1 is not yet validated end-to-end (gsm8k parity fails),
        # so fail closed here rather than silently produce wrong output.
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size > 1:
            raise NotImplementedError(
                "DCP sparse indexer currently supports only "
                f"cp_kv_cache_interleave_size=1 (got "
                f"{self.cp_kv_cache_interleave_size})."
            )
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = (
            self.vllm_config.attention_config.use_fp4_indexer_cache
        )

        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
        # NOTE: SM100 datacenter GPUs support any next_n natively via the
        # multi-atom paged MQA logits kernels (FP8 and FP4 indexer
        # caches). Outside the SM100 family the FP8
        # paged MQA logits kernel only supports next_n in (1, 2)
        # (deepgemm smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there.
        self.use_flattening = not current_platform.is_device_capability_family(
            100
        ) and next_n not in (1, 2)
        # SM100 supports the varlen paged MQA logits kernel (indices-selected,
        # next_n == 1 rows). Only compact spec-decode verification batches opt
        # into it; uniform DFlash draft proposal should keep the native path.
        self.use_varlen = (
            _supports_varlen_paged_mqa_logits()
            and _uses_varlen_dspark_capacity(self.vllm_config)
        )
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s use_varlen=%s "
            "(next_n=%d, use_fp4_indexer_cache=%s)",
            self.use_flattening,
            self.use_varlen,
            next_n,
            self.use_fp4_indexer_cache,
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
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.global_decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Per-row request ids for the SM100 varlen paged kernel.
        self.decode_indices_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        # Keep the expanded decode table layout identical to the model
        # runner's block table. The runner right-pads rows to a 128-token
        # boundary for attention backends; sizing this buffer to only the
        # logical page count can therefore produce a different row stride
        # (for example 1875 here versus 1876 in the runner at 480k,
        # block_size=64, DCP=4). Besides breaking variable-width MTP's
        # repeat_interleave, that divergent stride is passed to the paged
        # indexer kernels during flattened decode.
        max_num_blocks_per_req = _align_block_table_width(
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

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # Persistent live-active-width buffer for the b12x indexer decode
        # scorer window. Filled host-side each build() (outside cudagraph
        # capture) and read by the captured kernel at a stable address.
        self.b12x_active_width_buffer = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # DCP writes the indexer cache through rank-local pages. DeepSeek V4
        # additionally maps only the tokens retained by KV compression.
        if self.compress_ratio > 1 or self.dcp_world_size > 1:
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _dcp_localize_decode_seq_lens(
        self,
        seq_lens: torch.Tensor,
        num_decodes: int,
        seq_lens_is_buffer_view: bool,
    ) -> torch.Tensor:
        local_seq_lens = get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        if seq_lens_is_buffer_view:
            seq_lens.copy_(local_seq_lens)
            return seq_lens

        out = self.decode_seq_lens_buffer[:num_decodes]
        out.copy_(local_seq_lens)
        return out

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
        force_flatten: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        The flatten path (not use_native, max_decode_len > 1 or force_flatten)
        expands each multi-token decode request into individual single-token
        entries so the kernel always sees next_n=1. ``force_flatten`` keeps the
        varlen path on the flatten buffers even for all-single-token batches so
        captured CUDA graphs always read the same tensors at replay.

        The native path (use_native or max_decode_len == 1) preserves plain
        decode or spec-decode with 2D per-token context lengths.

        Args:
            seq_lens: Per-request context lengths.
            block_table: Per-request KV block table.
            decode_lens: Device tensor with each request's decode width.
            decode_lens_cpu: CPU mirror of ``decode_lens``.
            query_start_loc: Cumulative query offsets.
            num_decodes: Number of decode requests.
            num_decode_tokens: Total active decode tokens.
            use_native: Whether the backend accepts the native layout.
            next_n: Native query width expected by the backend.
            max_decode_len: Largest decode width in the batch.
            force_flatten: Whether to preserve the flatten-buffer address even
                for a uniform single-token batch.

        Returns:
            A tuple of prepared sequence lengths, block table, decode lengths,
            effective batch size, and whether kernel-side padding is required.

        Layout details:
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.
          ``seq_lens`` is 1D ``(batch_size,)`` for flatten/plain and 2D
          ``(B, max_decode_len)`` for native MTP.
        """
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native and (max_decode_len > 1 or force_flatten):
            assert self.decode_seq_lens_buffer.dim() == 1
            if block_table.shape[1] != self.expanded_block_table_buffer.shape[1]:
                raise ValueError(
                    "MTP block-table row width does not match its expanded buffer: "
                    f"{block_table.shape[1]} != "
                    f"{self.expanded_block_table_buffer.shape[1]}"
                )
            if (
                min_decode_len == max_decode_len
                and num_decodes * max_decode_len == num_decode_tokens
            ):
                # Uniform decode lengths with no cudagraph token padding.
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
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                # FULL graphs may pad the compact varlen token batch past the
                # final real row. B12X paged scoring cannot launch a zero-K
                # row, so point graph-only rows at one safe compressed token;
                # their slot mappings stay padded and their outputs are ignored.
                padding_seq_len = (
                    self.compress_ratio
                    if force_flatten and envs.VLLM_USE_B12X_SPARSE_INDEXER
                    else 0
                )
                self.decode_seq_lens_buffer[actual_expanded:num_decode_tokens] = (
                    padding_seq_len
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request. The expanded buffer uses the same backend-aligned
                # row width as the model runner's source table.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table,
                        decode_lens,
                        dim=0,
                        output_size=actual_expanded,
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
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def _prepare_global_decode_seq_lens(
        self,
        global_seq_lens: torch.Tensor | None,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decode_tokens: int,
        use_native: bool,
        max_decode_len: int,
    ) -> torch.Tensor | None:
        if global_seq_lens is None:
            return None
        if use_native or max_decode_len <= 1:
            return global_seq_lens

        actual_expanded = int(decode_lens_cpu.sum().item())
        if actual_expanded > 0:
            expanded_offsets = torch.repeat_interleave(
                global_seq_lens - decode_lens - query_start_loc,
                decode_lens,
                output_size=actual_expanded,
            )
            self.global_decode_seq_lens_buffer[:actual_expanded] = (
                expanded_offsets + self.arange_buffer[:actual_expanded] + 1
            )
        self.global_decode_seq_lens_buffer[actual_expanded:num_decode_tokens] = 0
        return self.global_decode_seq_lens_buffer[:num_decode_tokens]

    def _decode_topk_max_seq_len_from_cpu(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decodes: int,
        decode_lens_np: np.ndarray,
        max_decode_len: int,
        use_native: bool,
        dcp_local_seq_lens_cpu: torch.Tensor | None,
    ) -> int | None:
        """Return the exact decode max seq-len without reading CUDA tensors.

        B12X needs a host scalar for metadata/scheduling.  Computing it with
        ``seq_lens.max().item()`` synchronizes every decode step.  The model
        runner already maintains CPU shadows for normal decode; use those
        directly and mirror only the shape transforms that can affect the max.
        If async-spec has made the CPU shadow non-authoritative, return None so
        the caller can fall back to a safe graph-stable bound.
        """
        cpu_seq_lens = (
            dcp_local_seq_lens_cpu
            if dcp_local_seq_lens_cpu is not None
            else common_attn_metadata._seq_lens_cpu
        )
        if cpu_seq_lens is None:
            return None
        if cpu_seq_lens.device.type != "cpu":
            return None

        seq_lens_np = cpu_seq_lens[:num_decodes].numpy()
        if seq_lens_np.size == 0:
            return 0

        # Flattened variable-length MTP writes only real decode tokens and
        # zero-fills the tail.  Ignore padded decode rows to match that max.
        if not use_native and max_decode_len > 1:
            valid_decode_rows = decode_lens_np[:num_decodes] > 0
            if not bool(np.any(valid_decode_rows)):
                return 0
            seq_lens_np = seq_lens_np[valid_decode_rows]

        max_seq_len = int(seq_lens_np.max())
        if self.compress_ratio > 1:
            max_seq_len //= self.compress_ratio
        return max_seq_len

    def _maybe_build_b12x_schedule_metadata(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_decode_tokens: int,
        requires_padding: bool,
    ) -> torch.Tensor | None:
        if not envs.VLLM_USE_B12X_SPARSE_INDEXER or requires_padding:
            return None

        schedule_seq_lens = seq_lens
        if schedule_seq_lens.dim() == 2:
            batch_size, next_n = schedule_seq_lens.shape
            if num_decode_tokens != int(batch_size * next_n):
                return None
            schedule_seq_lens = schedule_seq_lens.reshape(-1)
        if schedule_seq_lens.dim() != 1:
            return None

        from sparkinfer.attention.nsa_indexer import (
            plan_paged_schedule as build_paged_mqa_schedule_metadata,
        )
        from sparkinfer.attention.nsa_indexer import (
            uses_paged_schedule as uses_paged_mqa_schedule,
        )

        if not uses_paged_mqa_schedule(
            q_rows=int(schedule_seq_lens.shape[0]),
            max_pages=int(block_table.shape[1]),
        ):
            return None

        return build_paged_mqa_schedule_metadata(
            schedule_seq_lens.contiguous(),
            self.kv_cache_spec.storage_block_size,
            self.num_sms,
            out=self.scheduler_metadata_buffer,
        )

    def _build_varlen_decode_indices(
        self,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        max_decode_len: int,
    ) -> torch.Tensor:
        """Per-flattened-row request id for the SM100 varlen paged kernel.

        Rows are in request-then-token order (matching the per-token expansion
        in ``_prepare_decode_tensors``); adjacent equal ids form one run. The
        result always has ``num_decode_tokens`` rows so it matches the
        (possibly cudagraph-padded) context_lens rows.
        ``decode_lens`` must be the original per-request counts, read before the
        expansion overwrites the buffer.

        Args:
            decode_lens: Device tensor with each request's decode width.
            decode_lens_cpu: CPU mirror of ``decode_lens``.
            num_decodes: Number of decode requests.
            num_decode_tokens: Number of flattened decode rows.
            max_decode_len: Largest decode width in the batch.

        Returns:
            A persistent tensor mapping each flattened row to its request id.
        """
        indices = self.decode_indices_buffer[:num_decode_tokens]
        if max_decode_len <= 1:
            # One query token per request: row r is request r, and any
            # qsl-padded rows past the last real request naturally form
            # singleton runs. Copy into the persistent buffer: captured CUDA
            # graphs bake this buffer's address, so returning arange_buffer
            # directly would leave the graph reading stale ids.
            indices.copy_(self.arange_buffer[:num_decode_tokens])
            return indices

        min_decode_len = int(decode_lens_cpu.min().item())
        if (
            min_decode_len == max_decode_len
            and num_decodes * max_decode_len == num_decode_tokens
        ):
            # Uniform with no token padding: row r belongs to request
            # r // max_decode_len. Static closed form, no device sync.
            indices.copy_(self.arange_buffer[:num_decode_tokens] // max_decode_len)
        else:
            # Variable (eager only): repeat each request id by its decode_len.
            # Pad the tail with non-merging trailing ids so masked pad rows form
            # singleton runs instead of extending the last real request's run.
            actual_expanded = int(decode_lens_cpu.sum().item())
            indices[:actual_expanded] = torch.repeat_interleave(
                self.arange_buffer[:num_decodes],
                decode_lens,
                output_size=actual_expanded,
            )
            if actual_expanded < num_decode_tokens:
                pad = num_decode_tokens - actual_expanded
                indices[actual_expanded:num_decode_tokens] = (
                    num_decodes + self.arange_buffer[:pad]
                )
        return indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        use_dcp_local_kv = self.dcp_world_size > 1 and dcp_local_seq_lens is not None
        use_varlen_decode = self.use_varlen and common_attn_metadata.max_req_tokens > 0

        # Short extends ride the decode path (default): their per-token causal
        # context is the same shape as spec-verify rows, and the boundary must
        # agree with the other DSv4 builders (sparse_swa/sparse_mla) so that
        # varlen FULL cudagraphs, which are captured all-decode, stay valid.
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                require_uniform=not (self.use_flattening or use_varlen_decode),
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1 or use_dcp_local_kv:
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size if use_dcp_local_kv else 1,
                dcp_rank=self.dcp_rank if use_dcp_local_kv else 0,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                # The b12x paged prefill streams one supertile at a time and
                # row-shares the page table, which requires single-request
                # chunks. Budget each request by the supertile K window.
                chunk_specs = []
                b12x_budget_seq_lens = torch.tensor(
                    [_get_b12x_paged_indexer_supertile_k()],
                    dtype=compressed_seq_lens_cpu.dtype,
                )
                for prefill_idx in range(num_prefills):
                    req_idx = num_decodes + prefill_idx
                    chunk_specs.extend(
                        split_indexer_prefill_chunks(
                            b12x_budget_seq_lens,
                            prefill_query_lens_cpu[prefill_idx : prefill_idx + 1],
                            self.max_prefill_buffer_size,
                            max_logits_bytes,
                            request_offset=req_idx,
                        )
                    )
            else:
                chunk_specs = split_indexer_prefill_chunks(
                    compressed_seq_lens_cpu[num_decodes:],
                    prefill_query_lens_cpu,
                    self.max_prefill_buffer_size,
                    max_logits_bytes,
                    request_offset=num_decodes,
                )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=self.dcp_rank,
                    dcp_world_size=self.dcp_world_size,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

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

            # Under DCP the per-token decode bounds must be localized AFTER the
            # per-token expansion below, not before. Expanding from a
            # request-level localized length subtracts decode offsets in local
            # space and yields too-short bounds (e.g. world=2, rank=1, global
            # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
            # the first decode token would run top-k against too short a local KV
            # range and miss valid tokens. Keep the global seq_lens here and
            # localize the expanded bounds further down.
            global_seq_lens_for_decode: torch.Tensor | None = None
            if dcp_local_seq_lens is not None:
                global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_varlen = _needs_varlen_decode(
                use_varlen_decode=use_varlen_decode,
                all_uniform_width=bool(
                    (decode_lens_cpu == max_decode_len).all().item()
                ),
                max_decode_len=max_decode_len,
                max_query_len=common_attn_metadata.max_query_len,
            )
            use_native = (
                not (self.use_flattening or use_varlen) and max_decode_len <= next_n
            )

            global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
                global_seq_lens=global_seq_lens_for_decode,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                max_decode_len=max_decode_len,
            )

            # Build the varlen per-row request ids from the original per-request
            # decode_lens, before _prepare_decode_tensors overwrites the buffer.
            decode_indices = None
            if use_varlen:
                decode_indices = self._build_varlen_decode_indices(
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    max_decode_len=max_decode_len,
                )

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
                    force_flatten=use_varlen,
                )
            )

            seq_lens_is_buffer_view = (use_native and next_n > 1) or (
                not use_native and (max_decode_len > 1 or use_varlen)
            )

            # Uncompressed DCP localizes after per-token expansion. Compressed
            # DCP must first convert global logical lengths to compressed-token
            # lengths and only then shard those retained tokens across ranks.
            if dcp_local_seq_lens is not None and self.compress_ratio == 1:
                seq_lens = self._dcp_localize_decode_seq_lens(
                    seq_lens, num_decodes, seq_lens_is_buffer_view
                )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                    if dcp_local_seq_lens is not None:
                        seq_lens.copy_(
                            get_dcp_local_seq_lens(
                                seq_lens,
                                self.dcp_world_size,
                                self.dcp_rank,
                                self.cp_kv_cache_interleave_size,
                            )
                        )
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    compressed_decode_seq_lens = seq_lens // self.compress_ratio
                    if dcp_local_seq_lens is not None:
                        compressed_decode_seq_lens = get_dcp_local_seq_lens(
                            compressed_decode_seq_lens,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        )
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        compressed_decode_seq_lens
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            active_width: torch.Tensor | None = None
            decode_topk_max_seq_len: int | None = None
            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                # Live scorer window in cache tokens. ceil(max_seq_len /
                # compress_ratio) is an upper bound on the max compressed
                # context across the batch, so windowing to it is
                # top-k-identical to the capacity cap and only skips wasted
                # k-tiles. Computed on the host here (metadata-prep, outside
                # cudagraph capture) and filled into the persistent buffer the
                # captured kernel reads.
                active_width_tokens = int(common_attn_metadata.max_seq_len)
                if self.compress_ratio > 1:
                    active_width_tokens = -(-active_width_tokens // self.compress_ratio)
                self.b12x_active_width_buffer.fill_(active_width_tokens)
                active_width = self.b12x_active_width_buffer
                if self.compress_ratio > 1:
                    # Compressed-MLA models already bound the B12X scorer with
                    # active_width. Avoiding a host reduction here removes a
                    # per-decode sync without changing the scorer window.
                    decode_topk_max_seq_len = active_width_tokens
                else:
                    # GLM/Kimi-style uncompressed indexer rows need the exact
                    # scalar for best throughput; use the CPU shadow maintained
                    # by the runner instead of synchronizing on seq_lens.
                    decode_topk_max_seq_len = self._decode_topk_max_seq_len_from_cpu(
                        common_attn_metadata,
                        num_decodes,
                        decode_lens_cpu.numpy(),
                        max_decode_len,
                        use_native,
                        (
                            common_attn_metadata.dcp_local_seq_lens_cpu[:num_decodes]
                            if dcp_local_seq_lens is not None
                            and common_attn_metadata.dcp_local_seq_lens_cpu is not None
                            else None
                        ),
                    )
                    if decode_topk_max_seq_len is None:
                        # Async-spec can make GPU seq_lens authoritative.  In
                        # that mode there is no exact host scalar without a
                        # sync, so use the same graph-stable live-window bound
                        # already consumed by the B12X scorer.
                        decode_topk_max_seq_len = active_width_tokens
                schedule_metadata = self._maybe_build_b12x_schedule_metadata(
                    seq_lens,
                    block_table,
                    num_decode_tokens,
                    requires_padding,
                )
            else:
                # DeepGEMM is required for the paged MQA logits on CUDA devices
                if current_platform.is_cuda() and has_deep_gemm():
                    self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                        seq_lens,
                        self.kv_cache_spec.storage_block_size,
                        self.num_sms,
                        indices=decode_indices,
                    )
                schedule_metadata = self.scheduler_metadata_buffer

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=schedule_metadata,
                max_seq_len=decode_topk_max_seq_len,
                indices=decode_indices,
                global_seq_lens=global_seq_lens_for_decode,
                active_width=active_width,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens
    if dcp_world_size > 1:
        # Per-rank local KV length under interleave-aware DCP sharding, shape
        # [num_reqs, dcp_world_size]. Reuse the canonical CP helper so the
        # sharding matches the rest of the DCP pipeline (decode/prefill).
        local_seq_lens = get_dcp_local_seq_lens(
            compressed_seq_lens[start_idx:end_idx],
            dcp_world_size,
            None,
            cp_kv_cache_interleave_size,
        )
        this_rank_counts = local_seq_lens[:, dcp_rank].to(torch.int32)
        local_cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        torch.cumsum(this_rank_counts, dim=0, out=local_cu_seq_lens[1:])
        local_total_seq_lens = int(local_cu_seq_lens[-1].item())
        max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item())

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    # Under DCP the kernel writes this rank's local row bounds into
    # cu_seq_len_ks/ke; otherwise local_cu_seq_lens aliases cu_seq_lens.
    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
    )


@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Row-start base for cu_seq_len_ks/ke: local cumulative lens under DCP,
    # aliases cu_compressed_seq_lens_ptr otherwise.
    row_start_cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    DCP_RANK,
    DCP_WORLD,
    DCP_INTERLEAVE,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    # Row start for the (possibly localized) cu_seq_len_ks/ke. Equals seq_start
    # when DCP is disabled (the pointer aliases cu_compressed_seq_lens_ptr).
    row_start = tl.load(row_start_cu_compressed_seq_lens_ptr + batch_idx)

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # cu_seq_len_ks: row start in the gathered K buffer.
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, row_start, mask=mask)

        # cu_seq_len_ke: row start + per-token context length. Under DCP the
        # global per-token length is sharded across ranks.
        global_ctx = start_pos + 1 + offset
        len_per_token = global_ctx // COMPRESS_RATIO
        if DCP_WORLD > 1:
            # Per-rank local context length under interleave-aware DCP, matching
            # get_dcp_local_seq_lens. K == 1 reduces to (len + world-1-rank)//world.
            base = (len_per_token // DCP_INTERLEAVE // DCP_WORLD) * DCP_INTERLEAVE
            remainder = len_per_token - base * DCP_WORLD
            remainder = tl.minimum(
                tl.maximum(remainder - DCP_RANK * DCP_INTERLEAVE, 0), DCP_INTERLEAVE
            )
            len_per_token = base + remainder
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            row_start + len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)
