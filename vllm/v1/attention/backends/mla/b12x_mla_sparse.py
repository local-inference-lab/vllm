# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``SparseMLASm120Backend`` (FlashInfer V32 v2). Same envelope --
``fp8_ds_mla`` KV cache (656 B/token), head_size = 576, paged block_size = 64,
V32-family models with an ``index_topk`` config (DeepSeek V3.2, GLM-5.1, Kimi
K2.5) -- but the decode/extend kernels come from b12x's unified SM120 backend
via the ``b12x.integration.mla`` front door (``sparse_mla_decode_forward`` /
``sparse_mla_extend_forward``). On SM120+ CUDA those front-door functions route
to ``b12x/attention/mla/unified_sm120`` automatically (GLM_NSA q_head_dim==576
contract). Selecting this backend also selects b12x's sparse indexer/top-k path.

Scratch philosophy (eager PLAN -> BIND -> KERNEL; no workspace/arena, ever):
b12x workspaces/arenas are sglang-only and forbidden here. We build a caller-
owned-scratch ``plan_sparse_mla_scratch`` PLAN once per mode (decode / extend),
and each forward maps a vLLM ``current_workspace_manager()`` scratch tensor into
a plain ``B12XSparseMLAScratch`` views CONTAINER via ``plan.bind(...)`` -- a pure
narrow()+view() mapping that allocates nothing and constructs no workspace. The
binding holds views (never a ``B12XAttentionWorkspace``); the unified SM120
sparse-MLA decode/extend kernels duck-type the container's
``tmp_output`` / ``tmp_lse`` / ``output_buffer`` / ``final_lse`` /
``num_chunks_ptr`` / ``set_split_chunk_config`` fields, so the binding is a
drop-in with no kernel-signature change. q-concat and the scratch are borrowed in
ONE ``get_simultaneous`` call so they never alias.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_dcp_global_index_to_local_index,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# Split-K tile width. Mirrors SparseMLASm120's _DECODE_SPLIT_TILE: the number of
# split-K chunks is ceil(topk / tile). This bounds the chunk dim of the borrowed
# mid_out/mid_lse scratch and the workspace ``max_chunks_per_row`` cap; b12x's
# wave-balanced planner picks num_splits <= this cap.
_DECODE_SPLIT_TILE = 64
_PREFILL_HEADS_PER_BLOCK = 16
_EXTEND_PREWARM_DONE: set[tuple[int | None, int, int, int, int, int, bool]] = set()


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


@triton.jit
def _compact_page_table_valid_prefix_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid_col = offs < width
    vals = tl.load(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        mask=valid_col,
        other=-1,
    )
    # B12X consumes page_table_1 as a dense prefix of length nsa_cache_seqlens.
    is_valid = valid_col & (vals >= 0)
    compact_pos = tl.cumsum(is_valid.to(tl.int32), 0) - 1
    valid_count = tl.sum(is_valid.to(tl.int32), axis=0)
    row_base = page_table_ptr + row * page_stride0
    tl.store(row_base + compact_pos * page_stride1, vals, mask=is_valid)
    tl.store(
        row_base + offs * page_stride1,
        -1,
        mask=valid_col & (offs >= valid_count),
    )
    tl.store(nsa_len_ptr + row, valid_count)


def _compact_page_table_valid_prefix(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = triton.next_power_of_2(width)
    _compact_page_table_valid_prefix_kernel[(page_table.shape[0],)](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


class B12xMLASparseBackend(AttentionBackend):
    """b12x unified sparse-MLA backend (SM120 / SM121).

    Same envelope as ``SparseMLASm120Backend`` (head 576, fp8_ds_mla, block 64,
    index_topk) but driven by b12x's unified decode/extend kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # aliases for fp8_ds_mla on this backend
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); the unified b12x decode/extend kernels dispatch
        # page_block_size == 64 natively (matches the fp8_ds_mla layout).
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope_head_dim
        # (64) = 576. The unified decode raises on any other q_head_dim.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The unified b12x kernels gate on
        # get_sm_version(device) >= 120 internally.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped (index_topk) model, same as SPARSE_MLA_SM120.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "B12X_MLA_SPARSE requires a model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # = 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA / SPARSE_MLA_SM120
            # layout; b12x's GLM_NSA decode reads the same record.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    # Per-token causal KV length; clamped to topk to form nsa_cache_seqlens.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor
    page_table_1: torch.Tensor
    nsa_cache_seqlens: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048


class B12xMLASparseMetadataBuilder(AttentionMetadataBuilder[B12xMLASparseMetadata]):
    """Builder for B12X_MLA_SPARSE attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_exact_metadata_reuse: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.req_id_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        self.page_table_1_buffer = torch.empty(
            (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
        )
        self.nsa_cache_seqlens_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.req_ids_arange = torch.arange(
            max_tokens, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )

        # Per-token causal KV length. Hot path (pure decode, one token per req)
        # stays entirely on device and uses stable preallocated buffers.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            req_id_per_token_tensor = self.req_ids_arange[:num_tokens]
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                seq_lens_for_req[:num_tokens], non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
        else:
            if cm.batch_topology is not None:
                starts = cm.batch_topology.query_start_loc_np[: cm.num_reqs + 1]
                query_lens = cm.batch_topology.query_lens_np
                req_id_per_token_np = cm.batch_topology.req_id_per_token_np
            else:
                starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
                query_lens = np.diff(starts)
                req_id_per_token_np = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )

            req_ids = np.zeros((num_tokens,), dtype=np.int32)
            if num_query_tokens:
                req_ids[:num_query_tokens] = req_id_per_token_np

            # Avoid the blocking seq_lens device->host sync. cm.seq_lens_cpu is a
            # lazy `.to("cpu")`; under --async-scheduling the runner keeps the GPU
            # tensor authoritative (_seq_lens_cpu=None), so reading it here forces a
            # full D2H copy every (MTP) decode step and serializes the pipeline that
            # async scheduling exists to overlap. The indexer that selects the
            # top-k for this same step already reads seq_lens_cpu_upper_bound; mirror
            # it. The per-token context length only feeds cache_seq_lens_per_token,
            # which forward_mqa clamps via torch.minimum(nsa, per_token_cache) and
            # the kernel masks past nsa_len, so an optimistic (>=) bound is safe.
            seq_lens_cpu_src = (
                cm.seq_lens_cpu_upper_bound
                if cm.seq_lens_cpu_upper_bound is not None
                else cm.seq_lens_cpu
            )
            seq_lens_cpu = seq_lens_cpu_src.numpy().astype(np.int32, copy=False)
            per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, q_len in enumerate(query_lens):
                if q_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(q_len)
                if cm.dcp_local_seq_lens is not None:
                    global_per_token_lens = torch.arange(
                        context_len + 1,
                        context_len + int(q_len) + 1,
                        dtype=torch.int32,
                    )
                    per_token_lens[start:end] = get_dcp_local_seq_lens(
                        global_per_token_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).numpy()
                else:
                    per_token_lens[start:end] = np.arange(
                        context_len + 1,
                        context_len + int(q_len) + 1,
                        dtype=np.int32,
                    )

            req_ids_t = torch.from_numpy(req_ids)
            per_token_lens_t = torch.from_numpy(per_token_lens)
            if req_ids_t.device.type == "cpu":
                req_ids_t = req_ids_t.pin_memory()
            if per_token_lens_t.device.type == "cpu":
                per_token_lens_t = per_token_lens_t.pin_memory()
            self.req_id_per_token_buffer[:num_tokens].copy_(
                req_ids_t, non_blocking=True
            )
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
            req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]

        cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
        cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[: cm.num_reqs]

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            seq_lens=cache_seq_lens_per_req,
            cache_seq_lens_per_req=cache_seq_lens_per_req,
            cache_seq_lens_per_token=cache_seq_lens_per_token,
            page_table_1=self.page_table_1_buffer[:num_tokens],
            nsa_cache_seqlens=self.nsa_cache_seqlens_buffer[:num_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(SparseMLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "B12X_MLA_SPARSE does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_MLA_SPARSE only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA dims (absorbed: Q post-projection is [T, H, kv_lora_rank + rope]).
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args.get("v_head_dim", 512)
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.force_contiguous_mla_bmm_input = True
        self.force_contiguous_mla_bmm_weight = True
        self.force_contiguous_mla_bmm_output = True

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, (
            "B12X_MLA_SPARSE requires sparse-MLA top-k indices "
            "(model with index_topk in its config)."
        )
        self.topk_tokens = int(self.topk_indices_buffer.shape[-1])

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )

        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64
        self._workspace_num_heads = self.num_heads * max(1, self.dcp_world_size)

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._prefill_num_heads = (
            _cdiv(self._workspace_num_heads, _PREFILL_HEADS_PER_BLOCK)
            * _PREFILL_HEADS_PER_BLOCK
        )
        if self._prefill_num_heads != self._workspace_num_heads:
            logger.info_once(
                "Padding B12X_MLA_SPARSE prefill heads from %d to %d.",
                self._workspace_num_heads,
                self._prefill_num_heads,
            )

        self.spec_decode_max_q = _env_int("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", 8)
        # The decode kernel handles independent one-token query rows. MTP
        # verification has multiple query rows per request, and later rows must
        # attend to earlier draft rows in the same verifier batch. Route those
        # batches through the extend path unless explicitly overridden.
        self.spec_extend_as_decode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "0") != "0"
        )

        # Decode query rows per request (1, plus speculative draft tokens).
        q_per_req = 1
        spec = getattr(vllm_config, "speculative_config", None)
        if spec is not None and getattr(spec, "num_speculative_tokens", None):
            q_per_req = 1 + int(spec.num_speculative_tokens)
        if self.spec_extend_as_decode:
            q_per_req = max(q_per_req, self.spec_decode_max_q)
        self._decode_max_rows = min(max_num_seqs * q_per_req, max_batched)
        if self._decode_max_rows < max_num_seqs:
            self._decode_max_rows = max_num_seqs

        self._max_batched = int(max_batched)

        # Lazily import b12x only on this opt-in path.
        from b12x.integration.mla import (
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )
        from b12x.integration.sparse_mla_scratch import (
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
        )

        self._sparse_mla_decode_forward = sparse_mla_decode_forward
        self._sparse_mla_extend_forward = sparse_mla_extend_forward

        # Eager PLAN -> BIND -> KERNEL (no b12x workspace/arena, ever). We build a
        # caller-owned-scratch PLAN once per mode; each forward maps a vLLM
        # workspace-manager scratch tensor into a plain B12XSparseMLAScratch views
        # CONTAINER via plan.bind(). The unified SM120 sparse-MLA decode/extend
        # kernels duck-type the container's tmp_output/tmp_lse/output_buffer/
        # final_lse/num_chunks_ptr/set_split_chunk_config fields, so no kernel
        # signature change is needed. The container also pre-seeds a stable,
        # width-derived split count (so the merge's num_chunks is well-defined and
        # CUDA-graph-stable; run_unified_decode pins the same value via
        # forced_num_splits below) and pre-materializes final_lse as a view (so the
        # legacy lazy torch.empty(final_lse) never fires inside a captured graph).
        def _make_plan(
            mode: str, max_q_rows: int, num_q_heads: int, max_batch: int
        ) -> Any:
            return plan_sparse_mla_scratch(
                B12XSparseMLAScratchCaps(
                    device=self.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=int(max_q_rows),
                    max_width=self.topk_tokens,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=self.q_head_dim,
                    v_head_dim=self.kv_lora_rank,
                    mode=mode,
                    max_batch=int(max_batch),
                    max_chunks_per_row=self._num_splits_cap,
                    page_size=self.block_size,
                )
            )

        self._decode_plan = _make_plan(
            "decode",
            self._decode_max_rows,
            self._workspace_num_heads,
            self._decode_max_rows,
        )
        self._extend_plan = _make_plan(
            "extend", max_batched, self._prefill_num_heads, max_num_seqs
        )
        # One caller-owned uint8 scratch tensor covers either path (the larger
        # layout); the per-mode materializer carves its views from the prefix.
        self._scratch_nbytes = max(
            int(self._decode_plan.layout.nbytes),
            int(self._extend_plan.layout.nbytes),
        )

        # Pre-touch q-concat + the attention scratch TOGETHER so the workspace
        # manager grows during warmup (before lock_workspace() runs
        # post-cudagraph-capture) and so the two always come from ONE
        # get_simultaneous call -> distinct, non-overlapping offsets. The manager
        # packs every call from offset 0, so borrowing q and the scratch the kernel
        # writes in separate calls would alias them.
        current_workspace_manager().get_simultaneous(
            ((max_batched, self.num_heads, self.q_head_dim), torch.bfloat16),
            ((self._scratch_nbytes,), torch.uint8),
        )
        self._prewarm_extend_kernels_once(max_batched)

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    def _sync_dcp_warmup(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.dcp_world_size <= 1:
            return
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_group = get_dcp_group()
            dcp_group.barrier()
        except Exception:
            return
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def _prewarm_extend_kernels_once(self, max_batched: int) -> None:
        if self.device.type != "cuda":
            return
        key = (
            self.device.index,
            self.q_head_dim,
            self.kv_lora_rank,
            self._prefill_num_heads,
            int(self.topk_tokens),
            int(self.block_size),
            bool(self.need_to_return_lse_for_decode),
        )
        if key in _EXTEND_PREWARM_DONE:
            return
        _EXTEND_PREWARM_DONE.add(key)

        rows_to_warm = (1, 2, 4, max(1, int(max_batched)))
        seen_rows: set[int] = set()
        # GLM fp8_ds_mla cache records are 656 B/token. One page is enough:
        # prewarm top-k indices all point at slot zero.
        kv_cache = torch.zeros(
            (self.block_size, 1, 656), dtype=torch.uint8, device=self.device
        )
        for rows in rows_to_warm:
            rows = int(rows)
            if rows in seen_rows:
                continue
            seen_rows.add(rows)
            q = torch.zeros(
                (rows, self._prefill_num_heads, self.q_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            selected_indices = torch.zeros(
                (rows, self.topk_tokens), dtype=torch.int32, device=self.device
            )
            cache_seqlens = torch.full(
                (1,), self.block_size, dtype=torch.int32, device=self.device
            )
            nsa_cache_seqlens = torch.ones(
                (rows,), dtype=torch.int32, device=self.device
            )
            scratch_storage = torch.empty(
                (self._scratch_nbytes,), dtype=torch.uint8, device=self.device
            )
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                )
            else:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                )
            self._sync_dcp_warmup()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives as (mqa_ql_nope[T, H, kv_lora_rank], mqa_q_pe[T, H, rope]);
        # b12x's GLM_NSA contract wants a single contiguous [T, H, 576] tensor.
        # Co-allocate the q-concat buffer and the per-call attention scratch in ONE
        # get_simultaneous call so they receive distinct, non-overlapping offsets:
        # the kernel reads q while writing the scratch (tmp_output/output), and the
        # manager packs every call from offset 0, so separate calls would alias q
        # with the scratch and corrupt the result.
        manager = current_workspace_manager()
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            num_actual_toks = ql_nope.shape[0]
            q_concat, scratch_storage = manager.get_simultaneous(
                (
                    (self._max_batched, self.num_heads, self.q_head_dim),
                    torch.bfloat16,
                ),
                ((self._scratch_nbytes,), torch.uint8),
            )
            q_all = q_concat[:num_actual_toks]
            ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            q_all = q.contiguous()
            num_actual_toks = q_all.shape[0]
            (scratch_storage,) = manager.get_simultaneous(
                ((self._scratch_nbytes,), torch.uint8),
            )

        num_actual_heads = q_all.shape[1]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Per-request topk indices -> physical cache slot ids. B12X consumes the
        # selected KV rows as a dense valid prefix of length nsa_cache_seqlens, so
        # reuse graph-stable metadata buffers and compact holes before launch.
        page_table_1 = attn_metadata.page_table_1[
            :num_actual_toks, : topk_indices.shape[1]
        ]
        nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_actual_toks]
        if self.dcp_world_size > 1:
            triton_convert_dcp_global_index_to_local_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                out=page_table_1,
                valid_counts=nsa_cache_seqlens,
            )
        else:
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
                out=page_table_1,
                valid_counts=nsa_cache_seqlens,
            )
            _compact_page_table_valid_prefix(page_table_1, nsa_cache_seqlens)
        per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
        torch.minimum(
            nsa_cache_seqlens,
            per_token_cache,
            out=nsa_cache_seqlens,
        )
        _mask_page_table_after_nsa_len(page_table_1, nsa_cache_seqlens)

        # KV cache -> flat (num_slots, 1, nbytes) uint8 (b12x requires rank-3
        # uint8; page_size tells it the per-block stride). page_table_1 are
        # physical slot ids consistent with block_size == page_size.
        kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
        kv_cache = kv_u8.reshape(-1, 1, kv_u8.shape[-1])
        if not kv_cache.is_contiguous():
            kv_cache = kv_cache.contiguous()

        use_decode_kernel = attn_metadata.max_query_len <= 1 or (
            self.spec_extend_as_decode
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and num_actual_toks <= attn_metadata.num_reqs * self.spec_decode_max_q
            and num_actual_toks <= self._decode_max_rows
        )
        if use_decode_kernel:
            cache_seqlens = (
                attn_metadata.cache_seq_lens_per_req
                if attn_metadata.max_query_len <= 1
                else attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            )
            # Eager bind: map the caller scratch into a views container (no
            # workspace) and call the kernel with binding=. forced_num_splits pins
            # a stable, width-derived split count so the merge's num_chunks matches
            # across every captured graph (the container seeds the same value).
            binding = self._decode_plan.bind(
                scratch=scratch_storage,
                q=q_all,
                selected_indices=page_table_1,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                return cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )
            out = cast(
                torch.Tensor,
                self._sparse_mla_decode_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    forced_num_splits=self._num_splits_cap,
                ),
            )
            return out, None
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). The b12x prefill
            # kernel currently requires head blocks of 16, so high-TP shards
            # with fewer local heads are padded and sliced back after launch.
            cache_seqlens = attn_metadata.cache_seq_lens_per_req
            prefill_num_heads = (
                _cdiv(num_actual_heads, _PREFILL_HEADS_PER_BLOCK)
                * _PREFILL_HEADS_PER_BLOCK
            )
            if prefill_num_heads == num_actual_heads:
                prefill_q = q_all
            else:
                prefill_q = q_all.new_zeros(
                    (num_actual_toks, prefill_num_heads, self.q_head_dim)
                )
                prefill_q[:, :num_actual_heads, :].copy_(q_all)

            # Eager bind into the extend views container (single-pass prefill;
            # no split-K, output_buffer is the only scratch the kernel writes).
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=prefill_q,
                selected_indices=page_table_1,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            lse = None
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                    ),
                )
            else:
                out = cast(
                    torch.Tensor,
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                    ),
                )
            if prefill_num_heads != num_actual_heads:
                dense_out = q_all.new_empty(
                    (num_actual_toks, num_actual_heads, self.kv_lora_rank),
                    dtype=q_all.dtype,
                )
                dense_out.copy_(out[:, :num_actual_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, :num_actual_heads].contiguous()
        return out, lse
