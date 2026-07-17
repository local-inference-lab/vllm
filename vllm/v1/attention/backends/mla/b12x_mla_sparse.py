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

import inspect
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.distributed as dist

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
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
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
_HEAD_ALIGNMENT = 8
_BF16_BYTES = 2
_EXTEND_PREWARM_DONE: set[
    tuple[int | None, int, int, int, int, int, bool, str, bool]
] = set()
_FP8_ROPE_WRITER_LOADED = False
_KV_FP8_ROPE_REQUESTED = os.getenv("KV_FP8_ROPE", "0") == "1"


_IS_GLM_MOE_DSA_CACHE: bool | None = None


def _is_glm_moe_dsa_model() -> bool:
    """Return true only for GLM or its in-process MTP draft model.

    Robust to being called before the vLLM config context is established (e.g.
    during KV-cache shape resolution / cudagraph compilation in a worker, where
    get_current_vllm_config() raises): fall back to the explicit KV_FP8_ROPE
    request and re-resolve once the config becomes available. Correctness is
    preserved because the fallback is only reached when the user set
    KV_FP8_ROPE=1 for their GLM model; KV_FP8_ROPE=0 short-circuits earlier.
    """
    global _IS_GLM_MOE_DSA_CACHE
    if _IS_GLM_MOE_DSA_CACHE is not None:
        return _IS_GLM_MOE_DSA_CACHE
    from vllm.config import get_current_vllm_config

    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return _KV_FP8_ROPE_REQUESTED
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type == "glm_moe_dsa":
        _IS_GLM_MOE_DSA_CACHE = True
        return True
    speculative_config = getattr(vllm_config, "speculative_config", None)
    target_model_config = getattr(speculative_config, "target_model_config", None)
    target_model_type = (
        getattr(target_model_config.hf_config, "model_type", None)
        if target_model_config is not None
        else None
    )
    result = model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa"
    _IS_GLM_MOE_DSA_CACHE = result
    return result


def _kv_fp8_rope_enabled() -> bool:
    """Strict public gate plus literal GLM architecture selection."""
    return _KV_FP8_ROPE_REQUESTED and _is_glm_moe_dsa_model()


def _load_fp8_rope_writer() -> None:
    """Load the private 368-byte writer without modifying the stock writer."""
    global _FP8_ROPE_WRITER_LOADED
    if _FP8_ROPE_WRITER_LOADED:
        return
    library = os.getenv("KV_FP8_ROPE_WRITER_LIB", "/opt/fp8rope/_C_fp8_rope.so")
    if not os.path.isfile(library):
        raise RuntimeError(
            "KV_FP8_ROPE=1 requires the standalone writer library at "
            f"{library!r} (override with KV_FP8_ROPE_WRITER_LIB)"
        )
    torch.ops.load_library(library)
    namespace = getattr(torch.ops, "_C_fp8_rope_ops", None)
    if namespace is None or not hasattr(
        namespace, "concat_and_cache_nvfp4_mla_fp8_rope"
    ):
        raise RuntimeError(
            f"FP8-RoPE writer library {library!r} did not register the expected op"
        )
    _FP8_ROPE_WRITER_LOADED = True


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
        "nvfp4_ds_mla",
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
        if cache_dtype_str == "nvfp4_ds_mla":
            # NVFP4 MLA latent: 256 B E2M1 NoPE data + 32 B E4M3 group-16
            # scales. The stock record has 16 B pad + 128 B BF16 RoPE (432 B).
            # KV_FP8_ROPE=1 reuses the pad for one FP32 amax scale and stores
            # 64 E4M3 bytes at the original RoPE offset (368 B total).
            return (
                num_blocks,
                block_size,
                368 if _kv_fp8_rope_enabled() else 432,
            )
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    # DCP keeps global logical top-k ids until forward_mqa maps the entries
    # owned by this rank to local physical slots. These buffers are unnecessary
    # for the direct native-slot path when DCP is disabled.
    req_id_per_token: torch.Tensor | None
    page_table_1: torch.Tensor | None
    nsa_cache_seqlens: torch.Tensor | None
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    # Per-token causal KV length consumed directly by the sparse MLA kernel.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor

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
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        if self.dcp_world_size > 1:
            self.req_id_per_token_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
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
        else:
            self.req_id_per_token_buffer = None
            self.page_table_1_buffer = None
            self.nsa_cache_seqlens_buffer = None
            self.req_ids_arange = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            num_decode_tokens = num_tokens
            num_prefill_tokens = 0
        elif cm.batch_topology is not None:
            _, _, num_decode_tokens, num_prefill_tokens = (
                cm.batch_topology.split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        else:
            _, _, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                cm,
                decode_threshold=1,
                treat_short_extends_as_decodes=True,
            )
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        use_dcp = self.dcp_world_size > 1
        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if use_dcp and cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )
        req_id_per_token_tensor = None

        # Per-token causal KV length. In pure decode the common metadata already
        # has exactly the graph-stable tensor both b12x consumers need, so bind it
        # directly instead of staging two identical D2D copies.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            if use_dcp:
                assert self.req_ids_arange is not None
                req_id_per_token_tensor = self.req_ids_arange[:num_tokens]
                self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                    seq_lens_for_req[:num_tokens], non_blocking=True
                )
                self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                    seq_lens_for_req[: cm.num_reqs], non_blocking=True
                )
                cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[
                    :num_tokens
                ]
                cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[
                    : cm.num_reqs
                ]
            else:
                cache_seq_lens_per_token = seq_lens_for_req[:num_tokens]
                cache_seq_lens_per_req = seq_lens_for_req[: cm.num_reqs]
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

            req_ids = None
            if use_dcp:
                req_ids = np.zeros((num_tokens,), dtype=np.int32)
                if num_query_tokens:
                    req_ids[:num_query_tokens] = req_id_per_token_np

            if not use_dcp and cm.positions is not None and cm.positions.ndim == 1:
                # Async scheduling intentionally exposes only an optimistic CPU
                # sequence-length bound. That bound can lag when a finished slot is
                # recycled for a shorter request, so it is not a valid causal mask
                # for multi-token verification. Positions are authoritative on the
                # GPU and give the exact per-token KV length for DCP1.
                per_token_lens_t = cm.positions[:num_tokens].to(torch.int32) + 1
            else:
                # DCP needs rank-local lengths rather than global positions. Avoid
                # the blocking lazy seq_lens D2H copy and convert the scheduler's
                # conservative CPU lengths to each rank's local interleaving.
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
                    if use_dcp:
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

                per_token_lens_t = torch.from_numpy(per_token_lens)
                if per_token_lens_t.device.type == "cpu":
                    per_token_lens_t = per_token_lens_t.pin_memory()
            if req_ids is not None:
                assert self.req_id_per_token_buffer is not None
                req_ids_t = torch.from_numpy(req_ids)
                if req_ids_t.device.type == "cpu":
                    req_ids_t = req_ids_t.pin_memory()
                self.req_id_per_token_buffer[:num_tokens].copy_(
                    req_ids_t, non_blocking=True
                )
                req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
            cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
            cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[: cm.num_reqs]

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            page_table_1=(
                self.page_table_1_buffer[:num_tokens]
                if self.page_table_1_buffer is not None
                else None
            ),
            nsa_cache_seqlens=(
                self.nsa_cache_seqlens_buffer[:num_tokens]
                if self.nsa_cache_seqlens_buffer is not None
                else None
            ),
            seq_lens=cache_seq_lens_per_req,
            cache_seq_lens_per_req=cache_seq_lens_per_req,
            cache_seq_lens_per_token=cache_seq_lens_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(SparseMLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    can_return_lse_for_decode: bool = True
    supports_dcp_project_before_merge: bool = True
    supports_dcp_gather_query_in_workspace: bool = True
    supports_dcp_project_before_merge_in_workspace: bool = True
    supports_dcp_reduce_scatter_output_in_workspace: bool = True

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
        self._kv_fp8_rope = bool(
            self.kv_cache_dtype == "nvfp4_ds_mla" and _kv_fp8_rope_enabled()
        )
        if _KV_FP8_ROPE_REQUESTED and not _is_glm_moe_dsa_model():
            logger.warning(
                "KV_FP8_ROPE=1 ignored: compact MLA records are restricted to "
                "model_type=glm_moe_dsa and its associated MTP draft"
            )
        if _kv_fp8_rope_enabled() and self.kv_cache_dtype != "nvfp4_ds_mla":
            logger.warning(
                "KV_FP8_ROPE=1 has no effect for kv_cache_dtype=%s; the compact "
                "record is GLM nvfp4_ds_mla-only",
                self.kv_cache_dtype,
            )
        if self._kv_fp8_rope:
            _load_fp8_rope_writer()

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
        self.dcp_workspace_non_dbo = not bool(parallel_config.enable_dbo)
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.tp_world_size = int(parallel_config.tensor_parallel_size)
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = self.dcp_world_size > 1

        expects_physical_slots = self.dcp_world_size == 1
        if (
            indexer is not None
            and bool(indexer.output_physical_slots) != expects_physical_slots
        ):
            expected = "physical" if expects_physical_slots else "logical"
            raise RuntimeError(
                f"B12X_MLA_SPARSE requires {expected} sparse-indexer output "
                f"when dcp_world_size={self.dcp_world_size}"
            )

        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64
        # NVFP4 MLA record selection: ScaleFormat.NVFP4_E4M3 (2) rides every
        # decode/extend call so the CuTeDSL kernels specialize on the packed
        # E2M1+E4M3 record instead of the 656 B fp8_ds_mla record. KV_FP8_ROPE
        # only changes its RoPE tail; the latent format and outer-scale
        # correction are deliberately untouched.
        self._b12x_scale_format = 2 if self.kv_cache_dtype == "nvfp4_ds_mla" else None
        self._kv_record_bytes = (
            (368 if self._kv_fp8_rope else 432)
            if self.kv_cache_dtype == "nvfp4_ds_mla"
            else 656
        )
        logger.info(
            "B12X GLM MLA KV format: KV_FP8_ROPE=%d kv_gmem_stride=%d "
            "kv_cache_dtype=%s",
            int(self._kv_fp8_rope),
            self._kv_record_bytes,
            self.kv_cache_dtype,
        )
        # MLAAttention all-gathers the local query-head shard before entering a
        # DCP backend. The kernel must therefore plan for, and return, the full
        # gathered head set; the outer layer reduces/scatters it back afterward.
        self._input_num_heads = self.num_heads * self.dcp_world_size

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._kernel_num_heads = (
            _cdiv(self._input_num_heads, _HEAD_ALIGNMENT) * _HEAD_ALIGNMENT
        )
        self._pad_heads = self._kernel_num_heads != self._input_num_heads

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

        if self._b12x_scale_format is not None:
            required_kwargs = {"latent_scale", "scale_format"}
            unsupported_forwards = [
                mode
                for mode, forward in (
                    ("decode", sparse_mla_decode_forward),
                    ("extend", sparse_mla_extend_forward),
                )
                if not required_kwargs.issubset(inspect.signature(forward).parameters)
            ]
            if unsupported_forwards:
                raise RuntimeError(
                    "B12X_MLA_SPARSE with kv_cache_dtype='nvfp4_ds_mla' "
                    "requires a b12x build with NVFP4 sparse-MLA API support; "
                    "unsupported forwards: " + ", ".join(unsupported_forwards)
                )

        # Eager PLAN -> BIND -> KERNEL (no b12x workspace/arena, ever). We build a
        # caller-owned-scratch PLAN once per mode; each forward maps a vLLM
        # workspace-manager scratch tensor into a plain B12XSparseMLAScratch views
        # CONTAINER via plan.bind(). The unified SM120 sparse-MLA decode/extend
        # kernels duck-type the container's tmp_output/tmp_lse/output_buffer/
        # final_lse fields. The planner fixes the split count for each captured
        # graph and the merge specializes on that count, so no device-side control
        # scalar initialization is needed. final_lse is pre-materialized as a view
        # so the legacy lazy torch.empty(final_lse) never fires during capture.
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
            self._kernel_num_heads,
            self._decode_max_rows,
        )
        self._extend_plan = _make_plan(
            "extend", max_batched, self._kernel_num_heads, max_num_seqs
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
        workspace_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (max_batched, self._kernel_num_heads, self.q_head_dim),
                torch.bfloat16,
            )
        ]
        if self._pad_heads:
            workspace_specs.append(
                (
                    (max_batched, self._input_num_heads, self.kv_lora_rank),
                    torch.bfloat16,
                )
            )
        workspace_specs.append(((self._scratch_nbytes,), torch.uint8))
        self._workspace_specs = tuple(workspace_specs)
        self._borrow_workspaces()
        self._prewarm_extend_kernels_once(max_batched)

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write the post-RoPE key using the selected runtime cache format.

        The disabled branch delegates to the shipped implementation unchanged,
        including its stock 432-byte NVFP4 writer.  The enabled branch calls a
        separate operator so loading this overlay cannot replace or perturb the
        stock operator used by KV_FP8_ROPE=0.
        """
        if not self._kv_fp8_rope:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype != "nvfp4_ds_mla":
            raise RuntimeError(
                f"KV_FP8_ROPE writer reached a non-NVFP4 cache: {kv_cache_dtype!r}"
            )
        k_pe_flat = k_pe.squeeze(1)
        if kv_c_normed.shape[-1] != 512 or k_pe_flat.shape[-1] != 64:
            raise RuntimeError(
                "KV_FP8_ROPE is GLM MLA-only and requires latent=512, rope=64; "
                f"got {tuple(kv_c_normed.shape)} and {tuple(k_pe.shape)}"
            )
        kv_u8 = kv_cache.view(torch.uint8)
        if kv_u8.shape[-1] != 368:
            raise RuntimeError(
                "KV_FP8_ROPE expected a 368-byte cache record, got "
                f"shape={tuple(kv_u8.shape)}"
            )
        torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope(
            kv_c_normed,
            k_pe_flat,
            kv_cache,
            slot_mapping.flatten(),
            k_scale,
        )

    def _borrow_workspaces(self) -> list[torch.Tensor]:
        return current_workspace_manager().get_simultaneous(*self._workspace_specs)

    def _borrow_workspace_parts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        workspace_tensors = self._borrow_workspaces()
        expected_count = 3 if self._pad_heads else 2
        if len(workspace_tensors) != expected_count:
            raise RuntimeError(
                "B12X DCP prefill borrowed an unexpected workspace count: "
                f"{len(workspace_tensors)} != {expected_count}"
            )
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        expected_q_shape = (
            self._max_batched,
            self._kernel_num_heads,
            self.q_head_dim,
        )
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != self.device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X DCP prefill borrowed an invalid query workspace: "
                f"shape={tuple(q_workspace.shape)}, dtype={q_workspace.dtype}, "
                f"device={q_workspace.device}"
            )
        if dense_out_workspace is not None and (
            tuple(dense_out_workspace.shape)
            != (self._max_batched, self._input_num_heads, self.kv_lora_rank)
            or dense_out_workspace.dtype != torch.bfloat16
            or dense_out_workspace.device != self.device
            or not dense_out_workspace.is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid dense output")
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != self.device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid raw scratch")
        return q_workspace, dense_out_workspace, scratch_storage

    def _validate_dcp_prefill_workspace_contract(self, num_tokens: int) -> None:
        supported_topologies = {
            (4, 4),
            (6, 2),
            (6, 3),
            (6, 6),
            (8, 2),
            (8, 4),
            (8, 8),
        }
        if (
            not 1025 <= num_tokens <= self._max_batched
            or not self.dcp_workspace_non_dbo
            or (self.tp_world_size, self.dcp_world_size) not in supported_topologies
            or self.dcp_world_size <= 1
            or self.num_heads <= 0
            or self._input_num_heads != self.num_heads * self.dcp_world_size
            or self._kernel_num_heads < self._input_num_heads
            or self._kernel_num_heads % _HEAD_ALIGNMENT != 0
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
            or self.v_head_dim != 256
        ):
            raise RuntimeError(
                "The DCP prefill workspace path received an unsupported "
                "topology or geometry: "
                f"tokens={num_tokens}/{self._max_batched}, "
                f"TP/DCP={self.tp_world_size}/{self.dcp_world_size}, "
                f"local/input/kernel heads={self.num_heads}/"
                f"{self._input_num_heads}/{self._kernel_num_heads}, "
                f"pad_heads={self._pad_heads}, dimensions="
                f"{self.q_head_dim}/{self.kv_lora_rank}/{self.v_head_dim}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The DCP prefill workspace path is eager-only")

    def dcp_all_gather_query_in_workspace(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Gather local DCP query heads through the borrowed MLA workspaces."""
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if ql_nope.ndim != 3 or q_pe.ndim != 3:
                raise ValueError("DCP workspace tuple queries must be rank-3")
            num_tokens, local_heads, nope_dim = ql_nope.shape
            if tuple(q_pe.shape) != (num_tokens, local_heads, 64) or nope_dim != 512:
                raise ValueError("DCP workspace requires noPE/RoPE dimensions 512/64")
            if ql_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                raise TypeError("DCP workspace queries must be BF16")
            tuple_input = True
            query_device = ql_nope.device
        else:
            if q.ndim != 3:
                raise ValueError("DCP workspace query must be rank-3")
            num_tokens, local_heads, head_dim = q.shape
            if head_dim != self.q_head_dim or not q.is_contiguous():
                raise ValueError("DCP workspace tensor query has invalid layout")
            tuple_input = False
            query_device = q.device

        self._validate_dcp_prefill_workspace_contract(int(num_tokens))
        if local_heads != self.num_heads or query_device != self.device:
            raise ValueError("DCP workspace query does not match the MLA plan")

        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        process_group = dcp_group.device_group
        if (
            dcp_group.world_size != self.dcp_world_size
            or dcp_group.rank_in_group != self.dcp_rank
        ):
            raise RuntimeError("DCP workspace group does not match the MLA plan")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError("DCP query and scratch workspaces overlap")

        world_size = self.dcp_world_size
        head_dim = self.q_head_dim
        bytes_per_chunk_row = world_size * local_heads * head_dim * _BF16_BYTES
        chunk_capacity = scratch_storage.numel() // bytes_per_chunk_row
        if chunk_capacity <= 0:
            raise RuntimeError("DCP scratch cannot hold one gathered query row")

        q_workspace_flat = q_workspace.view(-1)
        chunk_start = 0
        while chunk_start < num_tokens:
            chunk_rows = min(chunk_capacity, num_tokens - chunk_start)
            if tuple_input:
                local_offset = chunk_start * self._kernel_num_heads * head_dim
                local_numel = chunk_rows * local_heads * head_dim
                local_chunk = q_workspace_flat.narrow(
                    0, local_offset, local_numel
                ).view(chunk_rows, local_heads, head_dim)
                ops.concat_mla_q(
                    ql_nope.narrow(0, chunk_start, chunk_rows),
                    q_pe.narrow(0, chunk_start, chunk_rows),
                    local_chunk,
                )
            else:
                local_chunk = cast(torch.Tensor, q).narrow(0, chunk_start, chunk_rows)

            gather_numel = world_size * chunk_rows * local_heads * head_dim
            gathered = (
                scratch_storage.narrow(0, 0, gather_numel * _BF16_BYTES)
                .view(torch.bfloat16)
                .view(world_size * chunk_rows, local_heads, head_dim)
            )
            dist.all_gather_into_tensor(
                gathered,
                local_chunk,
                group=process_group,
                async_op=False,
            )
            rank_major = gathered.view(world_size, chunk_rows, local_heads, head_dim)
            destination = q_workspace.narrow(0, chunk_start, chunk_rows)
            for source_rank in range(world_size):
                destination.narrow(1, source_rank * local_heads, local_heads).copy_(
                    rank_major[source_rank]
                )
            chunk_start += chunk_rows

        global_query = q_workspace[:num_tokens, : self._input_num_heads]
        expected_stride = (self._kernel_num_heads * head_dim, head_dim, 1)
        if (
            tuple(global_query.shape) != (num_tokens, self._input_num_heads, head_dim)
            or tuple(global_query.stride()) != expected_stride
        ):
            raise RuntimeError("DCP workspace produced an invalid query view")
        return global_query

    def dcp_project_before_merge_in_workspace(
        self,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Project DCP partials from 512 to 256 in borrowed MLA storage."""
        num_tokens = int(attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        if (
            tuple(attn_out.shape)
            != (num_tokens, self._input_num_heads, self.kv_lora_rank)
            or not attn_out.is_contiguous()
            or attn_out.dtype != torch.bfloat16
            or tuple(w_uv.shape)
            != (self._input_num_heads, self.kv_lora_rank, self.v_head_dim)
            or not w_uv.is_contiguous()
            or w_uv.dtype != torch.bfloat16
            or tuple(lse.shape) != (num_tokens, self._input_num_heads)
            or lse.dtype != torch.float32
        ):
            raise ValueError(
                "DCP workspace projection received an invalid tensor layout"
            )

        q_workspace, dense_out_workspace, scratch_storage = (
            self._borrow_workspace_parts()
        )
        input_numel = self._input_num_heads * num_tokens * self.kv_lora_rank
        projected_numel = self._input_num_heads * num_tokens * self.v_head_dim
        projected_nbytes = projected_numel * _BF16_BYTES
        if (
            q_workspace.numel() < input_numel
            or scratch_storage.numel() < projected_nbytes
        ):
            raise RuntimeError("DCP projection workspace is too small")
        expected_attn_storage = (
            dense_out_workspace if self._pad_heads else scratch_storage
        )
        assert expected_attn_storage is not None
        if attn_out.untyped_storage().data_ptr() != (
            expected_attn_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "DCP attention output is not backed by the expected MLA workspace"
            )

        projection_input = q_workspace.view(-1)[:input_numel].view(
            self._input_num_heads, num_tokens, self.kv_lora_rank
        )
        projected_head_major = (
            scratch_storage[:projected_nbytes]
            .view(torch.bfloat16)
            .view(self._input_num_heads, num_tokens, self.v_head_dim)
        )
        projection_input.copy_(attn_out.transpose(0, 1))
        torch.bmm(projection_input, w_uv, out=projected_head_major)
        return projected_head_major.transpose(0, 1)

    def dcp_reduce_scatter_output_in_workspace(
        self,
        corrected_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the dead query prefix as DCP reduce-scatter output."""
        num_tokens = int(corrected_attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        input_head_major = corrected_attn_out.movedim(0, 1)
        if (
            tuple(corrected_attn_out.shape)
            != (num_tokens, self._input_num_heads, self.v_head_dim)
            or corrected_attn_out.dtype != torch.bfloat16
            or not input_head_major.is_contiguous()
        ):
            raise ValueError("DCP reduce-scatter input has an invalid layout")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        if (
            corrected_attn_out.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "DCP corrected input is not backed by the MLA scratch workspace"
            )
        output_numel = self.num_heads * num_tokens * self.v_head_dim
        output_head_major = q_workspace.view(-1)[:output_numel].view(
            self.num_heads, num_tokens, self.v_head_dim
        )
        output = output_head_major.transpose(0, 1)
        if not output_head_major.is_contiguous():
            raise RuntimeError("DCP reduce-scatter output is not contiguous")
        return output

    def _sync_warmup(self) -> None:
        if self.device.type == "cuda":
            torch.accelerator.synchronize(self.device)
        if self.dcp_world_size <= 1:
            return
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            get_dcp_group().barrier()
        except Exception:
            return
        finally:
            if self.device.type == "cuda":
                torch.accelerator.synchronize(self.device)

    def _b12x_kernel_format_kwargs(self, latent_scale: float = 1.0) -> dict[str, Any]:
        if self._b12x_scale_format is None:
            return {}
        return {
            "latent_scale": float(latent_scale),
            "scale_format": self._b12x_scale_format,
        }

    def _prewarm_extend_kernels_once(self, max_batched: int) -> None:
        if self.device.type != "cuda":
            return
        key = (
            self.device.index,
            self.q_head_dim,
            self.kv_lora_rank,
            self._kernel_num_heads,
            int(self.topk_tokens),
            int(self.block_size),
            bool(self.need_to_return_lse_for_decode),
            self.kv_cache_dtype,
            bool(self._kv_fp8_rope),
        )
        if key in _EXTEND_PREWARM_DONE:
            return
        _EXTEND_PREWARM_DONE.add(key)
        kernel_format_kwargs = self._b12x_kernel_format_kwargs()

        rows_to_warm = (1, 2, 4, max(1, int(max_batched)))
        seen_rows: set[int] = set()
        # GLM fp8_ds_mla cache records are 656 B/token; the real KV cache is
        # laid out (num_blocks, block_size, 656) (see the allocator at the
        # block-shape branch above), so a page's stride(0) = block_size*656.
        # The prewarm dummy must match that layout -- (1, block_size, 656) --
        # so _cache_block_stride_bytes sees stride >= page_size*656. The prior
        # (block_size, 1, 656) shape put block_size in dim 0, giving stride(0)
        # = 656 < page_size*656, which tripped the SM120 stride assertion
        # whenever this prewarm ran (i.e. spec + cudagraphs, the first config
        # to reach here; verifier-only and eager-snap both skipped it).
        # One page is enough: prewarm top-k indices all point at slot zero.
        kv_cache = torch.zeros(
            (1, self.block_size, self._kv_record_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        for rows in rows_to_warm:
            rows = int(rows)
            if rows in seen_rows:
                continue
            seen_rows.add(rows)
            q = torch.zeros(
                (rows, self._kernel_num_heads, self.q_head_dim),
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
                    **kernel_format_kwargs,
                )
            else:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    **kernel_format_kwargs,
                )
            self._sync_warmup()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Stored by MultiHeadLatentAttentionWrapper as a host float. Avoid
        # reading device state or allocating per-call CUDA state; the CuTe
        # launch receives this as a runtime scalar.
        latent_scale = float(getattr(layer, "_nvfp4_mla_outer_scale", 1.0))
        kernel_format_kwargs = self._b12x_kernel_format_kwargs(latent_scale)
        # q arrives as (mqa_ql_nope[T, H, kv_lora_rank], mqa_q_pe[T, H, rope]);
        # b12x's GLM_NSA contract wants a single contiguous [T, H, 576] tensor.
        # Co-allocate the q-concat buffer and the per-call attention scratch in ONE
        # get_simultaneous call so they receive distinct, non-overlapping offsets:
        # the kernel reads q while writing the scratch (tmp_output/output), and the
        # manager packs every call from offset 0, so separate calls would alias q
        # with the scratch and corrupt the result.
        workspace_tensors = self._borrow_workspaces()
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            num_actual_toks = ql_nope.shape[0]
            num_input_heads = ql_nope.shape[1]
            if num_input_heads != self._input_num_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {self._input_num_heads}."
                )
            q_buffer = q_workspace[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            num_actual_toks = q.shape[0]
            num_input_heads = q.shape[1]
            if num_input_heads != self._input_num_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {self._input_num_heads}."
                )
            q_buffer = q_workspace[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr() == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q.contiguous())

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
        if self.dcp_world_size > 1:
            # The indexer globally merges logical top-k ids across DCP ranks.
            # Compact just this rank's winners into local physical cache slots;
            # the outer MLA layer combines the rank-local outputs using LSE.
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.page_table_1 is not None
            assert attn_metadata.nsa_cache_seqlens is not None
            selected_indices = attn_metadata.page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_actual_toks]
            # Zero-copy: the kernel scatters directly into the persistent
            # CUDA-graph-stable views consumed by the b12x planned kernels.
            triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                out=selected_indices,
                valid_counts=nsa_cache_seqlens,
            )
            torch.minimum(
                nsa_cache_seqlens,
                per_token_cache,
                out=nsa_cache_seqlens,
            )
            _mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)
        else:
            # Without DCP, the b12x indexer writes flat physical cache slots
            # directly into the shared top-k buffer.
            selected_indices = topk_indices
            nsa_cache_seqlens = per_token_cache

        # KV cache -> paged rank-3 uint8. B12X unified SM120 kernels consume
        # flat slot ids in selected_indices, but compute raw byte offsets as:
        #   block = slot // page_size, local = slot % page_size
        # so the cache tensor itself must expose a per-block stride of
        # block_size * record_bytes. The older split path used a token-flat
        # (num_slots, 1, bytes) view; that makes stride(0) one record and breaks
        # the unified block-stride contract.
        kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
        if kv_u8.ndim == 3 and kv_u8.shape[1] == self.block_size:
            kv_cache = kv_u8
        elif kv_u8.ndim == 3 and kv_u8.shape[1] == 1:
            if kv_u8.shape[0] % self.block_size != 0:
                raise ValueError(
                    "B12X_MLA_SPARSE flat KV cache rows must be divisible by "
                    f"block_size={self.block_size}; got {kv_u8.shape[0]}"
                )
            kv_cache = kv_u8.reshape(-1, self.block_size, kv_u8.shape[-1])
        else:
            raise ValueError(
                f"B12X_MLA_SPARSE expected {self.kv_cache_dtype} KV cache as "
                f"(blocks,{self.block_size},bytes) or (slots,1,bytes), got "
                f"{tuple(kv_u8.shape)}"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "B12X_MLA_SPARSE requires a contiguous native paged KV cache; "
                f"got stride={tuple(kv_cache.stride())}"
            )

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
            decode_q = q_all
            if self._pad_heads:
                decode_q = q_buffer[:, : self._kernel_num_heads]
                decode_q[:, self._input_num_heads :, :].zero_()
            # Eager bind maps caller-owned scratch into views. forced_num_splits
            # pins the planner choice for this captured graph; the merge kernel is
            # specialized on that count and needs no device-side control fill.
            binding = self._decode_plan.bind(
                scratch=scratch_storage,
                q=decode_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                        **kernel_format_kwargs,
                    ),
                )
                if self._pad_heads:
                    assert dense_out_workspace is not None
                    dense_out = dense_out_workspace[:num_actual_toks]
                    dense_out.copy_(out[:, : self._input_num_heads, :])
                    out = dense_out
                    lse = lse[:, : self._input_num_heads]
                return out, lse
            out = cast(
                torch.Tensor,
                self._sparse_mla_decode_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    forced_num_splits=self._num_splits_cap,
                    **kernel_format_kwargs,
                ),
            )
            if self._pad_heads:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
            return out, None
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). b12x supports 8-head
            # granularity, so only a non-aligned local tail is padded here.
            cache_seqlens = attn_metadata.cache_seq_lens_per_req
            prefill_q = q_all
            if self._pad_heads:
                prefill_q = q_buffer[:, : self._kernel_num_heads]
                prefill_q[:, self._input_num_heads :, :].zero_()

            # Eager bind into the extend views container (single-pass prefill;
            # no split-K, output_buffer is the only scratch the kernel writes).
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=prefill_q,
                selected_indices=selected_indices,
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
                        **kernel_format_kwargs,
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
                        **kernel_format_kwargs,
                    ),
                )
            if self._pad_heads:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, : self._input_num_heads]
        return out, lse
