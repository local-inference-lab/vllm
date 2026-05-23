# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``FlashInferMLASparseBackend`` (SM100 / datacenter Blackwell)
and ``FlashMLASparseBackend`` (Hopper). Targets the V32 (DSv3.2 family) sparse
MLA decode kernel shipped in FlashInfer (``BatchSparseMLAPagedAttentionWrapper``).

Same envelope as the FlashMLA path: ``fp8_ds_mla`` KV cache layout (656 B/token
INLINE: 512 NoPE + 16 scales + 128 RoPE), head_size = 576, paged block_size =
64. Works for V32-family models (DeepSeek V3.2, GLM-5.1, Kimi K2.5, ...).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from flashinfer import BatchSparseMLAPagedAttentionWrapper

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms.interface import DeviceCapability
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
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


class SparseMLASm120Backend(AttentionBackend):
    """SM120 FlashInfer sparse-MLA backend (BatchSparseMLAPagedAttentionWrapper).

    V32-family (DSv3.2 / GLM-5.1 / Kimi K2.5) — fp8_ds_mla KV cache, 576
    head_size, paged block_size = 64.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla on this backend (auto-converted by MLAAttention)
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); vLLM's prepare_kernel_block_sizes requires a
        # common block size across the indexer cache and the main sparse-MLA
        # cache, so we cannot offer pbs=1 here. The flashinfer V32 v2 decode
        # kernel dispatches PAGE_BLOCK_SIZE=64 natively.
        return [64]

    @staticmethod
    def get_name() -> str:
        return "SPARSE_MLA_SM120"

    @staticmethod
    def get_impl_cls() -> type["SparseMLASm120Impl"]:
        return SparseMLASm120Impl

    @staticmethod
    def get_builder_cls() -> type["SparseMLASm120MetadataBuilder"]:
        return SparseMLASm120MetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell: RTX PRO 6000 (SM120a) and SM121a variants.
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
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped model. The wrapper's dispatch table covers
        # h ∈ {8,16,32,64,128} × topk ∈ {128,512,1024,2048}; both are the
        # standard V32-family envelope, so we don't gate further here.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "SPARSE_MLA_SM120 requires a model with index_topk config"
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
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA layout.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)


@dataclass
class SparseMLASm120Metadata(AttentionMetadata):
    """Attention metadata for SPARSE_MLA_SM120 backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048


class SparseMLASm120MetadataBuilder(AttentionMetadataBuilder[SparseMLASm120Metadata]):
    """Builder for SPARSE_MLA_SM120 attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

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

        # req_id_per_token: scratch buffer for build(), sized to the max
        # batched-token bound so cudagraph capture sees stable allocations.
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SparseMLASm120Metadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )

        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]

        return SparseMLASm120Metadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            seq_lens=cm.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class SparseMLASm120Impl(SparseMLAAttentionImpl[SparseMLASm120Metadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

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
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "SPARSE_MLA_SM120 does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "SPARSE_MLA_SM120 only supports decoder self-attention"
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

        assert indexer is not None, (
            "SPARSE_MLA_SM120 requires a sparse-MLA indexer (model with "
            "index_topk in its config)."
        )
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

        # BatchSparseMLAPagedAttentionWrapper is sized by max_num_batched_tokens
        # and the per-rank head count. Construction allocates the V32 decode
        # workspace + LSE buffer once and reuses them across calls (cudagraph
        # friendly).
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._wrapper = BatchSparseMLAPagedAttentionWrapper(
            max_num_tokens=max_tokens,
            max_num_heads=num_heads,
            d_v=self.kv_lora_rank,  # latent V dim (= 512 for V32 family)
        )

        # Q is passed pre-quantized to fp8 only when the kernel asks for it;
        # V32 v2 takes BF16 Q today and quantizes inside the kernel.
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: SparseMLASm120Metadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives as (mqa_ql_nope[T, H, kv_lora_rank], mqa_q_pe[T, H, rope]);
        # the wrapper expects a single contiguous [T, H, kv_lora_rank + rope]
        # tensor (= 576 for V32 family).
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Per-request indices → global cache slots. Matches the conversion in
        # FlashMLASparseImpl / FlashInferMLASparseImpl exactly.
        topk_indices_physical = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        # kv_c_and_k_pe_cache is laid out as (num_blocks, block_size, 656)
        # uint8 per the fp8_ds_mla spec; the wrapper accepts either 3-D
        # [num_blocks, page_bytes] or 4-D [num_blocks, page_block_size, 1, bpt]
        # and extracts the block stride from .stride(0). Unsqueeze to the 4-D
        # form (singleton kv-head dim) matching the FlashMLA convention.
        kv_cache_4d = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2)

        self._wrapper.run(
            q=q,
            kv_cache=kv_cache_4d,
            indices=topk_indices_physical,
            output=output,
            sm_scale=self.scale,
        )
        return output, None
