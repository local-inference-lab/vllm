# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X MLA sparse attention backend.

This backend adapts vLLM's existing DeepSeek/GLM sparse-indexer output to
the B12X sparse MLA runtime contract. The vLLM indexer produces per-query
logical token offsets; the helper below resolves them through vLLM's block
table into physical KV-cache rows, which is the `page_table_1` format consumed
by B12X.
"""

from dataclasses import dataclass
import inspect
import os
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.distributed.parallel_state import get_dcp_group
from vllm.triton_utils import tl, triton


from vllm import _custom_ops as ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.b12x_contract import (
    B12X_KV_CACHE_DTYPES,
)
from vllm.platforms import current_platform
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
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
_B12X_SYNC_DEBUG = os.getenv("VLLM_B12X_SYNC_DEBUG", "0") != "0"
_B12X_DECODE_ACCEPTS_METADATA: bool | None = None
_B12X_DECODE_ACCEPTS_RETURN_LSE: bool | None = None
_B12X_EXTEND_ACCEPTS_METADATA: bool | None = None
_B12X_EXTEND_ACCEPTS_RETURN_LSE: bool | None = None


def _b12x_sparse_mla_signature_flags(fn, *, mode: str) -> tuple[bool, bool]:
    global _B12X_DECODE_ACCEPTS_METADATA
    global _B12X_DECODE_ACCEPTS_RETURN_LSE
    global _B12X_EXTEND_ACCEPTS_METADATA
    global _B12X_EXTEND_ACCEPTS_RETURN_LSE

    if mode == "decode":
        if _B12X_DECODE_ACCEPTS_METADATA is not None:
            return (
                _B12X_DECODE_ACCEPTS_METADATA,
                bool(_B12X_DECODE_ACCEPTS_RETURN_LSE),
            )
    elif _B12X_EXTEND_ACCEPTS_METADATA is not None:
        return (
            _B12X_EXTEND_ACCEPTS_METADATA,
            bool(_B12X_EXTEND_ACCEPTS_RETURN_LSE),
        )

    params = inspect.signature(fn).parameters
    accepts_metadata = "metadata" in params
    accepts_return_lse = "return_lse" in params
    if mode == "decode":
        _B12X_DECODE_ACCEPTS_METADATA = accepts_metadata
        _B12X_DECODE_ACCEPTS_RETURN_LSE = accepts_return_lse
    else:
        _B12X_EXTEND_ACCEPTS_METADATA = accepts_metadata
        _B12X_EXTEND_ACCEPTS_RETURN_LSE = accepts_return_lse
    return accepts_metadata, accepts_return_lse


def _sparse_mla_decode_forward_vllm_metadata(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    workspace,
    sm_scale: float,
    v_head_dim: int,
):
    from b12x.integration.mla import sparse_mla_decode_forward

    accepts_metadata, _ = _b12x_sparse_mla_signature_flags(
        sparse_mla_decode_forward,
        mode="decode",
    )
    if accepts_metadata:
        return sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )

    return sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=metadata.page_table_1,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def _sparse_mla_decode_forward_with_lse_vllm_metadata(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    workspace,
    sm_scale: float,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from b12x.integration.mla import sparse_mla_decode_forward

    accepts_metadata, accepts_return_lse = _b12x_sparse_mla_signature_flags(
        sparse_mla_decode_forward,
        mode="decode",
    )
    if not accepts_return_lse:
        raise RuntimeError("Installed b12x sparse MLA decode does not support LSE")
    if accepts_metadata:
        return sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
            return_lse=True,
            lse_scale="natural",
        )

    return sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=metadata.page_table_1,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        return_lse=True,
        lse_scale="natural",
    )


def _sparse_mla_extend_forward_with_lse_vllm_metadata(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    workspace,
    sm_scale: float,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from b12x.integration.mla import sparse_mla_extend_forward

    accepts_metadata, accepts_return_lse = _b12x_sparse_mla_signature_flags(
        sparse_mla_extend_forward,
        mode="extend",
    )
    if not accepts_return_lse:
        raise RuntimeError("Installed b12x sparse MLA extend does not support LSE")
    if accepts_metadata:
        return sparse_mla_extend_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
            return_lse=True,
            lse_scale="natural",
        )

    return sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_token_offsets=metadata.selected_token_offsets,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        return_lse=True,
        lse_scale="natural",
    )


def _sparse_mla_extend_forward_vllm_metadata(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    workspace,
    sm_scale: float,
    v_head_dim: int,
):
    from b12x.integration.mla import sparse_mla_extend_forward

    accepts_metadata, _ = _b12x_sparse_mla_signature_flags(
        sparse_mla_extend_forward,
        mode="extend",
    )
    if accepts_metadata:
        return sparse_mla_extend_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )

    return sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_token_offsets=metadata.selected_token_offsets,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def _b12x_sync_debug(stage: str) -> None:
    if not _B12X_SYNC_DEBUG:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.warning("B12X_SYNC_DEBUG passed: %s", stage)


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
    # DCP can leave invalid entries interspersed with valid local KV offsets, so
    # compact valid entries to the row prefix before handing metadata to B12X.
    is_valid = valid_col & (vals >= 0)
    compact_pos = tl.cumsum(is_valid.to(tl.int32), 0) - 1
    valid_count = tl.sum(is_valid.to(tl.int32), axis=0)
    row_base = page_table_ptr + row * page_stride0
    # Do not clear the full row here. The compact writes below target the row
    # prefix, so a full-row clear races with them inside the same Triton program
    # and can leave -1 values inside the prefix while nsa_len stays high.
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


B12X_WORKSPACES: dict[tuple[object, ...], object] = {}
B12X_ARENAS: dict[tuple[object, ...], object] = {}


def clear_b12x_mla_workspace_cache(*, clear_arenas: bool = True) -> None:
    B12X_WORKSPACES.clear()
    if clear_arenas:
        B12X_ARENAS.clear()


def _get_b12x_covering_arena(
    *,
    device_type: str,
    device_index: int | None,
    mode: str,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    head_size: int,
    kv_lora_rank: int,
    indexer_num_q_heads: int,
    topk: int,
    decode_max_total_q: int,
    caps_extend_max_total_q: int,
    caps_extend_max_batch: int,
    caps_extend_max_kv_rows: int,
    reserve_extend_indexer_logits: bool,
    extend_indexer_tile_logits_k_rows: int,
    use_joint_arena: bool,
    moe_backend: str,
):
    for key, arena in B12X_ARENAS.items():
        if len(key) != 18:
            continue
        (
            key_device_type,
            key_device_index,
            key_mode,
            key_dtype,
            key_kv_dtype,
            key_num_q_heads,
            key_head_size,
            key_kv_lora_rank,
            key_indexer_num_q_heads,
            key_topk,
            key_decode_max_total_q,
            key_caps_extend_max_total_q,
            key_caps_extend_max_batch,
            key_caps_extend_max_kv_rows,
            key_reserve_extend_indexer_logits,
            key_extend_indexer_tile_logits_k_rows,
            key_use_joint_arena,
            key_moe_backend,
        ) = key
        if (
            key_device_type != device_type
            or key_device_index != device_index
            or key_mode != mode
            or key_dtype != dtype
            or key_kv_dtype != kv_dtype
            or key_head_size != head_size
            or key_kv_lora_rank != kv_lora_rank
            or key_indexer_num_q_heads < indexer_num_q_heads
            or key_topk != topk
            or key_decode_max_total_q < decode_max_total_q
            or key_caps_extend_max_total_q < caps_extend_max_total_q
            or key_caps_extend_max_batch < caps_extend_max_batch
            or key_caps_extend_max_kv_rows < caps_extend_max_kv_rows
            or key_use_joint_arena != use_joint_arena
            or key_moe_backend != moe_backend
        ):
            continue
        if key_num_q_heads < num_q_heads:
            continue
        if reserve_extend_indexer_logits and not key_reserve_extend_indexer_logits:
            continue
        if (
            not reserve_extend_indexer_logits
            and key_extend_indexer_tile_logits_k_rows
            < extend_indexer_tile_logits_k_rows
        ):
            continue
        return arena
    return None


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


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Ignoring negative %s=%r; using %d", name, value, default)
        return default
    return parsed


def _default_decode_q_per_req(
    num_speculative_tokens: int,
    *,
    spec_extend_as_decode: bool,
    spec_decode_max_q: int,
) -> int:
    return max(
        1,
        1 + int(num_speculative_tokens),
        int(spec_decode_max_q) if spec_extend_as_decode else 1,
    )


def _prime_b12x_sm_scale(workspace: object, device: torch.device, scale: float) -> None:
    """Initialize b12x sm_scale before CUDA graph capture.

    b12x lazily materializes a device tensor from the Python float scale. Doing
    that first write during graph capture is illegal, so vLLM primes it when the
    workspace is created.
    """
    sm_scale = float(scale)
    sm_scale_tensor = getattr(workspace, "sm_scale_tensor", None)
    if (
        sm_scale_tensor is None
        or sm_scale_tensor.device != device
        or sm_scale_tensor.dtype != torch.float32
    ):
        sm_scale_tensor = torch.empty((1,), dtype=torch.float32, device=device)
        setattr(workspace, "sm_scale_tensor", sm_scale_tensor)
    sm_scale_tensor.fill_(sm_scale)
    setattr(workspace, "sm_scale_value", sm_scale)


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

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
        del head_size, dtype, block_size, has_sink
        if not use_mla or not use_sparse:
            return "B12X sparse MLA requires an MLA sparse model"
        if device_capability.major != 12:
            return f"B12X sparse MLA requires SM120, got {device_capability}"
        if kv_cache_dtype not in B12X_KV_CACHE_DTYPES:
            return "B12X sparse MLA requires fp8/fp8_e4m3/fp8_ds_mla KV cache"

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_config, "index_topk"):
                return "B12X sparse MLA requires index_topk in model config"
            if getattr(hf_config, "kv_lora_rank", None) != 512:
                return "B12X sparse MLA currently requires kv_lora_rank=512"
            if getattr(hf_config, "qk_rope_head_dim", None) != 64:
                return "B12X sparse MLA currently requires qk_rope_head_dim=64"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, head_size
        if cache_dtype_str in ("fp8", "fp8_ds_mla"):
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, 576)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    req_id_per_token: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    cache_seq_lens_per_token: torch.Tensor
    block_table: torch.Tensor
    page_table_1: torch.Tensor
    nsa_cache_seqlens: torch.Tensor
    nsa_cu_seqlens: torch.Tensor
    nsa_cu_seqlens_k: torch.Tensor
    block_size: int
    topk_tokens: int


class B12xMLASparseMetadataBuilder(
    AttentionMetadataBuilder[B12xMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        del vllm_config, kv_cache_spec
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        del layer_names
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
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
        self.req_ids_arange = torch.arange(max_tokens, dtype=torch.int32, device=device)
        self.nsa_cu_seqlens_buffer = torch.arange(
            max_tokens + 1, dtype=torch.int32, device=device
        )
        self.nsa_cu_seqlens_k_buffer = torch.empty(
            (max_tokens + 1,), dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens

        # Pure decode is the hot path. Avoid `seq_lens_cpu`, which can force a
        # device-to-host sync, and reuse fixed device buffers so CUDA graph
        # capture sees stable tensor addresses.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            seq_lens = (
                cm.dcp_local_seq_lens
                if cm.dcp_local_seq_lens is not None
                else cm.seq_lens
            )
            self.req_id_per_token_buffer[:num_tokens].copy_(
                self.req_ids_arange[:num_tokens]
            )
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                seq_lens[:num_tokens], non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens[: cm.num_reqs], non_blocking=True
            )
        else:
            starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )
            req_ids = np.zeros((num_tokens,), dtype=np.int32)
            if num_query_tokens:
                req_ids[:num_query_tokens] = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )

            seq_lens_for_req = (
                cm.dcp_local_seq_lens
                if cm.dcp_local_seq_lens is not None
                else cm.seq_lens
            )
            seq_lens_cpu = cm.seq_lens_cpu.numpy().astype(np.int32, copy=False)
            # `cm.num_actual_tokens` can be CUDA-graph padded. query_start_loc
            # still describes only real query rows, so fill padding rows with
            # zero-length metadata instead of trying to copy a shorter req-id
            # vector into the padded buffer.
            per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, q_len in enumerate(query_lens):
                if q_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(q_len)
                global_per_token_lens = torch.arange(
                    context_len + 1, context_len + int(q_len) + 1, dtype=torch.int32
                )
                if cm.dcp_local_seq_lens is not None:
                    per_token_lens[start:end] = get_dcp_local_seq_lens(
                        global_per_token_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).numpy()
                else:
                    per_token_lens[start:end] = global_per_token_lens.numpy()

            # Pin the host source tensors so non_blocking=True copies are
            # legal during CUDA graph capture. Without `.pin_memory()`,
            # capturing a forward pass through this branch (max_query_len>1
            # path, e.g. MTP=3 decode where max_query_len=4) raises:
            #   "Cannot copy between CPU and CUDA tensors during CUDA graph
            #    capture unless the CPU tensor is pinned."
            # The buffers themselves are CUDA (allocated at line 245-247
            # with device=device), so the copy_ direction is CPU→CUDA;
            # pinning the source side satisfies the capture-time constraint.
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

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            cache_seq_lens_per_req=self.cache_seq_lens_per_req_buffer[: cm.num_reqs],
            cache_seq_lens_per_token=self.cache_seq_lens_per_token_buffer[:num_tokens],
            block_table=cm.block_table_tensor,
            page_table_1=self.page_table_1_buffer[:num_tokens],
            nsa_cache_seqlens=self.nsa_cache_seqlens_buffer[:num_tokens],
            nsa_cu_seqlens=self.nsa_cu_seqlens_buffer[: num_tokens + 1],
            nsa_cu_seqlens_k=self.nsa_cu_seqlens_k_buffer[: num_tokens + 1],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(SparseMLAAttentionImpl[B12xMLASparseMetadata]):
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
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        del topk_indice_buffer, kv_sharing_target_layer_name
        unsupported = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported):
            raise NotImplementedError(
                "B12xMLASparseImpl does not support alibi_slopes, "
                "sliding_window, or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("B12xMLASparseImpl only supports decoder MLA")
        if kv_cache_dtype not in B12X_KV_CACHE_DTYPES:
            raise NotImplementedError(
                "B12X sparse MLA requires fp8/fp8_e4m3/fp8_ds_mla KV cache"
            )
        assert indexer is not None, "Indexer required for sparse MLA"

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_text_config
        self.hf_config = hf_config
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_total_q = max_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dcp_world_size = max(
            1, int(vllm_config.parallel_config.decode_context_parallel_size)
        )
        self.indexer_num_q_heads = int(getattr(hf_config, "index_n_heads", 32))
        self.arena_max_running_requests = _env_int(
            "VLLM_B12X_MLA_ARENA_MAX_RUNNING_REQUESTS", self.max_num_seqs
        )
        self.arena_extend_max_total_q = _env_int(
            "VLLM_B12X_MLA_ARENA_EXTEND_MAX_TOTAL_Q", self.max_total_q
        )
        self.arena_extend_max_batch = _env_int(
            "VLLM_B12X_MLA_ARENA_EXTEND_MAX_BATCH",
            min(self.max_num_seqs, self.arena_max_running_requests),
        )
        self.arena_extend_max_kv_rows = _env_int(
            "VLLM_B12X_MLA_ARENA_EXTEND_MAX_KV_ROWS", self.max_model_len
        )
        moe_backend = self._moe_backend_name(vllm_config)
        self.use_arena = True
        self.use_joint_arena = moe_backend == "b12x"
        self.decode_topk_is_physical = (
            os.getenv("VLLM_B12X_MLA_RAW_DECODE_TOPK", "0") != "0"
            or os.getenv("VLLM_B12X_MLA_DECODE_TOPK_PHYSICAL", "0") != "0"
        )
        self.decode_fast_nsa_seqlens = (
            os.getenv("VLLM_B12X_MLA_FAST_NSA_SEQLENS", "0") != "0"
        )
        self.spec_decode_max_q = _env_int("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", 8)
        # MTP verify rows are short extend batches. Forcing them through the
        # B12X sparse MLA decode path can corrupt FULL CUDA graph capture.
        self.spec_extend_as_decode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "0") != "0"
        )
        self.dcp_topk_per_rank = _env_int("VLLM_B12X_MLA_DCP_TOPK_PER_RANK", 0)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        num_speculative_tokens = int(
            getattr(speculative_config, "num_speculative_tokens", 0) or 0
        )
        # MTP/speculative verification sends target + draft rows through the
        # decode kernel. Short speculative extend batches can also use the same
        # decode path, so size the graph-stable workspace for the largest
        # per-request query length that path accepts, not just active requests.
        decode_q_per_req = _env_int(
            "VLLM_B12X_MLA_DECODE_Q_PER_REQ",
            _default_decode_q_per_req(
                num_speculative_tokens,
                spec_extend_as_decode=self.spec_extend_as_decode,
                spec_decode_max_q=self.spec_decode_max_q,
            ),
        )
        self.decode_max_total_q = _env_int(
            "VLLM_B12X_MLA_DECODE_MAX_TOTAL_Q",
            self.max_num_seqs * decode_q_per_req,
        )
        self.decode_max_total_q = max(
            int(self.decode_max_total_q), int(self.max_num_seqs)
        )
        self.debug_b12x_mla = os.getenv("VLLM_DEBUG_B12X_MLA", "0") == "1"
        self.debug_b12x_mla_file = os.getenv(
            "VLLM_B12X_MLA_DEBUG_FILE",
            f"/tmp/vllm_b12x_mla_debug_{os.getpid()}.log",
        )
        self.compact_nsa_page_table = (
            os.getenv("VLLM_B12X_MLA_RECOUNT_NSA_SEQLENS", "1") != "0"
        )
        self.extend_use_cuda_graph = (
            os.getenv("VLLM_B12X_MLA_EXTEND_CUDA_GRAPH", "1") != "0"
        )
        self.decode_use_cuda_graph = (
            os.getenv("VLLM_B12X_MLA_DECODE_CUDA_GRAPH", "1") != "0"
        )
        self.decode_workspace_per_layer = (
            os.getenv("VLLM_B12X_MLA_DECODE_WORKSPACE_PER_LAYER", "0") != "0"
        )
        self.decode_workspace_ring = _env_int(
            "VLLM_B12X_MLA_DECODE_WORKSPACE_RING", 1
        )
        # B12X native LSE mode currently uses the split sparse-MLA path. Keep a
        # bounded split reserve for extend: non-DCP mirrors the sglang b12x
        # contract so short extend/prefill does not collapse to the one-pass
        # q x head_tiles launch, while DCP keeps a smaller cap because
        # all-gathered heads multiply split scratch.
        self.extend_max_chunks_per_row = _env_int(
            "VLLM_B12X_MLA_EXTEND_MAX_CHUNKS",
            4 if self.dcp_world_size > 1 else 32,
        )
        self.extend_indexer_tile_logits_k_rows = _env_nonnegative_int(
            "VLLM_B12X_MLA_EXTEND_TOPK_SUPERTILE_K",
            _env_nonnegative_int("B12X_NSA_EXTEND_TOPK_SUPERTILE_K", 32768),
        )
        self.clamp_nsa_to_valid_prefix = (
            os.getenv("VLLM_B12X_MLA_CLAMP_NSA_TO_VALID_PREFIX", "0") != "0"
        )
        self.extend_clamp_nsa_to_cache = (
            os.getenv("VLLM_B12X_MLA_EXTEND_CLAMP_NSA_TO_CACHE", "0") != "0"
        )

        (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
            ((max_tokens, num_heads, head_size), torch.bfloat16),
        )
        self._b12x_joint_arena_preinstalled = False

    @staticmethod
    def _moe_backend_name(vllm_config: VllmConfig) -> str:
        kernel_config = getattr(vllm_config, "kernel_config", None)
        return str(getattr(kernel_config, "moe_backend", "") or "").lower()

    def _decode_workspace_layer_key(self, layer_key: str | None) -> str | None:
        if self.decode_workspace_per_layer:
            return layer_key
        if self.decode_workspace_ring <= 1:
            return None
        layer_idx = None
        if layer_key:
            for part in reversed(layer_key.split(".")):
                if part.isdigit():
                    layer_idx = int(part)
                    break
        if layer_idx is None:
            layer_idx = 0
        return f"ring{layer_idx % self.decode_workspace_ring}"

    def _build_b12x_moe_arena_caps(self, device: torch.device, dtype: torch.dtype):
        if not self.use_joint_arena:
            return None
        if self._moe_backend_name(self.vllm_config) != "b12x":
            return None

        try:
            from b12x.integration import B12XMoEArenaCaps
        except ImportError as exc:
            raise RuntimeError(
                "B12X sparse MLA attention with B12X MoE requires a b12x "
                "build with shared MoE arena capacity support"
            ) from exc

        cfg = self.hf_config
        weight_e = getattr(cfg, "n_routed_experts", None)
        if weight_e is None:
            weight_e = getattr(cfg, "num_experts", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        intermediate_size = getattr(cfg, "moe_intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(cfg, "intermediate_size", None)
        num_topk = getattr(cfg, "num_experts_per_tok", None)
        if num_topk is None:
            num_topk = getattr(cfg, "top_k", None)

        missing = [
            name
            for name, value in (
                ("n_routed_experts/num_experts", weight_e),
                ("hidden_size", hidden_size),
                ("moe_intermediate_size/intermediate_size", intermediate_size),
                ("num_experts_per_tok/top_k", num_topk),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "B12X joint arena cannot size MoE workspace; missing "
                f"{', '.join(missing)}"
            )

        tp_size = max(1, int(self.vllm_config.parallel_config.tensor_parallel_size))
        intermediate_size = int(intermediate_size)
        if intermediate_size % tp_size != 0:
            raise RuntimeError(
                "B12X joint arena expected MoE intermediate_size divisible by TP: "
                f"intermediate_size={intermediate_size} tp_size={tp_size}"
            )

        decode_tokens = max(1, int(self.decode_max_total_q))
        extend_tokens = max(1, int(self.arena_extend_max_total_q))
        max_tokens = max(decode_tokens, extend_tokens)
        return B12XMoEArenaCaps(
            device=device,
            dtype=dtype,
            weight_E=int(weight_e),
            k=int(hidden_size),
            n=intermediate_size // tp_size,
            num_topk=int(num_topk),
            max_tokens=max_tokens,
            core_token_counts=(extend_tokens, decode_tokens),
            route_num_experts=int(weight_e),
            route_logits_dtype=dtype,
        )

    def _build_b12x_attention_arena_caps(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        topk: int,
        max_page_table_width: int,
        max_chunks_per_row: int,
        caps_extend_max_total_q: int,
        caps_extend_max_batch: int,
        caps_extend_max_kv_rows: int,
        reserve_extend_indexer_logits: bool,
        extend_indexer_tile_logits_k_rows: int,
    ):
        from b12x.integration.mla import B12XAttentionArenaCaps

        kwargs = dict(
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            indexer_num_q_heads=self.indexer_num_q_heads,
            head_dim=self.head_size,
            max_v_head_dim=self.kv_lora_rank,
            topk=topk,
            max_page_table_width=max_page_table_width,
            extend_max_total_q=caps_extend_max_total_q,
            extend_max_batch=caps_extend_max_batch,
            extend_max_kv_rows=caps_extend_max_kv_rows,
            paged_max_q_rows=self.decode_max_total_q,
            paged_max_batch=self.decode_max_total_q,
            page_size=64,
            max_chunks_per_row=max_chunks_per_row,
        )
        return B12XAttentionArenaCaps(
            **kwargs,
            reserve_extend_indexer_logits=reserve_extend_indexer_logits,
            extend_indexer_tile_logits_k_rows=extend_indexer_tile_logits_k_rows,
        )

    def _allocate_b12x_attention_arena(self, attention_caps, moe_caps):
        from b12x.integration.mla import B12XAttentionArena

        if self.use_joint_arena:
            try:
                from b12x.integration import (
                    B12XJointArenaSpec,
                    ensure_b12x_execution_lane_arena,
                )

                lane = ensure_b12x_execution_lane_arena(
                    B12XJointArenaSpec(
                        device=attention_caps.device,
                        attention_caps=attention_caps,
                        moe_caps=moe_caps,
                    )
                )
                if lane.arena is not None and lane.arena.attention_arena is not None:
                    return lane.arena.attention_arena
            except (AttributeError, ImportError, RuntimeError) as exc:
                raise RuntimeError(
                    "B12X sparse MLA attention with B12X MoE requires a shared "
                    "execution-lane arena and cannot fall back to a standalone "
                    "attention arena"
                ) from exc
            raise RuntimeError(
                "B12X sparse MLA attention with B12X MoE requires a shared "
                "execution-lane arena with an attention arena"
            )

        return B12XAttentionArena.allocate(attention_caps)

    def _preinstall_b12x_joint_arena(self) -> None:
        """Install the b12x joint lane before the first MoE forward.

        vLLM creates the MoE workspace lazily in the first model forward. If
        that happens before MLA asks for an attention arena, b12x has to create
        a standalone MoE lane and refuses to replace it with a joint arena once
        workspaces are live. Preinstalling the decode-sized attention arena here
        mirrors SGLang's eager NSA backend initialization order.
        """
        if (
            not self.use_arena
            or not self.use_joint_arena
            or self._moe_backend_name(self.vllm_config) != "b12x"
            or not current_platform.is_cuda()
        ):
            return
        if getattr(self, "_b12x_joint_arena_preinstalled", False):
            return

        from b12x.integration.mla import B12XAttentionWorkspaceContract

        device = torch.device("cuda", torch.cuda.current_device())
        dcp_size = max(
            1,
            int(
                getattr(
                    self.vllm_config.parallel_config,
                    "decode_context_parallel_size",
                    1,
                )
            ),
        )
        num_q_heads = int(self.num_heads) * dcp_size
        topk = int(self.topk_indices_buffer.shape[1])
        max_chunks_per_row = int(getattr(self, "extend_max_chunks_per_row", 64))
        attention_caps = self._build_b12x_attention_arena_caps(
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=num_q_heads,
            topk=topk,
            max_page_table_width=topk,
            max_chunks_per_row=max_chunks_per_row,
            caps_extend_max_total_q=self.arena_extend_max_total_q,
            caps_extend_max_batch=self.arena_extend_max_batch,
            caps_extend_max_kv_rows=self.arena_extend_max_kv_rows,
            reserve_extend_indexer_logits=False,
            extend_indexer_tile_logits_k_rows=int(
                getattr(self, "extend_indexer_tile_logits_k_rows", 32768)
            ),
        )
        moe_caps = self._build_b12x_moe_arena_caps(device, torch.bfloat16)
        arena_key = (
            device.type,
            device.index,
            "decode",
            torch.bfloat16,
            torch.uint8,
            num_q_heads,
            self.head_size,
            self.kv_lora_rank,
            self.indexer_num_q_heads,
            topk,
            self.decode_max_total_q,
            self.arena_extend_max_total_q,
            self.arena_extend_max_batch,
            self.arena_extend_max_kv_rows,
            False,
            int(getattr(self, "extend_indexer_tile_logits_k_rows", 32768)),
            self.use_joint_arena,
            self._moe_backend_name(self.vllm_config),
        )
        arena = B12X_ARENAS.get(arena_key)
        if arena is None:
            arena = self._allocate_b12x_attention_arena(attention_caps, moe_caps)
            B12X_ARENAS[arena_key] = arena
        decode_only_arena_key = (
            device.type,
            device.index,
            "decode",
            torch.bfloat16,
            torch.uint8,
            num_q_heads,
            self.head_size,
            self.kv_lora_rank,
            self.indexer_num_q_heads,
            topk,
            self.decode_max_total_q,
            1,
            1,
            1,
            False,
            0,
            self.use_joint_arena,
            self._moe_backend_name(self.vllm_config),
        )
        B12X_ARENAS.setdefault(decode_only_arena_key, arena)

        workspace_key = (
            device.type,
            device.index,
            "decode",
            None,
            torch.bfloat16,
            torch.uint8,
            num_q_heads,
            self.head_size,
            self.kv_lora_rank,
            self.indexer_num_q_heads,
            topk,
            self.decode_max_total_q,
            self.decode_max_total_q,
            0,
            topk,
            64,
            self.use_arena,
            self.arena_max_running_requests,
            self.arena_extend_max_total_q,
            self.arena_extend_max_batch,
            self.arena_extend_max_kv_rows,
            0,
            self.use_joint_arena,
            self._moe_backend_name(self.vllm_config),
        )
        workspace = B12X_WORKSPACES.get(workspace_key)
        if workspace is None:
            contract = B12XAttentionWorkspaceContract(
                mode="decode",
                max_total_q=self.decode_max_total_q,
                max_batch=self.decode_max_total_q,
                max_paged_q_rows=self.decode_max_total_q,
                max_kv_rows=0,
                v_head_dim=self.kv_lora_rank,
                indexer_num_q_heads=self.indexer_num_q_heads,
                max_page_table_width=topk,
            )
            workspace = arena.make_workspace(
                contract, use_cuda_graph=self.decode_use_cuda_graph
            )
            B12X_WORKSPACES[workspace_key] = workspace
        self._ensure_decode_split_chunk_config(
            workspace,
            page_table_width=topk,
        )
        _prime_b12x_sm_scale(workspace, device, self.scale)
        self._b12x_joint_arena_preinstalled = True

    def _ensure_decode_split_chunk_config(
        self,
        workspace: object,
        *,
        page_table_width: int,
    ) -> None:
        if getattr(workspace, "tmp_lse", None) is None:
            return
        from b12x.attention.mla.split import (
            forced_sparse_mla_split_decode_config_for_width,
        )

        cfg = forced_sparse_mla_split_decode_config_for_width(
            int(page_table_width),
            max_chunks=int(getattr(workspace, "max_chunks_per_row", 64)),
        )
        if cfg is None:
            return
        if (
            getattr(workspace, "kv_chunk_size_value", None) == int(cfg.chunk_size)
            and getattr(workspace, "num_chunks_value", None) == int(cfg.num_chunks)
        ):
            return
        workspace.set_decode_chunk_config(
            kv_chunk_size=int(cfg.chunk_size),
            num_chunks=int(cfg.num_chunks),
        )

    def _get_workspace(
        self,
        mode: str,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        *,
        max_kv_rows: int | None = None,
        page_table_width: int | None = None,
        layer_key: str | None = None,
    ):
        from b12x.integration.mla import (
            B12XAttentionArenaCaps,
            B12XAttentionWorkspaceContract,
        )

        num_q_heads = int(q.shape[1])

        # Decode CUDA-graph captures operate on graph-stable rows. With MTP,
        # speculative verify rows are target+draft tokens per active sequence.
        # Keep row capacity separate from request-batch capacity: b12x arena caps
        # reserve paged q rows for MTP verification, but paged batch remains the
        # number of concurrently running requests.
        # Extend also needs a stable capacity contract: b12x kernels consume
        # live sequence lengths as runtime metadata, but CUTE host-launcher
        # cache keys still include tensor/workspace shapes.
        max_total_q = (
            max(int(self.decode_max_total_q), int(q.shape[0]))
            if mode == "decode"
            else max(1, int(q.shape[0]), int(self.arena_extend_max_total_q))
        )
        topk = int(self.topk_indices_buffer.shape[1])
        max_chunks_per_row = 64
        if mode != "decode":
            # DCP all-gathers Q heads before the B12X call. Keeping the decode
            # 64-way split reserve for extend would allocate Q x heads x 64 x V
            # scratch. Use the configured extend cap instead: non-DCP defaults
            # to a sglang-aligned 32 chunks for occupancy on short extend, while
            # DCP defaults lower because the all-gathered head count multiplies
            # scratch.
            #
            # DCP is different: the post-attention reducer needs softmax LSE
            # from each rank. B12X's single-pass sparse MLA kernel currently
            # returns only output; LSE is produced by the split path via
            # workspace.tmp_lse. Use a small split-chunk cap by default in DCP
            # so topk=2048 selects 4x512 chunks instead of falling through to
            # single-pass with uninitialized LSE.
            #
            # Only cap split chunks when the installed b12x split selectors
            # accept the cap. Older b12x builds ignore the workspace limit and
            # then assert while applying a 64-chunk split config.
            max_chunks_per_row = int(self.extend_max_chunks_per_row)
        max_kv_rows = (
            0
            if mode == "decode"
            else max(
                1,
                int(max_kv_rows or self.max_model_len),
                int(self.arena_extend_max_kv_rows),
            )
        )
        arena_max_total_q = (
            self.decode_max_total_q
            if mode == "decode"
            else self.arena_extend_max_total_q
        )
        arena_max_batch = (
            self.decode_max_total_q
            if mode == "decode"
            else self.arena_extend_max_batch
        )
        arena_max_kv_rows = 0 if mode == "decode" else self.arena_extend_max_kv_rows
        max_batch = (
            self.decode_max_total_q
            if mode == "decode"
            else self.arena_extend_max_batch
        )
        max_page_table_width = max(1, int(page_table_width or topk))
        workspace_layer_key = (
            self._decode_workspace_layer_key(layer_key)
            if mode == "decode"
            else None
        )
        workspace_extend_indexer_tile_logits_k_rows = (
            0
            if mode == "decode"
            else int(getattr(self, "extend_indexer_tile_logits_k_rows", 32768))
        )
        key = (
            q.device.type,
            q.device.index,
            mode,
            workspace_layer_key,
            q.dtype,
            kv_cache.dtype,
            num_q_heads,
            self.head_size,
            self.kv_lora_rank,
            self.indexer_num_q_heads,
            topk,
            max_total_q,
            max_batch,
            max_kv_rows,
            max_page_table_width,
            max_chunks_per_row,
            self.use_arena,
            self.arena_max_running_requests,
            self.arena_extend_max_total_q,
            self.arena_extend_max_batch,
            self.arena_extend_max_kv_rows,
            workspace_extend_indexer_tile_logits_k_rows,
            self.use_joint_arena,
            self._moe_backend_name(self.vllm_config),
        )
        workspace = B12X_WORKSPACES.get(key)
        if workspace is not None:
            if mode == "decode":
                self._ensure_decode_split_chunk_config(
                    workspace,
                    page_table_width=max_page_table_width,
                )
            return workspace

        # B12X sparse MLA always uses the b12x arena/workspace contract.  Decode
        # and extend keep separate arena keys so the decode hot path does not
        # inherit long-context extend scratch reservations.
        use_arena_for_mode = self.use_arena
        if not use_arena_for_mode:
            raise RuntimeError(
                "B12X sparse MLA requires the b12x attention arena/workspace "
                "contract"
            )

        if use_arena_for_mode:
            decode_only_arena = mode == "decode"
            caps_extend_max_total_q = (
                1 if decode_only_arena else self.arena_extend_max_total_q
            )
            caps_extend_max_batch = (
                1 if decode_only_arena else self.arena_extend_max_batch
            )
            caps_extend_max_kv_rows = (
                1 if decode_only_arena else self.arena_extend_max_kv_rows
            )
            reserve_extend_indexer_logits = False
            extend_indexer_tile_logits_k_rows = (
                0
                if decode_only_arena
                else int(getattr(self, "extend_indexer_tile_logits_k_rows", 32768))
            )
            arena_key = (
                q.device.type,
                q.device.index,
                mode,
                q.dtype,
                kv_cache.dtype,
                num_q_heads,
                self.head_size,
                self.kv_lora_rank,
                self.indexer_num_q_heads,
                topk,
                self.decode_max_total_q,
                caps_extend_max_total_q,
                caps_extend_max_batch,
                caps_extend_max_kv_rows,
                reserve_extend_indexer_logits,
                extend_indexer_tile_logits_k_rows,
                self.use_joint_arena,
                self._moe_backend_name(self.vllm_config),
            )
            arena = B12X_ARENAS.get(arena_key)
            if arena is None and self.use_joint_arena:
                arena = _get_b12x_covering_arena(
                    device_type=q.device.type,
                    device_index=q.device.index,
                    mode=mode,
                    dtype=q.dtype,
                    kv_dtype=kv_cache.dtype,
                    num_q_heads=num_q_heads,
                    head_size=self.head_size,
                    kv_lora_rank=self.kv_lora_rank,
                    indexer_num_q_heads=self.indexer_num_q_heads,
                    topk=topk,
                    decode_max_total_q=self.decode_max_total_q,
                    caps_extend_max_total_q=caps_extend_max_total_q,
                    caps_extend_max_batch=caps_extend_max_batch,
                    caps_extend_max_kv_rows=caps_extend_max_kv_rows,
                    reserve_extend_indexer_logits=reserve_extend_indexer_logits,
                    extend_indexer_tile_logits_k_rows=(
                        extend_indexer_tile_logits_k_rows
                    ),
                    use_joint_arena=self.use_joint_arena,
                    moe_backend=self._moe_backend_name(self.vllm_config),
                )
                if arena is not None:
                    B12X_ARENAS[arena_key] = arena
            if arena is None:
                caps = self._build_b12x_attention_arena_caps(
                    device=q.device,
                    dtype=q.dtype,
                    kv_dtype=kv_cache.dtype,
                    num_q_heads=num_q_heads,
                    topk=topk,
                    max_page_table_width=topk,
                    max_chunks_per_row=max_chunks_per_row,
                    caps_extend_max_total_q=caps_extend_max_total_q,
                    caps_extend_max_batch=caps_extend_max_batch,
                    caps_extend_max_kv_rows=caps_extend_max_kv_rows,
                    reserve_extend_indexer_logits=reserve_extend_indexer_logits,
                    extend_indexer_tile_logits_k_rows=(
                        extend_indexer_tile_logits_k_rows
                    ),
                )
                moe_caps = self._build_b12x_moe_arena_caps(q.device, q.dtype)
                arena = self._allocate_b12x_attention_arena(caps, moe_caps)
                B12X_ARENAS[arena_key] = arena

            contract = B12XAttentionWorkspaceContract(
                mode=mode,
                max_total_q=arena_max_total_q,
                max_batch=arena_max_batch,
                max_paged_q_rows=(
                    self.decode_max_total_q if mode == "decode" else 1
                ),
                max_kv_rows=arena_max_kv_rows,
                v_head_dim=self.kv_lora_rank,
                indexer_num_q_heads=self.indexer_num_q_heads,
                max_page_table_width=max_page_table_width,
            )
            workspace = arena.make_workspace(
                contract,
                use_cuda_graph=(
                    self.decode_use_cuda_graph
                    if mode == "decode"
                    else self.extend_use_cuda_graph
                ),
            )
            _prime_b12x_sm_scale(workspace, q.device, self.scale)
            if mode == "decode":
                self._ensure_decode_split_chunk_config(
                    workspace,
                    page_table_width=max_page_table_width,
                )
            B12X_WORKSPACES[key] = workspace
            return workspace

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q_all = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            q_all = q

        num_actual_toks = q_all.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if self.dcp_world_size > 1 and self.dcp_topk_per_rank > 0:
            topk_indices = topk_indices[:, : self.dcp_topk_per_rank]
        page_table_1_buffer = attn_metadata.page_table_1[:, : topk_indices.shape[1]]
        is_decode = attn_metadata.max_query_len <= 1
        decode_topk_is_physical = is_decode and self.decode_topk_is_physical
        decode_fast_nsa_seqlens = is_decode and self.decode_fast_nsa_seqlens
        compacted_page_table = False
        if decode_topk_is_physical:
            page_table_1 = topk_indices
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
                : attn_metadata.cache_seq_lens_per_req.shape[0]
            ]
            torch.clamp(
                attn_metadata.cache_seq_lens_per_req,
                max=topk_indices.shape[1],
                out=nsa_cache_seqlens,
            )
        elif decode_fast_nsa_seqlens:
            page_table_1 = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=False,
                out=page_table_1_buffer,
            )
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
                : attn_metadata.cache_seq_lens_per_req.shape[0]
            ]
            torch.clamp(
                attn_metadata.cache_seq_lens_per_req,
                max=topk_indices.shape[1],
                out=nsa_cache_seqlens,
            )
        else:
            compacted_page_table = self.compact_nsa_page_table
            if compacted_page_table:
                page_table_1 = triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token,
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=False,
                    out=page_table_1_buffer,
                )
                nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
                    : page_table_1.shape[0]
                ]
            else:
                page_table_1, nsa_cache_seqlens = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token,
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        out=page_table_1_buffer,
                        valid_counts=attn_metadata.nsa_cache_seqlens,
                    )
                )
            if compacted_page_table:
                _compact_page_table_valid_prefix(page_table_1, nsa_cache_seqlens)

        if self.clamp_nsa_to_valid_prefix:
            valid_mask = page_table_1 >= 0
            invalid_mask = ~valid_mask
            prefix_counts = torch.argmax(invalid_mask.to(torch.int32), dim=1)
            prefix_counts = torch.where(
                invalid_mask.any(dim=1),
                prefix_counts,
                torch.full_like(prefix_counts, page_table_1.shape[1]),
            )
            torch.minimum(
                nsa_cache_seqlens[: prefix_counts.shape[0]],
                prefix_counts,
                out=nsa_cache_seqlens[: prefix_counts.shape[0]],
            )

        if self.extend_clamp_nsa_to_cache:
            extend_cache_seqlens = (
                attn_metadata.cache_seq_lens_per_token
                if attn_metadata.max_query_len > 1
                else attn_metadata.cache_seq_lens_per_req
            )
            row_count = min(page_table_1.shape[0], extend_cache_seqlens.shape[0])
            nsa_rows = nsa_cache_seqlens[:row_count]
            torch.minimum(
                nsa_rows,
                extend_cache_seqlens[:row_count],
                out=nsa_rows,
            )
            _mask_page_table_after_nsa_len(page_table_1[:row_count], nsa_rows)

        kv_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        ).contiguous()
        self._preinstall_b12x_joint_arena()

        use_decode_kernel = attn_metadata.max_query_len <= 1 or (
            self.spec_extend_as_decode
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and attn_metadata.num_actual_tokens
            <= attn_metadata.num_reqs * self.spec_decode_max_q
        )
        if use_decode_kernel:
            from b12x.integration.mla import MLASparseDecodeMetadata

            cache_seqlens = (
                attn_metadata.cache_seq_lens_per_req
                if attn_metadata.max_query_len <= 1
                else attn_metadata.cache_seq_lens_per_token
            )
            if (
                self.debug_b12x_mla
                and attn_metadata.max_query_len > 1
            ):
                debug_count = getattr(self, "_debug_b12x_mla_count", 0)
                if debug_count < 24:
                    try:
                        rows = min(4, q_all.shape[0])
                        cols = min(16, topk_indices.shape[1])
                        invalid_counts = (
                            topk_indices[:rows]
                            >= cache_seqlens[:rows].unsqueeze(1)
                        ).sum(dim=1)
                        debug_payload = (
                            f"layer={getattr(layer, 'layer_name', '')} "
                            f"max_q={attn_metadata.max_query_len} "
                            f"rows={q_all.shape[0]} heads={q_all.shape[1]} "
                            f"cache={cache_seqlens[:rows].detach().cpu().tolist()} "
                            f"nsa={nsa_cache_seqlens[:rows].detach().cpu().tolist()} "
                            f"invalid={invalid_counts.detach().cpu().tolist()} "
                            f"topk={topk_indices[:rows, :cols].detach().cpu().tolist()} "
                            f"page={page_table_1[:rows, :cols].detach().cpu().tolist()}"
                        )
                        logger.warning("B12X_MLA_DEBUG[%d] %s", debug_count,
                                       debug_payload)
                        with open(
                            self.debug_b12x_mla_file, "a", encoding="utf-8"
                        ) as f:
                            f.write(
                                f"{time.time():.6f} B12X_MLA_DEBUG[{debug_count}] "
                                f"{debug_payload}\n"
                            )
                    except Exception:
                        pass
                    setattr(self, "_debug_b12x_mla_count", debug_count + 1)
            if not (decode_topk_is_physical or decode_fast_nsa_seqlens):
                # The generic req-index -> physical-index converter can only
                # tell whether a selected index maps to an allocated block. For
                # speculative verify rows (q_len > 1), that is weaker than the
                # causal per-token context length: top-k padding / future draft
                # slots may still map to physical cache rows. B12X consumes
                # nsa_cache_seqlens as the number of selected rows to attend, so
                # clamp it to the effective per-token KV length here.
                nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
                    : cache_seqlens.shape[0]
                ]
                if compacted_page_table:
                    torch.minimum(
                        nsa_cache_seqlens,
                        cache_seqlens[: nsa_cache_seqlens.shape[0]],
                        out=nsa_cache_seqlens,
                    )
                else:
                    torch.clamp(
                        cache_seqlens,
                        max=topk_indices.shape[1],
                        out=nsa_cache_seqlens,
                    )
                _mask_page_table_after_nsa_len(page_table_1, nsa_cache_seqlens)
            workspace = self._get_workspace(
                "decode",
                q_all,
                kv_cache,
                page_table_width=int(page_table_1.shape[1]),
                layer_key=getattr(layer, "layer_name", ""),
            )
            metadata = MLASparseDecodeMetadata(
                page_table_1=page_table_1,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
                max_seq_len_k=attn_metadata.max_seq_len,
            )
            if self.need_to_return_lse_for_decode:
                return _sparse_mla_decode_forward_with_lse_vllm_metadata(
                    q_all=q_all,
                    kv_cache=kv_cache,
                    metadata=metadata,
                    workspace=workspace,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                )
            out = _sparse_mla_decode_forward_vllm_metadata(
                q_all=q_all,
                kv_cache=kv_cache,
                metadata=metadata,
                workspace=workspace,
                sm_scale=self.scale,
                v_head_dim=self.kv_lora_rank,
            )
            if not self.need_to_return_lse_for_decode:
                return out, None
            raise RuntimeError("B12X sparse MLA DCP LSE path was not selected")

        from b12x.integration.mla import MLASparseExtendMetadata

        workspace = self._get_workspace(
            "extend",
            q_all,
            kv_cache,
            max_kv_rows=int(attn_metadata.max_seq_len),
            page_table_width=int(page_table_1.shape[1]),
        )
        nsa_cu_seqlens_k = attn_metadata.nsa_cu_seqlens_k[
            : nsa_cache_seqlens.shape[0] + 1
        ]
        nsa_cu_seqlens_k[:1].zero_()
        torch.cumsum(nsa_cache_seqlens, dim=0, out=nsa_cu_seqlens_k[1:])
        metadata = MLASparseExtendMetadata(
            selected_token_offsets=page_table_1,
            cache_seqlens_int32=attn_metadata.cache_seq_lens_per_req,
            nsa_cache_seqlens_int32=nsa_cache_seqlens,
            nsa_cu_seqlens_q=attn_metadata.nsa_cu_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            max_seq_len_q=attn_metadata.max_query_len,
            max_seq_len_k=attn_metadata.max_seq_len,
            mode="extend",
        )
        if self.debug_b12x_mla:
            debug_count = getattr(self, "_debug_b12x_mla_extend_count", 0)
            if debug_count < 24:
                try:
                    rows = min(4, q_all.shape[0])
                    cols = min(16, page_table_1.shape[1])
                    kv_rows = int(kv_cache.shape[0])
                    page_preview = page_table_1[:rows, :cols]
                    nsa_preview = nsa_cache_seqlens[:rows]
                    valid_counts = (page_table_1 >= 0).sum(dim=1)
                    prefix_invalid = page_table_1 < 0
                    prefix_counts = torch.argmax(
                        prefix_invalid.to(torch.int32), dim=1
                    )
                    prefix_counts = torch.where(
                        prefix_invalid.any(dim=1),
                        prefix_counts,
                        torch.full_like(prefix_counts, page_table_1.shape[1]),
                    )
                    bad_valid = int(
                        (valid_counts < nsa_cache_seqlens).sum().item()
                    )
                    bad_prefix = int(
                        (prefix_counts < nsa_cache_seqlens).sum().item()
                    )
                    invalid_neg = int((page_table_1 < 0).sum().item())
                    invalid_oob = int((page_table_1 >= kv_rows).sum().item())
                    page_min = int(page_table_1.min().item())
                    page_max = int(page_table_1.max().item())
                    nsa_max = int(nsa_cache_seqlens.max().item())
                    debug_payload = (
                        f"layer={getattr(layer, 'layer_name', '')} "
                        f"mode=extend max_q={attn_metadata.max_query_len} "
                        f"rows={q_all.shape[0]} heads={q_all.shape[1]} "
                        f"num_reqs={attn_metadata.num_reqs} "
                        f"dcp_world={self.dcp_world_size} "
                        f"compact={compacted_page_table} "
                        f"kv_rows={kv_rows} page_min={page_min} "
                        f"page_max={page_max} nsa_max={nsa_max} "
                        f"valid_min={int(valid_counts.min().item())} "
                        f"prefix_min={int(prefix_counts.min().item())} "
                        f"bad_valid={bad_valid} bad_prefix={bad_prefix} "
                        f"invalid_neg={invalid_neg} invalid_oob={invalid_oob} "
                        f"cache_req={attn_metadata.cache_seq_lens_per_req[:rows].detach().cpu().tolist()} "
                        f"nsa={nsa_preview.detach().cpu().tolist()} "
                        f"valid={valid_counts[:rows].detach().cpu().tolist()} "
                        f"prefix={prefix_counts[:rows].detach().cpu().tolist()} "
                        f"page={page_preview.detach().cpu().tolist()}"
                    )
                    logger.warning("B12X_MLA_DEBUG_EXTEND[%d] %s", debug_count,
                                   debug_payload)
                    with open(self.debug_b12x_mla_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"{time.time():.6f} B12X_MLA_DEBUG_EXTEND[{debug_count}] "
                            f"{debug_payload}\n"
                        )
                except Exception:
                    pass
                setattr(self, "_debug_b12x_mla_extend_count", debug_count + 1)
        if self.need_to_return_lse_for_decode:
            out, lse = _sparse_mla_extend_forward_with_lse_vllm_metadata(
                q_all=q_all,
                kv_cache=kv_cache,
                metadata=metadata,
                workspace=workspace,
                sm_scale=self.scale,
                v_head_dim=self.kv_lora_rank,
            )
            _b12x_sync_debug("mla.sparse_mla_extend_forward_with_lse")
            return out, lse

        out = _sparse_mla_extend_forward_vllm_metadata(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=metadata,
            workspace=workspace,
            sm_scale=self.scale,
            v_head_dim=self.kv_lora_rank,
        )
        _b12x_sync_debug("mla.sparse_mla_extend_forward")
        return out, None
