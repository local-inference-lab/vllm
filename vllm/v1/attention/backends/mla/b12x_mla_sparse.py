# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

from dataclasses import dataclass, replace
from math import gcd
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_ag_rs,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_sparse_mla,
    set_b12x_preparation_provider,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    _remap_tiling,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    refresh_dcp_local_seq_lens_,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata


_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_GLM_DSA_MODEL_TYPES = frozenset(("glm_moe_dsa",))
_GLM_NEXT_CACHE_RECORD_BYTES = 528
_GLM_NEXT_NVFP4_CACHE_RECORD_BYTES = 304
_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN = 132 // 4
_GLM_NEXT_INDEX_PAGE_BYTES = 64 * 132
_GLM_DSA_NVFP4_CACHE_RECORD_BYTES = 368

logger = init_logger(__name__)


@runtime_checkable
class B12xPhysicalSelectionProvider(Protocol):
    def get_b12x_physical_selection(
        self,
        *,
        num_tokens: int,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None: ...


def _is_glm_next_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


def _is_glm_dsa_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_DSA_MODEL_TYPES


def _current_hf_text_config() -> object | None:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return vllm_config.model_config.hf_text_config


def _is_glm_next_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm5_next":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_next_config(hf_config)


def _is_glm_dsa_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm_moe_dsa":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_dsa_config(hf_config)


def _nvfp4_run_options(*, is_glm_next: bool) -> dict[str, bool | int]:
    if is_glm_next:
        # GLM5Next has no RoPE payload and stores one latent scale per token.
        return {
            "scale_format": 2,
            "fp8_rope": False,
            "latent_scale_per_token": True,
        }
    # GLM-5.2/5.3 DSA stores its 64-dimensional RoPE tail as E4M3.
    return {"scale_format": 2, "fp8_rope": True}


def _glm_next_recipe_error(hf_config: object) -> str | None:
    expected = {
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
    }
    mismatches = [
        f"{name}={getattr(hf_config, name, None)!r} (expected {value})"
        for name, value in expected.items()
        if getattr(hf_config, name, None) != value
    ]
    if mismatches:
        return "B12X GLM5Next sparse MLA requires " + ", ".join(mismatches)
    return None


def _glm_next_dcp_error(vllm_config: VllmConfig) -> str | None:
    parallel_config = vllm_config.parallel_config
    dcp_size = int(parallel_config.decode_context_parallel_size)
    if dcp_size <= 1:
        return None
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    if interleave % 4:
        return (
            "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
        )
    return None


def _selected_index_block_stride_rows(
    kv_cache: torch.Tensor,
    *,
    block_size: int,
) -> int:
    # B12X selected indices are physical token slots. The kernel applies the
    # cache's byte page stride when it addresses the selected page.
    if int(kv_cache.shape[1]) != block_size:
        raise ValueError(
            "B12X sparse MLA cache page size does not match attention metadata: "
            f"cache={int(kv_cache.shape[1])}, metadata={block_size}"
        )
    return block_size


def _max_speculative_decode_query_len(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return 1
    num_speculative_tokens = int(spec_config.num_speculative_tokens or 0)
    multiplier = 2 if spec_config.parallel_drafting else 1
    return 1 + multiplier * num_speculative_tokens


def _is_speculative_decode_batch(
    common_attn_metadata: "CommonAttentionMetadata",
    max_speculative_decode_query_len: int,
) -> bool:
    is_prefilling = getattr(common_attn_metadata, "is_prefilling", None)
    return (
        max_speculative_decode_query_len > 1
        and 1 < common_attn_metadata.max_query_len <= max_speculative_decode_query_len
        and is_prefilling is not None
        and not bool(torch.any(is_prefilling[: common_attn_metadata.num_reqs]))
    )


def _is_glm_next_ckv_source_layout(
    kv_cache: torch.Tensor,
    *,
    page_size: int,
    record_bytes: int,
) -> bool:
    return (
        kv_cache.dtype == torch.uint8
        and kv_cache.ndim == 3
        and tuple(kv_cache.shape[1:]) == (page_size, record_bytes)
        and kv_cache.stride(1) == record_bytes
        and kv_cache.stride(2) == 1
    )


def _use_b12x_full_ckv_gather(
    *,
    enabled: bool,
    is_glm_next: bool,
    dcp_world_size: int,
    max_query_len: int,
    num_tokens: int,
    num_decode_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> bool:
    return (
        enabled
        and is_glm_next
        and dcp_world_size > 1
        and max_query_len > 1
        and num_decode_tokens == 0
        and num_tokens > min_tokens
        and num_tokens <= max_tokens
    )


def _ckv_rank_token_alignment(page_size: int, dcp_world_size: int) -> int:
    """Per-rank padding that makes the gathered CKV page-addressable.

    All-gather concatenates one equally sized token span from every DCP rank.
    Individual rank boundaries do not need to coincide with KV-page boundaries;
    only the concatenated span must contain an integral number of pages.
    """
    if page_size <= 0 or dcp_world_size <= 0:
        raise ValueError("CKV page size and DCP world size must be positive")
    return page_size // gcd(page_size, dcp_world_size)


def _round_up_ckv_rank_tokens(
    token_count: int,
    *,
    page_size: int,
    dcp_world_size: int,
) -> int:
    alignment = _ckv_rank_token_alignment(page_size, dcp_world_size)
    return (token_count + alignment - 1) // alignment * alignment


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if output_tensor.numel() != input_tensor.numel() * group.world_size:
        raise ValueError("CKV all-gather tensors have incompatible sizes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    output_tensor.copy_(group.all_gather(input_tensor, dim=0))


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


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Compute each query token's causal length in the gathered global cache.

    Args:
        global_seq_lens: Global cache sequence length for each request.
        query_start_loc: Start offset of each request's query chunk.
        req_id_per_token: Request identifier for each query token.
        num_actual_tokens: Number of active query tokens.

    Returns:
        Causal cache length for each active query token.
    """
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_tokens].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_tokens,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


@triton.jit
def _map_global_topk_to_gathered_ckv_kernel(
    req_id_ptr,
    token_indices_ptr,
    rank_req_starts_ptr,
    rank_req_lens_ptr,
    out_ptr,
    valid_count_ptr,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_TOPK_TOKENS
    req = tl.load(req_id_ptr + row)
    tok = tl.load(
        token_indices_ptr + row * ti_stride0 + cols * ti_stride1,
        mask=col_mask,
        other=-1,
    )
    owner = (tok // DCP_INTERLEAVE) % DCP_SIZE
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
    valid_tok = col_mask & (tok >= 0)
    req_start = tl.load(
        rank_req_starts_ptr + owner * starts_stride0 + req * starts_stride1,
        mask=valid_tok,
        other=0,
    )
    req_len = tl.load(
        rank_req_lens_ptr + owner * lens_stride0 + req * lens_stride1,
        mask=valid_tok,
        other=0,
    )
    valid = valid_tok & (local_idx >= 0) & (local_idx < req_len)
    gathered_slot = owner * padded_rank_tokens + req_start + local_idx
    valid_i32 = valid.to(tl.int32)
    local_offset = tl.cumsum(valid_i32) - valid_i32
    tile_valid_count = tl.sum(valid_i32)
    if tl.constexpr(BLOCK_N >= NUM_TOPK_TOKENS):
        output_base = 0
        tl.store(valid_count_ptr + row, tile_valid_count)
        tl.store(
            out_ptr + row * out_stride0 + cols * out_stride1,
            -1,
            mask=col_mask & (cols >= tile_valid_count),
        )
    else:
        output_base = tl.atomic_add(valid_count_ptr + row, tile_valid_count)
    tl.store(
        out_ptr + row * out_stride0 + (output_base + local_offset) * out_stride1,
        gathered_slot,
        mask=valid,
    )


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")

    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        token_indices.shape[1], 128, True
    )
    if not single_tile:
        out.fill_(-1)
        valid_counts.zero_()
    _map_global_topk_to_gathered_ckv_kernel[(token_indices.shape[0], tiles_per_row)](
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        rank_req_starts.stride(0),
        rank_req_starts.stride(1),
        rank_req_lens.stride(0),
        rank_req_lens.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
        padded_rank_tokens,
        DCP_SIZE=dcp_size,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS=token_indices.shape[1],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        is_glm_next = _is_glm_next_spec(spec)
        is_glm_dsa_nvfp4 = isinstance(spec, MLAAttentionSpec) and (
            spec.cache_dtype_str == "nvfp4_ds_mla" and _is_glm_dsa_spec(spec)
        )
        if not is_glm_next and not is_glm_dsa_nvfp4:
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        if is_glm_dsa_nvfp4:
            if spec.head_size != 576:
                raise ValueError(
                    "B12X GLM DSA NVFP4 sparse MLA requires head_size=576, got "
                    f"{spec.head_size}."
                )
            return replace(
                spec,
                state_content_bytes=_GLM_DSA_NVFP4_CACHE_RECORD_BYTES,
                model_version="glm_moe_dsa",
            )
        if spec.head_size != 512:
            raise ValueError(
                "B12X GLM5Next sparse MLA requires head_size=512, got "
                f"{spec.head_size}."
            )
        record_bytes = (
            _GLM_NEXT_NVFP4_CACHE_RECORD_BYTES
            if spec.cache_dtype_str == "nvfp4_ds_mla"
            else _GLM_NEXT_CACHE_RECORD_BYTES
        )
        return replace(
            spec,
            state_content_bytes=record_bytes,
            alignment=_GLM_NEXT_INDEX_PAGE_BYTES,
            page_tail_bytes_per_token=_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The DSA kernels consume one layer-compact, token-major cache view.
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

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
        from vllm.config import get_current_vllm_config

        module = get_b12x_sparse_mla()
        if module is None:
            return "B12X sparse MLA requires the optional b12x package"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if getattr(hf_config, "index_topk", None) is None:
                return "B12X sparse MLA requires a model with index_topk"
            if _is_glm_next_config(hf_config):
                if recipe_error := _glm_next_recipe_error(hf_config):
                    return recipe_error
                if head_size != 512:
                    return "B12X GLM5Next sparse MLA requires head_size=512"
                if dcp_error := _glm_next_dcp_error(vllm_config):
                    return dcp_error
                return None
            if kv_cache_dtype == "nvfp4_ds_mla" and not _is_glm_dsa_config(hf_config):
                return (
                    "B12X nvfp4_ds_mla requires GLM5Next or the "
                    "GLM-5.2/5.3 DSA architecture"
                )
            if head_size != 576:
                return "B12X sparse MLA requires head_size=576"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


class B12xGLMDSAMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM DSA sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm_moe_dsa"))


class B12xGLM5NextMLASparseBackend(B12xMLASparseBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # GLM-Next's pooled index tail shares manager blocks with its MLA
        # latent page. Keep the layer dimension inside the manager block so
        # copies and swaps carry both cache regions together.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_builder_cls() -> type["B12xGLM5NextMLASparseMetadataBuilder"]:
        return B12xGLM5NextMLASparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm5_next"))

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Keep the hybrid manager page intact so its FP8 pooled-index tail is
        # copied and recycled with the corresponding MLA page.
        return [MultipleOf(64)]


@dataclass
class B12xMLASparseMetadata(SparseMLACommonMetadata):
    dcp_global_seq_lens: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens_per_token: torch.Tensor | None = None
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None
    is_spec_decode: bool = False
    ckv_selected_indices: torch.Tensor | None = None
    ckv_active_counts: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_ckv_gather_eligible: bool = False


class B12xMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[B12xMLASparseMetadata]
):
    metadata_cls = B12xMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    requires_glm_next_selector_metadata: bool

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        self.requires_glm_next_selector_metadata = _is_glm_next_config(hf_config)
        if self.requires_glm_next_selector_metadata and (
            dcp_error := _glm_next_dcp_error(vllm_config)
        ):
            raise ValueError(dcp_error)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # All step-dependent state is persistent. Generic DSA DCP additionally
        # refreshes its rank-local sequence lengths in place between steps.
        self.supports_draft_decode_metadata_update = True
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        scheduler_config = vllm_config.scheduler_config
        max_tokens = scheduler_config.max_num_batched_tokens
        max_reqs = int(scheduler_config.max_num_seqs)
        self._ckv_max_reqs = max_reqs
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        if self.requires_glm_next_selector_metadata:
            self._capture_default_state_slot_ids = torch.arange(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_slot_ids = torch.empty(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_is_fresh = torch.ones(
                max_reqs, dtype=torch.bool, device=device
            )
            self._capture_num_accepted_tokens = torch.ones(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_is_prefilling = torch.zeros(
                max_reqs, dtype=torch.bool, device=device
            )
        self._ckv_gather_requested = (
            self.requires_glm_next_selector_metadata
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        if self._ckv_gather_requested:
            hf_config = vllm_config.model_config.hf_text_config
            ckv_topk_tokens = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
            self.ckv_selected_indices_buffer = torch.empty(
                (max_tokens, ckv_topk_tokens), dtype=torch.int32, device=device
            )
            self.ckv_active_counts_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_lens_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_starts_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_local_cu_seq_lens_buffer = torch.empty(
                (max_reqs + 1,), dtype=torch.int32, device=device
            )
        else:
            self.ckv_selected_indices_buffer = None
            self.ckv_active_counts_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None
        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )

    def _stage_glm_next_selector_metadata(
        self,
        *,
        num_reqs: int,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None,
        selector_state_is_fresh: torch.Tensor | None,
        selector_num_accepted_tokens: torch.Tensor | None,
        selector_is_prefilling: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        values = (
            selector_state_slot_ids,
            selector_state_is_fresh,
            selector_num_accepted_tokens,
            selector_is_prefilling,
        )
        if not self.requires_glm_next_selector_metadata:
            if any(value is not None for value in values):
                raise TypeError(
                    "GLM5Next selector metadata was provided to a non-GLM "
                    "B12X sparse MLA builder"
                )
            return (None, None, None, None)

        capacity = int(self._capture_state_slot_ids.numel())
        if not 0 <= num_reqs <= capacity:
            raise ValueError(
                "GLM5Next selector request count exceeds the metadata buffer "
                f"capacity: num_reqs={num_reqs}, capacity={capacity}"
            )
        if not for_cudagraph_capture and any(value is None for value in values):
            raise RuntimeError(
                "B12X GLM5Next sparse MLA requires selector state slots, fresh "
                "flags, accepted-token counts, and prefill flags"
            )

        if for_cudagraph_capture:
            self._capture_state_slot_ids[:num_reqs].copy_(
                self._capture_default_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
        else:
            typed_values = tuple(value for value in values if value is not None)
            if any(
                value.ndim != 1 or value.numel() < num_reqs for value in typed_values
            ):
                raise ValueError(
                    "GLM5Next selector metadata must be one-dimensional and "
                    "cover every padded request row"
                )
            self._capture_state_slot_ids[:num_reqs].fill_(-1)
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
            assert selector_state_slot_ids is not None
            assert selector_state_is_fresh is not None
            assert selector_num_accepted_tokens is not None
            assert selector_is_prefilling is not None
            self._capture_state_slot_ids[:num_reqs].copy_(
                selector_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].copy_(
                selector_state_is_fresh[:num_reqs]
            )
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                selector_num_accepted_tokens[:num_reqs]
            )
            self._capture_is_prefilling[:num_reqs].copy_(
                selector_is_prefilling[:num_reqs]
            )

        return (
            self._capture_state_slot_ids[:num_reqs],
            self._capture_state_is_fresh[:num_reqs],
            self._capture_num_accepted_tokens[:num_reqs],
            self._capture_is_prefilling[:num_reqs],
        )

    def _build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        *,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        use_dcp = self.dcp_world_size > 1
        seq_lens = (
            common.dcp_local_seq_lens
            if use_dcp and common.dcp_local_seq_lens is not None
            else common.seq_lens
        )
        metadata.seq_lens = seq_lens
        metadata.dcp_global_seq_lens = (
            common.seq_lens[: common.num_reqs] if use_dcp else None
        )

        if self.supports_varlen_decode_cudagraph:
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
        elif common.max_query_len <= 1 and num_tokens == common.num_reqs:
            per_token_lens = seq_lens[:num_tokens]
        elif not use_dcp and common.positions is not None:
            # The decode kernel binds these lengths, so they must live in the
            # builder's buffer: a FULL CUDA graph captures the buffer address
            # and replays against whatever a later build wrote there.
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]
            per_token_lens.copy_(common.positions[:num_tokens])
            per_token_lens += 1
        else:
            starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            seq_lens_cpu_source = (
                common.seq_lens_cpu_upper_bound
                if common.seq_lens_cpu_upper_bound is not None
                else common.seq_lens_cpu
            )
            seq_lens_cpu = seq_lens_cpu_source.numpy().astype(np.int32, copy=False)
            host_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, query_len in enumerate(query_lens):
                if query_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(query_len)
                request_lens = torch.arange(
                    context_len + 1,
                    context_len + int(query_len) + 1,
                    dtype=torch.int32,
                )
                if use_dcp:
                    request_lens = get_dcp_local_seq_lens(
                        request_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                host_lens[start:end] = request_lens.numpy()
            host_tensor = torch.from_numpy(host_lens).pin_memory()
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                host_tensor, non_blocking=True
            )
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]

        metadata.cache_seq_lens_per_token = per_token_lens
        metadata.is_spec_decode = _is_speculative_decode_batch(
            common,
            self._max_speculative_decode_query_len,
        )
        if metadata.num_prefills:
            prefill_start = metadata.num_decodes
            prefill_end = prefill_start + metadata.num_prefills + 1
            metadata.prefill_query_lens_cpu = torch.diff(
                common.query_start_loc_cpu[prefill_start:prefill_end]
            )
            seq_lens_cpu_source = (
                common.seq_lens_cpu_upper_bound
                if common.seq_lens_cpu_upper_bound is not None
                else common.seq_lens_cpu
            )
            metadata.prefill_seq_lens_cpu = seq_lens_cpu_source[
                prefill_start : prefill_start + metadata.num_prefills
            ].clone()
        if _use_b12x_full_ckv_gather(
            enabled=self._ckv_gather_requested,
            is_glm_next=self.requires_glm_next_selector_metadata,
            dcp_world_size=self.dcp_world_size,
            max_query_len=common.max_query_len,
            num_tokens=num_tokens,
            num_decode_tokens=metadata.num_decode_tokens,
            min_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS,
            max_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS,
        ):
            assert self.ckv_selected_indices_buffer is not None
            assert self.ckv_active_counts_buffer is not None
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            global_seq_lens = common.seq_lens[: common.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_lens.copy_(all_rank_lens)
            rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_starts[:, 0].zero_()
            if common.num_reqs > 1:
                torch.cumsum(rank_req_lens[:, :-1], dim=1, out=rank_req_starts[:, 1:])
            local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: common.num_reqs + 1]
            local_cu_seq_lens[0].zero_()
            torch.cumsum(
                rank_req_lens[self.dcp_rank],
                dim=0,
                out=local_cu_seq_lens[1:],
            )
            rank_totals = rank_req_lens.sum(dim=1).tolist()
            local_total_tokens = int(rank_totals[self.dcp_rank])
            page_size = int(self.kv_cache_spec.block_size)
            padded_total_tokens = _round_up_ckv_rank_tokens(
                max(int(total) for total in rank_totals),
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            max_local_capacity = _round_up_ckv_rank_tokens(
                (envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS + self.dcp_world_size - 1)
                // self.dcp_world_size
                + self._ckv_max_reqs * self.cp_kv_cache_interleave_size,
                page_size=page_size,
                dcp_world_size=self.dcp_world_size,
            )
            if 0 < padded_total_tokens <= max_local_capacity:
                metadata.ckv_selected_indices = self.ckv_selected_indices_buffer[
                    :num_tokens
                ]
                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]
                metadata.dcp_rank_req_lens = rank_req_lens
                metadata.dcp_rank_req_starts = rank_req_starts
                metadata.dcp_local_cu_seq_lens = local_cu_seq_lens
                metadata.global_cache_seq_lens_per_req = global_seq_lens
                metadata.dcp_local_total_tokens = local_total_tokens
                metadata.dcp_padded_total_tokens = padded_total_tokens
                metadata.dcp_ckv_gather_eligible = True
        (
            metadata.selector_state_slot_ids,
            metadata.selector_state_is_fresh,
            metadata.selector_num_accepted_tokens,
            metadata.selector_is_prefilling,
        ) = self._stage_glm_next_selector_metadata(
            num_reqs=common.num_reqs,
            for_cudagraph_capture=for_cudagraph_capture,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            for_cudagraph_capture=False,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xMLASparseMetadata,
    ) -> None:
        if self.dcp_world_size > 1:
            global_seq_lens = metadata.dcp_global_seq_lens
            if global_seq_lens is None:
                raise RuntimeError(
                    "B12X fused DCP draft decode requires global sequence lengths"
                )
            refresh_dcp_local_seq_lens_(
                metadata.seq_lens,
                global_seq_lens,
                metadata.num_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )

        if self.requires_glm_next_selector_metadata:
            accepted = metadata.selector_num_accepted_tokens
            if accepted is None:
                raise RuntimeError(
                    "GLM5Next draft decode metadata requires accepted-token counts"
                )
            accepted.fill_(1)


@triton.jit
def _glm_device_token_metadata_kernel(
    query_start_loc,
    seq_lens,
    request_ids,
    causal_lens,
    NUM_REQS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), NUM_REQS, tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + tl.minimum(mid + 1, NUM_REQS))
        advance = (lo < hi) & (end <= token)
        hi = tl.where((lo < hi) & ~advance, mid, hi)
        lo = tl.where(advance, mid + 1, lo)
    valid = (token < NUM_TOKENS) & (lo < NUM_REQS)
    start = tl.load(query_start_loc + lo, mask=valid, other=0)
    end = tl.load(query_start_loc + lo + 1, mask=valid, other=0)
    length = tl.load(seq_lens + lo, mask=valid, other=0)
    length = tl.maximum(length - (end - start) + token - start + 1, 0)
    if DCP_SIZE > 1:
        base = length // (INTERLEAVE * DCP_SIZE) * INTERLEAVE
        length = base + tl.minimum(
            tl.maximum(length - base * DCP_SIZE - DCP_RANK * INTERLEAVE, 0),
            INTERLEAVE,
        )
    tl.store(request_ids + token, tl.where(valid, lo, 0), token < NUM_TOKENS)
    tl.store(causal_lens + token, tl.where(valid, length, 0), token < NUM_TOKENS)


class B12xGLM5NextMLASparseMetadataBuilder(B12xMLASparseMetadataBuilder):
    # The pooled selector must commit every fresh or extended prompt row
    # through run_prefill; decode commits only accepted prior rows.
    treat_short_extends_as_decodes: ClassVar[bool] = False
    supports_varlen_decode_cudagraph: ClassVar[bool] = True

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        # Adaptive verification redistributes only the decode prefix. Its total
        # length and the CPU prefill boundaries remain exact.
        if num_tokens:
            _glm_device_token_metadata_kernel[(triton.cdiv(num_tokens, 128),)](
                common.query_start_loc,
                common.seq_lens,
                self.req_id_per_token_buffer,
                self.cache_seq_lens_per_token_buffer,
                NUM_REQS=common.num_reqs,
                NUM_TOKENS=num_tokens,
                SEARCH_STEPS=common.num_reqs.bit_length(),
                DCP_SIZE=self.dcp_world_size,
                DCP_RANK=self.dcp_rank,
                INTERLEAVE=self.cp_kv_cache_interleave_size,
                BLOCK=128,
            )
        return self.req_id_per_token_buffer[:num_tokens]


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
    supports_pcp = False
    # B12X maps selections into caller-owned scratch or uses the indexer's
    # prepared physical selection. Generic group buffers are never consumed.
    uses_index_group = False

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
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "B12X sparse MLA does not support ALiBi, sliding window, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X sparse MLA supports decoder self-attention only."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_text_config
        self._is_glm_next = _is_glm_next_config(hf_config)
        self._is_glm_dsa = _is_glm_dsa_config(hf_config)
        self._physical_selection_provider = (
            indexer if isinstance(indexer, B12xPhysicalSelectionProvider) else None
        )
        self.supports_mtp_with_cp_non_trivial_interleave_size = self._is_glm_next
        if self._is_glm_next:
            if recipe_error := _glm_next_recipe_error(hf_config):
                raise ValueError(recipe_error)
            if dcp_error := _glm_next_dcp_error(vllm_config):
                raise ValueError(dcp_error)
            if head_size != 512:
                raise ValueError("B12X GLM5Next sparse MLA requires head_size=512.")
        else:
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "B12X sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64."
                )
            if head_size != 576:
                raise ValueError("B12X sparse MLA requires head_size=576.")
        if self.topk_indices_buffer is None:
            raise ValueError("B12X sparse MLA requires a top-k index buffer.")
        uses_nvfp4_cache = kv_cache_dtype == "nvfp4_ds_mla"
        self._uses_glm_dsa_nvfp4_cache = self._is_glm_dsa and uses_nvfp4_cache
        if uses_nvfp4_cache and not (
            self._is_glm_next or self._uses_glm_dsa_nvfp4_cache
        ):
            raise ValueError(
                "B12X nvfp4_ds_mla requires GLM5Next or the "
                "GLM-5.2/5.3 DSA architecture."
            )
        if kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla"):
            raise ValueError(
                "B12X sparse MLA requires a packed fp8_ds_mla or "
                "nvfp4_ds_mla KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )
        self._uses_nvfp4_cache = uses_nvfp4_cache

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        required_symbols = ["Caps", "bind", "plan", "run"]
        if self._is_glm_next:
            required_symbols.append(
                "concat_and_cache_glm_next_mla_nvfp4"
                if uses_nvfp4_cache
                else "concat_and_cache_glm_next_mla_fp8"
            )
        if self._uses_glm_dsa_nvfp4_cache:
            required_symbols.append("concat_and_cache_nvfp4_mla_fp8_rope")
        for name in required_symbols:
            getattr(module, name)
        self._bind = module.bind
        self._run = module.run
        self._model_type: int | None = None
        self._concat_and_cache_nvfp4_mla_fp8_rope = None
        self._concat_and_cache_glm_next_mla = None
        if self._is_glm_next:
            self._model_type = int(module.ModelType.GLM_NEXT)
            self._concat_and_cache_glm_next_mla = (
                module.concat_and_cache_glm_next_mla_nvfp4
                if self._uses_nvfp4_cache
                else module.concat_and_cache_glm_next_mla_fp8
            )
        elif self._uses_glm_dsa_nvfp4_cache:
            self._model_type = int(module.ModelType.GLM_NSA)
            self._concat_and_cache_nvfp4_mla_fp8_rope = (
                module.concat_and_cache_nvfp4_mla_fp8_rope
            )

        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(scheduler_config.max_num_batched_tokens)
        max_seqs = int(scheduler_config.max_num_seqs)
        self._input_num_heads = self.num_heads * self.dcp_world_size
        self._q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._topk_tokens = int(self.topk_indices_buffer.shape[-1])
        if self._is_glm_next:
            expected_width = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
            if self._topk_tokens != expected_width:
                raise ValueError(
                    "B12X GLM5Next sparse MLA requires a selector output width "
                    f"of {expected_width}, got {self._topk_tokens}."
                )
        self._max_tokens = max_tokens
        self._max_seqs = max_seqs
        self._max_speculative_decode_query_len = _max_speculative_decode_query_len(
            vllm_config
        )
        self._decode_max_rows = min(
            max_tokens,
            max_seqs * self._max_speculative_decode_query_len,
        )
        self._kv_dtype = torch.uint8
        kernel_page_size = (
            int(vllm_config.cache_config.block_size) if self._is_glm_next else 64
        )
        self._ckv_gather_enabled = (
            self._is_glm_next
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        max_ckv_tokens = envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        cp_kv_cache_interleave_size = int(
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self._ckv_capacity_tokens = (
            max_ckv_tokens + self.dcp_world_size - 1
        ) // self.dcp_world_size + max_seqs * cp_kv_cache_interleave_size
        self._ckv_local_capacity = 0
        self._module = module
        self._kernel_page_size = 0
        self._kernel_page_size_finalized = not self._is_glm_next
        self._plans: dict[tuple[str, int], object] = {}
        self._plan_caps: dict[tuple[str, int], object] = {}
        self._cache_writer_plan: object | None = None
        self._bound_kv_cache: torch.Tensor | None = None
        self._set_kernel_page_size(kernel_page_size)
        self.supports_quant_query_input = False
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return
        # Plan declarations read the page size.
        self._kernel_page_size = kernel_page_size
        keys = [("decode", self._decode_max_rows), ("extend", self._max_tokens)]
        if self._ckv_gather_enabled:
            keys.append(("ckv_extend", self._max_tokens))
        self._plans = {}
        self._plan_caps = {}
        for key in keys:
            self._plan_caps[key], self._plans[key] = self._declare_plan(*key)
        self._cache_record_bytes = int(self._plan_caps[keys[0]].cache_record_bytes)
        # This family's scratch layout is a deterministic function of Caps
        # alone (see plan_sparse_mla_scratch) and grows with the row capacity,
        # so the largest plan declared here also covers every decode plan
        # declared later for fewer rows.
        self._scratch_nbytes = max(
            int(plan.scratch_specs()[0].nbytes) for plan in self._plans.values()
        )
        if self._is_glm_next and self.dcp_world_size > 1:
            # Query gathering finishes before attention consumes scratch. Its
            # rank-major receive storage can therefore reuse the kernel arena.
            self._scratch_nbytes = max(
                self._scratch_nbytes,
                self._max_tokens
                * self._input_num_heads
                * self._q_head_dim
                * torch.bfloat16.itemsize,
            )
        self._ckv_local_capacity = _round_up_ckv_rank_tokens(
            self._ckv_capacity_tokens,
            page_size=kernel_page_size,
            dcp_world_size=self.dcp_world_size,
        )

    def _declare_plan(self, mode: str, rows: int):
        """Declare the plan for mode at rows query rows; returns (caps, plan).

        The decode launch is exact in its row count, so decode plans are
        declared per row count. The extend modes accept any row count up to
        their declared capacity.
        """
        caps_kwargs = dict(
            device=torch.device("cuda", torch.accelerator.current_device_index()),
            num_q_heads=(
                self.num_heads if mode == "ckv_extend" else self._input_num_heads
            ),
            max_q_rows=rows,
            max_width=self._topk_tokens,
            softmax_scale=self.scale,
            dtype=torch.bfloat16,
            kv_dtype=self._kv_dtype,
            head_dim=self._q_head_dim,
            v_head_dim=self.kv_lora_rank,
            mode="extend" if mode == "ckv_extend" else mode,
            max_batch=rows,
            max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
            page_size=self._kernel_page_size,
            return_lse=self.need_to_return_lse_for_decode,
            lse_scale="natural",
        )
        if self._model_type is not None:
            caps_kwargs["model_type"] = self._model_type
        if self._uses_nvfp4_cache:
            caps_kwargs.update(_nvfp4_run_options(is_glm_next=self._is_glm_next))
        caps = self._module.Caps(**caps_kwargs)
        return caps, self._module.plan(caps)

    def _preparation_prefix(self) -> str:
        return f"attention.sparse_mla.{id(self):x}"

    def _request_name(self, key: tuple[str, int]) -> str:
        mode, rows = key
        return f"{self._preparation_prefix()}.{mode}.m{rows}"

    def _plan(self, key: tuple[str, int]):
        """The plan for key, declared on first use with its default configuration."""
        plan = self._plans.get(key)
        if plan is None:
            caps, plan = self._declare_plan(*key)
            self._plans[key] = plan
            self._plan_caps[key] = caps
        return plan

    def _make_prepare_call(self, state, caps):
        from b12x.preparation import PreparedCall

        if self._bound_kv_cache is None or self._bound_kv_cache.numel() == 0:
            raise PreparationResourceUnavailableError(
                "sparse MLA benchmark requires the published KV cache"
            )
        rows = int(caps.max_q_rows)
        q = torch.empty(
            (rows, int(caps.num_q_heads), int(caps.head_dim)),
            dtype=caps.dtype,
            device=caps.device,
        )
        # Selection reads the real cache.  Page zero is a legal physical page
        # and avoids a synthetic cache or a pool-sized checkpoint.
        kv_cache = self._bound_kv_cache
        selected = torch.zeros(
            (rows, int(caps.max_width)), dtype=torch.int32, device=caps.device
        )
        cache_lengths = torch.full(
            (int(caps.max_batch),),
            int(caps.page_size),
            dtype=torch.int32,
            device=caps.device,
        )
        selected_lengths = torch.ones(rows, dtype=torch.int32, device=caps.device)
        (scratch_spec,) = state.scratch_specs()
        # Preparation calls execute serially. Borrow caller-owned scratch so
        # exact-row plans do not each allocate a workspace.
        workspace = current_workspace_manager()
        workspace.reserve_all(((self._scratch_nbytes,), torch.uint8))
        (scratch,) = workspace.get_simultaneous(
            (scratch_spec.shape, scratch_spec.dtype)
        )
        binding = state.bind(
            scratch=scratch,
            q=q,
            selected_indices=selected,
            cache_seqlens_int32=cache_lengths,
            nsa_cache_seqlens_int32=selected_lengths,
            kv_cache=kv_cache,
        )
        # Launcher construction belongs to the declaration's preparation
        # callback: runtime only consumes these retained launchers.
        state.prime(binding, kv_cache=kv_cache)

        def produce():
            q.fill_(1)

        return PreparedCall(
            run=lambda: state.run(binding, kv_cache=kv_cache),
            produce=produce,
            # Closures retain probes until preparation has synchronized. The
            # published launcher binds live inputs and must not pin these probes.
        )

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if layer is not self:
            raise ValueError("sparse MLA preparation owner mismatch")
        if self._bound_kv_cache is None or self._bound_kv_cache.numel() == 0:
            return ()
        requests = []
        # Decode launches are exact in their row count, so every planned token
        # count that can route to decode gets its own plan.
        for rows in workload.token_counts:
            key = ("decode", rows)
            if rows <= self._decode_max_rows and key not in self._plans:
                self._plan_caps[key], self._plans[key] = self._declare_plan(*key)
        for key, declaration in self._plans.items():
            caps = self._plan_caps[key]
            requests.append(
                declaration.request(
                    name=self._request_name(key),
                    prepare_call=lambda state, caps=caps: self._make_prepare_call(
                        state, caps
                    ),
                    benchmark_call=lambda state, caps=caps: self._make_prepare_call(
                        state, caps
                    ),
                )
            )
        if self._is_glm_next:
            from b12x.preparation import PreparedCall

            kv_probe = torch.empty(
                (self._max_tokens, self.kv_lora_rank),
                dtype=torch.bfloat16,
                device=self._bound_kv_cache.device,
            )
            ignored_slots = torch.full(
                (self._max_tokens,),
                -1,
                dtype=torch.int64,
                device=self._bound_kv_cache.device,
            )
            writer_plan = self._module.plan_cache_writer(
                kv_probe, self._bound_kv_cache, ignored_slots
            )
            self._cache_writer_plan = writer_plan

            def prepare_writer(state):
                probe = torch.empty_like(kv_probe)
                # The writer touches only slot zero; retain that single page
                # for restoration instead of cloning the serving KV pool.
                slots = torch.zeros_like(ignored_slots)
                saved_page = state.kv_cache[0].clone()

                def produce():
                    probe.fill_(1)

                def restore():
                    state.kv_cache[0].copy_(saved_page)

                return PreparedCall(
                    run=lambda: state.run(probe, state.kv_cache, slots),
                    produce=produce,
                    restore=restore,
                    owners=(probe, slots, saved_page),
                )

            requests.append(
                writer_plan.request(
                    name=f"{self._preparation_prefix()}.cache_writer",
                    prepare_call=prepare_writer,
                    benchmark_call=prepare_writer,
                    dependencies=tuple(request.name for request in requests),
                )
            )
        provider = getattr(self, "_physical_selection_provider", None)
        selection_request = getattr(
            provider, "get_b12x_physical_selection_preparation_request", None
        )
        if selection_request is not None and self.dcp_world_size == 1:
            requests.append(selection_request())
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="sparse MLA",
                key=self._preparation_prefix(),
                requests=tuple(requests),
                stage="state",
            ),
        )

    def _use_decode_execution(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if attn_metadata.max_query_len <= 1:
            return True
        return (
            attn_metadata.is_spec_decode
            and attn_metadata.max_query_len <= self._max_speculative_decode_query_len
            and num_tokens
            <= attn_metadata.num_reqs * self._max_speculative_decode_query_len
            and num_tokens <= self._decode_max_rows
        )

    def _workspace_specs(
        self,
        *,
        input_num_heads: int,
        include_ckv: bool,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        q_spec = (
            (self._max_tokens, input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        # Full-CKV attention has only local query heads and does not gather
        # queries. Its cache receive buffer need not coexist with global-head
        # query-gather scratch or the decode plan's split-K intermediates.
        scratch_nbytes = (
            int(self._plan(("ckv_extend", self._max_tokens)).scratch_specs()[0].nbytes)
            if include_ckv
            else self._scratch_nbytes
        )
        scratch_spec = ((scratch_nbytes,), torch.uint8)
        ckv_specs = (
            (
                (
                    self.dcp_world_size * self._ckv_local_capacity,
                    self._cache_record_bytes,
                ),
                torch.uint8,
            ),
        )
        return (
            q_spec,
            scratch_spec,
            *(ckv_specs if include_ckv else ()),
        )

    def _borrow_workspaces(
        self,
        *,
        input_num_heads: int | None = None,
        include_ckv: bool = False,
    ) -> list[torch.Tensor]:
        input_num_heads = (
            self._input_num_heads if input_num_heads is None else input_num_heads
        )
        specs = self._workspace_specs(
            input_num_heads=input_num_heads,
            include_ckv=include_ckv,
        )
        return current_workspace_manager().get_simultaneous(*specs)

    def supports_fused_mla_query_output(
        self,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> bool:
        return bool(
            self.dcp_world_size == 1
            and output_dtype == torch.bfloat16
            and num_heads == self._input_num_heads
            and self._q_head_dim == 576
        )

    def reduce_scatter_dcp_output(self, output: torch.Tensor) -> torch.Tensor | None:
        """Reuse consumed query storage for the prefill output collective.

        The corrected attention output remains in disjoint kernel scratch.
        Pack it into the query buffer in head-major order, then let NCCL write
        each rank's reduction at its in-place receive offset. The resulting
        head-major matrices stay live until the following value projection.
        Decode and symmetric-memory transports retain their existing path.
        """
        if not self._is_glm_next or output.shape[0] <= self._decode_max_rows:
            return None
        group = get_dcp_group()
        communicator = getattr(group, "device_communicator", None)
        comm = getattr(communicator, "pynccl_comm", None)
        if comm is None or comm.disabled or should_nccl_symm_mem_ag_rs():
            return None

        rows, heads, head_dim = output.shape
        q_buffer = self._borrow_workspaces()[0]
        assert output.dtype == q_buffer.dtype
        packed = q_buffer.view(-1)[: output.numel()].view(heads, rows, head_dim)
        packed.copy_(output.transpose(0, 1))
        local_heads = heads // self.dcp_world_size
        local = packed.narrow(0, group.rank_in_group * local_heads, local_heads)
        comm.reduce_scatter(local, packed)
        return local.transpose(0, 1)

    def gather_dcp_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather GLM prefill heads into caller-owned attention query storage.

        The rank-major receive view borrows kernel scratch only until the
        layout copy completes on the execution stream. The attention call
        rebinds that scratch while retaining the final query at the same address.
        Decode and transport-specific all-gathers keep their established path.
        """
        group = get_dcp_group()
        if not self._is_glm_next or query.shape[0] <= self._decode_max_rows:
            return group.all_gather(query, dim=1)
        communicator = getattr(group, "device_communicator", None)
        b12x_comm = getattr(communicator, "b12x_ar_comm", None)
        if (
            b12x_comm is not None
            and not b12x_comm.disabled
            and getattr(b12x_comm, "should_all_gather", None) is not None
            and b12x_comm.should_all_gather(query, 1)
        ):
            return group.all_gather(query, dim=1)

        rows, local_heads, head_dim = query.shape
        q_buffer, scratch = self._borrow_workspaces()[:2]
        output = q_buffer[:rows]
        receive = scratch[: output.numel() * output.element_size()].view(query.dtype)
        receive = receive.view(self.dcp_world_size, rows, local_heads, head_dim)
        local = receive[group.rank_in_group]
        local.copy_(query)
        _dcp_all_gather_current_stream(group, local.view(-1), receive.view(-1))
        output.view(rows, self.dcp_world_size, local_heads, head_dim).copy_(
            receive.movedim(0, 1)
        )
        return output

    def get_fused_mla_query_output(
        self,
        num_tokens: int,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            not self.supports_fused_mla_query_output(num_heads, output_dtype)
            or num_tokens <= 0
            or num_tokens > self._max_tokens
        ):
            return None
        q_buffer = self._borrow_workspaces()[0]
        output = q_buffer[:num_tokens, :num_heads]
        if not output.is_contiguous():
            raise RuntimeError("B12X fused MLA query output must be contiguous.")
        return output

    def _plan_key(
        self, attn_metadata: B12xMLASparseMetadata, num_tokens: int
    ) -> tuple[str, int]:
        """The route and declared row count serving one batch."""
        if self.uses_full_ckv_dcp(attn_metadata, num_tokens):
            return ("ckv_extend", self._max_tokens)
        if self._use_decode_execution(attn_metadata, num_tokens):
            return ("decode", num_tokens)
        return ("extend", self._max_tokens)

    def finalize_kv_cache_geometry(self, kernel_page_size: int) -> None:
        """Finalize kernel plans and workspace memory before KV profiling.

        GLM5Next hybrid cache alignment resolves the physical page size after
        model construction. Full-CKV gather workspace depends on that value,
        so every execution slot must be sized while the memory profiler can
        still subtract the allocation from the KV-cache budget.

        Args:
            kernel_page_size: Resolved physical KV-cache page size in tokens.

        Raises:
            RuntimeError: If an established page size is changed.
        """
        if not self._is_glm_next:
            return
        if self._kernel_page_size_finalized:
            if kernel_page_size != self._kernel_page_size:
                raise RuntimeError(
                    "B12X GLM5Next KV-cache page size is immutable after "
                    f"finalization: {self._kernel_page_size} != {kernel_page_size}."
                )
            return
        self._set_kernel_page_size(kernel_page_size)
        self._kernel_page_size_finalized = True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self._uses_glm_dsa_nvfp4_cache and (
            kv_cache.ndim != 3
            or int(kv_cache.shape[-1]) != _GLM_DSA_NVFP4_CACHE_RECORD_BYTES
            or kv_cache.dtype != torch.uint8
        ):
            raise ValueError(
                "B12X GLM DSA NVFP4 cache must have shape "
                f"[pages, page_size, {_GLM_DSA_NVFP4_CACHE_RECORD_BYTES}] uint8"
            )
        if self._is_glm_next:
            if (
                kv_cache.ndim != 3
                or int(kv_cache.shape[-1]) != self._cache_record_bytes
            ):
                raise ValueError(
                    "B12X GLM5Next cache must have shape "
                    f"[pages, page_size, {self._cache_record_bytes}], got "
                    f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}, "
                    f"dtype={kv_cache.dtype}"
                )
            cache_page_size = int(kv_cache.shape[1])
            if self._kernel_page_size_finalized:
                if cache_page_size != self._kernel_page_size:
                    raise RuntimeError(
                        "B12X GLM5Next bound cache does not match the finalized "
                        f"page size: {cache_page_size} != {self._kernel_page_size}."
                    )
            elif self._ckv_gather_enabled:
                raise RuntimeError(
                    "B12X GLM5Next full-CKV gather requires page geometry "
                    "finalization before KV-cache memory profiling."
                )
            else:
                self._set_kernel_page_size(cache_page_size)
        # The cache writer plan is bound to the outgoing generation's tensor;
        # get_b12x_preparation_units declares a fresh one for the new pool.
        self._cache_writer_plan = None
        self._bound_kv_cache = kv_cache

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self._uses_glm_dsa_nvfp4_cache:
            if kv_cache.numel() == 0:
                return
            assert self._concat_and_cache_nvfp4_mla_fp8_rope is not None
            self._concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                k_scale,
            )
            return
        if not self._is_glm_next:
            return super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
            )
        if kv_cache.numel() == 0:
            return
        plan = self._cache_writer_plan
        if plan is None:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix()} has no declared cache_writer plan"
            )
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "B12X GLM5Next cache updates require a zero-width RoPE tensor, "
                f"got shape={tuple(k_pe.shape)}."
            )
        assert self._concat_and_cache_glm_next_mla is not None
        self._concat_and_cache_glm_next_mla(
            kv_c_normed, kv_cache, slot_mapping.flatten(), plan=plan
        )

    def uses_full_ckv_dcp(
        self, attn_metadata: B12xMLASparseMetadata, num_tokens: int
    ) -> bool:
        if torch.cuda.is_current_stream_capturing():
            return False
        return (
            self._ckv_gather_enabled
            and self._kernel_page_size_finalized
            and attn_metadata.dcp_ckv_gather_eligible
            and attn_metadata.num_decode_tokens == 0
            and num_tokens == attn_metadata.num_actual_tokens
            and 0 < attn_metadata.dcp_padded_total_tokens <= self._ckv_local_capacity
            and attn_metadata.dcp_local_total_tokens
            <= attn_metadata.dcp_padded_total_tokens
            and all(
                value is not None
                for value in (
                    attn_metadata.ckv_selected_indices,
                    attn_metadata.ckv_active_counts,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    attn_metadata.dcp_local_cu_seq_lens,
                    attn_metadata.global_cache_seq_lens_per_req,
                )
            )
        )

    def _gather_full_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        gathered_buffer: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_full_ckv_dcp(attn_metadata, attn_metadata.num_actual_tokens):
            raise RuntimeError("full CKV gather called for an ineligible batch")
        if not _is_glm_next_ckv_source_layout(
            kv_cache,
            page_size=self._kernel_page_size,
            record_bytes=self._cache_record_bytes,
        ):
            raise ValueError(
                "GLM5Next CKV gather requires native "
                f"{self._cache_record_bytes}-byte records; "
                f"got shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}"
            )
        expected_gathered_shape = (
            self.dcp_world_size * self._ckv_local_capacity,
            self._cache_record_bytes,
        )
        if tuple(gathered_buffer.shape) != expected_gathered_shape:
            raise RuntimeError("CKV gathered workspace has an invalid shape")

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        local_tokens = attn_metadata.dcp_local_total_tokens
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        group = get_dcp_group()
        # NCCL in-place all-gather reads each rank's contribution from its
        # receive slice. The stride is this call's padded count, not capacity.
        local_buffer = gathered_buffer.narrow(
            0, group.rank_in_group * padded_tokens, padded_tokens
        )
        if local_tokens:
            ops.cp_gather_cache(
                src_cache=kv_cache,
                dst=local_buffer[:local_tokens],
                block_table=attn_metadata.block_table,
                cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                batch_size=attn_metadata.num_reqs,
            )
        if local_tokens < padded_tokens:
            local_buffer[local_tokens:padded_tokens].zero_()
        _dcp_all_gather_current_stream(
            group,
            local_buffer[:padded_tokens].view(-1),
            gathered_buffer[: self.dcp_world_size * padded_tokens].view(-1),
        )
        return gathered_buffer.view(
            -1, self._kernel_page_size, self._cache_record_bytes
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])
        metadata_page_size = int(attn_metadata.block_size)
        if self._is_glm_next and (
            cache_page_size != self._kernel_page_size
            or metadata_page_size != self._kernel_page_size
        ):
            raise RuntimeError(
                "B12X GLM5Next page geometry does not match the bound plan: "
                f"cache={cache_page_size}, metadata={metadata_page_size}, "
                f"plan={self._kernel_page_size}"
            )
        num_tokens = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        plan_key = self._plan_key(attn_metadata, num_tokens)
        plan = self._plan(plan_key)
        use_ckv_gather = plan_key[0] == "ckv_extend"
        if use_ckv_gather:
            logger.info_once("Using full-CKV gather for GLM5Next B12X DCP prefill")
        input_num_heads = self.num_heads if use_ckv_gather else self._input_num_heads
        workspace_specs = self._workspace_specs(
            input_num_heads=input_num_heads,
            include_ckv=use_ckv_gather,
        )
        workspaces = current_workspace_manager().get_simultaneous(*workspace_specs)
        q_buffer = workspaces[0]
        scratch = workspaces[1]

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q_all = q_buffer[:num_tokens]
            if int(q_pe.shape[-1]) == 0:
                q_all.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_all)
        else:
            q_all = q_buffer[:num_tokens]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr() == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q)

        if int(q_all.shape[1]) != input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        kv_cache_for_run = kv_c_and_k_pe_cache
        if use_ckv_gather:
            gathered_buffer = workspaces[2]
            kv_cache_for_run = self._gather_full_ckv(
                kv_c_and_k_pe_cache,
                attn_metadata,
                gathered_buffer,
            )
            assert attn_metadata.ckv_selected_indices is not None
            assert attn_metadata.ckv_active_counts is not None
            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            selected_indices = attn_metadata.ckv_selected_indices[
                :num_tokens, : topk_indices.shape[1]
            ]
            active_counts = attn_metadata.ckv_active_counts[:num_tokens]
            _map_global_topk_to_gathered_ckv(
                attn_metadata.req_id_per_token[:num_tokens],
                topk_indices,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                selected_indices,
                active_counts,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
            )
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            cache_seq_lens = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_tokens,
            ).contiguous()
            torch.minimum(active_counts, cache_seq_lens, out=active_counts)
            _mask_page_table_after_nsa_len(selected_indices, active_counts)
        else:
            physical_selection = (
                self._physical_selection_provider.get_b12x_physical_selection(
                    num_tokens=num_tokens,
                    num_prefills=int(attn_metadata.num_prefills),
                    num_decode_tokens=int(attn_metadata.num_decode_tokens),
                )
                if self._physical_selection_provider is not None
                else None
            )
            if physical_selection is not None:
                selected_indices, active_counts = physical_selection
            elif self.dcp_world_size > 1:
                block_stride_rows = _selected_index_block_stride_rows(
                    kv_c_and_k_pe_cache,
                    block_size=attn_metadata.block_size,
                )
                selected_indices, active_counts = triton_filter_and_convert_dcp_index(
                    attn_metadata.req_id_per_token[:num_tokens],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    BLOCK_SIZE=attn_metadata.block_size,
                    BLOCK_STRIDE_ROWS=block_stride_rows,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                )
            elif not self._is_glm_next:
                selected_indices = topk_indices
                cache_seq_lens_per_token = attn_metadata.cache_seq_lens_per_token
                assert cache_seq_lens_per_token is not None
                active_counts = cache_seq_lens_per_token[:num_tokens]
            else:
                block_stride_rows = _selected_index_block_stride_rows(
                    kv_c_and_k_pe_cache,
                    block_size=attn_metadata.block_size,
                )
                selected_indices, active_counts = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_tokens],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        BLOCK_STRIDE_ROWS=block_stride_rows,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        compact_valid_to_front=False,
                    )
                )

        if not use_ckv_gather:
            if self._is_glm_next:
                cache_seq_lens = attn_metadata.cache_seq_lens_per_token
                assert cache_seq_lens is not None
                cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()
            else:
                cache_seq_lens = attn_metadata.seq_lens[
                    : attn_metadata.num_reqs
                ].contiguous()
        binding = self._bind(
            plan,
            scratch=scratch,
            q=q_all,
            kv_cache=kv_cache_for_run,
            selected_indices=selected_indices,
            cache_lengths=cache_seq_lens,
            selected_lengths=active_counts,
        )
        result = self._run(binding)
        # The extend kernel returns LSE even when decode does not request it.
        output, lse = result if isinstance(result, tuple) else (result, None)
        if self.need_to_return_lse_for_decode:
            assert lse is not None
            return output, lse
        return output, None
