# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x QSA owner for Qwen4Exp on NVIDIA SM12x."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_projection_workspace_sizes,
    get_b12x_projection_workspaces,
    get_b12x_qsa,
    get_b12x_scratch_buffers,
    set_b12x_preparation_provider,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    aux_stream,
    canonicalize_singleton_dim_strides,
    current_stream,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.b12x import (
    B12xPagedAttentionBackend,
    B12xPagedMetadata,
    B12xPagedMetadataBuilder,
)
from vllm.v1.attention.backends.mla.b12x_indexer import _merge_dcp_topk
from vllm.v1.attention.ops.dcp import MLADCPManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import (
    retain_cuda_graph_capture_resource,
    use_preallocated_workspace,
)

from ..common.b12x_qsa_cache import (
    canonical_qsa_rope_positions,
    qsa_compressed_cache_view,
    qsa_logical_positions,
    qsa_padded_page_size_bytes,
)
from ..config import Qwen4ExpTextConfig
from .b12x_indexer_qsa import QSAIndexer

_QSA_COMPRESS_RATIO = 4
_QSA_INDEX_HEAD_DIM = 128
# The supported speculative envelope is zero to four draft tokens. Its raw
# ring is either 4 or 8 rows, so eight is the static manager-page alignment
# that is valid for every supported runtime configuration.
_QSA_MANAGER_BLOCK_ALIGNMENT = 8
_QSA_MAX_SPECULATIVE_TOKENS = 4
_QSA_SPLITTING_OP = "vllm::qwen4_exp_b12x_qsa_with_output"
_QSA_PROJECTED_READ_OP = "vllm::qwen4_exp_b12x_qsa_run_projected"


def _qsa_prefill_context_capacities(
    max_seq_len: int,
    min_seq_len: int,
) -> tuple[int, ...]:
    """Return bounded prefill capacities while retaining the configured limit."""
    if max_seq_len <= 0 or min_seq_len <= 0:
        raise ValueError("QSA sequence-length capacities must be positive")
    capacities: list[int] = []
    capacity = 1 << (min_seq_len - 1).bit_length()
    while capacity < max_seq_len:
        capacities.append(capacity)
        capacity *= 2
    capacities.append(max_seq_len)
    return tuple(capacities)


def _dcp_local_length(length: int, size: int, rank: int, interleave: int) -> int:
    round_width = size * interleave
    full_rounds, remainder = divmod(length, round_width)
    return full_rounds * interleave + min(
        max(remainder - rank * interleave, 0), interleave
    )


def _compressed_dcp_interleave(size: int, interleave: int, ratio: int) -> int:
    if size == 1:
        return 1
    return interleave // ratio


def _qsa_dcp_group_kv_geometry(
    tp_size: int,
    dcp_size: int,
    total_kv_heads: int,
) -> tuple[int, int]:
    projected_kv_heads = max(1, total_kv_heads // tp_size)
    replicas_per_kv_head = max(1, tp_size // total_kv_heads)
    if max(dcp_size, replicas_per_kv_head) % min(dcp_size, replicas_per_kv_head):
        raise ValueError("QSA DCP groups must align with replicated KV heads")
    replicas_in_group = min(dcp_size, replicas_per_kv_head)
    return dcp_size * projected_kv_heads // replicas_in_group, replicas_in_group


def _register_qsa_compilation_context(
    compilation_config: Any,
    layer_name: str,
    layer: nn.Module,
) -> None:
    static_context = compilation_config.static_forward_context
    if layer_name in static_context:
        raise ValueError(f"duplicate layer name: {layer_name}")
    static_context[layer_name] = layer
    splitting_ops = compilation_config.splitting_ops
    if splitting_ops is not None:
        for op in (_QSA_SPLITTING_OP, _QSA_PROJECTED_READ_OP):
            if op not in splitting_ops:
                splitting_ops.append(op)


def _without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


@dataclass
class Qwen4ExpQSAMetadata(B12xPagedMetadata):
    """Main-cache metadata plus persistent selector-state ownership."""

    request_ids: torch.Tensor | None = None
    is_prefilling: torch.Tensor | None = None
    qsa_state_slot_ids: torch.Tensor | None = None
    qsa_state_is_fresh: torch.Tensor | None = None
    qsa_num_accepted_tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class _StagedQSAMetadata:
    """Caller-owned request metadata shared only within one model forward."""

    request_ids: torch.Tensor
    logical_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    state_slot_ids: torch.Tensor
    state_is_fresh: torch.Tensor
    num_accepted_tokens: torch.Tensor
    query_start_loc: torch.Tensor
    is_prefilling: torch.Tensor
    num_requests: int


@dataclass
class _QSAContextPlan:
    """Immutable declaration metadata and generation-owned preparation buffers."""

    max_seq_len: int
    caps: Any
    plan: Any
    main_block_table: torch.Tensor | None = None
    compressed_block_table: torch.Tensor | None = None
    prepared_plan: Any | None = None


class Qwen4ExpQSAMetadataBuilder(B12xPagedMetadataBuilder):
    """Build QSA metadata without introducing another KV-cache owner."""

    requires_qsa_metadata: ClassVar[bool] = True
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        scheduler = vllm_config.scheduler_config
        max_tokens = int(scheduler.max_num_batched_tokens)
        max_reqs = int(scheduler.max_num_seqs)
        self.max_speculative_tokens = int(vllm_config.num_speculative_tokens)
        self._request_ids = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self._capture_state_slot_ids = torch.arange(
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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        qsa_state_slot_ids: torch.Tensor | None = None,
        qsa_state_is_fresh: torch.Tensor | None = None,
        qsa_num_accepted_tokens: torch.Tensor | None = None,
        qsa_is_prefilling: torch.Tensor | None = None,
    ) -> Qwen4ExpQSAMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        num_reqs = int(cm.seq_lens.shape[0])
        if qsa_state_slot_ids is not None:
            self._capture_state_slot_ids[:num_reqs].copy_(qsa_state_slot_ids[:num_reqs])
        if qsa_state_is_fresh is not None:
            self._capture_state_is_fresh[:num_reqs].copy_(qsa_state_is_fresh[:num_reqs])
        if qsa_num_accepted_tokens is not None:
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                qsa_num_accepted_tokens[:num_reqs]
            )
        qsa_state_slot_ids = self._capture_state_slot_ids[:num_reqs]
        qsa_state_is_fresh = self._capture_state_is_fresh[:num_reqs]
        qsa_num_accepted_tokens = self._capture_num_accepted_tokens[:num_reqs]
        is_prefilling = qsa_is_prefilling
        if is_prefilling is None:
            is_prefilling = cm.is_prefilling
        if is_prefilling is None:
            raise RuntimeError(
                "Qwen3.8 Flash Next QSA requires explicit per-request "
                "is_prefilling metadata"
            )
        self._capture_is_prefilling[:num_reqs].copy_(is_prefilling[:num_reqs])
        is_prefilling = self._capture_is_prefilling[:num_reqs]
        request_ids = cm.token_to_req_indices(self._request_ids)
        num_mapped_tokens = int(cm.query_start_loc_cpu[-1])
        if num_mapped_tokens < cm.num_actual_tokens:
            request_ids[num_mapped_tokens:].fill_(-1)
        return Qwen4ExpQSAMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            causal=cast(bool, cm.causal),
            request_ids=request_ids,
            is_prefilling=is_prefilling,
            qsa_state_slot_ids=qsa_state_slot_ids,
            qsa_state_is_fresh=qsa_state_is_fresh,
            qsa_num_accepted_tokens=qsa_num_accepted_tokens,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xPagedMetadata,
    ) -> None:
        qsa_metadata = cast(Qwen4ExpQSAMetadata, metadata)
        if qsa_metadata.qsa_num_accepted_tokens is None:
            raise RuntimeError(
                "QSA draft decode metadata requires accepted-token counts"
            )
        qsa_metadata.qsa_num_accepted_tokens.fill_(1)

    def update_block_table(
        self,
        metadata: B12xPagedMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Qwen4ExpQSAMetadata:
        if not isinstance(metadata, Qwen4ExpQSAMetadata):
            raise TypeError(f"expected QSA metadata, got {type(metadata)!r}")
        # build_attn_metadata applies this ownership rule during capture too;
        # replay therefore reads the first builder's live request-state buffers.
        return replace(
            metadata,
            block_table=blk_table,
            slot_mapping=slot_mapping,
        )


class Qwen4ExpQSABackend(B12xPagedAttentionBackend):
    """Sparse QSA backend using one padded, unsplit BLHNC manager page."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_B12X_QSA"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        packed = B12xPagedAttentionBackend.customize_spec(spec)
        return replace(
            packed,
            page_size_padded=qsa_padded_page_size_bytes(
                packed,
                compress_ratio=_QSA_COMPRESS_RATIO,
                index_head_dim=_QSA_INDEX_HEAD_DIM,
            ),
        )

    @classmethod
    def customize_hybrid_kv_cache_spec(
        cls,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
    ) -> AttentionSpec:
        parallel = vllm_config.parallel_config
        dcp_size = parallel.decode_context_parallel_size
        if dcp_size > 1:
            num_kv_heads, _ = _qsa_dcp_group_kv_geometry(
                parallel.tensor_parallel_size,
                dcp_size,
                vllm_config.model_config.get_total_num_kv_heads(),
            )
            spec = replace(
                spec,
                num_kv_heads=num_kv_heads,
            )
        return cls.customize_spec(spec)

    @classmethod
    def get_impl_cls(cls) -> type[Qwen4ExpQSAImpl]:
        return Qwen4ExpQSAImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(_QSA_MANAGER_BLOCK_ALIGNMENT)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) % _QSA_MANAGER_BLOCK_ALIGNMENT == 0

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return (
            (int(default_block_size) + _QSA_MANAGER_BLOCK_ALIGNMENT - 1)
            // _QSA_MANAGER_BLOCK_ALIGNMENT
            * _QSA_MANAGER_BLOCK_ALIGNMENT
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [256]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Block-copy copies page padding only when the raw allocation is block
        # outer. LBHNC would copy the logical main view and lose selector tails.
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        del capability
        return True

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
        del use_sparse, device_capability
        if dtype != torch.bfloat16:
            return "Qwen4Exp QSA requires BF16 queries"
        if kv_cache_dtype not in (None, "auto", "bfloat16", "fp8", "fp8_e4m3"):
            return "Qwen4Exp QSA requires BF16 or FP8 E4M3 KV cache"
        if use_mla or has_sink or use_mm_prefix:
            return "QSA does not support MLA, attention sinks, or MM-prefix attention"
        if not cls.supports_block_size(block_size):
            return (
                "QSA manager block size must be a multiple of "
                f"{_QSA_MANAGER_BLOCK_ALIGNMENT}"
            )
        qsa = get_b12x_qsa()
        if qsa is None:
            return "Install the b12x backend with `pip install vllm[b12x]`"
        return None


class Qwen4ExpQSAImpl(AttentionImpl[Qwen4ExpQSAMetadata]):
    """Native vLLM main-K/V writer for the merged QSA owner."""

    is_sparse: ClassVar[bool] = True
    supports_dense_mha_prefill: ClassVar[bool] = False
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if alibi_slopes is not None or sliding_window is not None or sinks is not None:
            raise NotImplementedError(
                "QSA does not support ALiBi, sliding window, or sinks"
            )
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("QSA does not support logits soft cap")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("QSA supports causal decoder attention only")
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError("QSA requires BF16 or FP8 E4M3 main KV cache")
        if head_size != 256 or num_heads % num_kv_heads:
            raise ValueError("QSA requires head_dim=256 and valid grouped-query heads")
        if not math.isclose(scale, head_size**-0.5, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("QSA requires canonical head_dim**-0.5 scaling")
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.output_head_size = self.head_size
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.supports_quant_query_input = False

    def _kv_cache_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = key_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        value_cache = value_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        expected_dtype = (
            current_platform.fp8_dtype()
            if self.kv_cache_dtype
            in (
                "fp8",
                "fp8_e4m3",
            )
            else torch.bfloat16
        )
        if key_cache.dtype == torch.uint8 and expected_dtype != torch.bfloat16:
            key_cache = key_cache.view(expected_dtype)
        if value_cache.dtype == torch.uint8 and expected_dtype != torch.bfloat16:
            value_cache = value_cache.view(expected_dtype)
        if key_cache.dtype != expected_dtype or value_cache.dtype != expected_dtype:
            raise TypeError(
                f"QSA main K/V cache views must be {expected_dtype}, got "
                f"{key_cache.dtype}/{value_cache.dtype}"
            )
        return key_cache, value_cache

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            *self._kv_cache_views(kv_cache),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Qwen4ExpQSAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        raise RuntimeError("QSA must run through its merged write-before-read owner")


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact scalar/MRoPE composition to QSA."""

    num_tokens, _, head_dim = tensor.shape
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        shape = tensor.shape
        rotated, _ = triton_mrope(
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.is_neox_style,
        )
        return rotated.reshape(shape)
    rotated = rotary_emb.apply_rotary_emb.forward_cuda(
        tensor[..., :rotary_dim], cos, sin
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


@triton.jit(do_not_specialize=["num_reqs"])
def _reset_fresh_qsa_state_kernel(
    state_slot_ids,
    state_is_fresh,
    sequence_lengths,
    query_start_loc,
    num_accepted_tokens,
    raw_k_ring,
    raw_logical_positions,
    raw_rope_positions,
    raw_interval_start_positions,
    raw_k_slot_stride,
    raw_k_ring_stride,
    raw_tag_slot_stride,
    raw_rope_slot_stride,
    raw_rope_ring_stride,
    raw_interval_stride,
    num_reqs,
    MAX_STATE_SLOTS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    POSITION_AXES: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = request < num_reqs
    fresh = tl.load(state_is_fresh + request, mask=active, other=0)
    state_slot = tl.load(state_slot_ids + request, mask=active, other=-1).to(tl.int64)
    query_start = tl.load(query_start_loc + request, mask=active, other=0)
    query_end = tl.load(query_start_loc + request + 1, mask=active, other=0)
    valid = (
        active
        & fresh
        & (query_end > query_start)
        & (state_slot >= 0)
        & (state_slot < MAX_STATE_SLOTS)
    )

    key_rows = offsets // INDEX_HEAD_DIM
    key_dims = offsets % INDEX_HEAD_DIM
    key_mask = valid & (key_rows < RING_CAPACITY)
    tl.store(
        raw_k_ring
        + state_slot * raw_k_slot_stride
        + key_rows * raw_k_ring_stride
        + key_dims,
        0.0,
        mask=key_mask,
    )
    tag_mask = valid & (offsets < RING_CAPACITY)
    tl.store(
        raw_logical_positions + state_slot * raw_tag_slot_stride + offsets,
        -1,
        mask=tag_mask,
    )
    rope_rows = offsets // POSITION_AXES
    rope_axes = offsets % POSITION_AXES
    rope_mask = valid & (rope_rows < RING_CAPACITY)
    tl.store(
        raw_rope_positions
        + state_slot * raw_rope_slot_stride
        + rope_rows * raw_rope_ring_stride
        + rope_axes,
        -1,
        mask=rope_mask,
    )
    if tl.program_id(1) == 0:
        sequence_length = tl.load(sequence_lengths + request, mask=valid, other=0)
        accepted = tl.load(num_accepted_tokens + request, mask=valid, other=1)
        first_position = sequence_length - (query_end - query_start)
        anchor = tl.where(query_end > query_start, first_position - accepted, -1)
        tl.store(
            raw_interval_start_positions + state_slot * raw_interval_stride,
            anchor.to(tl.int64),
            mask=valid,
        )


@triton.jit
def _stage_qsa_request_state_kernel(
    source_state_slot_ids,
    source_state_is_fresh,
    source_num_accepted_tokens,
    query_start_loc,
    state_slot_ids,
    state_is_fresh,
    num_accepted_tokens,
    num_reqs,
) -> None:
    request = tl.program_id(0)
    in_range = request < num_reqs
    query_start = tl.load(query_start_loc + request, mask=in_range, other=0)
    query_end = tl.load(query_start_loc + request + 1, mask=in_range, other=0)
    active = in_range & (query_end > query_start)
    state_slot = tl.load(source_state_slot_ids + request, mask=active, other=-1)
    fresh = tl.load(source_state_is_fresh + request, mask=active, other=0)
    accepted = tl.load(source_num_accepted_tokens + request, mask=active, other=0)
    tl.store(state_slot_ids + request, tl.where(active, state_slot, -1))
    tl.store(state_is_fresh + request, tl.where(active, fresh, 0))
    tl.store(num_accepted_tokens + request, tl.where(active, accepted, 0))


@triton.jit
def _stage_qsa_rope_positions_kernel(
    source,
    request_ids,
    output,
    source_row_stride,
    source_axis_stride,
    output_row_stride,
    rows,
    POSITION_AXES: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    axes = tl.arange(0, 4)
    axis_mask = axes < POSITION_AXES
    active = row < rows
    request_id = tl.load(request_ids + row, mask=active, other=-1)
    positions = tl.load(
        source + row * source_row_stride + axes * source_axis_stride,
        mask=active & axis_mask,
        other=-1,
    )
    positions = tl.where(request_id >= 0, positions, -1)
    tl.store(
        output + row * output_row_stride + axes,
        positions,
        mask=active & axis_mask,
    )


@triton.jit
def _pack_dcp_qkv_kernel(
    query,
    key,
    value,
    packed,
    query_row_stride,
    key_row_stride,
    value_row_stride,
    rows,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    row = tl.program_id(1)
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM
    is_query = head < NUM_QUERY_HEADS
    is_key = (head >= NUM_QUERY_HEADS) & (head < NUM_QUERY_HEADS + NUM_KV_HEADS)
    kv_head = head - NUM_QUERY_HEADS
    value_head = kv_head - NUM_KV_HEADS
    data = tl.load(
        query + row * query_row_stride + head * HEAD_DIM + dims,
        mask=is_query & dim_mask,
        other=0.0,
    )
    data += tl.load(
        key + row * key_row_stride + kv_head * HEAD_DIM + dims,
        mask=is_key & dim_mask,
        other=0.0,
    )
    data += tl.load(
        value + row * value_row_stride + value_head * HEAD_DIM + dims,
        mask=~(is_query | is_key) & dim_mask,
        other=0.0,
    )
    tl.store(
        packed + (head * rows + row) * HEAD_DIM + dims,
        data,
        mask=dim_mask,
    )


@triton.jit
def _unpack_dcp_qkv_kernel(
    gathered,
    output,
    rows,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_PROJECTED_KV_HEADS: tl.constexpr,
    TOTAL_QUERY_HEADS: tl.constexpr,
    TOTAL_KV_HEADS: tl.constexpr,
    KV_REPLICAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> None:
    output_head = tl.program_id(0)
    row = tl.program_id(1)
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM
    local_heads = NUM_QUERY_HEADS + 2 * NUM_PROJECTED_KV_HEADS
    is_query = output_head < TOTAL_QUERY_HEADS
    is_key = (output_head >= TOTAL_QUERY_HEADS) & (
        output_head < TOTAL_QUERY_HEADS + TOTAL_KV_HEADS
    )

    query_rank = output_head // NUM_QUERY_HEADS
    query_head = output_head % NUM_QUERY_HEADS
    kv_head = output_head - TOTAL_QUERY_HEADS
    value_head = kv_head - TOTAL_KV_HEADS
    selected_kv_head = tl.where(is_key, kv_head, value_head) * KV_REPLICAS
    kv_rank = selected_kv_head // NUM_PROJECTED_KV_HEADS
    local_kv_head = selected_kv_head % NUM_PROJECTED_KV_HEADS
    source_head = tl.where(
        is_query,
        query_rank * local_heads + query_head,
        kv_rank * local_heads
        + NUM_QUERY_HEADS
        + local_kv_head
        + tl.where(is_key, 0, NUM_PROJECTED_KV_HEADS),
    )
    data = tl.load(
        gathered + (source_head * rows + row) * HEAD_DIM + dims,
        mask=dim_mask,
    )

    query_elements = rows * TOTAL_QUERY_HEADS * HEAD_DIM
    key_elements = rows * TOTAL_KV_HEADS * HEAD_DIM
    query_offset = (row * TOTAL_QUERY_HEADS + output_head) * HEAD_DIM
    key_offset = query_elements + (row * TOTAL_KV_HEADS + kv_head) * HEAD_DIM
    value_offset = (
        query_elements + key_elements + (row * TOTAL_KV_HEADS + value_head) * HEAD_DIM
    )
    output_offset = tl.where(
        is_query,
        query_offset,
        tl.where(is_key, key_offset, value_offset),
    )
    tl.store(output + output_offset + dims, data, mask=dim_mask)


class Qwen4ExpQSAAttention(nn.Module, AttentionLayerBase):
    """Merged main attention, selector state, and b12x QSA transaction."""

    supports_dcp = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("QSA currently requires BF16 activations")
        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError("QSA requires BF16 or FP8 E4M3 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("QSA does not support KV-cache quantization")
        parallel = vllm_config.parallel_config
        if parallel.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "QSA does not support prefill context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("QSA requires causal decoder attention")
        if getattr(config, "dual_chunk_attention_config", None) is not None:
            raise NotImplementedError("QSA does not support dual-chunk RoPE")
        if (
            int(config.indexer_compress_ratio) != _QSA_COMPRESS_RATIO
            or int(config.indexer_head_dim) != _QSA_INDEX_HEAD_DIM
        ):
            raise NotImplementedError(
                "The SM12x QSA integration requires compress_ratio=4 and "
                "index_head_dim=128"
            )

        self.config = config
        self.layer_id = int(layer_id)
        self.overlap_input_projections = envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP
        if self.overlap_input_projections:
            aux_stream()
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.dcp_size = int(parallel.decode_context_parallel_size)
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.cp_kv_cache_interleave_size = int(parallel.cp_kv_cache_interleave_size)
        if self.dcp_size > 1:
            if self.cp_kv_cache_interleave_size % _QSA_COMPRESS_RATIO:
                raise ValueError(
                    "QSA DCP requires cp_kv_cache_interleave_size divisible by "
                    f"{_QSA_COMPRESS_RATIO}"
                )
            self.overlap_input_projections = False
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA query heads must divide across TP ranks")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must divide across TP ranks")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must divide replicated QSA KV heads")
        self.num_projected_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.dcp_num_heads = self.num_heads * self.dcp_size
        self.dcp_num_kv_heads, self.dcp_kv_replicas = _qsa_dcp_group_kv_geometry(
            tp_size,
            self.dcp_size,
            self.total_num_kv_heads,
        )
        self.num_kv_heads = (
            self.dcp_num_kv_heads if self.dcp_size > 1 else self.num_projected_kv_heads
        )
        self.head_dim = int(config.head_dim)
        if self.head_dim != 256:
            raise NotImplementedError("The SM12x QSA integration requires head_dim=256")
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_projected_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=False,
            quant_config=_without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        self.kv_cache_kernel_dtype = (
            current_platform.fp8_dtype()
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3")
            else self.kv_cache_torch_dtype
        )
        if self.kv_cache_kernel_dtype not in (
            torch.bfloat16,
            current_platform.fp8_dtype(),
        ):
            raise NotImplementedError(
                "QSA cache storage must resolve to BF16 or FP8 E4M3"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSABackend
        self.impl = Qwen4ExpQSAImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )

        scheduler = vllm_config.scheduler_config
        self.max_tokens = int(scheduler.max_num_batched_tokens)
        self.max_seqs = int(scheduler.max_num_seqs)
        self.max_seq_len = int(model_config.max_model_len)
        self._qsa_model_config = model_config
        self._qsa_cache_config = vllm_config.cache_config
        self.max_speculative_tokens = int(vllm_config.num_speculative_tokens)
        if self.max_speculative_tokens > _QSA_MAX_SPECULATIVE_TOKENS:
            raise NotImplementedError(
                "QSA currently supports at most four speculative tokens"
            )
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.budget = int(config.indexer_budget)
        self.index_heads = int(config.indexer_n_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.position_axes = 3 if model_config.uses_mrope else 1
        self.raw_ring_capacity = self.compress_ratio * math.ceil(
            (self.compress_ratio + self.max_speculative_tokens) / self.compress_ratio
        )
        self.max_decode_rows = self.max_seqs * (1 + self.max_speculative_tokens)
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.device = device
        self.dcp_manager = (
            MLADCPManager(
                vllm_config=vllm_config,
                device=device,
                num_heads=self.num_heads,
                query_head_dim=self.head_dim,
                output_head_dim=self.head_dim,
                query_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                padded_num_heads=None,
                is_lse_base_on_e=True,
                use_pcp=False,
            )
            if self.dcp_size > 1
            else None
        )
        self._selector_done = (
            torch.cuda.Event() if self.overlap_input_projections else None
        )
        self._index_ready = (
            torch.cuda.Event() if self.overlap_input_projections else None
        )
        if self._selector_done is not None:
            self._selector_done.record(aux_stream())
        self._selector_index_inputs: tuple[torch.Tensor, ...] | None = None
        self._selector_metadata_views: _StagedQSAMetadata | None = None
        self._native_selection_width = self.budget + self.compress_ratio - 1
        self._share_mtp_indices = bool(
            self.layer_id >= int(config.num_hidden_layers)
            and config.index_share_for_mtp_iteration
            and self.max_speculative_tokens > 1
        )
        self.skip_topk = False
        self._mtp_anchor_state: Any | None = None
        self._mtp_anchor_storage: torch.Tensor | None
        self.register_buffer("_mtp_anchor_storage", None, persistent=False)
        if self._share_mtp_indices:
            self.register_buffer(
                "_mtp_source_rows",
                torch.full((self.max_seqs,), -1, dtype=torch.int64, device=device),
                persistent=False,
            )

        self.register_buffer(
            "_raw_k_ring",
            torch.zeros(
                self.max_seqs,
                self.raw_ring_capacity,
                self.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_logical_positions",
            torch.full(
                (self.max_seqs, self.raw_ring_capacity),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_rope_positions",
            torch.full(
                (self.max_seqs, self.raw_ring_capacity, self.position_axes),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_positions",
            torch.full((self.max_seqs,), -1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_snapshot",
            torch.empty(self.max_seqs, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_qsa_output",
            torch.empty(
                self.max_tokens,
                self.dcp_num_heads if self.dcp_size > 1 else self.num_heads,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rope_position_input",
            torch.empty(
                self.max_tokens,
                self.position_axes,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_selected_positions",
            torch.empty(
                self.max_tokens,
                self.budget + self.compress_ratio - 1,
                dtype=torch.int32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_sequence_lengths",
            torch.zeros(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_state_slot_ids",
            torch.full((self.max_seqs,), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_state_is_fresh",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_accepted_tokens",
            torch.ones(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_is_prefilling",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self._main_block_table: torch.Tensor | None = None
        self._compressed_cache: torch.Tensor | None = None
        self._qsa_decode_context: _QSAContextPlan | None = None
        self._qsa_prefill_bindings: tuple[_QSAContextPlan, ...] = ()
        self._b12x_diagnostic_request_ids: torch.Tensor | None = None
        self._b12x_preparation_prefix = self.layer_name
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)
        _register_qsa_compilation_context(
            vllm_config.compilation_config,
            self.layer_name,
            self,
        )

    def _qsa_request_name(self, mode: str, context: _QSAContextPlan, rows: int) -> str:
        return (
            f"{self._b12x_preparation_prefix}.qsa.{mode}."
            f"context{context.max_seq_len}.m{rows}"
        )

    def _qsa_caps(self, **geometry):
        sections = None
        if self.position_axes == 3:
            sections = getattr(self.rotary_emb, "mrope_section", None)
            if sections is None:
                raise RuntimeError("QSA M-RoPE requires mrope_section")
            sections = tuple(map(int, sections))
        api = get_b12x_qsa()
        assert api is not None
        return api.Caps(
            max_batch=self.max_seqs,
            max_raw_state_slots=self.max_seqs,
            max_speculative_tokens=self.max_speculative_tokens,
            q_heads=(self.dcp_num_heads if self.dcp_size > 1 else self.num_heads),
            kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            index_heads=self.index_heads,
            index_kv_heads=1,
            index_head_dim=self.index_head_dim,
            index_rotary_dim=int(self.rotary_emb.rotary_dim),
            compress_ratio=self.compress_ratio,
            budget=self.budget,
            position_axes=self.position_axes,
            mrope_sections=sections,
            mrope_interleaved=bool(
                getattr(self.rotary_emb, "mrope_interleaved", False)
            ),
            rms_norm_eps=float(self.indexer.q_layernorm.variance_epsilon),
            dtype=torch.bfloat16,
            kv_dtype=self.kv_cache_kernel_dtype,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            **geometry,
        )

    def _ensure_qsa_context_buffers(self, context):
        """Materialize only after the registry admitted the complete envelope."""
        caps = context.caps
        if self._selector_done is not None and self._selector_done.device is None:
            with torch.accelerator.device_index(caps.device.index):
                self._selector_done.record(current_stream())
        if context.main_block_table is None:
            context.main_block_table = torch.full(
                (caps.max_batch, caps.main_table_width),
                -1,
                dtype=torch.int32,
                device=caps.device,
            )
            context.compressed_block_table = (
                context.main_block_table
                if caps.main_table_width == caps.compressed_table_width
                else torch.full(
                    (caps.max_batch, caps.compressed_table_width),
                    -1,
                    dtype=torch.int32,
                    device=caps.device,
                )
            )
        if context is self._qsa_prefill_bindings[-1]:
            self._main_block_table = context.main_block_table
        if self._share_mtp_indices and self._mtp_anchor_storage is None:
            full_caps = self._qsa_prefill_bindings[-1].caps
            api = get_b12x_qsa()
            assert api is not None
            layout = api.DraftSelectionPlan(
                caps.device,
                full_caps.max_q_rows,
                full_caps.selection_width,
            )
            (spec,) = layout.storage_specs()
            self._mtp_anchor_storage = torch.empty(
                spec.shape, dtype=spec.dtype, device=spec.device
            )
            self._mtp_anchor_state = layout.bind(storage=self._mtp_anchor_storage)
            self._mtp_anchor_state.reset()
            self._mtp_source_rows.fill_(-1)

    def get_b12x_preparation_units(
        self,
        layer: torch.nn.Module,
        workload: B12xWorkload,
    ) -> tuple[B12xPreparationUnit, ...]:
        from b12x._lib.scratch import scratch_buffer_spec
        from b12x.preparation import MemoryRequirements, PersistentMemory

        if layer is not self:
            raise ValueError("QSA preparation owner mismatch")
        if self._qsa_decode_context is None or not self._qsa_prefill_bindings:
            return ()
        if max(workload.token_counts) > self.max_tokens:
            raise ValueError("QSA workload exceeds declared token capacity")
        contexts = (
            *(("prefill", context) for context in self._qsa_prefill_bindings),
            ("decode", self._qsa_decode_context),
        )
        requests = []
        # QSA specializes on planned capacity, not exact live M. Each original
        # context's native compiler covers its dynamic decode/prefill row paths.
        for mode, context in contexts:

            def memory(config, detected, context=context):
                native = context.plan._memory_requirements(config, detected)
                caps = context.caps
                widths = (
                    (caps.main_table_width,)
                    if caps.main_table_width == caps.compressed_table_width
                    else (
                        caps.main_table_width,
                        caps.compressed_table_width,
                    )
                )
                required = caps.max_batch * sum(widths) * torch.int32.itemsize
                resident = 0
                if context.main_block_table is not None:
                    resident = context.main_block_table.numel() * torch.int32.itemsize
                    if context.compressed_block_table is not context.main_block_table:
                        resident += (
                            context.compressed_block_table.numel()
                            * torch.int32.itemsize
                        )
                persistent = [
                    PersistentMemory(
                        key=(self, "qsa_tables", id(context)),
                        required_nbytes=required,
                        resident_nbytes=resident,
                    )
                ]
                if (
                    self._share_mtp_indices
                    and context is self._qsa_prefill_bindings[-1]
                ):
                    api = get_b12x_qsa()
                    assert api is not None
                    layout = api.DraftSelectionPlan(
                        caps.device,
                        caps.max_q_rows,
                        caps.selection_width,
                    )
                    required_anchor = sum(
                        spec.nbytes for spec in layout.storage_specs()
                    )
                    resident_anchor = (
                        0
                        if self._mtp_anchor_storage is None
                        else self._mtp_anchor_storage.numel()
                    )
                    persistent.append(
                        PersistentMemory(
                            key=(self, "qsa_anchor"),
                            required_nbytes=required_anchor,
                            resident_nbytes=resident_anchor,
                        )
                    )
                scratch = native.scratch
                if (
                    self.overlap_input_projections
                    and context is self._qsa_decode_context
                ):
                    sizes = get_b12x_projection_workspace_sizes(
                        caps.max_q_rows,
                        self.qkv_proj,
                        self.indexer.index_qk_proj,
                    )
                    scratch += tuple(
                        scratch_buffer_spec(name, nbytes=size, device=caps.device)
                        for name, size in zip(
                            ("qkv_projection", "index_projection"), sizes
                        )
                    )
                return MemoryRequirements(
                    scratch=scratch,
                    persistent=(*native.persistent, *persistent),
                )

            declaration = replace(context.plan, _memory_requirements=memory)
            context.prepared_plan = declaration
            requests.append(
                declaration.request(
                    name=self._qsa_request_name(mode, context, context.caps.max_q_rows),
                    prepare_call=self._qsa_prepare_call(context),
                    benchmark_call=self._qsa_prepare_call(context),
                )
            )
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="QSA",
                key=self._b12x_preparation_prefix,
                requests=tuple(requests),
                stage="state",
            ),
        )

    def _qsa_prepare_call(self, context: _QSAContextPlan):
        from b12x.preparation import PreparedCall

        def prepare(state):
            caps, device = state.caps, state.caps.device
            self._ensure_qsa_context_buffers(context)
            compressed_cache = self._compressed_cache
            main_block_table = context.main_block_table
            compressed_block_table = context.compressed_block_table
            assert compressed_cache is not None
            assert main_block_table is not None
            assert compressed_block_table is not None
            impl = cast(Qwen4ExpQSAImpl, self.impl)
            main_k, main_v = impl._kv_cache_views(self.kv_cache)
            rows = min(
                caps.max_q_rows,
                caps.max_seq_len,
                main_k.shape[0] * caps.main_page_size,
                compressed_cache.shape[0]
                * caps.compressed_page_size
                * caps.compress_ratio,
            )
            if context is self._qsa_decode_context:
                rows = min(rows, caps.max_speculative_tokens + 1)
            if rows < 1:
                raise PreparationResourceUnavailableError(
                    "QSA primer requires a live cache page"
                )
            local_rows = _dcp_local_length(
                rows,
                caps.dcp_size,
                caps.dcp_rank,
                caps.cp_kv_cache_interleave_size,
            )
            local_groups = _dcp_local_length(
                rows // caps.compress_ratio,
                caps.dcp_size,
                caps.dcp_rank,
                _compressed_dcp_interleave(
                    caps.dcp_size,
                    caps.cp_kv_cache_interleave_size,
                    caps.compress_ratio,
                ),
            )
            main_pages = triton.cdiv(local_rows, caps.main_page_size)
            compressed_pages = triton.cdiv(local_groups, caps.compressed_page_size)
            output = torch.empty(
                (rows, caps.q_heads, caps.head_dim), dtype=torch.bfloat16, device=device
            )
            query = torch.empty_like(output)
            source = torch.randn_like(query)
            projection = torch.empty(
                (rows, (caps.index_heads + caps.index_kv_heads) * caps.index_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            projection_source = torch.randn_like(projection)
            index_query = projection[
                :, : caps.index_heads * caps.index_head_dim
            ].unflatten(
                -1,
                (caps.index_heads, caps.index_head_dim),
            )
            raw_index_key = projection[:, caps.index_heads * caps.index_head_dim :]
            request_ids = torch.zeros(rows, dtype=torch.int32, device=device)
            positions = torch.arange(rows, dtype=torch.int64, device=device)
            rope_positions = (
                positions[:, None].expand(rows, caps.position_axes).contiguous()
            )
            sequence_lengths = torch.zeros(
                caps.max_batch, dtype=torch.int32, device=device
            )
            sequence_lengths[0] = rows
            starts = torch.full(
                (caps.max_batch + 1,), rows, dtype=torch.int32, device=device
            )
            starts[0] = 0
            slots = torch.full_like(self._raw_state_slot_ids, -1)
            slots[0] = 0
            accepted = torch.zeros(caps.max_batch, dtype=torch.int32, device=device)
            accepted[0] = 1
            prefilling = torch.zeros(caps.max_batch, dtype=torch.bool, device=device)
            prefilling[0] = context is not self._qsa_decode_context
            staged = _StagedQSAMetadata(
                request_ids=request_ids,
                logical_positions=positions,
                sequence_lengths=sequence_lengths,
                state_slot_ids=slots,
                state_is_fresh=torch.ones(
                    caps.max_batch, dtype=torch.bool, device=device
                ),
                num_accepted_tokens=accepted,
                query_start_loc=starts,
                is_prefilling=prefilling,
                num_requests=1,
            )
            binding = self._bind_qsa_context(context, staged, output, state=state)
            targets = [
                self.kv_cache[:main_pages],
                self._raw_k_ring[:1],
                compressed_cache[:compressed_pages],
                self._raw_logical_positions[:1],
                self._raw_rope_positions[:1],
                self._raw_interval_start_positions[:1],
                main_block_table,
                self._selected_positions[:rows],
            ]
            if compressed_block_table is not main_block_table:
                targets.append(compressed_block_table)
            if self._mtp_anchor_storage is not None:
                targets.append(self._mtp_anchor_storage)
            snapshots = tuple((target, target.clone()) for target in targets)
            key_source = (
                torch.randn(main_k[:main_pages].shape, device=device)
                .mul_(0.25)
                .to(main_k.dtype)
            )
            value_source = (
                torch.randn(main_v[:main_pages].shape, device=device)
                .mul_(0.25)
                .to(main_v.dtype)
            )
            main_ids = torch.arange(main_pages, dtype=torch.int32, device=device)
            compressed_ids = torch.arange(
                compressed_pages, dtype=torch.int32, device=device
            )

            def reset():
                main_k[:main_pages].copy_(key_source)
                main_v[:main_pages].copy_(value_source)
                compressed_cache[:compressed_pages].zero_()
                self._raw_k_ring[:1].zero_()
                self._raw_logical_positions[:1].fill_(-1)
                self._raw_rope_positions[:1].fill_(-1)
                self._raw_interval_start_positions[:1].fill_(-1)
                main_block_table.fill_(-1)
                main_block_table[0, :main_pages].copy_(main_ids)
                if compressed_block_table is not main_block_table:
                    compressed_block_table.fill_(-1)
                compressed_block_table[0, :compressed_pages].copy_(compressed_ids)
                output.fill_(float("nan"))
                if self._mtp_anchor_state is not None:
                    self._mtp_anchor_state.reset()

            def produce():
                query.copy_(source)
                projection.copy_(projection_source)

            def restore():
                from b12x.preparation.types import _close_all

                _close_all(
                    (lambda target=target, saved=saved: target.copy_(saved))
                    for target, saved in snapshots
                )

            return PreparedCall(
                run=lambda: state.run_for_preparation(
                    binding,
                    query=query,
                    index_query=index_query,
                    raw_index_key=raw_index_key,
                    request_ids=request_ids,
                    query_positions=positions,
                    rope_positions=rope_positions,
                    sequence_lengths=sequence_lengths,
                    query_start_loc=starts,
                    num_accepted_tokens=accepted,
                    is_prefilling=prefilling,
                ),
                output=output,
                produce=produce,
                reset=reset,
                restore=restore,
            )

        return prepare

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        base = FullAttentionSpec(
            block_size=int(vllm_config.cache_config.block_size),
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        return self.attn_backend.customize_spec(base)

    def snapshot_speculative_interval_starts(self) -> None:
        self._raw_interval_start_snapshot.copy_(self._raw_interval_start_positions)

    def get_recurrent_checkpoint_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self._raw_k_ring,
            self._raw_logical_positions,
            self._raw_rope_positions,
            self._raw_interval_start_positions,
        )

    def restore_speculative_interval_starts(self) -> None:
        self._raw_interval_start_positions.copy_(self._raw_interval_start_snapshot)

    def set_recurrent_checkpoint_anchor(
        self, slot: int, anchor: torch.Tensor | int
    ) -> None:
        self._raw_interval_start_positions[slot] = anchor

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_gate, key, value = qkv.split(
            (self.q_size * 2, self.kv_size, self.kv_size), dim=-1
        )
        q_gate = q_gate.unflatten(-1, (self.num_heads, self.head_dim * 2))
        query, gate = q_gate.chunk(2, dim=-1)
        query = self.q_norm(query).flatten(-2)
        key = self.k_norm(
            key.unflatten(-1, (self.num_projected_kv_heads, self.head_dim))
        ).flatten(-2)
        query, key = self.rotary_emb(positions, query, key)
        return query, key, value, gate.flatten(-2)

    def _combine_dcp_output(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        staged: _StagedQSAMetadata,
    ) -> torch.Tensor:
        if partial_output.shape[0] <= self.max_decode_rows:
            from vllm.v1.attention.ops.dcp import mask_dcp_empty_shards_

            mask_dcp_empty_shards_(
                partial_lse,
                staged.sequence_lengths,
                staged.query_start_loc,
            )
            group = get_dcp_group()
            lses = group.all_gather(partial_lse.contiguous(), dim=0).view(
                self.dcp_size, *partial_lse.shape
            )
            lses = torch.where(
                torch.isfinite(lses), lses, lses.new_full((), float("-inf"))
            )
            global_lse = torch.logsumexp(lses, dim=0)
            weights = torch.exp(partial_lse - global_lse)
            weights = torch.where(torch.isfinite(weights), weights, 0.0)
            corrected = partial_output.float() * weights.unsqueeze(-1)
            combined = group.reduce_scatter(
                corrected.transpose(0, 1).contiguous(), dim=0
            )
            return combined.transpose(0, 1).to(partial_output.dtype)
        assert self.dcp_manager is not None
        return self.dcp_manager.combine(
            partial_output,
            partial_lse,
            seq_lens=staged.sequence_lengths,
            query_start_loc=staged.query_start_loc,
        )

    def _gather_dcp_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = get_dcp_group().all_gather(
            tensor.transpose(0, 1).contiguous(), dim=0
        )
        return gathered.transpose(0, 1).contiguous()

    def _gather_dcp_qkv_fused_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.flatten(1)
        key = key.flatten(1)
        value = value.flatten(1)
        rows = int(query.shape[0])
        local_heads = self.num_heads + 2 * self.num_projected_kv_heads
        packed = torch.empty(
            (local_heads, rows, self.head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        block_dim = triton.next_power_of_2(self.head_dim)
        _pack_dcp_qkv_kernel[(local_heads, rows)](
            query,
            key,
            value,
            packed,
            query.stride(0),
            key.stride(0),
            value.stride(0),
            rows,
            NUM_QUERY_HEADS=self.num_heads,
            NUM_KV_HEADS=self.num_projected_kv_heads,
            HEAD_DIM=self.head_dim,
            BLOCK_DIM=block_dim,
            num_warps=4,
        )
        gathered = get_dcp_group().all_gather(packed, dim=0)
        output_heads = self.dcp_num_heads + 2 * self.dcp_num_kv_heads
        output = torch.empty(
            rows * output_heads * self.head_dim,
            dtype=query.dtype,
            device=query.device,
        )
        _unpack_dcp_qkv_kernel[(output_heads, rows)](
            gathered,
            output,
            rows,
            NUM_QUERY_HEADS=self.num_heads,
            NUM_PROJECTED_KV_HEADS=self.num_projected_kv_heads,
            TOTAL_QUERY_HEADS=self.dcp_num_heads,
            TOTAL_KV_HEADS=self.dcp_num_kv_heads,
            KV_REPLICAS=self.dcp_kv_replicas,
            HEAD_DIM=self.head_dim,
            BLOCK_DIM=block_dim,
            num_warps=4,
        )
        query_elements = rows * self.dcp_num_heads * self.head_dim
        kv_elements = rows * self.dcp_num_kv_heads * self.head_dim
        query = output[:query_elements].view(rows, self.dcp_num_heads, self.head_dim)
        key = output[query_elements : query_elements + kv_elements].view(
            rows, self.dcp_num_kv_heads, self.head_dim
        )
        value = output[query_elements + kv_elements :].view_as(key)
        return query, key, value

    def _gather_dcp_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather Q/K/V and remove replicated KV heads."""

        if query.is_cuda and query.shape[0] <= self.max_decode_rows:
            return self._gather_dcp_qkv_fused_cuda(query, key, value)
        if not query.is_cuda:
            local_heads = self.num_heads + 2 * self.num_projected_kv_heads
            packed = torch.cat((query, key, value), dim=1)
            gathered = self._gather_dcp_heads(packed).view(
                query.shape[0], self.dcp_size, local_heads, self.head_dim
            )
            query = (
                gathered[:, :, : self.num_heads]
                .contiguous()
                .view(query.shape[0], self.dcp_num_heads, self.head_dim)
            )
            kv_start = self.num_heads
            key = (
                gathered[:, :, kv_start : kv_start + self.num_projected_kv_heads]
                .contiguous()
                .view(query.shape[0], -1, self.head_dim)
            )
            value = (
                gathered[:, :, kv_start + self.num_projected_kv_heads :]
                .contiguous()
                .view(query.shape[0], -1, self.head_dim)
            )
        else:
            query = self._gather_dcp_heads(query)
            key = self._gather_dcp_heads(key)
            value = self._gather_dcp_heads(value)
        if key.ndim == 2:
            key = key.view(
                query.shape[0],
                self.dcp_size * self.num_projected_kv_heads,
                self.head_dim,
            )
            value = value.view_as(key)
        return (
            query,
            key[:, :: self.dcp_kv_replicas].contiguous(),
            value[:, :: self.dcp_kv_replicas].contiguous(),
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        self.max_seq_len = int(self._qsa_model_config.max_model_len)
        if kv_cache.ndim != 4 or int(kv_cache.shape[1]) != 2:
            raise ValueError(
                "QSA requires a native B12x [pages,2,page,packed_kv] cache"
            )
        main_page_size = int(kv_cache.shape[2])
        if main_page_size % self.raw_ring_capacity:
            raise ValueError(
                "QSA manager block must be divisible by raw_ring_capacity: "
                f"{main_page_size} % {self.raw_ring_capacity} != 0"
            )
        if main_page_size % _QSA_MANAGER_BLOCK_ALIGNMENT:
            raise ValueError("QSA manager block does not satisfy the backend alignment")
        if self.max_decode_rows > self.max_tokens:
            raise ValueError(
                "QSA max_num_batched_tokens must cover the largest verifier batch"
            )

        impl = cast(Qwen4ExpQSAImpl, self.impl)
        main_k_cache, main_v_cache = impl._kv_cache_views(kv_cache)
        compressed_cache = qsa_compressed_cache_view(
            kv_cache,
            compress_ratio=self.compress_ratio,
            index_head_dim=self.index_head_dim,
        )
        self._compressed_cache = compressed_cache
        compressed_page_size = int(compressed_cache.shape[1])
        qsa_max_seq_len = (
            (self.max_seq_len + self.compress_ratio - 1)
            // self.compress_ratio
            * self.compress_ratio
        )
        dcp_round = self.dcp_size * self.cp_kv_cache_interleave_size
        full_rounds, remainder = divmod(qsa_max_seq_len, dcp_round)
        local_max_seq_len = full_rounds * self.cp_kv_cache_interleave_size + min(
            max(
                remainder - self.dcp_rank * self.cp_kv_cache_interleave_size,
                0,
            ),
            self.cp_kv_cache_interleave_size,
        )
        table_width = math.ceil(local_max_seq_len / main_page_size)
        compressed_table_width = math.ceil(
            (local_max_seq_len // self.compress_ratio) / compressed_page_size
        )
        if compressed_table_width != table_width:
            raise RuntimeError(
                "QSA packed main/compressed caches require one shared table width"
            )
        # CUDA-graph memory profiling binds a deliberately small throwaway
        # cache. Caps describe the maximum cache page counts admitted by the
        # plan; bind() separately validates actual tensors against those maxima.
        planned_main_cache_pages = max(int(main_k_cache.shape[0]), table_width)
        planned_compressed_cache_pages = max(
            int(compressed_cache.shape[0]), compressed_table_width
        )
        qsa = get_b12x_qsa()
        if qsa is None or not qsa.is_supported():
            raise RuntimeError("b12x QSA is unavailable on the current device")

        def make_context(*, max_q_rows: int, max_seq_len: int):
            caps = self._qsa_caps(
                device=kv_cache.device,
                max_q_rows=max_q_rows,
                max_seq_len=max_seq_len,
                main_page_size=main_page_size,
                compressed_page_size=compressed_page_size,
                num_main_cache_pages=planned_main_cache_pages,
                num_compressed_cache_pages=planned_compressed_cache_pages,
            )
            projection_row_stride = (
                caps.index_heads + caps.index_kv_heads
            ) * caps.index_head_dim
            declaration = qsa.plan(
                caps,
                invocation=qsa.invocation_from_descriptors(
                    caps,
                    operands={
                        "request_ids": {"dtype": "int32", "strides": (1,)},
                        "rope_positions": {
                            "dtype": "int64",
                            "strides": (caps.position_axes, 1),
                        },
                        "index_query": {
                            "dtype": "bfloat16",
                            "strides": (
                                projection_row_stride,
                                caps.index_head_dim,
                                1,
                            ),
                        },
                        "raw_index_key": {
                            "dtype": "bfloat16",
                            "strides": (projection_row_stride, 1),
                        },
                        "main_k_cache": {
                            "dtype": str(main_k_cache.dtype).removeprefix("torch."),
                            "strides": tuple(main_k_cache.stride()),
                        },
                        "main_v_cache": {
                            "dtype": str(main_v_cache.dtype).removeprefix("torch."),
                            "strides": tuple(main_v_cache.stride()),
                        },
                        "main_block_table": {
                            "dtype": "int32",
                            "strides": (caps.main_table_width, 1),
                        },
                        "compressed_k_cache": {
                            "dtype": str(compressed_cache.dtype).removeprefix("torch."),
                            "strides": tuple(compressed_cache.stride()),
                        },
                        "compressed_block_table": {
                            "dtype": "int32",
                            "strides": (caps.compressed_table_width, 1),
                        },
                        **{
                            name: {
                                "dtype": str(tensor.dtype).removeprefix("torch."),
                                "strides": tuple(tensor.stride()),
                            }
                            for name, tensor in (
                                ("raw_k_ring", self._raw_k_ring),
                                ("raw_logical_positions", self._raw_logical_positions),
                                ("raw_rope_positions", self._raw_rope_positions),
                                (
                                    "raw_interval_start_positions",
                                    self._raw_interval_start_positions,
                                ),
                                ("raw_state_slot_ids", self._raw_state_slot_ids),
                                (
                                    "index_q_norm_weight",
                                    self.indexer.q_layernorm.weight,
                                ),
                                (
                                    "index_k_norm_weight",
                                    self.indexer.k_layernorm.weight,
                                ),
                                (
                                    "rope_cos",
                                    self.rotary_emb.cos_sin_cache.chunk(2, dim=-1)[0],
                                ),
                                (
                                    "rope_sin",
                                    self.rotary_emb.cos_sin_cache.chunk(2, dim=-1)[1],
                                ),
                            )
                        },
                    },
                ),
            )
            return _QSAContextPlan(max_seq_len=max_seq_len, caps=caps, plan=declaration)

        prefill_contexts = tuple(
            make_context(max_q_rows=self.max_tokens, max_seq_len=capacity)
            for capacity in _qsa_prefill_context_capacities(
                qsa_max_seq_len,
                min(qsa_max_seq_len, max(self.max_tokens, self.budget)),
            )
        )
        decode_context = make_context(
            max_q_rows=self.max_decode_rows,
            max_seq_len=qsa_max_seq_len,
        )

        cos_sin = self.rotary_emb.cos_sin_cache
        rope_cos, rope_sin = cos_sin.chunk(2, dim=-1)
        expected_rope_width = int(self.rotary_emb.rotary_dim) // 2
        if (
            int(rope_cos.shape[1]) != expected_rope_width
            or rope_sin.shape != rope_cos.shape
        ):
            raise RuntimeError("QSA received an unexpected main RoPE cache layout")
        self._qsa_decode_context = decode_context
        self._qsa_prefill_bindings = prefill_contexts
        self._main_block_table = None

    def _bind_qsa_context(
        self,
        context: _QSAContextPlan,
        staged: _StagedQSAMetadata | None = None,
        output: torch.Tensor | None = None,
        *,
        overlap: bool = False,
        state: Any | None = None,
    ):
        impl = cast(Qwen4ExpQSAImpl, self.impl)
        main_k_cache, main_v_cache = impl._kv_cache_views(self.kv_cache)
        rope_cos, rope_sin = self.rotary_emb.cos_sin_cache.chunk(2, dim=-1)
        rows = context.caps.max_q_rows if output is None else int(output.shape[0])
        if state is not None:
            self._ensure_qsa_context_buffers(context)
            binder = state.bind_for_preparation
            # Trial and prepare factories own their scratch; the runtime
            # binding below draws from the workspace manager instead.
            specs = tuple(state.scratch_specs())
            if len(specs) != 1:
                raise RuntimeError(
                    "QSA preparation requires exactly one scratch buffer"
                )
            scratch = torch.empty(
                specs[0].shape, dtype=specs[0].dtype, device=specs[0].device
            )
            max_q_rows = state.caps.max_q_rows
        else:
            plan = context.prepared_plan
            if plan is None:
                raise PreparationResourceUnavailableError(
                    f"{self._b12x_preparation_prefix} lacks prepared QSA context "
                    f"{context.max_seq_len}"
                )
            api = get_b12x_qsa()
            assert api is not None
            binder = lambda **kwargs: api.bind(plan, **kwargs)
            scratch = get_b12x_scratch_buffers(plan)[0]
            max_q_rows = context.caps.max_q_rows
        return binder(
            scratch=scratch,
            main_block_table=context.main_block_table,
            compressed_block_table=context.compressed_block_table,
            main_k_cache=main_k_cache,
            main_v_cache=main_v_cache,
            k_descale=self._k_scale,
            v_descale=self._v_scale,
            compressed_k_cache=self._compressed_cache,
            raw_k_ring=self._raw_k_ring,
            raw_logical_positions=self._raw_logical_positions,
            raw_rope_positions=self._raw_rope_positions,
            raw_interval_start_positions=self._raw_interval_start_positions,
            raw_state_slot_ids=(
                self._raw_state_slot_ids if staged is None else staged.state_slot_ids
            ),
            index_q_norm_weight=self.indexer.q_layernorm.weight,
            index_k_norm_weight=self.indexer.k_layernorm.weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            output=self._qsa_output[:rows] if output is None else output,
            selected_positions=self._selected_positions[:max_q_rows],
            selection_stream=aux_stream() if overlap else None,
            selection_done=self._selector_done if overlap else None,
            draft_selection=self._mtp_anchor_state,
        )

    def unbind_kv_cache(self) -> None:
        self._mtp_anchor_state = None
        self._mtp_anchor_storage = None
        self._qsa_decode_context = None
        self._qsa_prefill_bindings = ()
        self._compressed_cache = None
        self._main_block_table = None
        super().unbind_kv_cache()

    @staticmethod
    def _active_positions(positions: torch.Tensor, rows: int) -> torch.Tensor:
        return positions[:rows] if positions.ndim == 1 else positions[:, :rows]

    @staticmethod
    def _require_qsa_metadata(
        metadata: Qwen4ExpQSAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = (
            metadata.request_ids,
            metadata.qsa_state_slot_ids,
            metadata.qsa_state_is_fresh,
            metadata.qsa_num_accepted_tokens,
        )
        if any(tensor is None for tensor in fields):
            raise RuntimeError("QSA runtime metadata is incomplete")
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], fields
        )

    def _reset_fresh_selector_state(
        self,
        *,
        state_slot_ids: torch.Tensor,
        state_is_fresh: torch.Tensor,
        sequence_lengths: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if num_reqs <= 0:
            return
        if not HAS_TRITON:
            raise RuntimeError("QSA state reset requires Triton")
        block = 256
        elements = self.raw_ring_capacity * self.index_head_dim
        _reset_fresh_qsa_state_kernel[(num_reqs, triton.cdiv(elements, block))](
            state_slot_ids,
            state_is_fresh,
            sequence_lengths,
            query_start_loc,
            num_accepted_tokens,
            self._raw_k_ring,
            self._raw_logical_positions,
            self._raw_rope_positions,
            self._raw_interval_start_positions,
            self._raw_k_ring.stride(0),
            self._raw_k_ring.stride(1),
            self._raw_logical_positions.stride(0),
            self._raw_rope_positions.stride(0),
            self._raw_rope_positions.stride(1),
            self._raw_interval_start_positions.stride(0),
            num_reqs,
            MAX_STATE_SLOTS=self.max_seqs,
            RING_CAPACITY=self.raw_ring_capacity,
            INDEX_HEAD_DIM=self.index_head_dim,
            POSITION_AXES=self.position_axes,
            BLOCK=block,
            num_warps=4,
        )

    def _stage_runtime_metadata(
        self,
        metadata: Qwen4ExpQSAMetadata,
        rows: int,
        *,
        row_capacity: int,
        main_block_table: torch.Tensor | None = None,
        compressed_block_table: torch.Tensor | None = None,
    ) -> _StagedQSAMetadata:
        if main_block_table is None:
            main_block_table = self._main_block_table
        if compressed_block_table is None:
            compressed_block_table = main_block_table
        if main_block_table is None or compressed_block_table is None:
            raise RuntimeError("QSA main cache is not bound")
        request_ids, state_slots, state_is_fresh, accepted = self._require_qsa_metadata(
            metadata
        )
        if metadata.is_prefilling is None:
            raise RuntimeError("QSA prefill metadata is missing")
        num_reqs = int(metadata.seq_lens.shape[0])
        if rows > row_capacity or num_reqs > self.max_seqs:
            raise ValueError("QSA batch exceeds its planned capacity")
        table_widths = {
            int(main_block_table.shape[1]),
            int(compressed_block_table.shape[1]),
        }
        if int(metadata.block_table.shape[1]) < max(table_widths):
            raise ValueError("QSA block table is narrower than the planned context")

        block_tables = (
            (main_block_table,)
            if compressed_block_table is main_block_table
            else (main_block_table, compressed_block_table)
        )
        for block_table in block_tables:
            table_width = int(block_table.shape[1])
            block_table.fill_(-1)
            block_table[:num_reqs].copy_(metadata.block_table[:num_reqs, :table_width])

        # Cache groups retain their own page tables. Request metadata can share
        # staging only when all source tensors and capacity contracts match.
        cache = get_forward_context().additional_kwargs.setdefault(
            "qwen38_qsa_staging", {}
        )
        cache_key = (
            rows,
            row_capacity,
            self.max_seqs,
            self.max_speculative_tokens,
            num_reqs,
            request_ids.data_ptr(),
            state_slots.data_ptr(),
            state_is_fresh.data_ptr(),
            accepted.data_ptr(),
            metadata.is_prefilling.data_ptr(),
            metadata.seq_lens.data_ptr(),
            metadata.query_start_loc.data_ptr(),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            if not isinstance(cached, _StagedQSAMetadata):
                raise TypeError("QSA staging cache contains an invalid metadata owner")
            return cached
        self._sequence_lengths.zero_()
        self._sequence_lengths[:num_reqs].copy_(metadata.seq_lens[:num_reqs])
        # Unused request slots end at the live packed prefix, not the padded
        # graph row count. Copy the device boundary so replay sees batch changes.
        self._query_start_loc.copy_(metadata.query_start_loc[num_reqs : num_reqs + 1])
        self._query_start_loc[: num_reqs + 1].copy_(
            metadata.query_start_loc[: num_reqs + 1]
        )
        self._raw_state_slot_ids.fill_(-1)
        self._state_is_fresh.zero_()
        self._num_accepted_tokens.zero_()
        self._is_prefilling.zero_()
        self._is_prefilling[:num_reqs].copy_(metadata.is_prefilling[:num_reqs])
        if num_reqs:
            _stage_qsa_request_state_kernel[(num_reqs,)](
                state_slots,
                state_is_fresh,
                accepted,
                self._query_start_loc,
                self._raw_state_slot_ids,
                self._state_is_fresh,
                self._num_accepted_tokens,
                num_reqs,
                num_warps=1,
            )
        active_request_ids = request_ids[:rows]
        logical_positions = qsa_logical_positions(
            sequence_lengths=metadata.seq_lens[:num_reqs],
            query_start_loc=metadata.query_start_loc[: num_reqs + 1],
            request_ids=active_request_ids,
        )
        staged = _StagedQSAMetadata(
            request_ids=active_request_ids,
            logical_positions=logical_positions,
            sequence_lengths=self._sequence_lengths,
            state_slot_ids=self._raw_state_slot_ids,
            state_is_fresh=self._state_is_fresh,
            num_accepted_tokens=self._num_accepted_tokens,
            query_start_loc=self._query_start_loc,
            is_prefilling=self._is_prefilling,
            num_requests=num_reqs,
        )
        cache[cache_key] = staged
        return staged

    def _prepare_qsa_metadata(
        self,
        metadata: Qwen4ExpQSAMetadata,
        rows: int,
        main_block_table: torch.Tensor,
        compressed_block_table: torch.Tensor,
    ) -> _StagedQSAMetadata:
        staged = self._stage_runtime_metadata(
            metadata,
            rows,
            row_capacity=self.max_tokens,
            main_block_table=main_block_table,
            compressed_block_table=compressed_block_table,
        )
        self._reset_fresh_selector_state(
            state_slot_ids=staged.state_slot_ids,
            state_is_fresh=staged.state_is_fresh,
            sequence_lengths=staged.sequence_lengths,
            query_start_loc=staged.query_start_loc,
            num_accepted_tokens=staged.num_accepted_tokens,
            num_reqs=staged.num_requests,
        )
        return staged

    def _shared_qsa_rope_positions(
        self,
        positions: torch.Tensor,
        request_ids: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        source = canonical_qsa_rope_positions(
            self._active_positions(positions, rows),
            num_rows=rows,
            position_axes=self.position_axes,
        )
        cache = get_forward_context().additional_kwargs.setdefault(
            "qwen38_qsa_rope_staging", {}
        )
        key = (
            source.data_ptr(),
            request_ids.data_ptr(),
            rows,
            self.position_axes,
            *source.stride(),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        _stage_qsa_rope_positions_kernel[(rows,)](
            source,
            request_ids,
            self._rope_position_input,
            source.stride(0),
            source.stride(1),
            self._rope_position_input.stride(0),
            rows,
            POSITION_AXES=self.position_axes,
            num_warps=1,
        )
        staged = self._rope_position_input[:rows]
        cache[key] = staged
        return staged

    def set_skip_topk(self, skip: bool) -> None:
        """Start a recording round or reuse its accepted draft anchors."""
        self.skip_topk = bool(skip and self._share_mtp_indices)
        if self._share_mtp_indices and not self.skip_topk:
            if self._mtp_anchor_state is not None:
                self._mtp_anchor_state.reset()
            self._mtp_source_rows.fill_(-1)

    def compact_topk_indices(self, source_rows: torch.Tensor) -> None:
        """Map compacted requests to their accepted draft-selection anchors."""
        if not self._share_mtp_indices:
            return
        rows = int(source_rows.numel())
        if rows == 0:
            return
        if source_rows.ndim != 1 or rows > self.max_seqs:
            raise ValueError("QSA MTP capture exceeds the request-row capacity")
        if (
            source_rows.device != self._selected_positions.device
            or source_rows.dtype not in (torch.int32, torch.int64)
            or not source_rows.is_contiguous()
        ):
            raise TypeError("QSA MTP source rows must be contiguous CUDA int32/int64")
        self._mtp_source_rows[:rows].copy_(source_rows)
        if rows < self.max_seqs:
            self._mtp_source_rows[rows:].fill_(-1)

    def _run_reused_qsa(self, query, key, value, output) -> None:
        raw = get_forward_context().attn_metadata
        if isinstance(raw, list):
            raw = raw[0]
        if not isinstance(raw, dict):
            output.zero_()
            return
        metadata = raw.get(self.layer_name)
        if not isinstance(metadata, Qwen4ExpQSAMetadata):
            raise TypeError("QSA MTP selected read requires QSA metadata")
        rows = int(metadata.num_actual_tokens)
        if rows == 0:
            output.zero_()
            return
        if (
            not 0
            < rows
            <= min(
                self.max_seqs,
                query.shape[0],
                key.shape[0],
                value.shape[0],
                output.shape[0],
            )
        ):
            raise ValueError("QSA MTP selected read requires one row per request")
        if metadata.max_query_len != 1 or metadata.causal is not True:
            raise ValueError(
                "QSA MTP selection reuse requires causal single-token queries"
            )
        context = self._qsa_binding_for_workload(
            rows=rows, max_seq_len=int(metadata.max_seq_len)
        )
        staged = self._stage_runtime_metadata(
            metadata,
            rows,
            row_capacity=self.max_seqs,
            main_block_table=context.main_block_table,
            compressed_block_table=context.compressed_block_table,
        )
        self._b12x_diagnostic_request_ids = staged.request_ids
        if self.dcp_manager is not None:
            query, key, value = self._gather_dcp_qkv(
                query[:rows], key[:rows], value[:rows]
            )
        impl = cast(Qwen4ExpQSAImpl, self.impl)
        impl.do_kv_cache_update(
            self, key[:rows], value[:rows], self.kv_cache, metadata.slot_mapping[:rows]
        )
        qsa = get_b12x_qsa()
        if qsa is None:
            raise RuntimeError("b12x QSA is unavailable for MTP selection reuse")
        reuse = qsa.DraftSelectionReuse(source_rows=self._mtp_source_rows)
        if self.dcp_manager is None:
            qsa.run(
                self._bind_qsa_context(context, staged, output[:rows]),
                query=query[:rows],
                request_ids=staged.request_ids,
                query_positions=staged.logical_positions,
                reuse=reuse,
            )
        else:
            binding = self._bind_qsa_context(context, staged, self._qsa_output[:rows])
            partial_output, partial_lse = qsa.attend_reuse(
                binding,
                query=query,
                request_ids=staged.request_ids,
                query_positions=staged.logical_positions,
                reuse=reuse,
            )
            combined = self._combine_dcp_output(partial_output, partial_lse, staged)
            output[:rows].copy_(combined)
        if rows < output.shape[0]:
            output[rows:].zero_()

    def _qsa_binding_for_workload(
        self,
        *,
        rows: int,
        max_seq_len: int,
    ) -> _QSAContextPlan:
        if not self._qsa_prefill_bindings:
            raise RuntimeError("b12x QSA was not bound to its cache")
        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"QSA sequence length {max_seq_len} exceeds the configured limit "
                f"{self.max_seq_len}"
            )
        if rows <= self.max_decode_rows:
            if self._qsa_decode_context is None:
                raise RuntimeError("b12x QSA decode capacity was not planned")
            return self._qsa_decode_context
        for context in self._qsa_prefill_bindings:
            if max_seq_len <= context.max_seq_len:
                return context
        raise ValueError(
            f"QSA sequence length {max_seq_len} exceeds the configured limit "
            f"{self.max_seq_len}"
        )

    def _parallel_selector_metadata(
        self,
        projected_rows: int,
    ) -> Qwen4ExpQSAMetadata | None:
        # V2 uses runtime mode NONE inside its outer full-graph capture.
        # PIECEWISE cannot carry a pending selector across the read-op boundary.
        if (
            not self.overlap_input_projections
            or projected_rows > min(16, self.max_decode_rows)
            or not torch.cuda.is_current_stream_capturing()
            or get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        ):
            return None
        raw = get_forward_context().attn_metadata
        if isinstance(raw, list):
            raw = raw[0]
        metadata = raw.get(self.layer_name) if isinstance(raw, dict) else None
        if (
            not isinstance(metadata, Qwen4ExpQSAMetadata)
            or not 0 < metadata.num_actual_tokens <= projected_rows
            or metadata.max_query_len > 1 + self.max_speculative_tokens
        ):
            return None
        return metadata

    def _project_qsa_inputs(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.skip_topk:
            return self.qkv_proj(hidden_states)[0]
        metadata = self._parallel_selector_metadata(int(hidden_states.shape[0]))
        if metadata is None:
            return self.qkv_proj(hidden_states)[0]
        rows = int(metadata.num_actual_tokens)
        context = self._qsa_binding_for_workload(
            rows=rows,
            max_seq_len=int(metadata.max_seq_len),
        )
        if context.prepared_plan is None:
            raise PreparationResourceUnavailableError(
                "QSA projections lack a prepared plan"
            )
        # Selection can start before Q/K/V finishes, so all three need disjoint scratch.
        _, qkv_scratch, index_scratch = get_b12x_scratch_buffers(context.prepared_plan)
        stream = aux_stream()
        assert stream is not None and self._selector_done is not None
        assert self._index_ready is not None
        main_stream = current_stream()
        stream.wait_stream(main_stream)
        hidden_states.record_stream(stream)
        positions.record_stream(stream)
        with torch.cuda.stream(stream):
            staged = self._prepare_qsa_metadata(
                metadata,
                rows,
                context.main_block_table,
                context.compressed_block_table,
            )
            self._selector_metadata_views = staged
            self._b12x_diagnostic_request_ids = staged.request_ids
            rope_positions = self._shared_qsa_rope_positions(
                positions, staged.request_ids, rows
            )
            with use_preallocated_workspace(index_scratch):
                index_query, raw_index_key = self.indexer.project(hidden_states)
            self._selector_index_inputs = (
                index_query[:rows],
                raw_index_key[:rows],
                rope_positions,
            )
            self._index_ready.record(stream)
        # The complete B12X run consumes these inputs and joins its selection
        # stream after the independent main Q/K/V producer has been enqueued.
        with use_preallocated_workspace(qkv_scratch):
            return self.qkv_proj(hidden_states)[0]

    def _run_projected_qsa(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if self.skip_topk:
            self._run_reused_qsa(query, key, value, output)
            return
        metadata = self._parallel_selector_metadata(int(hidden_states.shape[0]))
        if metadata is None:
            index_query, raw_index_key = self.indexer.project(hidden_states)
            self._run_qsa(
                positions, query, key, value, index_query, raw_index_key, output
            )
            return
        rows = int(metadata.num_actual_tokens)
        context = self._qsa_binding_for_workload(
            rows=rows,
            max_seq_len=int(metadata.max_seq_len),
        )
        impl = cast(Qwen4ExpQSAImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key[:rows],
            value[:rows],
            self.kv_cache,
            metadata.slot_mapping[:rows],
        )
        qsa = get_b12x_qsa()
        if qsa is None or metadata.request_ids is None:
            raise RuntimeError("b12x QSA selection metadata is unavailable")
        staged = self._selector_metadata_views
        if staged is None or self._selector_index_inputs is None:
            raise RuntimeError("QSA attention has no matching selector metadata")
        index_query, raw_index_key, rope_positions = self._selector_index_inputs
        qsa.run(
            self._bind_qsa_context(context, staged, output[:rows], overlap=True),
            query=query[:rows],
            index_query=index_query,
            raw_index_key=raw_index_key,
            request_ids=staged.request_ids,
            query_positions=staged.logical_positions,
            rope_positions=rope_positions,
            sequence_lengths=staged.sequence_lengths,
            query_start_loc=staged.query_start_loc,
            num_accepted_tokens=staged.num_accepted_tokens,
            is_prefilling=staged.is_prefilling,
            index_ready=self._index_ready,
        )
        if rows < output.shape[0]:
            output[rows:].zero_()

    def _run_b12x_qsa(
        self,
        *,
        metadata: Qwen4ExpQSAMetadata,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        raw_index_key: torch.Tensor,
        output: torch.Tensor,
        rows: int,
    ) -> None:
        context = self._qsa_binding_for_workload(
            rows=rows,
            max_seq_len=int(metadata.max_seq_len),
        )
        retain_cuda_graph_capture_resource(self)
        retain_cuda_graph_capture_resource(context)
        staged = self._prepare_qsa_metadata(
            metadata,
            rows,
            context.main_block_table,
            context.compressed_block_table,
        )
        # The model-state diagnostic reads this stable staged view after CUDA
        # graph replay, when the custom-op body itself does not execute in Python.
        self._b12x_diagnostic_request_ids = staged.request_ids
        rope_positions = self._shared_qsa_rope_positions(
            positions, staged.request_ids, rows
        )
        if self.dcp_manager is not None:
            query, key, value = self._gather_dcp_qkv(
                query[:rows], key[:rows], value[:rows]
            )
        impl = cast(Qwen4ExpQSAImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key[:rows],
            value[:rows],
            self.kv_cache,
            metadata.slot_mapping[:rows],
        )
        qsa = get_b12x_qsa()
        if qsa is None:
            raise RuntimeError("b12x QSA disappeared after cache binding")
        if self.dcp_manager is None:
            binding = self._bind_qsa_context(context, staged, output[:rows])
            qsa.run(
                binding,
                query=query[:rows],
                index_query=index_query[:rows],
                raw_index_key=raw_index_key[:rows],
                request_ids=staged.request_ids,
                query_positions=staged.logical_positions,
                rope_positions=rope_positions,
                sequence_lengths=staged.sequence_lengths,
                query_start_loc=staged.query_start_loc,
                num_accepted_tokens=staged.num_accepted_tokens,
                is_prefilling=staged.is_prefilling,
            )
        else:
            binding = self._bind_qsa_context(context, staged, self._qsa_output[:rows])
            selection = qsa.select(
                binding,
                query=query,
                index_query=index_query[:rows],
                raw_index_key=raw_index_key[:rows],
                request_ids=staged.request_ids,
                query_positions=staged.logical_positions,
                rope_positions=rope_positions,
                sequence_lengths=staged.sequence_lengths,
                query_start_loc=staged.query_start_loc,
                num_accepted_tokens=staged.num_accepted_tokens,
                is_prefilling=staged.is_prefilling,
            )
            _merge_dcp_topk(
                selection.group_ids,
                selection.scores,
                self.dcp_rank,
                self.dcp_size,
                _compressed_dcp_interleave(
                    self.dcp_size,
                    self.cp_kv_cache_interleave_size,
                    self.compress_ratio,
                ),
            )
            partial_output, partial_lse = qsa.attend(
                binding,
                query=query,
                request_ids=staged.request_ids,
                query_positions=staged.logical_positions,
                selection=selection,
            )
            combined = self._combine_dcp_output(partial_output, partial_lse, staged)
            output[:rows].copy_(combined)
        if rows < output.shape[0]:
            output[rows:].zero_()

    def _run_qsa(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        raw_index_key: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raw_metadata = get_forward_context().attn_metadata
        if isinstance(raw_metadata, list):
            raw_metadata = raw_metadata[0]
        if not isinstance(raw_metadata, dict):
            output.zero_()
            return
        metadata = raw_metadata.get(self.layer_name)
        if not isinstance(metadata, Qwen4ExpQSAMetadata):
            raise TypeError(
                f"{self.layer_name} expected QSA metadata, got "
                f"{type(metadata).__name__}"
            )
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        rows = int(metadata.num_actual_tokens)
        if rows < 0 or rows > min(
            query.shape[0],
            key.shape[0],
            value.shape[0],
            index_query.shape[0],
            raw_index_key.shape[0],
            output.shape[0],
        ):
            raise ValueError("QSA metadata exceeds the projected token rows")
        if rows == 0:
            output.zero_()
            return
        if metadata.causal is not True:
            raise NotImplementedError("QSA supports causal attention only")

        self._run_b12x_qsa(
            metadata=metadata,
            positions=positions,
            query=query,
            key=key,
            value=value,
            index_query=index_query,
            raw_index_key=raw_index_key,
            output=output,
            rows=rows,
        )
        if rows < output.shape[0]:
            output[rows:].zero_()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.overlap_input_projections or self._share_mtp_indices:
            layer_name = _encode_layer_name(self.layer_name)
            qkv = torch.ops.vllm.qwen4_exp_b12x_qsa_project_inputs(
                positions,
                hidden_states,
                self._selected_positions,
                self.qkv_proj.output_size_per_partition,
                layer_name,
            )
            query, key, value, gate = self._project_qkv_gate(qkv, positions)
            rows = int(hidden_states.shape[0])
            query = query.view(rows, self.num_heads, self.head_dim)
            key = key.view(rows, self.num_projected_kv_heads, self.head_dim)
            value = value.view(rows, self.num_projected_kv_heads, self.head_dim)
            output = torch.empty_like(query)
            torch.ops.vllm.qwen4_exp_b12x_qsa_run_projected(
                positions,
                hidden_states,
                query,
                key,
                value,
                output,
                self._selected_positions,
                layer_name,
            )
            gated = output.flatten(-2) * torch.sigmoid(gate)
            return self.o_proj(gated)[0]
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value, gate = self._project_qkv_gate(qkv, positions)
        index_query, raw_index_key = self.indexer.project(hidden_states)
        num_tokens = int(hidden_states.shape[0])
        query = query.view(num_tokens, self.num_heads, self.head_dim)
        key = key.view(num_tokens, self.num_projected_kv_heads, self.head_dim)
        value = value.view(num_tokens, self.num_projected_kv_heads, self.head_dim)
        output = torch.empty_like(query)
        layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_b12x_qsa_with_output(
                positions,
                query,
                key,
                value,
                index_query,
                raw_index_key,
                output,
                layer_name,
            )
        else:
            qwen4_exp_b12x_qsa_with_output(
                positions,
                query,
                key,
                value,
                index_query,
                raw_index_key,
                output,
                layer_name,
            )
        gated = output.flatten(-2) * torch.sigmoid(gate)
        projected, _ = self.o_proj(gated)
        return projected


def _qsa_project_inputs(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    selected_positions: torch.Tensor,
    qkv_size: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    if selected_positions.data_ptr() != layer._selected_positions.data_ptr():
        raise ValueError("QSA selection must use its bound caller-owned buffer")
    return layer._project_qsa_inputs(positions, hidden_states)


def _qsa_project_inputs_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    selected_positions: torch.Tensor,
    qkv_size: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return hidden_states.new_empty((*hidden_states.shape[:-1], qkv_size))


direct_register_custom_op(
    op_name="qwen4_exp_b12x_qsa_project_inputs",
    op_func=_qsa_project_inputs,
    # This buffer is also the ordering token for the hidden selector inputs.
    mutates_args=["selected_positions"],
    fake_impl=_qsa_project_inputs_fake,
)


def _qsa_run_projected(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    if selected_positions.data_ptr() != layer._selected_positions.data_ptr():
        raise ValueError("QSA attention must consume the matching selector buffer")
    layer._run_projected_qsa(positions, hidden_states, query, key, value, output)


def _qsa_run_projected_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return None


direct_register_custom_op(
    op_name="qwen4_exp_b12x_qsa_run_projected",
    op_func=_qsa_run_projected,
    mutates_args=["output", "selected_positions"],
    fake_impl=_qsa_run_projected_fake,
)


def _qsa_input_projections(
    hidden_states: torch.Tensor,
    qkv_size: int,
    index_size: int,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    qkv_scratch, index_scratch = get_b12x_projection_workspaces(
        hidden_states.shape[0], layer.qkv_proj, layer.indexer.index_qk_proj
    )
    if hidden_states.shape[0] > 16 or not torch.cuda.is_current_stream_capturing():
        with use_preallocated_workspace(qkv_scratch):
            qkv, _ = layer.qkv_proj(hidden_states)
        with use_preallocated_workspace(index_scratch):
            index_qk, _ = layer.indexer.index_qk_proj(hidden_states)
        return qkv, index_qk

    stream = aux_stream()
    assert stream is not None
    main_stream = current_stream()
    stream.wait_stream(main_stream)
    hidden_states.record_stream(stream)
    with torch.cuda.stream(stream), use_preallocated_workspace(index_scratch):
        index_qk, _ = layer.indexer.index_qk_proj(hidden_states)
    with use_preallocated_workspace(qkv_scratch):
        qkv, _ = layer.qkv_proj(hidden_states)
    main_stream.wait_stream(stream)
    index_qk.record_stream(main_stream)
    return qkv, index_qk


def _qsa_input_projections_fake(
    hidden_states: torch.Tensor,
    qkv_size: int,
    index_size: int,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        hidden_states.new_empty((*hidden_states.shape[:-1], qkv_size)),
        hidden_states.new_empty((*hidden_states.shape[:-1], index_size)),
    )


direct_register_custom_op(
    op_name="qwen4_exp_b12x_qsa_input_projections",
    op_func=_qsa_input_projections,
    fake_impl=_qsa_input_projections_fake,
)


def qwen4_exp_b12x_qsa_with_output(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the package-owned QSA write, selector, and sparse-read transaction."""

    resolved_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[resolved_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{resolved_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        positions,
        query,
        key,
        value,
        index_query,
        raw_index_key,
        output,
    )


def _qwen4_exp_b12x_qsa_with_output_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del (
        positions,
        query,
        key,
        value,
        index_query,
        raw_index_key,
        output,
        layer_name,
    )


direct_register_custom_op(
    op_name="qwen4_exp_b12x_qsa_with_output",
    op_func=qwen4_exp_b12x_qsa_with_output,
    mutates_args=["output"],
    fake_impl=_qwen4_exp_b12x_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSABackend",
    "Qwen4ExpQSAImpl",
    "Qwen4ExpQSAMetadata",
    "Qwen4ExpQSAMetadataBuilder",
    "apply_qsa_rope",
    "qwen4_exp_b12x_qsa_with_output",
]
