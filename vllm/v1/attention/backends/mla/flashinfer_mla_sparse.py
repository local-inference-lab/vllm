# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA attention backend."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)

_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
# FlashInfer's SM120 GLM_NSA kernel is instantiated for a 576-wide query and
# a 656-byte packed record: 512 FP8 latent bytes, 16 bytes of scales, and a
# 64-element BF16 RoPE payload. GLM5Next has no RoPE payload, so its backend
# implementation zero-pads that final region rather than publishing an
# unsupported 528-byte kernel layout.
_GLM_NEXT_CACHE_RECORD_BYTES = 656
# The selector consumes 33 bytes/token (one 132-byte C4 record per four
# tokens).  With a 656-byte FlashInfer state record, four extra padding bytes
# per token make every 256-token unit an exact multiple of the C4 page size.
# This also preserves the invariant for the 2,304-token hybrid-manager page.
# The pooled indexer leaves that padding unused.
_GLM_NEXT_PACKED_TAIL_BYTES_PER_TOKEN = 37


def _current_hf_text_config() -> object | None:
    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return None
    return config.model_config.hf_text_config


def _is_glm_next_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm5_next":
        return True
    hf_config = _current_hf_text_config()
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


class _FlashInferMLASparseBackendBase(AttentionBackend):
    """Common metadata for concrete FlashInfer sparse MLA backends."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class FlashInferMLASparseTRTLLMBackend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend using the TRTLLM-gen launcher."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseTRTLLMMetadataBuilder"]:
        return FlashInferMLASparseTRTLLMMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

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

        vllm_config = get_current_vllm_config()
        if kv_cache_dtype == "fp8_ds_mla":
            return (
                "FLASHINFER_MLA_SPARSE SM10 does not support fp8_ds_mla kv-cache dtype"
            )

        # FlashInfer MLA sparse SM10 kernel requires qk_nope_head_dim in [128, 192].
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None


class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend for SM120."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not _is_glm_next_spec(spec):
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "FlashInfer SM120 GLM5Next sparse MLA requires an "
                f"MLAAttentionSpec, got {type(spec).__name__}."
            )
        if spec.head_size != 512:
            raise ValueError(
                "FlashInfer SM120 GLM5Next sparse MLA requires head_size=512, "
                f"got {spec.head_size}."
            )
        return replace(
            spec,
            state_content_bytes=_GLM_NEXT_CACHE_RECORD_BYTES,
            page_tail_bytes_per_token=_GLM_NEXT_PACKED_TAIL_BYTES_PER_TOKEN,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Keep the GLM pooled-index tail adjacent to its parent MLA page.
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # The SM120 launcher accepts both DeepSeek's 512+64 MLA query and
        # GLM5Next's 512-wide non-RoPE query. The shared SM10 backend remains
        # restricted to the 576-wide DeepSeek layout.
        return [512, 576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        hf_config = _current_hf_text_config()
        if getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES:
            # The pooled-index tail is appended to the manager page. Keeping
            # that page intact preserves the semantic-state/tail boundary;
            # splitting it into smaller kernel pages would interleave neither
            # region correctly under a block-outermost layout.
            return [MultipleOf(64)]
        return [64, 256]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
        )

        return FlashInferMLASparseSM120Impl

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
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            index_topk = getattr(hf_text_config, "index_topk", None)
            if index_topk is None:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires a model with "
                    "index_topk config"
                )
            if int(index_topk) != 2048:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; "
                    f"got {index_topk}"
                )
        return None


@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    # Query start locations
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    # Sequence lengths for all requests (context + query)
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None

    # Sparse-specific
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None


class FlashInferMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    metadata_cls = FlashInferMLASparseMetadata
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
        self.requires_glm_next_selector_metadata = (
            getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES
        )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = (
            self.requires_glm_next_selector_metadata
        )
        if self.requires_glm_next_selector_metadata:
            max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
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

    def _should_treat_short_extends_as_decodes(self) -> bool:
        # Every fresh or extended GLM prompt row must update the pooled
        # selector before it can be routed as decode.
        return not self.requires_glm_next_selector_metadata

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
                    "FlashInfer sparse MLA builder"
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
                "FlashInfer GLM5Next sparse MLA requires selector state slots, "
                "fresh flags, accepted-token counts, and prefill flags"
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
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        if metadata.num_prefills:
            prefill_start = metadata.num_decodes
            prefill_end = prefill_start + metadata.num_prefills + 1
            metadata.prefill_query_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[prefill_start:prefill_end]
            )
        (
            metadata.selector_state_slot_ids,
            metadata.selector_state_is_fresh,
            metadata.selector_num_accepted_tokens,
            metadata.selector_is_prefilling,
        ) = self._stage_glm_next_selector_metadata(
            num_reqs=common_attn_metadata.num_reqs,
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
    ) -> FlashInferMLASparseMetadata:
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
    ) -> FlashInferMLASparseMetadata:
        return self._build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def update_draft_decode_metadata(
        self,
        metadata: FlashInferMLASparseMetadata,
    ) -> None:
        accepted = metadata.selector_num_accepted_tokens
        if not self.requires_glm_next_selector_metadata or accepted is None:
            raise RuntimeError(
                "GLM5Next draft decode metadata requires accepted-token counts"
            )
        accepted.fill_(1)


class FlashInferMLASparseTRTLLMMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """Metadata builder for the SM100 TRT-LLM sparse MLA kernel."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        return common_attn_metadata.token_to_req_indices(self.req_id_per_token_buffer)


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None


def _get_workspace_buffer(device: torch.device) -> torch.Tensor:
    global _fi_sparse_workspace
    if _fi_sparse_workspace is None:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_sparse_workspace = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    can_return_lse_for_decode: bool = True
    lse_base_on_e: bool = False

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
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
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

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # fp8 query quantization is required when using fp8 kv_cache,
        # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
        # for query and kv_cache (mixed bf16+fp8 is not supported).
        self.supports_quant_query_input = True

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        # Single-token sparse decode. trtllm-gen requires the q_len_per_request
        # dim, but the sparse attention mask is fully per-token (each query token
        # carries its own top-k index row), so unsqueeze is sufficient and
        # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
        # deferred until MTP is validated end-to-end for this backend.
        query = q.unsqueeze(1)
        block_tables = topk_indices_physical.unsqueeze(1)
        seq_lens_arg = seq_lens

        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_arg,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            o = kernel_out
            lse = None

        out = o.view(-1, o.shape[-2], o.shape[-1])
        if lse is not None:
            lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
            empty_rows = (topk_indices_physical == -1).all(dim=-1)
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens, num_heads)
        # or 3D ((num_tokens, num_heads, 1) / (num_tokens, 1, num_heads)).
        # Collapse all of these to the (num_tokens, num_heads) the shared DCP
        # reducer expects.
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse
