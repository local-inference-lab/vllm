# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_sparse_mla
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
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata


_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_GLM_NEXT_CACHE_RECORD_BYTES = 528
_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN = 132 // 4


def _is_glm_next_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


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
    is_glm_next: bool,
) -> int:
    if is_glm_next:
        # GLM_NEXT selected indices are physical token slots. The b12x kernel
        # applies the cache's byte page stride itself.
        return block_size
    record_width = int(kv_cache.shape[-1])
    return int(kv_cache.stride(0)) // record_width


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
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
        if not _is_glm_next_spec(spec):
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        if spec.head_size != 512:
            raise ValueError(
                "B12X GLM5Next sparse MLA requires head_size=512, got "
                f"{spec.head_size}."
            )
        return replace(
            spec,
            state_content_bytes=_GLM_NEXT_CACHE_RECORD_BYTES,
            page_tail_bytes_per_token=_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Sparse index caches share manager blocks with their MLA layer. Keep
        # the layer dimension inside the manager's block so block copies and
        # swaps carry both cache regions together. DeepSeek-V4's index backend
        # already imposes the same constraint, so this preserves its layout.
        return (KVCacheLayout.BLHNC,)

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
            if head_size != 576:
                return "B12X sparse MLA requires head_size=576"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


class B12xGLM5NextMLASparseBackend(B12xMLASparseBackend):
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
class B12xMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1
    cache_seq_lens_per_token: torch.Tensor | None = None
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None


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
        self.supports_draft_decode_metadata_update = (
            self.requires_glm_next_selector_metadata
        )
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        scheduler_config = vllm_config.scheduler_config
        max_tokens = scheduler_config.max_num_batched_tokens
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        if self.requires_glm_next_selector_metadata:
            max_reqs = int(scheduler_config.max_num_seqs)
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

        if common.max_query_len <= 1 and num_tokens == common.num_reqs:
            per_token_lens = seq_lens[:num_tokens]
        elif not use_dcp and common.positions is not None:
            per_token_lens = common.positions[:num_tokens].to(torch.int32) + 1
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
        accepted = metadata.selector_num_accepted_tokens
        if not self.requires_glm_next_selector_metadata or accepted is None:
            raise RuntimeError(
                "GLM5Next draft decode metadata requires accepted-token counts"
            )
        accepted.fill_(1)


class B12xGLM5NextMLASparseMetadataBuilder(B12xMLASparseMetadataBuilder):
    # The pooled selector must commit every fresh or extended prompt row
    # through run_prefill; decode commits only accepted prior rows.
    treat_short_extends_as_decodes: ClassVar[bool] = False


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
    supports_pcp = False

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
        if kv_cache_dtype != "fp8_ds_mla":
            raise ValueError(
                "B12X sparse MLA requires the packed fp8_ds_mla KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        for name in ("Caps", "plan", "run_decode", "run_extend"):
            getattr(module, name)
        self._run_decode = module.run_decode
        self._run_extend = module.run_extend
        self._model_type: int | None = None
        self._concat_and_cache_glm_next_mla = None
        if self._is_glm_next:
            self._model_type = int(module.ModelType.GLM_NEXT)
            self._concat_and_cache_glm_next_mla = module.concat_and_cache_glm_next_mla

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
        self._kv_dtype = torch.uint8
        kernel_page_size = (
            int(vllm_config.cache_config.block_size) if self._is_glm_next else 64
        )

        self._module = module
        self._kernel_page_size = 0
        self._set_kernel_page_size(kernel_page_size)
        self.supports_quant_query_input = False

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return

        def make_plan(mode: str):
            caps_kwargs = dict(
                device=torch.device("cuda", torch.accelerator.current_device_index()),
                num_q_heads=self._input_num_heads,
                max_q_rows=self._max_tokens,
                max_width=self._topk_tokens,
                dtype=torch.bfloat16,
                kv_dtype=self._kv_dtype,
                head_dim=self._q_head_dim,
                v_head_dim=self.kv_lora_rank,
                mode=mode,
                max_batch=self._max_tokens,
                max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
                page_size=kernel_page_size,
            )
            if self._model_type is not None:
                caps_kwargs["model_type"] = self._model_type
            return self._module.plan(self._module.Caps(**caps_kwargs))

        decode_plan = make_plan("decode")
        extend_plan = make_plan("extend")
        self._decode_plan = decode_plan
        self._extend_plan = extend_plan
        self._kernel_page_size = kernel_page_size

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self._is_glm_next:
            if kv_cache.ndim != 3 or int(kv_cache.shape[-1]) != 528:
                raise ValueError(
                    "B12X GLM5Next cache must have shape "
                    "[pages, page_size, 528], got "
                    f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}, "
                    f"dtype={kv_cache.dtype}"
                )
            self._set_kernel_page_size(int(kv_cache.shape[1]))

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if not self._is_glm_next:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        del k_scale
        if kv_cache.numel() == 0:
            return
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "B12X GLM5Next cache updates require a zero-width RoPE tensor, "
                f"got shape={tuple(k_pe.shape)}."
            )
        assert self._concat_and_cache_glm_next_mla is not None
        self._concat_and_cache_glm_next_mla(
            kv_c_normed,
            kv_cache,
            slot_mapping.flatten(),
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
        plan = (
            self._decode_plan if attn_metadata.max_query_len <= 1 else self._extend_plan
        )
        q_spec = (
            (self._max_tokens, self._input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        workspaces = current_workspace_manager().get_simultaneous(
            q_spec, *plan.shapes_and_dtypes()
        )
        q_buffer = workspaces[0]
        scratch = workspaces[1:]

        if isinstance(q, tuple):
            q_nope, q_pe = q
            num_tokens = int(q_nope.shape[0])
            q_all = q_buffer[:num_tokens]
            if int(q_pe.shape[-1]) == 0:
                q_all.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_all)
        else:
            num_tokens = int(q.shape[0])
            q_all = q_buffer[:num_tokens]
            q_all.copy_(q)

        if int(q_all.shape[1]) != self._input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {self._input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        block_stride_rows = _selected_index_block_stride_rows(
            kv_c_and_k_pe_cache,
            block_size=attn_metadata.block_size,
            is_glm_next=self._is_glm_next,
        )
        if self.dcp_world_size > 1:
            selected_indices, active_counts = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            selected_indices, active_counts = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        cache_seq_lens = attn_metadata.cache_seq_lens_per_token
        assert cache_seq_lens is not None
        cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()
        binding = plan.bind(
            scratch=scratch,
            q=q_all,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seq_lens,
            nsa_cache_seqlens_int32=active_counts,
        )
        run = self._run_decode if plan is self._decode_plan else self._run_extend
        run_kwargs = dict(
            binding=binding,
            kv_cache=kv_c_and_k_pe_cache,
            sm_scale=self.scale,
            v_head_dim=self.kv_lora_rank,
            return_lse=self.need_to_return_lse_for_decode,
            lse_scale="natural",
        )
        if self._model_type is not None:
            run_kwargs["model_type"] = self._model_type
        result = run(**run_kwargs)
        if self.need_to_return_lse_for_decode:
            output, lse = result
            return output, lse
        assert isinstance(result, torch.Tensor)
        return result, None
