# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native b12x dense MLA decode backend for Kimi K3 on SM120/SM121."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_K3_ABSORBED_HEAD_DIM = 576
_K3_KV_LORA_RANK = 512
_K3_QK_NOPE_HEAD_DIM = 128
_K3_QK_ROPE_HEAD_DIM = 64
_K3_QK_HEAD_DIM = 192
_K3_V_HEAD_DIM = 128
_MAX_B12X_QUERY_ROWS = 1024
_MAX_B12X_CACHE_TOKENS = 1_048_576
_MAX_I32 = torch.iinfo(torch.int32).max


def _load_dense_mla() -> Any:
    from sparkinfer.attention import dense_mla

    return dense_mla


def _page_table_width(max_cache_tokens: int, page_size: int) -> int:
    width = (max_cache_tokens + page_size - 1) // page_size
    # vLLM pads block tables to a 128-token boundary for page sizes <= 128.
    if page_size <= 128:
        alignment = 128 // page_size
        width = ((width + alignment - 1) // alignment) * alignment
    return width


def _planned_kv_dtype(vllm_config: VllmConfig) -> torch.dtype:
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype == "auto":
        return vllm_config.model_config.dtype
    if cache_dtype == "bfloat16":
        return torch.bfloat16
    if cache_dtype in ("fp8", "fp8_e4m3"):
        fp8_dtype = current_platform.fp8_dtype()
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "B12X_MLA requires native E4M3 FP8 KV storage; "
                f"this platform selected {fp8_dtype}."
            )
        return fp8_dtype
    raise ValueError(
        f"B12X_MLA supports only BF16 or E4M3 KV cache storage, got {cache_dtype!r}."
    )


def _create_dense_mla_plan(
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    page_size: int,
    num_q_heads: int,
) -> Any:
    dense_mla = _load_dense_mla()
    max_total_q = int(vllm_config.scheduler_config.max_num_seqs)
    max_cache_tokens = int(vllm_config.model_config.max_model_len)
    if max_total_q > _MAX_B12X_QUERY_ROWS:
        raise ValueError(
            "B12X_MLA supports at most "
            f"{_MAX_B12X_QUERY_ROWS} simultaneous decode rows, got {max_total_q}."
        )
    if max_cache_tokens > _MAX_B12X_CACHE_TOKENS:
        raise ValueError(
            "B12X_MLA supports at most "
            f"{_MAX_B12X_CACHE_TOKENS} cache tokens, got {max_cache_tokens}."
        )

    caps = dense_mla.Caps(
        device=device,
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=_planned_kv_dtype(vllm_config),
        num_q_heads=num_q_heads,
        page_size=page_size,
        max_total_q=max_total_q,
        max_batch=max_total_q,
        max_cache_tokens=max_cache_tokens,
        max_page_table_width=_page_table_width(max_cache_tokens, page_size),
        # This is a validation ceiling only; physical page count is selected
        # after memory profiling and can exceed max_model_len / page_size when
        # the server has capacity for multiple requests.
        num_cache_pages=_MAX_I32,
        use_cuda_graph=True,
    )
    return dense_mla.plan(caps)


@dataclass
class B12xMLAMetadata(MLACommonMetadata):
    """Common MLA metadata plus the capture-static b12x launch plan."""

    dense_mla_plan: Any | None = None


class B12xMLAMetadataBuilder(MLACommonMetadataBuilder[B12xMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            B12xMLAMetadata,
        )
        # The builder receives the final kernel block size, including K3's
        # hybrid-cache alignment (944 in the production launcher). Planning in
        # the layer constructor would incorrectly see the initial size of 16.
        self._dense_mla_plan = _create_dense_mla_plan(
            vllm_config,
            device,
            page_size=self.page_size,
            num_q_heads=self.num_heads,
        )
        self._workspace_specs = self._dense_mla_plan.shapes_and_dtypes()
        if is_workspace_manager_initialized():
            current_workspace_manager().get_simultaneous(*self._workspace_specs)
        logger.info_once(
            "B12X dense K3 MLA plan: heads=%d, page_size=%d, "
            "max_decode_rows=%d, max_cache_tokens=%d, splits=%d",
            self.num_heads,
            self.page_size,
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.model_config.max_model_len,
            self._dense_mla_plan.num_splits,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLAMetadata:
        metadata = cast(
            B12xMLAMetadata,
            super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            ),
        )
        metadata.dense_mla_plan = self._dense_mla_plan
        return metadata


class B12xMLABackend(MLACommonBackend):
    """Opt-in dense Kimi K3 MLA backend backed by sparkinfer."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Keep the literal visible to the generated backend capability table.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA"

    @staticmethod
    def get_impl_cls() -> type[B12xMLAImpl]:
        return B12xMLAImpl

    @staticmethod
    def get_builder_cls() -> type[B12xMLAMetadataBuilder]:
        return B12xMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)

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
        try:
            _load_dense_mla()
        except (ImportError, AttributeError):
            return "B12X_MLA requires a b12x build that provides dense_mla"

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        if model_config is None:
            return None
        hf_text_config = model_config.hf_text_config
        if getattr(hf_text_config, "model_type", None) != "kimi_linear":
            return "B12X_MLA currently supports only Kimi K3"

        dims = (
            getattr(hf_text_config, "kv_lora_rank", None),
            getattr(hf_text_config, "qk_nope_head_dim", None),
            getattr(hf_text_config, "qk_rope_head_dim", None),
            getattr(hf_text_config, "v_head_dim", None),
        )
        required_dims = (
            _K3_KV_LORA_RANK,
            _K3_QK_NOPE_HEAD_DIM,
            _K3_QK_ROPE_HEAD_DIM,
            _K3_V_HEAD_DIM,
        )
        if dims != required_dims:
            return (
                "B12X_MLA requires K3 MLA dimensions "
                "(kv_lora=512, qk_nope=128, qk_rope=64, v=128), "
                f"got {dims}"
            )

        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size != 1:
            return "B12X_MLA does not yet support decode context parallelism"
        local_heads = model_config.get_num_attention_heads(parallel_config)
        if local_heads <= 0 or local_heads % 8:
            return (
                "B12X_MLA requires a positive multiple of 8 query heads per "
                f"rank, got {local_heads}"
            )
        if vllm_config.scheduler_config.max_num_seqs > _MAX_B12X_QUERY_ROWS:
            return (
                "B12X_MLA max_num_seqs exceeds its 1024-row decode capacity: "
                f"{vllm_config.scheduler_config.max_num_seqs}"
            )
        if model_config.max_model_len > _MAX_B12X_CACHE_TOKENS:
            return (
                "B12X_MLA max_model_len exceeds its 1048576-token capacity: "
                f"{model_config.max_model_len}"
            )
        return None


class B12xMLAImpl(MLACommonImpl[B12xMLAMetadata]):
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
        **mla_args: Any,
    ) -> None:
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
            **mla_args,
        )

        if any(
            feature is not None
            for feature in (alibi_slopes, sliding_window, logits_soft_cap)
        ):
            raise NotImplementedError(
                "B12xMLAImpl does not support alibi, sliding windows, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("B12xMLAImpl supports decoder attention only.")
        if num_kv_heads != 1:
            raise ValueError(f"B12xMLAImpl requires one KV head, got {num_kv_heads}.")

        actual_dims = (
            head_size,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        required_dims = (
            _K3_ABSORBED_HEAD_DIM,
            _K3_KV_LORA_RANK,
            _K3_QK_NOPE_HEAD_DIM,
            _K3_QK_ROPE_HEAD_DIM,
            _K3_QK_HEAD_DIM,
            _K3_V_HEAD_DIM,
        )
        if actual_dims != required_dims:
            raise ValueError(
                f"B12xMLAImpl received non-K3 MLA dimensions {actual_dims}; "
                f"required {required_dims}."
            )
        if num_heads <= 0 or num_heads % 8:
            raise ValueError(
                "B12xMLAImpl requires a positive multiple of 8 query heads, "
                f"got {num_heads}."
            )
        if get_current_vllm_config().parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError(
                "B12xMLAImpl does not yet support decode context parallelism."
            )

        self._dense_mla = _load_dense_mla()
        self._compiled_bindings: set[tuple[object, ...]] = set()
        self._fallback_scratch: dict[int, torch.Tensor] = {}

    def _borrow_scratch(self, plan: Any, device: torch.device) -> torch.Tensor:
        specs = plan.shapes_and_dtypes()
        if is_workspace_manager_initialized():
            (scratch,) = current_workspace_manager().get_simultaneous(*specs)
            return scratch

        # Unit-test/minimal-runner fallback. Production initializes and
        # pre-sizes WorkspaceManager before metadata construction.
        key = id(plan)
        scratch = self._fallback_scratch.get(key)
        if scratch is None:
            if len(specs) != 1:
                raise RuntimeError("B12X_MLA expected exactly one scratch buffer.")
            shape, dtype = specs[0]
            scratch = torch.empty(shape, dtype=dtype, device=device)
            self._fallback_scratch[key] = scratch
        return scratch

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kv_c_and_k_pe_cache.numel() == 0:
            raise ValueError("B12X_MLA received an empty KV cache.")
        if attn_metadata.decode is None:
            raise ValueError("B12X_MLA requires decode metadata.")
        plan = attn_metadata.dense_mla_plan
        if plan is None:
            raise RuntimeError("B12X_MLA metadata is missing its dense MLA plan.")

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()

        batch = int(attn_metadata.decode.seq_lens.shape[0])
        if int(q.shape[0]) != batch:
            raise ValueError(
                "B12X_MLA's single-token decode path requires one query row per "
                f"request, got {q.shape[0]} rows for {batch} requests."
            )
        if int(q.shape[1]) != self.num_heads:
            raise ValueError(
                f"B12X_MLA expected {self.num_heads} query heads, got {q.shape[1]}."
            )

        output = torch.empty(
            (batch, self.num_heads, self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=q.device,
        )
        scratch = self._borrow_scratch(plan, q.device)
        quantized = q.dtype == torch.float8_e4m3fn
        binding = self._dense_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_c_and_k_pe_cache,
            output=output,
            page_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc[: batch + 1],
            q_scale=layer._q_scale if quantized else None,
            kv_scale=layer._k_scale if quantized else None,
            sm_scale=self.scale,
        )

        compile_key = (
            id(plan),
            q.dtype,
            tuple(q.stride()),
            tuple(kv_c_and_k_pe_cache.stride()),
            tuple(output.stride()),
        )
        if compile_key not in self._compiled_bindings:
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X_MLA encountered an uncompiled layout during CUDA graph "
                    "capture; the eager graph warmup did not exercise this cache "
                    "layout."
                )
            self._dense_mla.compile(binding=binding)
            self._compiled_bindings.add(compile_key)

        return self._dense_mla.run(binding=binding)


__all__ = [
    "B12xMLABackend",
    "B12xMLAImpl",
    "B12xMLAMetadata",
    "B12xMLAMetadataBuilder",
]
