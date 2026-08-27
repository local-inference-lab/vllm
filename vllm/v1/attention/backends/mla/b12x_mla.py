# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native B12X dense MLA decode backend for Kimi K3 on SM120/SM121."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
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

logger = init_logger(__name__)

_K3_ABSORBED_HEAD_DIM = 576
_K3_KV_LORA_RANK = 512
_K3_QK_NOPE_HEAD_DIM = 128
_K3_QK_ROPE_HEAD_DIM = 64
_K3_QK_HEAD_DIM = 192
_K3_V_HEAD_DIM = 128
_MAX_B12X_QUERY_ROWS = 1024
_MAX_B12X_CACHE_TOKENS = 1_048_576
_B12X_QUERY_HEAD_TILE = 8
_MAX_I32 = torch.iinfo(torch.int32).max


def _load_dense_mla() -> Any:
    from b12x.attention import dense_mla

    return dense_mla


def _page_table_width(max_cache_tokens: int, page_size: int) -> int:
    width = (max_cache_tokens + page_size - 1) // page_size
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
                "B12X requires native E4M3 FP8 KV storage; "
                f"this platform selected {fp8_dtype}."
            )
        return fp8_dtype
    raise ValueError(
        f"B12X supports only BF16 or E4M3 KV cache storage, got {cache_dtype!r}."
    )


def _max_dcp_local_cache_tokens(
    vllm_config: VllmConfig, *, dcp_size: int | None = None
) -> int:
    """Return the largest interleaved KV shard held by one DCP rank."""
    parallel_config = vllm_config.parallel_config
    dcp_size = int(
        parallel_config.decode_context_parallel_size if dcp_size is None else dcp_size
    )
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    if dcp_size <= 0 or interleave <= 0:
        raise ValueError(
            "B12X requires positive DCP and KV-interleave sizes, got "
            f"DCP={dcp_size}, interleave={interleave}."
        )
    max_model_len = int(vllm_config.model_config.max_model_len)
    partitions = dcp_size * interleave
    return ((max_model_len + partitions - 1) // partitions) * interleave


def _kernel_query_heads(local_heads: int, dcp_size: int = 1) -> int:
    """Return the tiled head count after an optional DCP query gather."""
    if local_heads <= 0 or dcp_size <= 0:
        raise ValueError(
            "B12X requires positive query-head and DCP sizes, got "
            f"heads={local_heads}, DCP={dcp_size}."
        )
    effective_heads = local_heads * dcp_size
    if dcp_size > 1 and effective_heads % _B12X_QUERY_HEAD_TILE:
        raise ValueError(
            "B12X requires a multiple of eight query heads after DCP "
            f"gather, got local={local_heads}, DCP={dcp_size}, "
            f"effective={effective_heads}."
        )
    return (
        (effective_heads + _B12X_QUERY_HEAD_TILE - 1)
        // _B12X_QUERY_HEAD_TILE
        * _B12X_QUERY_HEAD_TILE
    )


def _active_dense_mla_splits(plan: Any, max_seq_len: int | None) -> int:
    """Return the split-plan prefix that can contain live cache rows."""
    num_splits = int(getattr(plan, "num_splits", 1))
    chunks_per_split = int(getattr(plan, "chunks_per_split", 1))
    if num_splits <= 0 or chunks_per_split <= 0:
        raise ValueError(
            "B12X received invalid split geometry: "
            f"splits={num_splits}, chunks_per_split={chunks_per_split}."
        )
    if max_seq_len is None:
        return num_splits
    valid_chunks = max(1, (max(0, int(max_seq_len)) + 63) // 64)
    return min(
        num_splits,
        (valid_chunks + chunks_per_split - 1) // chunks_per_split,
    )


def _create_dense_mla_plan(
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    page_size: int,
    num_q_heads: int,
    max_total_q: int | None = None,
    dcp_size: int | None = None,
    max_cache_tokens: int | None = None,
) -> Any:
    dense_mla = _load_dense_mla()
    max_total_q = int(
        max_total_q
        if max_total_q is not None
        else vllm_config.scheduler_config.max_num_seqs
    )
    max_cache_tokens = int(
        max_cache_tokens
        if max_cache_tokens is not None
        else _max_dcp_local_cache_tokens(vllm_config, dcp_size=dcp_size)
    )
    if max_total_q > _MAX_B12X_QUERY_ROWS:
        raise ValueError(
            "B12X supports at most "
            f"{_MAX_B12X_QUERY_ROWS} simultaneous decode rows, got {max_total_q}."
        )
    if max_cache_tokens > _MAX_B12X_CACHE_TOKENS:
        raise ValueError(
            "B12X supports at most "
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
        num_cache_pages=_MAX_I32,
        use_cuda_graph=True,
    )
    return dense_mla.plan(caps)


@dataclass
class B12xMLAMetadata(MLACommonMetadata):
    """Common MLA metadata plus the capture-static B12X launch plan."""

    dense_mla_plan: Any | None = None
    dense_mla_scratch: torch.Tensor | None = None
    dense_mla_padded_q: torch.Tensor | None = None
    dense_mla_padded_output: torch.Tensor | None = None
    dense_mla_flat_block_table: torch.Tensor | None = None
    dense_mla_flat_seq_lens: torch.Tensor | None = None
    dense_mla_flat_query_start_loc: torch.Tensor | None = None
    dense_mla_dcp_world_size: int = 1


class B12xMLAMetadataBuilder(MLACommonMetadataBuilder[B12xMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True

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
            supports_dcp_with_varlen=True,
        )
        self._dcp_rank = (
            int(get_dcp_group().rank_in_group) if self.dcp_world_size > 1 else 0
        )
        max_dense_mla_rows = int(vllm_config.scheduler_config.max_num_seqs) * int(
            self.reorder_batch_threshold
        )
        if max_dense_mla_rows > _MAX_B12X_QUERY_ROWS:
            raise ValueError(
                "B12X query capacity exceeds its limit: "
                f"rows={max_dense_mla_rows}, limit={_MAX_B12X_QUERY_ROWS}."
            )
        self._max_dense_mla_rows = max_dense_mla_rows
        self._effective_heads = self.num_heads * self.dcp_world_size
        self._kernel_heads = _kernel_query_heads(self.num_heads, self.dcp_world_size)
        max_cache_tokens = _max_dcp_local_cache_tokens(
            vllm_config, dcp_size=self.dcp_world_size
        )
        sliding_window = getattr(kv_cache_spec, "sliding_window", None)
        if sliding_window is not None:
            max_cache_tokens = min(max_cache_tokens, int(sliding_window))
        self._dense_mla_plan = _create_dense_mla_plan(
            vllm_config,
            device,
            page_size=self.page_size,
            num_q_heads=self._kernel_heads,
            max_total_q=max_dense_mla_rows,
            dcp_size=self.dcp_world_size,
            max_cache_tokens=max_cache_tokens,
        )
        self._workspace_specs = self._dense_mla_plan.shapes_and_dtypes()
        if len(self._workspace_specs) != 1:
            raise RuntimeError("B12X expected exactly one scratch buffer.")
        scratch_shape, scratch_dtype = self._workspace_specs[0]
        # Every attention layer represented by this builder executes serially
        # on the model stream. One builder-owned buffer therefore gives each
        # eager bind a stable caller-owned address without a backend workspace
        # cache or one allocation per layer.
        self._dense_mla_scratch = torch.empty(
            scratch_shape,
            dtype=scratch_dtype,
            device=device,
        )
        self._dense_mla_padded_q: torch.Tensor | None = None
        self._dense_mla_padded_output: torch.Tensor | None = None
        if self._kernel_heads != self._effective_heads:
            self._dense_mla_padded_q = torch.empty(
                (max_dense_mla_rows, self._kernel_heads, _K3_ABSORBED_HEAD_DIM),
                dtype=_planned_kv_dtype(vllm_config),
                device=device,
            )
            self._dense_mla_padded_output = torch.empty(
                (max_dense_mla_rows, self._kernel_heads, _K3_KV_LORA_RANK),
                dtype=torch.bfloat16,
                device=device,
            )
        max_table_width = int(self._dense_mla_plan.caps.max_page_table_width)
        self._dense_mla_flat_block_table = torch.zeros(
            (max_dense_mla_rows, max_table_width),
            dtype=torch.int32,
            device=device,
        )
        self._dense_mla_flat_seq_lens = torch.empty(
            max_dense_mla_rows,
            dtype=torch.int32,
            device=device,
        )
        self._dense_mla_flat_query_start_loc = torch.arange(
            max_dense_mla_rows + 1,
            dtype=torch.int32,
            device=device,
        )
        self._dense_mla_causal_offsets = torch.arange(
            1 - int(self.reorder_batch_threshold),
            1,
            dtype=torch.int32,
            device=device,
        )
        self._dense_mla_flat_global_seq_lens = (
            torch.empty(
                max_dense_mla_rows,
                dtype=torch.int32,
                device=device,
            )
            if self.dcp_world_size > 1
            else None
        )
        self._dense_mla_flat_dcp_remainder = (
            torch.empty(
                max_dense_mla_rows,
                dtype=torch.int32,
                device=device,
            )
            if self.dcp_world_size > 1
            else None
        )
        logger.info_once(
            "B12X dense K3 MLA plan: local_heads=%d, effective_heads=%d, "
            "kernel_heads=%d, page_size=%d, "
            "max_decode_rows=%d, max_cache_tokens=%d, splits=%d",
            self.num_heads,
            self._effective_heads,
            self._kernel_heads,
            self.page_size,
            max_dense_mla_rows,
            max_cache_tokens,
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
        metadata.dense_mla_scratch = self._dense_mla_scratch
        metadata.dense_mla_padded_q = self._dense_mla_padded_q
        metadata.dense_mla_padded_output = self._dense_mla_padded_output
        metadata.dense_mla_dcp_world_size = self.dcp_world_size
        decode_metadata = metadata.decode
        flatten_decode = False
        if decode_metadata is not None and metadata.num_decodes > 0:
            flatten_decode = metadata.num_decode_tokens > metadata.num_decodes or int(
                decode_metadata.block_table.shape[1]
            ) > int(self._dense_mla_plan.caps.max_page_table_width)
        if flatten_decode:
            assert decode_metadata is not None
            total_q = int(metadata.num_decode_tokens)
            if total_q > self._max_dense_mla_rows:
                raise ValueError(
                    "B12X query block exceeds its flattened capacity: "
                    f"rows={total_q}, capacity={self._max_dense_mla_rows}."
                )
            if total_q % metadata.num_decodes:
                raise ValueError(
                    "B12X requires a uniform query block, got "
                    f"tokens={total_q}, requests={metadata.num_decodes}."
                )
            query_len = total_q // metadata.num_decodes
            source_table = decode_metadata.block_table
            flat_table = self._dense_mla_flat_block_table[:total_q]
            # A bounded speculative cache can retain a position-indexed worker
            # table wider than the resident cache. Sequence lengths make the
            # omitted suffix unreachable by the dense-MLA kernel.
            source_width = min(int(source_table.shape[1]), int(flat_table.shape[1]))
            flat_table[:, :source_width].copy_(
                source_table[:, None, :source_width]
                .expand(-1, query_len, -1)
                .reshape(total_q, source_width)
            )
            flat_lens = self._dense_mla_flat_seq_lens[:total_q]
            if metadata.causal:
                offsets = self._dense_mla_causal_offsets[-query_len:]
                if self.dcp_world_size > 1:
                    global_source_lens = decode_metadata.dcp_tot_seq_lens
                    if global_source_lens is None:
                        raise RuntimeError(
                            "B12X causal DCP verification requires global "
                            "decode sequence lengths."
                        )
                    assert self._dense_mla_flat_global_seq_lens is not None
                    assert self._dense_mla_flat_dcp_remainder is not None
                    global_flat_lens = self._dense_mla_flat_global_seq_lens[:total_q]
                    torch.add(
                        global_source_lens[:, None],
                        offsets,
                        out=global_flat_lens.view(metadata.num_decodes, query_len),
                    )
                    virtual_block = (
                        self.dcp_world_size * self.cp_kv_cache_interleave_size
                    )
                    torch.div(
                        global_flat_lens,
                        virtual_block,
                        rounding_mode="floor",
                        out=flat_lens,
                    )
                    flat_lens.mul_(self.cp_kv_cache_interleave_size)
                    remainder = self._dense_mla_flat_dcp_remainder[:total_q]
                    torch.remainder(global_flat_lens, virtual_block, out=remainder)
                    remainder.sub_(self._dcp_rank * self.cp_kv_cache_interleave_size)
                    remainder.clamp_(
                        min=0,
                        max=self.cp_kv_cache_interleave_size,
                    )
                    flat_lens.add_(remainder)
                else:
                    torch.add(
                        decode_metadata.seq_lens[:, None],
                        offsets,
                        out=flat_lens.view(metadata.num_decodes, query_len),
                    )
            else:
                flat_lens.copy_(
                    decode_metadata.seq_lens[:, None]
                    .expand(-1, query_len)
                    .reshape(total_q)
                )
            metadata.dense_mla_flat_block_table = flat_table
            metadata.dense_mla_flat_seq_lens = flat_lens
            metadata.dense_mla_flat_query_start_loc = (
                self._dense_mla_flat_query_start_loc[: total_q + 1]
            )
        return metadata


class B12xMLABackend(MLACommonBackend):
    """Opt-in dense Kimi K3 MLA backend backed by B12X."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
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
        return "B12X"

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
            return "B12X requires a B12X build that provides dense_mla"

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        if model_config is None:
            return None
        hf_text_config = model_config.hf_text_config
        if getattr(hf_text_config, "model_type", None) not in (
            "kimi_linear",
            "k3_dspark",
        ):
            return "B12X currently supports only Kimi K3 and K3 DSpark"

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
                "B12X requires K3 MLA dimensions "
                "(kv_lora=512, qk_nope=128, qk_rope=64, v=128), "
                f"got {dims}"
            )

        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size != 1:
            return "B12X does not support prefill context parallelism"
        dcp_size = int(parallel_config.decode_context_parallel_size)
        local_heads = model_config.get_num_attention_heads(parallel_config)
        try:
            _kernel_query_heads(local_heads, dcp_size)
        except ValueError as exc:
            return str(exc)
        if vllm_config.scheduler_config.max_num_seqs > _MAX_B12X_QUERY_ROWS:
            return (
                "B12X max_num_seqs exceeds its 1024-row decode capacity: "
                f"{vllm_config.scheduler_config.max_num_seqs}"
            )
        local_cache_tokens = _max_dcp_local_cache_tokens(vllm_config)
        if local_cache_tokens > _MAX_B12X_CACHE_TOKENS:
            return (
                "B12X local DCP cache exceeds its 1048576-token capacity: "
                f"{local_cache_tokens}"
            )
        return None

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


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
        if num_heads <= 0:
            raise ValueError(
                f"B12xMLAImpl requires a positive query-head count, got {num_heads}."
            )
        vllm_config = get_current_vllm_config()
        self.dcp_world_size = int(
            vllm_config.parallel_config.decode_context_parallel_size
        )
        if vllm_config.parallel_config.prefill_context_parallel_size != 1:
            raise NotImplementedError(
                "B12xMLAImpl does not support prefill context parallelism."
            )
        self._dense_mla = _load_dense_mla()
        self._compiled_bindings: set[tuple[object, ...]] = set()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kv_c_and_k_pe_cache.numel() == 0:
            raise ValueError("B12X received an empty KV cache.")
        if attn_metadata.decode is None:
            raise ValueError("B12X requires decode metadata.")
        plan = attn_metadata.dense_mla_plan
        if plan is None:
            raise RuntimeError("B12X metadata is missing its dense MLA plan.")

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        flat_block_table = getattr(attn_metadata, "dense_mla_flat_block_table", None)
        if flat_block_table is not None:
            block_table = flat_block_table
            seq_lens = getattr(attn_metadata, "dense_mla_flat_seq_lens", None)
            query_start_loc = getattr(
                attn_metadata, "dense_mla_flat_query_start_loc", None
            )
            if seq_lens is None or query_start_loc is None:
                raise RuntimeError("B12X metadata is missing flattened decode rows.")

        batch = int(seq_lens.shape[0])
        total_q = int(q.shape[0])
        if total_q != batch:
            raise ValueError(
                "B12X requires one query row per prepared decode sequence, "
                f"got {total_q} rows for {batch} sequences."
            )

        metadata_dcp_world_size = int(
            getattr(attn_metadata, "dense_mla_dcp_world_size", self.dcp_world_size)
        )
        if metadata_dcp_world_size not in (1, self.dcp_world_size):
            raise ValueError(
                "B12X metadata uses an unsupported DCP KV shard count: "
                f"metadata={metadata_dcp_world_size}, runtime={self.dcp_world_size}."
            )
        effective_heads = self.num_heads * metadata_dcp_world_size
        kernel_heads = _kernel_query_heads(self.num_heads, metadata_dcp_world_size)

        actual_heads = int(q.shape[1])
        if actual_heads != effective_heads:
            raise ValueError(
                "B12X gathered an unexpected query-head count: "
                f"expected {effective_heads}, got {actual_heads}."
            )
        if kernel_heads == effective_heads:
            output = torch.empty(
                (total_q, effective_heads, self.kv_lora_rank),
                dtype=torch.bfloat16,
                device=q.device,
            )
        else:
            padded_q = getattr(attn_metadata, "dense_mla_padded_q", None)
            output = getattr(attn_metadata, "dense_mla_padded_output", None)
            if output is None or (kernel_heads != effective_heads and padded_q is None):
                raise RuntimeError(
                    "B12X metadata is missing caller-owned padded query buffers."
                )
            query_capacity = (
                int(padded_q.shape[0]) if padded_q is not None else int(output.shape[0])
            )
            if query_capacity < total_q or int(output.shape[0]) < total_q:
                raise ValueError(
                    "B12X padded query capacity is smaller than the decode "
                    f"batch: query={query_capacity}, output={output.shape[0]}, "
                    f"required={total_q}."
                )
            output = output[:total_q]
            if kernel_heads != effective_heads:
                assert padded_q is not None
                padded_q = padded_q[:total_q]
                if padded_q.dtype != q.dtype:
                    raise TypeError(
                        "B12X padded query dtype does not match the live query: "
                        f"buffer={padded_q.dtype}, query={q.dtype}."
                    )
                padded_q[:, :effective_heads].copy_(q)
                padded_q[:, effective_heads:].zero_()
                q = padded_q
        scratch = getattr(attn_metadata, "dense_mla_scratch", None)
        if scratch is None:
            raise RuntimeError(
                "B12X metadata is missing caller-owned dense MLA scratch."
            )
        quantized = q.dtype == torch.float8_e4m3fn
        # Direct CUDA graph capture fixes the launch grid and therefore uses
        # every planned split. Piecewise eager attention may omit only trailing
        # splits whose first 64-token chunk lies beyond every live sequence.
        active_splits = (
            int(plan.num_splits)
            if q.is_cuda and torch.cuda.is_current_stream_capturing()
            else _active_dense_mla_splits(
                plan,
                getattr(attn_metadata, "max_seq_len", None),
            )
        )
        binding = self._dense_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_c_and_k_pe_cache,
            output=output,
            page_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=query_start_loc[: batch + 1],
            q_scale=layer._q_scale if quantized else None,
            kv_scale=layer._k_scale if quantized else None,
            sm_scale=self.scale,
            active_splits=active_splits,
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
                    "B12X encountered an uncompiled layout during CUDA graph "
                    "capture; eager warmup did not exercise this cache layout."
                )
            self._dense_mla.compile(binding=binding)
            self._compiled_bindings.add(compile_key)

        output, lse = self._dense_mla.run(binding=binding)
        output = output[:, :effective_heads]
        lse = lse[:, :effective_heads]
        return output, lse


__all__ = [
    "B12xMLABackend",
    "B12xMLAImpl",
    "B12xMLAMetadata",
    "B12xMLAMetadataBuilder",
]
