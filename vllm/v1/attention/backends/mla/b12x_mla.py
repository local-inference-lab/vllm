# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native B12X dense MLA decode backend for Kimi K3 on SM120/SM121."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm import envs
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
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_a2a_lse_reduce,
    dcp_b12x_all_gather_heads,
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
                "B12X_MLA requires native E4M3 FP8 KV storage; "
                f"this platform selected {fp8_dtype}."
            )
        return fp8_dtype
    raise ValueError(
        f"B12X_MLA supports only BF16 or E4M3 KV cache storage, got {cache_dtype!r}."
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
            "B12X_MLA requires positive DCP and KV-interleave sizes, got "
            f"DCP={dcp_size}, interleave={interleave}."
        )
    max_model_len = int(vllm_config.model_config.max_model_len)
    partitions = dcp_size * interleave
    return ((max_model_len + partitions - 1) // partitions) * interleave


def _kernel_query_heads(local_heads: int, dcp_size: int = 1) -> int:
    """Return the tiled head count after an optional DCP query gather."""
    if local_heads <= 0 or dcp_size <= 0:
        raise ValueError(
            "B12X_MLA requires positive query-head and DCP sizes, got "
            f"heads={local_heads}, DCP={dcp_size}."
        )
    effective_heads = local_heads * dcp_size
    if dcp_size > 1 and effective_heads % _B12X_QUERY_HEAD_TILE:
        raise ValueError(
            "B12X_MLA requires a multiple of eight query heads after DCP "
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
            "B12X_MLA received invalid split geometry: "
            f"splits={num_splits}, chunks_per_split={chunks_per_split}."
        )
    if max_seq_len is None:
        return num_splits
    valid_chunks = max(1, (max(0, int(max_seq_len)) + 63) // 64)
    return min(
        num_splits,
        (valid_chunks + chunks_per_split - 1) // chunks_per_split,
    )


def _dense_mla_plan_row_caps(max_rows: int) -> tuple[int, ...]:
    """Return CUDA-graph-friendly row capacities through ``max_rows``."""
    if max_rows <= 0:
        raise ValueError("dense MLA row capacity must be positive")
    caps: list[int] = []
    row_cap = 1
    while row_cap < max_rows:
        caps.append(row_cap)
        row_cap *= 2
    caps.append(max_rows)
    return tuple(caps)


def _select_dense_mla_plan(
    plans: dict[int, Any],
    total_rows: int,
) -> Any:
    """Select the smallest launch plan that covers the live query rows."""
    row_caps = tuple(sorted(plans))
    index = bisect_left(row_caps, total_rows)
    if total_rows <= 0 or index >= len(row_caps):
        raise ValueError(
            "B12X_MLA query rows exceed the planned capacities: "
            f"rows={total_rows}, capacities={row_caps}"
        )
    return plans[row_caps[index]]


def _create_dense_mla_plan(
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    page_size: int,
    num_q_heads: int,
    max_total_q: int | None = None,
    max_batch: int | None = None,
    mode: str = "decode",
    uses_query_cache_seqlens: bool = False,
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
    max_batch = int(max_total_q if max_batch is None else max_batch)
    dcp_size = int(
        vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size is None
        else dcp_size
    )

    def local_tokens(global_tokens: int) -> int:
        return (max(int(global_tokens), 0) + dcp_size - 1) // dcp_size

    sparse_stride = int(envs.VLLM_K3_DYNAMIC_SPARSE_STRIDE)
    if sparse_stride > 1:
        logger.warning_once(
            "Kimi-K3 dynamic sparse MLA is enabled with stride=%d. This "
            "changes attention semantics and requires model-quality "
            "qualification; stride=1 is the exact production default.",
            sparse_stride,
        )
    sparse_min_tokens = local_tokens(envs.VLLM_K3_DYNAMIC_SPARSE_MIN_TOKENS)
    sparse_sink_chunks = (
        local_tokens(envs.VLLM_K3_DYNAMIC_SPARSE_SINK_TOKENS) + 63
    ) // 64
    sparse_recent_chunks = (
        local_tokens(envs.VLLM_K3_DYNAMIC_SPARSE_RECENT_TOKENS) + 63
    ) // 64
    # A local DCP shard length can differ from its peers at an interleave
    # boundary. Until the kernel accepts a global refresh clock, disabling the
    # periodic dense refresh under DCP avoids mixing dense and sparse shards in
    # one exact LSE reduction. The sparse policy itself remains rank-consistent.
    sparse_refresh_interval = (
        local_tokens(envs.VLLM_K3_DYNAMIC_SPARSE_REFRESH_INTERVAL)
        if dcp_size == 1
        else 0
    )
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
        mode=mode,
        dtype=torch.bfloat16,
        kv_dtype=_planned_kv_dtype(vllm_config),
        num_q_heads=num_q_heads,
        page_size=page_size,
        max_total_q=max_total_q,
        max_batch=max_batch,
        max_cache_tokens=max_cache_tokens,
        max_page_table_width=_page_table_width(max_cache_tokens, page_size),
        num_cache_pages=_MAX_I32,
        use_cuda_graph=True,
        uses_query_cache_seqlens=uses_query_cache_seqlens,
        sparse_stride=sparse_stride,
        sparse_min_tokens=sparse_min_tokens,
        sparse_sink_chunks=sparse_sink_chunks,
        sparse_recent_chunks=sparse_recent_chunks,
        sparse_refresh_interval=sparse_refresh_interval,
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
    dense_mla_verify_block_table: torch.Tensor | None = None
    dense_mla_query_cache_seq_lens: torch.Tensor | None = None
    dense_mla_dcp_world_size: int = 1


class B12xMLAMetadataBuilder(MLACommonMetadataBuilder[B12xMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_direct_dcp_kv_gather: ClassVar[bool] = True

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
                "B12X_MLA query capacity exceeds its limit: "
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
        self._dense_mla_plans = {
            rows: _create_dense_mla_plan(
                vllm_config,
                device,
                page_size=self.page_size,
                num_q_heads=self._kernel_heads,
                max_total_q=rows,
                dcp_size=self.dcp_world_size,
                max_cache_tokens=max_cache_tokens,
            )
            for rows in _dense_mla_plan_row_caps(max_dense_mla_rows)
        }
        self._dense_mla_verify_plans: dict[int, Any] = {}
        if _planned_kv_dtype(vllm_config) == torch.float8_e4m3fn:
            self._dense_mla_verify_plans = {
                batch: _create_dense_mla_plan(
                    vllm_config,
                    device,
                    page_size=self.page_size,
                    num_q_heads=self._kernel_heads,
                    max_total_q=batch * 4,
                    max_batch=batch,
                    mode="verify",
                    uses_query_cache_seqlens=True,
                    dcp_size=self.dcp_world_size,
                    max_cache_tokens=max_cache_tokens,
                )
                for batch in range(
                    1,
                    int(vllm_config.scheduler_config.max_num_seqs) + 1,
                )
            }
        self._dense_mla_plan = self._dense_mla_plans[max_dense_mla_rows]
        workspace_specs = [
            plan.shapes_and_dtypes()
            for plan in (
                *self._dense_mla_plans.values(),
                *self._dense_mla_verify_plans.values(),
            )
        ]
        if any(len(specs) != 1 for specs in workspace_specs):
            raise RuntimeError("B12X_MLA expected exactly one scratch buffer per plan.")
        scratch_dtype = workspace_specs[0][0][1]
        if any(specs[0][1] != scratch_dtype for specs in workspace_specs):
            raise RuntimeError("B12X_MLA plan scratch dtypes do not match.")
        scratch_shape = max(
            (specs[0][0] for specs in workspace_specs),
            key=lambda shape: shape[0],
        )
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
        if self._kernel_heads != self.num_heads:
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
            "B12X dense K3 MLA plans: local_heads=%d, effective_heads=%d, "
            "kernel_heads=%d, page_size=%d, "
            "max_decode_rows=%d, max_cache_tokens=%d, rows/splits=%s, "
            "verify_batch/splits=%s",
            self.num_heads,
            self._effective_heads,
            self._kernel_heads,
            self.page_size,
            max_dense_mla_rows,
            max_cache_tokens,
            ",".join(
                f"{rows}/{plan.num_splits}"
                for rows, plan in self._dense_mla_plans.items()
            ),
            ",".join(
                f"{batch}/{plan.num_splits}"
                for batch, plan in self._dense_mla_verify_plans.items()
            )
            or "disabled",
        )

    def _materialize_query_cache_seq_lens(
        self,
        metadata: B12xMLAMetadata,
        decode_metadata: Any,
        *,
        query_len: int,
        total_q: int,
    ) -> torch.Tensor:
        flat_lens = self._dense_mla_flat_seq_lens[:total_q]
        if not metadata.causal:
            flat_lens.copy_(
                decode_metadata.seq_lens[:, None].expand(-1, query_len).reshape(total_q)
            )
            return flat_lens

        offsets = self._dense_mla_causal_offsets[-query_len:]
        if self.dcp_world_size == 1:
            torch.add(
                decode_metadata.seq_lens[:, None],
                offsets,
                out=flat_lens.view(metadata.num_decodes, query_len),
            )
            return flat_lens

        global_source_lens = decode_metadata.dcp_tot_seq_lens
        if global_source_lens is None:
            raise RuntimeError(
                "B12X_MLA causal DCP verification requires global decode "
                "sequence lengths."
            )
        assert self._dense_mla_flat_global_seq_lens is not None
        assert self._dense_mla_flat_dcp_remainder is not None
        global_flat_lens = self._dense_mla_flat_global_seq_lens[:total_q]
        torch.add(
            global_source_lens[:, None],
            offsets,
            out=global_flat_lens.view(metadata.num_decodes, query_len),
        )
        virtual_block = self.dcp_world_size * self.cp_kv_cache_interleave_size
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
        return flat_lens

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
        live_rows = max(1, int(metadata.num_decode_tokens))
        plans = getattr(
            self,
            "_dense_mla_plans",
            {self._max_dense_mla_rows: self._dense_mla_plan},
        )
        metadata.dense_mla_plan = _select_dense_mla_plan(plans, live_rows)
        metadata.dense_mla_scratch = self._dense_mla_scratch
        metadata.dense_mla_padded_q = self._dense_mla_padded_q
        metadata.dense_mla_padded_output = self._dense_mla_padded_output
        metadata.dense_mla_dcp_world_size = self.dcp_world_size
        decode_metadata = metadata.decode
        if decode_metadata is None or metadata.num_decodes <= 0:
            return metadata
        multi_query = metadata.num_decode_tokens > metadata.num_decodes
        table_too_wide = int(decode_metadata.block_table.shape[1]) > int(
            self._dense_mla_plan.caps.max_page_table_width
        )
        if not (multi_query or table_too_wide):
            return metadata

        total_q = int(metadata.num_decode_tokens)
        if total_q > self._max_dense_mla_rows:
            raise ValueError(
                "B12X_MLA query block exceeds its flattened capacity: "
                f"rows={total_q}, capacity={self._max_dense_mla_rows}."
            )
        if total_q % metadata.num_decodes:
            raise ValueError(
                "B12X_MLA requires a uniform query block, got "
                f"tokens={total_q}, requests={metadata.num_decodes}."
            )
        query_len = total_q // metadata.num_decodes
        source_table = decode_metadata.block_table
        flat_lens = self._materialize_query_cache_seq_lens(
            metadata,
            decode_metadata,
            query_len=query_len,
            total_q=total_q,
        )
        verify_plans = getattr(self, "_dense_mla_verify_plans", {})
        tiled_verify = (
            metadata.causal and query_len == 4 and metadata.num_decodes in verify_plans
        )
        if tiled_verify:
            verify_table = self._dense_mla_flat_block_table[: metadata.num_decodes]
            source_width = min(
                int(source_table.shape[1]),
                int(verify_table.shape[1]),
            )
            verify_table[:, :source_width].copy_(source_table[:, :source_width])
            metadata.dense_mla_plan = verify_plans[metadata.num_decodes]
            metadata.dense_mla_verify_block_table = verify_table
            metadata.dense_mla_query_cache_seq_lens = flat_lens
            return metadata

        flat_table = self._dense_mla_flat_block_table[:total_q]
        source_width = min(int(source_table.shape[1]), int(flat_table.shape[1]))
        flat_table[:, :source_width].copy_(
            source_table[:, None, :source_width]
            .expand(-1, query_len, -1)
            .reshape(total_q, source_width)
        )
        metadata.dense_mla_flat_block_table = flat_table
        metadata.dense_mla_flat_seq_lens = flat_lens
        metadata.dense_mla_flat_query_start_loc = self._dense_mla_flat_query_start_loc[
            : total_q + 1
        ]
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
            return "B12X_MLA requires a B12X build that provides dense_mla"

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        if model_config is None:
            return None
        hf_text_config = model_config.hf_text_config
        if getattr(hf_text_config, "model_type", None) not in (
            "kimi_linear",
            "k3_dspark",
        ):
            return "B12X_MLA currently supports only Kimi K3 and K3 DSpark"

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
        if parallel_config.prefill_context_parallel_size != 1:
            return "B12X_MLA does not support prefill context parallelism"
        dcp_size = int(parallel_config.decode_context_parallel_size)
        local_heads = model_config.get_num_attention_heads(parallel_config)
        try:
            _kernel_query_heads(local_heads, dcp_size)
        except ValueError as exc:
            return str(exc)
        if vllm_config.scheduler_config.max_num_seqs > _MAX_B12X_QUERY_ROWS:
            return (
                "B12X_MLA max_num_seqs exceeds its 1024-row decode capacity: "
                f"{vllm_config.scheduler_config.max_num_seqs}"
            )
        local_cache_tokens = _max_dcp_local_cache_tokens(vllm_config)
        if local_cache_tokens > _MAX_B12X_CACHE_TOKENS:
            return (
                "B12X_MLA local DCP cache exceeds its 1048576-token capacity: "
                f"{local_cache_tokens}"
            )
        return None

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


class B12xMLAImpl(MLACommonImpl[B12xMLAMetadata]):
    can_return_lse_for_decode: bool = True
    owns_decode_dcp_collectives: bool = True

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
        self._dcp_comm_backend = vllm_config.parallel_config.dcp_comm_backend
        self._dcp_max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        self.dcp_q_replicate = False
        self._compiled_bindings: set[tuple[object, ...]] = set()

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

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        query_cache_seq_lens = getattr(
            attn_metadata,
            "dense_mla_query_cache_seq_lens",
            None,
        )
        verify_block_table = getattr(
            attn_metadata,
            "dense_mla_verify_block_table",
            None,
        )
        if verify_block_table is not None:
            block_table = verify_block_table
        flat_block_table = getattr(attn_metadata, "dense_mla_flat_block_table", None)
        if flat_block_table is not None:
            block_table = flat_block_table
            seq_lens = getattr(attn_metadata, "dense_mla_flat_seq_lens", None)
            query_start_loc = getattr(
                attn_metadata, "dense_mla_flat_query_start_loc", None
            )
            if seq_lens is None or query_start_loc is None:
                raise RuntimeError(
                    "B12X_MLA metadata is missing flattened decode rows."
                )

        batch = int(seq_lens.shape[0])
        total_q = int(q.shape[0])
        if query_cache_seq_lens is None and total_q != batch:
            raise ValueError(
                "B12X_MLA requires one query row per prepared decode sequence, "
                f"got {total_q} rows for {batch} sequences."
            )
        metadata_dcp_world_size = int(
            getattr(attn_metadata, "dense_mla_dcp_world_size", self.dcp_world_size)
        )
        if metadata_dcp_world_size not in (1, self.dcp_world_size):
            raise ValueError(
                "B12X_MLA metadata uses an unsupported DCP KV shard count: "
                f"metadata={metadata_dcp_world_size}, runtime={self.dcp_world_size}."
            )
        effective_heads = self.num_heads * metadata_dcp_world_size
        kernel_heads = _kernel_query_heads(self.num_heads, metadata_dcp_world_size)
        qrep_decode = self.dcp_q_replicate and metadata_dcp_world_size > 1
        expected_input_heads = effective_heads if qrep_decode else self.num_heads
        if int(q.shape[1]) != expected_input_heads:
            raise ValueError(
                f"B12X_MLA expected {expected_input_heads} query heads, "
                f"got {q.shape[1]}."
            )

        dcp_group = None
        if metadata_dcp_world_size > 1:
            dcp_group = get_dcp_group()
            if not qrep_decode:
                gathered_q = getattr(attn_metadata, "dense_mla_padded_q", None)
                if gathered_q is None:
                    raise RuntimeError(
                        "B12X_MLA DCP metadata is missing caller-owned query storage."
                    )
                if int(gathered_q.shape[0]) < total_q:
                    raise ValueError(
                        "B12X_MLA DCP query capacity is smaller than the decode "
                        f"batch: capacity={gathered_q.shape[0]}, required={total_q}."
                    )
                if gathered_q.dtype != q.dtype:
                    raise TypeError(
                        "B12X_MLA DCP query storage does not match the live query: "
                        f"buffer={gathered_q.dtype}, query={q.dtype}."
                    )
                gathered_q = gathered_q[:total_q, :effective_heads]
                q = dcp_b12x_all_gather_heads(
                    q,
                    dcp_group,
                    max_batch_size=self._dcp_max_batch_size,
                    output_head_dim=self.kv_lora_rank,
                    out=gathered_q,
                )

        actual_heads = int(q.shape[1])
        if actual_heads != effective_heads:
            raise ValueError(
                "B12X_MLA gathered an unexpected query-head count: "
                f"expected {effective_heads}, got {actual_heads}."
            )
        if kernel_heads == effective_heads and metadata_dcp_world_size == 1:
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
                    "B12X_MLA metadata is missing caller-owned padded query buffers."
                )
            query_capacity = (
                int(padded_q.shape[0]) if padded_q is not None else int(output.shape[0])
            )
            if query_capacity < total_q or int(output.shape[0]) < total_q:
                raise ValueError(
                    "B12X_MLA padded query capacity is smaller than the decode "
                    f"batch: query={query_capacity}, output={output.shape[0]}, "
                    f"required={total_q}."
                )
            output = output[:total_q]
            if kernel_heads != effective_heads:
                assert padded_q is not None
                padded_q = padded_q[:total_q]
                if padded_q.dtype != q.dtype:
                    raise TypeError(
                        "B12X_MLA padded query dtype does not match the live query: "
                        f"buffer={padded_q.dtype}, query={q.dtype}."
                    )
                padded_q[:, :effective_heads].copy_(q)
                padded_q[:, effective_heads:].zero_()
                q = padded_q
        scratch = getattr(attn_metadata, "dense_mla_scratch", None)
        if scratch is None:
            raise RuntimeError(
                "B12X_MLA metadata is missing caller-owned dense MLA scratch."
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
            query_cache_seqlens=query_cache_seq_lens,
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
                    "B12X_MLA encountered an uncompiled layout during CUDA graph "
                    "capture; eager warmup did not exercise this cache layout."
                )
            self._dense_mla.compile(binding=binding)
            self._compiled_bindings.add(compile_key)

        output, lse = self._dense_mla.run(binding=binding)
        output = output[:, :effective_heads]
        lse = lse[:, :effective_heads]
        if dcp_group is None:
            return output, lse
        if self._dcp_comm_backend == "a2a":
            # B12X dense MLA writes -inf LSE for rows with no visible local KV
            # chunks. The B12X DCP reduction converts every non-finite LSE to
            # zero weight before reading its payload, so empty rows already
            # contribute the exact neutral value without a sanitization pass.
            reduced = dcp_a2a_lse_reduce(
                output,
                lse,
                dcp_group,
                is_lse_base_on_e=True,
                use_b12x=True,
                b12x_max_batch_size=self._dcp_max_batch_size,
                b12x_query_head_dim=_K3_ABSORBED_HEAD_DIM,
            )
        else:
            reduced = cp_lse_ag_out_rs(
                output,
                lse,
                dcp_group,
                is_lse_base_on_e=True,
            )
        return reduced, None


__all__ = [
    "B12xMLABackend",
    "B12xMLAImpl",
    "B12xMLAMetadata",
    "B12xMLAMetadataBuilder",
]
