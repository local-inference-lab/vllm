# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared B12X dense MLA with caller-owned DCP collectives and scratch."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_scratch_buffers,
    set_b12x_preparation_provider,
)
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf


def _local_cache_capacity(
    config, dcp_size: int, window: int = 0, page_size: int = 1
) -> int:
    interleave = config.parallel_config.cp_kv_cache_interleave_size
    capacity = (
        triton.cdiv(config.model_config.max_model_len, dcp_size * interleave)
        * interleave
    )
    # Draft windows retain the complete page containing their first token.
    return min(capacity, window + page_size - 1) if window else capacity


@triton.jit
def _flatten_decode_metadata(
    query_start,
    local_lengths,
    global_lengths,
    page_table,
    flat_lengths,
    flat_table,
    num_reqs,
    source_width,
    source_stride,
    target_width,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK: tl.constexpr,
    WINDOW: tl.constexpr = 0,
    PAGE_SIZE: tl.constexpr = 1,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # upper_bound(query_start, row) also skips zero-length padded requests.
    low = 0
    high = num_reqs
    while low < high:
        middle = (low + high) // 2
        end = tl.load(query_start + middle + 1)
        right = row >= end
        low = tl.where(right, middle + 1, low)
        high = tl.where(right, high, middle)
    active = low < num_reqs
    request = tl.minimum(low, num_reqs - 1).to(tl.int64)
    if CAUSAL:
        end = tl.load(query_start + request + 1)
        if DCP_SIZE > 1:
            length = tl.load(global_lengths + request) + row + 1 - end
            length = tl.maximum(length, 0)
            cycle = DCP_SIZE * INTERLEAVE
            remainder = tl.minimum(
                tl.maximum(length % cycle - DCP_RANK * INTERLEAVE, 0), INTERLEAVE
            )
            length = (length // cycle) * INTERLEAVE + remainder
        else:
            length = tl.load(local_lengths + request) + row + 1 - end
    else:
        length = tl.load(local_lengths + request)
    page_offset = 0
    if WINDOW > 0:
        page_offset = tl.maximum(length - WINDOW, 0) // PAGE_SIZE
        length -= page_offset * PAGE_SIZE
    if tl.program_id(1) == 0:
        tl.store(flat_lengths + row, tl.where(active, tl.maximum(length, 0), 0))
    source_columns = columns + page_offset
    pages = tl.load(
        page_table + request * source_stride + source_columns,
        mask=active & (source_columns < source_width) & (columns < target_width),
        other=0,
    )
    tl.store(flat_table + row * target_width + columns, pages, columns < target_width)


@dataclass
class B12xMLAMetadata(MLACommonMetadata):
    flat_block_table: torch.Tensor | None = None
    flat_seq_lens: torch.Tensor | None = None
    flat_query_start_loc: torch.Tensor | None = None


class B12xMLAMetadataBuilder(MLACommonMetadataBuilder[B12xMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=B12xMLAMetadata,
            supports_dcp_with_varlen=True,
        )
        self.supports_draft_decode_metadata_update = False
        self._dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self._draft_window = getattr(kv_cache_spec, "sliding_window", 0) or 0
        if self._draft_window and (
            not self.non_causal_multi_token_decode or self.dcp_world_size != 1
        ):
            raise ValueError("B12X_MLA windows require replicated non-causal draft KV")
        rows = vllm_config.scheduler_config.max_num_seqs * self.reorder_batch_threshold
        width = triton.cdiv(
            _local_cache_capacity(
                vllm_config, self.dcp_world_size, self._draft_window, self.page_size
            ),
            self.page_size,
        )
        self._flat_table = torch.empty((rows, width), dtype=torch.int32, device=device)
        self._flat_lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self._flat_query_start = torch.arange(
            rows + 1, dtype=torch.int32, device=device
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        decode = metadata.decode
        if decode is None:
            return metadata
        rows = metadata.num_decode_tokens
        if rows > self._flat_lengths.numel():
            raise ValueError(
                "B12X_MLA decode rows exceed the configured query capacity"
            )
        if (
            rows == metadata.num_decodes
            and decode.block_table.shape[1] <= self._flat_table.shape[1]
            and not self._draft_window
        ):
            return metadata
        if (
            metadata.causal
            and self.dcp_world_size > 1
            and decode.dcp_tot_seq_lens is None
        ):
            raise ValueError("Causal DCP verification requires global sequence lengths")
        flat_table = self._flat_table[:rows]
        flat_lengths = self._flat_lengths[:rows]
        _flatten_decode_metadata[(rows, triton.cdiv(flat_table.shape[1], 128))](
            metadata.query_start_loc,
            decode.seq_lens,
            decode.dcp_tot_seq_lens,
            decode.block_table,
            flat_lengths,
            flat_table,
            metadata.num_decodes,
            decode.block_table.shape[1],
            decode.block_table.stride(0),
            flat_table.shape[1],
            self.dcp_world_size,
            self._dcp_rank,
            self.cp_kv_cache_interleave_size,
            metadata.causal,
            128,
            WINDOW=self._draft_window,
            PAGE_SIZE=self.page_size,
        )
        metadata.flat_block_table = flat_table
        metadata.flat_seq_lens = flat_lengths
        metadata.flat_query_start_loc = self._flat_query_start[: rows + 1]
        return metadata


class B12xMLABackend(MLACommonBackend):
    supports_dcp_replicated: ClassVar[bool] = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA"

    @staticmethod
    def get_impl_cls():
        return B12xMLAImpl

    @staticmethod
    def get_builder_cls():
        return B12xMLAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576, 1088]

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


class B12xMLAImpl(MLACommonImpl[B12xMLAMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True

    def __init__(
        self,
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
    ):
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
            value is not None
            for value in (alibi_slopes, sliding_window, logits_soft_cap)
        ):
            raise NotImplementedError(
                "B12X_MLA does not support ALiBi, windows or logit caps"
            )
        if attn_type != AttentionType.DECODER or num_kv_heads != 1:
            raise NotImplementedError(
                "B12X_MLA requires decoder attention and one KV head"
            )
        from b12x.attention import dense_mla

        self._dense_mla = dense_mla
        self._config = get_current_vllm_config()
        self._plans = {}
        self._capacities = ()
        set_b12x_preparation_provider(self, self)

    def _cache_view(self, layer):
        cache = layer.kv_cache
        if self.kv_cache_dtype.startswith("fp8") and cache.dtype == torch.uint8:
            cache = cache.view(torch.float8_e4m3fn)
        return cache

    def get_b12x_preparation_units(self, layer, workload: B12xWorkload):
        cache = self._cache_view(layer)
        if not cache.numel():
            return ()
        speculative_config = self._config.speculative_config
        # Match AttentionMetadataBuilder's decode threshold: parallel drafting
        # admits the bonus token plus two speculative blocks as decode queries.
        spec_blocks = (
            2
            if (speculative_config is not None and speculative_config.parallel_drafting)
            else 1
        )
        max_rows = min(
            workload.max_tokens,
            workload.max_seqs * (1 + spec_blocks * workload.speculative_tokens),
        )
        self._capacities = tuple(
            sorted(
                {min(max_rows, capacity) for capacity in (1, 4, 16, 64, 256, max_rows)}
            )
        )
        heads = self.num_heads * self.dcp_world_size
        capacity = _local_cache_capacity(
            self._config,
            self.dcp_world_size,
            getattr(layer, "draft_kv_window", 0),
            cache.shape[1],
        )
        width = triton.cdiv(capacity, cache.shape[1])
        fp8 = cache.dtype == torch.float8_e4m3fn
        plans = {}
        requests = []
        for rows in self._capacities:
            caps = self._dense_mla.Caps(
                device=cache.device,
                mode="decode",
                kv_dtype=cache.dtype,
                num_q_heads=heads,
                page_size=cache.shape[1],
                max_total_q=rows,
                max_batch=rows,
                max_cache_tokens=capacity,
                max_page_table_width=width,
                num_cache_pages=cache.shape[0],
                head_dim=self.head_size,
                v_head_dim=self.kv_lora_rank,
                physical_record_width=cache.shape[2],
                use_cuda_graph=True,
            )

            def descriptor(shape, strides, dtype):
                return {
                    "shape": shape,
                    "strides": strides,
                    "dtype": str(dtype).removeprefix("torch."),
                    "alignment": 16,
                }

            scale_desc = descriptor((1,), (1,), torch.float32) if fp8 else None
            invocation = self._dense_mla.invocation_from_descriptors(
                caps,
                operands={
                    "q": descriptor(
                        (rows, heads, self.head_size),
                        (heads * self.head_size, self.head_size, 1),
                        cache.dtype,
                    ),
                    "kv_cache": descriptor(cache.shape, cache.stride(), cache.dtype),
                    "output": descriptor(
                        (rows, heads, self.kv_lora_rank),
                        (heads * self.kv_lora_rank, self.kv_lora_rank, 1),
                        torch.bfloat16,
                    ),
                    "page_table": descriptor((rows, width), (width, 1), torch.int32),
                    "cache_seqlens": descriptor((rows,), (1,), torch.int32),
                    "cu_seqlens_q": descriptor((rows + 1,), (1,), torch.int32),
                    "q_scale": scale_desc,
                    "kv_scale": scale_desc,
                },
            )
            plan = self._dense_mla.plan(caps, invocation=invocation)
            plans[rows] = plan
            requests.append(
                plan.request(
                    name=f"attention.dense_mla.{id(layer):x}.q{rows}",
                    prepare_call=lambda state, rows=rows: self._prepared_call(
                        layer, state, rows
                    ),
                )
            )
        self._plans = plans
        return (
            B12xPreparationUnit(
                name="B12X dense MLA",
                key=id(layer),
                requests=tuple(requests),
                stage="state",
                autotune=False,
            ),
        )

    def _prepared_call(self, layer, state, rows):
        from b12x.preparation import PreparedCall

        from vllm.v1.worker.workspace import current_workspace_manager

        cache = self._cache_view(layer)
        heads = self.num_heads * self.dcp_world_size
        # Reserve every declared capacity before graph capture locks storage.
        # Serial preparation calls borrow scratch rather than pinning one
        # allocation for each layer and capacity.
        scratch_bytes = max(
            spec.nbytes
            for plan in self._plans.values()
            for spec in plan.scratch_specs()
        )
        manager = current_workspace_manager()
        manager.get_simultaneous(((scratch_bytes,), torch.uint8))
        manager.reserve_by_lane()
        (scratch,) = get_b12x_scratch_buffers(state)
        q = torch.zeros(
            (rows, heads, self.head_size), dtype=cache.dtype, device=cache.device
        )
        output = torch.empty(
            (rows, heads, self.kv_lora_rank), dtype=torch.bfloat16, device=cache.device
        )
        # Empty sequences prime the launch ABI without reading uninitialized KV.
        lengths = torch.zeros(rows, dtype=torch.int32, device=cache.device)
        cu = torch.arange(rows + 1, dtype=torch.int32, device=cache.device)
        pages = torch.zeros((rows, 1), dtype=torch.int32, device=cache.device)
        fp8 = cache.dtype == torch.float8_e4m3fn
        binding = state.bind(
            scratch=scratch,
            q=q,
            kv_cache=cache,
            output=output,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            q_scale=layer._q_scale if fp8 else None,
            kv_scale=layer._k_scale if fp8 else None,
            sm_scale=self.scale,
        )
        state.prime(binding)
        return PreparedCall(
            run=lambda: state.run(binding),
            output=output,
            owners=(scratch, q, output, lengths, cu, pages, binding),
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        decode = attn_metadata.decode
        if decode is None:
            raise ValueError("B12X_MLA requires decode metadata")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()
        rows = q.shape[0]
        capacity = next((size for size in self._capacities if rows <= size), None)
        if capacity is None:
            raise PreparationResourceUnavailableError(
                f"B12X_MLA query rows {rows} exceed prepared capacities "
                f"{self._capacities}"
            )
        plan = self._plans[capacity]
        table = attn_metadata.flat_block_table
        if table is None:
            table, lengths = decode.block_table, decode.seq_lens
            cu = attn_metadata.query_start_loc[: lengths.shape[0] + 1]
        else:
            lengths, cu = (
                attn_metadata.flat_seq_lens,
                attn_metadata.flat_query_start_loc,
            )
        if rows != lengths.shape[0]:
            raise ValueError("B12X_MLA requires one flattened sequence per query")
        output = torch.empty(
            (rows, q.shape[1], self.kv_lora_rank), dtype=torch.bfloat16, device=q.device
        )
        (scratch,) = get_b12x_scratch_buffers(plan)
        fp8 = kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fn
        binding = self._dense_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_c_and_k_pe_cache,
            output=output,
            page_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            q_scale=layer._q_scale if fp8 else None,
            kv_scale=layer._k_scale if fp8 else None,
            sm_scale=self.scale,
        )
        return self._dense_mla.run(plan=plan, binding=binding)
