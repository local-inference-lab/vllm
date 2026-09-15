# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x DSA indexer for non-compressed sparse MLA models."""

import os
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    get_b12x_dsa_indexer,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    split_indexer_prefill_chunks,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.workspace import current_workspace_manager

_INDEX_HEAD_DIM = 128
_INDEX_SCALE_BYTES = 4
_INDEX_PAGE_SIZE = 64
_INDEX_PAGE_WIDTH = _INDEX_PAGE_SIZE * (_INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
_PREFILL_PROFILE_SUPERTILE_K = 32 * 1024


def _prefill_profile_q_rows(max_q_rows: int) -> int:
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    supertile_k = int(
        os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K", _PREFILL_PROFILE_SUPERTILE_K)
    )
    supertile_k = max(supertile_k, 256)
    return min(max(int(max_q_rows), 1), max(1, max_logits_elems // supertile_k))


def _is_current_stream_capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


@dataclass
class B12xIndexerDecodeMetadata(DeepSeekV32IndexerDecodeMetadata):
    active_width: torch.Tensor | None = None


class B12xIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, block_table_width=block_table_width, **kwargs)
        self.use_flattening = False
        self.supports_varlen = False
        self.active_width_buffer = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )

    def _supports_native_decode(self, next_n: int) -> bool:
        return True

    def _split_prefill_chunks(
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        num_decodes: int,
        max_logits_bytes: int,
    ) -> list[tuple[slice, slice]]:
        return [
            chunk
            for prefill_idx in range(len(prefill_query_lens_cpu))
            for chunk in split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[
                    num_decodes + prefill_idx : num_decodes + prefill_idx + 1
                ],
                prefill_query_lens_cpu[prefill_idx : prefill_idx + 1],
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes + prefill_idx,
            )
        ]

    def build(self, *args, **kwargs) -> DeepseekV32IndexerMetadata:
        metadata = super().build(*args, **kwargs)
        self.active_width_buffer.fill_(int(metadata.max_seq_len))
        if metadata.decode is not None:
            decode = metadata.decode
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            fields = vars(decode).copy()
            fields["seq_lens"] = seq_lens
            fields["schedule_metadata"] = None
            metadata.decode = B12xIndexerDecodeMetadata(
                **fields,
                active_width=self.active_width_buffer,
            )
        return metadata


class B12xIndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "B12X_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[B12xIndexerMetadataBuilder]:
        return B12xIndexerMetadataBuilder


class B12xIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[B12xIndexerBackend]:
        return B12xIndexerBackend


def _require_b12x_indexer() -> Any:
    module = get_b12x_dsa_indexer()
    if module is None:
        raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
    if not module.is_supported():
        raise RuntimeError("B12X sparse indexer is not supported on this device.")
    if int(module.PAGED_INDEX_PAGE_SIZE) != _INDEX_PAGE_SIZE:
        raise RuntimeError(
            "B12X sparse indexer page size changed: expected "
            f"{_INDEX_PAGE_SIZE}, got {module.PAGED_INDEX_PAGE_SIZE}."
        )
    for name in (
        "Caps",
        "bind",
        "plan",
        "run",
        "scratch_specs",
    ):
        getattr(module, name)
    return module


def _flatten_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_tail = (_INDEX_PAGE_SIZE, _INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.uint8
        or tuple(kv_cache.shape[1:]) != expected_tail
    ):
        raise RuntimeError(
            "B12X indexer cache must have shape "
            f"[num_blocks, {expected_tail[0]}, {expected_tail[1]}] and dtype "
            f"uint8, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}."
        )
    if kv_cache.stride(1) != expected_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "B12X indexer cache requires contiguous page payloads, got stride "
            f"{tuple(kv_cache.stride())}."
        )
    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_paged_topk(
    *,
    module: Any,
    plan: object,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    active_width: torch.Tensor | None,
    output: torch.Tensor,
    scores: torch.Tensor | None,
) -> None:
    if active_width is None:
        raise RuntimeError("B12X DSA requires a device active-width scalar.")
    scratch = current_workspace_manager().get_simultaneous(
        *((spec.shape, spec.dtype) for spec in module.scratch_specs(plan, device=q.device))
    )
    binding = module.bind(
        plan,
        scratch=scratch,
        q_fp8=q,
        query_weights=weights,
        index_k_cache=_flatten_index_cache(kv_cache),
        page_table=block_table,
        cache_lengths=seq_lens,
        active_width=active_width,
        output_indices=output,
        output_scores=scores,
    )
    module.run(binding)


@triton.jit
def _pack_dcp_candidates_kernel(
    indices,
    scores,
    packed,
    index_stride,
    score_stride,
    packed_row_stride,
    packed_col_stride,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    interleave: tl.constexpr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    mask = col < topk
    local_idx = tl.load(indices + row * index_stride + col, mask=mask, other=-1)
    score = tl.load(scores + row * score_stride + col, mask=mask, other=-float("inf"))
    valid = local_idx >= 0
    safe_idx = tl.maximum(local_idx, 0)
    global_idx = (
        (safe_idx // interleave) * (dcp_world_size * interleave)
        + dcp_rank * interleave
        + safe_idx % interleave
    )
    global_idx = tl.where(valid, global_idx, -1)
    score = tl.where(valid, score, -float("inf"))
    base = packed + row * packed_row_stride + col * packed_col_stride
    tl.store(base, score, mask=mask)
    tl.store(base + 1, global_idx.to(tl.float32), mask=mask)


def _merge_dcp_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave: int,
) -> None:
    if dcp_world_size <= 1 or indices.numel() == 0:
        return
    topk = int(indices.shape[1])
    if topk not in (512, 1024, 2048):
        raise RuntimeError(
            "B12X DCP indexer merge requires index_topk in (512, 1024, 2048), "
            f"got {topk}."
        )
    packed = torch.empty(
        (indices.shape[0], topk, 2), dtype=torch.float32, device=indices.device
    )
    _pack_dcp_candidates_kernel[(indices.shape[0], triton.cdiv(topk, 512))](
        indices,
        scores,
        packed,
        indices.stride(0),
        scores.stride(0),
        packed.stride(0),
        packed.stride(1),
        dcp_rank,
        dcp_world_size,
        interleave,
        topk,
        512,
        num_warps=8,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )

    stable_topk_from_gathered_candidates_cutedsl(gathered, topk, out=indices)


class B12xSparseIndexer(nn.Module):
    """Non-compressed FP8 DSA indexer consuming only installed executions."""

    def __init__(
        self, k_cache, quant_block_size: int, scale_fmt: str, topk_tokens: int,
        head_dim: int, max_model_len: int, max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None, skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False, compress_ratio: int = 1,
        num_q_heads: int | None = None, output_physical_slots: bool = False,
    ) -> None:
        super().__init__()
        del quant_block_size, scale_fmt, max_total_seq_len
        if not skip_k_cache_insert or use_fp4_cache or compress_ratio != 1:
            raise ValueError("B12X requires the fused FP8 non-compressed DSA cache path.")
        if head_dim != _INDEX_HEAD_DIM or topk_indices_buffer is None:
            raise ValueError("B12X requires the FP8 index head layout and output buffer.")
        if num_q_heads is None or int(num_q_heads) <= 0:
            raise ValueError("B12X indexing requires a positive index query head count.")
        self._module, self.k_cache = _require_b12x_indexer(), k_cache
        self.topk_tokens, self.max_model_len = int(topk_tokens), int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        self.output_physical_slots, self.num_q_heads = bool(output_physical_slots), int(num_q_heads)
        self.active_width_cap = torch.full((1,), self.max_model_len, dtype=torch.int32, device=topk_indices_buffer.device)
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        parallel = config.parallel_config
        self._max_num_seqs = int(config.scheduler_config.max_num_seqs)
        self._max_page_table_width = get_block_table_width(max(1, (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE), _INDEX_PAGE_SIZE)
        self.dcp_world_size = parallel.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel.cp_kv_cache_interleave_size
        self._prepared_plans: dict[tuple[str, int], object] = {}
        self._preparation_prefix = f"{getattr(k_cache, 'prefix', type(self).__qualname__)}.dsa_indexer"
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)
        self._register_score_collectives()

    def _register_score_collectives(self) -> None:
        if self.dcp_world_size <= 1:
            return
        from vllm.distributed.parallel_state import register_b12x_collective_describer
        from vllm.distributed.device_communicators.b12x_pcie_all_reduce import B12xPcieInvocation
        def describe(requirements):
            return tuple(B12xPcieInvocation(name=f"{self._preparation_prefix}.score_all_reduce.m{rows}.lane{requirements.workspace_lane}", operation="all_reduce", shape=(rows, self.topk_tokens), dtype=torch.float32) for rows in requirements.token_counts)
        register_b12x_collective_describer(self, describe, group=get_dcp_group())

    def _request_name(self, mode: str, rows: int) -> str:
        return f"{self._preparation_prefix}.{mode}.m{rows}"

    def _page_table_width(self, max_model_len: int) -> int:
        return max(self._max_page_table_width, (max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE)

    def _caps(self, mode: str, rows: int, width: int):
        return self._module.Caps(device=self.topk_indices_buffer.device, num_q_heads=self.num_q_heads, max_q_rows=rows, max_page_table_width=width, topk=self.topk_tokens, mode=mode, max_batch=rows if mode == "decode" else self._max_num_seqs, output_index_space="physical" if self.output_physical_slots else "logical")

    def _invocation(self, caps, scores: bool):
        descriptor = lambda shape, dtype: {"shape": tuple(shape), "strides": tuple(torch.empty(shape, device="meta").stride()), "dtype": dtype, "alignment": 16}
        return self._module.invocation_from_descriptors(caps, operands={
            "q_fp8": descriptor((caps.max_q_rows, caps.num_q_heads, _INDEX_HEAD_DIM), "float8_e4m3fn"),
            "query_weights": descriptor((caps.max_q_rows, caps.num_q_heads), "float32"),
            "index_k_cache": descriptor((max(int(self.k_cache.kv_cache.shape[0]), 1), _INDEX_PAGE_WIDTH), "uint8"),
            "page_table": descriptor((caps.max_q_rows, caps.max_page_table_width), "int32"),
            "cache_lengths": descriptor((caps.max_q_rows,), "int32"), "active_width": descriptor((1,), "int32"),
            "output_indices": descriptor((caps.max_q_rows, self.topk_tokens), "int32"),
            "output_scores": descriptor((caps.max_q_rows, self.topk_tokens), "float32") if scores else None,
        })

    def _declare_plan(self, caps):
        return self._module.plan(caps, invocation=self._invocation(caps, self.dcp_world_size > 1))

    def _prepare_call(self, mode: str, caps):
        from b12x.preparation import PreparedCall
        cache = self.k_cache.kv_cache

        def make_call(state):
            # Index selection only reads the published cache.  Use its
            # first physical page rather than manufacturing a cache or
            # copying the whole pool for a timing trial.
            q = torch.empty((caps.max_q_rows, caps.num_q_heads, _INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=caps.device)
            weights = torch.empty((caps.max_q_rows, caps.num_q_heads), dtype=torch.float32, device=caps.device)
            lengths = torch.full((caps.max_q_rows,), min(self.max_model_len, _INDEX_PAGE_SIZE), dtype=torch.int32, device=caps.device)
            pages = torch.zeros((caps.max_q_rows, caps.max_page_table_width), dtype=torch.int32, device=caps.device)
            output = torch.empty((caps.max_q_rows, self.topk_tokens), dtype=torch.int32, device=caps.device)
            scores = torch.empty_like(output, dtype=torch.float32) if self.dcp_world_size > 1 else None
            # Trial and prepare factories own their scratch; the
            # runtime binding in _run_paged_topk draws from the
            # workspace manager instead.
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=caps.device)
                for spec in state.layout.scratch_specs()
            )
            binding = state.bind(scratch=scratch, real_page_table=pages, cache_seqlens_int32=lengths, active_width=self.active_width_cap, expected_num_q_heads=caps.num_q_heads, shared_page_table=mode == "prefill", output_physical_slots=self.output_physical_slots)
            def produce():
                q.fill_(1)
                weights.fill_(1)
            return PreparedCall(
                run=lambda: state.run(binding, q_fp8=q, query_weights=weights, index_k_cache=_flatten_index_cache(cache), output_indices=output, output_scores=scores),
                produce=produce,
                owners=(q, weights, lengths, pages, output, scores, scratch, binding),
            )

        return make_call

    def _plan(self, mode: str, rows: int) -> object:
        """Reuse a prepared capacity within the same native execution mode."""
        rows = int(rows)
        capacity = min(
            (count for plan_mode, count in self._prepared_plans
             if plan_mode == mode and count >= rows), default=rows,
        )
        plan = self._prepared_plans.get((mode, capacity))
        if plan is None:
            caps = self._caps(mode, capacity, self._page_table_width(self.max_model_len))
            plan = self._declare_plan(caps)
            self._prepared_plans[(mode, capacity)] = plan
        return plan

    def get_b12x_preparation_units(self, layer: torch.nn.Module, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        if layer is not self:
            raise ValueError("DSA indexer preparation owner mismatch")
        cache = self.k_cache.kv_cache
        if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
            return ()
        width = self._page_table_width(workload.max_model_len)
        requests = []
        prepared_plans: dict[tuple[str, int], object] = {}
        capacities = {
            "decode": sorted({self._max_num_seqs, *workload.fixed_token_counts}),
            "prefill": (_prefill_profile_q_rows(workload.max_tokens),),
        }
        for mode, counts in capacities.items():
            for rows in counts:
                caps = self._caps(mode, rows, width)
                plan = self._declare_plan(caps)
                prepared_plans[(mode, rows)] = plan
                call = self._prepare_call(mode, caps)
                requests.append(plan.request(name=self._request_name(mode, rows), prepare_call=call, benchmark_call=call))
        self._prepared_plans = prepared_plans
        if not requests:
            return ()
        return (B12xPreparationUnit(
            name="DSA indexer",
            key=self._preparation_prefix,
            requests=tuple(requests),
            stage="state",
            autotune=not workload.eager_only,
        ),)

    def forward(self, hidden_states: torch.Tensor, q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor], k: torch.Tensor | None, weights: torch.Tensor) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12X indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12X index K must be written by the fused cache path.")
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            return self.topk_indices_buffer
        metadata = cast(DeepseekV32IndexerMetadata, context.attn_metadata[self.k_cache.prefix])
        scores = torch.empty((q_quant.shape[0], self.topk_tokens), dtype=torch.float32, device=q_quant.device) if self.dcp_world_size > 1 else None
        if metadata.prefill is not None:
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError("B12X sparse prefill requires single-request chunks.")
                start, end = chunk.token_start, chunk.token_end
                q = q_quant[start:end].contiguous()
                local = chunk.local_total_seq_lens if self.dcp_world_size > 1 else chunk.total_seq_lens
                pages = min(max(1, (int(local) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE), int(chunk.block_table.shape[1]))
                output, score = self.topk_indices_buffer[start:end, :self.topk_tokens], scores[start:end] if scores is not None else None
                _run_paged_topk(module=self._module, plan=self._plan("prefill", int(q.shape[0])), q=q, weights=weights[start:end].contiguous(), kv_cache=self.k_cache.kv_cache, seq_lens=(chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous(), block_table=chunk.block_table[:1, :pages].expand(int(q.shape[0]), pages), active_width=self.active_width_cap, output=output, scores=score)
                if score is not None:
                    _merge_dcp_topk(output, score, self.dcp_rank, self.dcp_world_size, self.cp_kv_cache_interleave_size)
        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12X sparse decode does not support padded rows.")
            lengths, tables = decode.seq_lens.reshape(-1).contiguous(), decode.block_table
            if int(tables.shape[0]) != int(lengths.shape[0]):
                if int(lengths.shape[0]) % int(tables.shape[0]):
                    raise RuntimeError("B12X sparse decode could not align lengths and page tables.")
                tables = tables.repeat_interleave(int(lengths.shape[0]) // int(tables.shape[0]), dim=0)
            rows = metadata.num_decode_tokens
            output, score = self.topk_indices_buffer[:rows, :self.topk_tokens], scores[:rows] if scores is not None else None
            _run_paged_topk(module=self._module, plan=self._plan("decode", rows), q=q_quant[:rows].contiguous(), weights=weights[:rows].contiguous(), kv_cache=self.k_cache.kv_cache, seq_lens=lengths[:rows], block_table=tables[:rows].contiguous(), active_width=getattr(decode, "active_width", None), output=output, scores=score)
            if score is not None:
                _merge_dcp_topk(output, score, self.dcp_rank, self.dcp_world_size, self.cp_kv_cache_interleave_size)
        return self.topk_indices_buffer

