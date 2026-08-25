# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x DSA indexer for non-compressed sparse MLA models."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import get_b12x_dsa_indexer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    split_indexer_prefill_chunks,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

_INDEX_HEAD_DIM = 128
_INDEX_SCALE_BYTES = 4
_INDEX_PAGE_SIZE = 64
_INDEX_PAGE_WIDTH = _INDEX_PAGE_SIZE * (_INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
_PREFILL_ROUTE = "packed_contiguous"


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
        self.prefill_k_rows = _require_b12x_indexer().resolve_paged_prefill_k_rows(
            max_page_table_width=block_table_width,
            page_size=_INDEX_PAGE_SIZE,
        )
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
        budget_seq_lens = torch.full_like(
            compressed_seq_lens_cpu[num_decodes:], self.prefill_k_rows
        )
        return [
            chunk
            for prefill_idx in range(len(prefill_query_lens_cpu))
            for chunk in split_indexer_prefill_chunks(
                budget_seq_lens[prefill_idx : prefill_idx + 1],
                prefill_query_lens_cpu[prefill_idx : prefill_idx + 1],
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes + prefill_idx,
            )
        ]

    def build(self, *args, **kwargs) -> DeepseekV32IndexerMetadata:
        metadata = super().build(*args, **kwargs)
        if metadata.decode is not None:
            module = _require_b12x_indexer()
            decode = metadata.decode
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            schedule_metadata = None
            if module.uses_paged_schedule(
                q_rows=int(seq_lens.shape[0]),
                max_pages=int(decode.block_table.shape[1]),
            ):
                schedule_metadata = module.plan_paged_schedule(
                    seq_lens,
                    _INDEX_PAGE_SIZE,
                    self.num_sms,
                    out=self.scheduler_metadata_buffer,
                )
            self.active_width_buffer.fill_(int(metadata.max_seq_len))
            fields = vars(decode).copy()
            fields["seq_lens"] = seq_lens
            fields["schedule_metadata"] = schedule_metadata
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
        "SOURCE_LAYOUT_PAGED",
        "index_topk_fp8",
        "plan",
        "plan_paged_schedule",
        "resolve_paged_prefill_k_rows",
        "uses_paged_schedule",
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


def _assert_prefill_route(obj: object) -> None:
    route = getattr(obj, "route", None)
    if route is None:
        route = getattr(getattr(obj, "layout", None), "route", None)
    if route != _PREFILL_ROUTE:
        raise RuntimeError(
            f"B12X sparse prefill requires route {_PREFILL_ROUTE!r}, got {route!r}."
        )


def _run_paged_topk(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    active_width: torch.Tensor | None,
    output: torch.Tensor,
    scores: torch.Tensor | None,
    topk: int,
    shared_page_table: bool,
) -> None:
    module = _require_b12x_indexer()
    plan = module.plan(
        module.Caps(
            device=q.device,
            source_layout=module.SOURCE_LAYOUT_PAGED,
            num_q_heads=int(q.shape[1]),
            max_q_rows=max(int(q.shape[0]), 1),
            max_page_table_width=max(int(block_table.shape[1]), 1),
            topk=int(topk),
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
        )
    )
    if shared_page_table:
        _assert_prefill_route(plan)
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=int(q.shape[1]),
        shared_page_table=shared_page_table,
        output_physical_slots=False,
    )
    if shared_page_table:
        _assert_prefill_route(binding)
    module.index_topk_fp8(
        q_fp8=q,
        weights=weights,
        index_k_cache=_flatten_index_cache(kv_cache),
        binding=binding,
        page_size=module.PAGED_INDEX_PAGE_SIZE,
        expected_num_q_heads=int(q.shape[1]),
        out_indices=output,
        out_scores=scores,
    )


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
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        del quant_block_size, scale_fmt, max_total_seq_len
        if not skip_k_cache_insert:
            raise ValueError("B12X requires the fused DSA index-cache insert path.")
        if use_fp4_cache:
            raise ValueError("B12X indexing requires the FP8 index cache.")
        if compress_ratio != 1:
            raise ValueError(
                "The non-compressed B12X indexer requires compress_ratio=1."
            )
        if head_dim != _INDEX_HEAD_DIM:
            raise ValueError(
                f"B12X indexing requires head_dim={_INDEX_HEAD_DIM}, got {head_dim}."
            )
        if topk_indices_buffer is None:
            raise ValueError("B12X indexing requires a top-k output buffer.")
        _require_b12x_indexer()
        self.k_cache = k_cache
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        from vllm.config import get_current_vllm_config

        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

    def _reserve_profile_workspace(self, q: torch.Tensor) -> None:
        module = _require_b12x_indexer()
        page_table_width = max(
            1, (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
        )
        for shared_page_table in (False, True):
            plan = module.plan(
                module.Caps(
                    device=q.device,
                    source_layout=module.SOURCE_LAYOUT_PAGED,
                    num_q_heads=int(q.shape[1]),
                    max_q_rows=max(int(q.shape[0]), 1),
                    max_page_table_width=page_table_width,
                    topk=self.topk_tokens,
                    mode="prefill" if shared_page_table else "decode",
                    shared_page_table=shared_page_table,
                )
            )
            if shared_page_table:
                _assert_prefill_route(plan)
            current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12X indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12X index K must be written by the fused cache path.")

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and forward_context.batch_descriptor is not None
            ):
                self._reserve_profile_workspace(q_quant)
            return self.topk_indices_buffer

        metadata = cast(DeepseekV32IndexerMetadata, attn_metadata[self.k_cache.prefix])
        scores = None
        if self.dcp_world_size > 1:
            scores = torch.empty(
                (q_quant.shape[0], self.topk_tokens),
                dtype=torch.float32,
                device=q_quant.device,
            )

        if metadata.prefill is not None:
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12X sparse prefill requires single-request chunks."
                    )
                start, end = chunk.token_start, chunk.token_end
                q_chunk = q_quant[start:end].contiguous()
                weights_chunk = weights[start:end].contiguous()
                output = self.topk_indices_buffer[start:end, : self.topk_tokens]
                output.fill_(-1)
                score_chunk = scores[start:end] if scores is not None else None
                seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()
                local_rows = (
                    chunk.local_total_seq_lens
                    if self.dcp_world_size > 1
                    else chunk.total_seq_lens
                )
                active_pages = max(
                    1, (int(local_rows) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
                )
                active_pages = min(active_pages, int(chunk.block_table.shape[1]))
                block_table = chunk.block_table[:1, :active_pages].expand(
                    int(q_chunk.shape[0]), active_pages
                )
                _run_paged_topk(
                    q=q_chunk,
                    weights=weights_chunk,
                    kv_cache=self.k_cache.kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
                    active_width=None,
                    output=output,
                    scores=score_chunk,
                    topk=self.topk_tokens,
                    shared_page_table=True,
                )
                if score_chunk is not None:
                    _merge_dcp_topk(
                        output,
                        score_chunk,
                        self.dcp_rank,
                        self.dcp_world_size,
                        self.cp_kv_cache_interleave_size,
                    )

        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12X sparse decode does not support padded rows.")
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            block_table = decode.block_table
            if int(block_table.shape[0]) != int(seq_lens.shape[0]):
                if int(seq_lens.shape[0]) % int(block_table.shape[0]) != 0:
                    raise RuntimeError(
                        "B12X sparse decode could not align lengths and page tables."
                    )
                block_table = block_table.repeat_interleave(
                    int(seq_lens.shape[0]) // int(block_table.shape[0]), dim=0
                )
            num_tokens = metadata.num_decode_tokens
            output = self.topk_indices_buffer[:num_tokens, : self.topk_tokens]
            output.fill_(-1)
            score_slice = scores[:num_tokens] if scores is not None else None
            _run_paged_topk(
                q=q_quant[:num_tokens].contiguous(),
                weights=weights[:num_tokens].contiguous(),
                kv_cache=self.k_cache.kv_cache,
                seq_lens=seq_lens[:num_tokens],
                block_table=block_table[:num_tokens].contiguous(),
                schedule_metadata=decode.schedule_metadata,
                active_width=getattr(decode, "active_width", None),
                output=output,
                scores=score_slice,
                topk=self.topk_tokens,
                shared_page_table=False,
            )
            if score_slice is not None:
                _merge_dcp_topk(
                    output,
                    score_slice,
                    self.dcp_rank,
                    self.dcp_world_size,
                    self.cp_kv_cache_interleave_size,
                )

        return self.topk_indices_buffer
