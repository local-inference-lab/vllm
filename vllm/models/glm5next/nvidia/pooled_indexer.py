# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 C4 pooling around the shared b12x FP8 paged indexer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.models.deepseek_v4.nvidia.b12x_indexer import (
    B12xC4SparseIndexer,
)
from vllm.v1.attention.backends.mla.b12x_indexer import _merge_dcp_topk

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
        B12xMLASparseMetadata,
    )

from .ops.glm_kpool import (
    expand_c4_block_table,
    expand_pool_ids,
    fwht128_quant_fp8,
    gather_c4_block_table_rows,
    pool_seq_lens,
    update_decode_pools,
)

_INDEX_HEADS = 32
_INDEX_HEAD_DIM = 128
_POOL_SIZE = 4
_POOL_TOPK = 512
_TOPK_TOKENS = 2048
_SELECTION_WIDTH = 2051
_INDEX_CACHE_WIDTH = 132
_INDEX_PAGE_SIZE = 64
_INDEX_PAGE_BYTES = _INDEX_PAGE_SIZE * _INDEX_CACHE_WIDTH
_MLA_RECORD_BYTES = 528


class _PersistentPooledSelectorArena:
    """Fixed-address selector metadata and workspaces."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        pool_table_width: int,
        device: torch.device,
    ) -> None:
        if min(max_num_seqs, max_num_batched_tokens, pool_table_width) <= 0:
            raise ValueError("GLM selector arena capacities must be positive")
        self.request_rows = max_num_seqs
        self.token_rows = max_num_batched_tokens
        self.pool_table_width = pool_table_width
        self.request_block_table = torch.empty(
            (max_num_seqs, pool_table_width),
            dtype=torch.int32,
            device=device,
        )
        self.token_block_table = torch.empty(
            (max_num_batched_tokens, pool_table_width),
            dtype=torch.int32,
            device=device,
        )
        self.pool_seq_lens = torch.empty(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.pool_scores = torch.empty(
            (max_num_batched_tokens, _POOL_TOPK),
            dtype=torch.float32,
            device=device,
        )
        self.query_start_loc = torch.empty(
            max_num_seqs + 1, dtype=torch.int32, device=device
        )
        self.row_request_ids = torch.empty(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )

    def require_fits(
        self, *, num_reqs: int, num_tokens: int, pool_table_width: int
    ) -> None:
        if num_reqs > self.request_rows:
            raise ValueError("GLM selector request capacity exceeded")
        if num_tokens > self.token_rows:
            raise ValueError("GLM selector token capacity exceeded")
        if pool_table_width != self.pool_table_width:
            raise ValueError("GLM selector pool-table geometry changed")

    def stage_replay_metadata(
        self,
        *,
        query_start_loc: torch.Tensor,
        req_id_per_token: torch.Tensor,
        num_reqs: int,
        live_rows: int,
    ) -> None:
        self.query_start_loc[: num_reqs + 1].copy_(query_start_loc[: num_reqs + 1])
        self.row_request_ids[:live_rows].copy_(req_id_per_token[:live_rows])


class Glm5NextPooledIndexer(nn.Module):
    """Produce GLM C4 pools, select them with b12x, and expand token IDs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        pool_topk_indices_buffer: torch.Tensor | None,
        *,
        main_layer_name: str,
        prefix: str,
    ) -> None:
        super().__init__()
        if cache_config is None:
            raise ValueError("GLM pooled selection requires a paged cache")
        if int(cache_config.block_size) % (_POOL_SIZE * _INDEX_PAGE_SIZE):
            raise ValueError(
                "GLM C4 indexing requires a model block size divisible by 256"
            )
        if topk_indices_buffer is None or tuple(topk_indices_buffer.shape[1:]) != (
            _SELECTION_WIDTH,
        ):
            raise ValueError(
                "GLM pooled selection requires an int32 [rows, 2051] buffer"
            )
        if pool_topk_indices_buffer is None or tuple(
            pool_topk_indices_buffer.shape[1:]
        ) != (_POOL_TOPK,):
            raise ValueError(
                "GLM pooled selection requires an int32 [rows, 512] buffer"
            )
        if topk_indices_buffer.dtype != torch.int32:
            raise TypeError("GLM token selection buffer must use int32")
        if pool_topk_indices_buffer.dtype != torch.int32:
            raise TypeError("GLM pool selection buffer must use int32")
        geometry = {
            "index_topk": _TOPK_TOKENS,
            "index_n_heads": _INDEX_HEADS,
            "index_head_dim": _INDEX_HEAD_DIM,
            "index_kpool": _POOL_SIZE,
            "qk_rope_head_dim": 0,
        }
        for name, expected in geometry.items():
            if int(getattr(config, name, -1)) != expected:
                raise ValueError(
                    f"GLM-5.3-Flash requires {name}={expected}, "
                    f"got {getattr(config, name, None)!r}"
                )

        self.topk_tokens = _TOPK_TOKENS
        self.topk_indices_buffer = topk_indices_buffer
        self.pool_topk_indices_buffer = pool_topk_indices_buffer
        self.main_layer_name = main_layer_name
        self.max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.block_size = int(cache_config.block_size)
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = int(parallel_config.decode_context_parallel_size)
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        token_interleave = int(parallel_config.cp_kv_cache_interleave_size)
        if self.dcp_world_size > 1 and token_interleave % _POOL_SIZE:
            raise ValueError(
                "GLM C4 DCP requires cp_kv_cache_interleave_size to be "
                f"divisible by {_POOL_SIZE}, got {token_interleave}"
            )
        self.pool_interleave = max(token_interleave // _POOL_SIZE, 1)
        if int(topk_indices_buffer.shape[0]) != self.max_tokens:
            raise ValueError("GLM token selection buffer has the wrong row capacity")
        if int(pool_topk_indices_buffer.shape[0]) != self.max_tokens:
            raise ValueError("GLM pool selection buffer has the wrong row capacity")

        device = topk_indices_buffer.device
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(
                (_POOL_SIZE, _INDEX_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(
                (_INDEX_HEAD_DIM, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            _INDEX_HEADS * _INDEX_HEAD_DIM,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            _INDEX_HEAD_DIM,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            _INDEX_HEADS,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(_INDEX_HEAD_DIM, eps=1e-6)

        self.indexer_op = B12xC4SparseIndexer(
            None,
            quant_block_size=_INDEX_HEAD_DIM,
            scale_fmt="ue8m0",
            topk_tokens=_POOL_TOPK,
            head_dim=_INDEX_HEAD_DIM,
            max_model_len=math.ceil(self.max_model_len / _POOL_SIZE),
            max_total_seq_len=math.ceil(self.max_model_len / _POOL_SIZE),
            topk_indices_buffer=pool_topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=False,
            compress_ratio=_POOL_SIZE,
        )

        self.register_buffer(
            "_tail",
            torch.empty(
                (self.max_seqs, 2, _POOL_SIZE, _INDEX_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_tail_snapshot",
            torch.empty_like(self._tail),
            persistent=False,
        )
        self.register_buffer(
            "_q_fp8",
            torch.empty(
                (self.max_tokens, _INDEX_HEADS, _INDEX_HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_q_scale",
            torch.empty(
                (self.max_tokens, _INDEX_HEADS),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        self._selector_arena: _PersistentPooledSelectorArena | None = None
        self._weights_proj_fp32: torch.Tensor | None = None
        self._index_cache: torch.Tensor | None = None
        self._parent_table_width = 0
        self._subpages_per_parent = 0
        self._parent_stride_pages = 0

    @property
    def _aligned_max_seq_len(self) -> int:
        return math.ceil(self.max_model_len / _POOL_SIZE) * _POOL_SIZE

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = int(max_model_len)
        self.indexer_op.max_model_len = math.ceil(max_model_len / _POOL_SIZE)

    @staticmethod
    def _max_parent_table_width(
        max_model_len: int,
        block_size: int,
        dcp_world_size: int,
    ) -> int:
        return math.ceil(max_model_len / (block_size * dcp_world_size))

    @staticmethod
    def _index_cache_view(
        main_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if (
            main_cache.ndim != 3
            or main_cache.dtype != torch.uint8
            or int(main_cache.shape[-1]) != _MLA_RECORD_BYTES
        ):
            raise ValueError("GLM MLA cache must be uint8 [pages, block, 528]")
        pages, block_size, _ = map(int, main_cache.shape)
        if pages <= 0 or block_size % (_POOL_SIZE * _INDEX_PAGE_SIZE):
            raise ValueError(
                "GLM MLA pages must contain a whole number of 64-pool C4 pages"
            )
        if tuple(map(int, main_cache.stride()[1:])) != (_MLA_RECORD_BYTES, 1):
            raise ValueError("GLM MLA cache records must be contiguous within a page")

        subpages_per_parent = block_size // (_POOL_SIZE * _INDEX_PAGE_SIZE)
        parent_stride_bytes = int(main_cache.stride(0))
        semantic_page_bytes = block_size * _MLA_RECORD_BYTES
        index_tail_bytes = subpages_per_parent * _INDEX_PAGE_BYTES
        if parent_stride_bytes < semantic_page_bytes + index_tail_bytes:
            raise ValueError("GLM MLA cache page does not contain its FP8 index tail")
        if parent_stride_bytes % _INDEX_PAGE_BYTES:
            raise ValueError(
                "GLM MLA parent-page stride must be an exact number of C4 pages"
            )
        parent_stride_pages = parent_stride_bytes // _INDEX_PAGE_BYTES
        max_virtual_page = (pages - 1) * parent_stride_pages + subpages_per_parent - 1
        if max_virtual_page > torch.iinfo(torch.int32).max:
            raise ValueError(
                "GLM virtual C4 page IDs exceed the int32 page-table range"
            )

        tail_offset = int(main_cache.storage_offset()) + semantic_page_bytes
        tail_end = (
            tail_offset + max_virtual_page * _INDEX_PAGE_BYTES + _INDEX_PAGE_BYTES
        )
        if tail_end > int(main_cache.untyped_storage().nbytes()):
            raise ValueError(
                "GLM MLA cache storage does not include its FP8 index tail"
            )
        index_cache = torch.as_strided(
            main_cache,
            size=(max_virtual_page + 1, _INDEX_PAGE_SIZE, _INDEX_CACHE_WIDTH),
            stride=(_INDEX_PAGE_BYTES, _INDEX_CACHE_WIDTH, 1),
            storage_offset=tail_offset,
        )
        return index_cache, subpages_per_parent, parent_stride_pages

    def bind_main_kv_cache(self, main_cache: torch.Tensor) -> None:
        index_cache, subpages, parent_stride_pages = self._index_cache_view(main_cache)
        block_size = int(main_cache.shape[1])
        parent_table_width = self._max_parent_table_width(
            self._aligned_max_seq_len,
            block_size,
            self.dcp_world_size,
        )
        pool_table_width = parent_table_width * subpages
        device = main_cache.device
        arena = self._selector_arena
        if arena is None:
            arena = _PersistentPooledSelectorArena(
                max_num_seqs=self.max_seqs,
                max_num_batched_tokens=self.max_tokens,
                pool_table_width=pool_table_width,
                device=device,
            )
            self._selector_arena = arena
        else:
            arena.require_fits(
                num_reqs=self.max_seqs,
                num_tokens=self.max_tokens,
                pool_table_width=pool_table_width,
            )
        self._index_cache = index_cache
        self._parent_table_width = parent_table_width
        self._subpages_per_parent = subpages
        self._parent_stride_pages = parent_stride_pages
        self.block_size = block_size

    def unbind_main_kv_cache(self) -> None:
        self._index_cache = None

    def _project_head_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._weights_proj_fp32 is None:
            self._weights_proj_fp32 = self.weights_proj.weight.detach().float()
        return F.linear(hidden_states.float(), self._weights_proj_fp32)

    @staticmethod
    def _state_slots(metadata: Any) -> torch.Tensor:
        slots = getattr(metadata, "selector_state_slot_ids", None)
        if slots is None:
            raise RuntimeError("GLM selector state-slot metadata is missing")
        return cast(torch.Tensor, slots)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module | None,
    ) -> torch.Tensor:
        del rotary_emb
        if q_lora is None:
            raise RuntimeError("GLM pooled selection requires q_lora_rank")
        rows = int(hidden_states.shape[0])
        if rows > self.max_tokens:
            raise ValueError("GLM selector batch exceeds its row capacity")
        if positions.shape != (rows,) or positions.dtype != torch.int64:
            raise ValueError("GLM selector positions must be int64 [rows]")

        query = self.wq_b(q_lora)[0].view(rows, _INDEX_HEADS, _INDEX_HEAD_DIM)
        normalized_key = self.k_norm(self.wk(hidden_states)[0])
        gate = F.linear(hidden_states, self.index_kpool_compress_gate)
        weights = self._project_head_weights(hidden_states)

        q_fp8 = self._q_fp8[:rows]
        q_scale = self._q_scale[:rows]
        fwht128_quant_fp8(
            query.contiguous().view(-1, _INDEX_HEAD_DIM),
            q_fp8.view(-1, _INDEX_HEAD_DIM),
            q_scale.view(-1),
        )
        weights.mul_(q_scale)
        weights.mul_((_INDEX_HEAD_DIM * _INDEX_HEADS) ** -0.5)

        forward_context = get_forward_context()
        raw_metadata = forward_context.attn_metadata
        if not isinstance(raw_metadata, dict):
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and forward_context.batch_descriptor is not None
            ):
                self.indexer_op.reserve_profile_workspace(q_fp8)
            output = self.topk_indices_buffer[:rows]
            output.fill_(-1)
            return output

        main_metadata = cast(
            "B12xMLASparseMetadata | None", raw_metadata.get(self.main_layer_name)
        )
        if main_metadata is None:
            raise RuntimeError("GLM selector metadata is incomplete")
        index_cache = self._index_cache
        if index_cache is None:
            raise RuntimeError("GLM selector cache is not bound")
        state_slots = self._state_slots(main_metadata)
        num_reqs = int(main_metadata.num_reqs)
        live_rows = int(main_metadata.num_actual_tokens)
        decode_rows = int(main_metadata.num_decode_tokens)
        if not 0 <= decode_rows <= live_rows <= rows:
            raise RuntimeError(
                "GLM selector token counts are inconsistent: "
                f"decode={decode_rows}, live={live_rows}, capacity={rows}"
            )
        actual_table_width = int(main_metadata.block_table.shape[1])
        if actual_table_width < self._parent_table_width:
            raise RuntimeError(
                "GLM selector block table is narrower than the runtime context: "
                f"actual={actual_table_width}, required={self._parent_table_width}"
            )

        arena = self._selector_arena
        if arena is None:
            raise RuntimeError("GLM selector arena is not bound")
        arena.require_fits(
            num_reqs=num_reqs,
            num_tokens=live_rows,
            pool_table_width=(self._parent_table_width * self._subpages_per_parent),
        )
        arena.stage_replay_metadata(
            query_start_loc=main_metadata.query_start_loc,
            req_id_per_token=main_metadata.req_id_per_token,
            num_reqs=num_reqs,
            live_rows=live_rows,
        )
        update_decode_pools(
            index_cache,
            self._tail,
            state_slots,
            arena.query_start_loc,
            normalized_key,
            gate,
            self.index_kpool_compress_ape,
            main_metadata.slot_mapping[:rows],
            positions,
            num_reqs,
            model_block_size=self.block_size,
            parent_stride_pages=self._parent_stride_pages,
        )
        expand_c4_block_table(
            main_metadata.block_table[:num_reqs, : self._parent_table_width],
            arena.request_block_table,
            rows=num_reqs,
            subpages_per_parent=self._subpages_per_parent,
            parent_stride_pages=self._parent_stride_pages,
        )
        seq_lens = arena.pool_seq_lens[:live_rows]
        pool_seq_lens(
            positions[:live_rows],
            seq_lens,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            pool_interleave=self.pool_interleave,
        )
        pool_ids = self.pool_topk_indices_buffer[:rows]
        pool_ids.fill_(-1)
        pool_scores = arena.pool_scores[:live_rows] if self.dcp_world_size > 1 else None
        token_block_table = arena.token_block_table[:live_rows]
        gather_c4_block_table_rows(
            arena.request_block_table,
            arena.row_request_ids[:live_rows],
            token_block_table,
        )
        self.indexer_op.run_paged_topk(
            q=q_fp8[:live_rows],
            weights=weights[:live_rows],
            kv_cache=index_cache,
            seq_lens=seq_lens,
            block_table=token_block_table,
            output=pool_ids[:live_rows],
            scores=pool_scores,
            shared_page_table=False,
        )
        if pool_scores is not None:
            _merge_dcp_topk(
                pool_ids[:live_rows],
                pool_scores,
                self.dcp_rank,
                self.dcp_world_size,
                self.pool_interleave,
            )

        output = self.topk_indices_buffer[:rows]
        output.fill_(-1)
        expand_pool_ids(pool_ids[:live_rows], positions[:live_rows], output[:live_rows])
        return output

    def snapshot_speculative_interval_starts(self) -> None:
        self._tail_snapshot.copy_(self._tail)

    def restore_speculative_interval_starts(self) -> None:
        self._tail.copy_(self._tail_snapshot)


__all__ = ["Glm5NextPooledIndexer"]
