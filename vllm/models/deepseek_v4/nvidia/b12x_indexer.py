# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse indexer for DeepSeek V4."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_dsa_indexer
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
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
class DeepseekV4B12xIndexerDecodeMetadata(DeepSeekV32IndexerDecodeMetadata):
    active_width: torch.Tensor | None = None


class DeepseekV4B12xIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
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
            compressed_seq_lens_cpu[num_decodes:],
            self.prefill_k_rows,
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
            active_width = (
                int(metadata.max_seq_len) + int(self.compress_ratio) - 1
            ) // int(self.compress_ratio)
            self.active_width_buffer.fill_(active_width)
            decode_fields = vars(decode).copy()
            decode_fields["schedule_metadata"] = schedule_metadata
            metadata.decode = DeepseekV4B12xIndexerDecodeMetadata(
                **decode_fields,
                active_width=self.active_width_buffer,
            )
        return metadata


class DeepseekV4B12xIndexerBackend(DeepseekV4IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_B12X_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4B12xIndexerMetadataBuilder]:
        return DeepseekV4B12xIndexerMetadataBuilder


def _require_b12x_indexer() -> Any:
    module = get_b12x_dsa_indexer()
    if module is None:
        raise RuntimeError(
            "DeepSeek V4 B12x attention requires `pip install vllm[b12x]`."
        )
    if not module.is_supported():
        raise RuntimeError("B12x sparse indexer is not supported on this device.")
    if int(module.PAGED_INDEX_PAGE_SIZE) != _INDEX_PAGE_SIZE:
        raise RuntimeError(
            "B12x sparse indexer page size changed: expected "
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
            "B12x indexer cache must have shape "
            f"[num_blocks, {expected_tail[0]}, {expected_tail[1]}] and dtype "
            f"uint8, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}."
        )
    if kv_cache.stride(1) != expected_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "B12x indexer cache requires contiguous page payloads, got stride "
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
            f"B12x sparse prefill requires the packed-contiguous route, got {route!r}."
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


def _run_deepgemm_prefill_topk(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    gather_cu_seq_lens: torch.Tensor,
    row_starts: torch.Tensor,
    output: torch.Tensor,
    topk: int,
    total_k_rows: int,
    workspace_k_rows: int,
) -> None:
    """Gather a shared paged cache and score prefill rows with DeepGEMM."""
    q_rows = int(q.shape[0])
    expected_cache_tail = (_INDEX_PAGE_SIZE, _INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
    if q.ndim != 3 or int(q.shape[2]) != _INDEX_HEAD_DIM:
        raise ValueError(
            "DeepGEMM C4 prefill queries must have shape [rows, heads, 128]"
        )
    if weights.shape != (q_rows, int(q.shape[1])) or weights.dtype != torch.float32:
        raise ValueError("DeepGEMM C4 prefill weights must be FP32 [rows, heads]")
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.uint8
        or tuple(kv_cache.shape[1:]) != expected_cache_tail
    ):
        raise ValueError("DeepGEMM C4 prefill cache must be uint8 [pages, 64, 132]")
    if seq_lens.shape != (q_rows,) or seq_lens.dtype != torch.int32:
        raise ValueError("DeepGEMM C4 prefill lengths must be int32 [rows]")
    if block_table.ndim != 2 or int(block_table.shape[0]) != 1:
        raise ValueError("DeepGEMM C4 prefill requires one shared page-table row")
    if gather_cu_seq_lens.shape != (2,) or gather_cu_seq_lens.dtype != torch.int32:
        raise ValueError("DeepGEMM C4 gather boundaries must be int32 [2]")
    if row_starts.shape != (q_rows,) or row_starts.dtype != torch.int32:
        raise ValueError("DeepGEMM C4 row starts must be int32 [rows]")
    if output.shape != (q_rows, int(topk)) or output.dtype != torch.int32:
        raise ValueError("DeepGEMM C4 prefill output must be int32 [rows, topk]")
    if not 0 <= int(total_k_rows) <= int(workspace_k_rows):
        raise ValueError(
            "DeepGEMM C4 gathered length exceeds workspace capacity: "
            f"length={total_k_rows}, capacity={workspace_k_rows}"
        )

    output.fill_(-1)
    if q_rows == 0 or total_k_rows == 0:
        return

    active_pages = (int(total_k_rows) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
    if active_pages > int(block_table.shape[1]):
        raise ValueError(
            "DeepGEMM C4 page table does not cover the gathered prefix: "
            f"required={active_pages}, available={block_table.shape[1]}"
        )

    k_quant_full, k_scale_full = current_workspace_manager().get_simultaneous(
        ((max(int(workspace_k_rows), 1), _INDEX_HEAD_DIM), q.dtype),
        ((max(int(workspace_k_rows), 1), _INDEX_SCALE_BYTES), torch.uint8),
    )
    k_quant = k_quant_full[:total_k_rows]
    k_scale_bytes = k_scale_full[:total_k_rows]
    ops.cp_gather_indexer_k_quant_cache(
        kv_cache,
        k_quant,
        k_scale_bytes,
        block_table[:, :active_pages],
        gather_cu_seq_lens,
    )
    logits = fp8_fp4_mqa_logits(
        (q, None),
        (k_quant, k_scale_bytes.view(torch.float32).squeeze(-1)),
        weights,
        row_starts,
        seq_lens,
        clean_logits=False,
    )
    ops.top_k_per_row_prefill(
        logits,
        row_starts,
        seq_lens,
        output,
        q_rows,
        int(logits.stride(0)),
        int(logits.stride(1)),
        int(topk),
    )


class B12xC4SparseIndexer(nn.Module):
    """Shared C4 FP8 paged indexer used by DeepSeek V4 and GLM-5.3-Flash."""

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
            raise ValueError("B12x C4 indexing requires a model-owned cache writer.")
        if use_fp4_cache:
            raise ValueError("B12x C4 indexing requires the FP8 index cache.")
        if compress_ratio != 4:
            raise ValueError(
                f"B12x C4 indexing requires compress_ratio=4, got {compress_ratio}."
            )
        if head_dim != _INDEX_HEAD_DIM:
            raise ValueError(
                f"B12x C4 indexing requires head_dim={_INDEX_HEAD_DIM}, got {head_dim}."
            )
        if topk_indices_buffer is None:
            raise ValueError("B12x C4 indexing requires a top-k output buffer.")
        _require_b12x_indexer()
        self.k_cache = k_cache
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer

    def _reserve_profile_workspace(self, q: torch.Tensor) -> None:
        module = _require_b12x_indexer()
        q_rows = max(int(q.shape[0]), 1)
        page_table_width = max(
            1,
            (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE,
        )
        for shared_page_table in (False, True):
            plan = module.plan(
                module.Caps(
                    device=q.device,
                    source_layout=module.SOURCE_LAYOUT_PAGED,
                    num_q_heads=int(q.shape[1]),
                    max_q_rows=q_rows,
                    max_page_table_width=page_table_width,
                    topk=self.topk_tokens,
                    mode="prefill" if shared_page_table else "decode",
                    shared_page_table=shared_page_table,
                )
            )
            if shared_page_table:
                _assert_prefill_route(plan)
            current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
        current_workspace_manager().get_simultaneous(
            ((max(self.max_model_len, 1), _INDEX_HEAD_DIM), q.dtype),
            (
                (max(self.max_model_len, 1), _INDEX_SCALE_BYTES),
                torch.uint8,
            ),
        )
        torch.empty(
            (max(q_rows, 1), max(self.max_model_len, 1)),
            dtype=torch.float32,
            device=q.device,
        )

    def reserve_profile_workspace(self, q: torch.Tensor) -> None:
        self._reserve_profile_workspace(q)

    def run_paged_topk(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        output: torch.Tensor,
        scores: torch.Tensor | None = None,
        shared_page_table: bool,
        schedule_metadata: torch.Tensor | None = None,
        active_width: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the shared C4 scorer against caller-owned paged metadata."""
        if output.shape != (int(q.shape[0]), self.topk_tokens):
            raise ValueError(
                "B12x C4 output must have shape "
                f"{(int(q.shape[0]), self.topk_tokens)}, got {tuple(output.shape)}."
            )
        if scores is not None and (
            scores.shape != output.shape or scores.dtype != torch.float32
        ):
            raise ValueError(
                "B12x C4 scores must be float32 with the same shape as output"
            )
        output.fill_(-1)
        _run_paged_topk(
            q=q,
            weights=weights,
            kv_cache=kv_cache,
            seq_lens=seq_lens,
            block_table=block_table,
            schedule_metadata=schedule_metadata,
            active_width=active_width,
            output=output,
            scores=scores,
            topk=self.topk_tokens,
            shared_page_table=shared_page_table,
        )
        return output

    def run_deepgemm_prefill_topk(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        gather_cu_seq_lens: torch.Tensor,
        row_starts: torch.Tensor,
        output: torch.Tensor,
        total_k_rows: int,
    ) -> torch.Tensor:
        """Use DeepGEMM for shared-page-table C4 prefill selection."""
        _run_deepgemm_prefill_topk(
            q=q,
            weights=weights,
            kv_cache=kv_cache,
            seq_lens=seq_lens,
            block_table=block_table,
            gather_cu_seq_lens=gather_cu_seq_lens,
            row_starts=row_starts,
            output=output,
            topk=self.topk_tokens,
            total_k_rows=int(total_k_rows),
            workspace_k_rows=self.max_model_len,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12x C4 indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12x C4 index K must be written before selection.")

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and forward_context.batch_descriptor is not None
            ):
                self._reserve_profile_workspace(q_quant)
            return self.topk_indices_buffer

        metadata = cast(
            DeepseekV32IndexerMetadata,
            attn_metadata[self.k_cache.prefix],
        )
        if metadata.prefill is not None:
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12x sparse prefill requires single-request chunks."
                    )
                q_chunk = q_quant[chunk.token_start : chunk.token_end].contiguous()
                weights_chunk = weights[
                    chunk.token_start : chunk.token_end
                ].contiguous()
                output = self.topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : self.topk_tokens
                ]
                seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()
                active_pages = max(
                    1,
                    (int(chunk.total_seq_lens) + _INDEX_PAGE_SIZE - 1)
                    // _INDEX_PAGE_SIZE,
                )
                active_pages = min(active_pages, int(chunk.block_table.shape[1]))
                block_table = chunk.block_table[:1, :active_pages].expand(
                    int(q_chunk.shape[0]), active_pages
                )
                output.fill_(-1)
                _run_paged_topk(
                    q=q_chunk,
                    weights=weights_chunk,
                    kv_cache=self.k_cache.kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
                    active_width=None,
                    output=output,
                    scores=None,
                    topk=self.topk_tokens,
                    shared_page_table=True,
                )

        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12x sparse decode does not support padded rows.")
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            block_table = decode.block_table
            if int(block_table.shape[0]) != int(seq_lens.shape[0]):
                if int(seq_lens.shape[0]) % int(block_table.shape[0]) != 0:
                    raise RuntimeError(
                        "B12x sparse decode could not align sequence lengths with "
                        "page-table rows."
                    )
                block_table = block_table.repeat_interleave(
                    int(seq_lens.shape[0]) // int(block_table.shape[0]), dim=0
                )
            num_tokens = metadata.num_decode_tokens
            output = self.topk_indices_buffer[:num_tokens, : self.topk_tokens]
            output.fill_(-1)
            active_width = getattr(decode, "active_width", None)
            _run_paged_topk(
                q=q_quant[:num_tokens].contiguous(),
                weights=weights[:num_tokens].contiguous(),
                kv_cache=self.k_cache.kv_cache,
                seq_lens=seq_lens[:num_tokens],
                block_table=block_table[:num_tokens].contiguous(),
                schedule_metadata=decode.schedule_metadata,
                active_width=active_width,
                output=output,
                scores=None,
                topk=self.topk_tokens,
                shared_page_table=False,
            )

        return self.topk_indices_buffer


# Preserve the DeepSeek-specific import surface while GLM imports the shared name.
DeepseekV4B12xSparseIndexer = B12xC4SparseIndexer


def b12x_indexer_is_supported() -> bool:
    module = get_b12x_dsa_indexer()
    return bool(
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and module is not None
        and module.is_supported()
    )
