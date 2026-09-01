# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse indexer for DeepSeek V4."""

from typing import Any, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_dsa_indexer
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
            decode_fields = vars(decode).copy()
            decode_fields["schedule_metadata"] = schedule_metadata
            metadata.decode = DeepSeekV32IndexerDecodeMetadata(**decode_fields)
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
    module: Any,
    plan: Any,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    output: torch.Tensor,
    scores: torch.Tensor | None,
    topk: int,
    shared_page_table: bool,
) -> None:
    if shared_page_table:
        _assert_prefill_route(plan)
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
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
        self._b12x_indexer = _require_b12x_indexer()
        self.k_cache = k_cache
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        self._b12x_plans: dict[tuple[object, ...], Any] = {}

    def _plan_paged_topk(
        self,
        *,
        q: torch.Tensor,
        block_table: torch.Tensor,
        shared_page_table: bool,
    ) -> Any:
        key = (
            q.device,
            int(q.shape[1]),
            max(int(q.shape[0]), 1),
            max(int(block_table.shape[1]), 1),
            bool(shared_page_table),
        )
        plan = self._b12x_plans.get(key)
        if plan is None:
            module = self._b12x_indexer
            plan = module.plan(
                module.Caps(
                    device=q.device,
                    source_layout=module.SOURCE_LAYOUT_PAGED,
                    num_q_heads=int(q.shape[1]),
                    max_q_rows=max(int(q.shape[0]), 1),
                    max_page_table_width=max(int(block_table.shape[1]), 1),
                    topk=self.topk_tokens,
                    mode="prefill" if shared_page_table else "decode",
                    shared_page_table=shared_page_table,
                )
            )
            if shared_page_table:
                _assert_prefill_route(plan)
            self._b12x_plans[key] = plan
        return plan

    def _reserve_profile_workspace(self, q: torch.Tensor) -> None:
        module = self._b12x_indexer
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
        _run_paged_topk(
            module=self._b12x_indexer,
            plan=self._plan_paged_topk(
                q=q,
                block_table=block_table,
                shared_page_table=shared_page_table,
            ),
            q=q,
            weights=weights,
            kv_cache=kv_cache,
            seq_lens=seq_lens,
            block_table=block_table,
            schedule_metadata=schedule_metadata,
            output=output,
            scores=scores,
            topk=self.topk_tokens,
            shared_page_table=shared_page_table,
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
                _run_paged_topk(
                    module=self._b12x_indexer,
                    plan=self._plan_paged_topk(
                        q=q_chunk,
                        block_table=block_table,
                        shared_page_table=True,
                    ),
                    q=q_chunk,
                    weights=weights_chunk,
                    kv_cache=self.k_cache.kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    schedule_metadata=None,
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
            _run_paged_topk(
                module=self._b12x_indexer,
                plan=self._plan_paged_topk(
                    q=q_quant[:num_tokens],
                    block_table=block_table[:num_tokens],
                    shared_page_table=False,
                ),
                q=q_quant[:num_tokens].contiguous(),
                weights=weights[:num_tokens].contiguous(),
                kv_cache=self.k_cache.kv_cache,
                seq_lens=seq_lens[:num_tokens],
                block_table=block_table[:num_tokens].contiguous(),
                schedule_metadata=decode.schedule_metadata,
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
