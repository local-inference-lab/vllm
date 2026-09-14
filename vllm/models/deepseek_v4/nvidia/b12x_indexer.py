# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse indexer for DeepSeek V4."""

from typing import Any, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    set_b12x_preparation_provider,
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_dsa_indexer,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
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
        self.use_flattening = True
        self.supports_varlen = False

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


class DeepseekV4B12xIndexerBackend(DeepseekV4IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Flattened rows use the device query lengths after draft trimming.
        return True

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
        "Binding",
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
    if route is None:
        plan = getattr(obj, "plan", None)
        route = getattr(getattr(plan, "layout", None), "route", None)
    if route != _PREFILL_ROUTE:
        raise RuntimeError(
            f"B12x sparse prefill requires the packed-contiguous route, got {route!r}."
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
    active_width: torch.Tensor,
    output: torch.Tensor,
    scores: torch.Tensor | None,
    shared_page_table: bool,
) -> None:
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
    if shared_page_table:
        _assert_prefill_route(binding.runtime)
    module.run(binding)


class B12xC4SparseIndexer(nn.Module):
    """Shared C4 FP8 paged indexer with session-prepared C4 executions."""

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
        prefix: str | None = None,
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
        self._index_cache = getattr(k_cache, "kv_cache", None)
        self._index_num_q_heads: int | None = None
        self._score_output = False
        self._score_collective_registered = False
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        self._plans: dict[tuple[str, int], object] = {}
        # Request names must be unique across every indexer in the model. A
        # caller whose k_cache carries no prefix (GLM passes None) names its layer.
        if prefix is None:
            prefix = getattr(k_cache, "prefix", type(self).__qualname__)
        self._preparation_prefix = f"{prefix}.c4_indexer"
        self.register_buffer(
            "_active_width",
            torch.empty((1,), dtype=torch.int32, device=topk_indices_buffer.device),
            persistent=False,
        )
        set_b12x_preparation_provider(self, self)
    def _set_active_width(
        self, seq_lens: torch.Tensor, block_table: torch.Tensor
    ) -> torch.Tensor:
        torch.amax(seq_lens, dim=0, keepdim=True, out=self._active_width)
        return self._active_width.clamp_(
            min=0, max=int(block_table.shape[1]) * _INDEX_PAGE_SIZE
        )

    def _plan_for(self, mode: str, rows: int) -> object:
        rows = int(rows)
        capacity = min((count for plan_mode, count in self._plans
                        if plan_mode == mode and count >= rows), default=rows)
        plan = self._plans.get((mode, capacity))
        if plan is None:
            width = max(1, (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE)
            caps = self._caps(mode=mode, rows=capacity, max_page_table_width=width)
            plan = self._b12x_indexer.plan(
                caps, invocation=self._invocation(caps, scores=self._score_output),
            )
            self._plans[(mode, capacity)] = plan
        return plan

    def _request_name(self, mode: str, rows: int) -> str:
        return f"{self._preparation_prefix}.{mode}.m{rows}"

    def _caps(self, *, mode: str, rows: int, max_page_table_width: int):
        return self._b12x_indexer.Caps(
            device=self.topk_indices_buffer.device,
            num_q_heads=int(self._num_q_heads),
            max_q_rows=rows,
            max_page_table_width=max_page_table_width,
            topk=self.topk_tokens,
            mode=mode,
            route="packed_contiguous" if mode == "prefill" else "auto",
        )

    def set_b12x_index_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        num_q_heads: int | None = None,
        score_output: bool | None = None,
    ) -> None:
        """Publish the real C4 storage before its owner is collected."""
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            raise PreparationResourceUnavailableError("C4 index K cache is unavailable")
        next_heads = self._index_num_q_heads if num_q_heads is None else int(num_q_heads)
        if next_heads is not None and next_heads <= 0:
            raise ValueError("C4 index query head count must be positive")
        next_scores = self._score_output if score_output is None else bool(score_output)
        self._index_cache = kv_cache
        self._index_num_q_heads = next_heads
        self._score_output = next_scores
        if self._score_output and not self._score_collective_registered:
            from vllm.distributed import get_dcp_group
            from vllm.distributed.parallel_state import register_b12x_collective_describer
            from vllm.distributed.device_communicators.b12x_pcie_all_reduce import B12xPcieInvocation
            def describe(workload):
                return tuple(B12xPcieInvocation(
                    name=f"{self._preparation_prefix}.score_all_reduce.m{rows}.lane{workload.lane}",
                    operation="all_reduce", shape=(rows, self.topk_tokens),
                    dtype=torch.float32,
                ) for rows in workload.token_counts)
            self._score_collective_registered = register_b12x_collective_describer(
                self, describe, group=get_dcp_group()
            )
    @property
    def _num_q_heads(self) -> int:
        return self._index_num_q_heads or int(getattr(self.k_cache, "num_q_heads", 1))

    def _invocation(self, caps, *, scores: bool):
        rows, width = caps.max_q_rows, caps.max_page_table_width
        descriptor = lambda shape, dtype: {
            "shape": tuple(shape),
            "strides": tuple(torch.empty(shape, device="meta").stride()),
            "dtype": dtype,
            "alignment": 16,
        }
        return self._b12x_indexer.invocation_from_descriptors(
            caps,
            operands={
                "q_fp8": descriptor((rows, caps.num_q_heads, _INDEX_HEAD_DIM), "float8_e4m3fn"),
                "query_weights": descriptor((rows, caps.num_q_heads), "float32"),
                "index_k_cache": descriptor((max(int(self._index_cache.shape[0]), 1), _INDEX_PAGE_WIDTH), "uint8"),
                "page_table": descriptor((rows, width), "int32"),
                "cache_lengths": descriptor((rows,), "int32"),
                "active_width": descriptor((1,), "int32"),
                "output_indices": descriptor((rows, self.topk_tokens), "int32"),
                "output_scores": descriptor((rows, self.topk_tokens), "float32") if scores else None,
            },
        )

    def get_b12x_preparation_units(
        self, layer: nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        kv_cache = self._index_cache
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            return ()
        width = max(1, (workload.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE)
        requests = []
        plans: dict[tuple[str, int], object] = {}
        from b12x.preparation import PreparedCall
        capacities = {
            "decode": tuple(sorted({workload.max_seqs, *workload.fixed_token_counts})),
            "prefill": (workload.max_tokens,),
        }
        for mode, counts in capacities.items():
            for rows in counts:
                caps = self._caps(mode=mode, rows=rows, max_page_table_width=width)
                plan = self._b12x_indexer.plan(
                    caps, invocation=self._invocation(caps, scores=self._score_output)
                )
                plans[(mode, rows)] = plan

                def make_call(state, *, caps=caps, mode=mode):
                    # Keep the trial on the published C4 storage.  Indexing is
                    # read-only, so page zero is a safe representative page and
                    # does not need a pool-sized snapshot/restore.
                    q = torch.empty((caps.max_q_rows, caps.num_q_heads, _INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=caps.device)
                    weights = torch.empty((caps.max_q_rows, caps.num_q_heads), dtype=torch.float32, device=caps.device)
                    lengths = torch.full((caps.max_q_rows,), min(self.max_model_len, _INDEX_PAGE_SIZE), dtype=torch.int32, device=caps.device)
                    pages = torch.zeros((caps.max_q_rows, caps.max_page_table_width), dtype=torch.int32, device=caps.device)
                    output = torch.empty((caps.max_q_rows, self.topk_tokens), dtype=torch.int32, device=caps.device)
                    scores = torch.empty_like(output, dtype=torch.float32) if self._score_output else None
                    scratch = [torch.empty(spec.shape, dtype=spec.dtype, device=caps.device) for spec in state.layout.scratch_specs()]
                    binding = state.bind(scratch=scratch, real_page_table=pages, cache_seqlens_int32=lengths, active_width=self._active_width, expected_num_q_heads=caps.num_q_heads, shared_page_table=mode == "prefill", output_physical_slots=False)
                    def produce():
                        q.fill_(1)
                        weights.fill_(1)
                        self._set_active_width(lengths, pages)
                    return PreparedCall(
                        run=lambda: state.run(binding, q_fp8=q, query_weights=weights, index_k_cache=_flatten_index_cache(kv_cache), output_indices=output, output_scores=scores),
                        produce=produce,
                        owners=(q, weights, lengths, pages, output, scores, scratch, binding),
                    )

                requests.append(plan.request(
                    name=self._request_name(mode, rows),
                    prepare_call=make_call,
                    benchmark_call=make_call,
                ))
        self._plans = plans
        return (B12xPreparationUnit(
            name="B12xC4SparseIndexer",
            key=(self._preparation_prefix, width, tuple(capacities.items())),
            requests=tuple(requests), stage="state", autotune=not workload.eager_only,
        ),)

    def reserve_profile_workspace(self, q: torch.Tensor) -> None:
        del q

    def run_paged_topk(
        self, *, q: torch.Tensor, weights: torch.Tensor, kv_cache: torch.Tensor,
        seq_lens: torch.Tensor, block_table: torch.Tensor, output: torch.Tensor,
        scores: torch.Tensor | None = None, shared_page_table: bool,
        schedule_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del schedule_metadata
        if output.shape != (int(q.shape[0]), self.topk_tokens):
            raise ValueError(f"B12x C4 output must have shape {(int(q.shape[0]), self.topk_tokens)}, got {tuple(output.shape)}.")
        if scores is not None and (scores.shape != output.shape or scores.dtype != torch.float32):
            raise ValueError("B12x C4 scores must be float32 with the same shape as output")
        _run_paged_topk(
            module=self._b12x_indexer,
            plan=self._plan_for("prefill" if shared_page_table else "decode", int(q.shape[0])),
            q=q, weights=weights, kv_cache=kv_cache, seq_lens=seq_lens,
            block_table=block_table, active_width=self._set_active_width(seq_lens, block_table),
            output=output, scores=scores, shared_page_table=shared_page_table,
        )
        return output

    def forward(self, hidden_states: torch.Tensor, q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor], k: torch.Tensor | None, weights: torch.Tensor) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12x C4 indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12x C4 index K must be written before selection.")
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer
        metadata = cast(DeepseekV32IndexerMetadata, attn_metadata[self.k_cache.prefix])
        if metadata.prefill is not None:
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError("B12x sparse prefill requires single-request chunks.")
                q_chunk = q_quant[chunk.token_start:chunk.token_end].contiguous()
                output = self.topk_indices_buffer[chunk.token_start:chunk.token_end, :self.topk_tokens]
                seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()
                active_pages = min(max(1, (int(chunk.total_seq_lens) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE), int(chunk.block_table.shape[1]))
                self.run_paged_topk(q=q_chunk, weights=weights[chunk.token_start:chunk.token_end].contiguous(), kv_cache=self.k_cache.kv_cache, seq_lens=seq_lens, block_table=chunk.block_table[:1, :active_pages].expand(int(q_chunk.shape[0]), active_pages), output=output, shared_page_table=True)
        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12x sparse decode does not support padded rows.")
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            block_table = decode.block_table
            if int(block_table.shape[0]) != int(seq_lens.shape[0]):
                if int(seq_lens.shape[0]) % int(block_table.shape[0]):
                    raise RuntimeError("B12x sparse decode could not align sequence lengths with page-table rows.")
                block_table = block_table.repeat_interleave(int(seq_lens.shape[0]) // int(block_table.shape[0]), dim=0)
            rows = metadata.num_decode_tokens
            self.run_paged_topk(q=q_quant[:rows].contiguous(), weights=weights[:rows].contiguous(), kv_cache=self.k_cache.kv_cache, seq_lens=seq_lens[:rows], block_table=block_table[:rows].contiguous(), output=self.topk_indices_buffer[:rows, :self.topk_tokens], shared_page_table=False)
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
