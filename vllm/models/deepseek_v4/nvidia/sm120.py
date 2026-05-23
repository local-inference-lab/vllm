# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 (consumer Blackwell) sparse-MLA impl for DeepSeek-V4.

Counterpart to :class:`DeepseekV4FlashMLASparseImpl` (Hopper / SM10x). The
forward path is driven by flashinfer's :class:`BatchSparseMLAPagedAttention
Wrapper` — the same wrapper used by the V32-family SPARSE_MLA_SM120 backend —
which auto-dispatches decode (num_tokens <= 64) and prefill internally and
accepts the SWA + compressed-indexer dual cache through its ``extra_kv_cache``
parameter.

Selected by ``_select_v4_sparse_impl()`` in :mod:`vllm.models.deepseek_v4
.nvidia.ops.attention` when the runtime compute capability is SM120; the
flashinfer wrapper itself lives on the layer (``layer._sparse_mla_wrapper``)
so it can be allocated once per layer at construction time and reused
across forward calls.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.nvidia.ops.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4SM120SparseBackend(DeepseekV4FlashMLASparseBackend):
    """SM120 variant. Geometry is identical to the FlashMLA parent (same KV
    layout, head size, block size); the only thing that changes is the impl
    class returned by ``get_impl_cls``."""

    @staticmethod
    def get_name() -> str:
        return "DSV4_SPARSE_MLA_SM120"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SM120SparseImpl"]:
        return DeepseekV4SM120SparseImpl


class DeepseekV4SM120SparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """SM120 flashinfer-wrapper-driven sparse-MLA impl for DeepseekV4.

    The wrapper auto-dispatches decode (num_tokens <= 64) and prefill on
    num_tokens, so this impl issues a single ``wrapper.run`` per chunk —
    no separate prefill kernel call, no plan() step.
    """

    backend_cls: ClassVar[type[AttentionBackend]] = DeepseekV4SM120SparseBackend

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run before metadata is built — flashinfer wrapper
            # carries its own pre-allocated workspace, so there's nothing
            # additional to reserve here.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation; layer.kv_cache may be empty after profiling cleanup.
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_kv_cache = layer.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            cls._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            cls._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // layer.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert layer.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    layer.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Treat queries in the same seq as independent queries (attended
        # purely by the generated indices). q arrives pre-padded to
        # layer.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)
        swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        assert layer._sparse_mla_wrapper is not None, (
            "DeepseekV4SM120SparseImpl requires layer._sparse_mla_wrapper; "
            "the flashinfer wrapper must be constructed in the layer __init__."
        )
        layer._sparse_mla_wrapper.run(
            q=q,
            kv_cache=swa_cache,
            indices=swa_indices,
            output=output,
            sm_scale=layer.scale,
            topk_length=swa_lens,
            attn_sink=layer.attn_sink,
            extra_kv_cache=kv_cache if not swa_only else None,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
        )

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        # `_dummy_run` passes synthetic non-None attn_metadata for swa-only
        # layers during cudagraph capture, so check compress_ratio directly.
        swa_only = layer.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        topk_indices: torch.Tensor | None
        if swa_only:
            topk_indices = None
        elif layer.compress_ratio == 4:
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            # C128A: pre-computed during metadata build.
            assert attn_metadata is not None
            topk_indices = attn_metadata.c128a_prefill_topk_indices

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        assert layer._sparse_mla_wrapper is not None

        # unsqueeze(-2) adds the h_kv=1 axis without copying.
        swa_kv_paged = swa_k_cache.unsqueeze(-2)
        extra_kv_paged = (
            compressed_k_cache.unsqueeze(-2) if not swa_only else None
        )

        num_chunks = (
            num_prefills + cls.PREFILL_CHUNK_SIZE - 1
        ) // cls.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * cls.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + cls.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start]
                - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end]
                - prefill_token_base
            )

            extra_indices_chunk = (
                topk_indices[query_start:query_end]
                if topk_indices is not None
                else None
            )

            layer._sparse_mla_wrapper.run(
                q=q[query_start:query_end],
                kv_cache=swa_kv_paged,
                indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                output=output[query_start:query_end],
                sm_scale=layer.scale,
                topk_length=swa_metadata.prefill_swa_lens[query_start:query_end],
                attn_sink=layer.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_indices=extra_indices_chunk,
            )
