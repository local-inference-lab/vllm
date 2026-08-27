# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


_LOGICAL_TOPK = 2048
_DECODE_TAIL_KERNEL_TOPK = 512
_SM120_DECODE_MAX_TOKENS = 64
_MAX_PHYSICAL_TOPK = 2176


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

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
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.rope_pad = 0
        if self.qk_rope_head_dim == 0:
            if self.kv_lora_rank != 512:
                raise NotImplementedError(
                    "FLASHINFER_MLA_SPARSE_SM120 maps NoPE MLA onto the "
                    "576-wide GLM_NSA kernel geometry, which requires "
                    f"kv_lora_rank=512; got {self.kv_lora_rank}."
                )
            self.rope_pad = 64
        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None
        physical_topk = int(self.topk_indices_buffer.shape[1])
        if self.rope_pad and not (_LOGICAL_TOPK <= physical_topk <= _MAX_PHYSICAL_TOPK):
            raise ValueError(
                "The GLM SM120 FlashInfer adapter requires a 2048-wide "
                "logical top-k buffer with at most 128 physical tail columns; "
                f"got {physical_topk}."
            )

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor, num_tokens: int, num_heads: int
    ) -> torch.Tensor:
        if lse.dim() == 3 and lse.shape[1] == 1:
            lse = lse.squeeze(1)
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer SM120 sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse

    def _run_partition(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        physical_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        kernel_topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one supported fixed-width kernel partition with exact row bounds."""
        num_tokens = int(q.shape[0])
        empty_rows = valid_counts == 0
        physical_indices[:, 0] = physical_indices[:, 0].masked_fill(empty_rows, 0)
        valid_counts = valid_counts.clamp(min=1)

        output = q.new_empty(
            (num_tokens, self.num_heads, self.kv_lora_rank), dtype=q.dtype
        )
        lse = torch.empty(
            (num_tokens, self.num_heads), dtype=torch.float32, device=q.device
        )

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        kernel_result = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache,
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.kernel_qk_rope_head_dim,
            block_tables=physical_indices.unsqueeze(1),
            seq_lens=valid_counts,
            max_seq_len=kernel_topk,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=kernel_topk,
            lse=lse,
            return_lse=True,
            kv_scale_format=self.kv_scale_format,
        )
        assert isinstance(kernel_result, tuple)
        out, out_lse = kernel_result
        out = out.squeeze(1)
        out_lse = self._normalize_lse(out_lse, num_tokens, self.num_heads)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        out_lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, out_lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.rope_pad:
            q = torch.nn.functional.pad(q, (0, self.rope_pad))

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if not self.rope_pad:
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            output = q.new_empty(
                (num_actual_toks, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
            )
            if self._workspace_buffer is None:
                self._workspace_buffer = _get_workspace_buffer(q.device)
            from vllm.utils.flashinfer import (
                flashinfer_trtllm_batch_decode_with_kv_cache_mla,
            )

            out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1),
                kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.kernel_qk_rope_head_dim,
                block_tables=topk_indices_physical.unsqueeze(1),
                seq_lens=None,
                max_seq_len=attn_metadata.topk_tokens,
                out=output.unsqueeze(1),
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=attn_metadata.topk_tokens,
                kv_scale_format=self.kv_scale_format,
            )
            assert isinstance(out, torch.Tensor)
            return out.squeeze(1), None

        main_topk = topk_indices[:, :_LOGICAL_TOPK]
        main_indices, main_counts = cast(
            tuple[torch.Tensor, torch.Tensor],
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                main_topk,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=_LOGICAL_TOPK,
                return_valid_counts=True,
            ),
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        packed_kv_cache = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1)
        main_output, main_lse = self._run_partition(
            q=q,
            kv_cache=packed_kv_cache,
            physical_indices=main_indices,
            valid_counts=main_counts,
            kernel_topk=_LOGICAL_TOPK,
        )

        physical_topk = int(topk_indices.shape[1])
        if physical_topk == _LOGICAL_TOPK:
            if self.need_to_return_lse_for_decode:
                return main_output, main_lse
            return main_output, None

        tail_topk = topk_indices[:, _LOGICAL_TOPK:physical_topk]
        converted_tail, tail_counts = cast(
            tuple[torch.Tensor, torch.Tensor],
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                tail_topk,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=tail_topk.shape[1],
                return_valid_counts=True,
            ),
        )
        # FlashInfer's GLM_NSA SM120 prefill kernel is instantiated only at
        # top-k 2048. Its decode dispatcher also supports top-k 512 and is
        # selected for calls of at most 64 query tokens.
        tail_kernel_topk = (
            _DECODE_TAIL_KERNEL_TOPK
            if num_actual_toks <= _SM120_DECODE_MAX_TOKENS
            else _LOGICAL_TOPK
        )
        tail_indices = topk_indices.new_full((num_actual_toks, tail_kernel_topk), -1)
        tail_indices[:, : converted_tail.shape[1]].copy_(converted_tail)
        tail_output, tail_lse = self._run_partition(
            q=q,
            kv_cache=packed_kv_cache,
            physical_indices=tail_indices,
            valid_counts=tail_counts,
            kernel_topk=tail_kernel_topk,
        )

        merged_output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank), dtype=q.dtype
        )
        merged_lse = torch.empty(
            (num_actual_toks, self.num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        merge_attn_states(
            output=merged_output,
            prefix_output=main_output,
            prefix_lse=main_lse.transpose(0, 1),
            suffix_output=tail_output,
            suffix_lse=tail_lse.transpose(0, 1),
            output_lse=merged_lse.transpose(0, 1),
        )
        if self.need_to_return_lse_for_decode:
            return merged_output, merged_lse
        return merged_output, None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self.rope_pad:
            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
