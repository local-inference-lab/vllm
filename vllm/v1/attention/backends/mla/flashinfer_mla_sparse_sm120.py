# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING

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
    triton_filter_and_convert_dcp_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

_FLASHINFER_DECODE_MAX_TOKENS = 64


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
                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "
                    "576-wide GLM_NSA geometry, which requires "
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

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

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

        # GLM's pooled selector emits 2048 history candidates followed by up
        # to three unpooled recent tokens. FlashInfer instantiates a fixed
        # 2048-wide GLM_NSA kernel, so retain the recent tail and drop the same
        # number of lowest-ranked history candidates.
        kernel_topk = attn_metadata.topk_tokens
        extra_topk = topk_indices.shape[1] - kernel_topk
        if extra_topk < 0:
            raise ValueError(
                "FlashInfer sparse MLA candidate buffer is narrower than its "
                f"kernel width: {topk_indices.shape[1]} < {kernel_topk}."
            )
        if extra_topk:
            topk_indices = torch.cat(
                (
                    topk_indices[:, : kernel_topk - extra_topk],
                    topk_indices[:, -extra_topk:],
                ),
                dim=1,
            )

        topk_indices_physical, topk_lens = triton_filter_and_convert_dcp_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            dcp_size=1,
            dcp_rank=0,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=kernel_topk,
            return_valid_counts=True,
        )

        # FlashInfer's standalone SM120 sparse-decode kernels require a
        # 64-token physical cache page. GLM's pooled selector intentionally
        # keeps a larger manager page so its C4 index tail remains adjacent to
        # the MLA cache. For a decode-sized batch on that layout, pad the row
        # count just beyond FlashInfer's decode cutoff and use its paged
        # attention implementation. The dummy rows have no valid candidates
        # and are discarded below.
        cache_page_size = int(kv_c_and_k_pe_cache.shape[-2])
        if num_actual_toks <= _FLASHINFER_DECODE_MAX_TOKENS and cache_page_size != 64:
            padded_num_toks = _FLASHINFER_DECODE_MAX_TOKENS + 1
            pad_rows = padded_num_toks - num_actual_toks
            q = torch.cat(
                (q, q.new_zeros((pad_rows, *q.shape[1:]))),
                dim=0,
            )
            topk_indices_physical = torch.cat(
                (
                    topk_indices_physical,
                    topk_indices_physical.new_full(
                        (pad_rows, topk_indices_physical.shape[1]), -1
                    ),
                ),
                dim=0,
            )
            topk_lens = torch.cat(
                (topk_lens, topk_lens.new_zeros((pad_rows,))),
                dim=0,
            )

        output = q.new_empty(
            (q.shape[0], self.num_heads, self.kv_lora_rank),
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
            # The fused index conversion compacts every valid slot to the row
            # prefix, and these lengths mask its remaining -1 tail.
            seq_lens=topk_lens,
            max_seq_len=kernel_topk,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=kernel_topk,
            backend="sparse",
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1)[:num_actual_toks], None

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
            if int(k_pe.shape[-1]) != 0:
                raise ValueError(
                    "FlashInfer GLM5Next cache padding requires a zero-width "
                    f"RoPE tensor, got shape={tuple(k_pe.shape)}."
                )
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], self.rope_pad))
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )
