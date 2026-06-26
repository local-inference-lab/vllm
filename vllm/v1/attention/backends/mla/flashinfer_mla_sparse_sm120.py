# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``.

PATCHED (glm52-patches/fimla-dcp): adds decode-context-parallel (DCP>1) support
so the SM120 FlashInfer sparse-MLA fast path can serve with a sharded KV cache.

Contract (see model_executor/layers/attention/mla_attention.py:839-903):
  * When dcp_world_size>1 the layer all-gathers the query across DCP ranks in the
    HEAD dim BEFORE calling forward_mqa, so q arrives as [B, H*dcp, 576] and we must
    compute every gathered head against THIS rank's LOCAL KV shard.
  * forward_mqa must return (out[B, H*dcp, kv_lora_rank], lse[B, H*dcp]); the layer
    then merges across ranks via cp_lse_ag_out_rs (ag_rs) / dcp_a2a_lse_reduce (a2a),
    the latter of which reduce-scatters back to [B, H, ...] per rank.
  * The impl must advertise can_return_lse_for_decode=True or the layer raises.

Blueprint: b12x_mla_sparse.py (the existing DCP-capable sparse-MLA backend).
TWO runtime-verify points are marked [VERIFY] below — they are why this needs the
numerical equivalence test (validate_fimla_dcp_lse.py), not just a compile check.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_dcp_global_index_to_local_index,
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

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
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        topk_indices_buffer = mla_args.get("topk_indices_buffer")
        if indexer is not None:
            topk_indices_buffer = indexer.topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError(
                "FLASHINFER_MLA_SPARSE_SM120 requires sparse-MLA top-k indices "
                "from an indexer or a shared topk_indices_buffer."
            )
        self.topk_indices_buffer: torch.Tensor = topk_indices_buffer
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        # HARDENING: eagerly allocate the 128MiB FlashInfer sparse workspace now
        # (at model-construction __init__, before vLLM's memory profiling) so it
        # lands in the profiling baseline and the KV cache is sized around it.
        # Lazily allocating it in forward_mqa (the decode path) meant profiling —
        # which may only exercise prefill — didn't count it, so at high
        # gpu-memory-utilization the runtime DCP-merge reduce_scatter buffer had no
        # headroom and OOM'd (forcing 0.90). The workspace is a module-global, so
        # only the first layer actually allocates; the rest reuse. Falls back to
        # lazy allocation if the worker device isn't selected yet.
        try:
            _dev = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._workspace_buffer: torch.Tensor | None = _get_workspace_buffer(_dev)
        except Exception:
            self._workspace_buffer = None

        # ---------------- DCP (decode context parallel) setup ----------------
        # This impl does NOT call super().__init__(), so set the CP attributes
        # the common MLA layer reads (mla_attention.py:730/839/857) explicitly,
        # mirroring b12x_mla_sparse.py. Base default is can_return_lse=False.
        self.can_return_lse_for_decode = True
        # query is pre-quantized only when supports_quant_query_input is True;
        # we keep bf16 query, so DCP + fp8-kv pre-quant guard is not triggered.
        self.supports_dcp_quant_query_input = False
        self.pcp_world_size = 1
        self.pcp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            _dcp = get_dcp_group()
            self.dcp_world_size = _dcp.world_size
            self.dcp_rank = _dcp.rank_in_group
        except (AssertionError, RuntimeError, KeyError):
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        # Under DCP, the layer all-gathered q in the head dim, so q.shape[1] is
        # num_heads * dcp_world_size. Always size buffers from q, not self.num_heads.
        num_actual_heads = q.shape[1]
        dcp = self.dcp_world_size > 1

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if dcp:
            # Global logical top-k ids -> THIS rank's local physical slots, plus the
            # per-token count of slots that actually live on this rank (-> seq_lens).
            seq_lens = torch.empty(
                num_actual_toks, dtype=torch.int32, device=q.device
            )
            topk_indices_physical, seq_lens = (
                triton_convert_dcp_global_index_to_local_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_world_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    valid_counts=seq_lens,
                )
            )
        else:
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
            seq_lens = None

        output = q.new_empty(
            (num_actual_toks, num_actual_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        want_lse = self.need_to_return_lse_for_decode
        # [VERIFY-1] lse buffer shape: the merge op cp_lse_ag_out_rs expects [B, H]
        # (float32). The flashinfer trtllm MLA kernel's lse layout for the sparse
        # SM120 path must be confirmed at runtime; we allocate [B, H] and reshape
        # the kernel's return below. validate_fimla_dcp_lse.py catches a mismatch.
        lse_buf = (
            q.new_empty((num_actual_toks, num_actual_heads), dtype=torch.float32)
            if want_lse
            else None
        )

        ret = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            # DCP: only attend this rank's valid local slots; non-DCP keeps None
            # (kernel attends the full top-k as today).
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
            lse=None if lse_buf is None else lse_buf.unsqueeze(1),
            return_lse=want_lse,
        )

        if not want_lse:
            # ret is the output tensor (writes into `out`), as in the stock impl.
            return ret.squeeze(1), None

        # [VERIFY-2] return convention: with return_lse=True the kernel returns
        # (out, lse). We also passed in-place buffers, so prefer the returned
        # tensors but fall back to our buffers if the kernel returns only `out`.
        if isinstance(ret, tuple):
            out_t, lse_t = ret
        else:
            out_t, lse_t = ret, lse_buf
        out_t = out_t.squeeze(1)
        # Normalize lse to [B, H] and slice to the gathered head count for the merge.
        lse_t = lse_t.reshape(num_actual_toks, -1)[:, :num_actual_heads].contiguous()
        return out_t, lse_t
