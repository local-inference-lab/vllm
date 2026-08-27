# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X sparse MLA adapter for GLM-5 Next checkpoints without RoPE."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import MLAAttentionImpl
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseBackend,
    B12xMLASparseImpl,
    B12xMLASparseMetadata,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import AttentionLayer


_LOGICAL_HEAD_SIZE = 512
_KERNEL_ROPE_HEAD_DIM = 64
_KERNEL_HEAD_SIZE = _LOGICAL_HEAD_SIZE + _KERNEL_ROPE_HEAD_DIM
_LOGICAL_TOPK = 2048
_TAIL_KERNEL_TOPK = 512
_MAX_PHYSICAL_TOPK = 2176


class Glm5NextB12xMLASparseBackend(B12xMLASparseBackend):
    """B12X sparse MLA with an exact zero-padded NoPE kernel layout."""

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return Glm5NextB12xMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_LOGICAL_HEAD_SIZE]


class Glm5NextB12xMLASparseImpl(B12xMLASparseImpl):
    """Map a 512-wide NoPE latent onto B12X's 576-wide SM120 recipe.

    The additional 64 query and key coordinates are zero. They therefore add
    exactly zero to every QK dot product, while the model's original attention
    scale remains unchanged.
    """

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        logical_rope_head_dim = int(mla_args["qk_rope_head_dim"])
        if head_size != _LOGICAL_HEAD_SIZE or logical_rope_head_dim != 0:
            raise ValueError(
                "The GLM-5 Next B12X adapter requires a 512-wide NoPE latent; "
                f"got head_size={head_size}, "
                f"qk_rope_head_dim={logical_rope_head_dim}."
            )

        kernel_args = dict(mla_args)
        kernel_args["qk_rope_head_dim"] = _KERNEL_ROPE_HEAD_DIM
        kernel_args["qk_head_dim"] = (
            int(kernel_args["qk_nope_head_dim"]) + _KERNEL_ROPE_HEAD_DIM
        )
        super().__init__(
            num_heads=num_heads,
            head_size=_KERNEL_HEAD_SIZE,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **kernel_args,
        )
        if not (_LOGICAL_TOPK <= self._physical_topk_tokens <= _MAX_PHYSICAL_TOPK):
            raise ValueError(
                "The GLM-5 Next B12X adapter requires a 2048-wide logical "
                "top-k buffer with at most 128 physical tail columns; got "
                f"{self._physical_topk_tokens}."
            )
        self._tail_topk_tokens = self._physical_topk_tokens - _LOGICAL_TOPK
        if self._tail_topk_tokens:
            self._decode_tail_plan = self._make_plan("decode", _TAIL_KERNEL_TOPK)
            self._extend_tail_plan = self._make_plan("extend", _TAIL_KERNEL_TOPK)

    def _get_kernel_topk_tokens(self, physical_topk_tokens: int) -> int:
        if physical_topk_tokens < _LOGICAL_TOPK:
            raise ValueError(
                "The GLM-5 Next B12X adapter requires at least 2048 top-k "
                f"columns; got {physical_topk_tokens}."
            )
        return _LOGICAL_TOPK

    def _copy_query_to_buffer(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        q_buffer: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        if isinstance(q, tuple):
            q_nope, q_pe = q
            if int(q_pe.shape[-1]) != 0:
                raise ValueError(
                    "The GLM-5 Next B12X adapter received a non-empty RoPE query."
                )
            logical_q = q_nope
        else:
            logical_q = q
        if int(logical_q.shape[-1]) != _LOGICAL_HEAD_SIZE:
            raise ValueError(
                "The GLM-5 Next B12X query must have 512 latent coordinates; "
                f"got {logical_q.shape[-1]}."
            )

        num_tokens = int(logical_q.shape[0])
        q_all = q_buffer[:num_tokens]
        q_all.zero_()
        q_all[..., :_LOGICAL_HEAD_SIZE].copy_(logical_q)
        return num_tokens, q_all

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "The GLM-5 Next B12X adapter received a non-empty RoPE key."
            )
        kernel_k_pe = k_pe.new_zeros((*k_pe.shape[:-1], _KERNEL_ROPE_HEAD_DIM))
        super().do_kv_cache_update(
            kv_c_normed,
            kernel_k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: "AttentionLayer",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend to the 2048-token selection and incomplete kpool tail.

        GLM-5.3 reserves a padded 128-column region after its 2048 selected
        tokens. Up to three columns contain the uncompressed remainder of the
        current four-token index pool. B12X SM120 kernels accept fixed sparse
        widths, so the tail is evaluated by a second 512-wide B12X launch and
        both partial softmax states are merged exactly from their LSE values.
        """
        if not self._tail_topk_tokens:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        del layer
        is_decode = attn_metadata.max_query_len <= 1
        main_plan = self._decode_plan if is_decode else self._extend_plan
        tail_plan = self._decode_tail_plan if is_decode else self._extend_tail_plan
        main_scratch_specs = main_plan.shapes_and_dtypes()
        tail_scratch_specs = tail_plan.shapes_and_dtypes()
        q_spec = (
            (self._max_tokens, self._input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        tail_indices_spec = (
            (self._max_tokens, _TAIL_KERNEL_TOPK),
            torch.int32,
        )
        merged_output_spec = (
            (self._max_tokens, self._input_num_heads, self.kv_lora_rank),
            torch.bfloat16,
        )
        merged_lse_spec = (
            (self._max_tokens, self._input_num_heads),
            torch.float32,
        )
        workspaces = current_workspace_manager().get_simultaneous(
            q_spec,
            tail_indices_spec,
            merged_output_spec,
            merged_lse_spec,
            *main_scratch_specs,
            *tail_scratch_specs,
        )
        q_buffer, tail_indices_buffer, merged_output_buffer, merged_lse_buffer = (
            workspaces[:4]
        )
        main_scratch_end = 4 + len(main_scratch_specs)
        main_scratch = workspaces[4:main_scratch_end]
        tail_scratch = workspaces[main_scratch_end:]

        num_tokens, q_all = self._copy_query_to_buffer(q, q_buffer)
        if int(q_all.shape[1]) != self._input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {self._input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        record_width = int(kv_c_and_k_pe_cache.shape[-1])
        block_stride_rows = int(kv_c_and_k_pe_cache.stride(0)) // record_width
        main_topk = self.topk_indices_buffer[:num_tokens, :_LOGICAL_TOPK]
        tail_topk = self.topk_indices_buffer[
            :num_tokens, _LOGICAL_TOPK : self._physical_topk_tokens
        ]
        main_indices, main_counts = self._convert_topk_indices(
            attn_metadata, main_topk, block_stride_rows
        )
        converted_tail, tail_counts = self._convert_topk_indices(
            attn_metadata, tail_topk, block_stride_rows
        )
        tail_indices = tail_indices_buffer[:num_tokens]
        tail_indices.fill_(-1)
        tail_indices[:, : self._tail_topk_tokens].copy_(converted_tail)

        cache_seq_lens = attn_metadata.cache_seq_lens_per_token
        assert cache_seq_lens is not None
        cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()

        main_binding = main_plan.bind(
            scratch=main_scratch,
            q=q_all,
            selected_indices=main_indices,
            cache_seqlens_int32=cache_seq_lens,
            nsa_cache_seqlens_int32=main_counts,
        )
        tail_binding = tail_plan.bind(
            scratch=tail_scratch,
            q=q_all,
            selected_indices=tail_indices,
            cache_seqlens_int32=cache_seq_lens,
            nsa_cache_seqlens_int32=tail_counts,
        )
        run = self._run_decode if is_decode else self._run_extend
        main_output, main_lse = run(
            binding=main_binding,
            kv_cache=kv_c_and_k_pe_cache,
            sm_scale=self.scale,
            v_head_dim=self.kv_lora_rank,
            return_lse=True,
            lse_scale="natural",
        )
        tail_output, tail_lse = run(
            binding=tail_binding,
            kv_cache=kv_c_and_k_pe_cache,
            sm_scale=self.scale,
            v_head_dim=self.kv_lora_rank,
            return_lse=True,
            lse_scale="natural",
        )

        merged_output = merged_output_buffer[:num_tokens]
        merged_lse = merged_lse_buffer[:num_tokens]
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


__all__ = ["Glm5NextB12xMLASparseBackend", "Glm5NextB12xMLASparseImpl"]
