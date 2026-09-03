# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse-MLA components for DSA models on NVIDIA GPUs."""

import torch

from vllm.config import VllmConfig
from vllm.models.deepseek_v32.attention import (
    DeepseekV32Attention,
    DeepseekV32Indexer,
)
from vllm.v1.attention.backends.mla.b12x_indexer import (
    B12xIndexerCache,
    B12xSparseIndexer,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLMDSAMLASparseBackend,
    B12xMLASparseBackend,
)


def _get_sparse_backend_cls(vllm_config: VllmConfig) -> type[B12xMLASparseBackend]:
    """Select the packed-cache contract required by the DSA architecture."""
    hf_config = vllm_config.model_config.hf_text_config
    if getattr(hf_config, "model_type", None) == "glm_moe_dsa":
        return B12xGLMDSAMLASparseBackend
    return B12xMLASparseBackend


class B12xDSAIndexer(DeepseekV32Indexer):
    indexer_cache_cls = B12xIndexerCache
    indexer_op_cls = B12xSparseIndexer

    @staticmethod
    def get_indexer_op_kwargs(vllm_config: VllmConfig) -> dict[str, int | bool]:
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError("B12X sparse MLA does not support PCP.")
        return {
            "skip_k_cache_insert": True,
            "num_q_heads": int(vllm_config.model_config.hf_text_config.index_n_heads),
            "output_physical_slots": (
                vllm_config.parallel_config.decode_context_parallel_size == 1
            ),
        }

    def run_indexer(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
        *,
        use_pcp: bool,
        dense_mha_metadata_layer_name: str,
        dcp_rank: int,
        dcp_world_size: int,
        cp_kv_cache_interleave_size: int,
    ) -> torch.Tensor:
        del (
            use_pcp,
            dense_mha_metadata_layer_name,
            dcp_rank,
            dcp_world_size,
            cp_kv_cache_interleave_size,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)


class DeepseekV32B12xAttention(DeepseekV32Attention):
    indexer_cls = B12xDSAIndexer

    def __init__(self, vllm_config, config, prefix, topk_indices_buffer=None):
        super().__init__(
            vllm_config,
            config,
            prefix,
            topk_indices_buffer,
            attn_backend=_get_sparse_backend_cls(vllm_config),
        )


__all__ = ["B12xDSAIndexer", "DeepseekV32B12xAttention"]
