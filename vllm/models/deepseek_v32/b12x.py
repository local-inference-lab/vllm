# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse-MLA components for DeepSeek V3.2-compatible models."""

import torch

from vllm.config import VllmConfig
from vllm.models.deepseek_v32.attention import DeepseekV32Indexer
from vllm.v1.attention.backends.mla.b12x_indexer import (
    B12xIndexerCache,
    B12xSparseIndexer,
)


class B12xDeepseekV32Indexer(DeepseekV32Indexer):
    indexer_cache_cls = B12xIndexerCache
    indexer_op_cls = B12xSparseIndexer

    @staticmethod
    def get_indexer_op_kwargs(vllm_config: VllmConfig) -> dict[str, bool]:
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError("B12X sparse MLA does not support PCP.")
        return {"skip_k_cache_insert": True}

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


__all__ = ["B12xDeepseekV32Indexer"]
