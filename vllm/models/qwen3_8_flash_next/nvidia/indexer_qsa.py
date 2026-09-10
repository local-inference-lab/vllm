# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated QSA index projection for Qwen3.8-Flash-Next."""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from ..config import Qwen3_8FlashNextTextConfig


class QSAIndexer(nn.Module):
    """Project raw index Q/K while exposing learned selector norm weights.

    The b12x QSA transaction owns Q/K normalization, RoPE, streaming
    compression, selection, and sparse attention for both prefill and decode.
    Keeping this module to checkpoint-owned parameters prevents a second cache
    policy from leaking into the model definition.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3_8FlashNextTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("QSA currently requires BF16 activations")

        self.index_q_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        if self.index_kv_heads != 1:
            raise ValueError("QSA requires exactly one index KV head")
        output_size = (self.index_q_heads + self.index_kv_heads) * self.index_head_dim
        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(config.rms_norm_eps),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(config.rms_norm_eps),
        )

    def project(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw ``[T, Hi, Di]`` index queries and ``[T, Di]`` keys."""

        projected, _ = self.index_qk_proj(hidden_states)
        return self.split_projection(projected)

    def split_projection(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_width = self.index_q_heads * self.index_head_dim
        index_query, raw_index_key = projected.split(
            (query_width, self.index_head_dim), dim=-1
        )
        return (
            index_query.unflatten(-1, (self.index_q_heads, self.index_head_dim)),
            raw_index_key,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.project(hidden_states)


__all__ = ["QSAIndexer"]
