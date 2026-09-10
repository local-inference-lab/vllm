# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    yarn_get_mscale,
)
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .pooled_indexer import Glm5NextPooledIndexer


def _select_sparse_backend(
    vllm_config: VllmConfig,
    attn_backend: type | None,
) -> type | None:
    if (
        attn_backend is None
        and vllm_config.attention_config.backend == AttentionBackendEnum.B12X
    ):
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
            B12xGLM5NextMLASparseBackend,
        )

        return B12xGLM5NextMLASparseBackend
    return attn_backend


class Glm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        pool_topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        skip_rope: bool | None = False,
        is_mtp_layer: bool = False,
        attn_backend: type | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError("num_heads must be divisible by tensor parallel size")
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        proj_input_size = input_size if input_size is not None else hidden_size

        if q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                kv_lora_rank + qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not skip_rope:
            if config.rope_parameters is None:
                raise ValueError("RoPE-enabled GLM MLA requires rope_parameters")
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if config.rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )
            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            if config.rope_parameters["rope_type"] == "deepseek_yarn":
                mscale = yarn_get_mscale(
                    config.rope_parameters["factor"],
                    float(config.rope_parameters.get("mscale_all_dim", False)),
                )
                self.scaling *= mscale * mscale
        else:
            self.rotary_emb = None

        self.is_sparse = getattr(config, "index_topk", None) is not None
        if self.is_sparse:
            attn_backend = _select_sparse_backend(vllm_config, attn_backend)
            if q_lora_rank is None:
                raise ValueError("GLM sparse MLA requires q_lora_rank")
            self.indexer: Glm5NextPooledIndexer | None = Glm5NextPooledIndexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                pool_topk_indices_buffer,
                main_layer_name=f"{prefix}.attn",
                prefix=f"{prefix}.indexer",
                # MTP compacts and reuses request-relative selections.
                emit_physical_selection=not is_mtp_layer,
            )
        else:
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=(
                self.fused_qkv_a_proj if q_lora_rank is not None else None
            ),
            kv_a_proj_with_mqa=(
                self.kv_a_proj_with_mqa if q_lora_rank is None else None
            ),
            q_a_layernorm=self.q_a_layernorm if q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if q_lora_rank is not None else None,
            q_proj=self.q_proj if q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=None,
            is_sparse=self.is_sparse,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            attn_backend=attn_backend,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


__all__ = ["Glm5NextMLAAttention"]
