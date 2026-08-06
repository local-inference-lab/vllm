# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA draft model for DSpark speculative decoding."""

from collections.abc import Iterable

import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.deepseek_v32.nvidia.fused_ops import fused_allreduce_rms_norm

from .deepseek_v2 import DeepseekV2MLAAttention
from .kimi_linear import KimiMLP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)


class K3DSparkMarkovHead(nn.Module):
    """Replicated sequential transition head used by K3 DSpark."""

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=True,
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        return logits_processor(self.markov_w2, markov_embed)


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        quant_config = get_draft_quant_config(vllm_config)
        self.self_attn = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id + layer_idx}.self_attn"
            ),
            reduce_results=False,
            non_causal_multi_token_decode=True,
        )
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        hidden_states = self.self_attn(positions, hidden_states, None)
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.embed_tokens: nn.Module | None = None

        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = K3DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            kv_lora = qkv_lora[..., attn.q_lora_rank :]
            kv_c, k_pe = kv_lora.split(
                [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
            )
            kv_c = attn.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            ops.rotary_embedding(
                context_positions,
                k_pe,
                None,
                attn.rotary_emb.head_size,
                attn.rotary_emb.cos_sin_cache,
                attn.rotary_emb.is_neox_style,
            )

            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            inner = attn.mla_attn.mla_attn
            inner.impl.do_kv_cache_update(
                kv_c,
                k_pe,
                inner.kv_cache,
                slot_mapping,
                inner.kv_cache_dtype,
                inner._k_scale,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states, residual, self.final_norm
        )
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.self_attn.mla_attn.mla_attn.layer_name for layer in self.model.layers
        ]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["confidence_head", "embed_tokens", "lm_head"],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
