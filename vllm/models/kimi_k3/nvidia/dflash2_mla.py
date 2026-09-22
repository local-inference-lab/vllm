# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DFlash2 draft with grouped convolutions and a candidate transition head."""

from collections.abc import Iterable

import torch

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dflash import _resolve_layer_attention
from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    DFlashGroupedConv,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkDecoderLayer,
    K3DSparkForCausalLM,
    K3DSparkModel,
)


class DFlash2KimiK3DecoderLayer(K3DSparkDecoderLayer):
    @staticmethod
    def get_attention_window(config, layer_idx: int) -> int:
        window, causal = _resolve_layer_attention(config, layer_idx)
        if causal:
            raise ValueError("Kimi-K3 MLA DFlash2 requires non-causal draft layers")
        # Zero explicitly preserves full attention instead of inheriting the
        # DSpark tail-window override. Sliding layers retain checkpoint windows.
        return window or 0

    def __init__(self, *, vllm_config, config, layer_idx, start_layer_id, prefix):
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft = config.dflash_config
        spec = vllm_config.speculative_config
        assert spec is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft["conv_kernel_size"]),
            group_size=int(draft["conv_group_size"]),
            block_size=1 + spec.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(self, positions, hidden_states, residual, rope_cos_sin_cache=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rope_cos_sin_cache=rope_cos_sin_cache,
        )
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.mlp_conv.finish(hidden_states, coefficients), residual


class DFlash2KimiK3Model(K3DSparkModel):
    decoder_layer_cls = DFlash2KimiK3DecoderLayer
    has_markov_head = False

    def __init__(self, *, vllm_config, start_layer_id, prefix):
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        draft = self.config.dflash_config
        self.input_embedding_scale = float(draft.get("input_embedding_scale", 1.0))
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft["selector_rank"]),
                top_k=int(draft["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids):
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2KimiK3ForCausalLM(K3DSparkForCausalLM):
    model_cls = DFlash2KimiK3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        draft = config.dflash_config
        if draft.get("attention_mode") != "mla" or config.q_lora_rank is None:
            raise ValueError("Kimi-K3 DFlash2 requires latent Q and KV projections")
        config.target_layer_ids = list(draft["target_layer_ids"])
        if not config.target_layer_ids:
            raise ValueError("Kimi-K3 DFlash2 requires target auxiliary layer IDs")
        config.target_hidden_size = getattr(
            config, "target_hidden_size", config.hidden_size
        )
        config.num_target_layers = len(config.target_layer_ids)
        config.draft_vocab_size = config.vocab_size
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        softcap = float(draft.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft.get("output_multiplier", 1.0)),
            soft_cap=softcap or None,
        )

    def compute_candidates(self, hidden_states):
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap():
            names = {
                "fc": "context_proj",
                "hidden_norm": "context_norm",
                "norm": "final_norm",
            }
            for name, tensor in weights:
                first, separator, suffix = name.partition(".")
                yield names.get(first, first) + separator + suffix, tensor

        return super().load_weights(remap())
