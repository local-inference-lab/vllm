# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 DSpark with Afterburner feature-conditioning modules."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.weight_transfer import allocate_weights
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
)


class AfterburnerBlock(nn.Module):
    def __init__(self, width: int, intermediate: int, residual: bool, prefix: str):
        super().__init__()
        self.residual = residual
        self.norm = RMSNorm(width, eps=1e-6, dtype=torch.float32)
        self.gate_up = ReplicatedLinear(
            width,
            2 * intermediate,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "gate_up"),
        )
        self.down = ReplicatedLinear(
            intermediate,
            width,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "down"),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(self.norm(hidden)).chunk(2, dim=-1)
        output = self.down(F.silu(gate) * up)
        return hidden + output if self.residual else output


class Afterburner(nn.Module):
    def __init__(self, input_width: int, output_width: int, config, prefix: str):
        super().__init__()
        self.projection = ReplicatedLinear(
            input_width,
            output_width,
            bias=True,
            return_bias=False,
            prefix=maybe_prefix(prefix, "projection"),
        )
        self.blocks = nn.ModuleList(
            [
                AfterburnerBlock(
                    output_width,
                    config.afterburner_intermediate_size,
                    config.afterburner_residual,
                    maybe_prefix(prefix, f"blocks.{i}"),
                )
                for i in range(config.afterburner_depth)
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class Glm53DSparkModel(DSparkDeepseekV4Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        spec = vllm_config.speculative_config
        assert spec is not None
        if config.target_hidden_size != config.hidden_size:
            raise ValueError("GLM-5.3 DSpark requires matching target and draft widths")
        self.context_afterburners = nn.ModuleList(
            [
                Afterburner(
                    config.target_hidden_size,
                    config.hidden_size,
                    config,
                    maybe_prefix(prefix, f"context_afterburners.{i}"),
                )
                for i in range(len(self.target_layer_ids))
            ]
        )
        self.token_afterburner = Afterburner(
            config.target_hidden_size,
            config.hidden_size,
            config,
            maybe_prefix(prefix, "token_afterburner"),
        )
        self.output_afterburner = Afterburner(
            config.hidden_size,
            config.target_hidden_size,
            config,
            maybe_prefix(prefix, "output_afterburner"),
        )
        self.noise_embedding = nn.Parameter(
            allocate_weights(torch.empty, config.hidden_size), requires_grad=False
        )
        max_queries = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs * spec.num_speculative_tokens,
        )
        self.register_buffer(
            "_noise_positions",
            (torch.arange(max_queries) % spec.num_speculative_tokens != 0).unsqueeze(
                -1
            ),
            persistent=False,
        )
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.float()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        taps = aux_hidden_states.chunk(len(self.context_afterburners), dim=-1)
        context = torch.cat(
            [
                afterburner(tap)
                for afterburner, tap in zip(self.context_afterburners, taps)
            ],
            dim=-1,
        )
        return self.main_norm(self.main_proj(context))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        query = self.token_afterburner(inputs_embeds)
        query = torch.where(
            self._noise_positions[: query.shape[0]], self.noise_embedding, query
        )
        return super().forward(input_ids, positions, inputs_embeds=query)


class Glm53DSparkForCausalLM(DSparkDeepseekV4ForCausalLM):
    model_cls = Glm53DSparkModel
    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.quant_config is not None:
            raise ValueError("GLM-5.3 DSpark requires unquantized draft weights")
        assert vllm_config.speculative_config is not None
        draft_config = copy.copy(vllm_config)
        draft_config.model_config = vllm_config.speculative_config.draft_model_config
        with set_current_vllm_config(draft_config):
            super().__init__(vllm_config=draft_config, prefix=prefix)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.model.output_afterburner(self.model.norm(hidden_states))
        return self.logits_processor(self.lm_head, projected)

    def _remap_dspark_name(self, name: str) -> str | None:
        if name == "lm_head.weight":
            return name
        if (
            name == "embed_tokens.weight"
            or name == "noise_embedding"
            or name.startswith(
                ("context_afterburners.", "token_afterburner.", "output_afterburner.")
            )
        ):
            return f"model.{name}"
        mapped = super()._remap_dspark_name(name)
        if mapped is None:
            raise ValueError(f"Unexpected GLM-5.3 DSpark tensor: {name}")
        return mapped
