# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X-backed HyperConnection modules for Qwen3.8-Flash-Next."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.weight_transfer import allocate_weights
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_hyperconnection


def _hyperconnection_api() -> Any:
    api = get_b12x_hyperconnection()
    if api is None:
        raise ImportError(
            "Qwen3.8-Flash-Next requires b12x.norm.hyperconnection; "
            "install the b12x serving extra"
        )
    return api


@dataclass(frozen=True)
class HyperConnectionConfig:
    hc_count: int
    hidden_size: int
    params_dtype: torch.dtype
    hc_lowrank: int
    rms_norm_eps: float
    hc_per_branch_norm: bool = True


class GroupedGemmaRMSNorm(nn.Module):
    """Checkpoint-compatible zero-centered grouped RMSNorm weight."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by group_size={group_size}"
            )
        self.eps = float(eps)
        self.group_size = group_size
        self.weight = nn.Parameter(
            allocate_weights(torch.zeros, hidden_size, dtype=dtype)
        )


class HyperConnectionWorkspace(nn.Module):
    """Fixed-capacity storage shared by all HC modules in one model."""

    def __init__(self, config: HyperConnectionConfig, max_tokens: int) -> None:
        super().__init__()
        if not config.hc_per_branch_norm:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next requires one RMSNorm group per HC stream"
            )
        api = _hyperconnection_api()
        device = current_platform.current_device()
        self.plan = api.plan(
            api.Caps(
                device=device,
                max_tokens=max_tokens,
                hidden_size=config.hidden_size,
                streams=config.hc_count,
                lowrank=config.hc_lowrank,
                dtype=config.params_dtype,
            )
        )
        width = config.hc_count * config.hidden_size
        factory = dict(device=device, dtype=config.params_dtype)
        self.register_buffer(
            "normalized", torch.empty(max_tokens, width, **factory), persistent=False
        )
        self.register_buffer(
            "bottleneck",
            torch.empty(max_tokens, config.hc_lowrank, **factory),
            persistent=False,
        )
        self.register_buffer(
            "block_input",
            torch.empty(max_tokens, config.hidden_size, **factory),
            persistent=False,
        )

        # Validate the fixed output layout before Dynamo traces live prefix
        # views. The registered buffers and their addresses do not change.
        self.bind(max_tokens)

    def bind(self, tokens: int):
        return self.plan.bind(
            normalized=self.normalized,
            bottleneck=self.bottleneck,
            block_input=self.block_input,
            tokens=tokens,
        )


class GatedResidual(nn.Module):
    """Learned HC mixer with B12X pointwise and residual kernels."""

    def __init__(
        self,
        config: HyperConnectionConfig,
        workspace: HyperConnectionWorkspace,
        *,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.params_dtype != torch.bfloat16:
            raise TypeError("Qwen3.8-Flash-Next HC requires BF16 parameters")
        self.config = config
        # The workspace is owned once by the enclosing model.  Keeping a plain
        # reference here avoids registering the same large buffer module under
        # every decoder block.
        object.__setattr__(self, "_workspace", workspace)
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = self.hyper_hidden_size
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=config.hidden_size,
            dtype=config.params_dtype,
        )

        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            sizes = [self.lora_rank, self.hc_count]
            if self.pad_size:
                sizes.append(self.pad_size)
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                sizes,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def workspace(self) -> HyperConnectionWorkspace:
        return self._workspace

    def _binding(self, hidden_states: torch.Tensor):
        return self.workspace.bind(hidden_states.shape[0])

    def _mix_normalized(self, normalized: torch.Tensor, binding):
        api = _hyperconnection_api()
        if self.use_combine:
            down_and_injection = self.input_mix_weight_down_block_inject(normalized)
            projected_down = down_and_injection[:, : self.lora_rank]
            injection_start = self.lora_rank
            # The projection owner stays live through the downstream residual
            # combine; readers consume row-strided slices without staging.
            injection = down_and_injection[
                :, injection_start : injection_start + self.hc_count
            ]
        else:
            projected_down = self.input_mix_weight_down(normalized)
            injection = None

        bottleneck = api.run_scaled_silu(projected_down, binding=binding)
        gate_logits = self.input_mix_weight_up(bottleneck)
        block_input = api.run_gate_mean(normalized, gate_logits, binding=binding)
        return block_input, injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        api = _hyperconnection_api()
        binding = self._binding(hidden_states)
        normalized = api.run_grouped_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            eps=self.config.rms_norm_eps,
            binding=binding,
        )
        block_input, injection = self._mix_normalized(normalized, binding)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        api = _hyperconnection_api()
        combined, normalized = api.run_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            eps=self.config.rms_norm_eps,
            plan=self.workspace.plan,
        )
        binding = self._binding(normalized)
        block_input, injection = self._mix_normalized(normalized, binding)
        return combined, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        api = _hyperconnection_api()
        return api.run_combine(
            hidden_states,
            block_output,
            injection,
            plan=self.workspace.plan,
        )


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
    "HyperConnectionWorkspace",
]
