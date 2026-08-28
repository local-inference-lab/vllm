# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter."""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("glm53_kda_decode_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        if backend not in ("auto", "b12x", "triton"):
            raise ValueError(
                "GLM-5.3 KDA decode backend must be 'auto', 'b12x', or "
                "'triton', "
                f"got {backend!r}."
            )
        self._glm53_kda_decode_backend = backend
        self.enable_b12x_kda_decode = backend != "triton"
        super().__init__(config, vllm_config, prefix)

    def _can_use_b12x_kda_decode(self, m: GDNAttentionMetadata) -> bool:
        backend = self._glm53_kda_decode_backend
        if backend == "triton":
            return False
        if backend == "auto" and m.spec_sequence_masks is None:
            return False
        return super()._can_use_b12x_kda_decode(m)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        super().forward(hidden_states, positions, output)
        return output


__all__ = ["Glm5NextLinearAttention"]
