# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


class Glm5NextKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    supports_varlen_decode_cudagraph = True

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Adaptive verification supplies packed boundaries on device, while the
        # CPU lengths describe an even distribution of the same token budget.
        self._reuse_spec_decode_inputs = False


class Glm5NextKDAAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KDA"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextKDAMetadataBuilder]:
        return Glm5NextKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def get_attn_backend(self) -> type[AttentionBackend]:
        if (
            self.speculative_config is not None
            and self.speculative_config.enable_adaptive_verification
            and self.cache_config.mamba_cache_mode == "align"
        ):
            return Glm5NextKDAAttentionBackend
        return super().get_attn_backend()

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        defer_tp_reduction: bool = False,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if defer_tp_reduction:
            super().forward(hidden_states, positions, output, defer_tp_reduction=True)
        else:
            super().forward(hidden_states, positions, output)
        return output


__all__ = ["Glm5NextLinearAttention"]
