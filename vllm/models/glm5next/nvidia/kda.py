# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter.

Decode-latency variant of the shared KDA forward: the two low-rank output-gate
projections (``g_a_proj`` then ``g_b_proj``) depend only on the layer input, so
they are issued on a side CUDA stream and overlap the large fused
``in_proj_qkvgfab`` GEMM instead of trailing it on the main stream. Kernels,
shapes and reduction orders are unchanged, so results are bitwise identical to
the sequential path; only the stream fork/join edges are new. The fork/join
uses ``wait_stream``, which CUDA graph capture records as dependency edges.
``VLLM_GLM53_KDA_GATE_SIDE_STREAM=0`` restores the sequential forward.
"""

import os

import torch
from einops import rearrange

from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)

_GATE_SIDE_STREAM = os.getenv("VLLM_GLM53_KDA_GATE_SIDE_STREAM", "1") != "0"
_side_streams: dict[int, torch.cuda.Stream] = {}


def _gate_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _side_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=torch.device("cuda", index))
        _side_streams[index] = stream
    return stream


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if _GATE_SIDE_STREAM and not self.use_full_rank_gate:
            self._forward_gate_overlap(hidden_states, output)
        else:
            super().forward(hidden_states, positions, output)
        return output

    def _forward_gate_overlap(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        main = torch.cuda.current_stream(hidden_states.device)
        side = _gate_stream(hidden_states.device)

        # Fork: the gate projections read only hidden_states.
        side.wait_stream(main)
        # Keep hidden_states alive for the side stream (allocator safety in
        # eager/warmup mode; a no-op inside graph capture).
        hidden_states.record_stream(side)
        with torch.cuda.stream(side):
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        # Same optional callback the shared forward offers after its first
        # projection (e.g. the L2 weight prefetch of o_proj).
        hook = getattr(self, "_l2_prefetch_hook", None)
        if hook is not None:
            hook(num_tokens)
        mixed_qkv, beta, f_a = projected_qkvgfab.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
            ],
            dim=-1,
        )
        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        # Join before anything reads the gate states.
        main.wait_stream(side)
        g_proj_states.record_stream(main)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]


__all__ = ["Glm5NextLinearAttention"]
