# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract for ``QwenGatedDeltaNetAttention._output_projection``.

Every ``forward_*`` already allocates ``core_attn_out`` and ``z`` as
``(N, H, D)``. The helper now norms those tensors as-is and flattens once
for ``out_proj``; it used to be rank-agnostic via ``reshape(z_shape_og)``.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_output_projection_norms_per_head_and_flattens(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    num_tokens, num_heads, head_dim = 3, 4, 8
    hidden = num_heads * head_dim

    layer = types.SimpleNamespace()
    layer.norm = RMSNormGated(
        head_dim,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device="cpu",
        dtype=dtype,
    )
    layer.out_proj = lambda x: (x, None)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )

    core_attn_out = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    out = layer._output_projection(core_attn_out, z)

    assert core_attn_out.shape == (num_tokens, num_heads, head_dim)
    assert z.shape == (num_tokens, num_heads, head_dim)
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype


@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
@torch.inference_mode()
def test_b12x_forward_keeps_prepared_prefill_and_decode_entrypoint(
    num_speculative_tokens: int,
) -> None:
    """B12X uses its prepared packed path even when CUDA MTP fusion is disabled."""
    num_tokens = 1 + num_speculative_tokens
    hidden_states = torch.empty(num_tokens, 8, dtype=torch.bfloat16)
    projected = torch.randn(num_tokens, 48, dtype=torch.bfloat16)
    gates = torch.randn(num_tokens, 4, dtype=torch.bfloat16)
    layer = types.SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        gdn_decode_kernel="b12x",
        enable_fused_gdn_spec_decode=False,
        overlap_input_projections=False,
        gqa_interleaved_layout=False,
        key_dim=8,
        value_dim=16,
        num_v_heads=2,
        tp_size=1,
        head_v_dim=8,
        norm=types.SimpleNamespace(weight=torch.ones(8, dtype=torch.bfloat16)),
        in_proj_qkvz=lambda _: (projected, None),
        in_proj_ba=lambda _: (gates, None),
        split_ba=lambda ba: ba.chunk(2, dim=-1),
        out_proj=lambda output: (output, None),
    )

    def packed_op(qkvz, ba, output, *, layer_name):
        assert qkvz is projected and ba is gates
        output.copy_(torch.arange(output.numel()).reshape_as(output))

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed,
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core",
            side_effect=AssertionError("Prepared B12X path was bypassed"),
        ),
    ):
        forward = types.MethodType(QwenGatedDeltaNetAttention.forward_cuda, layer)
        output = forward(hidden_states)
    packed.assert_called_once()
    expected = torch.arange(num_tokens * 16, dtype=torch.bfloat16).view(num_tokens, 16)
    torch.testing.assert_close(output, expected)
