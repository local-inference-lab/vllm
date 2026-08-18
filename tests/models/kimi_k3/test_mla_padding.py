# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_kimi_mla_absorbed_weight_preallocation_uses_local_heads():
    from vllm.model_executor.layers.attention.mla_attention import (
        _preallocate_absorbed_mla_weights,
    )

    projection = torch.nn.Module()
    projection.weight = torch.nn.Parameter(torch.empty((1, 1)))
    attention = SimpleNamespace(
        num_heads=96,
        num_local_heads=6,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        v_head_dim=128,
        kv_b_proj=projection,
    )

    w_uv, w_uk_t = _preallocate_absorbed_mla_weights(attention, torch.bfloat16)

    assert w_uv is not None
    assert w_uk_t is not None
    assert w_uv.shape == (6, 512, 128)
    assert w_uk_t.shape == (6, 128, 512)
    assert w_uv.is_contiguous()
    assert w_uk_t.is_contiguous()


def test_kimi_mla_decode_query_materializes_interleaved_heads(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_lora_rank = 5
    attention.W_UK_T = torch.nn.Parameter(
        torch.randn((2, 3, 5), dtype=torch.bfloat16), requires_grad=False
    )
    query_storage = torch.randn((4, 2, 7), dtype=torch.bfloat16)
    query = query_storage[..., :3]
    calls = []

    def safe_bmm(q, weight, output, *, use_safe_op):
        calls.append((q, weight, use_safe_op))
        torch.bmm(q.contiguous(), weight, out=output)

    monkeypatch.setattr(mla, "_run_mla_query_bmm", safe_bmm)

    result = attention._absorb_decode_query(query)

    assert len(calls) == 1
    captured_query, captured_weight, use_safe_op = calls[0]
    assert captured_query.data_ptr() != query.data_ptr()
    assert captured_query.shape == query.transpose(0, 1).shape
    assert captured_query.is_contiguous()
    assert captured_weight is attention.W_UK_T
    assert use_safe_op is True
    expected = torch.bmm(query.transpose(0, 1).contiguous(), attention.W_UK_T)
    torch.testing.assert_close(result, expected.transpose(0, 1))


def test_kimi_mla_defines_graph_padding_before_output_projection(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_name = "model.layers.0.self_attn"
    attention.rotary_emb = None

    metadata = SimpleNamespace(num_actual_tokens=2, num_decode_tokens=0)
    context = SimpleNamespace(
        attn_metadata={attention.layer_name: metadata},
        slot_mapping={attention.layer_name: torch.arange(4)},
    )
    monkeypatch.setattr(mla, "get_forward_context", lambda: context)

    def write_active_prefill(*args):
        args[-1].fill_(3)

    attention._forward_prefill_fused = write_active_prefill
    output = torch.full((4, 8), 9, dtype=torch.bfloat16)
    attention_method = type(attention)._attention
    invoke_attention = getattr(attention_method, "__wrapped__", attention_method)
    invoke_attention(
        attention,
        torch.arange(4),
        torch.zeros((4, 1, 8), dtype=torch.bfloat16),
        torch.zeros((4, 8), dtype=torch.bfloat16),
        torch.zeros((4, 8), dtype=torch.bfloat16),
        output,
    )

    torch.testing.assert_close(output[:2], torch.full_like(output[:2], 3))
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))
