# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


def _unquantized_output_projection(
    input_size: int,
    output_size: int,
    *,
    tp_size: int = 2,
):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    projection = torch.nn.Module()
    projection.quant_method = UnquantizedLinearMethod()
    projection.input_is_parallel = True
    projection.input_size_per_partition = input_size
    projection.output_size = output_size
    projection.reduce_results = True
    projection.tp_size = tp_size
    projection.register_parameter("bias", None)
    projection.weight = torch.nn.Parameter(
        torch.randn(output_size, input_size),
        requires_grad=False,
    )
    return projection


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


def test_kimi_mla_decode_query_uses_replicated_absorbed_weight(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_lora_rank = 5
    attention.dcp_q_replicate = True
    attention.W_UK_T = torch.nn.Parameter(
        torch.randn((2, 3, 5), dtype=torch.bfloat16), requires_grad=False
    )
    attention.W_UK_T_dcp_qrep = torch.randn((4, 3, 5), dtype=torch.bfloat16)
    query = torch.randn((3, 4, 3), dtype=torch.bfloat16)
    captured = {}

    def safe_bmm(q, weight, output, *, use_safe_op):
        captured["weight"] = weight
        torch.bmm(q.contiguous(), weight, out=output)

    monkeypatch.setattr(mla, "_run_mla_query_bmm", safe_bmm)

    result = attention._absorb_decode_query(query)

    assert captured["weight"] is attention.W_UK_T_dcp_qrep
    expected = torch.bmm(
        query.transpose(0, 1).contiguous(),
        attention.W_UK_T_dcp_qrep,
    )
    torch.testing.assert_close(result, expected.transpose(0, 1))


def test_kimi_mla_qrep_layer_allowlist(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        )
    )
    monkeypatch.setattr(mla.envs, "VLLM_DCP_Q_REPLICATE", True)
    monkeypatch.setattr(
        mla.envs,
        "VLLM_K3_DCP_Q_REPLICATE_LAYERS",
        "4,12-16,92",
    )

    assert mla._k3_dcp_qrep_enabled("model.layers.4.self_attn", config)
    assert mla._k3_dcp_qrep_enabled("model.layers.14.self_attn", config)
    assert not mla._k3_dcp_qrep_enabled("model.layers.20.self_attn", config)
    assert mla._k3_dcp_qrep_enabled("model.layers.92.self_attn", config)


def test_kimi_mla_qrep_requires_explicit_memory_policy(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        )
    )
    monkeypatch.setattr(mla.envs, "VLLM_DCP_Q_REPLICATE", True)
    monkeypatch.setattr(mla.envs, "VLLM_K3_DCP_Q_REPLICATE_LAYERS", None)

    with pytest.raises(ValueError, match="explicit layer list"):
        mla._k3_dcp_qrep_enabled("model.layers.4.self_attn", config)


def test_kimi_mla_qrep_explicit_all(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        )
    )
    monkeypatch.setattr(mla.envs, "VLLM_DCP_Q_REPLICATE", True)
    monkeypatch.setattr(mla.envs, "VLLM_K3_DCP_Q_REPLICATE_LAYERS", "all")

    assert mla._k3_dcp_qrep_enabled("model.layers.4.self_attn", config)


def test_kimi_mla_qrep_uses_dcp_group_head_width():
    from vllm.models.kimi_k3.nvidia import mla

    assert mla._k3_projected_query_heads(12, 8, True) == 96
    assert mla._k3_projected_query_heads(6, 8, True) == 48
    assert mla._k3_projected_query_heads(6, 8, False) == 6


def test_kimi_mla_qrep_rejects_invalid_layer_range(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        )
    )
    monkeypatch.setattr(mla.envs, "VLLM_DCP_Q_REPLICATE", True)
    monkeypatch.setattr(
        mla.envs,
        "VLLM_K3_DCP_Q_REPLICATE_LAYERS",
        "16-12",
    )

    with pytest.raises(ValueError, match="Invalid K3 qrep layer range"):
        mla._k3_dcp_qrep_enabled("model.layers.12.self_attn", config)


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


@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_kimi_mla_context_output_reuses_consumed_query_bytes(query_dtype):
    from vllm.models.kimi_k3.nvidia import mla

    query = torch.empty((4, 2, 256), dtype=query_dtype)
    output = torch.randn((4, 2, 128), dtype=torch.bfloat16)

    compact = mla._reuse_consumed_query_for_context_output(query, output)
    compact.copy_(output)

    assert compact.data_ptr() == query.data_ptr()
    assert compact.shape == output.shape
    assert compact.dtype == output.dtype
    assert compact.is_contiguous()
    torch.testing.assert_close(compact, output)


def test_kimi_mla_context_output_rejects_insufficient_query_storage():
    from vllm.models.kimi_k3.nvidia import mla

    query = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    output = torch.empty((4, 2, 128), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="too small"):
        mla._reuse_consumed_query_for_context_output(query, output)


def test_kimi_mla_caller_output_selection_preserves_decode_and_sp_paths():
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.o_proj = _unquantized_output_projection(3, 4)

    assert not attention.should_use_caller_output(torch.empty(8, 4))
    assert attention.should_use_caller_output(torch.empty(1024, 4))

    attention.o_proj.reduce_results = False
    assert not attention.should_use_caller_output(torch.empty(1024, 4))


def test_kimi_mla_forward_reuses_consumed_hidden_state(monkeypatch):
    from vllm.models.kimi_k3.nvidia import mla

    attention = object.__new__(mla.MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.o_proj = _unquantized_output_projection(3, 4)
    attention.g_proj = None
    attention._gate_events = None
    attention.aux_stream = None

    attn_out = torch.randn(1024, 3)
    attention._forward_attn = lambda *_args, **_kwargs: attn_out

    reduced_pointers: list[int] = []

    def reduce_output(value: torch.Tensor, tp_size: int) -> torch.Tensor:
        assert tp_size == 2
        reduced_pointers.append(value.data_ptr())
        value.mul_(2)
        return value

    monkeypatch.setattr(mla, "reduce_kimi_full_width_projection", reduce_output)

    hidden_states = torch.empty(1024, 4)
    output_pointer = hidden_states.data_ptr()
    expected = torch.mm(attn_out, attention.o_proj.weight.t()) * 2

    actual = attention(
        torch.arange(1024),
        hidden_states,
        output=hidden_states,
    )

    assert actual.data_ptr() == output_pointer
    assert reduced_pointers == [output_pointer]
    torch.testing.assert_close(actual, expected)
