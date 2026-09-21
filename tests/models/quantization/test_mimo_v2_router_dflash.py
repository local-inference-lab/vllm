# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router dtype and DFlash value scaling, with independent CPU expectations."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.mimo_v2 as mimo
import vllm.model_executor.models.qwen3_dflash as dflash

pytestmark = pytest.mark.cpu_test


def test_bf16_router_emits_fp32_logits(monkeypatch):
    for module in (
        mimo,
        __import__("vllm.model_executor.layers.linear", fromlist=["_"]),
        __import__("vllm.model_executor.parameter", fromlist=["_"]),
    ):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mimo, "get_ep_group", lambda: SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 1)))
    config = SimpleNamespace(
        hidden_size=32, n_routed_experts=8, hidden_act="silu",
        moe_router_dtype="bfloat16", num_experts_per_tok=2,
        moe_intermediate_size=64, norm_topk_prob=True, n_group=1, topk_group=1,
    )
    vconfig = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config), quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False,
            enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0)),
    )
    monkeypatch.setattr(mimo, "get_current_vllm_config", lambda: vconfig)
    captured = {}

    class Experts(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        def forward(self, hidden_states, router_logits):
            captured["logits"] = router_logits
            return hidden_states

    monkeypatch.setattr(mimo, "FusedMoEFactory", Experts)
    module = mimo.MiMoV2MoE(vconfig)
    with torch.no_grad():
        module.gate.weight.copy_(torch.arange(256).view(8, 32) / 256)
    hidden = torch.linspace(-1, 1, 96).view(3, 32).bfloat16()
    result = module(hidden)
    assert module.gate.weight.dtype == torch.bfloat16
    assert captured["router_logits_dtype"] == torch.float32
    assert module.gate.e_score_correction_bias.dtype == torch.float32
    reference = torch.nn.functional.linear(hidden, module.gate.weight).float()
    torch.testing.assert_close(captured["logits"], reference)
    torch.testing.assert_close(result, hidden)


@pytest.mark.parametrize("value_scale", [None, 0.612])
def test_dflash_query_values_scaled_once(value_scale):
    qkv = torch.arange(24, dtype=torch.float32).view(2, 12)
    captured = {}

    def attention(q, k, v):
        captured["v"] = v.clone()
        return v

    module = SimpleNamespace(
        qkv_proj=lambda _: (qkv, None), q_size=4, kv_size=4, head_dim=2,
        q_norm=lambda x: x, k_norm=lambda x: x,
        rotary_emb=lambda positions, q, k: (q, k), attn=attention,
        o_proj=lambda x: (x, None), v_scale=value_scale,
    )
    output = dflash.DFlashQwen3Attention.forward(
        module, torch.arange(2), torch.empty(2, 4)
    )
    expected = qkv[:, 8:] * (1 if value_scale is None else value_scale)
    torch.testing.assert_close(captured["v"], expected)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("value_scale", [None, 0.612])
@pytest.mark.parametrize("per_layer", [False, True])
def test_dflash_context_values_scaled_once(monkeypatch, value_scale, per_layer):
    # Preserve the complete context method and intercept only native RoPE and
    # the cache boundary. Different slot lengths exercise per-layer selection.
    monkeypatch.setattr(dflash.ops, "rotary_embedding", lambda *args: None)
    original = torch.arange(48, dtype=torch.float32).view(2, 3, 2, 4)
    captures = []
    inner = [SimpleNamespace(
        kv_cache=torch.empty(0), impl=SimpleNamespace(
            do_kv_cache_update=lambda layer, k, v, cache, slots:
                captures.append((v.clone(), slots.clone())))) for _ in range(2)]
    module = SimpleNamespace(
        _num_attn_layers=2, _kv_size=8, _head_dim=4, _num_kv_heads=2,
        _project_context_kv=lambda *_: (original.clone(), original.clone()),
        _normalize_context_k=lambda x: x,
        _rope_cos_sin_cache=torch.zeros(8, 4), _rope_head_size=4,
        _rope_is_neox=True, _attn_layers=inner,
        layers=[SimpleNamespace(self_attn=SimpleNamespace(v_scale=value_scale))
                for _ in range(2)],
    )
    slots = [torch.tensor([0, 1, -1]), torch.tensor([7, -1, -1])]
    mapping = slots if per_layer else slots[0]
    dflash.DFlashQwen3Model.precompute_and_store_context_kv(
        module, torch.empty(3, 8), torch.arange(3), mapping
    )
    assert len(captures) == 2
    for index, (values, actual_slots) in enumerate(captures):
        expected = original[index] * (1 if value_scale is None else value_scale)
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(actual_slots, slots[index] if per_layer else slots[0])
