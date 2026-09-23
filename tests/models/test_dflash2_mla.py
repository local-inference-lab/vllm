# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia.dflash2_mla import (
    DFlash2KimiK3DecoderLayer,
    DFlash2KimiK3ForCausalLM,
    DFlash2KimiK3Model,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkForCausalLM,
    K3DSparkModel,
)

pytestmark = pytest.mark.cpu_test


def test_mla_dflash2_registry_and_head_contract():
    assert (
        ModelRegistry._try_load_model_cls("DFlash2KimiK3Model")
        is DFlash2KimiK3ForCausalLM
    )
    assert DFlash2KimiK3ForCausalLM.model_cls is DFlash2KimiK3Model
    assert DFlash2KimiK3Model.decoder_layer_cls is DFlash2KimiK3DecoderLayer
    assert not DFlash2KimiK3Model.has_markov_head
    assert K3DSparkModel.has_markov_head
    assert not DFlash2KimiK3ForCausalLM.has_own_embed_tokens
    assert not DFlash2KimiK3ForCausalLM.has_own_lm_head


def test_mla_dflash2_maps_context_and_norm_checkpoint_names(monkeypatch):
    names = [
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "layers.0.self_attn.q_a_proj.weight",
        "candidate_selector.predecessor_codebook",
    ]
    tensor = torch.ones(1)
    expected = [
        "context_proj.weight",
        "context_norm.weight",
        "final_norm.weight",
        *names[3:],
    ]

    def load(self, weights):
        weights = list(weights)
        assert [name for name, _ in weights] == expected
        assert all(weight is tensor for _, weight in weights)
        return set(expected)

    monkeypatch.setattr(K3DSparkForCausalLM, "load_weights", load)
    model = object.__new__(DFlash2KimiK3ForCausalLM)
    assert model.load_weights((name, tensor) for name in names) == set(expected)


def test_mla_dflash2_uses_distributed_topk_without_full_logit_gather():
    processor = Mock()
    head = object()
    states = torch.zeros(3, 8)
    model = SimpleNamespace(
        lm_head=head,
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=16)),
        candidate_logits_processor=processor,
    )
    result = DFlash2KimiK3ForCausalLM.compute_candidates(model, states)
    processor.get_top_k_tokens.assert_called_once_with(head, states, 16)
    assert result is processor.get_top_k_tokens.return_value


@pytest.mark.parametrize("index,expected", [(0, 4096), (3, 4096), (4, 0)])
def test_mla_dflash2_preserves_checkpoint_attention_windows(
    default_vllm_config, monkeypatch, index, expected
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_DSPARK_DRAFT_KV_WINDOW", "2048")
    config = SimpleNamespace(
        is_causal=False,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
        dflash_config={},
        sliding_window=4096,
    )
    assert DFlash2KimiK3DecoderLayer.get_attention_window(config, index) == expected


def test_mla_dflash2_rejects_unsupported_causal_layer():
    config = SimpleNamespace(is_causal=True, layer_types=None, dflash_config={})
    with pytest.raises(ValueError, match="requires non-causal"):
        DFlash2KimiK3DecoderLayer.get_attention_window(config, 0)


def test_candidate_selector_does_not_prepare_markov_argmax():
    model = SimpleNamespace(model=SimpleNamespace(markov_head=None))
    assert not K3DSparkForCausalLM.supports_local_draft_argmax(model)


@pytest.mark.parametrize("tp", [8, 9, 10, 12, 16])
def test_padded_draft_mlp_preserves_unquantized_projection(
    default_vllm_config, monkeypatch, tp
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    monkeypatch.setattr(kimi_model, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: tp)
    generator = torch.Generator().manual_seed(717)
    hidden, intermediate = 64, 160
    gate = torch.randn(intermediate, hidden, generator=generator) / hidden**0.5
    up = torch.randn(intermediate, hidden, generator=generator) / hidden**0.5
    down = torch.randn(hidden, intermediate, generator=generator) / intermediate**0.5
    x = torch.randn(3, hidden, generator=generator)
    functional = torch.nn.functional
    expected = functional.linear(
        functional.silu(functional.linear(x, gate)) * functional.linear(x, up), down
    )
    result = torch.zeros_like(expected)
    for rank in range(tp):
        monkeypatch.setattr(
            linear, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        monkeypatch.setattr(
            parameter, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        mlp = kimi_model.KimiMLP(
            hidden, intermediate, "silu", reduce_results=False, pad_intermediate=True
        )
        weight = mlp.gate_up_proj.weight
        weight.weight_loader(weight, gate, 0)
        weight.weight_loader(weight, up, 1)
        weight_down = mlp.down_proj.weight
        weight_down.weight_loader(weight_down, down)
        local_gate, local_up = functional.linear(x, weight).chunk(2, dim=-1)
        result += functional.linear(functional.silu(local_gate) * local_up, weight_down)
        local_width = weight_down.shape[1]
        valid = max(0, min(local_width, intermediate - rank * local_width))
        assert torch.count_nonzero(weight_down[:, valid:]) == 0
        assert torch.count_nonzero(weight[valid:local_width]) == 0
        assert torch.count_nonzero(weight[local_width + valid :]) == 0
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
