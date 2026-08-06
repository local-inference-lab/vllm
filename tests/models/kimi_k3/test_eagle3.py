# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models import kimi_linear
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.kimi_linear import (
    KimiK3ForConditionalGeneration,
    KimiLinearModel,
)


def test_kimi_k3_advertises_eagle3_support():
    assert supports_eagle3(KimiK3ForConditionalGeneration)


def test_kimi_k3_exposes_language_model_for_speculative_decoding():
    target = object.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    language_model = torch.nn.Module()
    object.__setattr__(target, "language_model", language_model)

    assert target.get_language_model() is language_model


def test_kimi_k3_uses_shared_eagle3_layer_configuration():
    target = object.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    object.__setattr__(model, "layers", [None] * 93)
    language_model = SimpleNamespace(model=model)
    object.__setattr__(target, "language_model", language_model)

    target.set_aux_hidden_state_layers((8, 24, 52, 68, 84))

    assert model.aux_hidden_state_layers == (8, 24, 52, 68, 84)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 46, 90)


def test_kimi_linear_extracts_attn_res_aux_hidden_states(monkeypatch):
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    final_hidden_states = torch.tensor([[9.0, 10.0]])
    normalized_hidden_states = torch.tensor([[11.0, 12.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    object.__setattr__(model, "norm", Mock(return_value=normalized_hidden_states))
    monkeypatch.setattr(
        kimi_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    final_attn_res = Mock(return_value=final_hidden_states)
    monkeypatch.setattr(kimi_linear, "attn_res", final_attn_res)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, normalized_hidden_states)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], prefix_sum + layer_hidden_states)
    assert final_attn_res.call_args.args[2] is block_residual
