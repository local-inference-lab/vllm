# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia import kda as kimi_kda
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import tp_projection


class _UnquantizedOutputLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, tp_size: int = 2):
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.input_is_parallel = True
        self.input_size_per_partition = input_size
        self.output_size = output_size
        self.reduce_results = False
        self.tp_size = tp_size
        self.register_parameter("bias", None)
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size),
            requires_grad=False,
        )


class _CallerOutputAttention(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.output: torch.Tensor | None = None

    def should_use_caller_output(self, _hidden_states: torch.Tensor) -> bool:
        return self.enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if output is None:
            return hidden_states + 1
        self.output = output
        output.copy_(hidden_states + 1)
        return output


def test_full_width_projection_reduces_prefill_in_place(monkeypatch) -> None:
    output_parallel = torch.empty(4096, 4)
    reduce_in_place = Mock(side_effect=lambda value: value)
    functional_reduce = Mock(side_effect=AssertionError("allocating reduction used"))
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        reduce_in_place,
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        functional_reduce,
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 16)

    assert result is output_parallel
    reduce_in_place.assert_called_once_with(output_parallel)
    functional_reduce.assert_not_called()


@pytest.mark.parametrize("num_tokens", [1, 8, 1023])
def test_full_width_projection_preserves_decode_collective(
    monkeypatch,
    num_tokens: int,
) -> None:
    output_parallel = torch.empty(num_tokens, 4)
    reduced = torch.empty_like(output_parallel)
    functional_reduce = Mock(return_value=reduced)
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        functional_reduce,
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        Mock(side_effect=AssertionError("in-place reduction used for decode")),
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 16)

    assert result is reduced
    functional_reduce.assert_called_once_with(output_parallel)


def test_full_width_projection_is_identity_at_tp1(monkeypatch) -> None:
    output_parallel = torch.empty(4096, 4)
    functional_reduce = Mock(side_effect=AssertionError("collective used at TP1"))
    in_place_reduce = Mock(side_effect=AssertionError("collective used at TP1"))
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        functional_reduce,
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        in_place_reduce,
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 1)

    assert result is output_parallel
    functional_reduce.assert_not_called()
    in_place_reduce.assert_not_called()


def test_kda_projects_prefill_into_caller_storage() -> None:
    layer = object.__new__(kimi_kda.KimiK3DeltaAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=3, output_size=4)
    core_attn_out = torch.randn(1024, 3)
    output = torch.empty(1024, 4)
    expected = torch.mm(core_attn_out, layer.o_proj.weight.t())
    output_pointer = output.data_ptr()

    with torch.inference_mode():
        actual = layer._project_output_into(core_attn_out, output)

    assert actual.data_ptr() == output_pointer
    torch.testing.assert_close(actual, expected)


def test_kda_caller_output_selection_preserves_decode_path() -> None:
    layer = object.__new__(kimi_kda.KimiK3DeltaAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=3, output_size=4)

    assert not layer.should_use_caller_output(torch.empty(8, 4))
    assert layer.should_use_caller_output(torch.empty(1024, 4))
    assert not layer.should_use_caller_output(torch.empty(1024, 3))


def test_kda_caller_output_rejects_projection_input_alias() -> None:
    layer = object.__new__(kimi_kda.KimiK3DeltaAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=4, output_size=4)
    aliased = torch.empty(1024, 4)

    with pytest.raises(ValueError, match="must not alias"):
        layer._project_output_into(aliased, aliased)


@pytest.mark.parametrize(
    ("use_sequence_parallel", "attention_enabled", "expects_reuse"),
    [
        (False, True, True),
        (False, False, False),
        (True, True, False),
    ],
)
def test_decoder_selects_kda_caller_output(
    use_sequence_parallel: bool,
    attention_enabled: bool,
    expects_reuse: bool,
) -> None:
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = use_sequence_parallel
    layer.self_attn = _CallerOutputAttention(enabled=attention_enabled)
    hidden_states = torch.empty(1024, 4)

    output = layer._select_self_attn_output(hidden_states)

    assert (output is hidden_states) is expects_reuse


def test_decoder_passes_caller_output_to_kda() -> None:
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    attention = _CallerOutputAttention()
    layer.self_attn = attention
    hidden_states = torch.zeros(1024, 4)
    output_pointer = hidden_states.data_ptr()

    actual = layer._run_self_attn(
        torch.arange(1024),
        hidden_states,
        output=hidden_states,
    )

    assert actual.data_ptr() == output_pointer
    assert attention.output is hidden_states
    torch.testing.assert_close(actual, torch.ones_like(actual))
