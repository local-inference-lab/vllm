# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.model import KimiMLP


class _GateUp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor):
        output = torch.cat((x, x), dim=-1)
        self.output_ref = weakref.ref(output)
        return output, None


class _Activation(nn.Module):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        return gate_up.chunk(2, dim=-1)[0].clone()


class _Down(nn.Module):
    def __init__(self, gate_up: _GateUp) -> None:
        super().__init__()
        self.gate_up = gate_up

    def forward(self, x: torch.Tensor):
        assert self.gate_up.output_ref is not None
        assert self.gate_up.output_ref() is None
        return x, None


class _DownInto(nn.Module):
    def __init__(self, hidden_size: int, tp_size: int = 1) -> None:
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.input_is_parallel = True
        self.bias = None
        self.weight = nn.Parameter(torch.eye(hidden_size))
        self.output_size = hidden_size
        self.reduce_results = tp_size > 1
        self.tp_size = tp_size


def test_mlp_releases_gate_up_before_down_projection() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    gate_up = _GateUp()
    mlp.gate_up_proj = gate_up
    mlp.act_fn = _Activation()
    mlp.down_proj = _Down(gate_up)
    mlp.shard_sequence_parallel = False

    input_tensor = torch.randn(2, 4)
    torch.testing.assert_close(mlp(input_tensor), input_tensor)


def test_mlp_reuses_caller_output_for_unquantized_down_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    gate_up = _GateUp()
    mlp.gate_up_proj = gate_up
    mlp.act_fn = _Activation()
    mlp.down_proj = _DownInto(hidden_size=4, tp_size=2)
    mlp.shard_sequence_parallel = False

    reduced_ptrs: list[int] = []

    def _all_reduce_in_place(output: torch.Tensor) -> torch.Tensor:
        reduced_ptrs.append(output.data_ptr())
        output.mul_(2)
        return output

    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.model.tensor_model_parallel_all_reduce_in_place",
        _all_reduce_in_place,
    )

    input_tensor = torch.randn(2, 4)
    expected = input_tensor * 2
    input_ptr = input_tensor.data_ptr()
    with torch.inference_mode():
        output = mlp(input_tensor, output=input_tensor)

    assert output.data_ptr() == input_ptr
    assert reduced_ptrs == [input_ptr]
    torch.testing.assert_close(output, expected)
    assert gate_up.output_ref is not None
    assert gate_up.output_ref() is None


def test_mlp_rejects_activation_output_as_caller_output() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    mlp.down_proj = _DownInto(hidden_size=4)

    activation = torch.randn(2, 4)
    with pytest.raises(ValueError, match="must not alias"):
        mlp._down_proj_into(activation, activation)


def test_mlp_preserves_functional_decode_collective() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    mlp.down_proj = _DownInto(hidden_size=4, tp_size=2)

    assert not mlp.should_use_caller_output(torch.empty(8, 4))
    assert mlp.should_use_caller_output(torch.empty(1024, 4))
