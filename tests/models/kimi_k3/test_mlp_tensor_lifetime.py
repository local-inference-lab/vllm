# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.model import (
    KimiMLP,
    KimiPaddedRowParallelLinear,
    KimiRoutedOutputTransform,
)


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


def _make_rms_norm(hidden_size: int) -> RMSNorm:
    norm = object.__new__(RMSNorm)
    nn.Module.__init__(norm)
    norm.hidden_size = hidden_size
    norm.variance_epsilon = 1e-5
    norm.variance_size_override = None
    norm.has_weight = True
    norm.pass_weight = True
    norm.pass_weight_add = True
    norm.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
    return norm


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


def test_mlp_caller_output_selection_uses_prefill_threshold() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    mlp.down_proj = _DownInto(hidden_size=4, tp_size=2)
    mlp.shard_sequence_parallel = False

    assert not mlp.should_use_caller_output(torch.empty(8, 4))
    assert mlp.should_use_caller_output(torch.empty(1024, 4))


def test_mlp_caller_output_selection_rejects_sequence_parallel() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    mlp.down_proj = _DownInto(hidden_size=4, tp_size=2)
    mlp.shard_sequence_parallel = True

    assert not mlp.should_use_caller_output(torch.empty(1024, 4))


def test_routed_output_transform_reuses_full_width_prefill_input() -> None:
    projection = object.__new__(KimiPaddedRowParallelLinear)
    nn.Module.__init__(projection)
    projection.input_pad = 0
    projection.input_is_parallel = False
    projection.tp_size = 2
    projection.tp_rank = 0
    projection.output_size = 4
    projection.reduce_results = False
    projection.quant_method = UnquantizedLinearMethod()
    projection.register_parameter("bias", None)
    projection.weight = nn.Parameter(
        torch.arange(12).view(4, 3).float(), requires_grad=False
    )

    transform = KimiRoutedOutputTransform(None, projection, layer_idx=0)
    hidden_states = torch.arange(1024 * 6).view(1024, 6).float()
    output = torch.empty(1024, 4)
    output_ptr = output.data_ptr()
    expected = torch.mm(hidden_states[:, :3], projection.weight.t())

    assert transform.can_write_output(hidden_states, output)
    with torch.inference_mode():
        actual = transform(hidden_states, output=output)

    assert actual.data_ptr() == output_ptr
    torch.testing.assert_close(actual, expected)


def test_routed_output_transform_keeps_decode_on_allocating_path() -> None:
    projection = object.__new__(KimiPaddedRowParallelLinear)
    nn.Module.__init__(projection)
    projection.input_pad = 0
    projection.input_is_parallel = False
    projection.tp_size = 2
    projection.tp_rank = 0
    projection.output_size = 4
    projection.reduce_results = False
    projection.quant_method = UnquantizedLinearMethod()
    projection.register_parameter("bias", None)
    projection.weight = nn.Parameter(
        torch.arange(12).view(4, 3).float(), requires_grad=False
    )

    transform = KimiRoutedOutputTransform(None, projection, layer_idx=0)
    hidden_states = torch.arange(8 * 6).view(8, 6).float()
    output = torch.empty(8, 4)

    assert not transform.can_write_output(hidden_states, output)


def test_routed_output_transform_normalizes_consumed_prefill_latent_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    norm = _make_rms_norm(4)
    transform = KimiRoutedOutputTransform(norm, nn.Identity(), layer_idx=0)
    hidden_states = torch.arange(16).view(4, 4).float()
    expected = hidden_states.clone()
    input_pointer = hidden_states.data_ptr()
    calls: list[tuple[int, int, float]] = []

    def _rms_norm(
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        calls.append((output.data_ptr(), input_tensor.data_ptr(), epsilon))
        output.copy_(input_tensor.clone() * weight)

    monkeypatch.setattr(
        KimiRoutedOutputTransform,
        "can_normalize_routed_latent_in_place",
        lambda _self, _hidden_states: True,
    )
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.model.ops.rms_norm",
        _rms_norm,
    )

    with torch.inference_mode():
        actual = transform.normalize_routed_latent(hidden_states)

    assert actual.data_ptr() == input_pointer
    assert calls == [(input_pointer, input_pointer, 1e-5)]
    torch.testing.assert_close(actual, expected)


def test_routed_output_transform_keeps_cpu_latent_on_allocating_norm_path() -> None:
    norm = _make_rms_norm(4)
    transform = KimiRoutedOutputTransform(norm, nn.Identity(), layer_idx=0)
    hidden_states = torch.arange(4096).view(1024, 4).float()

    with torch.inference_mode():
        assert not transform.can_normalize_routed_latent_in_place(hidden_states)


def test_routed_output_transform_accumulates_prefill_in_bounded_tiles() -> None:
    projection = object.__new__(KimiPaddedRowParallelLinear)
    nn.Module.__init__(projection)
    projection.input_pad = 0
    projection.input_is_parallel = False
    projection.tp_size = 2
    projection.tp_rank = 0
    projection.output_size = 2048
    projection.reduce_results = False
    projection.quant_method = UnquantizedLinearMethod()
    projection.register_parameter("bias", None)
    projection.weight = nn.Parameter(
        torch.arange(2048 * 3).view(2048, 3).float() / 1024,
        requires_grad=False,
    )

    transform = KimiRoutedOutputTransform(None, projection, layer_idx=0)
    hidden_states = torch.arange(1024 * 6).view(1024, 6).float() / 1024
    residual = torch.arange(1024 * 2048).view(1024, 2048).float() / 2048
    expected = residual + torch.mm(hidden_states[:, :3], projection.weight.t())
    residual_ptr = residual.data_ptr()

    assert transform.can_accumulate_residual(hidden_states, residual)
    with torch.inference_mode():
        actual = transform(hidden_states, residual=residual)

    assert actual.data_ptr() == residual_ptr
    torch.testing.assert_close(actual, expected)
