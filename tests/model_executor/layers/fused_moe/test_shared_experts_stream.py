# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_module
import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


class _PartialOutputTransform(torch.nn.Module):
    output_is_tp_partial = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _BufferedPartialOutputTransform(torch.nn.Module):
    output_is_tp_partial = True

    def can_write_output(
        self, hidden_states: torch.Tensor, output: torch.Tensor
    ) -> bool:
        return hidden_states.shape == output.shape

    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert output is not None
        output.copy_(hidden_states)
        return output


class _ResidualPartialOutputTransform(torch.nn.Module):
    output_is_tp_partial = True

    def can_accumulate_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> bool:
        return hidden_states.shape == residual.shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert residual is not None
        residual.add_(hidden_states)
        return residual


class _InputReusingSharedLayer(torch.nn.Module):
    def should_use_caller_output(self, hidden_states: torch.Tensor) -> bool:
        return hidden_states.shape[0] >= 4

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        output.copy_(hidden_states + 1)
        return output


def test_aux_stream_output_lifetime_extends_to_consumer(monkeypatch) -> None:
    shared_experts = object.__new__(SharedExperts)
    aux_stream = Mock()
    consumer_stream = Mock()
    output = Mock()
    shared_experts_input = Mock()
    shared_experts._stream = aux_stream
    shared_experts._layer = Mock(return_value=output)

    @contextmanager
    def use_stream(stream):
        assert stream is aux_stream
        yield

    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(shared_module, "current_stream", lambda: consumer_stream)

    result = shared_experts._run_in_aux_stream(shared_experts_input)

    assert result is output
    shared_experts._layer.assert_called_once_with(shared_experts_input)
    consumer_stream.wait_stream.assert_called_once_with(aux_stream)
    output.record_stream.assert_called_once_with(consumer_stream)


def test_shared_expert_reuses_input_only_for_synchronous_execution() -> None:
    shared_experts = object.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts.enable_dbo = False
    shared_experts._output = [None, None]
    shared_experts._layer = _InputReusingSharedLayer()
    shared_experts._determine_shared_experts_order = MethodType(
        lambda self, hidden_states: SharedExpertsOrder.NO_OVERLAP,
        shared_experts,
    )

    shared_input = torch.arange(16).view(4, 4).float()
    expected = shared_input + 1
    input_ptr = shared_input.data_ptr()

    assert shared_experts.can_reuse_input(shared_input)
    shared_experts(
        shared_input,
        SharedExpertsOrder.NO_OVERLAP,
        reuse_input=True,
    )
    output = shared_experts.output

    assert output.data_ptr() == input_ptr
    torch.testing.assert_close(output, expected)


def test_tp_partial_output_transform_defers_shared_reduce(monkeypatch) -> None:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = _PartialOutputTransform()
    runner.routed_input_transform = None
    runner.routed_scaling_factor = 1.0
    runner.router = None
    runner.layer_name = "test"
    runner.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=1,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )

    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    reduced_inputs = []

    def all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(hidden_states.clone())
        return hidden_states * 2

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    hidden_states = torch.zeros_like(shared_output)
    actual = runner.forward(
        hidden_states,
        router_logits=torch.empty(2, 1),
        shared_experts_input=hidden_states,
    )

    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], shared_output + fused_output)
    torch.testing.assert_close(actual, (shared_output + fused_output) * 2)


def test_tp_partial_output_transform_reuses_dead_input(monkeypatch) -> None:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = _BufferedPartialOutputTransform()
    runner.routed_input_transform = None
    runner.routed_scaling_factor = 1.0
    runner.router = None
    runner.layer_name = "test"
    runner.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=1,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )

    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    functional_reduce = Mock()
    reduced_ptrs: list[int] = []

    def all_reduce_in_place(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_ptrs.append(hidden_states.data_ptr())
        hidden_states.mul_(2)
        return hidden_states

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce",
        functional_reduce,
    )
    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce_in_place",
        all_reduce_in_place,
    )

    dead_input = torch.zeros_like(shared_output)
    dead_input_ptr = dead_input.data_ptr()
    actual = runner.forward(
        torch.zeros_like(shared_output),
        router_logits=torch.empty(2, 1),
        shared_experts_input=dead_input,
    )

    assert actual.data_ptr() == dead_input_ptr
    assert reduced_ptrs == [dead_input_ptr]
    functional_reduce.assert_not_called()
    torch.testing.assert_close(actual, (shared_output + fused_output) * 2)


def test_tp_partial_output_transform_accumulates_into_reused_shared_input(
    monkeypatch,
) -> None:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = _ResidualPartialOutputTransform()
    runner.routed_input_transform = None
    runner.routed_scaling_factor = 1.0
    runner.router = None
    runner.layer_name = "test"
    runner.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=1,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )
    runner._shared_experts = SimpleNamespace(can_reuse_input=lambda value: True)

    fused_output = torch.full((2, 4), 3.0)

    def reuse_entry(*args):
        shared_input = args[2]
        shared_input.fill_(2)
        return fused_output

    runner._shared_input_reuse_entry = Mock(side_effect=reuse_entry)
    runner._forward_entry = Mock()
    reduced_ptrs: list[int] = []

    def all_reduce_in_place(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_ptrs.append(hidden_states.data_ptr())
        hidden_states.mul_(2)
        return hidden_states

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce_in_place",
        all_reduce_in_place,
    )

    shared_input = torch.zeros_like(fused_output)
    input_ptr = shared_input.data_ptr()
    actual = runner.forward(
        torch.zeros_like(fused_output),
        router_logits=torch.empty(2, 1),
        shared_experts_input=shared_input,
    )

    runner._forward_entry.assert_not_called()
    assert actual.data_ptr() == input_ptr
    assert reduced_ptrs == [input_ptr]
    torch.testing.assert_close(actual, torch.full_like(actual, 10.0))
