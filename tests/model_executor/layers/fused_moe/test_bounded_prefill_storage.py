# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.runner.moe_runner as runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


class _InputReusingLayer(torch.nn.Module):
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


class _ResidualTransform(torch.nn.Module):
    output_is_tp_partial = True

    def can_accumulate_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
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


class _OutputTransform(torch.nn.Module):
    output_is_tp_partial = True

    def can_write_output(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        return hidden_states.shape == output.shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        output.copy_(hidden_states)
        return output


def _make_runner(transform: torch.nn.Module) -> MoERunner:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = transform
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
        should_defer_moe_finalize=lambda _tokens: False,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: False),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )
    return runner


def test_shared_expert_reuses_input_only_for_synchronous_execution() -> None:
    shared_experts = object.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts.enable_dbo = False
    shared_experts._output = [None, None]
    shared_experts._layer = _InputReusingLayer()
    shared_experts._determine_shared_experts_order = MethodType(
        lambda self, hidden_states: SharedExpertsOrder.NO_OVERLAP,
        shared_experts,
    )

    shared_input = torch.arange(16).view(4, 4).float()
    expected = shared_input + 1
    input_pointer = shared_input.data_ptr()

    assert shared_experts.can_reuse_input(shared_input)
    shared_experts(
        shared_input,
        SharedExpertsOrder.NO_OVERLAP,
        reuse_input=True,
    )
    output = shared_experts.output

    assert output.data_ptr() == input_pointer
    torch.testing.assert_close(output, expected)


def test_reused_shared_input_bounds_latent_tail_storage(monkeypatch) -> None:
    runner = _make_runner(_ResidualTransform())
    runner._shared_experts = SimpleNamespace(can_reuse_input=lambda _value: True)
    fused_output = torch.full((2, 4), 3.0)

    def reuse_entry(*args):
        shared_input = args[2]
        shared_input.fill_(2)
        return fused_output

    runner._shared_input_reuse_entry = Mock(side_effect=reuse_entry)
    runner._forward_entry = Mock()
    functional_reduce = Mock()
    reduced_pointers: list[int] = []

    def all_reduce_in_place(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_pointers.append(hidden_states.data_ptr())
        hidden_states.mul_(2)
        return hidden_states

    monkeypatch.setattr(
        runner_module,
        "tensor_model_parallel_all_reduce",
        functional_reduce,
    )
    monkeypatch.setattr(
        runner_module,
        "tensor_model_parallel_all_reduce_in_place",
        all_reduce_in_place,
    )

    shared_input = torch.zeros_like(fused_output)
    shared_input_pointer = shared_input.data_ptr()
    fused_output_pointer = fused_output.data_ptr()
    actual = runner.forward(
        torch.zeros_like(fused_output),
        router_logits=torch.empty(2, 1),
        shared_experts_input=shared_input,
    )

    runner._forward_entry.assert_not_called()
    functional_reduce.assert_not_called()
    assert actual.data_ptr() == shared_input_pointer
    assert reduced_pointers == [fused_output_pointer, shared_input_pointer]
    torch.testing.assert_close(actual, torch.full_like(actual, 16.0))


def test_separate_shared_output_reuses_consumed_input(monkeypatch) -> None:
    runner = _make_runner(_OutputTransform())
    runner._shared_experts = SimpleNamespace(can_reuse_input=lambda _value: False)
    runner._shared_input_reuse_entry = Mock()
    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    reduced_pointers: list[int] = []

    def all_reduce_in_place(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_pointers.append(hidden_states.data_ptr())
        hidden_states.mul_(2)
        return hidden_states

    monkeypatch.setattr(
        runner_module,
        "tensor_model_parallel_all_reduce",
        lambda hidden_states: hidden_states.mul(2),
    )
    monkeypatch.setattr(
        runner_module,
        "tensor_model_parallel_all_reduce_in_place",
        all_reduce_in_place,
    )

    consumed_input = torch.zeros_like(fused_output)
    consumed_input_pointer = consumed_input.data_ptr()
    actual = runner.forward(
        torch.zeros_like(fused_output),
        router_logits=torch.empty(2, 1),
        shared_experts_input=consumed_input,
    )

    runner._shared_input_reuse_entry.assert_not_called()
    assert actual.data_ptr() == consumed_input_pointer
    assert reduced_pointers == [consumed_input_pointer]
    torch.testing.assert_close(actual, torch.full_like(actual, 16.0))


def test_shared_input_reuse_operator_declares_mutation() -> None:
    schema = str(torch.ops.vllm.moe_forward_shared_input_reuse.default._schema)

    assert "!)? shared_experts_input" in schema
