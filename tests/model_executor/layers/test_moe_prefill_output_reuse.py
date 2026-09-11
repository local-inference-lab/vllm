# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE


class _OutputReusingExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe_config = SimpleNamespace(skip_final_all_reduce=True)
        self.routed_output_transform = None
        self.output_buffer_pointer: int | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        output_buffer: torch.Tensor | None,
    ) -> torch.Tensor:
        assert router_logits.data_ptr() == hidden_states.data_ptr()
        assert output_buffer is not None
        self.output_buffer_pointer = output_buffer.data_ptr()
        output_buffer.add_(1)
        return output_buffer


def _make_moe() -> DeepseekV2MoE:
    moe = object.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.shared_experts = nn.Identity()
    moe.experts = _OutputReusingExperts()
    return moe


def test_large_inference_prefill_reuses_consumed_input() -> None:
    moe = _make_moe()
    hidden_states = torch.arange(4096 * 4).view(4096, 4).float()
    expected = hidden_states + 1
    input_pointer = hidden_states.data_ptr()

    with torch.inference_mode():
        assert moe.can_reuse_input_as_output(hidden_states)
        actual = moe(hidden_states, output_buffer=hidden_states)

    assert actual.data_ptr() == input_pointer
    assert moe.experts.output_buffer_pointer == input_pointer
    torch.testing.assert_close(actual, expected)


def test_output_reuse_excludes_decode_and_grad_paths() -> None:
    moe = _make_moe()

    with torch.inference_mode():
        assert not moe.can_reuse_input_as_output(torch.empty(16, 4))
        assert not moe.can_reuse_input_as_output(torch.empty(4, 1024).t())
    assert not moe.can_reuse_input_as_output(torch.empty(1024, 4))


def test_expert_sum_writes_into_distinct_output_storage() -> None:
    shared_output = torch.arange(32).view(8, 4).float()
    fused_output = torch.full_like(shared_output, 3)
    output_buffer = torch.empty_like(shared_output)
    output_pointer = output_buffer.data_ptr()

    with torch.inference_mode():
        actual = MoERunner._combine_expert_outputs(
            shared_output, fused_output, output_buffer
        )

    assert actual.data_ptr() == output_pointer
    torch.testing.assert_close(actual, shared_output + fused_output)


def test_expert_sum_rejects_output_alias() -> None:
    shared_output = torch.ones(8, 4)
    fused_output = torch.ones(8, 4)

    with torch.inference_mode(), pytest.raises(ValueError, match="must not alias"):
        MoERunner._combine_expert_outputs(shared_output, fused_output, shared_output)
