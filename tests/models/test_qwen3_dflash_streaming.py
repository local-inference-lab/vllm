# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


class _Accumulator:
    input_width = 8
    max_tokens = 2048

    def __init__(self, output: torch.Tensor):
        self.output = output
        self.tokens = 0
        self.inputs: list[torch.Tensor] = []

    def begin(self, tokens: int) -> None:
        self.tokens = tokens
        self.inputs.clear()

    def append(self, source: torch.Tensor) -> None:
        self.inputs.append(source.clone())

    def finish(self) -> torch.Tensor:
        self.output[: self.tokens].fill_(7)
        return self.output[: self.tokens]


def test_dflash_streams_auxiliary_states_and_claims_result_once() -> None:
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.use_aux_hidden_state = True
    model._target_hidden_size = 4
    model._streamed_aux_layer_ids = (2, 4)
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    scratch = torch.empty(2048, 4)
    accumulator = _Accumulator(scratch)
    model.bind_auxiliary_stream(accumulator, scratch)

    first = torch.ones(1024, 4)
    second = torch.full((1024, 4), 2.0)
    residual = torch.full((1024, 4), 3.0)
    assert model.can_stream_auxiliary_states((2, 4), first)

    model.begin_auxiliary_stream(first)
    model.accumulate_auxiliary_state(first, None)
    model.accumulate_auxiliary_state(second, residual)
    result = model.finish_auxiliary_stream()

    torch.testing.assert_close(accumulator.inputs[0], first)
    torch.testing.assert_close(accumulator.inputs[1], second + residual)
    assert model.is_streamed_context_states([result])
    assert not model.is_streamed_context_states([result])
