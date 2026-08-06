# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts


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
