# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def test_aux_stream_respects_expert_kernel_capability(monkeypatch) -> None:
    stream = Mock()
    supports_aux_stream = Mock(side_effect=lambda num_tokens: num_tokens <= 8)
    monkeypatch.setattr(shared_module, "aux_stream", Mock(return_value=stream))
    monkeypatch.setattr(shared_module.envs, "VLLM_DISABLE_SHARED_EXPERTS_STREAM", False)
    monkeypatch.setattr(
        shared_module,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    shared_experts = SharedExperts(
        layer=Mock(),
        moe_config=moe_config,
        enable_dbo=False,
        mk_can_overlap_shared_experts=Mock(return_value=False),
        experts_support_aux_stream=supports_aux_stream,
    )

    assert (
        shared_experts._determine_shared_experts_order(torch.empty(8, 16))
        == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    assert (
        shared_experts._determine_shared_experts_order(torch.empty(16, 16))
        == SharedExpertsOrder.NO_OVERLAP
    )
    assert supports_aux_stream.call_args_list == [call(8), call(16)]


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


def test_aux_shared_experts_are_enqueued_before_resident_moe() -> None:
    runner = object.__new__(MoERunner)
    events: list[object] = []
    fused_out = object()
    shared_out = object()

    runner._maybe_apply_shared_experts = lambda _input, order: events.append(order)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=True),
        forward_monolithic=lambda **_kwargs: events.append("routed") or fused_out,
    )
    runner._shared_experts = SimpleNamespace(output=shared_out)

    result = runner._apply_quant_method(
        hidden_states=Mock(),
        router_logits=Mock(),
        shared_experts_input=Mock(),
    )

    assert events == [
        SharedExpertsOrder.NO_OVERLAP,
        SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        "routed",
    ]
    assert result == (shared_out, fused_out)
