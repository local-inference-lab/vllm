# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check shared-expert output ownership across asynchronous CUDA streams.

The shared-expert wrapper must return storage that remains valid until the
caller's queued consumer completes, even after Python releases the output and
the producer stream allocates another equally sized tensor. The test exercises
the wrapper interface without loading a language model.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def shared_wrapper(layer, monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0")
    config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    return SharedExperts(layer, config, False, lambda: False)


class ConstantBytes(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.full(
            (16 * 1024 * 1024,), 17, dtype=torch.uint8, device=hidden_states.device
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("early_async", [False, True])
def test_shared_output_survives_producer_reuse(monkeypatch, early_async):
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    wrapper = shared_wrapper(ConstantBytes(), monkeypatch)
    hidden = torch.ones((1, 16), device="cuda")
    copied = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def invoke():
        if early_async:
            assert wrapper.maybe_forward_async(hidden)
            wrapper.wait()
        else:
            wrapper.maybe_sync_shared_experts_stream(hidden)
            wrapper(hidden, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
        return wrapper.output

    warmup = invoke()
    copied.copy_(warmup)
    torch.cuda._sleep(1)
    torch.accelerator.synchronize()
    del warmup
    torch.accelerator.empty_cache()

    result = invoke()
    producer = wrapper._stream
    pointer = result.data_ptr()
    consumer_done = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    copied.copy_(result)
    consumer_done.record()
    del result
    with torch.cuda.stream(producer):
        replacement = torch.full_like(copied, 93)
    producer.synchronize()
    pending = not consumer_done.query()
    torch.accelerator.synchronize()
    print(
        {
            "producer_reused_output": replacement.data_ptr() == pointer,
            "consumer_was_pending": pending,
            "copied_values": copied.unique().tolist(),
        },
        flush=True,
    )
    assert pending, "The producer must allocate while the consumer is pending"
    assert torch.all(copied == 17), "Shared output was reused before its consumer"
    assert torch.all(replacement == 93)


class DoubleInput(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


@pytest.mark.parametrize(
    "donate,alias_shared", [(False, False), (True, False), (True, True)]
)
def test_tp_partial_output_transform_defers_shared_reduce(
    monkeypatch, donate, alias_shared
):
    runner = object.__new__(moe_runner.MoERunner)
    torch.nn.Module.__init__(runner)

    class Transform(torch.nn.Module):
        def can_write_output(self, hidden, output):
            return donate

        def forward(self, hidden, output=None):
            return hidden if output is None else output.copy_(hidden)

    transform = Transform()
    transform.output_is_tp_partial = True
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
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            output_dtype=torch.float32,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None), runner
    )
    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    shared_input = shared_output if alias_shared else torch.zeros_like(shared_output)
    expected = shared_output + fused_output
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    reduced_inputs = []

    def all_reduce(tensor):
        reduced_inputs.append(tensor.clone())
        return tensor * 2

    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", all_reduce)

    def all_reduce_in_place(tensor):
        assert donate and not alias_shared
        assert tensor is shared_input
        reduced_inputs.append(tensor.clone())
        return tensor.mul_(2)

    monkeypatch.setattr(
        moe_runner, "tensor_model_parallel_all_reduce_in_place", all_reduce_in_place
    )
    actual = runner.forward(
        torch.zeros_like(shared_output),
        router_logits=torch.empty(2, 1),
        shared_experts_input=shared_input,
    )
    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], expected)
    torch.testing.assert_close(actual, expected * 2)
    assert (actual.data_ptr() == shared_input.data_ptr()) == (
        donate and not alias_shared
    )
    torch.testing.assert_close(shared_output, torch.full_like(shared_output, 2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("early_async", [False, True])
def test_shared_output_cuda_graph_replay(monkeypatch, early_async):
    wrapper = shared_wrapper(DoubleInput(), monkeypatch)
    hidden = torch.ones((1, 1024), device="cuda")

    def invoke():
        if early_async:
            assert wrapper.maybe_forward_async(hidden)
            wrapper.wait()
        else:
            wrapper.maybe_sync_shared_experts_stream(hidden)
            wrapper(hidden, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
        return wrapper.output + 1

    invoke()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke()
    for value in (3, 17, -5):
        hidden.fill_(value)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, value * 2 + 1))
