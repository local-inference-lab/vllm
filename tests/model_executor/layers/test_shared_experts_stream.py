# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check shared-expert output ownership across asynchronous CUDA streams.

The shared-expert wrapper must return storage that remains valid until the
caller's queued consumer completes, even after Python releases the output and
the producer stream allocates another equally sized tensor. The test exercises
the wrapper interface without loading a language model.
"""

from types import SimpleNamespace

import pytest
import torch

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
def test_shared_output_survives_producer_reuse(monkeypatch):
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    wrapper = shared_wrapper(ConstantBytes(), monkeypatch)
    hidden = torch.ones((1, 16), device="cuda")
    copied = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def invoke():
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shared_output_cuda_graph_replay(monkeypatch):
    wrapper = shared_wrapper(DoubleInput(), monkeypatch)
    hidden = torch.ones((1, 1024), device="cuda")

    def invoke():
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
