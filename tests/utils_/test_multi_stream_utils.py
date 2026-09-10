# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import pytest
import torch

from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


_ALLOCATION_BYTES = 16 * 1024 * 1024
_CONSUMER_DELAY_CYCLES = 500_000_000


def _assert_aux_output_lifetime(
    invoke: Callable[[torch.cuda.Stream], Any], extract: Callable[[Any], torch.Tensor]
) -> None:
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()

    consumer_stream = torch.cuda.current_stream()
    producer_stream = torch.cuda.Stream(priority=-1)
    copied = torch.empty(_ALLOCATION_BYTES, dtype=torch.uint8, device="cuda")

    # Resolve imports, events, and copy kernels before delaying the consumer.
    warmup = invoke(producer_stream)
    copied.copy_(extract(warmup))
    torch.cuda._sleep(1)
    torch.accelerator.synchronize()
    del warmup
    torch.accelerator.empty_cache()

    result = invoke(producer_stream)
    tensor = extract(result)
    consumer_done = torch.cuda.Event()

    torch.cuda._sleep(_CONSUMER_DELAY_CYCLES)
    copied.copy_(tensor)
    consumer_done.record()
    del tensor, result

    with torch.cuda.stream(producer_stream):
        replacement = torch.full(
            (_ALLOCATION_BYTES,), 93, dtype=torch.uint8, device="cuda"
        )

    producer_stream.synchronize()
    consumer_was_pending = not consumer_done.query()
    torch.accelerator.synchronize()
    assert consumer_was_pending, "The test must overwrite while the consumer is pending"
    assert torch.all(copied == 17)
    assert torch.all(replacement == 93)
    assert torch.cuda.current_stream() == consumer_stream


@requires_cuda
def test_maybe_execute_in_parallel_preserves_aux_output_lifetime() -> None:
    event0 = torch.cuda.Event()
    event1 = torch.cuda.Event()

    def invoke(producer_stream: torch.cuda.Stream) -> tuple[None, torch.Tensor]:
        return maybe_execute_in_parallel(
            lambda: None,
            lambda: torch.full(
                (_ALLOCATION_BYTES,), 17, dtype=torch.uint8, device="cuda"
            ),
            event0,
            event1,
            producer_stream,
        )

    _assert_aux_output_lifetime(invoke, lambda result: result[1])


@requires_cuda
def test_execute_in_parallel_preserves_nested_aux_output_lifetime() -> None:
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    def invoke(producer_stream: torch.cuda.Stream) -> tuple[None, list[Any]]:
        return execute_in_parallel(
            lambda: None,
            [
                lambda: {
                    "output": (
                        torch.full(
                            (_ALLOCATION_BYTES,),
                            17,
                            dtype=torch.uint8,
                            device="cuda",
                        ),
                    )
                }
            ],
            start_event,
            [done_event],
            [producer_stream],
            enable=True,
        )

    _assert_aux_output_lifetime(invoke, lambda result: result[1][0]["output"][0])


@pytest.mark.parametrize("helper", ["pair", "fanout"])
def test_sequential_execution_preserves_results_and_order(helper: str) -> None:
    calls = []

    def call(value):
        calls.append(value)
        return value

    if helper == "pair":
        result = maybe_execute_in_parallel(lambda: call(1), lambda: call(2), None, None)
        assert result == (1, 2)
    else:
        result = execute_in_parallel(
            lambda: call(1), [lambda: call(2), None, lambda: call(3)], None, []
        )
        assert result == (1, [2, None, 3])
    assert calls == ([1, 2] if helper == "pair" else [1, 2, 3])


@requires_cuda
@pytest.mark.parametrize("helper", ["pair", "fanout"])
def test_parallel_cuda_graph_replay_preserves_outputs(helper: str) -> None:
    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    producer = torch.cuda.Stream()
    start, done = torch.cuda.Event(), torch.cuda.Event()

    def operation():
        if helper == "pair":
            main, aux = maybe_execute_in_parallel(
                lambda: source + 1, lambda: {"value": source * 2}, start, done, producer
            )
        else:
            main, aux_results = execute_in_parallel(
                lambda: source + 1,
                [lambda: {"value": source * 2}],
                start,
                [done],
                [producer],
                enable=True,
            )
            aux = aux_results[0]
        return main + aux["value"]

    operation()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    for value in (3, 17, -5):
        source.fill_(value)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, value * 3 + 1))
