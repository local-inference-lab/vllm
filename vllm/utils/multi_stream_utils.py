# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

import torch

_vllm_cudagraph_capture_depth: ContextVar[int] = ContextVar(
    "vllm_cudagraph_capture_depth", default=0
)


@contextmanager
def vllm_cudagraph_capture_scope() -> Iterator[None]:
    """Mark Python execution as belonging to a vLLM CUDA graph capture.

    CUDA reports capture status per current stream. Python custom ops can run
    on auxiliary streams joined to a multi-stream graph, where querying only
    the current stream is insufficient to establish event-handle lifetime.
    """
    token = _vllm_cudagraph_capture_depth.set(_vllm_cudagraph_capture_depth.get() + 1)
    try:
        yield
    finally:
        _vllm_cudagraph_capture_depth.reset(token)


def is_vllm_cudagraph_capture_active() -> bool:
    return _vllm_cudagraph_capture_depth.get() > 0


class AuxStreamType(Enum):
    Attention = 1


class EventType(Enum):
    Main = 0
    Attention = 1


class CUDAGraphCaptureEventPool:
    """Keep CUDA event generations private to each graph or eager call.

    Reusing one event handle across independently captured graphs is unsafe
    when those graphs can be replayed at different shapes. A replay may record
    a new generation while another graph still has waits bound to the same
    handle. Some custom ops also execute eagerly between CUDA graph segments,
    where capture-state detection is false even though adjacent graph shapes
    can still overlap event generations.

    Every real capture gets a retained private set embedded only in that graph.
    Eager graph-break callers request a fresh set for every Python invocation;
    the wrappers stay alive through enqueue and are then released. CUDA event
    destruction is asynchronous for pending work, which is also the lifetime
    pattern used by :meth:`torch.cuda.Stream.wait_stream`. Event handles are
    never recycled into another graph artifact.
    """

    def __init__(self, num_events: int) -> None:
        if num_events < 1:
            raise ValueError("num_events must be at least one")
        self.num_events = num_events
        self.default_events = [torch.cuda.Event() for _ in range(num_events)]
        self._captured_event_sets: list[list[torch.cuda.Event]] = []

    @contextmanager
    def lease(self, *, private_eager: bool = False) -> Iterator[list[torch.cuda.Event]]:
        if (
            is_vllm_cudagraph_capture_active()
            or torch.cuda.is_current_stream_capturing()
        ):
            events = [torch.cuda.Event() for _ in range(self.num_events)]
            # CUDA graphs retain the event handles, and this list keeps the
            # wrappers alive for the same lifetime as the owning module.
            self._captured_event_sets.append(events)
            yield events
            return

        if not private_eager:
            yield self.default_events
            return

        # Keep these wrappers alive until the caller has enqueued every record
        # and wait. cudaEventDestroy then defers resource release until pending
        # device work completes, so no Python-side retention is required.
        events = [torch.cuda.Event() for _ in range(self.num_events)]
        yield events


def maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: torch.cuda.Stream | None = None,
) -> tuple[Any, Any]:
    """Run two functions potentially in parallel on separate CUDA streams.

    When aux_stream is provided, fn0 runs on the current (default) stream and
    fn1 runs on aux_stream, synchronized via CUDA events.  When aux_stream is
    None, both functions execute sequentially on the current stream.

    This design follows TensorRT-LLM's maybe_execute_in_parallel pattern
    (tensorrt_llm/_torch/modules/multi_stream_utils.py).

    Args:
        fn0: Callable for the default stream.
        fn1: Callable for the auxiliary stream.
        event0: CUDA event recorded before fn0 so aux_stream can wait.
        event1: CUDA event recorded after fn1 so default stream can wait.
        aux_stream: The second CUDA stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        Tuple of (fn0_result, fn1_result).
    """
    if aux_stream is not None:
        event0.record()
        result0 = fn0()
        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


def execute_in_parallel(
    default_fn: Callable[[], Any],
    aux_fns: list[Callable[[], Any] | None],
    start_event: torch.cuda.Event,
    done_events: list[torch.cuda.Event],
    aux_streams: list[torch.cuda.Stream] | None = None,
    enable: bool = False,
    *,
    enqueue_default_first: bool = False,
) -> tuple[Any, list[Any]]:
    """Run default_fn on the current stream and aux_fns concurrently on
    aux_streams.

    Generalizes maybe_execute_in_parallel to N aux callables. Slots where
    aux_fns[i] is None are skipped (no stream switch, no event record); their
    corresponding entry in the returned aux_results list is None.

    start_event fans out from the current stream to every launched aux stream;
    done_events[i] is recorded after aux_fns[i] so the current stream joins
    before returning. Falls back to sequential execution on the current stream
    when aux_streams is None or enable is False; in that case default_fn runs
    first, then aux_fns in order.

    Args:
        default_fn: Callable for the default (current) stream.
        aux_fns: Per-aux callables; entries may be None to skip.
        start_event: CUDA event recorded on the current stream before
            default_fn so each launched aux stream can wait on it.
        done_events: One CUDA event per aux slot, recorded after the
            corresponding aux_fn. Length must match aux_fns.
        aux_streams: Per-aux CUDA streams. Length must match aux_fns.
            Multi-stream is disabled when None.
        enable: Opt-in switch for the multi-stream path. Defaults to False,
            so callers that pass aux_streams must also pass enable=True
            (typically gated by an env var) to actually overlap. When False,
            execution falls back to sequential on the current stream.
        enqueue_default_first: Enqueue ``default_fn`` before walking the
            auxiliary callables. The CUDA dependency graph is unchanged: all
            branches still wait on ``start_event`` and the default stream joins
            every launched auxiliary branch before returning. This option is
            useful when an auxiliary callable performs many host-side launches;
            it prevents that launch loop from withholding independent default-
            stream work from the GPU.

    Returns:
        Tuple of (default_result, aux_results) where aux_results[i] is the
        result of aux_fns[i] (or None when skipped).
    """
    aux_results: list[Any]
    if aux_streams is None or not enable:
        default_result = default_fn()
        aux_results = [fn() if fn is not None else None for fn in aux_fns]
        return default_result, aux_results

    assert len(aux_fns) == len(aux_streams) == len(done_events), (
        "aux_fns, aux_streams, and done_events must be the same length"
    )

    aux_results = [None] * len(aux_fns)
    pending: list[torch.cuda.Event] = []

    start_event.record()
    default_result = default_fn() if enqueue_default_first else None
    for i, fn in enumerate(aux_fns):
        if fn is None:
            continue
        with torch.cuda.stream(aux_streams[i]):
            start_event.wait()
            aux_results[i] = fn()
            done_events[i].record()
        pending.append(done_events[i])

    if not enqueue_default_first:
        default_result = default_fn()

    for ev in pending:
        ev.wait()

    return default_result, aux_results
