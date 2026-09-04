# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker completes an in-flight asynchronous output before it answers an
RPC that is not part of a step.

The engine answers RPCs in order: the reply to an RPC issued between two
steps is read only after the output of the step before it. A speculator that
resolves its proposal lazily finishes that output during the next step, which
the engine does not issue while it waits, so the worker has to finish the
output itself before it handles the RPC."""

import queue
import threading
from types import SimpleNamespace

import cloudpickle
import pytest

from vllm.v1.executor import multiproc_executor as mpe
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker


class _DeferredOutput(AsyncModelRunnerOutput):
    """An output whose host copy is recorded by a later runner call."""

    def __init__(self, ready: threading.Event) -> None:
        self._ready = ready

    def get_output(self):
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError("the output was never completed")
        return "step-output"


class _Worker:
    def __init__(self, *, flushes: bool) -> None:
        self.calls: list[str] = []
        self._ready = threading.Event()
        if flushes:
            self.flush_deferred_output = self._flush

    def _flush(self) -> None:
        self.calls.append("flush")
        self._ready.set()

    def execute_model(self, scheduler_output):
        self.calls.append("execute_model")
        return None

    def sample_tokens(self, grammar_output):
        self.calls.append("sample_tokens")
        return _DeferredOutput(self._ready)

    def profile(self, is_start=True, profile_prefix=None):
        self.calls.append("profile")
        return None

    def release(self) -> None:
        self._ready.set()


class _EndOfScript(Exception):
    pass


class _Broadcast:
    def __init__(self, rpcs) -> None:
        self._rpcs = list(rpcs)

    def dequeue(self, indefinite: bool = False):
        if not self._rpcs:
            raise _EndOfScript
        return self._rpcs.pop(0)


class _Responses:
    def __init__(self) -> None:
        self.items: queue.Queue = queue.Queue()

    def enqueue(self, item) -> None:
        self.items.put(item)


def _worker_proc(worker: _Worker, rpcs) -> tuple[mpe.WorkerProc, _Responses]:
    proc = object.__new__(mpe.WorkerProc)
    proc.worker = worker
    proc.rank = 0
    proc.use_async_scheduling = True
    proc.async_output_queue = queue.Queue()
    proc.rpc_broadcast_mq = _Broadcast(rpcs)
    responses = _Responses()
    proc.worker_response_mq = responses
    return proc, responses


def _run(proc: mpe.WorkerProc) -> None:
    threading.Thread(
        target=proc.async_output_busy_loop, name="async-output", daemon=True
    ).start()
    with pytest.raises(_EndOfScript):
        proc.worker_busy_loop()


def _step_then(rpc):
    return [
        ("execute_model", ("scheduler-output",), {}, 0),
        ("sample_tokens", (None,), {}, 0),
        rpc,
    ]


def test_out_of_band_rpc_completes_the_in_flight_output_first():
    worker = _Worker(flushes=True)
    proc, responses = _worker_proc(
        worker, _step_then(("profile", (False, None), {}, None))
    )

    _run(proc)

    assert worker.calls == ["execute_model", "sample_tokens", "flush", "profile"]
    replies = [responses.items.get(timeout=5.0) for _ in range(3)]
    success = mpe.WorkerProc.ResponseStatus.SUCCESS
    assert replies == [(success, None), (success, "step-output"), (success, None)]


def test_without_the_flush_the_rpc_reply_waits_behind_the_incomplete_output():
    worker = _Worker(flushes=False)
    proc, responses = _worker_proc(
        worker, _step_then(("profile", (False, None), {}, None))
    )

    _run(proc)

    assert worker.calls == ["execute_model", "sample_tokens", "profile"]
    assert responses.items.get(timeout=5.0)[1] is None  # execute_model
    with pytest.raises(queue.Empty):
        responses.items.get(timeout=0.5)
    worker.release()
    assert responses.items.get(timeout=5.0)[1] == "step-output"
    assert responses.items.get(timeout=5.0)[1] is None  # profile


def test_step_rpcs_do_not_flush_and_pickled_callables_do():
    worker = _Worker(flushes=True)
    remote = cloudpickle.dumps(lambda w: w.calls.append("remote"))
    proc, _ = _worker_proc(worker, _step_then((remote, (), {}, None)))

    _run(proc)

    assert worker.calls == ["execute_model", "sample_tokens", "flush", "remote"]


def test_gpu_worker_flush_consumes_the_runner_proposal():
    worker = object.__new__(Worker)
    calls: list[str] = []
    worker.model_runner = SimpleNamespace(
        flush_deferred_draft=lambda: calls.append("flush")
    )

    worker.flush_deferred_output()
    assert calls == ["flush"]

    worker.model_runner = SimpleNamespace()
    worker.flush_deferred_output()
    assert calls == ["flush"]
