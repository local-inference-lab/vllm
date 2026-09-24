# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import vllm.envs as envs
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


def _make_executor_for_shutdown(
    events: list[str], rpc_error: Exception | None = None
) -> tuple[MultiprocExecutor, Mock]:
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.shutting_down = False
    executor._workers_initialized = True

    death_writer = SimpleNamespace(close=lambda: events.append("close_writer"))
    worker = SimpleNamespace(
        death_writer=death_writer,
        proc=object(),
        worker_response_mq=None,
    )
    executor.workers = [worker]
    executor.rpc_broadcast_mq = SimpleNamespace(
        shutdown=lambda: events.append("close_rpc_queue")
    )
    executor.response_mqs = [
        SimpleNamespace(shutdown=lambda: events.append("close_response_queue"))
    ]
    executor.futures_queue = []

    def collective_rpc(*args, **kwargs):
        events.append("worker_shutdown")
        if rpc_error is not None:
            raise rpc_error

    collective_rpc_mock = Mock(side_effect=collective_rpc)
    executor.collective_rpc = collective_rpc_mock
    executor._ensure_worker_termination = lambda procs: events.append(
        "terminate_workers"
    )
    return executor, collective_rpc_mock


def test_shutdown_stops_workers_before_closing_death_writers() -> None:
    events: list[str] = []
    executor, collective_rpc = _make_executor_for_shutdown(events)

    executor.shutdown()

    collective_rpc.assert_called_once_with(
        "shutdown", timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
    )
    assert events[:3] == [
        "worker_shutdown",
        "close_writer",
        "terminate_workers",
    ]

    executor.shutdown()
    collective_rpc.assert_called_once()


def test_shutdown_rpc_failure_still_forces_worker_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    executor, collective_rpc = _make_executor_for_shutdown(
        events, rpc_error=TimeoutError("worker shutdown timed out")
    )

    executor.shutdown()

    collective_rpc.assert_called_once()
    assert events[:3] == [
        "worker_shutdown",
        "close_writer",
        "terminate_workers",
    ]
    assert "failed to shut down workers gracefully" in caplog.text


@pytest.mark.parametrize(
    ("workers_initialized", "rpc_queue_available", "response_queues_available"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_shutdown_skips_worker_rpc_until_queues_are_ready(
    workers_initialized: bool,
    rpc_queue_available: bool,
    response_queues_available: bool,
) -> None:
    events: list[str] = []
    executor, collective_rpc = _make_executor_for_shutdown(events)
    executor._workers_initialized = workers_initialized
    if not rpc_queue_available:
        executor.rpc_broadcast_mq = None
    if not response_queues_available:
        executor.response_mqs = None

    executor.shutdown()

    collective_rpc.assert_not_called()
    assert events[:2] == ["close_writer", "terminate_workers"]
