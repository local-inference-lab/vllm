# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_failed_broadcast_drains_every_reply_before_abort(world):
    from collections import deque
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    success, failure = (
        WorkerProc.ResponseStatus.SUCCESS,
        WorkerProc.ResponseStatus.FAILURE,
    )
    for failed_rank in range(world):
        queues = [
            MagicMock(
                dequeue=MagicMock(
                    side_effect=[
                        (failure, "admission")
                        if rank == failed_rank
                        else (success, "prepared"),
                        (success, "aborted"),
                    ]
                )
            )
            for rank in range(world)
        ]
        executor = SimpleNamespace(
            rpc_broadcast_mq=MagicMock(),
            is_failed=False,
            response_mqs=queues,
            futures_queue=deque(),
        )
        with pytest.raises(RuntimeError, match="admission"):
            MultiprocExecutor.collective_rpc(executor, "prepare", timeout=1)
        assert (
            MultiprocExecutor.collective_rpc(executor, "abort", timeout=1)
            == ["aborted"] * world
        )
        assert all(queue.dequeue.call_count == 2 for queue in queues)


@pytest.mark.parametrize("release_fails", [False, True])
def test_resource_release_precedes_worker_termination(release_fails):
    from unittest.mock import MagicMock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    events = []

    def release(method, timeout):
        assert method == "shutdown" and timeout == 60
        events.append("release")
        if release_fails:
            raise RuntimeError("release rejected")
        return [None, None, None]

    workers = [
        SimpleNamespace(
            death_writer=MagicMock(close=lambda: events.append("retire")),
            proc=MagicMock(),
            worker_response_mq=MagicMock(),
        )
        for _ in range(3)
    ]
    executor = SimpleNamespace(
        workers=workers,
        response_mqs=[MagicMock()] * 3,
        rpc_broadcast_mq=MagicMock(),
        is_failed=False,
        collective_rpc=release,
        _ensure_worker_termination=lambda _: events.append("terminated"),
    )
    if release_fails:
        with pytest.raises(RuntimeError, match="not acknowledged"):
            MultiprocExecutor.shutdown(executor)
    else:
        MultiprocExecutor.shutdown(executor)
        MultiprocExecutor.shutdown(executor)
    assert events == ["release", "retire", "retire", "retire", "terminated"]


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
