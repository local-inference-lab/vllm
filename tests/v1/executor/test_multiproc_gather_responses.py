# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor's gather of every rank's reply does not wait in rank order.

A rank that never replies (for example, stuck in a collective another rank
failed out of) must not hide a failure another rank has already reported,
including a rank that does not own the reply of a single-rank RPC.
"""

import threading
import time

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import WorkerProc, _gather_responses

SUCCESS = WorkerProc.ResponseStatus.SUCCESS
FAILURE = WorkerProc.ResponseStatus.FAILURE


class _Queue:
    """Has its reply ready after ``ready_after`` readiness checks; never if None.

    A dequeue before the reply is ready fails the test: the gather must only
    read a queue that ready() reported, or it can lose a large reply.
    """

    def __init__(self, reply=None, ready_after: int = 0) -> None:
        self.reply = reply
        self.ready_after = ready_after
        self.checks = 0

    def ready(self) -> bool:
        self.checks += 1
        return self.reply is not None and self.checks > self.ready_after

    def dequeue(self, timeout=None):
        assert self.reply is not None and self.checks > self.ready_after
        reply, self.reply = self.reply, None
        return reply

    def wait_for_message(self, timeout_ms: int) -> None:
        self.waits = getattr(self, "waits", 0) + 1


def test_a_later_rank_failure_surfaces_while_rank_zero_hangs():
    queues = [_Queue(), _Queue((FAILURE, "capture invalidated")), _Queue()]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Worker 1 failed.*capture invalidated"):
        _gather_responses(queues, range(3), None, "compile_or_warm_up_model")
    assert time.monotonic() - started < 1.0


def test_replies_arriving_out_of_order_come_back_in_rank_order():
    queues = [
        _Queue((SUCCESS, "r0"), ready_after=5),
        _Queue((SUCCESS, "r1")),
        _Queue((SUCCESS, "r2"), ready_after=2),
    ]
    assert _gather_responses(queues, range(3), None, "m") == ["r0", "r1", "r2"]


def test_the_deadline_still_applies():
    queues = [_Queue(), _Queue((SUCCESS, "r1"))]
    with pytest.raises(TimeoutError, match="RPC call to m timed out"):
        _gather_responses(queues, range(2), time.monotonic() + 0.2, "m")


def test_a_reply_larger_than_a_ring_chunk_is_not_lost():
    """Real queues: a reply over max_chunk_bytes goes out over the socket after
    its slot is marked, so reading it must wait for the payload."""
    pairs = []
    try:
        for _ in range(2):
            writer = MessageQueue(
                n_reader=1, n_local_reader=1, max_chunk_bytes=1024, max_chunks=2
            )
            reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
            writer.wait_until_ready()
            reader.wait_until_ready()
            pairs.append((writer, reader))
        big = b"x" * 200_000

        def send() -> None:
            time.sleep(0.05)
            pairs[1][0].enqueue((SUCCESS, big))
            time.sleep(0.05)
            pairs[0][0].enqueue((SUCCESS, "small"))

        sender = threading.Thread(target=send)
        sender.start()
        replies = _gather_responses(
            [reader for _, reader in pairs], range(2), time.monotonic() + 10, "m"
        )
        sender.join()
        assert replies == ["small", big]
    finally:
        for writer, reader in pairs:
            writer.shutdown()
            reader.shutdown()


def _queue_pair(max_chunk_bytes: int = 1024) -> tuple[MessageQueue, MessageQueue]:
    writer = MessageQueue(
        n_reader=1, n_local_reader=1, max_chunk_bytes=max_chunk_bytes, max_chunks=2
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()
    return writer, reader


def test_a_watched_rank_failure_ends_a_single_rank_wait():
    """TP1 fails out of execute_model while TP0, the output rank, waits in a
    collective TP1 left: the engine must not wait out the RPC timeout."""
    output = _Queue()
    watched = [(1, _Queue((FAILURE, "CUDA out of memory"), ready_after=3))]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Worker 1 failed.*out of memory"):
        _gather_responses([output], (0,), time.monotonic() + 300, "m", watched)
    assert time.monotonic() - started < 1.0
    assert output.waits >= 3


def test_a_single_rank_reply_is_read_while_watching():
    output = _Queue((SUCCESS, "r0"), ready_after=2)
    quiet = _Queue()
    assert _gather_responses([output], (0,), None, "m", [(1, quiet)]) == ["r0"]
    assert quiet.checks >= 2 and quiet.reply is None


def test_an_unexpected_success_from_a_watched_rank_is_dropped():
    stale = _Queue((SUCCESS, "late"))
    output = _Queue((SUCCESS, "r0"), ready_after=1)
    assert _gather_responses([output], (0,), None, "m", [(1, stale)]) == ["r0"]
    assert stale.reply is None


def test_the_deadline_applies_while_watching():
    with pytest.raises(TimeoutError, match="RPC call to m timed out"):
        _gather_responses(
            [_Queue()], (0,), time.monotonic() + 0.2, "m", [(1, _Queue())]
        )


def test_a_watched_failure_on_real_queues_wakes_the_parked_gather():
    """Real queues with the output rank idle: the gather parks on the output
    rank's notification in slices and still sees the other rank's failure."""
    pairs = [_queue_pair(), _queue_pair()]
    try:

        def fail() -> None:
            time.sleep(0.2)
            pairs[1][0].enqueue((FAILURE, "CUDA out of memory"))

        failer = threading.Thread(target=fail)
        started = time.monotonic()
        failer.start()
        with pytest.raises(RuntimeError, match="Worker 1 failed"):
            _gather_responses(
                [pairs[0][1]], (0,), time.monotonic() + 30, "m", [(1, pairs[1][1])]
            )
        failer.join()
        assert time.monotonic() - started < 2.0
    finally:
        for writer, reader in pairs:
            writer.shutdown()
            reader.shutdown()


def test_wait_for_message_does_not_consume():
    writer, reader = _queue_pair()
    try:
        started = time.monotonic()
        reader.wait_for_message(100)
        assert not reader.ready()
        assert time.monotonic() - started < 5.0
        writer.enqueue((SUCCESS, "r"))
        reader.wait_for_message(1000)
        assert reader.ready() and reader.ready()
        assert reader.dequeue(timeout=1) == (SUCCESS, "r")
        assert not reader.ready()
    finally:
        writer.shutdown()
        reader.shutdown()


class _Worker:
    def fail(self):
        raise RuntimeError("CUDA out of memory")

    def ok(self):
        return "done"


class _ResponseQueue:
    def __init__(self) -> None:
        self.sent: list = []

    def enqueue(self, obj) -> None:
        self.sent.append(obj)


def _worker_proc(rank: int) -> WorkerProc:
    proc = object.__new__(WorkerProc)
    proc.worker = _Worker()
    proc.rank = rank
    proc.use_async_scheduling = False
    proc.worker_response_mq = _ResponseQueue()
    return proc


def test_a_rank_that_does_not_own_the_reply_reports_its_failure():
    proc = _worker_proc(rank=1)
    proc._execute_worker_rpc(("fail", (), {}, 0))
    assert proc.worker_response_mq.sent == [(FAILURE, "CUDA out of memory")]


def test_a_rank_that_does_not_own_the_reply_stays_quiet_on_success():
    proc = _worker_proc(rank=1)
    proc._execute_worker_rpc(("ok", (), {}, 0))
    assert proc.worker_response_mq.sent == []
    owner = _worker_proc(rank=0)
    owner._execute_worker_rpc(("ok", (), {}, 0))
    assert owner.worker_response_mq.sent == [(SUCCESS, "done")]


def test_a_single_reply_names_its_rank():
    # One reply is one blocking read (safe for any size), so no readiness check.
    boom = _Queue((FAILURE, "boom"), ready_after=-1)
    with pytest.raises(RuntimeError, match="Worker 3 failed"):
        _gather_responses([boom], (3,), None, "m")
    ok = _Queue((SUCCESS, "ok"), ready_after=-1)
    assert _gather_responses([ok], (3,), None, "m") == ["ok"]
