# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor's gather of every rank's reply does not wait in rank order.

A rank that never replies (for example, stuck in a collective another rank
failed out of) must not hide a failure another rank has already reported.
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


def test_a_single_reply_names_its_rank():
    # One reply is one blocking read (safe for any size), so no readiness check.
    boom = _Queue((FAILURE, "boom"), ready_after=-1)
    with pytest.raises(RuntimeError, match="Worker 3 failed"):
        _gather_responses([boom], (3,), None, "m")
    ok = _Queue((SUCCESS, "ok"), ready_after=-1)
    assert _gather_responses([ok], (3,), None, "m") == ["ok"]
