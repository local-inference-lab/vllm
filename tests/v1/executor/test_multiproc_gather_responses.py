# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor's gather of every rank's reply does not wait in rank order.

A rank that never replies (for example, stuck in a collective another rank
failed out of) must not hide a failure another rank has already reported.
"""

import time

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc, _gather_responses

SUCCESS = WorkerProc.ResponseStatus.SUCCESS
FAILURE = WorkerProc.ResponseStatus.FAILURE


class _Queue:
    """Replies once ``ready_after`` dequeue calls have passed; never if None."""

    def __init__(self, reply=None, ready_after: int = 0) -> None:
        self.reply = reply
        self.ready_after = ready_after
        self.calls = 0

    def dequeue(self, timeout=None):
        self.calls += 1
        if self.reply is not None and self.calls > self.ready_after:
            reply, self.reply = self.reply, None
            return reply
        if timeout:
            time.sleep(min(timeout, 0.01))
        raise TimeoutError


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


def test_a_single_reply_names_its_rank():
    with pytest.raises(RuntimeError, match="Worker 3 failed"):
        _gather_responses([_Queue((FAILURE, "boom"))], (3,), None, "m")
    assert _gather_responses([_Queue((SUCCESS, "ok"))], (3,), None, "m") == ["ok"]
