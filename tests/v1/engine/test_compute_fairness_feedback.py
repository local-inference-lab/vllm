# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import Mock, call, patch

import pytest

from vllm.v1.core.sched.compute_fairness import ComputeServiceClass
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test


def _scheduler_output(
    service_class: ComputeServiceClass | None,
    *,
    contended: bool = False,
    timing_enabled: bool | None = None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.compute_service_class = service_class
    output.compute_timing_enabled = (
        service_class is not None if timing_enabled is None else timing_enabled
    )
    output.compute_contention = contended
    return output


def test_disabled_output_does_not_install_execution_timing():
    completed_future: Future[None] = Future()
    completed_future.set_result(None)
    engine = object.__new__(EngineCore)
    engine.model_executor = Mock()
    engine.model_executor.execute_model.return_value = completed_future

    with patch(
        "vllm.v1.engine.core.time.perf_counter",
        side_effect=AssertionError("disabled timing must not read the clock"),
    ):
        future, timing = engine._execute_model(_scheduler_output(None))
        engine._record_compute_time(_scheduler_output(None), timing)

    assert future is completed_future
    assert timing is None


def test_enabled_execution_timer_returns_primitive_start_timestamp():
    execute_future: Future[None] = Future()
    execute_future.set_result(None)
    engine = object.__new__(EngineCore)
    engine.model_executor = Mock()
    engine.model_executor.execute_model.return_value = execute_future

    with patch("vllm.v1.engine.core.time.perf_counter", return_value=10.0):
        _, started_at = engine._execute_model(
            _scheduler_output("prefill", contended=True)
        )

    assert type(started_at) is float
    assert started_at == pytest.approx(10.0)


def test_queued_feedback_stays_paired_with_exact_batch():
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()
    engine._last_model_completion_time = None
    decode_output = _scheduler_output("decode", contended=True)
    prefill_output = _scheduler_output("prefill", contended=True)
    # Queued batches are consumed oldest-first; each carries its own timing and
    # class tag even when a later batch has already completed. The second
    # charge excludes its 90 ms queued behind the first batch.
    with patch("vllm.v1.engine.core.time.perf_counter", side_effect=[10.1, 10.3]):
        engine._record_compute_time(decode_output, 10.0)
        engine._record_compute_time(prefill_output, 10.01)

    assert engine.scheduler.record_compute_time.call_args_list == [
        call("decode", pytest.approx(0.1), contended=True, scheduled_tokens=0),
        call("prefill", pytest.approx(0.2), contended=True, scheduled_tokens=0),
    ]


def test_empty_transfer_step_does_not_record_compute():
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()

    engine._record_compute_time(_scheduler_output(None), None)

    engine.scheduler.record_compute_time.assert_not_called()


def test_transfer_step_advances_completion_boundary_without_compute_charge():
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()
    engine._last_model_completion_time = None

    transfer_output = _scheduler_output(None, timing_enabled=True)
    with patch("vllm.v1.engine.core.time.perf_counter", return_value=10.2):
        engine._record_compute_time(transfer_output, 10.0)

    engine.scheduler.record_compute_time.assert_not_called()
    assert engine._last_model_completion_time == pytest.approx(10.2)

    prefill_output = _scheduler_output("prefill", contended=True)
    with patch("vllm.v1.engine.core.time.perf_counter", return_value=10.5):
        engine._record_compute_time(prefill_output, 10.1)

    engine.scheduler.record_compute_time.assert_called_once_with(
        "prefill",
        pytest.approx(0.3),
        contended=True,
        scheduled_tokens=0,
    )


def _queued_engine(
    predecessor_tokens: int, *, successor_timed: bool, deferred: bool = False
):
    """Construct two executor results while retaining the real engine loop."""
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()
    engine.model_executor = Mock()
    engine._last_model_completion_time = None
    engine.batch_queue_size = 2
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.check_for_draft_tokens = False
    engine._should_throttle_prefills = Mock(return_value=False)
    engine.log_error_detail = lambda _: nullcontext()
    engine.capture_iteration_details = lambda _: nullcontext()
    engine._wait_for_boundary_checkpoint_copies = Mock()
    engine._process_aborts_queue = Mock()
    engine._attach_iteration_details = Mock()

    predecessor = _scheduler_output(None)
    predecessor.total_num_scheduled_tokens = predecessor_tokens
    prior_future: Future[Mock] = Future()
    prior_future.set_result(Mock())
    engine.batch_queue = deque(
        [(prior_future, predecessor, prior_future, None)], maxlen=2
    )
    successor = _scheduler_output(
        "prefill" if successor_timed else None, contended=successor_timed
    )
    successor.total_num_scheduled_tokens = 1
    successor.pending_structured_output_tokens = deferred
    completed: Future[Mock] = Future()
    completed.set_result(Mock())
    engine.scheduler.schedule.return_value = successor
    engine.model_executor.execute_model.return_value = completed
    engine.model_executor.sample_tokens.return_value = completed
    engine.scheduler.has_requests.side_effect = [True, False]
    return engine


@pytest.mark.parametrize("predecessor_tokens", [0, 1])
@pytest.mark.parametrize("deferred", [False, True])
def test_queued_contended_model_excludes_untimed_predecessor(
    predecessor_tokens, deferred
):
    engine = _queued_engine(predecessor_tokens, successor_timed=True, deferred=deferred)
    # The contended successor is dispatched at 10.1, before the transfer-only
    # or uncontended predecessor is observed complete at 10.2.
    with patch("vllm.v1.engine.core.time.perf_counter", side_effect=[10.1, 10.2]):
        engine.step_with_batch_queue()
    engine.scheduler.record_compute_time.assert_not_called()
    with patch("vllm.v1.engine.core.time.perf_counter", return_value=10.5):
        engine.step_with_batch_queue()
    engine.scheduler.record_compute_time.assert_called_once_with(
        "prefill", pytest.approx(0.3), contended=True, scheduled_tokens=0
    )


@pytest.mark.parametrize("predecessor_tokens", [0, 1])
@pytest.mark.parametrize("deferred", [False, True])
def test_untimed_executor_queue_never_reads_clock(predecessor_tokens, deferred):
    engine = _queued_engine(
        predecessor_tokens, successor_timed=False, deferred=deferred
    )
    with patch(
        "vllm.v1.engine.core.time.perf_counter",
        side_effect=AssertionError("An uncontended queue must not read the clock"),
    ):
        engine.step_with_batch_queue()
        engine.step_with_batch_queue()
    engine.scheduler.record_compute_time.assert_not_called()
    assert engine._last_model_completion_time is None
