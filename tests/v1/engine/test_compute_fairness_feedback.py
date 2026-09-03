# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from unittest.mock import Mock, call, patch

import pytest

from vllm.v1.core.sched.compute_fairness import ComputeServiceClass
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore, _ModelExecutionTiming

pytestmark = pytest.mark.cpu_test


def _scheduler_output(
    service_class: ComputeServiceClass | None,
    *,
    contended: bool = False,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.compute_service_class = service_class
    output.compute_contention = contended
    return output


def test_completion_callback_excludes_later_batch_queue_residency():
    future: Future[None] = Future()
    with patch("vllm.v1.engine.core.time.perf_counter", side_effect=[10.0, 10.25]):
        timing = _ModelExecutionTiming()
        timing.bind(future)
        future.set_result(None)

    # Reading much later must use the completion callback's timestamp.
    assert timing.elapsed_seconds == pytest.approx(0.25)


def test_disabled_output_does_not_install_execution_timing():
    completed_future: Future[None] = Future()
    completed_future.set_result(None)
    engine = object.__new__(EngineCore)
    engine.model_executor = Mock()
    engine.model_executor.execute_model.return_value = completed_future

    future, timing = engine._execute_model(_scheduler_output(None))

    assert future is completed_future
    assert timing is None


def test_enabled_execution_timer_waits_for_final_batch_future():
    execute_future: Future[None] = Future()
    execute_future.set_result(None)
    final_future: Future[None] = Future()
    engine = object.__new__(EngineCore)
    engine.model_executor = Mock()
    engine.model_executor.execute_model.return_value = execute_future

    _, timing = engine._execute_model(_scheduler_output("prefill", contended=True))
    assert timing is not None
    assert timing.completed_at is None

    timing.bind(final_future)
    assert timing.completed_at is None
    final_future.set_result(None)
    assert timing.completed_at is not None


def test_queued_feedback_stays_paired_with_exact_batch():
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()
    decode_output = _scheduler_output("decode", contended=True)
    prefill_output = _scheduler_output("prefill", contended=True)
    decode_timing = _ModelExecutionTiming()
    decode_timing.completed_at = decode_timing.started_at + 0.1
    prefill_timing = _ModelExecutionTiming()
    prefill_timing.completed_at = prefill_timing.started_at + 0.3

    # Queued batches are consumed oldest-first; each carries its own timing and
    # class tag even when a later batch has already completed.
    engine._record_compute_time(decode_output, decode_timing)
    engine._record_compute_time(prefill_output, prefill_timing)

    assert engine.scheduler.record_compute_time.call_args_list == [
        call("decode", pytest.approx(0.1), contended=True),
        call("prefill", pytest.approx(0.3), contended=True),
    ]


def test_empty_transfer_step_does_not_record_compute():
    engine = object.__new__(EngineCore)
    engine.scheduler = Mock()

    engine._record_compute_time(_scheduler_output(None), None)

    engine.scheduler.record_compute_time.assert_not_called()
