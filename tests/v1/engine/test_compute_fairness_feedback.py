# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
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

    future, timing = engine._execute_model(_scheduler_output(None))

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
