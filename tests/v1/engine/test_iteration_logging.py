# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace

import pytest

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(log_stats: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
    )


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_non_dp_prefill_schedule_interval_uses_scheduler_step():
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(prefill_schedule_interval=4)
        ),
        scheduler=SimpleNamespace(current_step=0),
    )

    expected_by_completed_step = (False, True, True, True, False)
    for completed_step, expected in enumerate(expected_by_completed_step):
        engine.scheduler.current_step = completed_step
        assert EngineCore._should_throttle_prefills(engine) is expected

    engine.vllm_config.scheduler_config.prefill_schedule_interval = 1
    engine.scheduler.current_step = 3
    assert not EngineCore._should_throttle_prefills(engine)


@pytest.mark.parametrize("current_step", [None, -1, True, 1.5])
def test_non_dp_prefill_schedule_interval_validates_scheduler_step(current_step):
    scheduler = SimpleNamespace()
    if current_step is not None:
        scheduler.current_step = current_step
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(prefill_schedule_interval=4)
        ),
        scheduler=scheduler,
    )

    with pytest.raises(RuntimeError, match="scheduler.current_step"):
        EngineCore._should_throttle_prefills(engine)
