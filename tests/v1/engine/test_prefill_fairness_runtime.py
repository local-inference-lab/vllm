# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("has_unfinished_requests", "batch_queue"),
    [(True, None), (False, [object()]), (True, [object()])],
)
def test_engine_core_applies_policy_change_under_live_load(
    has_unfinished_requests, batch_queue
):
    scheduler = Mock()
    scheduler.has_unfinished_requests.return_value = has_unfinished_requests
    scheduler.set_prefill_fairness.return_value = {
        "prefill_compute_share": "auto",
        "prefill_compute_half_life": "responsive",
    }
    core = SimpleNamespace(scheduler=scheduler, batch_queue=batch_queue)
    config = {
        "prefill_compute_share": "auto",
        "prefill_compute_half_life": "responsive",
    }

    result = EngineCore.set_prefill_fairness(core, config)

    assert result["applied"] is True
    assert result["config"] == scheduler.set_prefill_fairness.return_value
    scheduler.set_prefill_fairness.assert_called_once_with(config)


def test_engine_core_applies_policy_change_when_idle():
    scheduler = Mock()
    scheduler.has_unfinished_requests.return_value = False
    scheduler.set_prefill_fairness.return_value = {
        "prefill_compute_share": 0.5,
    }
    core = SimpleNamespace(scheduler=scheduler, batch_queue=None)
    config = {"prefill_compute_share": 0.5}

    result = EngineCore.set_prefill_fairness(core, config)

    assert result["applied"] is True
    assert result["config"] == scheduler.set_prefill_fairness.return_value
    scheduler.set_prefill_fairness.assert_called_once_with(config)


def test_engine_core_returns_invalid_policy_without_mutating_engine():
    scheduler = Mock()
    scheduler.has_unfinished_requests.return_value = False
    scheduler.set_prefill_fairness.side_effect = ValueError("bad policy")
    scheduler.get_prefill_fairness.return_value = {"prefill_compute_share": None}
    core = SimpleNamespace(scheduler=scheduler, batch_queue=None)

    result = EngineCore.set_prefill_fairness(core, {"prefill_compute_share": "bad"})

    assert result["applied"] is False
    assert result["reason"] == "invalid"
    assert result["config"] == {"prefill_compute_share": None}


def test_scheduler_rejects_removed_selector_fields():
    scheduler = SimpleNamespace()

    with pytest.raises(ValueError, match="unknown prefill compute-share fields"):
        Scheduler.set_prefill_fairness(
            scheduler,
            {"fairness_engine": "micro_slicing"},
        )
