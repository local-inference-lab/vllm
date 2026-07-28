# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Late or duplicate KV-transfer completions must not kill the engine core.

A worker-side connector can report finished_recving/finished_sending for a
request the scheduler no longer tracks (abort racing an async KV load, or the
same request reported twice) or tracks in an unexpected status. The scheduler
must ignore such completions instead of raising AssertionError.
"""

import copy

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _deliver_kv_connector_output(
    scheduler: Scheduler, scheduler_output: SchedulerOutput, **kwargs
) -> None:
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(**kwargs)
    scheduler.update_from_output(scheduler_output, model_runner_output)


def test_finished_sending_for_untracked_request_ignored():
    """finished_sending for an unknown request id is a no-op."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    scheduler_output = scheduler.schedule()
    _deliver_kv_connector_output(
        scheduler, scheduler_output, finished_sending={"id-unknown"}
    )

    assert_scheduler_empty(scheduler)


def test_finished_recving_for_untracked_request_ignored():
    """finished_recving for an unknown request id is a no-op."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    scheduler_output = scheduler.schedule()
    _deliver_kv_connector_output(
        scheduler, scheduler_output, finished_recving={"id-unknown"}
    )

    assert_scheduler_empty(scheduler)


def test_finished_recving_for_running_request_keeps_blocks():
    """A completion for a live request that is not waiting for remote KVs
    must not free its blocks; the request keeps running and finishes
    normally."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(max_tokens=4)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request], finished_recving={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.RUNNING
    assert request.request_id not in scheduler.finished_recving_kv_req_ids

    # The request still finishes cleanly and releases everything.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert request.is_finished()

    scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_duplicate_finished_sending_after_free_ignored():
    """A second finished_sending for a request whose blocks were already
    freed by the first completion is ignored."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(do_remote_decode=True)
    scheduler.add_request(request)
    request_id = request.request_id

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert request.is_finished()

    # Pass the finished request to the persistent batch.
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    # The first completion frees the blocks.
    scheduler_output = scheduler.schedule()
    _deliver_kv_connector_output(
        scheduler, scheduler_output, finished_sending={request_id}
    )
    assert_scheduler_empty(scheduler)

    # The duplicate completion is ignored.
    scheduler_output = scheduler.schedule()
    _deliver_kv_connector_output(
        scheduler, scheduler_output, finished_sending={request_id}
    )
    assert_scheduler_empty(scheduler)
