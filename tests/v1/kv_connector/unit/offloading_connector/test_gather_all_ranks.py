# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The executor gathers every worker's output only while the offloading
connector has transfers outstanding."""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor


def _connector(pending: list[bool]) -> OffloadingConnector:
    conn = OffloadingConnector.__new__(OffloadingConnector)
    conn.connector_scheduler = SimpleNamespace(
        has_pending_push_work=lambda: pending.pop(0)
    )
    return conn


def test_gathers_while_jobs_are_outstanding_and_two_steps_after():
    conn = _connector([True, False, False, False, True, False])
    got = [conn.requires_all_worker_outputs() for _ in range(6)]
    assert got == [True, True, True, False, True, True]


def test_other_connectors_always_gather():
    assert KVConnectorBase_V1.requires_all_worker_outputs(object()) is True
    assert (
        SchedulerOutput.__dataclass_fields__["kv_connector_gather_all_ranks"].default
        is True
    )


def test_executor_drops_the_aggregator_on_steps_that_need_no_gather():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.kv_output_aggregator = aggregator = object()
    # before any step: gather (the conservative default)
    assert executor._step_kv_output_aggregator() is aggregator
    executor._kv_gather_all_ranks = False
    assert executor._step_kv_output_aggregator() is None
    executor._kv_gather_all_ranks = True
    assert executor._step_kv_output_aggregator() is aggregator
