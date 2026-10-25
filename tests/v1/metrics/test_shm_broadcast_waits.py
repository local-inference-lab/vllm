# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blocked shared-memory broadcast waits reach the frontend's registry.

The waits are taken in the engine-core and worker processes, whose Prometheus
registries nothing scrapes, so the counts must travel on the model-runner
output and the scheduler stats to the process that serves ``/metrics``.
"""

import prometheus_client
import pytest

from tests.v1.core.utils import create_scheduler
from vllm.config import ModelConfig, ObservabilityConfig, VllmConfig
from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test

TEST_MODEL = "facebook/opt-125m"


@pytest.fixture(autouse=True)
def isolate_wait_totals(monkeypatch: pytest.MonkeyPatch):
    """Keep each test's counters out of the process-wide accumulator."""
    monkeypatch.setattr(shm_broadcast, "_shm_broadcast_wait_totals", {})


def test_a_blocked_writer_is_counted_for_the_stats_path():
    """A writer held up by an unread block publishes its wait for the drain."""
    writer = MessageQueue(
        n_reader=1, n_local_reader=1, max_chunk_bytes=1024, max_chunks=1
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()
    try:
        writer.enqueue("first")
        with pytest.raises(TimeoutError):
            writer.enqueue("second", timeout=0.05)
    finally:
        writer.shutdown()
        reader.shutdown()

    waits = shm_broadcast.drain_shm_broadcast_wait_metrics()
    assert waits["writer"][0] == 1
    # The wait is measured from when the block is first noticed, so it lands
    # just short of the 50 ms budget rather than at or above it.
    assert waits["writer"][1] == pytest.approx(0.05, abs=0.01)


def test_a_drain_reports_each_role_once():
    """Draining resets the totals, so no wait is counted twice."""
    shm_broadcast._report_shm_broadcast_wait_metrics("reader", 2, 0.25)
    shm_broadcast._report_shm_broadcast_wait_metrics("writer", 1, 0.5)

    assert shm_broadcast.drain_shm_broadcast_wait_metrics() == {
        "reader": (2, pytest.approx(0.25)),
        "writer": (1, pytest.approx(0.5)),
    }
    assert shm_broadcast.drain_shm_broadcast_wait_metrics() == {}

    shm_broadcast._report_shm_broadcast_wait_metrics("reader", 1, 0.1)
    assert shm_broadcast.drain_shm_broadcast_wait_metrics() == {
        "reader": (1, pytest.approx(0.1))
    }


def test_scheduler_reports_engine_and_worker_waits():
    """SchedulerStats carries the engine's drains plus the worker's reply."""
    scheduler = create_scheduler()
    scheduler_output = scheduler.schedule()
    shm_broadcast._report_shm_broadcast_wait_metrics("writer", 3, 0.5)
    model_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    model_output.shm_broadcast_waits = {"reader": (1, 0.25)}

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    stats = next(iter(outputs.values())).scheduler_stats
    assert stats is not None
    assert stats.shm_broadcast_waits == {
        "writer": (3, pytest.approx(0.5)),
        "reader": (1, pytest.approx(0.25)),
    }
    # Reported once: the next report starts from an empty accumulator.
    reported = scheduler.make_stats()
    assert reported is not None
    assert reported.shm_broadcast_waits == {}


def test_prometheus_logger_exports_the_waits_by_role():
    """The frontend's registry gets the counts, labelled by queue role."""
    config = VllmConfig(
        model_config=ModelConfig(model=TEST_MODEL),
        observability_config=ObservabilityConfig(),
    )
    values: dict[tuple[str, str], float] = {}
    try:
        stat_logger = PrometheusStatLogger(config)
        stat_logger.record(
            SchedulerStats(shm_broadcast_waits={"reader": (2, 0.5)}), None
        )
        for metric in prometheus_client.REGISTRY.collect():
            if not metric.name.startswith("vllm:shm_broadcast_blocked"):
                continue
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    values[(metric.name, sample.labels["role"])] = sample.value
    finally:
        unregister_vllm_metrics()

    assert values == {
        ("vllm:shm_broadcast_blocked_waits", "reader"): 2,
        ("vllm:shm_broadcast_blocked_wait_seconds", "reader"): pytest.approx(0.5),
    }


def test_the_waits_survive_the_engine_to_frontend_codec():
    """The engine's msgpack message is what carries the counts to the frontend."""
    stats = SchedulerStats(shm_broadcast_waits={"reader": (2, 0.5), "writer": (1, 0.1)})
    encoded = MsgpackEncoder().encode(EngineCoreOutputs(scheduler_stats=stats))

    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.shm_broadcast_waits == {
        "reader": (2, 0.5),
        "writer": (1, 0.1),
    }
