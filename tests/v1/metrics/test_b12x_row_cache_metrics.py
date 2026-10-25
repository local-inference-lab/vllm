# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus collector for the B12X disk row-cache and prefetch counters."""

import prometheus_client
import pytest

from vllm.v1.metrics.b12x_row_cache import (
    _ENGINE_COUNTERS,
    _TABLE_COUNTERS,
    _TABLE_GAUGES,
    B12xRowCacheProm,
    B12xRowCacheStats,
    B12xTableStats,
)

LABELNAMES = ["model_name", "engine"]
LABELVALUES = {0: ["qwen4exp", "0"]}
BASE = {"model_name": "qwen4exp", "engine": "0"}


def _bound_classes(registry):
    """The metric classes the collector uses, pinned to one test registry."""

    class _Bound(B12xRowCacheProm):
        """The collector with its metric classes bound to the test registry."""

        _counter_cls = staticmethod(
            lambda **kwargs: prometheus_client.Counter(registry=registry, **kwargs)
        )
        _gauge_cls = staticmethod(
            lambda **kwargs: prometheus_client.Gauge(registry=registry, **kwargs)
        )

    return _Bound


@pytest.fixture
def collector():
    """A fresh registry and its collector, so families never leak between tests."""
    registry = prometheus_client.CollectorRegistry()
    return _bound_classes(registry)(LABELNAMES, LABELVALUES), registry


def _table(kind, **fields):
    """One table snapshot with only the fields the test cares about set."""
    return B12xTableStats(kind=kind, **fields)


def _counter(registry, name, **labels):
    """Read a cumulative counter, whose exposed name carries ``_total``."""
    return registry.get_sample_value(f"{name}_total", {**BASE, **labels})


def _gauge(registry, name, **labels):
    """Read a gauge, whose exposed name is the declared name unchanged."""
    return registry.get_sample_value(name, {**BASE, **labels})


def test_cumulative_counters_are_published_as_deltas(collector):
    """The worker sends totals; the registry must show the same running total."""
    prom, registry = collector
    payload = B12xRowCacheStats(
        submissions=5, dropped_submissions=2, staged_rows=11, tables=[_table("ple")]
    )

    prom.observe(payload)
    prom.observe(
        B12xRowCacheStats(
            submissions=8,
            dropped_submissions=2,
            staged_rows=30,
            tables=[_table("ple", hits=40, misses=10)],
        )
    )

    assert _counter(registry, "vllm:b12x_row_prefetch_submissions") == 8
    assert _counter(registry, "vllm:b12x_row_prefetch_staged_rows") == 30
    assert _counter(registry, "vllm:b12x_row_cache_hits", kind="ple") == 40
    assert _counter(registry, "vllm:b12x_row_cache_misses", kind="ple") == 10


def test_a_restarted_worker_rebaselines_instead_of_decrementing(collector):
    """Counters cannot go down, so smaller totals start a new accumulation."""
    prom, registry = collector
    prom.observe(B12xRowCacheStats(submissions=8, tables=[_table("ple", hits=40)]))
    assert _counter(registry, "vllm:b12x_row_prefetch_submissions") == 8

    # A fresh worker reports from zero again; the family keeps the first
    # worker's total and adds the new one, and never emits a decrease.
    prom.observe(B12xRowCacheStats(submissions=2, tables=[_table("ple", hits=5)]))
    assert _counter(registry, "vllm:b12x_row_prefetch_submissions") == 10
    assert _counter(registry, "vllm:b12x_row_cache_hits", kind="ple") == 45


def test_gauges_track_the_latest_snapshot_not_the_history(collector):
    """Residency and queue depth go down as well as up."""
    prom, registry = collector
    prom.observe(
        B12xRowCacheStats(
            tables=[
                _table(
                    "ple",
                    resident_rows=25,
                    pool_bytes=2250,
                    prefetch_queue_depth=4,
                )
            ]
        )
    )
    assert _gauge(registry, "vllm:b12x_row_cache_resident_rows", kind="ple") == 25
    assert _gauge(registry, "vllm:b12x_row_cache_pool_bytes", kind="ple") == 2250
    assert _gauge(registry, "vllm:b12x_row_cache_prefetch_queue_depth", kind="ple") == 4

    # The worker sends a whole snapshot every time, so a gauge that falls to
    # zero really does mean an empty cache or a drained queue.
    prom.observe(B12xRowCacheStats(tables=[_table("ple", resident_rows=7)]))
    assert _gauge(registry, "vllm:b12x_row_cache_resident_rows", kind="ple") == 7
    assert _gauge(registry, "vllm:b12x_row_cache_pool_bytes", kind="ple") == 0
    assert _gauge(registry, "vllm:b12x_row_cache_prefetch_queue_depth", kind="ple") == 0


def test_prefetch_disablement_is_visible_as_a_gauge(collector):
    """A prefetcher that gave up must be distinguishable from one never used."""
    prom, registry = collector
    prom.observe(B12xRowCacheStats(tables=[_table("ple")]))
    assert _gauge(registry, "vllm:b12x_row_prefetch_disabled") == 0

    prom.observe(B12xRowCacheStats(disabled=1, tables=[_table("ple")]))
    assert _gauge(registry, "vllm:b12x_row_prefetch_disabled") == 1


def test_each_kind_gets_its_own_series(collector):
    """PLE and Engram counters never merge into one unlabeled total."""
    prom, registry = collector
    prom.observe(
        B12xRowCacheStats(
            tables=[_table("ple", hits=40), _table("engram", hits=3, misses=7)]
        )
    )

    assert _counter(registry, "vllm:b12x_row_cache_hits", kind="ple") == 40
    assert _counter(registry, "vllm:b12x_row_cache_hits", kind="engram") == 3
    assert _counter(registry, "vllm:b12x_row_cache_misses", kind="engram") == 7


def test_every_declared_metric_is_reachable(collector):
    """A declared metric nobody writes is a dashboard panel that reads zero."""
    prom, registry = collector
    table_fields = {attribute: 3 for attribute, _name, _doc in _TABLE_COUNTERS}
    table_fields.update({attribute: 3 for attribute, _name, _doc in _TABLE_GAUGES})
    prom.observe(
        B12xRowCacheStats(
            submissions=1,
            dropped_submissions=1,
            staged_rows=1,
            disabled=1,
            tables=[_table("ple", **table_fields), _table("engram", **table_fields)],
        )
    )

    for _attribute, name, _documentation in _ENGINE_COUNTERS:
        assert _counter(registry, name) == 1, name
    assert _gauge(registry, "vllm:b12x_row_prefetch_disabled") == 1
    for _attribute, name, _documentation in _TABLE_COUNTERS:
        for kind in ("ple", "engram"):
            assert _counter(registry, name, kind=kind) == 3, f"{name} {kind}"
    for _attribute, name, _documentation in _TABLE_GAUGES:
        for kind in ("ple", "engram"):
            assert _gauge(registry, name, kind=kind) == 3, f"{name} {kind}"


def test_two_engines_keep_their_counters_apart(collector):
    """Per-engine children never share a cumulative baseline."""
    registry = prometheus_client.CollectorRegistry()
    prom = _bound_classes(registry)(
        LABELNAMES, {0: ["qwen4exp", "0"], 1: ["qwen4exp", "1"]}
    )

    prom.observe(B12xRowCacheStats(submissions=4, tables=[]), engine_idx=0)
    prom.observe(B12xRowCacheStats(submissions=9, tables=[]), engine_idx=1)
    prom.observe(B12xRowCacheStats(submissions=6, tables=[]), engine_idx=0)

    assert (
        registry.get_sample_value(
            "vllm:b12x_row_prefetch_submissions_total",
            {"model_name": "qwen4exp", "engine": "0"},
        )
        == 6
    )
    assert (
        registry.get_sample_value(
            "vllm:b12x_row_prefetch_submissions_total",
            {"model_name": "qwen4exp", "engine": "1"},
        )
        == 9
    )


def test_the_worker_payload_reaches_the_prometheus_exposition():
    """Counters relayed on SchedulerStats must show up on the real scrape."""
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.metrics.loggers import PrometheusStatLogger
    from vllm.v1.metrics.prometheus import unregister_vllm_metrics
    from vllm.v1.metrics.stats import SchedulerStats

    logger = PrometheusStatLogger(
        VllmConfig(model_config=ModelConfig(model="distilbert/distilgpt2"))
    )
    try:
        logger.record(
            scheduler_stats=SchedulerStats(
                b12x_row_cache_stats=B12xRowCacheStats(
                    submissions=4,
                    dropped_submissions=1,
                    staged_rows=12,
                    disabled=1,
                    tables=[_table("ple", hits=40, misses=10, resident_rows=25)],
                )
            ),
            iteration_stats=None,
        )
        found = {
            (sample.name, sample.labels.get("kind")): sample.value
            for metric in prometheus_client.REGISTRY.collect()
            if metric.name.startswith("vllm:b12x")
            for sample in metric.samples
        }
    finally:
        unregister_vllm_metrics()

    assert found[("vllm:b12x_row_prefetch_submissions_total", None)] == 4.0
    assert found[("vllm:b12x_row_prefetch_dropped_submissions_total", None)] == 1.0
    assert found[("vllm:b12x_row_prefetch_staged_rows_total", None)] == 12.0
    assert found[("vllm:b12x_row_prefetch_disabled", None)] == 1.0
    assert found[("vllm:b12x_row_cache_hits_total", "ple")] == 40.0
    assert found[("vllm:b12x_row_cache_misses_total", "ple")] == 10.0
    assert found[("vllm:b12x_row_cache_resident_rows", "ple")] == 25.0
    # The kind this step never touched is published as zero rather than absent,
    # so a dashboard cannot read "no series" as "no traffic".
    assert found[("vllm:b12x_row_cache_hits_total", "engram")] == 0.0


def test_the_payload_survives_the_engine_core_wire_format():
    """A multiprocess server decodes this on the frontend, so pin the wire.

    Everything else in this file calls the collector in process, which would
    keep passing if msgpack dropped the nested table list -- and a dropped
    list is silently empty metrics in every real deployment, not a red test.
    """
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    stats = SchedulerStats(
        b12x_row_cache_stats=B12xRowCacheStats(
            submissions=7,
            dropped_submissions=2,
            staged_rows=99,
            disabled=0,
            tables=[
                _table(
                    "ple",
                    hits=11,
                    misses=22,
                    resident_rows=33,
                    pool_bytes=44,
                    prefetch_rows=55,
                    prefetch_rows_loaded=66,
                    prefetch_errors=77,
                    prefetch_queue_depth=88,
                )
            ],
        )
    )

    decoded = MsgpackDecoder(EngineCoreOutputs).decode(
        MsgpackEncoder().encode(EngineCoreOutputs(scheduler_stats=stats))
    )

    payload = decoded.scheduler_stats.b12x_row_cache_stats
    assert payload is not None
    assert (payload.submissions, payload.dropped_submissions) == (7, 2)
    assert (payload.staged_rows, payload.disabled) == (99, 0)
    table = payload.tables[0]
    assert isinstance(table, B12xTableStats)
    assert (table.kind, table.hits, table.misses) == ("ple", 11, 22)
    assert (table.pool_bytes, table.prefetch_rows_loaded) == (44, 66)
    assert (table.prefetch_errors, table.prefetch_queue_depth) == (77, 88)
