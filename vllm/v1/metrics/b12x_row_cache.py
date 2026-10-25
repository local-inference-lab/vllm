# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X disk row-cache and advisory prefetch metrics for PLE and Engram.

The counters are collected in the worker, next to the prefetch thread and the
b12x caches they describe, and reach the Prometheus registry by the same
engine-output path the scheduler stats use: ``ModelRunnerOutput`` -> scheduler
-> ``SchedulerStats`` -> frontend :class:`PrometheusStatLogger`. Every b12x and
prefetcher counter is cumulative for the worker process, so the collector
publishes the deltas it observes and re-baselines when a worker restarts and
the counters go backwards.

Row-cache sizing and prefetch adequacy are the two questions these answer.
Host-row-cache hit rate, per table kind:

  rate(vllm:b12x_row_cache_hits_total[$interval])
    / (rate(vllm:b12x_row_cache_hits_total[$interval])
       + rate(vllm:b12x_row_cache_misses_total[$interval]))

Prefetch that cannot keep up with the disk, which means raising
``B12X_ROW_CACHE_BYTES`` or the b12x prefetch queue bound:

  rate(vllm:b12x_row_cache_prefetch_rows_dropped_total[$interval])

Prefetch work discarded before it is even queued, which means the pinned
staging ring is too shallow for the step cadence:

  rate(vllm:b12x_row_prefetch_dropped_submissions_total[$interval])
"""

from dataclasses import dataclass, field

import prometheus_client

from vllm.v1.metrics.utils import create_metric_per_engine

# The disk table families a worker can hold. Fixed and small, so the child for
# every kind is bound at construction instead of on first observation.
TABLE_KINDS = ("ple", "engram")

# (payload attribute, metric name, documentation). Drives both the declaration
# and the observation loop, so a counter cannot be declared but never fed.
_ENGINE_COUNTERS: tuple[tuple[str, str, str], ...] = (
    (
        "submissions",
        "vllm:b12x_row_prefetch_submissions",
        "Steps whose next-step rows were handed to the prefetch worker.",
    ),
    (
        "dropped_submissions",
        "vllm:b12x_row_prefetch_dropped_submissions",
        (
            "Steps the prefetcher could not queue because every pinned "
            "staging slot was still in flight. Steps skipped for other "
            "reasons are not counted: CUDA graph capture and a disabled "
            "prefetcher do no work to lose."
        ),
    ),
    (
        "staged_rows",
        "vllm:b12x_row_prefetch_staged_rows",
        "Distinct row ids the prefetcher handed to the disk tables to warm.",
    ),
)

_TABLE_COUNTERS: tuple[tuple[str, str, str], ...] = (
    (
        "hits",
        "vllm:b12x_row_cache_hits",
        "Row-cache lookups served from the host row cache.",
    ),
    (
        "misses",
        "vllm:b12x_row_cache_misses",
        "Row-cache lookups that had to read the rows from the checkpoint.",
    ),
    (
        "prefetch_rows",
        "vllm:b12x_row_cache_prefetch_rows",
        "Rows the b12x prefetch queue accepted for warming.",
    ),
    (
        "prefetch_rows_dropped",
        "vllm:b12x_row_cache_prefetch_rows_dropped",
        "Rows discarded because the b12x prefetch queue was full.",
    ),
    (
        "prefetch_rows_loaded",
        "vllm:b12x_row_cache_prefetch_rows_loaded",
        "Rows the b12x prefetch worker fetched and published ready.",
    ),
    (
        "prefetch_rows_cancelled",
        "vllm:b12x_row_cache_prefetch_rows_cancelled",
        "Rows the prefetch worker reserved and then cancelled unpublished.",
    ),
    (
        "prefetch_errors",
        "vllm:b12x_row_cache_prefetch_errors",
        "Source or worker errors the b12x prefetch path swallowed.",
    ),
)

_TABLE_GAUGES: tuple[tuple[str, str, str], ...] = (
    (
        "resident_rows",
        "vllm:b12x_row_cache_resident_rows",
        "Rows resident in the host row cache.",
    ),
    (
        "pool_bytes",
        "vllm:b12x_row_cache_pool_bytes",
        "Host row-cache pool budget in bytes.",
    ),
    (
        "prefetch_queue_depth",
        "vllm:b12x_row_cache_prefetch_queue_depth",
        "Prefetch jobs waiting in the b12x prefetch queue.",
    ),
)


@dataclass
class B12xTableStats:
    """Cumulative counters for every disk table of one kind in one worker.

    ``hits``, ``misses`` and the prefetch counters are b12x ``stats()`` totals
    summed across the tables of this kind; ``resident_rows``, ``pool_bytes``
    and ``prefetch_queue_depth`` are the current values.
    """

    kind: str = TABLE_KINDS[0]
    hits: int = 0
    misses: int = 0
    resident_rows: int = 0
    pool_bytes: int = 0
    prefetch_rows: int = 0
    prefetch_rows_dropped: int = 0
    prefetch_rows_loaded: int = 0
    prefetch_rows_cancelled: int = 0
    prefetch_errors: int = 0
    prefetch_queue_depth: int = 0


@dataclass
class B12xRowCacheStats:
    """Worker-side prefetch and row-cache counters for one engine step.

    The four leading fields count this worker's own advisory prefetch work;
    ``tables`` carries what the b12x caches report.
    """

    submissions: int = 0
    dropped_submissions: int = 0
    staged_rows: int = 0
    disabled: int = 0
    tables: list[B12xTableStats] = field(default_factory=list)


class B12xRowCacheProm:
    """Record B12X disk row-cache and prefetch metrics in Prometheus.

    Built by :class:`PrometheusStatLogger` only; the worker fills one of the
    dataclasses above and lets the ordinary msgpack engine-output path carry
    it to the frontend, so nothing is ever registered in the worker process.
    """

    _counter_cls = prometheus_client.Counter
    _gauge_cls = prometheus_client.Gauge

    def __init__(
        self,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        """Declare every family once per engine, with a kind child each.

        Building the children up front rather than on first observation keeps
        a kind that has never been sampled present as zero, so a dashboard
        cannot mistake an absent series for no traffic.
        """
        self.counters: dict[str, dict[int, prometheus_client.Counter]] = {}
        for _attribute, name, documentation in _ENGINE_COUNTERS:
            self.counters[name] = create_metric_per_engine(
                self._counter_cls(
                    name=name, documentation=documentation, labelnames=labelnames
                ),
                per_engine_labelvalues,
            )
        self.gauge_prefetch_disabled = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:b12x_row_prefetch_disabled",
                documentation=(
                    "One while the prefetcher has disabled itself after an "
                    "unusable row cache or an error; zero otherwise. Lookups "
                    "still serve, the worker only stops warming ahead of the "
                    "step."
                ),
                multiprocess_mode="mostrecent",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )

        kind_labelnames = [*labelnames, "kind"]
        self.table_counters: dict[str, dict[int, dict[str, object]]] = {}
        for _attribute, name, documentation in _TABLE_COUNTERS:
            base = self._counter_cls(
                name=name, documentation=documentation, labelnames=kind_labelnames
            )
            self.table_counters[name] = {
                idx: {kind: base.labels(*values, kind) for kind in TABLE_KINDS}
                for idx, values in per_engine_labelvalues.items()
            }
        self.table_gauges: dict[str, dict[int, dict[str, object]]] = {}
        for _attribute, name, documentation in _TABLE_GAUGES:
            base = self._gauge_cls(
                name=name,
                documentation=documentation,
                multiprocess_mode="mostrecent",
                labelnames=kind_labelnames,
            )
            self.table_gauges[name] = {
                idx: {kind: base.labels(*values, kind) for kind in TABLE_KINDS}
                for idx, values in per_engine_labelvalues.items()
            }

        # Last observed totals, for turning cumulative counters into deltas.
        self._last_engine: dict[int, B12xRowCacheStats] = {}
        self._last_table: dict[tuple[int, str], B12xTableStats] = {}

    def observe(self, stats: B12xRowCacheStats, engine_idx: int = 0) -> None:
        """Advance this engine's counters to a new cumulative worker snapshot.

        The worker reports process-lifetime totals, so each call publishes
        what arrived since the last one; gauges take the value outright.

        Every configured kind appears in every snapshot: the sampler creates a
        totals entry per table whether or not its cache answered, so a kind
        that has been observed cannot go missing and leave a gauge frozen.
        """
        previous = self._last_engine.get(engine_idx)
        for attribute, name, _documentation in _ENGINE_COUNTERS:
            delta = self._delta(
                getattr(previous, attribute, None), getattr(stats, attribute)
            )
            if delta:
                self.counters[name][engine_idx].inc(delta)
        self.gauge_prefetch_disabled[engine_idx].set(stats.disabled)

        for table in stats.tables:
            last = self._last_table.get((engine_idx, table.kind))
            for attribute, name, _documentation in _TABLE_COUNTERS:
                delta = self._delta(
                    getattr(last, attribute, None), getattr(table, attribute)
                )
                if delta:
                    counter = self.table_counters[name][engine_idx][table.kind]
                    counter.inc(delta)  # type: ignore[attr-defined]
            for attribute, name, _documentation in _TABLE_GAUGES:
                gauge = self.table_gauges[name][engine_idx][table.kind]
                gauge.set(getattr(table, attribute))  # type: ignore[attr-defined]
            self._last_table[(engine_idx, table.kind)] = table
        self._last_engine[engine_idx] = stats

    @staticmethod
    def _delta(previous: int | None, current: int) -> int:
        """Work added since this counter was last observed.

        A worker restart sends smaller cumulative values, so the counter is
        re-baselined from zero rather than decremented, which a Prometheus
        Counter cannot do.
        """
        if previous is None or current < previous:
            return current
        return current - previous
