# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import errno
import functools
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[str] = MEDIUM_FS

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_cache_size_bytes: int | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            max_cache_size_bytes: Optional byte limit for this tier's block
                files. When set, the least-recently-used unpinned blocks are
                removed before new blocks are stored. The limit applies only
                to this model/configuration namespace and requires exclusive
                ownership of that namespace by this vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self._events_lock = threading.Lock()
        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        if max_cache_size_bytes is not None:
            if isinstance(max_cache_size_bytes, bool) or not isinstance(
                max_cache_size_bytes, int
            ):
                raise TypeError("max_cache_size_bytes must be an integer or None")
            if max_cache_size_bytes < self._block_size:
                raise ValueError(
                    "max_cache_size_bytes must fit at least one filesystem "
                    f"cache block ({self._block_size} bytes)"
                )
        self._max_cache_size_bytes = max_cache_size_bytes

        # Capacity state is used only when max_cache_size_bytes is configured.
        # _cache_lru is oldest-first. Pending stores reserve their final file
        # size before I/O so concurrent writers cannot oversubscribe the limit.
        self._capacity_cv = threading.Condition(threading.Lock())
        self._cache_lru: OrderedDict[OffloadKey, int] = OrderedDict()
        self._cache_size_bytes = 0
        self._pending_store_keys: set[OffloadKey] = set()
        self._pending_store_bytes = 0
        self._request_pins: dict[str, set[OffloadKey]] = {}
        self._pin_counts: dict[OffloadKey, int] = {}
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        if self._max_cache_size_bytes is not None:
            self._initialize_capacity_index()

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        if not result:
            return LookupResult.MISS
        if self._max_cache_size_bytes is not None:
            pinned = self._pin_lookup_hit(key, req_context.req_id)
            if pinned is None:
                # A store has published the file but has not committed its
                # capacity reservation yet. Retry once that transition ends.
                return LookupResult.RETRY
            if not pinned:
                # The async existence result raced an eviction.
                return LookupResult.MISS
        return LookupResult.HIT

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        if self._max_cache_size_bytes is None:
            tasks = (
                functools.partial(
                    store_block,
                    self.file_mapper.get_file_name(key),
                    self._primary_kv_view,
                    int(bid) * self._block_size,
                    self._block_size,
                )
                for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
            )
        else:
            tasks = (
                functools.partial(
                    self._store_block_bounded,
                    key,
                    int(bid) * self._block_size,
                )
                for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
            )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self._max_cache_size_bytes is None:
            tasks = (
                functools.partial(
                    load_block,
                    self.file_mapper.get_file_name(key),
                    self._primary_kv_view,
                    int(bid) * self._block_size,
                    self._block_size,
                )
                for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
            )
        else:
            # Pin before enqueueing: load tasks can wait behind other work,
            # and capacity must not reclaim their source files in that gap.
            pinned_keys: list[OffloadKey] = []
            with self._capacity_cv:
                for key in job_metadata.keys:
                    if key in self._cache_lru:
                        self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
                        pinned_keys.append(key)
                        self._release_request_pin_locked(
                            job_metadata.req_context.req_id, key
                        )
                self._load_job_keys[job_metadata.job_id] = pinned_keys
            tasks = (
                functools.partial(
                    self._load_block_bounded,
                    key,
                    int(bid) * self._block_size,
                )
                for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
            )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if self._max_cache_size_bytes is not None:
                self._release_load_job_pins(job_id)
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    if self._max_cache_size_bytes is not None:
                        with self._capacity_cv:
                            # Removal events are idempotent invalidations. If a
                            # block was evicted before this completion poll,
                            # do not publish a stale store event for it.
                            keys = [key for key in keys if key in self._cache_lru]
                            if keys:
                                self._append_event(
                                    OffloadingEvent(
                                        keys=keys,
                                        medium=self.medium,
                                        removed=False,
                                        locality=self.locality,
                                    )
                                )
                    else:
                        self._append_event(
                            OffloadingEvent(
                                keys=keys,
                                medium=self.medium,
                                removed=False,
                                locality=self.locality,
                            )
                        )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            with self._events_lock:
                events = list(self.events)
                self.events.clear()
            yield from events

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if self._max_cache_size_bytes is None:
            return
        with self._capacity_cv:
            for key in keys:
                if key in self._cache_lru:
                    self._cache_lru.move_to_end(key)

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        if self._max_cache_size_bytes is not None:
            with self._capacity_cv:
                for key in self._request_pins.pop(req_context.req_id, ()):
                    self._unpin_key_locked(key)
                self._capacity_cv.notify_all()

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
        if self._max_cache_size_bytes is not None:
            with self._capacity_cv:
                for job_id in list(self._load_job_keys):
                    self._release_load_job_pins_locked(job_id)

    def _append_event(self, event: OffloadingEvent) -> None:
        if self.events is not None:
            with self._events_lock:
                self.events.append(event)

    def _key_from_cache_path(self, path: str) -> OffloadKey | None:
        """Recover an OffloadKey from a FileMapper-owned block path."""
        filename = os.path.basename(path)
        if not filename.endswith(".bin"):
            return None
        group_dir = os.path.basename(os.path.dirname(path))
        _, separator, group_idx = group_dir.rpartition("_g")
        if not separator:
            return None
        try:
            key = make_offload_key(
                bytes.fromhex(filename.removesuffix(".bin")), int(group_idx)
            )
        except (ValueError, OverflowError):
            return None
        expected = os.path.normpath(self.file_mapper.get_file_name(key))
        return key if expected == os.path.normpath(path) else None

    def _key_from_cache_temp_path(self, path: str) -> OffloadKey | None:
        """Recover an OffloadKey from a FileMapper-owned temporary path."""
        if not path.endswith(".tmp"):
            return None
        final_path, separator, suffix = path.removesuffix(".tmp").rpartition("_")
        if not separator or not suffix.isdecimal():
            return None
        return self._key_from_cache_path(final_path)

    def _initialize_capacity_index(self) -> None:
        """Index existing block files and trim an oversized namespace."""
        data_dir = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        entries: list[tuple[int, str, OffloadKey, int]] = []
        stale_temp_files_removed = 0
        for dir_path, _, filenames in os.walk(data_dir):
            for filename in filenames:
                path = os.path.join(dir_path, filename)
                key = self._key_from_cache_path(path)
                if key is None:
                    if self._key_from_cache_temp_path(path) is None:
                        continue
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        raise OSError(
                            f"Unable to remove stale filesystem KV temp file {path}: "
                            f"{exc}"
                        ) from exc
                    stale_temp_files_removed += 1
                    continue
                try:
                    stat_result = os.stat(path)
                except OSError:
                    continue
                entries.append(
                    (stat_result.st_mtime_ns, path, key, stat_result.st_size)
                )

        if stale_temp_files_removed:
            logger.info(
                "Removed %d stale filesystem KV temp files from %s",
                stale_temp_files_removed,
                data_dir,
            )

        # mtime is the restart-safe approximation of prior recency. Runtime
        # touches maintain exact LRU order after initialization.
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        with self._capacity_cv:
            for _, _, key, size in entries:
                previous = self._cache_lru.pop(key, None)
                if previous is not None:
                    self._cache_size_bytes -= previous
                self._cache_lru[key] = size
                self._cache_size_bytes += size
            if not self._evict_until_fits_locked(0):
                raise OSError(
                    "Unable to reduce filesystem KV cache to "
                    f"max_cache_size_bytes={self._max_cache_size_bytes}"
                )

        logger.info(
            "Filesystem KV cache capacity enabled: %d/%d bytes in %s",
            self._cache_size_bytes,
            self._max_cache_size_bytes,
            data_dir,
        )

    def _evict_until_fits_locked(self, required_bytes: int) -> bool:
        """Evict unpinned LRU entries until required_bytes can be reserved."""
        assert self._max_cache_size_bytes is not None
        if (
            self._cache_size_bytes + self._pending_store_bytes + required_bytes
            <= self._max_cache_size_bytes
        ):
            return True

        for key in tuple(self._cache_lru):
            if self._pin_counts.get(key, 0):
                continue
            size = self._cache_lru[key]
            path = self.file_mapper.get_file_name(key)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to evict filesystem KV block %s: %s", path, exc)
                continue
            del self._cache_lru[key]
            self._cache_size_bytes -= size
            self._append_event(
                OffloadingEvent(
                    keys=[key],
                    medium=self.medium,
                    removed=True,
                    locality=self.locality,
                )
            )
            if (
                self._cache_size_bytes + self._pending_store_bytes + required_bytes
                <= self._max_cache_size_bytes
            ):
                return True
        return False

    def _reserve_store(self, key: OffloadKey) -> bool:
        """Reserve one block. Return False when it is already resident."""
        path = self.file_mapper.get_file_name(key)
        with self._capacity_cv:
            while True:
                size = self._cache_lru.get(key)
                if size is not None:
                    if os.path.exists(path):
                        self._cache_lru.move_to_end(key)
                        return False
                    del self._cache_lru[key]
                    self._cache_size_bytes -= size
                    self._append_event(
                        OffloadingEvent(
                            keys=[key],
                            medium=self.medium,
                            removed=True,
                            locality=self.locality,
                        )
                    )

                if key in self._pending_store_keys:
                    self._capacity_cv.wait()
                    continue

                if self._evict_until_fits_locked(self._block_size):
                    self._pending_store_keys.add(key)
                    self._pending_store_bytes += self._block_size
                    return True

                # A concurrent store may become an evictable resident entry.
                if self._pending_store_bytes:
                    self._capacity_cv.wait()
                    continue

                raise OSError(
                    errno.ENOSPC,
                    "Filesystem KV cache is full and every resident block is "
                    "pinned by an active request or load job",
                )

    def _store_block_bounded(self, key: OffloadKey, offset: int) -> None:
        if not self._reserve_store(key):
            return

        path = self.file_mapper.get_file_name(key)
        success = False
        try:
            store_block(
                path,
                self._primary_kv_view,
                offset,
                self._block_size,
            )
            size = os.path.getsize(path)
            if size != self._block_size:
                try:
                    os.remove(path)
                except OSError as cleanup_exc:
                    logger.warning(
                        "Failed to remove invalid filesystem KV block %s: %s",
                        path,
                        cleanup_exc,
                    )
                raise OSError(
                    f"Invalid filesystem KV block size: expected "
                    f"{self._block_size} bytes, found {size} at {path}"
                )
            success = True
        finally:
            with self._capacity_cv:
                self._pending_store_keys.remove(key)
                self._pending_store_bytes -= self._block_size
                if success:
                    previous = self._cache_lru.pop(key, None)
                    if previous is not None:
                        self._cache_size_bytes -= previous
                    self._cache_lru[key] = self._block_size
                    self._cache_size_bytes += self._block_size
                self._capacity_cv.notify_all()

    def _load_block_bounded(self, key: OffloadKey, offset: int) -> None:
        path = self.file_mapper.get_file_name(key)
        try:
            load_block(
                path,
                self._primary_kv_view,
                offset,
                self._block_size,
            )
        except Exception:
            with self._capacity_cv:
                size = self._cache_lru.pop(key, None)
                if size is not None:
                    self._cache_size_bytes -= size
                    self._append_event(
                        OffloadingEvent(
                            keys=[key],
                            medium=self.medium,
                            removed=True,
                            locality=self.locality,
                        )
                    )
            raise
        else:
            with self._capacity_cv:
                if key in self._cache_lru:
                    self._cache_lru.move_to_end(key)

    def _pin_lookup_hit(self, key: OffloadKey, req_id: str) -> bool | None:
        """Pin a hit through request finalization to close lookup/load races."""
        with self._capacity_cv:
            req_keys = self._request_pins.setdefault(req_id, set())
            if key in req_keys:
                if key in self._cache_lru:
                    self._cache_lru.move_to_end(key)
                    return True
                req_keys.remove(key)
                self._unpin_key_locked(key)
                return False
            if key in self._pending_store_keys:
                return None
            if key not in self._cache_lru:
                return False
            req_keys.add(key)
            self._pin_counts[key] = self._pin_counts.get(key, 0) + 1
            self._cache_lru.move_to_end(key)
            return True

    def _release_load_job_pins(self, job_id: JobId) -> None:
        with self._capacity_cv:
            self._release_load_job_pins_locked(job_id)

    def _release_load_job_pins_locked(self, job_id: JobId) -> None:
        for key in self._load_job_keys.pop(job_id, ()):
            self._unpin_key_locked(key)
        self._capacity_cv.notify_all()

    def _release_request_pin_locked(self, req_id: str, key: OffloadKey) -> None:
        req_keys = self._request_pins.get(req_id)
        if req_keys is None or key not in req_keys:
            return
        req_keys.remove(key)
        if not req_keys:
            del self._request_pins[req_id]
        self._unpin_key_locked(key)

    def _unpin_key_locked(self, key: OffloadKey) -> None:
        pin_count = self._pin_counts[key] - 1
        if pin_count:
            self._pin_counts[key] = pin_count
        else:
            del self._pin_counts[key]
