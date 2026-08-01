# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Garbage collection for the fs secondary tier.

The fs tier keeps no index of its own -- the filesystem *is* the index, since
`lookup()` is a plain existence check -- so out of the box it has neither a
capacity limit nor eviction, and grows until the filesystem fills.

`FsGCManager` adds both, using file mtime as the single source of truth for
recency:

  - `touch()` stamps mtime when the scheduler marks a block recently used.
    `TieringOffloadingManager` calls this on every secondary tier, including
    for blocks served from the DRAM primary tier -- which never read this
    tier's files and would otherwise age out while still hot. Reads cannot
    supply this signal on their own: `store_block` sets mtime once at write
    time, reads never update file metadata, and `load_block` opens O_DIRECT so
    a read need not touch the device at all.
  - A background sweep walks the tree, and when the total exceeds `max_bytes`
    unlinks in mtime order until it is back under the low watermark.

Keeping recency on disk rather than in a private dict means the ordering
survives a restart (the fs tier's contents are reusable across runs when
PYTHONHASHSEED is pinned), the accounting cannot drift from reality since every
sweep re-measures it, and external tooling (`du`, `find -newermt`, a separate
reaper) sees the same truth.

Both `touch()` and the sweep run off the scheduler thread. `touch()` is called
once per request per scheduling attempt with *all* of the request's offload keys
(~1.9k keys for a 200k-token context), so stamping inline would cost
milliseconds of syscalls per call and dirty thousands of inodes; instead a key
is stamped at most once per `stamp_interval_s` and the `os.utime` calls happen
on a daemon thread.

Deleting a file that a promotion is about to read is not a correctness problem
-- the failed promotion frees the DRAM block and the tokens get recomputed --
but it wastes work and silently drops the block from disk, so the sweep protects
any key with an in-flight job (`protect`/`release`) and any file whose mtime is
within `grace_s`. Because `stamp_interval_s < grace_s`, every key used within
`grace_s - stamp_interval_s` is guaranteed to have a protected mtime. That
margin is eroded by clock skew wherever mtimes are not set by this host's
clock -- on NFS or a network PVC the server stamps them while the sweep cutoff
comes from local `time.time()` -- so on shared storage keep it well above the
sweep duration rather than tightening the two intervals toward each other.

Known limitation: sweeps do not emit KV events. When the owning tier publishes
BlockStored events (enable_kv_events), a sweep's unlinks have no removed=True
counterpart, so an external consumer of the event stream can believe evicted
blocks still exist. Emitting removals would require reconstructing OffloadKeys
from the swept paths and handing them across threads to the scheduler-owned
events list -- and the stream would still be incomplete, because failed-load
unlinks, peer instances sharing root_dir, and external reapers also remove
files without events. MEDIUM_FS presence events are best-effort by design;
consumers must tolerate a BlockStored block that is no longer there. This
instance's own scheduler already does: a failed load invalidates its cached
lookup and the tokens are recomputed.
"""

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Collection, Iterator

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.file_mapper import FileMapper

logger = init_logger(__name__)

# Bound the wait so shutdown() stays responsive when no work arrives.
_WAKE_TIMEOUT_S = 1.0

# Only block files are candidates. In particular `store_block` writes to
# "<dest>.tmp" before os.replace, and config.json must never be removed. A
# writer killed mid-store therefore leaks a .tmp that no sweep counts or
# reaps; the window is one block write wide, so this is bounded in practice.
_BLOCK_SUFFIX = ".bin"


class FsGCManager:
    """Bounds the fs tier's on-disk size, evicting least-recently-used blocks."""

    def __init__(
        self,
        file_mapper: FileMapper,
        max_bytes: int,
        low_watermark: float = 0.9,
        interval_s: float = 60.0,
        stamp_interval_s: float = 60.0,
        grace_s: float = 300.0,
        max_tracked: int = 200_000,
    ) -> None:
        if not 0.0 < low_watermark <= 1.0:
            raise ValueError(f"low_watermark must be in (0, 1], got {low_watermark}")
        if grace_s <= stamp_interval_s:
            raise ValueError(
                f"grace_s ({grace_s}) must exceed stamp_interval_s "
                f"({stamp_interval_s}), otherwise a block in active use can "
                f"have an mtime old enough to be swept"
            )

        self.file_mapper = file_mapper
        self.max_bytes = max_bytes
        self.target_bytes = int(max_bytes * low_watermark)
        self.interval_s = interval_s
        self.stamp_interval_s = stamp_interval_s
        self.grace_s = grace_s
        self.max_tracked = max_tracked

        # Blocks live under a per-rank sibling of the digest directory.
        self.block_root = f"{file_mapper.base_path}_r{file_mapper.rank}"

        # key -> monotonic time of last stamp, in LRU order so it can be
        # trimmed. Losing an entry only costs one redundant utime later.
        self._last_stamped: OrderedDict[OffloadKey, float] = OrderedDict()
        # Dedup set of keys awaiting a stamp; dict preserves insertion order.
        self._pending_stamps: dict[OffloadKey, None] = {}
        # Keys with in-flight store/load jobs, which must not be unlinked.
        self._protected: dict[OffloadKey, int] = {}

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._stamped = 0
        self._stamp_errors = 0

        self._thread = threading.Thread(
            target=self._run, name="vllm_kv_fs_gc", daemon=True
        )
        self._thread.start()
        logger.info(
            "fs tier GC enabled: cap %.1f GiB, sweeping to %.1f GiB every %gs "
            "(mtime tracks last use, so eviction is LRU and survives restarts)",
            max_bytes / 2**30,
            self.target_bytes / 2**30,
            interval_s,
        )

    # ------------------------------------------------------------------
    # Scheduler-thread entry points
    # ------------------------------------------------------------------

    def touch(self, keys: Collection[OffloadKey]) -> None:
        """Record keys as recently used. Cheap: one dict lookup per key."""
        now = time.monotonic()
        with self._lock:
            if self._stopping:
                return
            for key in keys:
                last = self._last_stamped.get(key)
                if last is not None and now - last < self.stamp_interval_s:
                    continue
                self._last_stamped[key] = now
                self._last_stamped.move_to_end(key)
                self._pending_stamps[key] = None
            while len(self._last_stamped) > self.max_tracked:
                self._last_stamped.popitem(last=False)
            has_work = bool(self._pending_stamps)
        if has_work:
            self._wake.set()

    def protect(self, keys: Collection[OffloadKey]) -> None:
        """Pin keys with an in-flight job so the sweep cannot unlink them."""
        with self._lock:
            for key in keys:
                self._protected[key] = self._protected.get(key, 0) + 1

    def release(self, keys: Collection[OffloadKey]) -> None:
        """Undo one `protect` for each key."""
        with self._lock:
            for key in keys:
                count = self._protected.get(key)
                if count is None:
                    continue
                if count <= 1:
                    del self._protected[key]
                else:
                    self._protected[key] = count - 1

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        next_sweep = time.monotonic() + self.interval_s
        while True:
            self._wake.wait(_WAKE_TIMEOUT_S)
            self._wake.clear()

            with self._lock:
                stopping = self._stopping
                batch = list(self._pending_stamps)
                self._pending_stamps.clear()
            try:
                self._stamp(batch)
            except Exception:
                # _stamp handles the expected per-key OSError itself; anything
                # else escaping here would kill this daemon thread and
                # silently un-bound the tier again, with no signal beyond the
                # absence of sweep logs.
                logger.exception("fs tier GC stamping failed")

            if stopping:
                return
            now = time.monotonic()
            if now >= next_sweep:
                next_sweep = now + self.interval_s
                try:
                    self.sweep()
                except Exception:
                    logger.exception("fs tier GC sweep failed")

    def _stamp(self, keys: list[OffloadKey]) -> None:
        for key in keys:
            # A key legitimately may have no file: the block can live only in
            # the DRAM primary tier, or its cascade may have failed.
            try:
                os.utime(self.file_mapper.get_file_name(key), None)
                self._stamped += 1
            except OSError:
                self._stamp_errors += 1

    def _iter_blocks(self, path: str) -> Iterator[tuple[str, int, float]]:
        """Yield (path, size, mtime) for every block file under `path`."""
        try:
            entries = list(os.scandir(path))
        except FileNotFoundError:
            return
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from self._iter_blocks(entry.path)
                elif entry.name.endswith(_BLOCK_SUFFIX):
                    stat = entry.stat(follow_symlinks=False)
                    yield entry.path, stat.st_size, stat.st_mtime
            except OSError:
                # Raced with a store or another sweep; skip it.
                continue

    def sweep(self) -> int:
        """Unlink LRU blocks until under the low watermark. Returns bytes freed."""
        started = time.monotonic()
        blocks = list(self._iter_blocks(self.block_root))
        total = sum(size for _, size, _ in blocks)
        if total <= self.max_bytes:
            logger.debug(
                "fs tier GC: %.2f GiB in %d blocks, under the %.2f GiB cap",
                total / 2**30,
                len(blocks),
                self.max_bytes / 2**30,
            )
            return 0

        with self._lock:
            protected_paths = {
                self.file_mapper.get_file_name(key) for key in self._protected
            }
        cutoff = time.time() - self.grace_s

        blocks.sort(key=lambda block: block[2])
        freed = 0
        removed = 0
        in_flight = 0
        within_grace = 0
        for path, size, mtime in blocks:
            if total - freed <= self.target_bytes:
                break
            # Check protection before recency: a block with an in-flight job must
            # survive however old its mtime is, and the two reasons are worth
            # telling apart. in_flight > 0 means the grace window alone was not
            # enough and protect() is doing real work; within_grace > 0 means the
            # working set itself is at least as large as the cap.
            if path in protected_paths:
                in_flight += 1
                continue
            if mtime > cutoff:
                within_grace += 1
                continue
            try:
                os.unlink(path)
            except OSError:
                continue
            freed += size
            removed += 1

        logger.info(
            "fs tier GC: %.2f GiB over the %.2f GiB cap, freed %.2f GiB in %d "
            "blocks, kept %d with jobs in flight and %d used within the %gs "
            "grace window, now %.2f GiB, took %.2fs",
            total / 2**30,
            self.max_bytes / 2**30,
            freed / 2**30,
            removed,
            in_flight,
            within_grace,
            self.grace_s,
            (total - freed) / 2**30,
            time.monotonic() - started,
        )
        return freed

    def shutdown(self) -> None:
        """Stop the sweep thread. Best-effort, and nothing depends on it.

        The engine reaches this through scheduler -> connector -> tier only if
        it gets that far: EngineCore.shutdown() tears the executor down first,
        and the API server's SIGTERM path force-kills the engine core with
        timeout=0s, so in practice this often does not run at all. That is
        safe -- the thread is a daemon, mtimes are already committed on disk
        and sweeps are idempotent -- but it means queued stamps can be dropped
        (costing at most `stamp_interval_s` of recency) and the summary below
        should not be relied on as a shutdown signal.
        """
        with self._lock:
            self._stopping = True
        self._wake.set()
        self._thread.join(timeout=5.0)
        logger.info(
            "fs tier GC stopped (%d mtime stamps, %d failed)",
            self._stamped,
            self._stamp_errors,
        )
