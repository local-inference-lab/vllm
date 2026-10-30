# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MultiNodeFileSystemTierManager: file system secondary tier for multi-node
tensor parallelism with node-local disks.

The CPU primary tier is a /dev/shm region per host. With TP across nodes,
each node's workers write their own slots of every chunk row into their own
host's region, so the scheduler (on the head node) sees only the head's
slots. The plain ``fs`` tier would persist and restore only those, leaving
the other nodes' slots empty after a restore.

This tier runs every store and load on every node:
  - the head node does its I/O in the scheduler process, as ``fs`` does;
  - every other node runs an agent thread inside its local-rank-0 worker
    (the process that maps that host's region), which receives the same jobs
    over ZMQ from the scheduler and does the same I/O against its own region
    and its own disk;
  - a job completes when every node has reported, and fails if any node
    failed (for loads, the shortest successful prefix is kept).

Each node persists only the bytes of its local slots, so a node's files hold
1/nnodes of each chunk row. Every node keeps its own size-capped LRU (by
mtime, refreshed on load). The head's files are the lookup index: when a
load fails on any node, the head drops its copies of the failed keys, so the
next lookup misses and the next store rewrites them everywhere.

Configuration (one entry of ``secondary_tiers``):
    {
        "type": "fs_multinode",
        "root_dir": "/kvcache",       # same path on every node
        "nnodes": 4,
        "master_addr": "10.0.0.1",    # the scheduler host, reachable from every node
        "port": 9871,
        "max_bytes": 68719476736,     # per-node cap (optional)
        "min_free_bytes": 42949672960 # keep this much free (optional)
    }
"""

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import zmq
from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.tiering.base import JobId, JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

_IO_ALIGNMENT = 4096


def _local_io_bytes(offloading_spec: "OffloadingSpec", nnodes: int) -> int:
    """Bytes of each chunk row that belong to one node's workers.

    Workers take slot ``device_index % world_size`` of each row, so a node
    with L local GPUs fills slots [0, L): a contiguous prefix of the row.
    Rounded up to the O_DIRECT alignment (within the row); the padding lands
    in slots no local worker reads.
    """
    row_stride = offloading_spec.kv_bytes_per_chunk
    page = offloading_spec.cpu_page_size_per_worker
    if offloading_spec.replicated_layout:
        local_bytes = page
    else:
        world_size = offloading_spec.config.parallel.world_size
        assert world_size % nnodes == 0, (
            f"world_size {world_size} is not divisible by nnodes {nnodes}"
        )
        local_bytes = page * (world_size // nnodes)
    return min(round_up(local_bytes, _IO_ALIGNMENT), row_stride)


def _touch(paths: list[str]) -> None:
    """Refresh mtime so the per-node LRU keeps recently loaded blocks."""
    for path in paths:
        with contextlib.suppress(OSError):
            os.utime(path)


class _DiskJanitor:
    """Keeps one node's tier directory under its byte cap and free-space floor.

    Evicts the oldest files (by mtime) down to 90% of the cap, and removes
    temp files abandoned by a crashed writer.
    """

    def __init__(
        self,
        root_dir: str,
        max_bytes: int | None,
        min_free_bytes: int | None,
        interval_s: float,
    ) -> None:
        self._root_dir = root_dir
        self._max_bytes = max_bytes
        self._min_free_bytes = min_free_bytes
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm_kv_fs_janitor", daemon=True
        )
        if max_bytes is not None or min_free_bytes is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.sweep()
            except Exception:
                logger.exception("KV fs janitor sweep failed at %s", self._root_dir)

    def _scan(self) -> list[tuple[float, int, str]]:
        files: list[tuple[float, int, str]] = []
        now = time.time()
        stack = [self._root_dir]
        while stack:
            directory = stack.pop()
            with contextlib.suppress(FileNotFoundError), os.scandir(directory) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue
                    if not entry.name.endswith((".bin", ".tmp")):
                        continue
                    with contextlib.suppress(FileNotFoundError):
                        st = entry.stat(follow_symlinks=False)
                        if entry.name.endswith(".tmp"):
                            if now - st.st_mtime > 600:
                                os.remove(entry.path)
                            continue
                        files.append((st.st_mtime, st.st_size, entry.path))
        return files

    def sweep(self) -> int:
        """Run one eviction pass; returns the number of bytes removed."""
        files = self._scan()
        total = sum(size for _, size, _ in files)
        excess = 0
        if self._max_bytes is not None and total > self._max_bytes:
            excess = total - int(self._max_bytes * 0.9)
        if self._min_free_bytes is not None:
            st = os.statvfs(self._root_dir)
            free = st.f_bavail * st.f_frsize
            if free < self._min_free_bytes:
                excess = max(excess, self._min_free_bytes - free)
        if excess <= 0:
            return 0
        files.sort()
        removed = 0
        for _, size, path in files:
            if removed >= excess:
                break
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
                removed += size
        logger.info(
            "KV fs janitor at %s evicted %.2f GB (tier held %.2f GB)",
            self._root_dir,
            removed / 1e9,
            total / 1e9,
        )
        return removed


def _submit_io(
    pool: DualQueueThreadPool,
    progress: dict[JobId, int],
    job_id: JobId,
    is_load: bool,
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    io_bytes: int,
    use_o_direct: bool,
) -> None:
    """Queue one store or load on ``pool``; a failed load records how many
    blocks loaded before the failure in ``progress[job_id]``."""
    if not is_load:
        pool.enqueue_store(
            job_id,
            1,
            [lambda: batch_store_block(paths, view, offsets, io_bytes, use_o_direct)],
        )
        return

    def load_task() -> None:
        try:
            batch_load_block(paths, view, offsets, io_bytes, use_o_direct)
        except OSError as exc:
            progress[job_id] = getattr(exc, "num_succeeded", 0)
            raise
        _touch(paths)

    pool.enqueue_load(job_id, 1, [load_task])


@dataclass
class _PendingJob:
    keys: list[OffloadKey]
    is_load: bool
    submitted: float
    # node -> number of blocks that succeeded (len(keys) on success)
    done: dict[int, int] = field(default_factory=dict)
    failed: bool = False
    # False when the failure came from this tier (not ready, timed out), not
    # from a node that tried the I/O: such a failure says nothing about the
    # files, so the head keeps its copies.
    node_reported: bool = True
    transfer_time: float = 0.0


class MultiNodeFileSystemTierManager(FileSystemTierManager):
    """``fs`` tier that mirrors every job on every node's local disk."""

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        nnodes: int,
        master_addr: str,
        port: int,
        max_bytes: int | None = None,
        min_free_bytes: int | None = None,
        janitor_interval_s: float = 30.0,
        job_timeout_s: float = 60.0,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            root_dir=root_dir,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
            enable_kv_events=enable_kv_events,
            locality=locality,
        )
        self._nnodes = int(nnodes)
        self._row_stride: int = self._block_size
        self._io_bytes = _local_io_bytes(offloading_spec, self._nnodes)
        if self._io_bytes % 512 != 0:
            self._use_o_direct = False
        self._job_timeout_s = job_timeout_s
        self._pending: dict[JobId, _PendingJob] = {}
        self._failed_locally: set[JobId] = set()
        self._degraded = False

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://0.0.0.0:{int(port)}")
        self._agents: dict[int, bytes] = {}
        self._root_dir = root_dir

        self._janitor = _DiskJanitor(
            root_dir, max_bytes, min_free_bytes, janitor_interval_s
        )
        logger.info(
            "fs_multinode tier at %s: %d nodes, %d of %d bytes per chunk row "
            "per node, O_DIRECT=%s, waiting for %d agents on port %d",
            root_dir,
            self._nnodes,
            self._io_bytes,
            self._row_stride,
            self._use_o_direct,
            self._nnodes - 1,
            port,
        )

    # ---- worker side ------------------------------------------------------

    @classmethod
    def start_worker_agent(
        cls,
        tier_config: dict[str, Any],
        offloading_spec: "OffloadingSpec",
        mmap_region: "SharedOffloadRegion",
    ) -> "_NodeAgent | None":
        from vllm.distributed.parallel_state import get_world_group

        nnodes = int(tier_config["nnodes"])
        group = get_world_group()
        local_world = group.world_size // nnodes
        node_rank = group.rank // local_world
        if node_rank == 0 or group.local_rank != 0:
            # The head node's I/O runs in the scheduler; other local ranks
            # share their node's region with local rank 0.
            return None
        return _NodeAgent(
            node_rank=node_rank,
            root_dir=tier_config["root_dir"],
            address=f"tcp://{tier_config['master_addr']}:{int(tier_config['port'])}",
            view=mmap_region.create_kv_memoryview(),
            io_bytes=_local_io_bytes(offloading_spec, nnodes),
            max_bytes=tier_config.get("max_bytes"),
            min_free_bytes=tier_config.get("min_free_bytes"),
            janitor_interval_s=float(tier_config.get("janitor_interval_s", 30.0)),
            n_read_threads=int(tier_config.get("n_read_threads", 16)),
            n_write_threads=int(tier_config.get("n_write_threads", 16)),
        )

    # ---- scheduler side ---------------------------------------------------

    def _ready(self) -> bool:
        return not self._degraded and len(self._agents) == self._nnodes - 1

    def _pump(self) -> None:
        """Drain agent messages without blocking."""
        while self._sock.poll(0):
            identity, payload = self._sock.recv_multipart()
            msg = json.loads(payload)
            if msg["op"] == "hello":
                node = int(msg["node"])
                self._agents[node] = identity
                logger.info(
                    "fs_multinode: agent for node %d registered (%d/%d)",
                    node,
                    len(self._agents),
                    self._nnodes - 1,
                )
                continue
            job = self._pending.get(int(msg["job"]))
            if job is None:
                continue
            node = int(msg["node"])
            job.done[node] = int(msg["n"])
            if not msg["ok"]:
                job.failed = True

    def _send(self, job_id: JobId, is_load: bool, paths: list[str], cids) -> None:
        payload = json.dumps(
            {
                "op": "load" if is_load else "store",
                "job": job_id,
                # relative: every node joins them to its own root_dir
                "paths": [os.path.relpath(p, self._root_dir) for p in paths],
                "chunks": [int(c) for c in cids],
            }
        ).encode()
        for identity in self._agents.values():
            self._sock.send_multipart([identity, payload])

    def _submit(self, job_metadata: TransferJob, is_load: bool) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        self._pending[job_id] = _PendingJob(
            keys=keys, is_load=is_load, submitted=time.monotonic()
        )
        if is_load:
            self._load_job_keys[job_id] = keys
        elif self.events is not None:
            self._store_job_keys[job_id] = keys
        if not self._ready():
            # Not every node can take part: fail without touching any disk.
            self._failed_locally.add(job_id)
            return
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(cid) * self._row_stride for cid in job_metadata.chunk_ids]
        self._send(job_id, is_load, paths, job_metadata.chunk_ids)
        _submit_io(
            self._pool,
            self._load_progress,
            job_id,
            is_load,
            paths,
            self._primary_kv_view,
            offsets,
            self._io_bytes,
            self._use_o_direct,
        )

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if not self._ready():
            # Agents register on the first poll; take it now so the first
            # request after a restart can already hit.
            self._pump()
        if not self._ready():
            return LookupResult.MISS
        return super().lookup(key, req_context)

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, is_load=False)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        self._submit(job_metadata, is_load=True)

    @override
    def has_pending_work(self) -> bool:
        return bool(self._pending)

    def _collect(self) -> None:
        for job_id, success, transfer_time in self._pool.get_finished():
            job = self._pending.get(job_id)
            if job is None:
                continue
            n = len(job.keys) if success else self._load_progress.pop(job_id, 0)
            job.done[0] = n
            job.failed |= not success
            job.transfer_time = transfer_time
        for job_id in self._failed_locally:
            job = self._pending.get(job_id)
            if job is not None:
                job.failed = True
                job.node_reported = False
                job.done = {node: 0 for node in range(self._nnodes)}
        self._failed_locally.clear()
        self._pump()

        now = time.monotonic()
        for job_id, job in self._pending.items():
            if len(job.done) < self._nnodes and (
                now - job.submitted > self._job_timeout_s
            ):
                missing = sorted(set(range(self._nnodes)) - set(job.done))
                logger.error(
                    "fs_multinode: job %d timed out waiting for nodes %s; "
                    "disabling the tier for this run",
                    job_id,
                    missing,
                )
                self._degraded = True
                job.failed = True
                job.node_reported = False
                for node in missing:
                    job.done[node] = 0

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        self._collect()
        results = []
        complete = [
            job_id
            for job_id, job in self._pending.items()
            if len(job.done) == self._nnodes
        ]
        for job_id in complete:
            job = self._pending.pop(job_id)
            self._load_job_keys.pop(job_id, None)
            self._load_progress.pop(job_id, None)
            store_keys = self._store_job_keys.pop(job_id, None)
            if not job.is_load:
                if self.events is not None and store_keys and not job.failed:
                    self.events.append(
                        OffloadingEvent(
                            keys=store_keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=not job.failed,
                        transfer_time=job.transfer_time,
                    )
                )
                continue
            if not job.failed:
                results.append(
                    JobResult(
                        job_id=job_id, success=True, transfer_time=job.transfer_time
                    )
                )
                continue
            num_succeeded = min(job.done.values())
            successful = job.keys[:num_succeeded]
            failed = job.keys[num_succeeded:]
            self._lookup_manager.mark_miss(failed)
            if job.node_reported:
                # Drop the head's copies so the index stops offering keys
                # some node cannot serve; the next store rewrites them
                # everywhere.
                for key in failed:
                    with contextlib.suppress(OSError):
                        os.remove(self.file_mapper.get_file_name(key))
            results.append(
                JobResult(
                    job_id=job_id,
                    success=False,
                    successful_keys=tuple(successful) if successful else None,
                    transfer_time=job.transfer_time,
                )
            )
        return results

    @override
    def drain_jobs(self) -> None:
        self._pool.wait_idle()
        deadline = time.monotonic() + self._job_timeout_s
        while any(len(j.done) < self._nnodes for j in self._pending.values()):
            if time.monotonic() > deadline:
                break
            self._sock.poll(10)
            self._collect()

    @override
    def shutdown(self) -> None:
        self._janitor.stop()
        super().shutdown()
        with contextlib.suppress(Exception):
            self._sock.close()


class _NodeAgent:
    """Runs the scheduler's tier jobs against one non-head node's region."""

    def __init__(
        self,
        node_rank: int,
        root_dir: str,
        address: str,
        view: memoryview,
        io_bytes: int,
        max_bytes: int | None,
        min_free_bytes: int | None,
        janitor_interval_s: float,
        n_read_threads: int,
        n_write_threads: int,
    ) -> None:
        self._node_rank = node_rank
        self._root_dir = root_dir
        self._view = view
        assert view.strides is not None
        self._row_stride: int = view.strides[0]
        self._io_bytes = io_bytes
        os.makedirs(root_dir, exist_ok=True)
        self._use_o_direct = probe_o_direct(root_dir) and io_bytes % 512 == 0
        self._pool = DualQueueThreadPool(
            n_read_threads, n_write_threads, thread_name_prefix="vllm_kv_fs_agent"
        )
        self._progress: dict[JobId, int] = {}
        self._sizes: dict[JobId, int] = {}
        self._janitor = _DiskJanitor(
            root_dir, max_bytes, min_free_bytes, janitor_interval_s
        )
        self._address = address
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm_kv_fs_agent", daemon=True
        )
        self._thread.start()
        logger.info(
            "fs_multinode agent for node %d: %d bytes per chunk row, "
            "O_DIRECT=%s, scheduler at %s",
            node_rank,
            io_bytes,
            self._use_o_direct,
            address,
        )

    def stop(self) -> None:
        self._stop.set()
        self._janitor.stop()

    def _run(self) -> None:
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._address)
        sock.send(json.dumps({"op": "hello", "node": self._node_rank}).encode())
        try:
            while not self._stop.is_set():
                if sock.poll(2):
                    self._handle(json.loads(sock.recv()))
                    while sock.poll(0):
                        self._handle(json.loads(sock.recv()))
                for job_id, success, _ in self._pool.get_finished():
                    size = self._sizes.pop(job_id)
                    n = size if success else self._progress.pop(job_id, 0)
                    sock.send(
                        json.dumps(
                            {
                                "op": "done",
                                "job": job_id,
                                "node": self._node_rank,
                                "ok": success,
                                "n": n,
                            }
                        ).encode()
                    )
        finally:
            sock.close()

    def _handle(self, msg: dict) -> None:
        job_id = int(msg["job"])
        paths = [os.path.join(self._root_dir, p) for p in msg["paths"]]
        self._sizes[job_id] = len(paths)
        _submit_io(
            self._pool,
            self._progress,
            job_id,
            msg["op"] == "load",
            paths,
            self._view,
            [int(c) * self._row_stride for c in msg["chunks"]],
            self._io_bytes,
            self._use_o_direct,
        )
