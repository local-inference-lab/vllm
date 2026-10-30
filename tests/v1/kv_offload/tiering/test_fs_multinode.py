# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fs_multinode tier: a scheduler (node 0) and one agent (node 1) in one
process, each with its own primary region and disk directory."""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.fs.multinode import (
    MultiNodeFileSystemTierManager,
    _DiskJanitor,
    _NodeAgent,
)

NUM_CHUNKS = 8
PAGE = 8192  # bytes per worker slot
WORLD = 2  # one worker per node
ROW = PAGE * WORLD


def _spec():
    parallel = SimpleNamespace(
        rank=0,
        world_size=WORLD,
        tp_size=WORLD,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        is_parallelism_agnostic=False,
    )
    config = SimpleNamespace(
        groups=(SimpleNamespace(tokens_per_block=16, layer_names=("l0",)),),
        model=SimpleNamespace(name="test/model", dtype="bfloat16"),
        cache=SimpleNamespace(tokens_per_hash=16),
        parallel=parallel,
        canonical_layout=False,
        replicated_layout=False,
        kv_cache_layout=None,
    )
    return SimpleNamespace(
        config=config,
        blocks_per_chunk=1,
        kv_bytes_per_chunk=ROW,
        cpu_page_size_per_worker=PAGE,
        replicated_layout=False,
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
    )


def _region():
    # mmap-backed so the buffer is page aligned (O_DIRECT)
    import mmap

    buf = mmap.mmap(-1, NUM_CHUNKS * ROW)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(NUM_CHUNKS, ROW)
    return buf, arr, memoryview(arr)


def _wait(tier, n, timeout=10.0):
    results = []
    deadline = time.monotonic() + timeout
    while len(results) < n and time.monotonic() < deadline:
        results.extend(tier.get_finished_jobs())
        time.sleep(0.01)
    assert len(results) == n, results
    return results


@pytest.fixture
def unused_tcp_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cluster(tmp_path, unused_tcp_port):
    spec = _spec()
    _, head_arr, head_view = _region()
    _, node_arr, node_view = _region()
    head_dir, node_dir = str(tmp_path / "head"), str(tmp_path / "node1")
    tier = MultiNodeFileSystemTierManager(
        spec,
        head_view,
        "fs_multinode",
        root_dir=head_dir,
        nnodes=2,
        master_addr="127.0.0.1",
        port=unused_tcp_port,
        n_read_threads=2,
        n_write_threads=2,
    )
    agent = _NodeAgent(
        node_rank=1,
        root_dir=node_dir,
        address=f"tcp://127.0.0.1:{unused_tcp_port}",
        view=node_view,
        io_bytes=PAGE,
        max_bytes=None,
        min_free_bytes=None,
        janitor_interval_s=30.0,
        n_read_threads=2,
        n_write_threads=2,
    )
    deadline = time.monotonic() + 5
    while not tier._ready() and time.monotonic() < deadline:
        tier._pump()
        time.sleep(0.01)
    assert tier._ready()
    yield SimpleNamespace(
        tier=tier,
        agent=agent,
        head=head_arr,
        node=node_arr,
        head_dir=head_dir,
        node_dir=node_dir,
    )
    agent.stop()
    tier.shutdown()


def _keys(n, salt=0):
    return [make_offload_key(bytes([salt, i]) * 16, 0) for i in range(n)]


def _job(job_id, keys, chunks):
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunks),
        is_promotion=False,
        req_context=ReqContext(req_id="r"),
    )


def _files(root):
    out = []
    for d, _, names in os.walk(root):
        out += [os.path.join(d, n) for n in names if n.endswith(".bin")]
    return out


def test_round_trip_keeps_each_nodes_slot(cluster):
    rng = np.random.default_rng(0)
    # Each node fills only its own slot (slot 0 on both, by device index).
    cluster.head[:, :PAGE] = rng.integers(0, 255, (NUM_CHUNKS, PAGE))
    cluster.node[:, :PAGE] = rng.integers(0, 255, (NUM_CHUNKS, PAGE))
    head_before = cluster.head.copy()
    node_before = cluster.node.copy()

    keys = _keys(3)
    cluster.tier.submit_store(_job(1, keys, [0, 1, 2]))
    (res,) = _wait(cluster.tier, 1)
    assert res.success
    for root in (cluster.head_dir, cluster.node_dir):
        files = _files(root)
        assert len(files) == 3
        assert all(os.path.getsize(f) == PAGE for f in files)

    # Load into different slots after wiping both regions.
    cluster.head[:] = 0
    cluster.node[:] = 0
    cluster.tier.submit_load(_job(2, keys, [5, 6, 7]))
    (res,) = _wait(cluster.tier, 1)
    assert res.success
    for src, dst in zip([0, 1, 2], [5, 6, 7]):
        assert (cluster.head[dst, :PAGE] == head_before[src, :PAGE]).all()
        assert (cluster.node[dst, :PAGE] == node_before[src, :PAGE]).all()


def test_remote_miss_fails_suffix_and_drops_head_copy(cluster):
    keys = _keys(3, salt=1)
    cluster.tier.submit_store(_job(1, keys, [0, 1, 2]))
    assert _wait(cluster.tier, 1)[0].success

    # Node 1 lost the second block (e.g. its janitor evicted it).
    victim = cluster.tier.file_mapper.get_file_name(keys[1])
    os.remove(victim.replace(cluster.head_dir, cluster.node_dir))

    cluster.tier.submit_load(_job(2, keys, [3, 4, 5]))
    (res,) = _wait(cluster.tier, 1)
    assert not res.success
    assert list(res.successful_keys) == keys[:1]
    # Head drops the keys that failed, so they stop being offered.
    assert os.path.exists(cluster.tier.file_mapper.get_file_name(keys[0]))
    assert not os.path.exists(cluster.tier.file_mapper.get_file_name(keys[1]))
    assert not os.path.exists(cluster.tier.file_mapper.get_file_name(keys[2]))


def test_not_ready_misses_and_fails_without_io(tmp_path, unused_tcp_port):
    _, _, view = _region()
    tier = MultiNodeFileSystemTierManager(
        _spec(),
        view,
        "fs_multinode",
        root_dir=str(tmp_path),
        nnodes=2,
        master_addr="127.0.0.1",
        port=unused_tcp_port,
        n_read_threads=1,
        n_write_threads=1,
    )
    try:
        key = _keys(1)[0]
        assert tier.lookup(key, ReqContext(req_id="r")).name == "MISS"
        tier.submit_store(_job(1, [key], [0]))
        (res,) = _wait(tier, 1)
        assert not res.success
        assert _files(str(tmp_path)) == []
    finally:
        tier.shutdown()


def test_janitor_evicts_oldest_to_cap(tmp_path):
    for i in range(10):
        p = tmp_path / "a" / f"{i}.bin"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x" * 1000)
        os.utime(p, (1000 + i, 1000 + i))
    janitor = _DiskJanitor(str(tmp_path), 5000, None, 3600)
    janitor.stop()
    removed = janitor.sweep()
    left = sorted(int(f.stem) for f in (tmp_path / "a").iterdir())
    # down to 90% of the cap: oldest six go
    assert removed == 6000
    assert left == [6, 7, 8, 9]
