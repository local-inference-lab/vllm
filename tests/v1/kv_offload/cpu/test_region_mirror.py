# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot mirroring between a remote worker's offload region and the head's."""

import json
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import zmq

from vllm.v1.kv_offload.base import TransferResult
from vllm.v1.kv_offload.cpu import region_mirror
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.region_mirror import (
    MirrorTopology,
    RegionMirrorClient,
    RegionMirrorServer,
    RegionMirrorWorker,
    SlotLayout,
    attach_region_mirror,
    multi_node_topology,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

NUM_CHUNKS = 4
SLOT = 4096
WORLD_SIZE = 3
REMOTE_RANK = 2
TOKEN = "secret-token"
REMOTES = frozenset({REMOTE_RANK})
ALLOWED = frozenset({"127.0.0.1"})
HEAD = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=SLOT * WORLD_SIZE, size=SLOT)
LOCAL = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=SLOT, size=SLOT)


def _region(layout: SlotLayout) -> SharedOffloadRegion:
    return SharedOffloadRegion(
        engine_id=f"test-mirror-{uuid.uuid4().hex}",
        num_chunks=NUM_CHUNKS,
        rank=None,
        kv_bytes_per_chunk=layout.row_stride,
        cpu_page_size=SLOT,
    )


def _head_slot(rows: np.ndarray, chunk: int, rank: int) -> np.ndarray:
    return rows[chunk, rank * SLOT : (rank + 1) * SLOT]


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture
def regions():
    head, remote = _region(HEAD), _region(LOCAL)
    yield head, remote
    head.cleanup()
    remote.cleanup()


@pytest.fixture
def mirror(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, server.address, TOKEN)
    yield HEAD.rows(head), LOCAL.rows(remote), client, server
    client.shutdown()
    server.shutdown()


def _raw_request(
    address: str, frames: list[bytes], *, source: str | None = None, wait_ms=5000
) -> list[bytes] | None:
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(
        address
        if source is None
        else f"tcp://{source}:0;{address.removeprefix('tcp://')}"
    )
    try:
        socket.send_multipart(frames)
        if not socket.poll(wait_ms):
            return None
        return socket.recv_multipart()
    finally:
        socket.close()
        context.term()


def _header(**fields) -> bytes:
    request = {
        "id": "r1",
        "token": TOKEN,
        "seq": 10**9,
        "epoch": "e",
        "op": "put",
        "rank": REMOTE_RANK,
        "chunks": [0],
    }
    request.update(fields)
    return json.dumps(request).encode()


def test_push_writes_only_the_remote_slot(mirror):
    head_rows, local_rows, client, _ = mirror
    local_rows[:] = 7
    client.push(np.array([1, 3]))

    for chunk in range(NUM_CHUNKS):
        expected = 7 if chunk in (1, 3) else 0
        assert (_head_slot(head_rows, chunk, REMOTE_RANK) == expected).all()
        assert (_head_slot(head_rows, chunk, 0) == 0).all()


def test_pull_reads_the_remote_slot(mirror):
    head_rows, local_rows, client, _ = mirror
    _head_slot(head_rows, 2, REMOTE_RANK)[:] = 5
    client.pull(np.array([2]))

    assert (local_rows[2] == 5).all()
    assert (local_rows[1] == 0).all()


@pytest.mark.parametrize(
    "frames",
    [
        [b"not json"],
        [_header(token="wrong"), b"\0" * SLOT],
        [_header(op="delete"), b"\0" * SLOT],
        [_header(rank=0), b"\0" * SLOT],
        [_header(rank=True), b"\0" * SLOT],
        [_header(rank=WORLD_SIZE), b"\0" * SLOT],
        [_header(seq="1"), b"\0" * SLOT],
        [_header(chunks=[-1]), b"\0" * SLOT],
        [_header(chunks=[1.0]), b"\0" * SLOT],
        [_header(chunks=[NUM_CHUNKS]), b"\0" * SLOT],
        [_header(chunks=[]), b""],
        [_header(), b"\0" * (SLOT - 1)],
        [_header(op="get", chunks=list(range(NUM_CHUNKS)) * 3000)],
    ],
)
def test_server_rejects_invalid_requests_and_keeps_serving(mirror, frames):
    head_rows, local_rows, client, server = mirror

    reply = _raw_request(server.address, frames)

    assert reply[-1] == b"error"
    assert not head_rows.any()
    local_rows[:] = 3
    client.push(np.array([0]))
    assert (_head_slot(head_rows, 0, REMOTE_RANK) == 3).all()


def test_server_rejects_a_put_older_than_one_applied(mirror):
    head_rows, _, _, server = mirror
    assert _raw_request(server.address, [_header(seq=5), b"\1" * SLOT])[-1] == b"ok"

    reply = _raw_request(server.address, [_header(seq=4), b"\2" * SLOT])

    assert reply[-1] == b"error"
    assert (_head_slot(head_rows, 0, REMOTE_RANK) == 1).all()


def test_ping_checks_the_token(mirror, regions):
    _, _, client, server = mirror
    client.ping()
    _, remote = regions
    impostor = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, server.address, "wrong", attempts=1
    )
    try:
        with pytest.raises(RuntimeError, match="rejected"):
            impostor.ping()
    finally:
        impostor.shutdown()


def test_large_copies_are_split_into_bounded_requests(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, server.address, TOKEN, max_request_bytes=SLOT
    )
    sent: list[int] = []
    request = client._request
    client._request = lambda op, chunk_ids, *payload: (
        sent.append(len(chunk_ids)) or request(op, chunk_ids, *payload)
    )
    head_rows, local_rows = HEAD.rows(head), LOCAL.rows(remote)
    try:
        local_rows[:] = 6
        client.push(np.arange(NUM_CHUNKS))
        head_rows[:, REMOTE_RANK * SLOT :] = 8
        client.pull(np.arange(NUM_CHUNKS))
    finally:
        client.shutdown()
        server.shutdown()

    assert sent == [1] * (2 * NUM_CHUNKS)
    assert (local_rows == 8).all()
    del head_rows, local_rows


class _ScriptedHead:
    """ROUTER that answers requests as scripted, on a background thread."""

    def __init__(self, script):
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        port = self.router.bind_to_random_port("tcp://127.0.0.1")
        self.address = f"tcp://127.0.0.1:{port}"
        self.thread = threading.Thread(target=script, args=(self.router,))
        self.thread.start()

    def close(self):
        self.thread.join()
        self.router.close()
        self.context.term()


def test_client_drops_replies_to_other_requests(regions):
    _, remote = regions

    def script(router):
        identity, header, _ = router.recv_multipart()
        request_id = json.loads(header)["id"].encode()
        router.send_multipart([identity, b"stale-id", b"error"])
        router.send_multipart([identity, request_id, b"ok"])

    head = _ScriptedHead(script)
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, head.address, TOKEN)
    try:
        client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()


def test_client_retries_on_a_fresh_socket(regions):
    _, remote = regions
    seen: list[tuple[bytes, int]] = []

    def script(router):
        for answer in (False, True):
            identity, header, *_ = router.recv_multipart()
            request = json.loads(header)
            seen.append((identity, request["seq"]))
            if answer:
                router.send_multipart([identity, request["id"].encode(), b"ok"])

    head = _ScriptedHead(script)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=200
    )
    try:
        client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()

    (first_socket, first_seq), (second_socket, second_seq) = seen
    assert first_socket != second_socket
    assert second_seq > first_seq


def test_client_gives_up_after_its_attempts(regions):
    _, remote = regions
    head = _ScriptedHead(lambda router: None)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=100, attempts=2
    )
    try:
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()


def _inner() -> MagicMock:
    inner = MagicMock()
    inner.submit_store.return_value = True
    inner.submit_load.return_value = True
    inner.get_finished.return_value = []
    return inner


def test_store_is_reported_only_after_its_push(mirror):
    head_rows, local_rows, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)

    assert worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    local_rows[0] = 9
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    first = worker.get_finished()
    inner.get_finished.return_value = []

    results = first + _wait_for(worker.get_finished)
    assert results == [TransferResult(job_id=1, success=True)]
    assert (_head_slot(head_rows, 0, REMOTE_RANK) == 9).all()
    worker.shutdown()


def test_load_dma_starts_only_after_its_pull(mirror):
    head_rows, local_rows, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    _head_slot(head_rows, 3, REMOTE_RANK)[:] = 4
    src, dst = CPULoadStoreSpec([3]), MagicMock()

    assert worker.submit_load(2, src, dst)
    _wait_for(lambda: worker.get_finished() is not None and inner.submit_load.called)

    inner.submit_load.assert_called_once_with(2, src, dst)
    assert (local_rows[3] == 4).all()
    worker.shutdown()


def test_wait_submits_a_pending_load_before_waiting(mirror):
    _, _, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    src, dst = CPULoadStoreSpec([1]), MagicMock()

    worker.submit_load(5, src, dst)
    worker.wait({5})

    inner.submit_load.assert_called_once_with(5, src, dst)
    inner.wait.assert_called_once_with({5})
    worker.shutdown()


def test_failed_push_reports_a_failed_store():
    inner = _inner()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = RuntimeError("head unreachable")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    assert worker.get_finished() == []
    inner.get_finished.return_value = []

    results = _wait_for(worker.get_finished)
    worker.shutdown()

    assert [(r.job_id, r.success) for r in results] == [(1, False)]


def test_a_failed_push_is_fenced_before_it_is_reported():
    inner = _inner()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = RuntimeError("head slow")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []
    _wait_for(worker.get_finished)
    worker.shutdown()

    client.fence.assert_called_once()
    # Only the shutdown stops the client.
    assert client.stop.call_args_list == [()]


def test_an_unfenced_failed_push_stops_the_mirror():
    inner = _inner()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = RuntimeError("head unreachable")
    client.fence.side_effect = RuntimeError("head unreachable")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []
    _wait_for(worker.get_finished)
    worker.shutdown()

    assert "fenced" in client.stop.call_args_list[0].args[0]


def test_a_fence_rejects_older_puts_of_its_rank(mirror):
    head_rows, _, _, server = mirror
    fence = [_header(op="fence", chunks=[], seq=10, epoch="e1")]
    assert _raw_request(server.address, fence)[-1] == b"ok"

    late = [_header(seq=9, epoch="e1"), b"\1" * SLOT]

    assert _raw_request(server.address, late)[-1] == b"error"
    assert not head_rows.any()


def test_failed_pull_reports_a_failed_load_without_its_dma():
    inner = _inner()
    inner.get_finished.return_value = []
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.pull.side_effect = RuntimeError("head unreachable")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_load(2, CPULoadStoreSpec([0]), MagicMock())
    results = _wait_for(worker.get_finished)
    worker.shutdown()

    assert [(r.job_id, r.success) for r in results] == [(2, False)]
    inner.submit_load.assert_not_called()


def test_shutdown_is_prompt_when_the_head_is_gone(regions):
    _, remote = regions
    head = _ScriptedHead(lambda router: None)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=30_000
    )
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    for job_id in range(5):
        worker.submit_store(job_id, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [
        TransferResult(job_id=j, success=True) for j in range(5)
    ]
    worker.get_finished()

    start = time.monotonic()
    worker.shutdown()

    assert time.monotonic() - start < 5
    inner.shutdown.assert_called_once()
    head.close()


def test_shutdown_releases_mirror_views_before_the_region_closes(monkeypatch):
    from vllm.v1.kv_offload.cpu import shared_offload_region

    warnings: list[str] = []
    monkeypatch.setattr(
        shared_offload_region.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg),
    )
    head, remote = _region(HEAD), _region(LOCAL)
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, server.address, TOKEN)
    head_inner, remote_inner = _inner(), _inner()
    head_inner.shutdown.side_effect = head.cleanup
    remote_inner.shutdown.side_effect = remote.cleanup

    RegionMirrorWorker(remote_inner, client=client).shutdown()
    RegionMirrorWorker(head_inner, server=server).shutdown()

    # A live mirror view would make mmap.close() fail with BufferError.
    assert warnings == []
    assert head.mmap_obj is None and remote.mmap_obj is None


class _Exchange:
    """In-process stand-in for the group all-gather between two ranks."""

    def __init__(self, world_size: int):
        self.barrier = threading.Barrier(world_size)
        self.slots: list = [None] * world_size

    def all_gather(self, topology, value):
        self.slots[topology.rank] = value
        self.barrier.wait(timeout=10)
        values = list(self.slots)
        self.barrier.wait(timeout=10)
        return values


def _run_ranks(
    monkeypatch,
    regions,
    *,
    server_error: Exception | None = None,
    host_ip: str | None = "127.0.0.1",
):
    if host_ip is None:
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    else:
        monkeypatch.setenv("VLLM_HOST_IP", host_ip)
    exchange = _Exchange(2)
    monkeypatch.setattr(region_mirror, "_all_gather", exchange.all_gather)
    monkeypatch.setattr(
        "vllm.utils.network_utils.get_ip",
        lambda: host_ip or "127.0.0.1",
        raising=True,
    )
    if server_error is not None:
        monkeypatch.setattr(
            region_mirror,
            "RegionMirrorServer",
            MagicMock(side_effect=server_error),
        )
    head, remote = regions
    outcomes: dict[int, object] = {}

    def rank(index: int, region: SharedOffloadRegion, layout: SlotLayout):
        topology = MirrorTopology(
            group=None, rank=index, on_head=index == 0, remote_ranks=frozenset({1})
        )
        try:
            outcomes[index] = attach_region_mirror(
                _inner(), region, HEAD, layout, topology
            )
        except Exception as e:
            outcomes[index] = e

    threads = [
        threading.Thread(target=rank, args=(0, head, HEAD)),
        threading.Thread(target=rank, args=(1, remote, LOCAL)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return outcomes


def test_attach_connects_remote_ranks_through_the_head(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions)

    head_worker, remote_worker = outcomes[0], outcomes[1]
    assert isinstance(head_worker, RegionMirrorWorker)
    assert isinstance(remote_worker, RegionMirrorWorker)
    assert remote_worker._client is not None
    remote_worker.shutdown()
    head_worker.shutdown()
    _, remote = regions
    # The remote region's name is released once every rank mapped it.
    assert not remote.created_file


def test_attach_fails_on_every_rank_when_the_head_cannot_serve(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions, server_error=OSError("bind failed"))

    assert isinstance(outcomes[0], OSError)
    assert isinstance(outcomes[1], RuntimeError)
    assert "mirror server" in str(outcomes[1])


def _group(world_size: int, rank: int = 0) -> SimpleNamespace:
    from torch.distributed import ProcessGroup

    return SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        cpu_group=MagicMock(spec=ProcessGroup),
    )


def _patch_groups(monkeypatch, *, inner, world, same_node):
    """same_node: a list for source rank 0, or a callable of the source rank."""
    from vllm.distributed import parallel_state

    def missing():
        raise AssertionError("not initialized")

    monkeypatch.setattr(
        parallel_state,
        "get_inner_dp_world_group",
        (lambda: inner) if inner is not None else missing,
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda group, source_rank: (
            same_node(source_rank) if callable(same_node) else same_node
        ),
    )


def test_topology_spans_the_engine_group(monkeypatch):
    _patch_groups(
        monkeypatch, inner=None, world=_group(3, rank=2), same_node=[True, False, False]
    )

    topology, spans_nodes = multi_node_topology(3)

    assert topology is not None and spans_nodes
    assert (topology.rank, topology.on_head) == (2, False)
    assert topology.remote_ranks == {1, 2}


def test_topology_ignores_single_node_data_parallel_replicas(monkeypatch):
    # The world group holds two single-node replicas of three workers each.
    _patch_groups(
        monkeypatch,
        inner=None,
        world=_group(6, rank=4),
        same_node=lambda source: [r // 3 == source // 3 for r in range(6)],
    )

    assert multi_node_topology(3) == (None, False)


def test_topology_reports_data_parallel_replicas_spanning_nodes(monkeypatch):
    # Replica 1 (ranks 3-5) puts rank 5 on another node; no inner group exists.
    nodes = [0, 0, 0, 1, 1, 2]
    _patch_groups(
        monkeypatch,
        inner=None,
        world=_group(6, rank=4),
        same_node=lambda source: [nodes[r] == nodes[source] for r in range(6)],
    )

    assert multi_node_topology(3) == (None, True)


def test_topology_is_none_on_one_node(monkeypatch):
    _patch_groups(monkeypatch, inner=None, world=_group(3), same_node=[True] * 3)

    assert multi_node_topology(3) == (None, False)


def test_wait_does_not_wait_for_pushes():
    gate = threading.Event()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = lambda chunk_ids: gate.wait(10)
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([2]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    assert worker.get_finished() == []  # the push is now in flight

    start = time.monotonic()
    worker.wait({1})

    assert time.monotonic() - start < 5
    inner.wait.assert_called_once_with({1})
    gate.set()
    worker.shutdown()


def test_loads_overtake_a_store_between_its_requests():
    order: list[str] = []
    started = threading.Event()
    gate = threading.Event()

    def push(chunk_ids):
        order.append("store")
        started.set()
        gate.wait(10)

    client = MagicMock(chunks_per_request=1, stopped=False)
    client.push.side_effect = push
    client.pull.side_effect = lambda chunk_ids: order.append("load")
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(0, MagicMock(), CPULoadStoreSpec([0, 1, 2]))
    inner.get_finished.return_value = [TransferResult(job_id=0, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []
    assert started.wait(10)
    worker.submit_load(1, CPULoadStoreSpec([3]), MagicMock())

    gate.set()
    _wait_for(lambda: len(order) == 4)
    worker.shutdown()

    assert order == ["store", "load", "store", "store"]


def test_waiting_stores_run_between_bursts_of_loads():
    order: list[str] = []
    gate = threading.Event()

    def pull(chunk_ids):
        gate.wait(10)
        order.append("load")

    client = MagicMock(chunks_per_request=64, stopped=False)
    client.pull.side_effect = pull
    client.push.side_effect = lambda chunk_ids: order.append("store")
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(0, MagicMock(), CPULoadStoreSpec([0]))
    for job_id in range(1, 8):
        worker.submit_load(job_id, CPULoadStoreSpec([job_id % NUM_CHUNKS]), MagicMock())
    inner.get_finished.return_value = [TransferResult(job_id=0, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []

    gate.set()
    _wait_for(lambda: len(order) == 8)
    worker.shutdown()

    # The first load was already running when the store was queued.
    assert order == ["load"] * 4 + ["store"] + ["load"] * 3


def test_slots_larger_than_the_request_limit_still_move(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head,
        HEAD,
        "127.0.0.1",
        TOKEN,
        REMOTES,
        allowed_hosts=ALLOWED,
        max_request_bytes=SLOT // 4,
    )
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, server.address, TOKEN, max_request_bytes=SLOT // 4
    )
    head_rows, local_rows = HEAD.rows(head), LOCAL.rows(remote)
    try:
        local_rows[:] = 12
        client.push(np.arange(NUM_CHUNKS))
    finally:
        client.shutdown()
        server.shutdown()

    assert (head_rows[:, REMOTE_RANK * SLOT :] == 12).all()
    del head_rows, local_rows


def test_a_restarted_client_is_not_taken_for_a_stale_one(mirror):
    # A new epoch restarts the sequence; within an epoch, older puts stay out.
    head_rows, _, _, server = mirror
    put = [_header(seq=5, epoch="old"), b"\1" * SLOT]
    assert _raw_request(server.address, put)[-1] == b"ok"
    stale = [_header(seq=4, epoch="old"), b"\2" * SLOT]
    assert _raw_request(server.address, stale)[-1] == b"error"

    restarted = [_header(seq=0, epoch="new"), b"\3" * SLOT]

    assert _raw_request(server.address, restarted)[-1] == b"ok"
    assert (head_rows[0, REMOTE_RANK * SLOT :] == 3).all()


def test_server_rejects_slots_of_head_node_ranks(mirror):
    head_rows, _, _, server = mirror

    reply = _raw_request(server.address, [_header(rank=1), b"\1" * SLOT])

    assert reply[-1] == b"error"
    assert not head_rows.any()


@pytest.mark.skipif(sys.platform != "linux", reason="binds a second loopback address")
def test_server_admits_only_the_remote_workers_hosts(mirror):
    # A host outside the allowlist cannot even queue a valid request.
    _, _, _, server = mirror
    ping = [_header(op="ping", chunks=[])]

    assert _raw_request(server.address, ping, source="127.0.0.2", wait_ms=1000) is None
    reply = _raw_request(server.address, ping, source="127.0.0.1")
    assert reply is not None and reply[-1] == b"ok"


def test_server_rejects_messages_with_extra_frames(mirror):
    head_rows, _, _, server = mirror

    reply = _raw_request(server.address, [_header(), b"\1" * SLOT, b"\2", b"\3"])

    assert reply[-1] == b"error"
    assert not head_rows.any()


def test_attach_rejects_an_unspecified_host_address(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions, host_ip="0.0.0.0")

    for outcome in outcomes.values():
        assert isinstance(outcome, RuntimeError)
        assert "not a host address" in str(outcome)


def test_attach_requires_an_explicit_host_address(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions, host_ip=None)

    assert isinstance(outcomes[0], RuntimeError)
    assert "VLLM_HOST_IP" in str(outcomes[0])
    assert isinstance(outcomes[1], RuntimeError)
