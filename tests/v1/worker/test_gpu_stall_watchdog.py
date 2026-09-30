# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the GPU stall watchdog writes a dump, and what the dump holds.

The stream write, the clock and cuda-gdb are faked, so this runs without a GPU;
the poll is driven directly instead of from the daemon thread.
"""

import os
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

import vllm.distributed.device_communicators.b12x_roce_all_reduce as roce_module
import vllm.v1.worker.gpu_stall_watchdog as watchdog_module
from vllm.distributed.device_communicators.b12x_roce_all_reduce import (
    B12xRoceAllReduce,
)
from vllm.v1.worker.gpu_stall_watchdog import STEP_MASK, GpuStallWatchdog


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _watchdog(monkeypatch, tmp_path, comms=()):
    """A watchdog whose GPU finishes a step only when the test says so."""
    clock = _Clock()
    monkeypatch.setattr(watchdog_module.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(watchdog_module.time, "monotonic", clock)
    monkeypatch.setattr(
        watchdog_module, "_stream_write_u32", lambda address, value: None
    )
    monkeypatch.setattr(
        GpuStallWatchdog, "_cuda_gdb", staticmethod(lambda: "kernel hung_kernel\n")
    )
    monkeypatch.setattr(roce_module, "live_roce_communicators", lambda: list(comms))
    watchdog = GpuStallWatchdog(rank=2, stall_seconds=60)
    # An unpinned host word stands in for the pinned one the stream writes.
    watchdog._done_tensor = torch.zeros(1, dtype=torch.int32)
    watchdog._done = watchdog._done_tensor.numpy().view(np.uint32)
    return watchdog, clock


def _finish_through(watchdog, step: int) -> None:
    """What the stream does when it reaches ``step``'s write."""
    watchdog._done[0] = step


def _dumps(tmp_path) -> list[str]:
    directory = tmp_path / "gpu_stall"
    if not directory.exists():
        return []
    return [path.read_text() for path in sorted(directory.iterdir())]


def test_finished_steps_never_dump(monkeypatch, tmp_path):
    watchdog, clock = _watchdog(monkeypatch, tmp_path)
    watchdog.mark()
    _finish_through(watchdog, 1)
    clock.now += 3600
    watchdog._poll()
    assert _dumps(tmp_path) == []


def test_one_dump_per_stall_and_rearmed_after_it_clears(monkeypatch, tmp_path):
    watchdog, clock = _watchdog(monkeypatch, tmp_path)
    watchdog.mark()
    clock.now += 59
    watchdog._poll()
    assert _dumps(tmp_path) == []  # under the threshold
    clock.now += 2
    watchdog._poll()
    clock.now += 600
    watchdog._poll()
    dumps = _dumps(tmp_path)
    assert len(dumps) == 1  # the stall is reported once, not once per poll
    assert "has not finished on rank 2" in dumps[0]
    assert "kernel hung_kernel" in dumps[0]
    assert "== Python threads" in dumps[0]
    _finish_through(watchdog, 1)
    watchdog._poll()
    watchdog.mark()
    clock.now += 61
    watchdog._poll()
    assert len(_dumps(tmp_path)) == 2


def test_the_oldest_unfinished_step_sets_the_age(monkeypatch, tmp_path):
    watchdog, clock = _watchdog(monkeypatch, tmp_path)
    watchdog.mark()
    clock.now += 50
    watchdog.mark()
    _finish_through(watchdog, 1)
    clock.now += 50  # step 2 is 50 s old, step 1 finished
    watchdog._poll()
    assert _dumps(tmp_path) == []


def test_poisoned_roce_dumps_once_with_its_snapshot(monkeypatch, tmp_path):
    snapshot = {"doorbell": 7, "completed": 6, "failed": 1, "error_peer": 0}
    comm = SimpleNamespace(
        name="ranks 0-1-2-3", poisoned=True, snapshot=lambda: snapshot
    )
    watchdog, _ = _watchdog(monkeypatch, tmp_path, comms=[comm])
    watchdog._poll()
    watchdog._poll()
    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "RoCEnante on rank 2 timed out" in dumps[0]
    assert f"RoCEnante ranks 0-1-2-3: {snapshot}" in dumps[0]


def test_step_numbers_wrap_at_32_bits(monkeypatch, tmp_path):
    watchdog, clock = _watchdog(monkeypatch, tmp_path)
    watchdog._next_step = STEP_MASK - 1
    watchdog.mark()
    watchdog.mark()
    assert [step for step, _ in watchdog._pending] == [STEP_MASK, 0]
    _finish_through(watchdog, STEP_MASK - 1)  # neither step reached yet
    clock.now += 61
    watchdog._poll()
    assert len(_dumps(tmp_path)) == 1
    _finish_through(watchdog, 0)  # the word wrapped past both
    watchdog._poll()
    assert not watchdog._pending
    watchdog.mark()
    clock.now += 61
    watchdog._poll()
    assert len(_dumps(tmp_path)) == 2  # re-armed: step 1 is a new stall


def test_a_step_blocked_on_the_host_dumps_once_armed(monkeypatch, tmp_path):
    watchdog, clock = _watchdog(monkeypatch, tmp_path)
    with watchdog.host_step():  # warm-up: not timed
        clock.now += 600
        watchdog._poll()
    assert _dumps(tmp_path) == []
    watchdog.arm_host_steps()
    with watchdog.host_step():  # blocks before any work is enqueued
        clock.now += 59
        watchdog._poll()
        assert _dumps(tmp_path) == []
        clock.now += 2
        watchdog._poll()
        watchdog._poll()
    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "worker thread on rank 2 has been inside one step for 61 s" in dumps[0]
    watchdog._poll()
    assert watchdog._stall_reported is False  # the step returned: re-armed


def test_snapshot_on_a_b12x_without_it():
    comm = B12xRoceAllReduce.__new__(B12xRoceAllReduce)  # no process group
    comm.disabled = False
    comm._runtime = SimpleNamespace(poisoned=False)
    assert comm.snapshot() == {"snapshot": "unavailable in this b12x"}
    comm._runtime.snapshot = lambda: {"doorbell": 3, "completed": 3}
    assert comm.snapshot() == {"doorbell": 3, "completed": 3}


def test_a_cuda_gdb_timeout_kills_its_whole_session(monkeypatch, tmp_path):
    """The shell and the debugger it started both go, and this process runs on."""
    pid_file = tmp_path / "gdb.pid"
    fake_gdb = tmp_path / "cuda-gdb"
    fake_gdb.write_text(f"#!/bin/bash\necho $$ > {pid_file}\nexec sleep 600\n")
    fake_gdb.chmod(0o755)
    monkeypatch.setattr(watchdog_module.shutil, "which", lambda name: str(fake_gdb))
    monkeypatch.setattr(watchdog_module, "CUDA_GDB_ATTACH_SECONDS", 0)
    monkeypatch.setattr(watchdog_module, "CUDA_GDB_TIMEOUT_SECONDS", 2)
    started = time.monotonic()
    output = GpuStallWatchdog._cuda_gdb()
    assert time.monotonic() - started < 30
    assert "cuda-gdb failed" in output
    gdb_pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(gdb_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(gdb_pid, signal.SIGKILL)
        raise AssertionError("the debugger outlived its timeout")
