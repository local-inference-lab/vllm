# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the GPU stall watchdog writes a dump, and what the dump holds.

The stream write, the clock and cuda-gdb are faked, so this runs without a GPU;
the poll is driven directly instead of from the daemon thread.
"""

from types import SimpleNamespace

import numpy as np
import torch

import vllm.distributed.device_communicators.b12x_roce_all_reduce as roce_module
import vllm.v1.worker.gpu_stall_watchdog as watchdog_module
from vllm.v1.worker.gpu_stall_watchdog import GpuStallWatchdog


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
