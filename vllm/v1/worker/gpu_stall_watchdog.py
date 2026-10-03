# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record what a rank's GPU is running when a step never finishes.

A multi-node step that wedges looks the same from outside whatever wedged it:
the output rank blocks in its host synchronization, the other ranks sit idle
waiting for the next command, and nothing is logged until the executor's
timeout tears the engine down along with every worker's evidence. This
watchdog runs on every rank, so the rank whose GPU stopped reports it itself.

After each step the worker enqueues, the stream itself writes the step's
number into a pinned host word (``cuStreamWriteValue32``) once it gets there,
and a daemon thread compares that word with the steps enqueued. The thread
makes no CUDA call: a query of an event or stream from another thread while
the worker captures a CUDA graph invalidates the capture. Once warm-up is
over (its steps compile for minutes), the worker also records when it enters
and leaves each step call, so a step that blocks on the host before its work
is enqueued is caught too. When either has been
outstanding for ``VLLM_GPU_STALL_DUMP_SECONDS``, or a RoCEnante
wait on this rank timed out, it writes one dump under
``$VLLM_CACHE_ROOT/gpu_stall/``: every RoCEnante runtime's host-side protocol
state, every Python thread's stack, and the kernels the GPU is running, read
by attaching ``cuda-gdb`` to this process (which needs ptrace permission: in a
container, ``SYS_PTRACE`` or ``--privileged``; without it the dump keeps the
stacks and RoCEnante state). It never tries to recover; a wedged GPU is past
that from inside the process.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import datetime
import faulthandler
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

POLL_SECONDS = 1.0
# Step numbers live in a 32-bit word the stream writes, so they wrap; a step
# counts as reached when the word is at most half the range past it.
STEP_MASK = 0xFFFFFFFF
STEP_HALF_RANGE = 1 << 31
CUDA_GDB_TIMEOUT_SECONDS = 120
# cuda-gdb finishes attaching to the GPU asynchronously: it resumes the process
# to run its attach stub and completes only while it waits for input, so the
# commands go in over stdin after a pause, not as -batch -ex options (which
# run before the attach completes and report no CUDA devices). The attach
# stops every thread of this process, the watchdog's included, so a shell
# feeds the commands and gdb writes to a file: a pipe this thread had to
# drain would fill and block gdb while the thread is stopped.
CUDA_GDB_ATTACH_SECONDS = 5
CUDA_GDB_COMMANDS = (
    "info cuda kernels",
    "info cuda blocks",
    "info cuda warps",
    "x/8i $pc",
    "detach",
    "quit",
)


def _stream_write_u32(address: int, value: int) -> None:
    """Enqueue a 32-bit write of ``value`` to ``address`` on the current stream."""
    from cuda.bindings import driver

    stream = torch.cuda.current_stream().cuda_stream
    (result,) = driver.cuStreamWriteValue32(stream, address, value, 0)
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamWriteValue32 failed: {result}")


@contextlib.contextmanager
def _sigint_ignored():
    """Drop SIGINT process-wide while cuda-gdb is attached.

    cuda-gdb interrupts the process with kill(pid, SIGINT) once its GPU attach
    completes, and the signal can land after the detach, where a worker takes
    it as a shutdown request. Python only lets the main thread install signal
    handlers, so the disposition is swapped with sigaction directly and the
    previous handler restored afterwards.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    # Any libc's struct sigaction fits in 512 bytes and starts with the
    # handler, so copying the current action and replacing its first pointer
    # yields the same action with SIG_IGN.
    previous = ctypes.create_string_buffer(512)
    if libc.sigaction(signal.SIGINT, None, previous) != 0:
        yield
        return
    ignore = ctypes.create_string_buffer(previous.raw, 512)
    ctypes.c_void_p.from_buffer(ignore).value = int(signal.SIG_IGN)
    libc.sigaction(signal.SIGINT, ignore, None)
    try:
        yield
    finally:
        libc.sigaction(signal.SIGINT, previous, None)


class GpuStallWatchdog:
    """Per-rank detector and recorder of a GPU step that never completes."""

    def __init__(self, rank: int, stall_seconds: float) -> None:
        self._rank = rank
        self._stall_seconds = stall_seconds
        self._dump_dir = os.path.join(envs.VLLM_CACHE_ROOT, "gpu_stall")
        self._lock = threading.Lock()
        # (step number, time enqueued) of every step the GPU has not reached.
        self._pending: collections.deque[tuple[int, float]] = collections.deque()
        self._next_step = 0
        # When the worker thread entered the step call it has not left, if any;
        # only timed once warm-up, whose steps compile for minutes, is over.
        self._host_steps_armed = False
        self._host_step_started: float | None = None
        # Pinned host word the stream writes each finished step's number into;
        # allocated on the first mark, from the worker thread.
        self._done_tensor: torch.Tensor | None = None
        self._done: np.ndarray | None = None
        # One dump per stall (re-armed once the stall clears) and one per
        # RoCEnante failure (terminal: a poisoned runtime stays poisoned).
        self._stall_reported = False
        self._roce_reported = False
        self._dump_count = 0

    def start(self) -> None:
        """Start polling on a daemon thread."""
        threading.Thread(
            target=self._run, name="gpu-stall-watchdog", daemon=True
        ).start()
        logger.info(
            "GPU stall watchdog on rank %d: dumps to %s after %.0f s",
            self._rank,
            self._dump_dir,
            self._stall_seconds,
        )

    def arm_host_steps(self) -> None:
        """Start timing step calls on the host; warm-up is over."""
        self._host_steps_armed = True

    @contextlib.contextmanager
    def host_step(self):
        """Time one step call on the worker thread, whatever it blocks on."""
        if not self._host_steps_armed:
            yield
            return
        self._host_step_started = time.monotonic()
        try:
            yield
        finally:
            self._host_step_started = None

    def mark(self) -> None:
        """Have the current stream report this step once it has run."""
        if self._done is None:
            self._done_tensor = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._done = self._done_tensor.numpy().view(np.uint32)
        assert self._done_tensor is not None
        self._next_step = (self._next_step + 1) & STEP_MASK
        _stream_write_u32(self._done_tensor.data_ptr(), self._next_step)
        with self._lock:
            self._pending.append((self._next_step, time.monotonic()))

    def _oldest_pending_age(self) -> float | None:
        """Seconds the oldest step the GPU has not reached has been outstanding."""
        if self._done is None:
            return None
        done = int(self._done[0])
        with self._lock:
            while (
                self._pending
                and (done - self._pending[0][0]) & STEP_MASK < STEP_HALF_RANGE
            ):
                self._pending.popleft()
            if not self._pending:
                return None
            return time.monotonic() - self._pending[0][1]

    def _host_step_age(self) -> float | None:
        """Seconds the worker thread has been inside its current step call."""
        started = self._host_step_started
        return None if started is None else time.monotonic() - started

    def _run(self) -> None:
        while True:
            time.sleep(POLL_SECONDS)
            try:
                self._poll()
            except Exception:
                # A watchdog that dies silently is worse than none.
                logger.exception(
                    "GPU stall watchdog poll failed on rank %d", self._rank
                )

    def _poll(self) -> None:
        from vllm.distributed.device_communicators.b12x_roce_all_reduce import (
            live_roce_communicators,
        )

        gpu_age = self._oldest_pending_age()
        host_age = self._host_step_age()
        if not self._roce_reported and any(
            comm.poisoned for comm in live_roce_communicators()
        ):
            self._roce_reported = True
            self._dump(f"RoCEnante on rank {self._rank} timed out or its proxy failed")
        stalled = [
            age
            for age in (gpu_age, host_age)
            if age is not None and age >= self._stall_seconds
        ]
        if not stalled:
            self._stall_reported = False
        elif not self._stall_reported:
            self._stall_reported = True
            if gpu_age is not None and gpu_age >= self._stall_seconds:
                reason = (
                    f"GPU work enqueued {gpu_age:.0f} s ago has not finished on "
                    f"rank {self._rank}"
                )
            else:
                reason = (
                    f"The worker thread on rank {self._rank} has been inside one "
                    f"step for {host_age:.0f} s"
                )
            self._dump(reason)

    def _dump(self, reason: str) -> None:
        from vllm.distributed.device_communicators.b12x_roce_all_reduce import (
            live_roce_communicators,
        )

        roce = [(comm.name, comm.snapshot()) for comm in live_roce_communicators()]
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(self._dump_dir, exist_ok=True)
        self._dump_count += 1
        path = os.path.join(
            self._dump_dir, f"rank{self._rank}-{stamp}-{self._dump_count}.txt"
        )
        # The reason and the RoCEnante state go to the log first: they need no
        # attach, and the snapshot names the missing peer and whether this
        # rank's collective is still in flight (doorbell ahead of completed).
        logger.error("%s. RoCEnante: %s. Writing %s", reason, roce or "none", path)
        with open(path, "w") as out:
            out.write(f"{reason}\npid {os.getpid()} rank {self._rank} at {stamp}\n")
            for name, snapshot in roce:
                out.write(f"RoCEnante {name}: {snapshot}\n")
            out.write("\n== Python threads\n")
            out.flush()
            faulthandler.dump_traceback(file=out, all_threads=True)
            out.write("\n== cuda-gdb\n")
            out.write(self._cuda_gdb())
        logger.error("GPU stall dump for rank %d written to %s", self._rank, path)

    @staticmethod
    def _cuda_gdb() -> str:
        """The kernels this process's GPU is running, by attaching cuda-gdb."""
        gdb = shutil.which("cuda-gdb") or "/usr/local/cuda/bin/cuda-gdb"
        commands = "\\n".join(CUDA_GDB_COMMANDS)
        script = (
            f"(sleep {CUDA_GDB_ATTACH_SECONDS}; printf '{commands}\\n') | "
            f"{shlex.quote(gdb)} -q -nx -iex 'set pagination off' "
            f"-iex 'set confirm off' -p {os.getpid()}"
        )
        with tempfile.TemporaryFile(mode="w+") as output, _sigint_ignored():
            try:
                process = subprocess.Popen(
                    ["bash", "-c", script],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                output.write(f"\ncuda-gdb failed: {exc}\n")
            else:
                try:
                    process.wait(timeout=CUDA_GDB_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired as exc:
                    # Kill the whole session, gdb included, not just the shell;
                    # a tracer that dies can leave this process stopped, so
                    # continue it.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    os.kill(os.getpid(), signal.SIGCONT)
                    output.write(f"\ncuda-gdb failed: {exc}\n")
            output.seek(0)
            return output.read()
