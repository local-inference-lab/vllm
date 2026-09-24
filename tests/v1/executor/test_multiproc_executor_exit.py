# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A process whose engine failed to start must not wait forever for workers."""

import contextlib
import multiprocessing
from multiprocessing.connection import Connection

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProcHandle

CONTEXT = multiprocessing.get_context("fork")


def _worker(death_reader: Connection, death_writer: Connection) -> None:
    """Stand-in worker: runs until its parent closes the death pipe."""
    death_writer.close()  # as real workers close their inherited fds
    with contextlib.suppress(EOFError):
        death_reader.recv()


def _engine_that_fails_after_starting_workers() -> None:
    death_reader, death_writer = CONTEXT.Pipe(duplex=False)
    # Not a daemon, like the real workers.
    worker = CONTEXT.Process(
        target=_worker, args=(death_reader, death_writer), daemon=False
    )
    worker.start()
    death_reader.close()
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.workers = [
        WorkerProcHandle(
            proc=worker,
            rank=0,
            worker_response_mq=None,
            peer_worker_response_mqs=[],
            death_writer=death_writer,
        )
    ]
    executor._register_exit_shutdown()
    # EngineCore construction fails; nothing calls executor.shutdown().
    raise RuntimeError("engine startup failed")


def test_failed_engine_start_exits_and_stops_its_workers():
    engine = CONTEXT.Process(target=_engine_that_fails_after_starting_workers)
    engine.start()
    engine.join(timeout=60)
    hung = engine.is_alive()
    if hung:
        engine.kill()  # closes the death pipe, so the worker exits too
        engine.join()
    assert not hung, "the engine process waited forever for its workers"
    assert engine.exitcode == 1
