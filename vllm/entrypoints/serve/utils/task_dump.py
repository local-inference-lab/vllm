# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show where every asyncio task of the API server is suspended.

A request held at an ``await`` is on no thread's stack: py-spy and
faulthandler only show the event loop waiting in ``select``. This walks each
task's chain of suspended coroutines and async generators down to the object
it waits on. Frames list local variable names and types; values are shown
only for numbers and request IDs, so prompts and outputs stay out of logs.

Send ``SIGUSR1`` to the API server process to log the dump.
"""

import asyncio
import gc
import inspect
import sys
import threading
import traceback
from types import FrameType
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Chains are short in practice; this only bounds a pathological cycle.
_MAX_DEPTH = 100
_ID_NAMES = ("request_id", "sub_request_id", "external_req_id", "session_id")


def _describe_frame(frame: FrameType) -> str:
    code = frame.f_code
    name = getattr(code, "co_qualname", code.co_name)
    local_parts = []
    for local_name, value in frame.f_locals.items():
        if isinstance(value, bool | int | float) or value is None:
            local_parts.append(f"{local_name}={value!r}")
        elif local_name in _ID_NAMES and isinstance(value, str):
            local_parts.append(f"{local_name}={value[:100]!r}")
        else:
            local_parts.append(f"{local_name}: {type(value).__qualname__}")
    return (
        f"    {name} ({code.co_filename}:{frame.f_lineno})\n"
        f"      locals: {', '.join(local_parts) or '-'}"
    )


def _step(awaitable: Any) -> tuple[FrameType | None, Any]:
    """Return the frame of a suspended awaitable and what it awaits next."""
    if inspect.iscoroutine(awaitable):
        return awaitable.cr_frame, awaitable.cr_await
    if inspect.isasyncgen(awaitable):
        return awaitable.ag_frame, awaitable.ag_await
    if inspect.isgenerator(awaitable):
        return awaitable.gi_frame, awaitable.gi_yieldfrom
    if type(awaitable).__name__.startswith("async_generator_"):
        # Awaiting an async generator's next item awaits a wrapper that does
        # not expose the generator, but references it.
        for referent in gc.get_referents(awaitable):
            if inspect.isasyncgen(referent):
                return None, referent
    return None, None


def _describe_task(task: asyncio.Task) -> str:
    lines = [f"  Task {task.get_name()!r}:"]
    awaitable: Any = task.get_coro()
    for _ in range(_MAX_DEPTH):
        if awaitable is None:
            break
        frame, awaited = _step(awaitable)
        if frame is not None:
            lines.append(_describe_frame(frame))
        elif awaited is None:
            lines.append(f"    waits on {type(awaitable).__qualname__}")
            break
        awaitable = awaited
    waiter = getattr(task, "_fut_waiter", None)
    if waiter is not None:
        # The future's callbacks name what resumes the task, e.g. another
        # task that it waits for.
        lines.append(f"    task is blocked on {waiter!r}")
    return "\n".join(lines)


def format_asyncio_tasks(loop: asyncio.AbstractEventLoop | None = None) -> str:
    """Describe every task of ``loop`` and the stack of every other thread.

    Args:
        loop: Event loop whose tasks are described; the running loop if None.

    Returns:
        A multi-line report, one section per task and per thread.
    """
    loop = loop or asyncio.get_running_loop()
    tasks = sorted(asyncio.all_tasks(loop), key=lambda task: task.get_name())
    sections = [f"{len(tasks)} asyncio tasks:"]
    sections.extend(_describe_task(task) for task in tasks)
    current = threading.get_ident()
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        if ident == current:
            continue
        stack = "".join(traceback.format_stack(frame)).rstrip()
        sections.append(f"  Thread {names.get(ident, ident)!r}:\n{stack}")
    return "\n".join(sections)


def log_asyncio_tasks() -> None:
    """Signal handler: log where each task and thread of this process waits."""
    try:
        logger.warning("Task dump requested by signal:\n%s", format_asyncio_tasks())
    except Exception:
        logger.exception("Task dump failed")
