# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Name the stage where a request or the engine-core loop stops moving.

The periodic stats line cannot tell a blocked engine from an idle one: both
stop producing tokens, and the stats logger goes quiet for both. A request
that waits outside the engine leaves no trace at all. These helpers log the
stage and its duration instead.
"""

import sys
import threading
import time
import traceback
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

logger = init_logger(__name__)

# Innermost loop-thread frames included in a stall report.
_STACK_FRAMES = 40


class RequestReceipt:
    """When the API server received an HTTP request and finished reading it."""

    __slots__ = ("received", "body_read")

    def __init__(self, received: float):
        self.received = received
        self.body_read: float | None = None


# Set by the HTTP receipt middleware for the task that serves the request.
REQUEST_RECEIPT: ContextVar[RequestReceipt | None] = ContextVar(
    "vllm_request_receipt", default=None
)


def _clock(timestamp: float) -> str:
    """Format a wall-clock time like the log line prefix, with milliseconds."""
    return time.strftime("%H:%M:%S", time.localtime(timestamp)) + (
        f".{int(timestamp % 1 * 1000):03d}"
    )


def _span(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "-"
    return f"{end - start:.2f} s"


class RequestTimeline:
    """Wall-clock times at which one request passed each API-server stage.

    Args:
        request_id: Request whose stages are recorded.
        arrival: The renderer's arrival stamp, taken when rendering started,
            or None when the prompt carried none.
        started: When the result generator started running, which is when
            the serving handler first asked it for output.

    Stages the request never reached stay None.
    """

    __slots__ = (
        "request_id",
        "received",
        "body_read",
        "arrival",
        "started",
        "processed",
        "submitted",
        "first_output",
        "final_output",
        "closed",
        "last_output",
    )

    def __init__(self, request_id: str, arrival: float | None, started: float):
        self.request_id = request_id
        receipt = REQUEST_RECEIPT.get()
        # HTTP receipt and complete request body, when served over HTTP.
        self.received = receipt.received if receipt is not None else None
        self.body_read = receipt.body_read if receipt is not None else None
        self.arrival = arrival
        self.started = started
        # Input processing finished and the output collector was created.
        self.processed: float | None = None
        # The engine-core client accepted the request.
        self.submitted: float | None = None
        self.first_output: float | None = None
        self.final_output: float | None = None
        # The result generator stopped: after the handler read the final
        # output, or when the request was aborted or failed.
        self.closed: float | None = None
        self.last_output: RequestOutput | None = None

    def output(self, output: "RequestOutput") -> None:
        """Record an output handed to the serving handler."""
        now = time.time()
        if self.first_output is None:
            self.first_output = now
        if output.finished:
            self.final_output = now
        self.last_output = output

    def outside_engine(self) -> float:
        """Longest stretch spent outside engine work, in seconds.

        That is HTTP receipt (or rendering) until the generator started, input
        processing plus submission, or the handler taking the final output to
        the client.
        """
        waits = [0.0]
        first = self._origin()
        waits.append(self.started - first)
        if self.submitted is not None:
            waits.append(self.submitted - self.started)
        if self.final_output is not None and self.closed is not None:
            waits.append(self.closed - self.final_output)
        return max(waits)

    def _origin(self) -> float:
        for stamp in (self.received, self.arrival):
            if stamp is not None:
                return stamp
        return self.started

    def describe(self, outcome: str) -> str:
        """Return a one-line account of every stage the request reached."""
        assert self.closed is not None
        origin = self._origin()
        parts = [
            f"Request {self.request_id} {outcome} after "
            f"{self.closed - origin:.2f} s (arrived {_clock(origin)})",
        ]
        if self.received is not None:
            parts += [
                f"HTTP receipt to body read {_span(self.received, self.body_read)}",
                f"body read to rendering {_span(self.body_read, self.arrival)}",
            ]
        parts += [
            f"arrival to generator start {_span(self.arrival, self.started)}",
            f"input processing {_span(self.started, self.processed)}",
            f"engine submission {_span(self.processed, self.submitted)}",
            f"submission to first output {_span(self.submitted, self.first_output)}",
            f"first to final output {_span(self.first_output, self.final_output)}",
            f"final output to stream end {_span(self.final_output, self.closed)}",
        ]
        output = self.last_output
        if output is not None:
            stats = output.metrics
            if stats is not None and stats.queued_ts and stats.scheduled_ts:
                parts.append(
                    f"engine queue {stats.scheduled_ts - stats.queued_ts:.2f} s"
                )
            if output.prompt_token_ids is not None:
                parts.append(
                    f"prompt {len(output.prompt_token_ids)} tokens "
                    f"({output.num_cached_tokens or 0} cached)"
                )
            if stats is not None:
                parts.append(f"{stats.num_generation_tokens} generated")
            if output.outputs and output.outputs[0].finish_reason is not None:
                parts.append(f"finish reason {output.outputs[0].finish_reason}")
        return ", ".join(parts)

    def close(self, outcome: str, threshold: float, always: bool) -> None:
        """Stamp the end of the request and log its stages when warranted.

        Args:
            outcome: How the request ended, e.g. "finished" or "aborted".
            threshold: Seconds above which a warning is logged: spent outside
                engine work, or in total by a request that ended without any
                output. 0 never warns.
            always: Log every request at INFO level.
        """
        self.closed = time.time()
        if self.final_output is not None:
            # A handler may stop iterating once it has the final output.
            outcome = "finished"
        if always:
            logger.info("%s", self.describe(outcome))
            return
        if threshold <= 0:
            return
        origin = self._origin()
        if self.outside_engine() > threshold or (
            self.first_output is None and self.closed - origin > threshold
        ):
            logger.warning("%s", self.describe(outcome))


def arrival_time(prompt: Any) -> float | None:
    """Return the renderer's arrival stamp carried by a prompt, if any."""
    value = (
        prompt.get("arrival_time")
        if isinstance(prompt, dict)
        else getattr(prompt, "arrival_time", None)
    )
    return value if isinstance(value, float | int) else None


class EngineLoopWatchdog:
    """Log the loop's stack when one engine-core loop step runs too long.

    The loop thread calls ``busy(stage)`` before each unit of work and
    ``idle()`` before blocking for new work. A daemon thread checks the
    current unit. Once a unit runs past the threshold it logs the loop
    thread's stack, again after each further threshold, and the loop thread
    logs the unit's total time when the unit ends.

    Args:
        threshold: Seconds a single unit may run before it is reported.
        poll_interval: Seconds between checks; defaults to a quarter of the
            threshold, at most five seconds.
    """

    def __init__(self, threshold: float, poll_interval: float | None = None):
        if threshold <= 0:
            raise ValueError("The engine loop stall threshold must be positive")
        self._threshold = threshold
        self._poll_interval = poll_interval or min(5.0, threshold / 4)
        # (stage, monotonic start, loop thread id) of the running unit.
        self._current: tuple[str, float, int] | None = None
        # The unit last reported, and how many times.
        self._reported: tuple[str, float, int] | None = None
        self._reports = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._watch, name="EngineLoopWatchdog", daemon=True
        )
        self._thread.start()

    def busy(self, stage: str) -> None:
        """Mark the start of a unit of loop work, ending the previous one."""
        self._finish()
        self._current = (stage, time.monotonic(), threading.get_ident())

    def idle(self) -> None:
        """Mark that the loop is about to block waiting for new work."""
        self._finish()
        self._current = None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._poll_interval + 1)

    def _finish(self) -> None:
        current = self._current
        if current is not None and current is self._reported:
            logger.warning(
                "Engine core %s finished after %.1f s",
                current[0],
                time.monotonic() - current[1],
            )

    def _watch(self) -> None:
        while not self._stop.wait(self._poll_interval):
            current = self._current
            if current is None:
                continue
            stage, started, thread_id = current
            reports = self._reports if current is self._reported else 0
            elapsed = time.monotonic() - started
            if elapsed < self._threshold * (reports + 1):
                continue
            frame = sys._current_frames().get(thread_id)
            stack = (
                "".join(traceback.format_stack(frame)[-_STACK_FRAMES:])
                if frame is not None
                else "  (stack unavailable)\n"
            )
            logger.warning(
                "Engine core has spent %.0f s in one %s without finishing; "
                "the engine is blocked, not idle. Loop thread stack:\n%s",
                elapsed,
                stage,
                stack.rstrip(),
            )
            self._reported = current
            self._reports = reports + 1
