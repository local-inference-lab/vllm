# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The capture_attention completion wait must be bounded (f9).

These are small tail copies (microseconds-to-ms healthy path); a wedged
kernel otherwise holds the engine-core loop forever on an unbounded
CUDA event synchronize. The wait is poll-bounded with a deadline and
raises on expiry — the loud-failure family: an incomplete capture must
not be read.

The CUDA surface (torch.cuda.Event query/synchronize/record) is
stubbed: the deadline mechanism is CPU-verifiable, but the CUDA event
poll/record behavior must be validated live before PR.
"""

import time
from types import SimpleNamespace

import pytest

# The module under test wraps its kernels in triton at import time; on a
# CPU-only host triton resolves to None, so substitute the minimal
# surface the module import touches (constexpr wrappers and @triton.jit
# decorators). Kernels are never invoked by these tests.
import vllm.triton_utils as _triton_utils

if getattr(getattr(_triton_utils, "tl", None), "constexpr", None) is None:

    class _FakeTl:
        # constexpr passes the value through; every other tl member used
        # at import time (dtype annotations) is a harmless callable.
        constexpr = staticmethod(lambda value: value)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def _fake_jit(fn=None, **_kwargs):
        return fn if fn is not None else (lambda inner: inner)

    _triton_utils.tl = _FakeTl()
    _triton_utils.triton = SimpleNamespace(jit=_fake_jit)

from vllm.v1.worker.gpu import boundary_checkpoint as bcp_module  # noqa: E402


class _StubCompletionEvent:
    """Fake torch.cuda.Event.

    query() replays `results` (False = copy in flight). synchronize()
    must never be called: the bounded wait polls instead of blocking.
    """

    def __init__(self, results: list[bool]) -> None:
        self._results = iter(results)
        self._last = bool(results[-1]) if results else True
        self.synchronize_calls = 0

    def query(self) -> bool:
        for result in self._results:
            self._last = result
            if not result:
                return result
        return self._last

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        raise AssertionError("synchronize() must not be used; the wait is poll-bounded")

    def record(self) -> None:
        return None


def _state(event: _StubCompletionEvent) -> bcp_module.BoundaryCheckpointState:
    state = bcp_module.BoundaryCheckpointState.__new__(
        bcp_module.BoundaryCheckpointState
    )
    state._completion = event
    return state


def test_wait_for_copies_returns_when_copies_complete(monkeypatch):
    """query() completing returns from wait_for_copies without a raise."""
    event = _StubCompletionEvent([False, False, True])
    state = _state(event)
    monkeypatch.setattr(bcp_module, "_COPY_WAIT_TIMEOUT", 30.0)

    state.wait_for_copies()

    assert event.synchronize_calls == 0


def test_wait_for_copies_raises_when_copies_never_complete(monkeypatch):
    """A wedged copy raises within the deadline instead of blocking the
    engine-core loop forever; an incomplete capture must not be read.
    """
    event = _StubCompletionEvent([False])
    state = _state(event)
    monkeypatch.setattr(bcp_module, "_COPY_WAIT_TIMEOUT", 0.1)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="did not complete within"):
        state.wait_for_copies()

    assert time.monotonic() - start < 10.0
    assert event.synchronize_calls == 0
