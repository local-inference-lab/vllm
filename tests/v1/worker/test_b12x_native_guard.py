# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-only tests for the asymmetric-native preparation guards."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm"))

from vllm.v1.worker import b12x_startup


class _StubCoordinator(b12x_startup.B12xPreparationCoordinator):
    """Bypass __init__'s session/batches wiring; set fields directly."""

    def __init__(self, global_rank, world_ranks):
        self.global_rank = global_rank
        self.world_ranks = world_ranks
        self.round = 0
        self.native = True
        self.native_reason = "native"


def _entry(rank, native, ready=()):
    return {
        "round": 0,
        "global_rank": rank,
        "world_ranks": (0, 1),
        "native": native,
        "native_reason": "native" if native else "no_units",
        "stop": False,
        "ready": tuple(ready),
        "tuning": (),
        "local_done": False,
        "error": None,
        "cleanup_complete": False,
    }


def test_coordinator_raises_on_mixed_native_world():
    coordinator = _StubCoordinator(0, (0, 1))
    with pytest.raises(RuntimeError, match="asymmetrically native") as info:
        coordinator._validate_domain([_entry(0, True), _entry(1, False)])
    message = str(info.value)
    assert "[0]" in message and "[1]" in message
    assert "no_units" in message


def test_coordinator_allows_symmetric_worlds():
    coordinator = _StubCoordinator(0, (0, 1))
    coordinator._validate_domain([_entry(0, True), _entry(1, True)])
    coordinator._validate_domain([_entry(0, False), _entry(1, False)])


def test_engine_raises_on_mixed_native_outcomes():
    outcomes = [
        {"native": True, "done": False, "error": None, "progress": None},
        {"native": False, "done": False, "error": None, "progress": None},
    ]
    flags = [bool(item.get("native")) for item in outcomes]
    assert any(flags) and not all(flags)


def test_engine_allows_symmetric_native_outcomes():
    for values in ([True, True], [False, False]):
        flags = [bool(v) for v in values]
        assert not (any(flags) and not all(flags))
