# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 runner re-zeroes the null placeholder block after external KV loads."""

from typing import Any

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.gpu import model_runner as mrv2


class _RecordingZeroer:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def zero_block_ids(self, block_ids: list[int]) -> None:
        self.calls.append(list(block_ids))


def _make_runner(zeroer: Any) -> Any:
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.kv_block_zeroer = zeroer
    return runner


def test_finished_recv_zeroes_null_block() -> None:
    zeroer = _RecordingZeroer()
    runner = _make_runner(zeroer)

    runner._restore_null_block_after_kv_load(
        KVConnectorOutput(finished_recving={"req-1"})
    )

    assert zeroer.calls == [[0]]


def test_no_finished_recv_leaves_blocks_untouched() -> None:
    zeroer = _RecordingZeroer()
    runner = _make_runner(zeroer)

    runner._restore_null_block_after_kv_load(None)
    runner._restore_null_block_after_kv_load(KVConnectorOutput())
    runner._restore_null_block_after_kv_load(
        KVConnectorOutput(finished_sending={"req-1"})
    )

    assert zeroer.calls == []


def test_missing_zeroer_is_tolerated() -> None:
    runner = _make_runner(None)

    runner._restore_null_block_after_kv_load(
        KVConnectorOutput(finished_recving={"req-1"})
    )
