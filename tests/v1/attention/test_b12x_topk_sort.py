# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch of the B12X selection sort (``b12x_topk_sort``): the row-count
gate, the index space of the indexer plans, and the side-stream fork/join
inside full CUDA graphs. The kernel is tested in the b12x repository; here
the op is a recorded stand-in."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_indexer, b12x_topk_sort


class _FakeOp:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sort_convert(self, indices, seq_lens, block_table, block_size, max_positions):
        stream = torch.cuda.current_stream().cuda_stream if indices.is_cuda else 0
        self.calls.append(
            ("sort", tuple(indices.shape), block_size, max_positions, stream)
        )

    def precompile(self, max_positions, device):
        self.calls.append(("precompile", max_positions))

    def is_supported(self, device=None):
        return True


@pytest.mark.parametrize(
    ("enabled", "max_tokens", "rows", "expected"),
    [
        (True, 8, 8, True),
        (True, 8, 9, False),
        (True, 0, 4096, True),
        (False, 8, 1, False),
    ],
)
def test_active_gate(monkeypatch, enabled, max_tokens, rows, expected) -> None:
    monkeypatch.setattr(b12x_topk_sort, "ENABLED", enabled)
    monkeypatch.setattr(b12x_topk_sort, "MAX_TOKENS", max_tokens)
    assert b12x_topk_sort.active(rows) is expected


@pytest.mark.parametrize(
    ("capture_sizes", "max_num_seqs", "sort_selection", "expected"),
    [
        # Capture sizes above the gate: the gate-sized plan is added so a
        # small decode batch does not select the 16-row physical-slot plan.
        ([16, 32], 32, True, [8, 16, 32]),
        # Capture sizes within the gate already provide sorting plans.
        ([1, 2, 4, 8, 16], 32, True, [1, 2, 4, 8, 16, 32]),
        # No capture sizes: the gate and the largest batch.
        ([], 32, True, [8, 32]),
        # The largest batch is within the gate: it sorts already.
        ([], 8, True, [8]),
        # Sort inactive: capture sizes and the largest batch only.
        ([16, 32], 32, False, [16, 32]),
        # Capture sizes above the largest batch are dropped.
        ([4, 64], 32, False, [4, 32]),
    ],
)
def test_decode_plan_sizes_include_the_sort_gate(
    monkeypatch, capture_sizes, max_num_seqs, sort_selection, expected
):
    monkeypatch.setattr(b12x_topk_sort, "MAX_TOKENS", 8)
    assert (
        b12x_indexer._decode_plan_sizes(capture_sizes, max_num_seqs, sort_selection)
        == expected
    )


@pytest.mark.parametrize(
    ("physical", "sort_selection", "rows", "expected"),
    [
        (True, True, 4, "logical"),
        (True, True, 8, "logical"),
        (True, True, 16, "physical"),
        (True, False, 4, "physical"),
        (False, True, 4, "logical"),
        (False, False, 4, "logical"),
    ],
)
def test_plan_output_space(monkeypatch, physical, sort_selection, rows, expected):
    monkeypatch.setattr(b12x_topk_sort, "ENABLED", True)
    monkeypatch.setattr(b12x_topk_sort, "MAX_TOKENS", 8)
    assert b12x_indexer._plan_output_space(physical, sort_selection, rows) == expected


def test_is_supported_without_the_b12x_op(monkeypatch) -> None:
    """An installed b12x without ``attention.topk_sort`` leaves the sort off
    (the indexer then keeps its unsorted selection) instead of raising at
    model construction."""

    def missing_op():
        raise ImportError("cannot import name 'topk_sort' from 'b12x.attention'")

    monkeypatch.setattr(b12x_topk_sort, "ENABLED", True)
    monkeypatch.setattr(b12x_topk_sort, "_op", missing_op)
    assert b12x_topk_sort.is_supported(torch.device("cpu")) is False
    monkeypatch.setattr(b12x_topk_sort, "ENABLED", False)
    assert b12x_topk_sort.is_supported(torch.device("cpu")) is False


def test_sorts_follows_the_plan_index_space(monkeypatch) -> None:
    indexer = b12x_indexer.B12xSparseIndexer.__new__(b12x_indexer.B12xSparseIndexer)
    indexer.sort_selection = True
    logical = SimpleNamespace(caps=SimpleNamespace(output_index_space="logical"))
    physical = SimpleNamespace(caps=SimpleNamespace(output_index_space="physical"))
    assert indexer._sorts(logical)
    assert not indexer._sorts(physical)
    indexer.sort_selection = False
    assert not indexer._sorts(logical)


def test_eager_sort_runs_in_line(monkeypatch) -> None:
    op = _FakeOp()
    monkeypatch.setattr(b12x_topk_sort, "_op", lambda: op)
    monkeypatch.setattr(b12x_topk_sort, "_graph_mode", lambda: False)
    indices = torch.full((4, 16), -1, dtype=torch.int32)
    seq_lens = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
    table = torch.zeros(4, 2, dtype=torch.int32)
    out = b12x_topk_sort.sort_convert_async(indices, seq_lens, table, 64, 4096)
    assert out is indices
    assert op.calls == [("sort", (4, 16), 64, 4096, 0)]
    assert not b12x_topk_sort._pending
    b12x_topk_sort.join(torch.device("cpu"))  # nothing pending: no-op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_mode_sort_forks_and_joins(monkeypatch) -> None:
    op = _FakeOp()
    monkeypatch.setattr(b12x_topk_sort, "_op", lambda: op)
    monkeypatch.setattr(b12x_topk_sort, "_graph_mode", lambda: True)
    device = torch.device("cuda")
    indices = torch.full((4, 16), -1, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([3, 4, 5, 6], dtype=torch.int32, device=device)
    table = torch.zeros(4, 2, dtype=torch.int32, device=device)
    main = torch.cuda.current_stream(device).cuda_stream
    table_ref = weakref.ref(table)
    b12x_topk_sort.sort_convert_async(indices, seq_lens, table, 64, 4096)
    assert len(op.calls) == 1
    assert op.calls[0][-1] != main, "the sort must run on the side stream"
    assert op.calls[0][-1] == b12x_topk_sort._stream(device).cuda_stream
    assert b12x_topk_sort._device_index(device) in b12x_topk_sort._pending
    # The caller's page table is a temporary: the fork must keep it alive
    # until the join so that no main-stream allocation reuses its memory
    # while the sort stream reads it.
    del table
    assert table_ref() is not None, "the fork must retain the page table"
    b12x_topk_sort.join(device)
    assert not b12x_topk_sort._pending
    assert table_ref() is None, "the join must release the page table"
    torch.accelerator.synchronize(device)


@pytest.mark.parametrize(
    ("caps", "physical"),
    [
        (SimpleNamespace(output_physical_slots=True), True),
        (SimpleNamespace(output_physical_slots=False), False),
        (SimpleNamespace(output_index_space="physical"), True),
        (SimpleNamespace(output_index_space="logical"), False),
        (SimpleNamespace(), False),
    ],
)
def test_plan_emits_physical_slots_reads_scratch_or_api_caps(caps, physical) -> None:
    """A compiled plan carries scratch caps (``output_physical_slots``); the
    API caps it was built from carry ``output_index_space``."""
    plan = SimpleNamespace(caps=caps)
    assert b12x_indexer._plan_emits_physical_slots(plan) is physical
