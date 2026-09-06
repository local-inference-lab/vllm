# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a split prefill chunk may be cut.

The B12X DMA ring reduces an all-reduce input as ``world`` chunks. With
``B12X_PCIE_RING_GRANULE_ROWS=g`` an element's summation order depends only
on its row index modulo ``world * g``, so a row slice reduces to the bits of
the whole tensor exactly when the slice starts and ends on a multiple of that
period. The split driver must therefore cut a prefill chunk on a period
boundary and leave a chunk that is not a whole number of periods unsplit;
without the granule mapping the order is relative to the row count and the
only boundary constraint left is the FlashKDA recurrent-state tile.
"""

import pytest

from vllm.v1.worker.gpu.k3_ubatch_prefill import (
    SPLIT_ALIGNMENT,
    current_split_point,
    ring_granule_rows,
    split_point,
)

WORLD = 9


@pytest.mark.parametrize(
    "rows,expected",
    [
        (4608, 2304),  # the served maximum chunk
        (3072, 1536),
        (1297, 656),  # 649 rounded up to the 16-row tile
        (1024, 512),
        (16, 0),  # a single tile cannot be cut in two
        (1, 0),
        (0, 0),
    ],
)
def test_row_count_relative_ring_cuts_on_the_kda_tile(rows, expected):
    assert split_point(rows) == expected
    assert expected == 0 or expected % SPLIT_ALIGNMENT == 0


@pytest.mark.parametrize(
    "granule,rows,expected",
    [
        (256, 4608, 2304),  # two periods of 2,304 rows
        (256, 2304, 0),  # one period: nothing to cut
        (256, 3072, 0),  # not a whole number of periods
        (256, 1297, 0),
        (128, 4608, 2304),  # four periods of 1,152 rows
        (128, 3456, 2304),  # three periods: the first half is the larger one
        (128, 1152, 0),
        (128, 1297, 0),
        (512, 4608, 0),  # one period of 4,608 rows
        (64, 4608, 2304),  # eight periods of 576 rows
        (32, 4608, 0),  # sixteen granules per chunk exceeds the ring's budget
    ],
)
def test_granule_ring_cuts_on_the_period(granule, rows, expected):
    block_rows = WORLD * granule
    got = split_point(rows, block_rows)
    assert got == expected
    if got:
        assert got % block_rows == 0
        assert (rows - got) % block_rows == 0
        assert got >= rows - got


@pytest.mark.parametrize("granule", [128, 256])
def test_granule_boundary_is_also_a_kda_tile_boundary(granule):
    # The recurrent-state hand-off constraint is implied by the period, so the
    # two rules never conflict for the granules the ring supports.
    assert (WORLD * granule) % SPLIT_ALIGNMENT == 0
    assert split_point(4608, WORLD * granule) % SPLIT_ALIGNMENT == 0


def test_granule_rows_from_environment(monkeypatch):
    monkeypatch.delenv("B12X_PCIE_RING_GRANULE_ROWS", raising=False)
    assert ring_granule_rows() == 0
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "256")
    assert ring_granule_rows() == 256
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "")
    assert ring_granule_rows() == 0
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "not-a-number")
    assert ring_granule_rows() == 0
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "-4")
    assert ring_granule_rows() == 0


def test_configured_granule_without_a_period_refuses_to_split(monkeypatch):
    # An unknown tensor-parallel size cannot be turned into a period, and the
    # granule is configured precisely so that the halves reduce like the
    # unsplit chunk: run the chunk whole instead of cutting it at a boundary
    # the ring does not respect.
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "256")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.k3_ubatch_prefill._tp_world_size",
        lambda: (_ for _ in ()).throw(AssertionError("no tensor-parallel group")),
    )
    assert current_split_point(4608) == 0


def test_configured_granule_uses_the_tensor_parallel_period(monkeypatch):
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "256")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.k3_ubatch_prefill._tp_world_size", lambda: WORLD
    )
    assert current_split_point(4608) == 2304
    assert current_split_point(3072) == 0


def test_single_rank_has_no_collective_to_keep_invariant(monkeypatch):
    monkeypatch.setenv("B12X_PCIE_RING_GRANULE_ROWS", "256")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.k3_ubatch_prefill._tp_world_size", lambda: 1
    )
    assert current_split_point(3072) == 1536


def test_served_configuration_keeps_the_tile_rule(monkeypatch):
    monkeypatch.delenv("B12X_PCIE_RING_GRANULE_ROWS", raising=False)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.k3_ubatch_prefill._tp_world_size", lambda: WORLD
    )
    assert current_split_point(4608) == 2304
    assert current_split_point(3072) == 1536
    assert current_split_point(1297) == 656
