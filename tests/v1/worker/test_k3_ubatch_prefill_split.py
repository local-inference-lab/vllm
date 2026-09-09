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


def test_mode_file_fails_safe_to_off(monkeypatch, tmp_path):
    """A configured mode file that is missing or empty selects ``off`` even
    when the environment enables the split; a present file selects its
    first word."""
    from vllm.v1.worker.gpu import k3_ubatch_prefill as mod

    path = tmp_path / "mode"
    monkeypatch.setenv("VLLM_K3_UBATCH_MODE_FILE", str(path))
    monkeypatch.setenv("VLLM_K3_UBATCH_PREFILL", "1")
    monkeypatch.setenv("VLLM_K3_UBATCH_PREFILL_OVERLAP", "1")

    def fresh():
        mod._MODE_CACHE[0] = 0.0
        mod._MODE_CACHE[1] = None

    fresh()
    assert mod.runtime_mode() == "off"
    assert not mod.ubatch_prefill_enabled()
    assert not mod.ubatch_prefill_overlap()
    path.write_text("")
    fresh()
    assert mod.runtime_mode() == "off"
    path.write_text("stage1\n")
    fresh()
    assert mod.runtime_mode() == "stage1"
    assert mod.ubatch_prefill_enabled()
    assert not mod.ubatch_prefill_overlap()
    path.write_text("overlap extra words\n")
    fresh()
    assert mod.ubatch_prefill_overlap()
    monkeypatch.delenv("VLLM_K3_UBATCH_MODE_FILE")
    fresh()
    assert mod.runtime_mode() is None
    assert mod.ubatch_prefill_enabled()
    fresh()


def test_configured_split_is_independent_of_boot_mode(tmp_path, monkeypatch):
    """Boot-time preparation keys on whether a split can ever be selected: a
    mode file saying ``off`` at boot still configures the split, while a
    process with neither the enable flag nor a mode file does not."""
    from vllm.v1.worker.gpu import k3_ubatch_prefill as mod

    path = tmp_path / "mode"
    path.write_text("off\n")
    monkeypatch.delenv("VLLM_K3_UBATCH_PREFILL", raising=False)
    monkeypatch.setenv("VLLM_K3_UBATCH_MODE_FILE", str(path))
    mod._MODE_CACHE[0] = 0.0
    mod._MODE_CACHE[1] = None
    assert not mod.ubatch_prefill_enabled()
    assert mod.ubatch_prefill_configured()
    monkeypatch.delenv("VLLM_K3_UBATCH_MODE_FILE")
    assert not mod.ubatch_prefill_configured()
    monkeypatch.setenv("VLLM_K3_UBATCH_PREFILL", "1")
    assert mod.ubatch_prefill_configured()


def test_each_half_keeps_its_own_slot_mapping(monkeypatch):
    """The runner computes slot mappings into one persistent buffer and
    returns a view of it. Both halves are prepared before either runs, so each
    half must run with a copy holding the mapping computed for its own rows,
    not with whatever the buffer holds after the second half was prepared
    (which sent the first half's keys into the second half's cache slots and
    left the first half's slots unwritten)."""
    import dataclasses
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np
    import torch

    from vllm.v1.worker.gpu import k3_ubatch_prefill as mod
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    rows = 2304
    monkeypatch.delenv("VLLM_K3_UBATCH_MODE_FILE", raising=False)
    monkeypatch.delenv("VLLM_K3_UBATCH_PREFILL_OVERLAP", raising=False)
    monkeypatch.delenv("B12X_PCIE_RING_GRANULE_ROWS", raising=False)
    monkeypatch.setattr(mod, "_tp_world_size", lambda: WORLD)
    monkeypatch.setattr(mod, "set_forward_context", lambda *a, **k: nullcontext())
    device = torch.device("cpu")
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=rows, device=device)
    batch = InputBatch.make_dummy(1, rows, buffers)
    batch = dataclasses.replace(
        batch,
        is_prefilling_np=np.array([True]),
        positions=torch.arange(rows, dtype=torch.int64),
        dcp_local_seq_lens=None,
    )

    shared = torch.full((1, rows), -1, dtype=torch.int64)
    calls: list[int] = []

    def prepare_attn(hb):
        # Marker per call: the row position, like a real slot for one page.
        calls.append(int(hb.positions[0]))
        shared[0, : hb.num_tokens] = hb.positions
        return (torch.zeros(1, 4, dtype=torch.int32),), shared[:, : hb.num_tokens]

    model_state = SimpleNamespace(
        preprocess_state=lambda *a, **k: None,
        prepare_attn=lambda *a, **k: {},
        prepare_inputs=lambda *a, **k: {},
    )
    runner = SimpleNamespace(
        prepare_attn=prepare_attn,
        model_state=model_state,
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32))
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]
        ),
        attn_groups=[],
        vllm_config=None,
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        model=None,
    )
    seen: list[torch.Tensor] = []

    def run_half(runner, half_index, hb, half_inputs, slot_mappings_by_layer):
        seen.append(slot_mappings_by_layer["layer"].clone())
        return torch.zeros(hb.num_tokens, 1)

    monkeypatch.setattr(mod, "_run_half", run_half)

    @dataclasses.dataclass
    class Descriptor:
        num_tokens: int

    out = mod.run_split_prefill(
        runner,
        scheduler_output=None,
        input_batch=batch,
        model_inputs={"input_ids": batch.input_ids, "positions": batch.positions},
        cudagraph_runtime_mode=None,
        num_tokens_across_dp=None,
        batch_descriptor=Descriptor(num_tokens=rows),
        skip_compiled=False,
    )
    split = rows // 2
    assert out.shape[0] == rows
    assert calls == [0, split]
    assert torch.equal(seen[0], torch.arange(0, split))
    assert torch.equal(seen[1], torch.arange(split, rows))
