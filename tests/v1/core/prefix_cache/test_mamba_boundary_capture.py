# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen boundary capture for align-mode Mamba + OffloadingConnector.

The connector's boundary store must never source the request's live running
column: a chunked step's column holds the state at the CHUNK end, and the
deferred DMA would read it only after the next forward migrated the column
onward. With capture enabled, each retention crossing the step actually
computes is copied into a dedicated single-writer pool block before the
column advances; that block carries the boundary hash and is the sole store
source, pinned until the store DMA ack.

These tests pin the scheduler-side contract (design invariants I1/I4/I5):
offers appear only after promotion, the offered id is the capture block and
never the column, the capture carries the boundary hash, and pins are
released exactly on store ack / drop / reset.
"""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _free_ids(manager):
    return {
        block.block_id
        for block in manager.block_pool.free_block_queue.get_all_free_blocks()
    }


def _make_capture_manager(*, boundary_capture: bool, num_blocks: int = 16):
    """Hybrid manager with mamba block size 4 and dense retention.

    A 4-token chunk ending on the block boundary is the toy analogue of a
    retention-grid crossing: cache_blocks records it, and only the next
    schedule pass may promote the capture.
    """
    block_size = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    boundary_capture=boundary_capture,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )
    return manager, block_size


def _mamba_manager(manager):
    return manager.coordinator.single_type_managers[1]


def test_capture_offer_names_the_frozen_block_not_the_column():
    """I1/I4: the (group, block, B) hand-off appears only at promotion, and
    its block is the dedicated capture -- never the live column the next
    forward migrates."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    req = make_request("p", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks["p"][0]

    # Recorded, not yet offered: the producing step is still the current one.
    assert "p" in mgr._pending_captures
    assert manager.take_boundary_state_offloads() == {}
    assert manager.take_kv_cache_block_copies() == ([], [])

    # Next schedule pass promotes: copy rail + offer, same scheduler output.
    req.num_computed_tokens = 4
    manager.new_step_starts()

    ((group_id, capture_id, boundary),) = manager.take_boundary_state_offloads()["p"]
    assert group_id == 1
    assert boundary == 4
    assert capture_id != column.block_id

    copies, retained = manager.take_kv_cache_block_copies()
    assert copies == [
        KVCacheBlockCopy(src_block_id=column.block_id, dst_block_id=capture_id)
    ]
    assert {b.block_id for b in retained} == {column.block_id, capture_id}

    # The capture itself carries the boundary hash for its group (local hits
    # past the last live column resume from it too). The live column keeps the
    # same key until it is re-hashed, so assert on the capture's own fields.
    block_hash = req.block_hashes[4 // block_size - 1]
    capture = manager.block_pool.blocks[capture_id]
    assert capture.block_hash == make_block_hash_with_group_id(block_hash, 1)
    assert capture.block_hash_num_tokens == 4

    # Pinned (I4) until the store ack: dropping the rail refs leaves exactly
    # the manager pin; the ack releases the last one.
    manager.block_pool.free_blocks(retained)
    assert capture.ref_cnt == 1
    mgr.release_boundary_capture(capture_id)
    assert capture.ref_cnt == 0
    assert capture_id in _free_ids(manager)

    manager.free(req)


def test_capture_survives_request_teardown_until_store_ack():
    """A promoted capture is pool-owned, not request-owned: aborting the
    producer before the copy step is processed must not release the frozen
    content or its hash; the store ack does."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    req = make_request("r", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks["r"][0]
    column_id = column.block_id

    req.num_computed_tokens = 4
    manager.new_step_starts()
    ((_, capture_id, boundary),) = manager.take_boundary_state_offloads()["r"]
    assert boundary == 4

    # The request dies before the copy step is processed (abort). The capture
    # stays pinned: its content is owed to the CPU tier, and it is no longer
    # reachable through any column.
    manager.free(req)
    assert manager.block_pool.blocks[capture_id].ref_cnt >= 1
    assert manager.block_pool.blocks[capture_id].block_hash is not None

    # Scheduler processed the copy step: rail refs drop; then the store ack.
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    assert manager.block_pool.blocks[capture_id].ref_cnt == 1
    mgr.release_boundary_capture(capture_id)
    assert capture_id in _free_ids(manager)
    # The column's rail ref went with the same release path.
    assert manager.block_pool.blocks[column_id].ref_cnt == 0


def test_no_capture_without_enablement_offers_live_column():
    """The capture path is opt-in: a deployment without boundary_capture
    keeps the legacy live-column hand-off and enqueues no copy."""
    manager, block_size = _make_capture_manager(boundary_capture=False)
    mgr = _mamba_manager(manager)

    req = make_request("x", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks["x"][0]

    ((group_id, block_id, boundary),) = manager.take_boundary_state_offloads()["x"]
    assert (group_id, block_id, boundary) == (1, column.block_id, 4)
    assert manager.take_kv_cache_block_copies() == ([], [])
    assert mgr._pending_captures == {}
    manager.free(req)


def test_release_is_noop_for_unknown_and_non_capture_ids():
    """The connector may pass fenced ids from any store kind; only capture
    pins react."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)
    mgr.release_boundary_capture(12345)  # unknown id: no-op

    req = make_request("n", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks["n"][0]
    column_id = column.block_id

    req.num_computed_tokens = 4
    manager.new_step_starts()
    ((_, capture_id, _),) = manager.take_boundary_state_offloads()["n"]
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)

    # A live column id is never in the pin dict: releasing it must not touch
    # the request's reference.
    before = manager.block_pool.blocks[column_id].ref_cnt
    assert before >= 1
    mgr.release_boundary_capture(column_id)
    assert manager.block_pool.blocks[column_id].ref_cnt == before
    mgr.release_boundary_capture(capture_id)
    assert manager.block_pool.blocks[capture_id].ref_cnt == 0
    manager.free(req)


def test_release_all_drops_pins_and_pending_records():
    """Cache reset releases every pin and every record-time column ref."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    req = make_request("a", list(range(12)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    first_column = mgr.req_to_blocks["a"][0]

    req.num_computed_tokens = 4
    manager.new_step_starts()
    ((_, capture_id, _),) = manager.take_boundary_state_offloads()["a"]
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    assert manager.block_pool.blocks[capture_id].ref_cnt == 1

    # Second chunk [4,8): records a new crossing that is never promoted.
    assert manager.allocate_slots(req, 4) is not None
    assert "a" in mgr._pending_captures
    second_column = mgr.req_to_blocks["a"][1]
    assert second_column.ref_cnt >= 2  # table + record-time ref

    mgr.release_all_boundary_captures()
    assert mgr._capture_pinned == {}
    assert mgr._pending_captures == {}
    assert manager.block_pool.blocks[capture_id].ref_cnt == 0
    # The record-time ref on the second column went with the dropped record.
    assert second_column.ref_cnt == 1
    manager.free(req)
    assert first_column.block_id in _free_ids(manager)


def test_crossing_capture_keyed_identically_across_diverging_prompts():
    """Key-level identity: two prompts sharing the prefix through B=4
    diverge after it; each crossing capture is registered under the SAME
    boundary key (the chain hash at 4), so a consumer of the store for one
    prompt resumes the other. The captures are distinct blocks -- content
    is copied independently of either live column."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    capture_ids = []
    for name, tail in (("p1", 8), ("p2", 9)):
        req = make_request(name, [0, 1, 2, 3, tail, tail + 1], block_size, sha256)
        # Both must PRODUCE a capture at 4: without the skip the second
        # prompt would simply resume from the first prompt's frozen block.
        req.skip_reading_prefix_cache = name == "p2"
        computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
        assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
        req.num_computed_tokens = 4
        manager.new_step_starts()
        ((_, capture_id, boundary),) = manager.take_boundary_state_offloads()[name]
        assert boundary == 4
        capture_ids.append(capture_id)
        _, retained = manager.take_kv_cache_block_copies()
        manager.block_pool.free_blocks(retained)

    first, second = capture_ids
    assert first != second  # dedicated block per producer, never shared
    # Both prompts hashed the same chain at 4 tokens (shared prefix).
    key = req.block_hashes[4 // block_size - 1]
    keyed = make_block_hash_with_group_id(key, 1)
    assert manager.block_pool.blocks[first].block_hash == keyed
    assert manager.block_pool.blocks[second].block_hash == keyed
    # Both pins drop only on the store acks.
    mgr.release_boundary_capture(first)
    mgr.release_boundary_capture(second)
    assert manager.block_pool.blocks[first].ref_cnt == 0
    assert manager.block_pool.blocks[second].ref_cnt == 0


def test_late_record_after_load_credit_is_withheld():
    """Patch A: a crossing whose producing step already completed must not
    be recorded. When cache_blocks only catches up to the boundary after the
    optimistic mirror already sits at it (delayed connector-load credit, the
    probe-#4 late-record class: nct==bnd at record, delta=prompt_final-bnd at
    drain), promotion would copy a column a later forward already advanced
    past the boundary -- state@prompt_final keyed at the boundary, the
    original mislabel defect. The boundary simply keeps no capture and hits
    fall back to recompute."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    req = make_request("d", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    # The producing step runs with caching delayed (connector-load shape).
    assert (
        manager.allocate_slots(
            req, 4, num_computed, computed_blocks, delay_cache_blocks=True
        )
        is not None
    )
    assert mgr._pending_captures == {}

    # Mirror advances to the boundary; a later pass catches cache_blocks up.
    req.num_computed_tokens = 4
    manager.coordinator.cache_blocks(req, 4)
    assert mgr._pending_captures == {}, "late catch-up must not record"

    manager.new_step_starts()
    assert manager.take_boundary_state_offloads() == {}
    manager.free(req)


def test_in_step_record_still_captured():
    """Control for the gate: the step being dispatched now still has the
    boundary ahead of it (nct < boundary) -- the record proceeds and the
    next pass promotes exactly as before."""
    manager, block_size = _make_capture_manager(boundary_capture=True)
    mgr = _mamba_manager(manager)

    req = make_request("c", list(range(8)), block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    assert "c" in mgr._pending_captures
    manager.free(req)
