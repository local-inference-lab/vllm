# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.core.test_prefix_caching import make_request
from vllm.distributed.kv_events import BlockStored
from vllm.multimodal.inputs import PlaceholderRange
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash, maybe_convert_block_hash


@pytest.fixture(autouse=True)
def init_hash():
    init_none_hash(sha256)


@pytest.mark.parametrize("retain_first", [False, True])
def test_context_survives_chunk_gap(retain_first):
    req = make_request("chunks", list(range(16)), 4, sha256)
    pool = BlockPool(5, True, 4, True)
    blocks = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, blocks, 0, 2, 4, 0, [retain_first, False])
    first = pool.take_events()
    assert len(first) == int(retain_first)
    pool.cache_full_blocks(req, blocks, 2, 4, 4, 0, [False, True])
    (event,) = pool.take_events()
    start = 4 if retain_first else 0
    assert event.token_ids == list(range(12, 16))
    assert event.skipped_token_ids == list(range(start, 12))
    assert event.skipped_extra_keys == [None] * ((12 - start) // 4)
    expected_parent = maybe_convert_block_hash(req.block_hashes[0]) if start else None
    assert event.skipped_parent_block_hash == expected_parent
    assert event.block_hashes == [maybe_convert_block_hash(req.block_hashes[3])]


@pytest.mark.parametrize("retain_full", [False, True])
def test_partial_replay_supplies_context(retain_full):
    req = make_request(
        "partial",
        list(range(16)),
        2,
        sha256,
        mm_positions=[PlaceholderRange(offset=8, length=2)],
        mm_hashes=["image"],
        cache_salt="tenant",
    )
    pool = BlockPool(3, True, 2, True)
    blocks = pool.get_new_blocks(2)
    if retain_full:
        pool.cache_full_blocks(req, blocks, 0, 1, 8, 0)
    pool.cache_partial_block(req, blocks[1], 14, 0, 8)
    pool.take_events()
    returned = [blocks[0] if retain_full else pool.null_block, blocks[1]]
    before = [(b.block_hash, b.block_hash_num_tokens, b.ref_cnt) for b in blocks]
    pool.emit_cached_block_events(req, returned, 14, 8, 0)
    events = pool.take_events()
    assert len(events) == 1 + int(retain_full)
    event = events[-1]
    start = 8 if retain_full else 0
    assert isinstance(event, BlockStored)
    assert event.token_ids == [12, 13]
    assert event.block_size == 2
    assert event.skipped_token_ids == list(range(start, 12))
    keys = [("tenant",), None, None, None, (("image", 0),), None]
    assert event.skipped_extra_keys == keys[start // 2 :]
    expected_parent = maybe_convert_block_hash(req.block_hashes[3]) if start else None
    assert event.skipped_parent_block_hash == expected_parent
    assert event.parent_block_hash == maybe_convert_block_hash(req.block_hashes[5])
    assert event.block_hashes == [maybe_convert_block_hash(req.block_hashes[6])]
    assert [
        (b.block_hash, b.block_hash_num_tokens, b.ref_cnt) for b in blocks
    ] == before


@pytest.mark.parametrize("boundary", ["group", "request", "reset"])
def test_context_stays_with_its_request_group_and_cache(boundary):
    req = make_request("reused-id", list(range(8)), 4, sha256)
    pool = BlockPool(5, True, 4, True)
    first = pool.get_new_blocks(1)
    pool.cache_full_blocks(req, first, 0, 1, 4, 0)
    pool.take_events()
    group = int(boundary == "group")
    if boundary == "request":
        req = make_request("reused-id", list(range(8)), 4, sha256)
    if boundary == "reset":
        pool.free_blocks(first)
        assert pool.reset_prefix_cache()
        pool.take_events()
    tail = pool.get_new_blocks(1)[0]
    pool.cache_full_blocks(req, [pool.null_block, tail], 1, 2, 4, group)
    (event,) = pool.take_events()
    assert event.skipped_parent_block_hash is None
    assert event.skipped_token_ids == list(range(4))


def test_partial_store_supplies_context():
    req = make_request("partial-store", list(range(16)), 2, sha256)
    pool = BlockPool(2, True, 2, True)
    block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(req, block, 14, 0, 8)
    (event,) = pool.take_events()
    assert event.token_ids == [12, 13]
    assert event.skipped_parent_block_hash is None
    assert event.skipped_token_ids == list(range(12))
    assert event.skipped_extra_keys == [None] * 6


def test_stale_anchor_restarts_context_from_root():
    req = make_request("stale-anchor", list(range(16)), 4, sha256)
    pool = BlockPool(5, True, 4, True)
    blocks = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, blocks, 0, 2, 4, 0)
    pool.take_events()
    # Streaming updates truncate a session in place and rebuild its hash
    # chain (scheduler._update_request_as_session), which can change the
    # request's block hashes under a saved anchor. A stale anchor must not
    # suppress skipped context, so validation restarts from the root.
    req._all_token_ids[1] = 999
    req.block_hashes = []
    req.update_block_hashes()
    pool.cache_full_blocks(req, blocks, 2, 4, 4, 0, [False, True])
    (event,) = pool.take_events()
    assert event.skipped_parent_block_hash is None
    assert event.skipped_token_ids == req.all_token_ids[:12]
    assert event.skipped_extra_keys == [None] * 3
    assert event.block_hashes == [maybe_convert_block_hash(req.block_hashes[3])]


def test_partial_only_replay_ignores_saved_anchor():
    req = make_request("partial-only-replay", list(range(16)), 2, sha256)
    pool = BlockPool(3, True, 2, True)
    full_block, partial_block = pool.get_new_blocks(2)
    pool.cache_full_blocks(req, [full_block], 0, 1, 8, 0)
    pool.cache_partial_block(req, partial_block, 14, 0, 8)
    pool.take_events()

    pool.emit_cached_block_events(req, [pool.null_block, partial_block], 14, 8, 0)
    (event,) = pool.take_events()
    assert event.token_ids == [12, 13]
    assert event.skipped_token_ids == list(range(12))
    assert event.skipped_parent_block_hash is None
    assert event.skipped_extra_keys == [None] * 6


@pytest.mark.parametrize("continuation", ["full", "partial"])
def test_full_replay_advances_incremental_context(continuation):
    producer = make_request("producer", list(range(32)), 2, sha256)
    req = make_request("replay-continuation", list(range(32)), 2, sha256)
    pool = BlockPool(4, True, 2, True)
    full_block, partial_block, tail = pool.get_new_blocks(3)
    pool.cache_full_blocks(producer, [full_block], 0, 1, 8, 0)
    pool.take_events()

    pool.emit_cached_block_events(req, [full_block], 8, 8, 0)
    (replay,) = pool.take_events()
    assert replay.token_ids == list(range(8))
    assert replay.skipped_token_ids is None
    if continuation == "partial":
        pool.cache_partial_block(req, partial_block, 14, 0, 8)
        (partial,) = pool.take_events()
        assert partial.token_ids == [12, 13]
        assert partial.skipped_token_ids == list(range(8, 12))
        assert partial.skipped_extra_keys == [None] * 2
        assert partial.skipped_parent_block_hash == maybe_convert_block_hash(
            req.block_hashes[3]
        )

    pool.cache_full_blocks(
        req, [full_block, pool.null_block, pool.null_block, tail], 1, 4, 8, 0
    )
    (event,) = pool.take_events()
    assert event.token_ids == list(range(24, 32))
    assert event.skipped_token_ids == list(range(8, 24))
    assert event.skipped_extra_keys == [None] * 2
    assert event.skipped_parent_block_hash == maybe_convert_block_hash(
        req.block_hashes[3]
    )
