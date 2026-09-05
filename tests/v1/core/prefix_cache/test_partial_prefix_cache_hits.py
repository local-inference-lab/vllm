# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fine-grained partial prefix-cache hits for hybrid (full attention + mamba
"align") models: scheduler chunk splitting, partial tail registration, CoW
on partial hits, and same-step deferral."""

import unittest
from dataclasses import replace
from math import lcm
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_block_hash,
    get_group_id,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def test_connector_without_divergent_hit_support_uses_common_lookup():
    common_blocks = MagicMock()
    manager = MagicMock()
    manager.get_computed_blocks.return_value = (common_blocks, 0, 0)
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(supports_divergent_local_hybrid_hits=False),
        kv_cache_manager=manager,
    )

    result = Scheduler._get_local_prefix_cache_hit(scheduler, MagicMock())

    assert result == (common_blocks, 0, 0, False)
    manager.get_computed_blocks_for_connector.assert_not_called()


def test_capable_connector_uses_divergent_partial_hit_lookup():
    per_group_blocks = MagicMock()
    manager = MagicMock()
    manager.get_computed_blocks_for_connector.return_value = (
        per_group_blocks,
        6,
        0,
        True,
    )
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(supports_divergent_local_hybrid_hits=True),
        kv_cache_manager=manager,
    )

    result = Scheduler._get_local_prefix_cache_hit(scheduler, MagicMock())

    assert result == (per_group_blocks, 6, 0, True)
    manager.get_computed_blocks.assert_not_called()


def make_full_mamba_manager(
    *,
    dcp_world_size: int,
    hash_block_size: int = 2,
    full_block_size: int = 4,
    mamba_block_size: int = 4,
    num_blocks: int = 32,
    use_eagle: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
):
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=full_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                ),
            ),
        ],
    )
    scheduler_block_size = lcm(
        full_block_size * dcp_world_size,
        mamba_block_size,
    )
    return make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        dcp_world_size=dcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
    )


@pytest.mark.parametrize("dcp_world_size", [1, 4])
def test_mamba_align_split_partial_tail_schedule(dcp_world_size: int):
    """Chunk ends with partial hits on: block-aligned chunks, one extra stop
    at the prompt's last hash boundary (registering the partial tail), then
    the remaining tokens. block=512, hash=32, prompt=10000, budget=8192:
    0 -> 8192 -> 9728 -> 9984 -> 10000."""
    block_size = 512
    scheduler_block_size = block_size * dcp_world_size
    hash_block_size = 32
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        drop_last_prefix_cache_block=False,
        hash_block_size=hash_block_size,
        dcp_world_size=dcp_world_size,
        scheduler_block_size=scheduler_block_size,
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    split = Scheduler._mamba_block_aligned_split

    req = make_request("0", [0] * 10000, hash_block_size, sha256)
    req.num_computed_tokens = 0
    assert split(self=mock, request=req, num_new_tokens=8192) == 8192
    req.num_computed_tokens = 8192
    # Stop at the last block boundary (9728).
    assert split(self=mock, request=req, num_new_tokens=1808) == 1536
    req.num_computed_tokens = 9728
    # Extra stop at the prompt's last hash boundary (9984).
    assert split(self=mock, request=req, num_new_tokens=272) == 256
    req.num_computed_tokens = 9984
    # Final 16 tokens run unchanged (no mid-block-resume stop: the next
    # block boundary is past the last block boundary).
    assert split(self=mock, request=req, num_new_tokens=16) == 16

    # Partial hits off: no extra stop, the tail runs in one chunk.
    mock.mamba_partial_cache_hit = False
    req.num_computed_tokens = 9728
    assert split(self=mock, request=req, num_new_tokens=272) == 272
    mock.mamba_partial_cache_hit = True

    # A request resumed mid-block (partial hash hit at 9984): the first chunk
    # stops at the next block boundary (10240), later chunk ends re-align.
    req2 = make_request("1", [0] * 12000, hash_block_size, sha256)
    req2.num_computed_tokens = 9984
    assert req2.num_computed_tokens % scheduler_block_size != 0
    assert split(self=mock, request=req2, num_new_tokens=2016) == 256
    req2.num_computed_tokens = 10240
    assert split(self=mock, request=req2, num_new_tokens=1000) == 512


def test_mamba_align_split_when_block_exceeds_scheduling_budget():
    """Sub-block chunks make progress only when no step can fit a full block."""
    block_size = 11392
    token_budget = 8192
    prompt_length = 30000
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=token_budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        drop_last_prefix_cache_block=False,
        hash_block_size=32,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    mock.max_num_scheduled_tokens = block_size
    assert split(self=mock, request=req, num_new_tokens=token_budget) == 0
    mock.max_num_scheduled_tokens = token_budget

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(token_budget, prompt_length - req.num_computed_tokens)
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= token_budget
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [8192, 3200, 8192, 3200, 7216]


def test_mamba_align_split_when_block_exceeds_long_prefill_threshold():
    """A long-prefill cap below the block size permits sub-block progress."""
    block_size = 512
    token_budget = 8192
    long_prefill_threshold = 384
    prompt_length = 1300
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        max_num_scheduled_tokens=token_budget,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=long_prefill_threshold
        ),
        use_eagle=False,
        drop_last_prefix_cache_block=False,
        hash_block_size=32,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
    )
    req = make_request("0", [0] * prompt_length, 32, sha256)
    split = Scheduler._mamba_block_aligned_split

    scheduled_chunks = []
    while req.num_computed_tokens < prompt_length:
        num_new_tokens = min(
            long_prefill_threshold,
            prompt_length - req.num_computed_tokens,
        )
        num_scheduled_tokens = split(
            self=mock,
            request=req,
            num_new_tokens=num_new_tokens,
        )
        assert 0 < num_scheduled_tokens <= long_prefill_threshold
        scheduled_chunks.append(num_scheduled_tokens)
        req.num_computed_tokens += num_scheduled_tokens

    assert scheduled_chunks == [384, 128, 384, 128, 276]


def test_hybrid_mamba_align_partial_hash_hit():
    hash_block_size = 2
    mamba_block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    blocks = manager.allocate_slots(req0, 6, num_computed, computed_blocks)
    assert blocks is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    assert partial_mamba_block[0].block_hash_num_tokens == 6

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [3, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block[0].block_id
    assert manager.get_blocks("1").get_block_ids()[1][1] == mamba_new_block_ids[0]
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_mamba_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block[0].block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )
    assert manager.get_blocks("1").blocks[1][1].block_hash_num_tokens == 8


def test_eagle_group_registers_unaligned_tail_under_partial_hash_hits():
    """An EAGLE group must not re-floor what partial hash hits leaves un-floored.

    ``cache_blocks`` decides once how far a request may be registered, and with
    fine-grained partial hash hits that bound is the raw token count. The EAGLE
    branch then re-derives its own bound for the lookahead block; if it rounds
    down to ``scheduler_block_size`` again, everything between the last aligned
    boundary and the tail stops being registered -- ``(n % scheduler_block_size)
    - manager.block_size`` tokens per call, which is most of a segment whenever
    the group's own block is much smaller than the scheduler block.
    """
    hash_block_size = 2
    mamba_block_size = 4 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=40,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    coordinator = manager.coordinator
    assert coordinator.enable_partial_hash_hits
    # The full-attention group is the EAGLE one, and its block is smaller than
    # the scheduler block -- the geometry where re-flooring loses tokens.
    eagle_manager = coordinator.single_type_managers[0]
    eagle_manager.use_eagle = True
    assert eagle_manager.block_size < coordinator.scheduler_block_size

    # Deliberately not a multiple of the scheduler block, so the two bounds
    # differ: floor(22/8)*8 + 2 = 18 against 22.
    num_tokens = coordinator.scheduler_block_size * 2 + hash_block_size * 3
    req = make_request("0", list(range(num_tokens)), hash_block_size, sha256)

    recorded: list[int] = []
    for single_type_manager in coordinator.single_type_managers:
        original = single_type_manager.cache_blocks

        def spy(request, num_tokens_to_cache, *args, _orig=original, **kwargs):
            recorded.append(num_tokens_to_cache)
            return _orig(request, num_tokens_to_cache, *args, **kwargs)

        single_type_manager.cache_blocks = spy

    # allocate_slots caches on the way out, so this exercises the real path.
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, num_tokens, num_computed, computed_blocks)

    # Every group, EAGLE or not, may register the whole unaligned tail.
    assert recorded == [num_tokens] * len(coordinator.single_type_managers)


def test_hybrid_mamba_partial_tail_owner_uses_cow_on_continue():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    new_blocks = manager.allocate_slots(req0, 1)
    assert new_blocks is not None

    # Reversed CoW for the owning request: it keeps its own block (the
    # worker's block table is append-only), and no new mamba block is handed
    # to the worker. The prefix-cache entry is moved to a private copy that
    # the queued block copy fills before the next forward.
    assert new_blocks.get_block_ids()[1] == []
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == partial_mamba_block_id)
    assert cow_copy.dst_block_id != partial_mamba_block_id
    # The source block gave up the hash; the copy target now owns the entry.
    assert partial_mamba_block[0].block_hash is None
    moved = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id
    assert get_block_hash(moved[0].block_hash) == partial_mamba_hash
    assert get_group_id(moved[0].block_hash) == 1
    assert moved[0].block_hash_num_tokens == 6


def test_partial_hit_then_internal_checkpoint_uses_distinct_mamba_blocks():
    hash_block_size = 2
    mamba_block_size = 4
    manager = make_full_mamba_manager(
        dcp_world_size=1,
        hash_block_size=hash_block_size,
        full_block_size=hash_block_size,
        mamba_block_size=mamba_block_size,
        num_prefill_checkpoint_blocks=1,
    )

    owner = make_request("owner", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(owner)
    assert manager.allocate_slots(owner, 6, num_computed, computed_blocks) is not None
    manager.free(owner)
    manager.new_step_starts()

    partial_hash = owner.block_hashes[2]
    partial_block = manager.block_pool.get_cached_block(partial_hash, [1])
    assert partial_block is not None
    partial_block_id = partial_block[0].block_id

    replay = make_request(
        "replay",
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        hash_block_size,
        sha256,
    )
    computed_blocks, num_computed, _ = manager.get_computed_blocks(replay)
    assert num_computed == 6

    realigned_blocks = manager.allocate_slots(replay, 2, num_computed, computed_blocks)
    assert realigned_blocks is not None
    mamba_cow_block_id = realigned_blocks.get_block_ids()[1][0]
    copies, retained = manager.take_kv_cache_block_copies()
    assert KVCacheBlockCopy(partial_block_id, mamba_cow_block_id) in copies
    manager.block_pool.free_blocks(retained)

    replay.num_computed_tokens = 8
    manager.new_step_starts()
    final_blocks = manager.allocate_slots(replay, 6)
    assert final_blocks is not None

    checkpoint_block_id, running_block_id = final_blocks.get_block_ids()[1]
    assert len({mamba_cow_block_id, checkpoint_block_id, running_block_id}) == 3
    mamba_blocks = manager.get_blocks(replay.request_id).blocks[1]
    assert mamba_blocks[2].block_id == checkpoint_block_id
    assert mamba_blocks[3].block_id == running_block_id


def test_internal_checkpoint_requires_block_aligned_start():
    hash_block_size = 2
    mamba_block_size = 16
    manager = make_full_mamba_manager(
        dcp_world_size=1,
        hash_block_size=hash_block_size,
        full_block_size=hash_block_size,
        mamba_block_size=mamba_block_size,
        num_prefill_checkpoint_blocks=1,
    )
    request = make_request("producer", list(range(50)), hash_block_size, sha256)

    # Compute token 0 first, so the next query starts at token 1, which is not
    # aligned to the Mamba block size.
    assert manager.allocate_slots(request, 1) is not None
    request.num_computed_tokens = 1
    manager.new_step_starts()

    new_blocks = manager.allocate_slots(request, 49)

    assert new_blocks is not None
    mamba_blocks = manager.get_blocks(request.request_id).blocks[1]
    checkpoint_block_idx = 48 // mamba_block_size - 1
    assert mamba_blocks[checkpoint_block_idx].is_null
    assert not mamba_blocks[-1].is_null
    checkpoint_hash = request.block_hashes[48 // hash_block_size - 1]
    assert manager.block_pool.get_cached_block(checkpoint_hash, [1]) is None


def test_external_mamba_hit_same_block_uses_running_cow_on_continue():
    """An external mid-block hit must become a running request even when its
    first continuation does not need another Mamba block."""
    hash_block_size = 2
    mamba_block_size = 4 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    request = make_request("0", [0] * 15, hash_block_size, sha256)
    loaded_blocks = manager.allocate_slots(
        request,
        num_new_tokens=0,
        num_external_computed_tokens=10,
        delay_cache_blocks=True,
    )
    assert loaded_blocks is not None

    request.num_computed_tokens = 10
    first_step_blocks = manager.allocate_slots(request, num_new_tokens=4)
    assert first_step_blocks is not None

    source_block_id = manager.get_blocks("0").get_block_ids()[1][1]
    partial_hash = request.block_hashes[14 // hash_block_size - 1]
    partial_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[1]
    )
    assert partial_block is not None
    assert partial_block[0].block_id == source_block_id

    request.num_computed_tokens = 14
    continuation_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert continuation_blocks is not None

    assert continuation_blocks.get_block_ids()[1] == []
    assert manager.get_blocks("0").get_block_ids()[1][1] == source_block_id
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == source_block_id)
    assert cow_copy.dst_block_id != source_block_id

    moved = manager.block_pool.get_cached_block(partial_hash, kv_cache_group_ids=[1])
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id


def test_take_partial_tail_offloads_returns_cow_target():
    """The connector offload hand-off exposes the mamba CoW *target* block Y
    (the durable boundary state), not the overwritten source X, and only at
    the CoW step."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                    num_prefill_checkpoint_blocks=1,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # Step A registered the partial tail but has not CoW'd yet: no offload.
    assert manager.take_partial_tail_offloads() == {}

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    source_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert source_block is not None
    source_block_id = source_block[0].block_id

    # Step B: the producer continues, triggering the CoW X->Y.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    offloads = manager.take_partial_tail_offloads()
    assert list(offloads.keys()) == ["0"]
    assert len(offloads["0"]) == 1
    group_id, block_id, boundary_tokens = offloads["0"][0]
    assert group_id == 1  # the mamba group
    assert boundary_tokens == 6
    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in copies if c.src_block_id == source_block_id)
    # The offload points at the durable CoW target Y, not the overwritten X.
    assert block_id == cow_copy.dst_block_id
    assert block_id != source_block_id
    # Draining clears it.
    assert manager.take_partial_tail_offloads() == {}

    # The hand-off pinned Y (its CoW retention is released after this step,
    # and Y is off the request block table); freeing the request unpins it.
    cow_block = manager.block_pool.blocks[block_id]
    pinned_ref = cow_block.ref_cnt
    assert pinned_ref >= 1
    manager.free(req0)
    assert cow_block.ref_cnt == pinned_ref - 1


def test_partial_tail_pin_survives_released_cow_retention():
    """If the CoW retention is released before the hand-off is drained
    (immediate-free mode), the drain must rescue the cow block from the free
    queue: a raw ref increment would leave a ref>0 block allocatable, and the
    next allocation would pop it and assert."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # Retention released before the drain (defer_block_free=False ordering).
    _copies, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)

    offloads = manager.take_partial_tail_offloads()
    ((_group_id, block_id, boundary_tokens),) = offloads["0"]
    assert boundary_tokens == 6
    cow_block = manager.block_pool.blocks[block_id]
    assert cow_block.ref_cnt == 1

    # The pinned block is out of the free queue: draining every free block
    # neither trips the allocator's ref_cnt assert nor hands it out.
    new_blocks = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks()
    )
    assert block_id not in {b.block_id for b in new_blocks}


def test_partial_tail_offload_dropped_when_request_freed_before_drain():
    """A hand-off recorded in the same scheduling pass as the request's death
    must not be drained: its release hook has already run, so draining would
    leak a pinned block."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # The request dies (preempt/abort) before the scheduler drains.
    manager.block_pool.free_blocks(manager.pop_blocks_for_free(req0))
    assert manager.take_partial_tail_offloads() == {}


def test_take_partial_tail_offloads_empty_without_partial_tail():
    """A prompt ending on a block boundary registers no partial tail, so there
    is nothing to offload."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    # 4-token prompt ends exactly on the mamba block boundary (block_size=4).
    req0 = make_request("0", [0, 0, 1, 1], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    assert manager.take_partial_tail_offloads() == {}

    req0.num_computed_tokens = 4
    req0.append_output_token_ids([2])
    assert manager.allocate_slots(req0, 1) is not None
    assert manager.take_partial_tail_offloads() == {}


def test_truncate_computed_blocks_preserves_sparse_prefix_positions():
    """truncate_computed_blocks slices each group by its own block size,
    keeps null placeholders in the retained prefix, and leaves the original
    lookup result untouched (pure view, no refcount changes)."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=2 * hash_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    producer = make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 6, num_computed, blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request(
        "consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256
    )
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in blocks.blocks] == [3, 2]
    assert blocks.blocks[1][0].is_null

    truncated = manager.truncate_computed_blocks(blocks, 4)

    assert [len(group) for group in truncated.blocks] == [2, 1]
    assert truncated.blocks[1][0].is_null
    assert [len(group) for group in blocks.blocks] == [3, 2]


def test_truncate_computed_blocks_allows_short_mamba_group_only():
    """External state may replace a short Mamba hit, but other groups must
    cover the aligned local endpoint."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=2 * hash_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    producer = make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, 6, num_computed, blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request(
        "consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256
    )
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in blocks.blocks] == [3, 2]

    short_mamba = manager.create_kv_cache_blocks((list(blocks.blocks[0]), []))
    truncated = manager.truncate_computed_blocks(short_mamba, 4)
    assert [len(group) for group in truncated.blocks] == [2, 0]

    short_full_attention = manager.create_kv_cache_blocks(
        (list(blocks.blocks[0][:1]), list(blocks.blocks[1]))
    )
    with pytest.raises(AssertionError):
        manager.truncate_computed_blocks(short_full_attention, 4)

    with pytest.raises(AssertionError):
        manager.truncate_computed_blocks(blocks, 6)

    # The lookup result itself is never mutated.
    assert [len(group) for group in blocks.blocks] == [3, 2]


def test_hybrid_mamba_partial_tail_owner_continue_preserves_later_hit():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    # The owner moved the prefix-cache entry to a private copy; capture its id.
    owner_copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(c for c in owner_copies if c.src_block_id == partial_mamba_block_id)
    moved_block_id = cow_copy.dst_block_id
    manager.new_step_starts()

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    # The later request hits the moved (private-copy) entry, not the source.
    assert computed_blocks.get_block_ids()[1][1] == moved_block_id

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != moved_block_id
    # The hitting request CoWs from the moved entry into its own private block.
    copies, _ = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=moved_block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in copies
    )


def test_hybrid_mamba_moved_partial_entry_defers_same_step_hit():
    """The owner's move re-arms the same-step guard: the moved entry is
    filled by this step's copy, and chained same-step copies read stale
    sources, so a request hitting it in the move step must be deferred."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.new_step_starts()

    # The owning request continues decoding: the partial entry moves to a
    # private copy in this step.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None

    # A request hitting the moved entry in the SAME step must be deferred.
    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is None

    # Next step the moved entry is consumable.
    manager.new_step_starts()
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert manager.allocate_slots(req1, 2, num_computed, computed_blocks) is not None


def test_hybrid_full_attention_partial_hash_hit_uses_cow():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_full_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_full_hash, kv_cache_group_ids=[0]
    )
    assert partial_full_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_ids = new_blocks.get_block_ids()[0]
    assert len(full_new_block_ids) == 1
    assert full_new_block_ids[0] != partial_full_block[0].block_id
    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_full_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    copies, retained = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_full_block[0].block_id,
            dst_block_id=full_new_block_ids[0],
        )
        in copies
    )
    assert partial_full_block[0].ref_cnt == 1
    manager.block_pool.free_blocks(retained)
    assert partial_full_block[0].ref_cnt == 0


def test_hybrid_partial_hit_cow_target_starts_uncached():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[0]
    )
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_hash, kv_cache_group_ids=[1]
    )
    assert partial_full_block is not None
    assert partial_mamba_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6

    new_blocks = manager.allocate_slots(
        req1,
        2,
        num_computed,
        computed_blocks,
        delay_cache_blocks=True,
    )
    assert new_blocks is not None

    full_cow_block = manager.get_blocks("1").blocks[0][1]
    mamba_cow_block = manager.get_blocks("1").blocks[1][1]
    assert full_cow_block.block_id != partial_full_block[0].block_id
    assert mamba_cow_block.block_id != partial_mamba_block[0].block_id
    assert full_cow_block.block_hash is None
    assert full_cow_block.block_hash_num_tokens is None
    assert mamba_cow_block.block_hash is None
    assert mamba_cow_block.block_hash_num_tokens is None

    assert partial_full_block[0].block_hash is not None
    assert get_block_hash(partial_full_block[0].block_hash) == partial_hash
    assert get_group_id(partial_full_block[0].block_hash) == 0
    assert partial_full_block[0].block_hash_num_tokens == 6
    assert partial_mamba_block[0].block_hash is not None
    assert get_block_hash(partial_mamba_block[0].block_hash) == partial_hash
    assert get_group_id(partial_mamba_block[0].block_hash) == 1
    assert partial_mamba_block[0].block_hash_num_tokens == 6


def test_hybrid_partial_hash_truncates_full_attention_hit_length():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    pool = manager.block_pool
    req = make_request(
        "0",
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        hash_block_size,
        sha256,
    )

    full_blocks = pool.get_new_blocks(3)
    pool.cache_full_blocks(
        request=req,
        blocks=full_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    pool.cache_partial_block(
        request=req,
        block=full_blocks[2],
        num_tokens=10,
        kv_cache_group_id=0,
        block_size=block_size,
    )

    mamba_block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(
        request=req,
        block=mamba_block,
        num_tokens=6,
        kv_cache_group_id=1,
        block_size=block_size,
    )

    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]


def test_cow_retained_blocks_returned_for_release():
    """new_step_starts returns the CoW copy retentions instead of freeing
    them; the scheduler owns releasing them once the copy has run."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    # The owner's move queues a copy and retains both endpoints.
    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    (cow_copy,), retained = manager.take_kv_cache_block_copies()
    assert {b.block_id for b in retained} == {
        cow_copy.src_block_id,
        cow_copy.dst_block_id,
    }
    # Not freed yet: the retention refs are still held.
    assert all(b.ref_cnt > 0 for b in retained)
    manager.block_pool.free_blocks(retained)


def test_free_cow_retained_blocks_defers_until_copy_step_processed():
    """Scheduler releases CoW retentions immediately when the copy's step has
    been processed (or deferral is off), and defers them otherwise."""
    from collections import deque

    freed: list = []
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=9)]
    mock = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(free_blocks=freed.extend)
        ),
        deferred_frees=deque(),
        defer_block_free=True,
        processed_step_seq=2,
    )
    free = Scheduler._free_cow_retained_blocks

    # Copy step still in flight: deferred with its fence.
    free(mock, list(blocks), fence_seq=3)
    assert not freed
    assert mock.deferred_frees == deque([(3, blocks[::-1])])

    # Copy step processed: freed immediately.
    mock.processed_step_seq = 3
    free(mock, list(blocks), fence_seq=3)
    assert freed == blocks

    # Deferral disabled: freed immediately regardless of the fence.
    freed.clear()
    mock.deferred_frees.clear()
    mock.defer_block_free = False
    mock.processed_step_seq = 0
    free(mock, list(blocks), fence_seq=3)
    assert freed == blocks


def test_full_attention_eagle_drops_one_hash_unit():
    """With fine-grained partial hits, eagle rewinds the hit by one hash unit
    instead of a whole cache block: the tail block's KV is append-only, so it
    still covers the reduced length and stays in the hit as a partial block."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    hash_block_size = 2
    block_size = 4
    pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=hash_block_size
    )
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    req = make_request("0", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)

    def find(drop_eagle_block):
        return FullAttentionManager.find_longest_cache_hit(
            block_hashes=req.block_hashes,
            max_length=8,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=hash_block_size,
        )

    # Two full cached blocks (hit 8): eagle rewinds to 6, keeping the last
    # block as a partial hit instead of dropping it to 4.
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    hit_blocks, hit_length = find(drop_eagle_block=False)
    assert (hit_length, len(hit_blocks[0])) == (8, 2)
    hit_blocks, hit_length = find(drop_eagle_block=True)
    assert (hit_length, len(hit_blocks[0])) == (6, 2)

    # A partial tail at 6 (block 1 not fully cached): eagle rewinds to the
    # block boundary and trims the tail block.
    pool2 = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=hash_block_size
    )
    pool = pool2
    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks[:1],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    assert (
        pool.cache_partial_block(
            request=req,
            block=blocks[1],
            num_tokens=6,
            kv_cache_group_id=0,
            block_size=block_size,
        )
        is not None
    )
    hit_blocks, hit_length = find(drop_eagle_block=False)
    assert (hit_length, len(hit_blocks[0])) == (6, 2)
    hit_blocks, hit_length = find(drop_eagle_block=True)
    assert (hit_length, len(hit_blocks[0])) == (4, 1)


def test_hybrid_partial_hit_with_eagle_stays_within_group_blocks():
    """Regression: with eagle, the mamba group must not receive the eagle
    lookup margin — its finder never applies the drop, so it could return a
    hit past the blocks the (dropped) full-attention group covers, crashing
    the consumer's CoW with block_idx >= len(req_blocks)."""
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
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
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )

    # The owner prefills in scheduler-split style: stop at the block boundary
    # (4), then at the prompt's last hash boundary (6, partial entries).
    req0 = make_request("0", [7] * 6, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 4
    manager.new_step_starts()
    assert manager.allocate_slots(req0, 2) is not None
    req0.num_computed_tokens = 6
    manager.new_step_starts()

    # A longer request with eagle: full attention drops the partial tail, so
    # the joint hit must fall back to the block boundary the FA blocks cover.
    req1 = make_request("1", [7] * 6 + [9] * 2, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 4
    assert all(
        len(group) * block_size >= num_computed for group in computed_blocks.blocks
    )
    assert manager.allocate_slots(req1, 4, num_computed, computed_blocks) is not None


def _make_hybrid_swa_eagle_config(
    hash_block_size: int,
    attn_block_size: int,
    mamba_block_size: int,
    sliding_window: int,
    num_blocks: int = 512,
) -> KVCacheConfig:
    """Full attention target + mamba "align" + an EAGLE sliding-window draft
    group (window == draft block), the GLM-5.3 + DFlash topology."""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=attn_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                ["swa_draft"],
                SlidingWindowSpec(
                    block_size=attn_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
                is_eagle_group=True,
            ),
        ],
    )


def _prefill_and_free(manager, request, chunk: int) -> int:
    """Prefill ``request`` in ``chunk``-token steps (so the mamba group
    checkpoints every block, as FlashKDA prefill checkpoints do), decode one
    token, free it. Returns the prefix-cache hit it started from."""
    num_tokens = len(request.prompt_token_ids)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(request)
    done = num_computed
    first = True
    while done < num_tokens:
        step = min(chunk, num_tokens - done)
        if first:
            result = manager.allocate_slots(
                request, step, num_computed, computed_blocks
            )
            first = False
        else:
            result = manager.allocate_slots(request, step)
        assert result is not None
        done += step
        request.num_computed_tokens = done
        manager.new_step_starts()
    assert manager.allocate_slots(request, 1) is not None
    request.num_computed_tokens = done + 1
    manager.new_step_starts()
    manager.free(request)
    manager.new_step_starts()
    return num_computed


def test_hybrid_sliding_window_group_enables_partial_hash_hits():
    """A sliding-window draft group no longer forces block-aligned hits: with
    the hash unit finer than the draft block, partial hash hits stay enabled
    and the draft's EAGLE rewind is one hash unit, not one draft block."""
    hash_block_size = 2
    sliding_window_block_size = 2 * hash_block_size
    mamba_block_size = 2 * sliding_window_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                ["swa_draft"],
                SlidingWindowSpec(
                    block_size=sliding_window_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window_block_size,
                ),
                is_eagle_group=True,
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )
    coordinator = manager.coordinator
    assert coordinator.enable_partial_hash_hits
    assert all(
        m.hit_alignment_tokens == hash_block_size
        for m in coordinator.single_type_managers
    )

    tokens = list(range(3 * sliding_window_block_size))
    request = make_request("0", tokens, hash_block_size, sha256)
    # Prefill in mamba-block steps, as the aligned scheduler split does.
    assert _prefill_and_free(manager, request, mamba_block_size) == 0

    cached_request = make_request(
        "1", tokens + [len(tokens), len(tokens) + 1], hash_block_size, sha256
    )
    computed_blocks, num_computed, _ = manager.get_computed_blocks(cached_request)
    # The mamba state at the last block boundary (8) is retained; the draft
    # matches one hash unit past it (10, inside its cached block 2) and
    # rewinds by that unit instead of by a whole draft block.
    assert num_computed == mamba_block_size
    assert len(computed_blocks.blocks[0]) * hash_block_size == num_computed


def test_sliding_window_fine_grained_hits():
    """Manager-level: hits on hash boundaries inside a draft block, the
    window-coverage requirement, the EAGLE rewind by one hash unit, and a
    covering entry registered further along the tail block."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    hash_block_size = 2
    block_size = 8
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size,
    )
    pool = BlockPool(
        num_gpu_blocks=16, enable_caching=True, hash_block_size=hash_block_size
    )
    req = make_request("0", list(range(100, 116)), hash_block_size, sha256)

    def find(max_length, drop_eagle_block):
        return SlidingWindowManager.find_longest_cache_hit(
            block_hashes=req.block_hashes,
            max_length=max_length,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=hash_block_size,
        )

    blocks = pool.get_new_blocks(2)
    # Block 0 fully cached (8 tokens); block 1 holds a partial entry at 14.
    pool.cache_full_blocks(
        request=req,
        blocks=blocks[:1],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    assert (
        pool.cache_partial_block(
            request=req,
            block=blocks[1],
            num_tokens=14,
            kv_cache_group_id=0,
            block_size=block_size,
        )
        is not None
    )

    # Hit at the registered partial boundary; window [7, 14) needs block 0.
    hit_blocks, hit_length = find(16, drop_eagle_block=False)
    assert (hit_length, [b.block_id for b in hit_blocks[0]]) == (
        14,
        [blocks[0].block_id, blocks[1].block_id],
    )
    # EAGLE rewinds one hash unit; the tail block still covers 12.
    hit_blocks, hit_length = find(16, drop_eagle_block=True)
    assert (hit_length, len(hit_blocks[0])) == (12, 2)
    # A re-query at the rewound length (no entry at 12) is served by the
    # covering entry at 14 of the same append-only tail block.
    hit_blocks, hit_length = find(12, drop_eagle_block=False)
    assert (hit_length, len(hit_blocks[0])) == (12, 2)
    # Block-aligned boundary: the full block 0 is the tail.
    hit_blocks, hit_length = find(8, drop_eagle_block=False)
    assert (hit_length, [b.block_id for b in hit_blocks[0]]) == (
        8,
        [blocks[0].block_id],
    )
    # Window coverage: with only the partial tail cached (no block 0), the
    # tail alone cannot serve 14 since its window reaches into block 0.
    pool = BlockPool(
        num_gpu_blocks=16, enable_caching=True, hash_block_size=hash_block_size
    )
    blocks = pool.get_new_blocks(2)
    assert (
        pool.cache_partial_block(
            request=req,
            block=blocks[1],
            num_tokens=14,
            kv_cache_group_id=0,
            block_size=block_size,
        )
        is not None
    )
    hit_blocks, hit_length = find(16, drop_eagle_block=False)
    assert hit_length == 0


def test_sliding_window_fine_grained_reachable_mask():
    """Sparse retention keeps the full blocks a fine-grained hit at each
    reachable boundary needs; dense retention keeps everything."""
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    mask = SlidingWindowManager.reachable_block_mask(
        start_block=0,
        end_block=4,
        alignment_tokens=2,
        kv_cache_spec=spec,
        use_eagle=True,
        retention_interval=0,
        reachable_boundaries=[27],  # aligned to 26, inside block 3
    )
    # Window (+1 EAGLE unit) before 26 spans [17, 26): blocks 2 and 3 (a
    # fully cached block 3 covers the boundary inside it), blocks 0-1 not.
    assert mask == [False, False, True, True]
    assert (
        SlidingWindowManager.reachable_block_mask(
            start_block=0,
            end_block=4,
            alignment_tokens=2,
            kv_cache_spec=spec,
            use_eagle=True,
            retention_interval=None,
            reachable_boundaries=[27],
        )
        is None
    )


def test_attention_only_hybrid_keeps_block_aligned_hits():
    """Without a recurrent group, hashing finer than an attention block does
    not turn partial hits on: attention-only hybrids keep the block-aligned
    behaviour they had before sliding-window groups learned fine-grained
    lookups."""
    hash_block_size = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=2 * hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["swa"],
                SlidingWindowSpec(
                    block_size=4 * hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=4 * hash_block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    coordinator = manager.coordinator
    assert not coordinator.enable_partial_hash_hits
    assert all(
        m.hit_alignment_tokens == coordinator.scheduler_block_size
        for m in coordinator.single_type_managers
    )


def test_sliding_window_eagle_margin_is_one_hash_unit_under_partial_hits():
    """With partial hits on, the coordinator hands the EAGLE sliding-window
    group a one-hash-unit lookahead margin (not a whole draft block) and the
    reconciled hit lands back on the candidate length."""
    hash_block_size = 2
    manager = make_kv_cache_manager(
        kv_cache_config=_make_hybrid_swa_eagle_config(
            hash_block_size=hash_block_size,
            attn_block_size=16,
            mamba_block_size=hash_block_size,
            sliding_window=16,
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )
    coordinator = manager.coordinator
    assert coordinator.enable_partial_hash_hits
    base = list(range(1000, 1045))
    _prefill_and_free(manager, make_request("0", base, hash_block_size, sha256), 2)

    seen_max_lengths: list[int] = []
    swa_manager = coordinator.single_type_managers[2]
    original = type(swa_manager).find_longest_cache_hit

    def spy(cls, *args, **kwargs):
        seen_max_lengths.append(kwargs["max_length"])
        return original(*args, **kwargs)

    type(swa_manager).find_longest_cache_hit = classmethod(spy)
    try:
        req = make_request("1", base, hash_block_size, sha256)
        _, hit_length, _ = coordinator.find_longest_cache_hit(
            req.block_hashes, len(base) - 1
        )
    finally:
        type(swa_manager).find_longest_cache_hit = original
    # Candidate 44 (the tail): the draft is asked for 44 + one hash unit
    # (capped by max_length 44), rewinds to 42; the re-query asks for exactly
    # the rewound length. Nothing is queried a whole draft block ahead.
    assert hit_length == 42
    assert seen_max_lengths and max(seen_max_lengths) <= 44
    assert 42 in seen_max_lengths


def _cached_mamba_boundaries(manager, group_id: int = 1) -> list[int]:
    return sorted(
        {
            block.block_hash_num_tokens
            for block in manager.block_pool.blocks
            if block.block_hash is not None
            and block.block_hash_num_tokens
            and get_group_id(block.block_hash) == group_id
        }
    )


@pytest.mark.parametrize("prefill_chunk", [2, 16])
def test_fine_grained_retention_keeps_scheduler_aligned_fallback(prefill_chunk: int):
    """Sparse retention (interval 0) under fine-grained hits must keep the
    scheduler-aligned recurrent state and its EAGLE predecessor next to the
    fine replay boundary. With prefill in scheduler-sized steps the recurrent
    state is only materialized at step ends, so the fine boundary alone would
    retain nothing and a repeat request would resume from zero."""
    hash_block_size = 2
    attn_block_size = 16  # also the scheduler block (LCM with the mamba block)
    manager = make_kv_cache_manager(
        kv_cache_config=_make_hybrid_swa_eagle_config(
            hash_block_size=hash_block_size,
            attn_block_size=attn_block_size,
            mamba_block_size=hash_block_size,
            sliding_window=attn_block_size,
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
        retention_interval=0,
    )
    assert manager.coordinator.enable_partial_hash_hits
    base = list(range(1000, 1045))  # replay boundary 44; scheduler-aligned 32
    _prefill_and_free(
        manager, make_request("0", base, hash_block_size, sha256), prefill_chunk
    )

    retained = _cached_mamba_boundaries(manager)
    # Scheduler-aligned fallback (32) and its predecessor (16) are always kept.
    assert {16, 32} <= set(retained)
    if prefill_chunk == hash_block_size:
        # Fine states exist: the replay boundary (44) and its EAGLE predecessor (42).
        assert {42, 44} <= set(retained)
        expected_repeat_hit = 42
    else:
        # Only step-end states exist; the fine boundary retains nothing extra.
        assert retained == [16, 32]
        expected_repeat_hit = 32
    repeat = _prefill_and_free(
        manager, make_request("1", base, hash_block_size, sha256), prefill_chunk
    )
    assert repeat == expected_repeat_hit


@pytest.mark.parametrize("num_prompt_tokens", [13, 45, 48])
def test_hybrid_decoupled_blocks_keep_fine_grained_reuse(num_prompt_tokens: int):
    """Large attention/draft blocks (page parity with the recurrent state)
    with a fine recurrent block and hash unit reuse prefixes exactly like an
    all-fine-block layout: a repeat or a shared-prefix request resumes from
    the last recurrent checkpoint before the prompt tail, minus the draft's
    EAGLE unit, and once the shared prefix is registered the next request
    resumes from the tail itself."""
    hash_block_size = 2

    def hits(attn_block_size: int, mamba_block_size: int) -> tuple[int, int, int]:
        manager = make_kv_cache_manager(
            kv_cache_config=_make_hybrid_swa_eagle_config(
                hash_block_size=hash_block_size,
                attn_block_size=attn_block_size,
                mamba_block_size=mamba_block_size,
                sliding_window=attn_block_size,
            ),
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=hash_block_size,
            use_eagle=True,
        )
        base = list(range(1000, 1000 + num_prompt_tokens))
        _prefill_and_free(
            manager, make_request("0", base, hash_block_size, sha256), mamba_block_size
        )
        identical = _prefill_and_free(
            manager, make_request("1", base, hash_block_size, sha256), mamba_block_size
        )
        suffix = _prefill_and_free(
            manager,
            make_request("2", base + [7, 7], hash_block_size, sha256),
            mamba_block_size,
        )
        suffix_again = _prefill_and_free(
            manager,
            make_request("3", base + [7, 7], hash_block_size, sha256),
            mamba_block_size,
        )
        return identical, suffix, suffix_again

    fine = hits(attn_block_size=2, mamba_block_size=2)
    decoupled = hits(attn_block_size=16, mamba_block_size=2)
    # A repeat may hit up to num_prompt - 1; a longer request up to the whole
    # shared prompt. Either way the draft's EAGLE unit comes off the tail.
    repeat_tail = (num_prompt_tokens - 1) // hash_block_size * hash_block_size
    shared_tail = num_prompt_tokens // hash_block_size * hash_block_size
    assert fine[:2] == (
        repeat_tail - hash_block_size,
        shared_tail - hash_block_size,
    )
    assert decoupled == fine


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_hybrid_partial_hash_hit_uses_cow_under_dcp(dcp_world_size: int):
    hash_block_size = 2
    physical_block_size = 4
    manager = make_full_mamba_manager(
        dcp_world_size=dcp_world_size,
        hash_block_size=hash_block_size,
        full_block_size=physical_block_size,
        mamba_block_size=physical_block_size,
    )
    assert manager.coordinator.enable_partial_hash_hits

    req0 = make_request("dcp-owner", [0, 0, 1, 1, 2, 2], 2, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_hash = req0.block_hashes[2]
    partial_full_block = manager.block_pool.get_cached_block(partial_hash, [0])
    partial_mamba_block = manager.block_pool.get_cached_block(partial_hash, [1])
    assert partial_full_block is not None
    assert partial_mamba_block is not None

    req1 = make_request("dcp-replay", [0, 0, 1, 1, 2, 2, 3, 3], 2, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 6
    full_block_size = physical_block_size * dcp_world_size
    assert [len(group) for group in computed_blocks.blocks] == [
        (6 + full_block_size - 1) // full_block_size,
        2,
    ]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_id = new_blocks.get_block_ids()[0][0]
    mamba_new_block_id = new_blocks.get_block_ids()[1][0]
    copies, retained = manager.take_kv_cache_block_copies()
    assert KVCacheBlockCopy(partial_full_block[0].block_id, full_new_block_id) in copies
    assert (
        KVCacheBlockCopy(partial_mamba_block[0].block_id, mamba_new_block_id) in copies
    )
    manager.block_pool.free_blocks(retained)


@pytest.mark.parametrize("dcp_world_size", [2, 4])
def test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot(
    dcp_world_size: int,
):
    block_size = 4
    manager = make_full_mamba_manager(
        dcp_world_size=dcp_world_size,
        hash_block_size=block_size,
        full_block_size=block_size,
        mamba_block_size=block_size,
    )
    assert manager.coordinator.enable_partial_hash_hits
    assert manager.coordinator._cache_hit_alignment_tokens == block_size
    assert manager.coordinator.single_type_managers[0].block_size == (
        block_size * dcp_world_size
    )
    assert manager.coordinator.single_type_managers[1].block_size == block_size

    prefix = list(range(12))
    req0 = make_request("snapshot-owner", prefix, block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 12, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    req1 = make_request(
        "snapshot-replay", prefix + list(range(12, 16)), block_size, sha256
    )

    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 12
    assert [len(group) for group in computed_blocks.blocks] == [
        (12 + block_size * dcp_world_size - 1) // (block_size * dcp_world_size),
        3,
    ]
    partial_full_block = computed_blocks.blocks[0][-1]
    new_blocks = manager.allocate_slots(req1, 4, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_id = new_blocks.get_block_ids()[0][0]
    copies, retained = manager.take_kv_cache_block_copies()
    assert KVCacheBlockCopy(partial_full_block.block_id, full_new_block_id) in copies
    assert all(
        copy.src_block_id != computed_blocks.blocks[1][-1].block_id for copy in copies
    )
    manager.block_pool.free_blocks(retained)


def test_dcp_joint_hit_is_bounded_by_replicated_mamba_snapshots():
    block_size = 4
    manager = make_full_mamba_manager(
        dcp_world_size=2,
        hash_block_size=block_size,
        full_block_size=block_size,
        mamba_block_size=block_size,
    )
    prefix = list(range(12))
    req0 = make_request("joint-owner", prefix, block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 8, num_computed, computed_blocks) is not None
    manager.new_step_starts()

    replay_tokens = prefix + list(range(12, 16))
    req1 = make_request("joint-before", replay_tokens, block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 8
    assert [len(group) for group in computed_blocks.blocks] == [1, 2]

    req0.num_computed_tokens = 8
    assert manager.allocate_slots(req0, 4) is not None
    manager.new_step_starts()

    req2 = make_request("joint-after", replay_tokens, block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req2)
    assert num_computed == 12
    assert [len(group) for group in computed_blocks.blocks] == [2, 3]


def test_dcp_partial_hit_with_eagle_rewinds_one_hash_unit():
    hash_block_size = 2
    manager = make_full_mamba_manager(
        dcp_world_size=2,
        hash_block_size=hash_block_size,
        full_block_size=4,
        mamba_block_size=4,
        use_eagle=True,
    )

    req0 = make_request("eagle-owner", [7] * 6, hash_block_size, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req0)
    assert manager.allocate_slots(req0, 4, num_computed, computed_blocks) is not None
    req0.num_computed_tokens = 4
    manager.new_step_starts()
    assert manager.allocate_slots(req0, 2) is not None
    req0.num_computed_tokens = 6
    manager.new_step_starts()

    req1 = make_request("eagle-replay", [7] * 6 + [9] * 2, 2, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 4
    assert [len(group) for group in computed_blocks.blocks] == [1, 1]
    assert manager.allocate_slots(req1, 4, num_computed, computed_blocks) is not None


class TestSemanticReplayCheckpoints(unittest.TestCase):
    def manager(self, block, draft):
        config = _make_hybrid_swa_eagle_config(
            block, 2048, block, 2048, num_blocks=4096
        )
        config.kv_cache_groups[1].kv_cache_spec = replace(
            config.kv_cache_groups[1].kv_cache_spec, num_prefill_checkpoint_blocks=1
        )
        if draft == "mtp":
            config.kv_cache_groups[2].kv_cache_spec = FullAttentionSpec(
                block_size=2048, num_kv_heads=1, head_size=1, dtype=torch.float32
            )
        return make_kv_cache_manager(
            config,
            max_model_len=524288,
            enable_caching=True,
            hash_block_size=block,
            use_eagle=True,
            retention_interval=0,
        )

    def test_large_chunk_materializes_a_reusable_replay_state(self):
        manager = self.manager(256, "mtp")
        scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=2048,
                prefix_cache_retention_interval=0,
                mamba_cache_mode="align",
            ),
            kv_cache_manager=manager,
            block_size=2048,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            drop_last_prefix_cache_block=True,
            mamba_has_prefill_checkpoint_blocks=True,
            use_eagle=True,
            max_num_scheduled_tokens=8192,
            hash_block_size=256,
            mamba_partial_cache_hit=True,
        )
        request = make_request("cold", list(range(65536)), 256, sha256)
        steps = 0
        while request.num_computed_tokens < request.num_prompt_tokens:
            budget = min(8192, request.num_prompt_tokens - request.num_computed_tokens)
            step = Scheduler._mamba_block_aligned_split(scheduler, request, budget)
            self.assertGreater(step, 0)
            self.assertLessEqual(step, budget)
            self.assertIsNotNone(manager.allocate_slots(request, step))
            request.num_computed_tokens += step
            manager.new_step_starts()
            steps += 1
            self.assertLessEqual(steps, 16)
        self.assertIsNotNone(manager.allocate_slots(request, 1))
        request.num_computed_tokens += 1
        manager.new_step_starts()
        manager.free(request)
        manager.new_step_starts()
        _, hit, _ = manager.get_computed_blocks(
            make_request("repeat", list(range(65536)), 256, sha256)
        )
        self.assertGreaterEqual(hit, 65024)

    def test_swa_preserves_the_materialized_recurrent_fallback(self):
        manager = self.manager(256, "dflash")
        _prefill_and_free(
            manager, make_request("cold", list(range(65536)), 256, sha256), 4096
        )
        retained = {
            b.block_hash_num_tokens
            for b in manager.block_pool.blocks
            if b.block_hash and get_group_id(b.block_hash) == 1
        }
        self.assertIn(61440, retained)
        _, hit, _ = manager.get_computed_blocks(
            make_request("repeat", list(range(65536)), 256, sha256)
        )
        self.assertGreaterEqual(hit, 61440)

    def test_fine_checkpoint_keeps_the_rewind_stop(self):
        manager = self.manager(256, "mtp")
        scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=2048, prefix_cache_retention_interval=0
            ),
            kv_cache_manager=manager,
            block_size=2048,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            drop_last_prefix_cache_block=True,
            mamba_has_prefill_checkpoint_blocks=True,
            use_eagle=True,
            max_num_scheduled_tokens=4096,
            hash_block_size=256,
            mamba_partial_cache_hit=True,
        )
        request = make_request("tail", list(range(8449)), 256, sha256)
        request.num_computed_tokens = 6144
        # The worker captures 8448 internally. Lookup also needs 8192.
        step = Scheduler._mamba_block_aligned_split(scheduler, request, 2305)
        self.assertEqual(step, 2048)


def test_semantic_checkpoints_support_unitary_recurrent_coordinator():
    config = _make_hybrid_swa_eagle_config(2048, 2048, 2048, 2048, num_blocks=128)
    config.kv_cache_groups = [config.kv_cache_groups[1]]
    manager = make_kv_cache_manager(
        config,
        max_model_len=524288,
        enable_caching=True,
        hash_block_size=2048,
        retention_interval=0,
    )
    assert not hasattr(manager.coordinator, "_cache_hit_alignment_tokens")
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048, prefix_cache_retention_interval=0
        ),
        kv_cache_manager=manager,
        block_size=2048,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        drop_last_prefix_cache_block=False,
        mamba_has_prefill_checkpoint_blocks=True,
        use_eagle=False,
        max_num_scheduled_tokens=8192,
        hash_block_size=2048,
        mamba_partial_cache_hit=False,
    )
    request = make_request("unitary-tail", list(range(65536)), 2048, sha256)
    request.num_computed_tokens = 61440
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 4096) == 2048
