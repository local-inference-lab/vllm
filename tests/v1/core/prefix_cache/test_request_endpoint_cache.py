# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-endpoint prefix-cache entries on a DCP9 hybrid layout.

A finished request publishes the pages holding its last computed token, the
draft window before it and one pool block for its recurrent state. A later
prompt that extends the same token sequence resumes at that token (not at
the last hash boundary) through the ordinary partial-hit copy-on-write path.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm import envs
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

UNIT = 2
MAMBA_BLOCK = 6
NUM_SPEC = 3
WINDOW = 6
ATTN_GROUP, MAMBA_GROUP, DRAFT_GROUP = 0, 1, 2


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


@pytest.fixture(autouse=True)
def _endpoint_cache_env(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE", True)
    monkeypatch.setattr(envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES", 8)
    monkeypatch.setattr(envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE", "")


def _dcp9_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target_attention"],
                FullAttentionSpec(
                    block_size=UNIT,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["target_recurrent"],
                MambaSpec(
                    block_size=MAMBA_BLOCK,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
            KVCacheGroupSpec(
                ["draft_attention"],
                SlidingWindowSpec(
                    block_size=UNIT,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=WINDOW,
                    dcp_replicated=True,
                ),
            ),
        ],
    )


def _two_recurrent_groups_config() -> KVCacheConfig:
    """The DCP9 layout with a second recurrent group (group 3)."""
    config = _dcp9_config()
    recurrent = config.kv_cache_groups[MAMBA_GROUP]
    return KVCacheConfig(
        num_blocks=config.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            *config.kv_cache_groups,
            KVCacheGroupSpec(["target_recurrent_2"], recurrent.kv_cache_spec),
        ],
    )


def _manager(kv_cache_config: KVCacheConfig | None = None):
    return make_kv_cache_manager(
        kv_cache_config=kv_cache_config or _dcp9_config(),
        max_model_len=128,
        enable_caching=True,
        scheduler_block_size=18,
        hash_block_size=UNIT,
        dcp_world_size=9,
        use_eagle=False,
    )


def _prefill(manager, request):
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=UNIT),
        mamba_block_size=MAMBA_BLOCK,
        max_num_scheduled_tokens=MAMBA_BLOCK,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        use_eagle_for_target_cache=False,
        mamba_eagle_block_sizes=(),
        hash_block_size=UNIT,
        mamba_partial_cache_hit=True,
    )
    while request.num_computed_tokens < request.num_tokens:
        remaining = request.num_tokens - request.num_computed_tokens
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, request, min(MAMBA_BLOCK, remaining)
        )
        assert chunk > 0
        assert manager.allocate_slots(request, chunk) is not None
        request.num_computed_tokens += chunk
        _end_step(manager)


def _decode_step(manager, request, num_drafts, num_accepted, next_token):
    """One speculative decode step: schedule ``num_drafts + 1`` rows from the
    committed position, then commit ``num_accepted + 1`` tokens."""
    request.spec_token_ids = [900 + i for i in range(num_drafts)]
    assert manager.allocate_slots(request, num_drafts + 1) is not None
    request.spec_token_ids = []
    request.num_computed_tokens += num_accepted + 1
    request.append_output_token_ids([next_token + i for i in range(num_accepted + 1)])
    _end_step(manager)


def _end_step(manager):
    """What the scheduler does per step: drain the copy-on-write copies and,
    once the step has run, release the blocks retained for them."""
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    manager.new_step_starts()


def _run_producer(manager, prompt_len=23, steps=((3, 1), (3, 1), (3, 1))):
    producer = make_request("producer", list(range(prompt_len)), UNIT, sha256)
    _prefill(manager, producer)
    # The first sampled token of the prompt's last chunk.
    producer.append_output_token_ids(500)
    token = 600
    for num_drafts, num_accepted in steps:
        _decode_step(manager, producer, num_drafts, num_accepted, token)
        token += 10
    assert producer.num_computed_tokens == producer.num_tokens - 1
    return producer, steps[-1]


def _mamba_blocks(manager, request):
    return list(
        manager.coordinator.single_type_managers[MAMBA_GROUP].req_to_blocks[
            request.request_id
        ]
    )


@pytest.mark.parametrize("in_flight", [False, True])
def test_next_turn_resumes_at_the_last_computed_token(in_flight):
    manager = _manager()
    producer, final_step = _run_producer(manager)
    num_tokens = producer.num_tokens - 1
    assert num_tokens == 29 and num_tokens % UNIT == 1
    producer_mamba = _mamba_blocks(manager, producer)
    producer_page = manager.coordinator.single_type_managers[ATTN_GROUP].req_to_blocks[
        producer.request_id
    ][(num_tokens - 1) // 18]

    assert manager.cache_endpoint(producer, final_step, in_flight=in_flight)
    assert manager.block_pool.num_endpoint_entries == 1
    manager.free(producer)
    manager.new_step_starts()

    copies, retained = manager.take_mamba_endpoint_copies()
    assert len(copies) == 1
    copy = copies[0]
    assert copy.req_id == "producer"
    assert copy.from_shadow == in_flight
    dst = next(block for block in retained if block.block_id == copy.dst_block_ids[0])
    assert dst.ref_cnt == 1 and dst.block_hash is not None
    num_drafts, num_accepted = final_step
    # The bias selects the conv shift and, for a shadow source, the temporal
    # shadow slot: the accepted count of the final step in both cases.
    assert copy.token_bias == num_accepted
    if in_flight:
        assert copy.conv_src_block_ids == () and copy.temporal_src_block_ids == ()
    else:
        # Final step: 4 rows from position 27 (column 5 after the crossing at
        # 30), 1 draft accepted -> temporal slot 1 of column 5, window bias 1.
        num_before = num_tokens - num_accepted - 1
        base = (num_before + num_drafts) // MAMBA_BLOCK
        assert copy.conv_src_block_ids == (producer_mamba[base].block_id,)
        assert copy.temporal_src_block_ids == (
            producer_mamba[base + num_accepted].block_id,
        )
        assert {block.block_id for block in retained} >= set(
            copy.conv_src_block_ids + copy.temporal_src_block_ids
        )
    # The step that runs the copy has completed: release the retentions.
    manager.block_pool.free_blocks(retained)
    assert dst.ref_cnt == 0 and dst.block_hash is not None

    consumer_tokens = list(producer.all_token_ids) + [700, 701, 702, 703, 704]
    consumer = make_request("consumer", consumer_tokens, UNIT, sha256)
    blocks, hit, boundary = manager.get_computed_blocks(consumer)
    assert hit == num_tokens and boundary == 0
    attn, mamba, draft = blocks.blocks
    assert len(attn) == (num_tokens - 1) // 18 + 1 and attn[-1] is producer_page
    assert [block.is_null for block in mamba] == [True] * ((num_tokens - 1) // 6) + [
        False
    ]
    assert mamba[-1] is dst
    first_window_block = max(0, num_tokens - WINDOW + 1) // UNIT
    assert len(draft) == (num_tokens - 1) // UNIT + 1
    assert all(block.is_null for block in draft[:first_window_block])
    assert not any(block.is_null for block in draft[first_window_block:])

    assert (
        manager.allocate_slots(consumer, consumer.num_tokens - hit, hit, blocks)
        is not None
    )
    cow_copies, _ = manager.take_kv_cache_block_copies()
    # Every group ends mid-block at 29 tokens: page, state and draft block
    # are all redirected to private copies before the consumer writes, and
    # the entry's recurrent state is one of the copied sources.
    assert len(cow_copies) >= 3
    sources = {copy.src_block_id for copy in cow_copies}
    assert dst.block_id in sources
    destinations = {copy.dst_block_id for copy in cow_copies}
    assert dst.block_id not in destinations
    # The entry itself is untouched by the consumer: the durable blocks keep
    # their endpoint keys.
    assert manager.block_pool.num_endpoint_entries == 1


@pytest.mark.parametrize("in_flight", [False, True])
def test_a_stop_inside_the_accepted_tokens_moves_the_endpoint_back(in_flight):
    """The final step accepts 2 drafts but the stop token is the first new
    token: the request keeps 1 new token, and the endpoint is the state
    after row 0 of that step (bias 0), one row before the committed one."""
    manager = _manager()
    producer = make_request("producer", list(range(23)), UNIT, sha256)
    _prefill(manager, producer)
    producer.append_output_token_ids(500)
    _decode_step(manager, producer, 3, 1, 600)  # 23 -> 25 computed
    # Final step: 4 rows from 25, 2 drafts accepted (committed 28), but the
    # output is trimmed to the first new token (num_tokens 27).
    producer.spec_token_ids = [900, 901, 902]
    assert manager.allocate_slots(producer, 4) is not None
    producer.spec_token_ids = []
    producer.num_computed_tokens += 3
    producer.append_output_token_ids(700)
    _end_step(manager)
    assert producer.num_computed_tokens == 28 and producer.num_tokens == 27
    producer_mamba = _mamba_blocks(manager, producer)

    assert manager.cache_endpoint(producer, (3, 2), in_flight=in_flight)
    manager.free(producer)
    manager.new_step_starts()
    copies, retained = manager.take_mamba_endpoint_copies()
    copy = copies[0]
    assert copy.token_bias == 0
    if not in_flight:
        base = (25 + 3) // MAMBA_BLOCK
        assert copy.conv_src_block_ids == (producer_mamba[base].block_id,)
        assert copy.temporal_src_block_ids == (producer_mamba[base].block_id,)
    manager.block_pool.free_blocks(retained)

    consumer = make_request(
        "consumer", list(producer.all_token_ids) + [800, 801], UNIT, sha256
    )
    _, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == 26


def test_a_stop_after_a_boundary_normalization_publishes_nothing():
    """The final step's accepted rows end exactly on a recurrent block
    boundary in the running column, so the post-step kernel shifted the
    window in place; a stop before that boundary has no restorable state."""
    manager = _manager()
    producer = make_request("producer", list(range(23)), UNIT, sha256)
    _prefill(manager, producer)
    producer.append_output_token_ids(500)
    _decode_step(manager, producer, 3, 1, 600)  # committed 25
    # Final step: rows 25..28, all 3 drafts accepted -> committed 29; the
    # rows end in column 4 which also holds the boundary at 30? No: use a
    # step whose last accepted row completes the block: from 26.
    _decode_step(manager, producer, 0, 0, 700)  # committed 26
    producer.spec_token_ids = [900, 901, 902]
    assert manager.allocate_slots(producer, 4) is not None
    producer.spec_token_ids = []
    producer.num_computed_tokens += 4  # rows 26..29 all accepted -> 30
    producer.append_output_token_ids(800)  # trimmed to one new token (27)
    _end_step(manager)
    assert producer.num_computed_tokens == 30 and producer.num_tokens == 28
    assert not manager.cache_endpoint(producer, (3, 3), in_flight=False)
    assert not manager.cache_endpoint(producer, (3, 3), in_flight=True)
    # Without the trim the normalized state itself is publishable.
    producer.append_output_token_ids([801, 802, 803])
    assert producer.num_tokens == 31
    assert manager.cache_endpoint(producer, (3, 3), in_flight=True)
    copy = manager.take_mamba_endpoint_copies()[0][0]
    assert copy.token_bias == 0


def test_a_different_tail_does_not_hit_the_endpoint():
    manager = _manager()
    producer, final_step = _run_producer(manager)
    num_tokens = producer.num_tokens - 1
    assert manager.cache_endpoint(producer, final_step, in_flight=True)
    manager.free(producer)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)

    tokens = list(producer.all_token_ids)
    tokens[num_tokens - 1] = 12345
    consumer = make_request("consumer", tokens + [700, 701], UNIT, sha256)
    _, hit, _ = manager.get_computed_blocks(consumer)
    # Falls back to the ordinary aligned hit below the endpoint.
    assert hit < num_tokens and hit % UNIT == 0


def test_replaying_exactly_the_producer_sequence_hits_the_endpoint():
    manager = _manager()
    producer, final_step = _run_producer(manager)
    num_tokens = producer.num_tokens - 1
    assert manager.cache_endpoint(producer, final_step, in_flight=True)
    manager.free(producer)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)

    consumer = make_request("consumer", list(producer.all_token_ids), UNIT, sha256)
    _, hit, _ = manager.get_computed_blocks(consumer)
    # Only the last token (the one that must be recomputed for logits) is left.
    assert hit == num_tokens == consumer.num_tokens - 1


def test_reusing_an_endpoint_block_drops_the_entry():
    manager = _manager()
    producer, final_step = _run_producer(manager)
    assert manager.cache_endpoint(producer, final_step, in_flight=True)
    manager.free(producer)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)
    assert manager.block_pool.num_endpoint_entries == 1

    # Take every free block: the LRU hands out the endpoint blocks as well.
    pool = manager.block_pool
    taken = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.num_endpoint_entries == 0
    assert all(block.block_hash is None for block in taken)
    pool.free_blocks(taken)

    consumer = make_request(
        "consumer", list(producer.all_token_ids) + [700, 701], UNIT, sha256
    )
    _, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == 0


def test_entry_cap_drops_the_oldest_entry(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES", 1)
    manager = _manager()
    first, final_step = _run_producer(manager)
    assert manager.cache_endpoint(first, final_step, in_flight=True)
    manager.free(first)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)

    second = make_request("second", list(range(100, 123)), UNIT, sha256)
    _prefill(manager, second)
    second.append_output_token_ids(500)
    _decode_step(manager, second, 3, 1, 600)
    assert manager.cache_endpoint(second, (3, 1), in_flight=True)
    assert manager.block_pool.num_endpoint_entries == 1
    manager.free(second)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)

    stale = make_request("stale", list(first.all_token_ids) + [700], UNIT, sha256)
    _, hit, _ = manager.get_computed_blocks(stale)
    assert hit < first.num_tokens - 1
    fresh = make_request("fresh", list(second.all_token_ids) + [700], UNIT, sha256)
    _, hit, _ = manager.get_computed_blocks(fresh)
    assert hit == second.num_tokens - 1


def test_reset_prefix_cache_forgets_endpoints():
    manager = _manager()
    producer, final_step = _run_producer(manager)
    assert manager.cache_endpoint(producer, final_step, in_flight=True)
    manager.free(producer)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    manager.block_pool.free_blocks(retained)
    assert manager.reset_prefix_cache()
    assert manager.block_pool.num_endpoint_entries == 0
    consumer = make_request(
        "consumer", list(producer.all_token_ids) + [700], UNIT, sha256
    )
    _, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == 0


def test_disabled_feature_registers_nothing(monkeypatch, tmp_path):
    manager = _manager()
    producer, final_step = _run_producer(manager)
    disable_file = tmp_path / "endpoint-cache-disable"
    disable_file.write_text("")
    monkeypatch.setattr(
        envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE", str(disable_file)
    )
    assert not manager.cache_endpoint(producer, final_step, in_flight=True)
    disable_file.unlink()
    monkeypatch.setattr(envs, "VLLM_K3_REQUEST_ENDPOINT_CACHE", False)
    assert not manager.cache_endpoint(producer, final_step, in_flight=True)
    assert manager.block_pool.num_endpoint_entries == 0
    assert manager.take_mamba_endpoint_copies() == ([], [])


def test_an_incomplete_request_publishes_nothing():
    manager = _manager()
    producer = make_request("producer", list(range(23)), UNIT, sha256)
    _prefill(manager, producer)
    # Aborted before its first token: nothing beyond the prompt is committed.
    assert not manager.cache_endpoint(producer, (0, 0), in_flight=False)
    assert manager.block_pool.num_endpoint_entries == 0


@pytest.mark.parametrize("in_flight", [False, True])
def test_every_recurrent_group_gets_its_own_durable_block(in_flight):
    """Block ids name the same physical page in every group, so an entry
    holds one distinct block per recurrent group and the copy record lists
    them in group order."""
    manager = _manager(_two_recurrent_groups_config())
    producer, final_step = _run_producer(manager)
    num_tokens = producer.num_tokens - 1
    free_before = manager.block_pool.get_num_free_blocks()
    assert manager.cache_endpoint(producer, final_step, in_flight=in_flight)
    assert manager.block_pool.get_num_free_blocks() == free_before - 2
    manager.free(producer)
    manager.new_step_starts()
    copies, retained = manager.take_mamba_endpoint_copies()
    (copy,) = copies
    assert len(copy.dst_block_ids) == 2
    assert len(set(copy.dst_block_ids)) == 2
    retained_ids = {block.block_id for block in retained}
    assert set(copy.dst_block_ids) <= retained_ids
    if not in_flight:
        assert len(copy.conv_src_block_ids) == 2
        assert len(copy.temporal_src_block_ids) == 2
    entry = manager.block_pool.find_endpoints(
        producer.block_hashes[num_tokens // UNIT - 1]
    )[0]
    recurrent_groups = [MAMBA_GROUP, 3]
    assert [entry.group_blocks[gid][1][0].block_id for gid in recurrent_groups] == list(
        copy.dst_block_ids
    )
    manager.block_pool.free_blocks(retained)

    consumer_tokens = list(producer.all_token_ids) + [700, 701, 702]
    consumer = make_request("consumer", consumer_tokens, UNIT, sha256)
    blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == num_tokens
    for gid, dst_id in zip(recurrent_groups, copy.dst_block_ids):
        assert blocks.blocks[gid][-1].block_id == dst_id


def test_endpoint_needs_a_spare_block_beyond_one_per_group():
    manager = _manager(_two_recurrent_groups_config())
    producer, final_step = _run_producer(manager)
    pool = manager.block_pool
    spare = pool.get_new_blocks(pool.get_num_free_blocks() - 2)
    assert pool.get_num_free_blocks() == 2
    assert not manager.cache_endpoint(producer, final_step, in_flight=True)
    assert pool.num_endpoint_entries == 0
    pool.free_blocks(spare)
    assert manager.cache_endpoint(producer, final_step, in_flight=True)
    assert pool.num_endpoint_entries == 1
    manager.free(producer)
    manager.new_step_starts()
    _, retained = manager.take_mamba_endpoint_copies()
    pool.free_blocks(retained)
