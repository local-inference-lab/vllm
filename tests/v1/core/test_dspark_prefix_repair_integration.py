# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the DSpark block-unaligned prefix repair.

A local prefix-cache hit whose token count is not block-aligned restores
tokens that never flowed through the target forward, so DFlash/DSpark draft
KV does not exist for them and drafting fail-closes for the whole batch.
The repair adopts only the block-aligned region and lets the residual tail
prefill normally, so the batch never reaches the fail-closed path.

The four-group tests model the DeepSeek-V4 hybrid geometry (full-attention
256 plus sliding-window 64, 4 and 8 token blocks). ``enable_partial_hash_hits``
is False without a Mamba ``align`` group, so ordinary lookups in this geometry
are 256-aligned and never exercise the repair; the tests therefore present the
257-token lookup directly at the cache-manager boundary and drive the real
admission and ``truncate_computed_blocks`` primitives. The truncated
per-group block lists must equal exactly what a normal 256-aligned admission
would adopt (1, 4, 64 and 32 blocks), which is why 256 is a valid clamp
endpoint here: it is divisible by every group block size.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_dspark_scheduler(async_scheduling: bool):
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=256,
        async_scheduling=async_scheduling,
    )
    scheduler.vllm_config.speculative_config = SimpleNamespace(method="dspark")
    return scheduler


def _stub_unaligned_hit(scheduler, monkeypatch, hit_tokens: int):
    """Let the cache lookup report an unaligned hit of hit_tokens.

    The blocks are real blocks from the block pool (2 blocks cover
    hit_tokens <= 512); truncate_computed_blocks and allocate_slots run for
    real, so the whole admission path is exercised.
    """
    pool = scheduler.kv_cache_manager.block_pool
    blocks = pool.get_new_blocks(2)

    def fake_lookup(request):
        wrapped = scheduler.kv_cache_manager.create_kv_cache_blocks(tuple([blocks]))
        return wrapped, hit_tokens, 0

    monkeypatch.setattr(scheduler.kv_cache_manager, "get_computed_blocks", fake_lookup)
    return blocks


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_unaligned_hit_is_clamped_during_admission(monkeypatch, async_scheduling: bool):
    scheduler = _make_dspark_scheduler(async_scheduling)
    request = create_requests(num_requests=1, num_tokens=300, block_size=256)[0]
    scheduler.add_request(request)
    _stub_unaligned_hit(scheduler, monkeypatch, 257)

    output = scheduler.schedule()

    new_req = output.scheduled_new_reqs[0]
    # Aligned restored prefix; the 1-token tail prefills via num_scheduled_tokens.
    assert new_req.num_computed_tokens == 256
    assert output.num_scheduled_tokens[request.request_id] == 300 - 256
    assert scheduler.num_dspark_prefix_repairs == 1
    assert scheduler.num_dspark_prefix_repair_tokens == 1

    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.dspark_prefix_repairs == 1
    assert stats.dspark_prefix_repair_tokens == 1
    assert stats.dspark_prefix_suppressed_batches == 0
    assert stats.dspark_prefix_suppressed_rows == 0


def test_mixed_batch_repairs_only_the_unaligned_request(monkeypatch):
    scheduler = _make_dspark_scheduler(async_scheduling=True)
    unaligned = create_requests(num_requests=1, num_tokens=300, block_size=256)[0]
    aligned = create_requests(
        num_requests=1, num_tokens=300, block_size=256, req_ids=["aligned"]
    )[0]
    scheduler.add_request(unaligned)
    scheduler.add_request(aligned)

    pool = scheduler.kv_cache_manager.block_pool
    blocks = pool.get_new_blocks(2)

    def fake_lookup(request):
        wrapped = scheduler.kv_cache_manager.create_kv_cache_blocks(tuple([blocks]))
        hit = 257 if request.request_id != "aligned" else 256
        return wrapped, hit, 0

    monkeypatch.setattr(scheduler.kv_cache_manager, "get_computed_blocks", fake_lookup)

    output = scheduler.schedule()

    by_id = {r.req_id: r for r in output.scheduled_new_reqs}
    assert by_id["aligned"].num_computed_tokens == 256
    assert by_id["0"].num_computed_tokens == 256
    assert scheduler.num_dspark_prefix_repairs == 1
    assert scheduler.num_dspark_prefix_repair_tokens == 1


def test_aligned_hit_is_untouched(monkeypatch):
    scheduler = _make_dspark_scheduler(async_scheduling=False)
    request = create_requests(num_requests=1, num_tokens=300, block_size=256)[0]
    scheduler.add_request(request)
    _stub_unaligned_hit(scheduler, monkeypatch, 256)

    output = scheduler.schedule()

    assert output.scheduled_new_reqs[0].num_computed_tokens == 256
    assert scheduler.num_dspark_prefix_repairs == 0
    assert scheduler.num_dspark_prefix_repair_tokens == 0


def test_worker_suppression_counters_flow_into_scheduler_stats():
    scheduler = _make_dspark_scheduler(async_scheduling=False)
    request = create_requests(num_requests=1, num_tokens=300, block_size=256)[0]
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        prompt_logprobs_dict={},
        dspark_suppressed_batches=1,
        dspark_suppressed_rows=1,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.dspark_prefix_suppressed_batches == 1
    assert stats.dspark_prefix_suppressed_rows == 1


# DeepSeek-V4-like hybrid geometry: full attention on 256-token blocks plus
# three sliding-window groups on 64, 4 and 8 token blocks. The scheduler
# block size is the LCM (256), which every group block size divides. The
# smallest config with the same manager and divisibility invariants; the
# real DeepSeek-V4 config only scales the head counts and dtypes.
FOUR_GROUP_BLOCK_SIZES = (256, 64, 4, 8)


def _four_group_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=FOUR_GROUP_BLOCK_SIZES[0],
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
    ]
    for block_size in FOUR_GROUP_BLOCK_SIZES[1:]:
        groups.append(
            KVCacheGroupSpec(
                [f"swa_{block_size}"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _four_group_manager():
    return make_kv_cache_manager(
        _four_group_kv_cache_config(num_blocks=2048),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=4,
        log_stats=True,
    )


def _unaligned_lookup_blocks(manager):
    """Per-group block lists a 257-token restore input presents.

    Each list covers ceil(257 / group_block_size) blocks; the sliding-window
    groups are null-padded like a real lookup, with only the in-window tail
    block backed by a real page. Truncating these lists at 256 tokens must
    produce exactly the lists a normal 256-aligned admission adopts
    (1, 4, 64 and 32 blocks).
    """
    pool = manager.block_pool
    null = pool.null_block
    return [
        pool.get_new_blocks(2),
        [null, null, null] + pool.get_new_blocks(2),
        [null] * 63 + pool.get_new_blocks(2),
        [null] * 31 + pool.get_new_blocks(2),
    ]


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_four_group_unaligned_hit_repairs_to_aligned_admission(
    monkeypatch, async_scheduling: bool
):
    scheduler = _make_dspark_scheduler(async_scheduling)
    manager = _four_group_manager()
    scheduler.kv_cache_manager = manager
    scheduler.kv_cache_config = manager.kv_cache_config

    request = create_requests(num_requests=1, num_tokens=257, block_size=256)[0]
    scheduler.add_request(request)

    hit_blocks = _unaligned_lookup_blocks(manager)

    def fake_lookup(request):
        wrapped = manager.create_kv_cache_blocks(tuple(hit_blocks))
        return wrapped, 257, 0

    monkeypatch.setattr(manager, "get_computed_blocks", fake_lookup)

    output = scheduler.schedule()

    new_req = output.scheduled_new_reqs[0]
    # 1. Only the 256-token block-aligned prefix is adopted.
    assert new_req.num_computed_tokens == 256
    # 2. The 1-token tail goes through normal target prefill.
    assert output.num_scheduled_tokens[request.request_id] == 1
    # 4. Exactly one repair, moving exactly one token.
    assert scheduler.num_dspark_prefix_repairs == 1
    assert scheduler.num_dspark_prefix_repair_tokens == 1
    # 5. The unaligned fail-closed DSpark suppression is never reached.
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.dspark_prefix_repairs == 1
    assert stats.dspark_prefix_repair_tokens == 1
    assert stats.dspark_prefix_suppressed_batches == 0
    assert stats.dspark_prefix_suppressed_rows == 0


def test_four_group_truncation_slices_every_group():
    """The real truncation must cut every group to the aligned endpoint.

    Exercises the actual ``KVCacheManager.truncate_computed_blocks`` route
    the repair uses: pure slicing at token count 256, which is divisible by
    all four group block sizes, and no mutation of the input lookup.
    """
    manager = _four_group_manager()
    hit_blocks = _unaligned_lookup_blocks(manager)
    blocks = manager.create_kv_cache_blocks(tuple(hit_blocks))

    truncated = manager.truncate_computed_blocks(blocks, 256)

    # 3. The four block lists after truncation hold exactly 1, 4, 64, 32
    # blocks.
    assert [len(group) for group in truncated.blocks] == [1, 4, 64, 32]
    # Non-mutating, byte-preserving slice: same block objects, input intact.
    assert truncated.blocks[0][0] is blocks.blocks[0][0]
    assert truncated.blocks[1][-1] is blocks.blocks[1][3]
    assert [len(group) for group in blocks.blocks] == [2, 5, 65, 33]


def test_four_group_aligned_hit_is_untouched(monkeypatch):
    """A 256-aligned hit stays byte- and behavior-equivalent to the base."""
    scheduler = _make_dspark_scheduler(async_scheduling=True)
    manager = _four_group_manager()
    scheduler.kv_cache_manager = manager
    scheduler.kv_cache_config = manager.kv_cache_config

    request = create_requests(num_requests=1, num_tokens=257, block_size=256)[0]
    scheduler.add_request(request)

    pool = manager.block_pool
    null = pool.null_block
    aligned_blocks = [
        pool.get_new_blocks(1),
        [null, null, null] + pool.get_new_blocks(1),
        [null] * 63 + pool.get_new_blocks(1),
        [null] * 31 + pool.get_new_blocks(1),
    ]

    def fake_lookup(request):
        wrapped = manager.create_kv_cache_blocks(tuple(aligned_blocks))
        return wrapped, 256, 0

    monkeypatch.setattr(manager, "get_computed_blocks", fake_lookup)

    output = scheduler.schedule()

    new_req = output.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == 256
    assert output.num_scheduled_tokens[request.request_id] == 1
    assert scheduler.num_dspark_prefix_repairs == 0
    assert scheduler.num_dspark_prefix_repair_tokens == 0
    # The aligned lookup lists are adopted unchanged: the request's first
    # full-attention block is the exact block the lookup returned.
    adopted = manager.get_block_ids(request.request_id)[0]
    assert adopted[0] == aligned_blocks[0][0].block_id
