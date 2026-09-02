# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LP14 regressions for EAGLE hybrid prefix-cache replay boundaries."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


@pytest.mark.parametrize("annotation", [None, "full_only", "both"])
def test_mamba_retains_eagle_reachable_boundary(annotation: str | None) -> None:
    block_size = 32
    num_spec = 3
    manager = make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                    ),
                    is_eagle_group=annotation in ("full_only", "both"),
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=block_size,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                        num_speculative_blocks=num_spec,
                    ),
                    is_eagle_group=annotation == "both",
                ),
            ],
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        retention_interval=0,
        use_eagle=True,
    )
    token_ids = [i for i in range(3) for _ in range(block_size)] + [3] * 31
    producer = make_request("producer", token_ids, block_size, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(producer)
    for chunk_end in (32, 64, 96, 127):
        blocks = manager.allocate_slots(
            producer,
            chunk_end - producer.num_computed_tokens,
            num_computed,
            computed,
            num_lookahead_tokens=num_spec,
        )
        assert blocks is not None, (
            producer.request_id,
            producer.num_computed_tokens,
            manager.usage,
        )
        producer.num_computed_tokens = chunk_end
    manager.free(producer)

    replay = make_request("replay", token_ids, block_size, sha256)
    _, hit_tokens, _ = manager.get_computed_blocks(replay)
    assert hit_tokens == 2 * block_size


def test_mamba_eagle_backoff_uses_alignment_unit() -> None:
    mamba_block = 32
    alignment = 64
    manager = make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=200,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full"],
                    FullAttentionSpec(
                        block_size=alignment,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                    ),
                    is_eagle_group=True,
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=mamba_block,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                    is_eagle_group=True,
                ),
            ],
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=mamba_block,
        retention_interval=0,
        use_eagle=True,
    )
    token_ids = [i for i in range(7) for _ in range(mamba_block)] + [7] * 31
    producer = make_request("producer", token_ids, mamba_block, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(producer)
    for chunk_end in (64, 128, 192, 255):
        blocks = manager.allocate_slots(
            producer,
            chunk_end - producer.num_computed_tokens,
            num_computed,
            computed,
        )
        assert blocks is not None, (
            producer.request_id,
            producer.num_computed_tokens,
            manager.usage,
        )
        producer.num_computed_tokens = chunk_end
    manager.free(producer)

    replay = make_request("replay", token_ids, mamba_block, sha256)
    _, hit_tokens, _ = manager.get_computed_blocks(replay)
    assert hit_tokens == 2 * alignment


def _compute_in_chunks(manager, request, chunk_size: int) -> None:
    while request.num_computed_tokens < request.num_tokens:
        chunk = min(chunk_size, request.num_tokens - request.num_computed_tokens)
        blocks = manager.allocate_slots(request, chunk)
        assert blocks is not None, (
            request.request_id,
            request.num_computed_tokens,
            manager.usage,
        )
        request.num_computed_tokens += chunk
        manager.new_step_starts()
    manager.cache_blocks(request, request.num_computed_tokens)
    manager.free(request)
    manager.new_step_starts()


def test_dcp1_dflash_growing_history_reuses_warm_boundary() -> None:
    """DCP1 must retain the boundary below the EAGLE-dropped candidate."""

    block_size = 2304
    manager = make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=2000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["target"],
                    MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        tokens_per_state=4,
                    ),
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=block_size,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                ),
                KVCacheGroupSpec(
                    ["draft"],
                    SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        sliding_window=2048,
                        dcp_replicated=False,
                        extra_retained_tokens=2048,
                    ),
                ),
            ],
            prefix_cache_retention_interval=0,
        ),
        max_model_len=524288,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
        scheduler_block_size=block_size,
        dcp_world_size=1,
    )
    warm_tokens = list(range(122871))
    producer = make_request("warm", warm_tokens, block_size, sha256)
    _compute_in_chunks(manager, producer, block_size)

    growing_tokens = [*warm_tokens[:122869], *range(200000, 212279)]
    replay = make_request("turn_1", growing_tokens, block_size, sha256)
    _, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    _, hit_tokens, _ = manager.get_computed_blocks(replay)
    assert per_group == (119808, 122112, 119808), per_group
    assert hit_tokens == 119808, hit_tokens


def _make_dcp1_glm_manager(num_blocks: int = 10000):
    block_size = 2304
    return make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["target"],
                    MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        tokens_per_state=4,
                    ),
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=block_size,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                ),
                KVCacheGroupSpec(
                    ["draft"],
                    SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        sliding_window=2048,
                        dcp_replicated=False,
                        extra_retained_tokens=2048,
                    ),
                ),
            ],
            prefix_cache_retention_interval=0,
        ),
        max_model_len=524288,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
        scheduler_block_size=block_size,
        dcp_world_size=1,
    )


def _resume_compute_and_release(manager, request, expected_hit: int) -> None:
    computed, hit_tokens, shared_boundary = manager.get_computed_blocks(request)
    assert hit_tokens == expected_hit
    request.shared_prefix_boundary = shared_boundary
    first = True
    while request.num_computed_tokens < request.num_tokens:
        already = hit_tokens if first else request.num_computed_tokens
        chunk = min(2304, request.num_tokens - already)
        blocks = manager.allocate_slots(
            request,
            chunk,
            num_new_computed_tokens=hit_tokens if first else 0,
            new_computed_blocks=computed if first else None,
        )
        assert blocks is not None, (
            request.request_id,
            request.num_computed_tokens,
            manager.usage,
        )
        request.num_computed_tokens = already + chunk
        manager.new_step_starts()
        first = False
    manager.cache_blocks(request, request.num_computed_tokens)
    manager.free(request)
    manager.new_step_starts()


def test_dcp1_twenty_growing_histories_reuse_each_turn() -> None:
    manager = _make_dcp1_glm_manager()
    histories: list[tuple[list[int], list[list[int]]]] = []
    for agent in range(20):
        start = agent * 1_000_000
        warm = list(range(start, start + 122871))
        turn_1 = [*warm[:122869], *range(start + 200000, start + 212279)]
        turn_2 = [*turn_1, *range(start + 300000, start + 312269)]
        turn_3 = [*turn_2, *range(start + 400000, start + 412269)]
        histories.append((warm, [turn_1, turn_2, turn_3]))

    for agent, (warm, _) in enumerate(histories):
        request = make_request(f"agent_{agent}_warm", warm, 2304, sha256)
        _compute_in_chunks(manager, request, 2304)

    expected_hits = (119808, 131328, 142848)
    total_hits = 0
    for turn, expected in enumerate(expected_hits):
        for agent, (_, turns) in enumerate(histories):
            request = make_request(
                f"agent_{agent}_turn_{turn}", turns[turn], 2304, sha256
            )
            _resume_compute_and_release(manager, request, expected)
            total_hits += expected

    assert total_hits == 7_879_680


def test_dcp4_dflash_growing_history_reuses_warm_boundary() -> None:
    block_size = 2304
    manager = make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=4000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["target"],
                    MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        tokens_per_state=4,
                    ),
                    is_eagle_group=True,
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=block_size,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                    is_eagle_group=True,
                ),
                KVCacheGroupSpec(
                    ["draft"],
                    SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float16,
                        sliding_window=2048,
                        dcp_replicated=True,
                        extra_retained_tokens=2048,
                    ),
                    is_eagle_group=True,
                ),
            ],
            prefix_cache_retention_interval=0,
        ),
        max_model_len=524288,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=False,
        hash_block_size=block_size,
        scheduler_block_size=4 * block_size,
        dcp_world_size=4,
    )
    warm_tokens = list(range(122871))
    producer = make_request("dcp4_warm", warm_tokens, block_size, sha256)
    _compute_in_chunks(manager, producer, block_size)

    growing_tokens = [*warm_tokens[:122869], *range(200000, 212279)]
    replay = make_request("dcp4_turn_1", growing_tokens, block_size, sha256)
    _, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    _, hit_tokens, _ = manager.get_computed_blocks(replay)
    assert per_group == (119808, 122112, 119808), per_group
    assert hit_tokens == 117504, hit_tokens
    assert min(per_group) - hit_tokens == block_size
