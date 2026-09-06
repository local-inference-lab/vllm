# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    RequestTracker,
)
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


@pytest.fixture(autouse=True)
def init_hash():
    init_none_hash(sha256)


def config(block=256, attention=2048, spec_blocks=0):
    return KVCacheConfig(
        num_blocks=512,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=attention,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=spec_blocks,
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=attention,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=attention,
                ),
                is_eagle_group=True,
            ),
        ],
    )


def test_external_fine_hit_gate_matches_engine():
    cfg = config()
    engine = make_kv_cache_manager(
        cfg,
        max_model_len=65536,
        hash_block_size=256,
        enable_caching=True,
        use_eagle=True,
        retention_interval=0,
    )
    external = MooncakeStoreCoordinator(
        cfg.kv_cache_groups, 2048, 256, use_eagle=True, retention_interval=0
    )
    assert engine.coordinator.enable_partial_hash_hits
    assert external.enable_partial_hash_hits, (
        "Engine enables fine hits but Mooncake disables them"
    )


def test_positive_retention_boundary_is_materialized():
    req = make_request("retention", list(range(16384)), 512, sha256)
    req.num_computed_tokens = 3584
    scheduler = NS(
        cache_config=NS(block_size=512, prefix_cache_retention_interval=4096),
        scheduler_config=NS(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=4096,
        use_eagle=True,
        drop_last_prefix_cache_block=False,
        mamba_has_prefill_checkpoint_blocks=True,
        hash_block_size=512,
        mamba_partial_cache_hit=False,
    )
    step = Scheduler._mamba_block_aligned_split(scheduler, req, 4089)
    assert step == 512, f"Crosses required token 4096 and ends at {3584 + step}"


def test_semantic_replay_expands_each_boundary_once():
    cfg = config()
    core = make_kv_cache_manager(
        cfg,
        max_model_len=65536,
        hash_block_size=256,
        enable_caching=True,
        use_eagle=True,
        retention_interval=0,
    )
    mamba = core.coordinator.single_type_managers[1]
    assert set(mamba._expand_reachable_boundaries([65535])) == {
        61440,
        63488,
        65024,
        65280,
    }
    for mgr in core.coordinator.single_type_managers:
        if isinstance(mgr.kv_cache_spec, SlidingWindowSpec):
            import inspect

            inspect.signature(mgr.cache_blocks).bind(object(), 0, alignment_tokens=256)


def test_positional_store_does_not_read_relocated_speculative_block():
    cfg = config(16, 16, 2)
    cfg.kv_cache_groups = cfg.kv_cache_groups[:2]
    core = make_kv_cache_manager(
        cfg,
        max_model_len=1024,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=True,
        retention_interval=48,
    )
    req = make_request("producer", list(range(128)), 16, sha256)
    first = core.allocate_slots(req, 32, num_lookahead_tokens=2)
    assert first is not None
    tracker = RequestTracker(
        req_id=req.request_id,
        token_len=32,
        allocated_block_ids=core.get_blocks(req.request_id).get_block_ids(),
        num_saved_tokens=0,
    )
    req.num_computed_tokens = 32
    core.new_step_starts()
    second = core.allocate_slots(req, 64, num_lookahead_tokens=2)
    assert second is not None
    tracker.update(second.get_block_ids())
    req.num_computed_tokens = 96
    current = core.get_blocks(req.request_id).get_block_ids()[1]
    old = tracker.allocated_block_ids[1]
    print("Mamba current block IDs:", current, "connector snapshot:", old)
    assert old[2] == current[5], (
        "Reproducer requires the 48-token slot to alias the 96-token state"
    )
    coord = MooncakeStoreCoordinator(
        cfg.kv_cache_groups, 16, 16, use_eagle=True, retention_interval=48
    )
    masks = coord.store_mask(96, 0, num_prompt_tokens=128)
    db = ChunkedTokenDatabase(
        KeyMetadata("review", 0, 0, 0, 0, group_id=1), block_size=16
    )
    stored = list(db.process_tokens(96, req.block_hashes, chunk_mask=masks[1]))
    print("Mamba store spans:", [(s, e) for s, e, _ in stored])
    assert not any(e == 48 for s, e, _ in stored), (
        "Stores the 96-token state under the valid 48-token prefix hash"
    )


def test_fine_swa_lookup_accepts_mooncake_hash_sequence():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        BlobBlockHashes,
    )
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    hashes = BlobBlockHashes(memoryview(bytes(range(32)) * 8), 32)
    pool = ExternalCachedBlockPool(32, set())
    spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=128,
    )
    blocks, hit = SlidingWindowManager.find_longest_cache_hit(
        hashes, 256, [0], pool, spec, False, alignment_tokens=32
    )
    assert hit == 0
