# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


class CountedBlocks(list):
    reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


def make_manager(draft_slots):
    return MambaManager(
        MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,),
                  mamba_cache_mode="align", num_speculative_blocks=draft_slots),
        BlockPool(128, enable_caching=True, hash_block_size=16),
        enable_caching=True, kv_cache_group_id=0, scheduler_block_size=64,
    )


@pytest.mark.parametrize("draft_slots", [0, 3, 7])
@pytest.mark.parametrize("chunk", [64, 256])
def test_async_sparse_cleanup(draft_slots, chunk):
    manager = make_manager(draft_slots)
    pool = manager.block_pool
    rid = "async"
    endpoints = []
    for step in range(1, 7):
        computed = (step - 1) * chunk
        processed = max(0, computed - chunk)
        manager.remove_skipped_blocks(rid, processed)
        before = list(manager.req_to_blocks[rid])
        limit = max(0, (processed - 1) // 16)
        assert all(b.is_null for b in before[:limit])
        for index, block in endpoints:
            if index >= limit:
                assert before[index] is block and block.ref_cnt == 1
            else:
                assert before[index].is_null
        manager.get_num_blocks_to_allocate(
            rid, step * chunk, (), computed, computed, step * chunk)
        manager.allocate_new_blocks(rid, step * chunk, step * chunk)
        table = manager.req_to_blocks[rid]
        index = step * chunk // 16 - 1
        endpoints = [(i, b) for i, b in endpoints if i >= limit]
        endpoints.append((index, table[index]))
        assert all(not b.is_null and b.ref_cnt == 1 for b in table[index:])
        assert sum(not b.is_null for b in table) <= 3 + draft_slots
    manager.free(rid)
    assert pool.get_num_free_blocks() == 127
    # Reusing an ID after teardown must not inherit its old cleanup cursor.
    manager.get_num_blocks_to_allocate(rid, chunk, (), 0, 0, chunk)
    manager.allocate_new_blocks(rid, chunk, chunk)
    manager.remove_skipped_blocks(rid, chunk + 16)
    assert all(b.is_null for b in manager.req_to_blocks[rid][:chunk // 16])
    manager.free(rid)
    assert pool.get_num_free_blocks() == 127


def test_cleanup_does_not_rescan_history_or_advance_past_table():
    manager = make_manager(3)
    rid = "bounded"
    manager.remove_skipped_blocks(rid, 16000)
    blocks = CountedBlocks([manager.block_pool.null_block] * 1000)
    old, current = manager.block_pool.get_new_blocks(2)
    blocks[10], blocks[999] = old, current
    manager.req_to_blocks[rid] = blocks
    manager.remove_skipped_blocks(rid, 16000)
    assert old.ref_cnt == 0
    assert current.ref_cnt == 1
    blocks.reads = 0
    for _ in range(100):
        manager.remove_skipped_blocks(rid, 16000)
        manager.remove_skipped_blocks(rid, 15999)
    assert blocks.reads == 0
    manager.free(rid)
    assert manager.block_pool.get_num_free_blocks() == 127
