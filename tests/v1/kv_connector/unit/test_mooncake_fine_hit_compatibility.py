# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    ExternalCachedBlockPool,
    partial_hash_hits_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def test_attention_can_need_fine_hits_when_mamba_matches_hash_size():
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=2048, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=256,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    assert partial_hash_hits_enabled(groups, 256)
    groups.append(
        KVCacheGroupSpec(
            ["chunked"],
            ChunkedLocalAttentionSpec(
                block_size=2048,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                attention_chunk_size=2048,
            ),
        )
    )
    assert not partial_hash_hits_enabled(groups, 256)


@pytest.mark.parametrize("present", [False, True])
def test_swa_fine_lookup_accepts_external_hash_sequence(present):
    hashes = BlobBlockHashes(memoryview(bytes(range(32)) * 8), 32)
    pool = ExternalCachedBlockPool(32, None if present else set())
    spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=128,
    )
    _, hit = SlidingWindowManager.find_longest_cache_hit(
        hashes, 256, [0], pool, spec, False, alignment_tokens=32
    )
    assert hit == (256 if present else 0)
