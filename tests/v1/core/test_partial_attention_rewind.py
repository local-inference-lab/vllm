# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse a cached attention tail below its published hash boundary."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("dcp,replicated", [(1, False), (4, False), (4, True)])
@pytest.mark.parametrize("leading_blocks", [0, 1])
@pytest.mark.parametrize("drop_eagle", [False, True])
def test_replay_below_published_attention_tail(
    dcp, replicated, leading_blocks, drop_eagle
):
    init_none_hash(sha256)
    unit = 256
    spec = FullAttentionSpec(
        block_size=2048,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        dcp_replicated=replicated,
    )
    effective = spec.block_size * (1 if replicated else dcp)
    boundary = leading_blocks * effective + 2048
    producer = make_request("producer", list(range(boundary)), unit, sha256)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=unit)
    blocks = pool.get_new_blocks(leading_blocks + 1)
    full_blocks = boundary // effective
    pool.cache_full_blocks(producer, blocks, 0, full_blocks, effective, 0)
    if boundary % effective:
        pool.cache_partial_block(producer, blocks[-1], boundary, 0, effective)
    pool.free_blocks(blocks)

    def lookup(tokens):
        request = make_request("consumer", tokens, unit, sha256)
        return FullAttentionManager.find_longest_cache_hit(
            request.block_hashes,
            max_length=boundary - 1,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=drop_eagle,
            alignment_tokens=unit,
            dcp_world_size=dcp,
        )

    found, hit = lookup(list(range(boundary)))
    expected = boundary - unit * (1 + int(drop_eagle))
    assert hit == expected
    assert list(found[0]) == blocks

    divergent = list(range(boundary))
    divergent[-1] += 1
    _, hit = lookup(divergent)
    assert hit == max(0, leading_blocks * effective - unit * int(drop_eagle))
