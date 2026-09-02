# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LP15 controls adapted from official vLLM hybrid-cache PRs.

PRs 52244 and 53802 cover exact prompt-boundary registration. PR 52371
pins the EAGLE/Mamba reconciliation loss, and PR 53945 adds a fine-grained
resume checkpoint. LP15 production does not enable fine-grained hashing, so
the aligned cases are hard regressions while the optional fine-grained case
stays an explicit upstream xfail.
"""

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
)


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _manager(block_size: int, hash_block_size: int):
    return make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=256,
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
                ),
            ],
        ),
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=True,
    )


def _seed(manager, request, stops: tuple[int, ...]) -> None:
    computed, num_computed, _ = manager.get_computed_blocks(request)
    first = True
    for stop in stops:
        step = stop - request.num_computed_tokens
        blocks = manager.allocate_slots(
            request,
            step,
            num_computed if first else 0,
            computed if first else None,
        )
        assert blocks is not None
        request.num_computed_tokens = stop
        manager.new_step_starts()
        first = False
    manager.cache_blocks(request, request.num_computed_tokens)
    manager.free(request)
    manager.new_step_starts()


def _hits(manager, token_ids: list[int]) -> tuple[int, tuple[int, ...]]:
    follower = make_request(
        "follower", token_ids, manager.block_pool.hash_block_size, sha256
    )
    _, joint, _ = manager.get_computed_blocks(follower)
    _, per_group = manager.coordinator.find_longest_cache_hit_per_group(
        follower.block_hashes, follower.num_tokens - 1
    )
    return joint, tuple(per_group)


def test_exact_aligned_prompt_retains_post_eagle_resume_state() -> None:
    block = 16
    manager = _manager(block, block)
    owner = make_request("owner", [7] * 48, block, sha256)
    _seed(manager, owner, (16, 32, 48))

    joint, per_group = _hits(manager, [7] * 48 + [9] * 8)

    assert per_group == (32, 48)
    assert joint == 32


def test_shared_system_boundary_retains_post_eagle_resume_state() -> None:
    block = 16
    manager = _manager(block, block)
    owner = make_request("owner", [7] * 32 + [8] * 16, block, sha256)
    _seed(manager, owner, (16, 32, 48))

    joint, per_group = _hits(manager, [7] * 32 + [9] * 8)

    assert per_group == (16, 32)
    assert joint == 16


@pytest.mark.xfail(
    strict=True,
    reason=(
        "official PR 53945 fine-grained resume checkpoints remain opt-in and "
        "LP15 production keeps hash and scheduler alignment equal"
    ),
)
def test_fine_grained_resume_checkpoint_is_not_claimed() -> None:
    block = 16
    hash_unit = 4
    manager = _manager(block, hash_unit)
    owner = make_request("owner", [7] * 48, hash_unit, sha256)
    _seed(manager, owner, (12, 16, 28, 32, 44, 48))

    joint, per_group = _hits(manager, [7] * 48 + [9] * 8)

    assert min(per_group) >= 44
    assert joint == 44
