# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSpark block-unaligned prefix repair.

A local prefix-cache hit whose token count is not block-aligned restores
tokens that never flowed through the target forward, so DFlash/DSpark draft
KV does not exist for them and drafting fail-closes for the whole batch.
The repair adopts only the block-aligned region and lets the residual tail
prefill normally. These tests cover the scheduler-side clamp, its counters
and its gating.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from .utils import create_scheduler


def _scheduler(speculative_method: str | None):
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=16)
    if speculative_method is not None:
        scheduler.vllm_config.speculative_config = SimpleNamespace(
            method=speculative_method
        )
    return scheduler


def _make_manager_stub(scheduler):
    calls = []

    def truncate_computed_blocks(blocks, num_computed_tokens):
        calls.append(num_computed_tokens)
        return blocks

    manager = SimpleNamespace(
        coordinator=scheduler.kv_cache_manager.coordinator,
        truncate_computed_blocks=truncate_computed_blocks,
    )
    return manager, calls


def _blocks():
    return KVCacheBlocks(tuple())


@pytest.mark.parametrize("speculative_method", ["dspark", "dflash"])
def test_unaligned_hit_is_repaired_for_draft_speculation(speculative_method: str):
    scheduler = _scheduler(speculative_method)
    manager, truncate_calls = _make_manager_stub(scheduler)
    scheduler.kv_cache_manager = manager

    blocks, aligned, partial_tail = scheduler._repair_unaligned_dspark_prefix(
        _blocks(), 33
    )

    assert aligned == 32
    assert partial_tail == 1
    assert truncate_calls == [32]
    assert scheduler.num_dspark_prefix_repairs == 1
    assert scheduler.num_dspark_prefix_repair_tokens == 1


def test_aligned_hit_is_untouched_for_dspark():
    scheduler = _scheduler("dspark")
    manager, truncate_calls = _make_manager_stub(scheduler)
    scheduler.kv_cache_manager = manager

    blocks, aligned, partial_tail = scheduler._repair_unaligned_dspark_prefix(
        _blocks(), 48
    )

    assert aligned == 48
    assert partial_tail == 0
    assert truncate_calls == []
    assert scheduler.num_dspark_prefix_repairs == 0
    assert scheduler.num_dspark_prefix_repair_tokens == 0


@pytest.mark.parametrize("speculative_method", [None, "eagle"])
def test_no_repair_without_supported_speculation(speculative_method: str | None):
    scheduler = _scheduler(speculative_method)
    manager, truncate_calls = _make_manager_stub(scheduler)
    scheduler.kv_cache_manager = manager

    blocks, aligned, partial_tail = scheduler._repair_unaligned_dspark_prefix(
        _blocks(), 33
    )

    assert aligned == 33
    assert partial_tail == 0
    assert truncate_calls == []
    assert scheduler.num_dspark_prefix_repairs == 0


def test_no_repair_when_connector_present():
    scheduler = _scheduler("dspark")
    manager, truncate_calls = _make_manager_stub(scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.connector = SimpleNamespace()

    blocks, aligned, partial_tail = scheduler._repair_unaligned_dspark_prefix(
        _blocks(), 33
    )

    assert aligned == 33
    assert partial_tail == 0
    assert truncate_calls == []
    assert scheduler.num_dspark_prefix_repairs == 0


@pytest.mark.parametrize(
    "unsupported_mode",
    ["dcp", "mamba_align", "partial_hash", "multiple_full_attention"],
)
def test_repair_preserves_excluded_cache_topologies(unsupported_mode: str):
    scheduler = _scheduler("dspark")
    manager, truncate_calls = _make_manager_stub(scheduler)
    scheduler.kv_cache_manager = manager

    if unsupported_mode == "dcp":
        scheduler.dcp_world_size = 2
    elif unsupported_mode == "mamba_align":
        scheduler.need_mamba_block_aligned_split = True
    elif unsupported_mode == "partial_hash":
        manager.coordinator.enable_partial_hash_hits = True
    else:
        group = scheduler.kv_cache_config.kv_cache_groups[0]
        scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[group, group])

    original_blocks = _blocks()
    blocks, aligned, partial_tail = scheduler._repair_unaligned_dspark_prefix(
        original_blocks, 33
    )

    assert blocks is original_blocks
    assert aligned == 33
    assert partial_tail == 0
    assert truncate_calls == []
    assert scheduler.num_dspark_prefix_repairs == 0
