# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The connector registers target caches without mutating runner-owned groups."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    _tp9_target_cache_view,
)


@dataclass
class CacheConfigView:
    kv_cache_groups: list


def config(tp=9, dcp=9):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=dcp
        ),
    )


def test_target_cache_view_preserves_original_groups_and_target_order():
    target = [
        SimpleNamespace(layer_names=["language_model.mla"]),
        SimpleNamespace(layer_names=["language_model.kda"]),
    ]
    draft = SimpleNamespace(layer_names=["dflash_model.layers.0.self_attn.attn"])
    original = CacheConfigView(target + [draft])
    selected, excluded = _tp9_target_cache_view(config(), original)
    assert selected.kv_cache_groups == target
    assert original.kv_cache_groups == target + [draft]
    assert excluded == frozenset(draft.layer_names)
    unchanged, excluded = _tp9_target_cache_view(config(8, 8), original)
    assert unchanged is original
    assert not excluded


def test_mixed_target_draft_groups_are_rejected():
    mixed = SimpleNamespace(layer_names=["language_model.mla", "dflash_model.attn"])
    with pytest.raises(ValueError, match="mixed target/draft"):
        _tp9_target_cache_view(config(), CacheConfigView([mixed]))


def test_filtering_does_not_renumber_target_groups_after_a_draft_group():
    groups = [
        SimpleNamespace(layer_names=["dflash_model.attn"]),
        SimpleNamespace(layer_names=["language_model.mla"]),
    ]
    with pytest.raises(ValueError, match="follow all target"):
        _tp9_target_cache_view(config(), CacheConfigView(groups))
