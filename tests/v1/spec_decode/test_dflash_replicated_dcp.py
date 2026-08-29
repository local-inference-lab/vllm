# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


def test_dflash_marks_windowed_and_full_draft_kv_replicated(monkeypatch):
    attn = object.__new__(DFlashAttention)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
    )
    window_spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        sliding_window=2048,
    )
    monkeypatch.setattr(
        Attention,
        "get_kv_cache_spec",
        lambda self, config: window_spec,
    )

    spec = DFlashAttention.get_kv_cache_spec(attn, vllm_config)

    assert isinstance(spec, SlidingWindowSpec)
    # The target's 256-token block must not replace the backend-selected
    # 64-token block of the windowed draft cache.
    assert spec.block_size == 64
    assert spec.sliding_window == 2048
    assert spec.num_kv_heads == 2
    assert spec.dcp_replicated is True

    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float8_e4m3fn,
    )
    monkeypatch.setattr(
        Attention,
        "get_kv_cache_spec",
        lambda self, config: full_spec,
    )

    spec = DFlashAttention.get_kv_cache_spec(attn, vllm_config)

    assert isinstance(spec, FullAttentionSpec)
    assert spec.dcp_replicated is True


def test_dflash_dcp1_does_not_mark_draft_kv_replicated(monkeypatch):
    attn = object.__new__(DFlashAttention)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )
    monkeypatch.setattr(
        Attention,
        "get_kv_cache_spec",
        lambda self, config: window_spec,
    )

    spec = DFlashAttention.get_kv_cache_spec(attn, vllm_config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.dcp_replicated is False


def test_replicated_group_uses_unsharded_scheduler_and_table_geometry():
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
        dcp_replicated=True,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        kv_transfer_config=None,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )

    assert resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) == (16, 16)
    assert spec.max_num_blocks_per_req(vllm_config, max_len=4096) == 256


def test_mixed_target_and_replicated_draft_keep_distinct_dcp_geometry():
    target = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    draft = FullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        dcp_replicated=True,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=64,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        kv_transfer_config=None,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], target),
            KVCacheGroupSpec(["draft"], draft),
        ],
    )

    # The target scheduler unit remains DCP4-sharded while the draft block
    # hashes and table width retain ordinary DCP1 geometry.
    assert resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) == (256, 64)
    assert target.max_num_blocks_per_req(vllm_config, max_len=4096) == 16
    assert draft.max_num_blocks_per_req(vllm_config, max_len=4096) == 64


def test_grouping_never_merges_replicated_draft_with_sharded_target():
    target = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )
    draft = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        dcp_replicated=True,
    )
    assert target.page_size_bytes == draft.page_size_bytes

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    )
    grouped = get_kv_cache_groups(
        vllm_config, {"target.layer": target, "draft.layer": draft}
    )

    assert [set(group.layer_names) for group in grouped] == [
        {"target.layer"},
        {"draft.layer"},
    ]
