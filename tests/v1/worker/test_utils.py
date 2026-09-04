# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import (
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
    should_defer_draft_after_partial_dcp_resume,
)


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def test_copy_kv_cache_blocks_uses_logical_strided_views() -> None:
    """CoW copies each logical page without copying backing-store padding."""
    num_blocks = 4
    backing = torch.full((num_blocks, 32), 0xEE, dtype=torch.uint8)
    first = backing.as_strided((num_blocks, 3, 4), (32, 4, 1), 0)
    second = backing.as_strided((num_blocks, 2, 4), (32, 4, 1), 16)
    first[1].fill_(0x11)
    second[1].fill_(0x22)
    first[3].zero_()
    second[3].zero_()
    padding_before = backing[3, 12:16].clone()

    copy_kv_cache_blocks_inplace([first, second], num_blocks, [KVCacheBlockCopy(1, 3)])

    assert torch.equal(first[3], first[1])
    assert torch.equal(second[3], second[1])
    assert torch.equal(backing[3, 12:16], padding_before)


def test_copy_kv_cache_blocks_detects_nonzero_block_axis() -> None:
    """K/V-first cache layouts copy along their unique block dimension."""
    cache = torch.zeros((2, 4, 3, 2), dtype=torch.int32)
    cache[:, 1].fill_(17)
    cache[:, 3].fill_(-1)

    copy_kv_cache_blocks_inplace([cache], 4, [KVCacheBlockCopy(1, 3)])

    assert torch.equal(cache[:, 3], cache[:, 1])


def test_copy_kv_cache_blocks_scopes_tagged_copy_to_group() -> None:
    """A group's CoW must not overwrite a live block in another group."""
    num_blocks = 8
    attention_cache = torch.zeros((num_blocks, 2), dtype=torch.int32)
    recurrent_cache = torch.zeros((num_blocks, 2), dtype=torch.int32)
    attention_cache[1].fill_(11)
    attention_cache[5].fill_(-1)
    recurrent_cache[1].fill_(22)
    recurrent_cache[5].fill_(55)

    copy_kv_cache_blocks_inplace(
        [attention_cache, recurrent_cache],
        num_blocks,
        [KVCacheBlockCopy(1, 5, kv_cache_group_id=0)],
        kv_cache_groups=[[attention_cache], [recurrent_cache]],
    )

    assert torch.equal(attention_cache[5], attention_cache[1])
    assert torch.equal(recurrent_cache[5], torch.full_like(recurrent_cache[5], 55))


def test_defer_draft_only_for_partial_packed_dcp_resume() -> None:
    kwargs = {
        "cache_dtype": "fp8_ds_mla",
        "dcp_size": 8,
        "block_size": 1536,
        "is_prefilling": np.array([True]),
    }

    assert should_defer_draft_after_partial_dcp_resume(
        **kwargs, num_cached_tokens=np.array([4608])
    )
    assert not should_defer_draft_after_partial_dcp_resume(
        **kwargs, num_cached_tokens=np.array([12288])
    )
    assert not should_defer_draft_after_partial_dcp_resume(
        **kwargs, num_cached_tokens=np.array([0])
    )
    assert not should_defer_draft_after_partial_dcp_resume(
        **(kwargs | {"cache_dtype": "fp8"}),
        num_cached_tokens=np.array([4608]),
    )
    assert not should_defer_draft_after_partial_dcp_resume(
        **(kwargs | {"is_prefilling": np.array([False])}),
        num_cached_tokens=np.array([4608]),
    )
