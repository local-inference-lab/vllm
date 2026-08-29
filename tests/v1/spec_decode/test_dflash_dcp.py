# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def test_dflash_sliding_window_cache_is_replicated_under_dcp():
    attention = SimpleNamespace(
        sliding_window=2048,
        attn_type=AttentionType.DECODER,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        kv_cache_torch_dtype=torch.float8_e4m3fn,
        kv_cache_dtype="fp8",
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
    )

    spec = DFlashAttention.get_kv_cache_spec(attention, config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.sliding_window == 2048
    assert spec.extra_retained_tokens == 2048
    assert spec.dcp_replicated is True
