# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV cache capability and scale-contract checks without GPU execution."""

import pytest
import torch

from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
def test_diffkv_retains_unquantized_queries(dtype):
    assert dtype in TritonAttentionDiffKVBackend.supported_kv_cache_dtypes
    impl = TritonAttentionDiffKVImpl(
        num_heads=16,
        head_size=192,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=128,
        kv_cache_dtype=dtype,
    )
    assert not impl.supports_quant_query_input


@pytest.mark.parametrize(
    "dtype", ["fp8_e5m2", "int8_per_token_head", "fp8_per_token_head"]
)
def test_diffkv_rejects_unimplemented_cache_modes(dtype):
    assert dtype not in TritonAttentionDiffKVBackend.supported_kv_cache_dtypes
    with pytest.raises(NotImplementedError):
        TritonAttentionDiffKVImpl(
            num_heads=16,
            head_size=192,
            scale=192**-0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=dtype,
        )


@pytest.mark.parametrize("scale", [None, torch.ones(2)])
def test_diffkv_requires_scalar_descales(scale):
    with pytest.raises(ValueError, match="(separate K and V|per-tensor K/V)"):
        unified_attention_diffkv(
            q=torch.empty(1, 16, 192, dtype=torch.bfloat16),
            k=torch.empty(1, 16, 1, 192, dtype=torch.float8_e4m3fn),
            v=torch.empty(1, 16, 1, 128, dtype=torch.float8_e4m3fn),
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=192**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=None,
            softcap=0,
            k_descale=scale,
            v_descale=scale,
        )
