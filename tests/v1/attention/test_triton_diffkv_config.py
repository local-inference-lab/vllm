# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV cache capability and scale-contract checks without GPU execution."""

import pytest
import torch

from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
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


@pytest.mark.parametrize("dtype", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("checkpoint_scales", [None, (0.037, 0.061)])
def test_cache_scales_initialize_once_and_remain_stable(dtype, checkpoint_scales):
    layer = torch.nn.Module()
    layer.kv_cache_dtype = dtype
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        layer.register_buffer(name, torch.tensor(1.0))
    layer._k_scale_cpu = torch.tensor(1.0)
    layer._v_scale_cpu = torch.tensor(1.0)
    method = BaseKVCacheMethod(None)
    method.create_weights(layer)
    if checkpoint_scales is not None:
        for name, value in zip(("k_scale", "v_scale"), checkpoint_scales):
            parameter = getattr(layer, name)
            parameter.weight_loader(parameter, torch.tensor([value]))
    method.process_weights_after_loading(layer)
    expected = (
        checkpoint_scales
        if dtype == "fp8_e4m3" and checkpoint_scales is not None
        else (1.0, 1.0)
    )
    for name, value in zip(("_k_scale", "_v_scale"), expected):
        torch.testing.assert_close(getattr(layer, name), torch.tensor(value))
    # Repeated initialization must not change scales for already populated pages.
    method.process_weights_after_loading(layer)
    for name, value in zip(("_k_scale", "_v_scale"), expected):
        torch.testing.assert_close(getattr(layer, name), torch.tensor(value))
