# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _qwen_gdn_warmup_config,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
    _warm_layer_norm_kernel,
)
from vllm.platforms import current_platform


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
        norm_weight=torch.ones(v, dtype=torch.bfloat16, device=device),
        norm_bias=None,
        norm_eps=1e-6,
        norm_group_size=v,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    _warm_layer_norm_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


def test_qwen_gdn_norm_warmup_preserves_per_head_shape(monkeypatch) -> None:
    """TP4 heads share a 128-wide norm weight, not a 1536-wide weight."""
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard

    norm = SimpleNamespace(
        weight=torch.ones(128),
        bias=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )
    layer = SimpleNamespace(
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        tp_size=4,
        norm=norm,
        A_log=torch.zeros(12),
        dt_bias=torch.zeros(12),
        kv_cache=(torch.empty(1, 2560, 3), torch.empty(1, 12, 128, 128)),
    )
    config = _qwen_gdn_warmup_config({"linear_attn": layer})
    assert config is not None
    calls = []

    def check_norm(x, weight, bias, eps, *, z, out, group_size, **kwargs):
        assert x.shape[1] == weight.numel() == group_size == 128
        assert x.shape[0] % 12 == 0
        assert z.shape == out.shape == x.shape
        calls.append(x.shape[0])

    monkeypatch.setattr(layernorm_guard, "layer_norm_fwd", check_norm)
    _warm_layer_norm_kernel(torch.device("cpu"), config)
    assert calls and min(calls) == 12
