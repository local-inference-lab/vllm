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
    qwen_triton_warmup,
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
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
@pytest.mark.usefixtures("default_vllm_config")
def test_qwen_gdn_warmup_uses_per_head_norm_weights(monkeypatch) -> None:
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.model_executor.layers.mamba import mamba_utils

    device = torch.device("cuda")
    h, hv, k, v = 2, 2, 16, 16
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: True)
    layer = SimpleNamespace(
        num_k_heads=h,
        num_v_heads=hv,
        head_k_dim=k,
        head_v_dim=v,
        conv_kernel_size=4,
        tp_size=1,
        kv_cache=(
            torch.zeros(8, 2 * h * k + hv * v, 3, device=device, dtype=torch.bfloat16),
            torch.zeros(8, hv, v, k, device=device, dtype=torch.float32),
        ),
        norm=RMSNormGated(
            v,
            norm_before_gate=True,
            activation="silu",
            device=device,
            dtype=torch.bfloat16,
        ),
        A_log=torch.zeros(hv, device=device, dtype=torch.float32),
        dt_bias=torch.zeros(hv, device=device, dtype=torch.float32),
    )
    runner = SimpleNamespace(
        device=device,
        max_num_tokens=16,
        is_pooling_model=True,
        compilation_config=SimpleNamespace(static_forward_context={"gdn": layer}),
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
        dtype=torch.bfloat16,
    )

    qwen_triton_warmup(runner, model_config)
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

    def check_norm(*, group_size, rows_per_token, **kwargs):
        assert group_size == norm.weight.numel() == 128
        assert rows_per_token == 12
        calls.append((group_size, rows_per_token))

    monkeypatch.setattr(layernorm_guard, "warmup_layer_norm_fwd", check_norm)
    _warm_gated_rms_norm_kernel(
        torch.device("cpu"), config, max_num_tokens=16, x_dtype=torch.float32
    )
    assert calls == [(128, 12)]
