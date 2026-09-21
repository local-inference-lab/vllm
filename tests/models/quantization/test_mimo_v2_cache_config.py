# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo target and native draft constructors retain the requested cache policy."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.mimo_v2 as mimo
import vllm.model_executor.models.mimo_v2_mtp as mtp

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("kind", ["global", "sliding", "mtp"])
@pytest.mark.parametrize("dtype", ["bfloat16", "fp8_e4m3"])
def test_mimo_attention_receives_cache_policy(monkeypatch, kind, dtype):
    observed = []

    class Attention(torch.nn.Module):
        def __init__(self, *, cache_config=None, **kwargs):
            super().__init__()
            observed.append(cache_config)

    for module in (mimo, mtp):
        monkeypatch.setattr(module, "MiMoV2Attention", Attention)
        monkeypatch.setattr(module, "MiMoV2MLP", lambda **kwargs: torch.nn.Identity())
        monkeypatch.setattr(module, "RMSNorm", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(mtp, "ReplicatedLinear", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(
        mtp, "VocabParallelEmbedding", lambda *a, **k: torch.nn.Identity()
    )
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=16,
        swa_num_key_value_heads=2,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window_size=128,
        attention_bias=False,
        intermediate_size=64,
        hidden_act="silu",
        layernorm_epsilon=1e-6,
        vocab_size=64,
        hybrid_layer_pattern=[int(kind == "sliding")],
    )
    cache_config = SimpleNamespace(
        cache_dtype=dtype, calculate_kv_scales=True, kv_cache_dtype_skip_layers=["3"]
    )
    vconfig = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, hf_config=config),
        quant_config=None,
        cache_config=cache_config,
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
    )
    if kind == "mtp":
        mtp.MiMoV2MultiTokenPredictor(vllm_config=vconfig, prefix="model")
    else:
        mimo.MiMoV2FlashDecoderLayer(vllm_config=vconfig, prefix="model.layers.0")
    assert len(observed) == 1
    # The whole policy must reach Attention, including scale calibration and
    # intentional skip layers, not just a copied dtype string.
    assert observed[0] is cache_config
