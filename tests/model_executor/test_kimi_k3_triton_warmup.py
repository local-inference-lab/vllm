# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    _get_kda_layer,
    _warm_chunk_kda_prefill,
    _warm_recurrent_kda,
)
from vllm.models.kimi_k3.nvidia.ops.third_party import kda
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent


def test_kda_layer_lookup_before_kv_cache_binding(monkeypatch) -> None:
    class FakeKimiK3DeltaAttention:
        pass

    layer = FakeKimiK3DeltaAttention()
    target_model = SimpleNamespace(modules=lambda: (layer,))
    worker = SimpleNamespace(
        get_model=lambda: target_model,
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={"layer": layer},
            )
        ),
    )
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention",
        FakeKimiK3DeltaAttention,
    )

    assert not hasattr(layer, "kv_cache")
    assert _get_kda_layer(worker) is layer


def test_kda_layer_lookup_ignores_speculative_draft(monkeypatch) -> None:
    class FakeKimiK3DeltaAttention:
        pass

    target_layer = FakeKimiK3DeltaAttention()
    draft_layer = FakeKimiK3DeltaAttention()
    worker = SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: (target_layer,)),
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "draft.layers.0": draft_layer,
                    "target.layers.0": target_layer,
                },
            )
        ),
    )
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention",
        FakeKimiK3DeltaAttention,
    )

    assert not hasattr(target_layer, "kv_cache")
    assert _get_kda_layer(worker) is target_layer


def test_speculative_kda_warmup_before_kv_cache_binding(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        fused_recurrent,
        "get_fused_recurrent_kda_fwd_warmup_profiles",
        lambda _num_heads: (2,),
    )
    monkeypatch.setattr(
        fused_recurrent,
        "fused_recurrent_kda",
        lambda **kwargs: calls.append(kwargs),
    )

    layer = SimpleNamespace(
        num_spec=7,
        local_num_heads=2,
        head_dim=4,
        A_log=torch.empty(2, dtype=torch.float32),
        dt_bias=torch.empty(8, dtype=torch.float32),
        gate_lower_bound=-10.0,
        get_state_shape=lambda: ((10, 4), (2, 4, 4)),
        get_state_dtype=lambda: (torch.bfloat16, torch.float32),
    )

    _warm_recurrent_kda(layer, torch.bfloat16)

    assert len(calls) == 1
    call = calls[0]
    assert call["initial_state"].shape == (1, 2, 4, 4)
    assert call["initial_state"].dtype == torch.float32
    assert call["q"].shape == (1, 16, 2, 4)
    assert call["ssm_state_indices"].shape == (2, 8)


def test_triton_kda_prefill_warmup_before_kv_cache_binding(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        kda,
        "chunk_kda_with_fused_gate",
        lambda **kwargs: calls.append(kwargs),
    )
    layer = SimpleNamespace(
        kda_prefill_backend="triton",
        local_num_heads=2,
        head_dim=4,
        A_log=torch.empty(2, dtype=torch.float32),
        dt_bias=torch.empty(8, dtype=torch.float32),
        gate_lower_bound=-10.0,
        get_state_shape=lambda: ((10, 4), (2, 4, 4)),
        get_state_dtype=lambda: (torch.bfloat16, torch.float32),
    )

    _warm_chunk_kda_prefill(layer, torch.bfloat16)

    assert len(calls) == 1
    call = calls[0]
    assert call["q"].shape == (1, 64, 2, 4)
    assert call["raw_g"].shape == (1, 64, 2, 4)
    assert call["raw_beta"].shape == (1, 64, 2)
    assert call["initial_state"].shape == (1, 2, 4, 4)
    assert call["initial_state"].dtype == torch.float32
    assert call["cu_seqlens"].tolist() == [0, 64]
    assert call["output_final_state"] is True
    assert call["use_qk_l2norm_in_kernel"] is True


def test_flashkda_prefill_does_not_warm_triton_kernels(monkeypatch) -> None:
    def fail_if_called(**_kwargs) -> None:
        raise AssertionError("FlashKDA must not compile the Triton prefill path")

    monkeypatch.setattr(kda, "chunk_kda_with_fused_gate", fail_if_called)
    layer = SimpleNamespace(kda_prefill_backend="flashkda")

    _warm_chunk_kda_prefill(layer, torch.bfloat16)
