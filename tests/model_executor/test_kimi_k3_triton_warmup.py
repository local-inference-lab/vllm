# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    _get_kda_layer,
    _warm_chunk_kda_prefill,
)
from vllm.models.kimi_k3.nvidia.ops.third_party import kda


def test_kda_layer_lookup_prefers_target_model(monkeypatch) -> None:
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

    assert _get_kda_layer(worker) is target_layer


def test_triton_kda_prefill_warmup_uses_production_shape(monkeypatch) -> None:
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

    assert _warm_chunk_kda_prefill(layer, torch.bfloat16) is True

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

    assert _warm_chunk_kda_prefill(layer, torch.bfloat16) is False
