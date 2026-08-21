# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models import kimi_k25_vit
from vllm.model_executor.warmup import kimi_k3_triton_warmup
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    _warm_vision_position_interpolation,
)


def test_warm_vision_position_interpolation(monkeypatch) -> None:
    model = torch.nn.Sequential(
        kimi_k25_vit.Learnable2DInterpPosEmbDivided_fixed(
            height=14,
            width=14,
            num_frames=4,
            dim=32,
        )
    )
    get_rope_shape = Mock(return_value=torch.empty(64 * 64, 32))
    monkeypatch.setattr(kimi_k25_vit, "get_rope_shape", get_rope_shape)

    assert _warm_vision_position_interpolation(model) == 1
    get_rope_shape.assert_called_once_with(
        model[0].weight,
        interpolation_mode="bicubic",
        shape=(64, 64),
    )


def test_warm_vision_position_interpolation_ignores_other_modules() -> None:
    assert _warm_vision_position_interpolation(torch.nn.Linear(2, 2)) == 0


def test_kimi_warmup_runs_vision_before_kda_state_exists(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    calls = []

    def record_vision(_model):
        calls.append("vision")
        return 1

    def record_kda(_worker):
        calls.append("kda")

    warm_vision = Mock(side_effect=record_vision)
    get_kda_layer = Mock(side_effect=record_kda)
    monkeypatch.setattr(kimi_k3_triton_warmup.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        kimi_k3_triton_warmup,
        "_warm_vision_position_interpolation",
        warm_vision,
    )
    monkeypatch.setattr(kimi_k3_triton_warmup, "_get_kda_layer", get_kda_layer)

    kimi_k3_triton_warmup.kimi_k3_triton_warmup(
        SimpleNamespace(get_model=lambda: model)
    )

    warm_vision.assert_called_once_with(model)
    get_kda_layer.assert_called_once()
    assert calls == ["vision", "kda"]
