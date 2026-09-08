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


def _full_grid_rope_reference(
    rope: kimi_k25_vit.Rope2DPosEmbRepeated,
    shapes: list[list[int]],
) -> torch.Tensor:
    flat_pos = torch.arange(rope.max_height * rope.max_width).float()
    x_pos = flat_pos % rope.max_width
    y_pos = flat_pos // rope.max_width
    dim_range = torch.arange(0, rope.dim, 4).float()
    freqs = 1.0 / (rope.theta_base ** (dim_range / rope.dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    table = torch.cat(
        [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
    ).reshape(rope.max_height, rope.max_width, -1)
    return torch.cat(
        [table[:h, :w].reshape(-1, rope.dim // 2).repeat(t, 1) for t, h, w in shapes]
    )


def test_vision_rope_materializes_only_requested_grids() -> None:
    rope = kimi_k25_vit.Rope2DPosEmbRepeated(
        dim=32,
        max_height=11,
        max_width=13,
    )
    shapes = [[1, 3, 5], [2, 4, 2], [1, 3, 5]]

    actual = rope.get_freqs_cis(shapes, device=torch.device("cpu"))
    expected = _full_grid_rope_reference(rope, shapes)

    assert torch.equal(actual, expected)
    assert actual.shape == (46, 16)
    assert not hasattr(rope, "freqs_cis")


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
