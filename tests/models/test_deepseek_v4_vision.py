# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.vl_model import (
    _stream_language_model_weights,
)


def test_language_model_weight_routing_is_lazy():
    source_events: list[str] = []
    side_weights: list[str] = []

    def weights():
        for name in (
            "vision.patch_embed.weight",
            "language_model.model.embed.weight",
            "aligner.w1.weight",
            "language_model.model.layers.0.weight",
        ):
            source_events.append(name)
            yield name, torch.empty(1)

    routed = iter(
        _stream_language_model_weights(
            weights(), lambda name, _: side_weights.append(name)
        )
    )
    assert source_events == []

    first_name, _ = next(routed)
    assert first_name == "model.embed.weight"
    assert source_events == [
        "vision.patch_embed.weight",
        "language_model.model.embed.weight",
    ]
    assert side_weights == ["vision.patch_embed.weight"]

    second_name, _ = next(routed)
    assert second_name == "model.layers.0.weight"
    assert source_events == [
        "vision.patch_embed.weight",
        "language_model.model.embed.weight",
        "aligner.w1.weight",
        "language_model.model.layers.0.weight",
    ]
    assert side_weights == ["vision.patch_embed.weight", "aligner.w1.weight"]
