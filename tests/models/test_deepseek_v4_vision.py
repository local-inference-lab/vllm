# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    DeepseekV4VLProcessingInfo,
)
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


class _TokenizerFixture:
    def __init__(self, vocabulary: dict[str, int], unknown_id: int = 0):
        self.vocabulary = vocabulary
        self.unknown_id = unknown_id

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary.get(token, self.unknown_id)


class _ProcessingInfoFixture:
    def __init__(self, tokenizer: _TokenizerFixture):
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> _TokenizerFixture:
        return self.tokenizer


def test_image_placeholder_requires_exact_vocabulary_entry():
    info = _ProcessingInfoFixture(_TokenizerFixture({}, unknown_id=17))

    with pytest.raises(ValueError, match="Token not found in tokenizer"):
        DeepseekV4VLProcessingInfo.get_image_placeholder_token_id(info)  # type: ignore[arg-type]


def test_image_placeholder_returns_exact_vocabulary_id():
    info = _ProcessingInfoFixture(
        _TokenizerFixture({IMAGE_PLACEHOLDER: 42}, unknown_id=17)
    )

    assert (
        DeepseekV4VLProcessingInfo.get_image_placeholder_token_id(info)  # type: ignore[arg-type]
        == 42
    )
