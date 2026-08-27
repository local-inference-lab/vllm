# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest

import vllm.models.kimi_k3.common.mm_preprocess as mm_preprocess


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|media_pad|>"
        return 42

    def decode(self, token_id: int) -> str:
        assert token_id == 42
        return "<|media_pad|>"


class _ProcessingContext:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None = None) -> None:
        multimodal_config = SimpleNamespace(
            mm_processor_kwargs=mm_processor_kwargs,
        )
        self.model_config = SimpleNamespace(
            model="moonshotai/Kimi-K3",
            revision="test-revision",
            trust_remote_code=True,
            get_multimodal_config=lambda: multimodal_config,
        )
        self._hf_config = SimpleNamespace(media_placeholder_token_id=42)

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def get_hf_config(self, *_args: Any) -> SimpleNamespace:
        return self._hf_config


def test_kimi_k3_applies_static_image_patch_limits(
    monkeypatch: pytest.MonkeyPatch,
):
    image_processor = SimpleNamespace(
        media_proc_cfg={
            "in_patch_limit": 65536,
            "patch_limit_on_one_side": 512,
        },
        media_tokens_calculator=lambda _media: 1,
    )
    monkeypatch.setattr(
        mm_preprocess,
        "cached_get_image_processor",
        lambda *_args, **_kwargs: image_processor,
    )

    info = mm_preprocess.KimiK3ProcessingInfo(  # type: ignore[arg-type]
        _ProcessingContext(
            {
                "in_patch_limit": 49152,
                "patch_limit_on_one_side": 384,
            }
        )
    )

    assert info.image_processor is not image_processor
    assert info.image_processor.media_proc_cfg == {
        "in_patch_limit": 49152,
        "patch_limit_on_one_side": 384,
    }
    assert image_processor.media_proc_cfg == {
        "in_patch_limit": 65536,
        "patch_limit_on_one_side": 512,
    }


@pytest.mark.parametrize(
    ("processor_kwargs", "message"),
    [
        ({"in_patch_limit": True}, "positive integer"),
        ({"in_patch_limit": 65537}, "exceeds the checkpoint limit"),
        ({"patch_limit_on_one_side": 513}, "exceeds the checkpoint limit"),
    ],
)
def test_kimi_k3_rejects_invalid_image_patch_limits(
    monkeypatch: pytest.MonkeyPatch,
    processor_kwargs: dict[str, object],
    message: str,
):
    image_processor = SimpleNamespace(
        media_proc_cfg={
            "in_patch_limit": 65536,
            "patch_limit_on_one_side": 512,
        },
        media_tokens_calculator=lambda _media: 1,
    )
    monkeypatch.setattr(
        mm_preprocess,
        "cached_get_image_processor",
        lambda *_args, **_kwargs: image_processor,
    )

    with pytest.raises(ValueError, match=message):
        mm_preprocess.KimiK3ProcessingInfo(  # type: ignore[arg-type]
            _ProcessingContext(processor_kwargs)
        )
