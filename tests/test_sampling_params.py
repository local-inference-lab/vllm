# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(ValueError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


def test_prompt_logprobs_all_rejected_when_operator_allows_full_vocab():
    params = SamplingParams(prompt_logprobs=-1)
    model_config = MockModelConfig(max_logprobs=-1)

    with pytest.raises(VLLMValidationError, match="prompt_logprobs=-1") as exc_info:
        params.verify(model_config, None, None, None)

    assert exc_info.value.parameter == "prompt_logprobs"
    assert exc_info.value.value == -1


@pytest.mark.parametrize("prompt_logprobs", [1024, 1025])
def test_prompt_logprobs_full_vocab_or_larger_rejected(prompt_logprobs: int):
    params = SamplingParams(prompt_logprobs=prompt_logprobs)
    model_config = MockModelConfig(max_logprobs=2000)

    with pytest.raises(VLLMValidationError, match="entire vocabulary") as exc_info:
        params.verify(model_config, None, None, None)

    assert exc_info.value.parameter == "prompt_logprobs"
    assert exc_info.value.value == prompt_logprobs


def test_openai_protocols_reject_unbounded_prompt_logprobs():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    requests = (
        (
            ChatCompletionRequest,
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "prompt_logprobs": -1,
            },
        ),
        (
            CompletionRequest,
            {
                "model": "test-model",
                "prompt": "test",
                "prompt_logprobs": -1,
            },
        ),
    )
    for request_type, payload in requests:
        with pytest.raises(ValueError, match="prompt_logprobs=-1"):
            request_type.model_validate(payload)


@pytest.mark.parametrize("prompt_logprobs", [None, 0, 20])
def test_bounded_prompt_logprobs_unchanged(prompt_logprobs: int | None):
    params = SamplingParams(prompt_logprobs=prompt_logprobs)
    params.verify(MockModelConfig(max_logprobs=-1), None, None, None)
