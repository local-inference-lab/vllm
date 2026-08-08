# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security test for prompt_logprobs=-1 rejection (M8)."""

from unittest.mock import MagicMock

import pytest

from vllm.sampling_params import SamplingParams


def _mock_model_config(vocab_size=154880, max_logprobs=-1):
    """Create a mock ModelConfig with a large vocab and max_logprobs=-1."""
    config = MagicMock()
    config.get_vocab_size.return_value = vocab_size
    config.max_logprobs = max_logprobs
    config.get_hidden_size.return_value = 4096
    config.is_encoder_decoder = False
    config.is_diffusion = False
    return config


def test_prompt_logprobs_minus_one_rejected():
    """M8: prompt_logprobs=-1 must be rejected regardless of max_logprobs.

    Without this fix, prompt_logprobs=-1 resolves to vocab_size (154,880 for
    GLM-5.2) and allocates a [num_prompt_tokens, vocab_size] tensor that OOMs
    the engine (upstream vllm-project/vllm#14239).
    """
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=-1)

    with pytest.raises(VLLMValidationError, match="prompt_logprobs=-1"):
        sp._validate_logprobs(model_config)


def test_prompt_logprobs_positive_still_works():
    """M8: a concrete prompt_logprobs value should still pass validation."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=20)
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_zero_still_works():
    """M8: prompt_logprobs=0 (disabled) should not trigger the rejection."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=0)
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_none_still_works():
    """M8: prompt_logprobs=None (unset) should not trigger the rejection."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams()
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_exceeds_max_still_rejected():
    """M8: prompt_logprobs > max_logprobs is still rejected by the existing
    check (confirms the existing behavior is preserved)."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=20)
    sp = SamplingParams(prompt_logprobs=100)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
