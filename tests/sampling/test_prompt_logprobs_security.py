# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security tests for prompt_logprobs resource-bound validation.

Replaces the original sentinel-only rejection (-1) with a resource bound
(``VLLM_MAX_PROMPT_LOGPROBS``).  The bound closes the bypass where an
operator-configured ``max_logprobs=-1`` makes the allowed maximum equal to
the vocabulary size, so ``prompt_logprobs=vocab_size`` (or any large value)
allocates the identical full-vocabulary tensor that ``-1`` would have.
"""

import os

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


# ---------------------------------------------------------------------------
# Original five tests (preserved, adapted to the cap-based approach)
# ---------------------------------------------------------------------------


def test_prompt_logprobs_minus_one_rejected():
    """B9: prompt_logprobs=-1 must be rejected regardless of max_logprobs.

    With the cap-based approach, -1 resolves to vocab_size (154,880 for
    GLM-5.2) which exceeds the default cap of 20.
    """
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=-1)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


def test_prompt_logprobs_positive_still_works():
    """A concrete prompt_logprobs value within the cap should pass."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=20)
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_zero_still_works():
    """prompt_logprobs=0 (disabled) should not trigger the rejection."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=0)
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_none_still_works():
    """prompt_logprobs=None (unset) should not trigger the rejection."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams()
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_exceeds_max_still_rejected():
    """prompt_logprobs > max_logprobs is still rejected by the existing
    check (confirms the existing behavior is preserved)."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=20)
    sp = SamplingParams(prompt_logprobs=100)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


# ---------------------------------------------------------------------------
# New tests: resource-bound bypass coverage (B9)
# ---------------------------------------------------------------------------


def test_prompt_logprobs_vocab_size_rejected():
    """B9: prompt_logprobs=vocab_size must be rejected even when
    max_logprobs=-1 (the bypass the reviewers found)."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=154880)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


def test_prompt_logprobs_vocab_size_minus_one_rejected():
    """B9: prompt_logprobs=vocab_size-1 must also be rejected; values just
    below vocab_size are equally expensive."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=154879)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


def test_prompt_logprobs_at_cap_accepted():
    """prompt_logprobs equal to the cap (VLLM_MAX_PROMPT_LOGPROBS, default 20)
    must be accepted."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=20)
    sp._validate_logprobs(model_config)


def test_prompt_logprobs_above_cap_rejected():
    """prompt_logprobs=cap+1 must be rejected."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=21)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


def test_prompt_logprobs_bypass_with_max_logprobs_minus_one():
    """B9: the original bypass — max_logprobs=-1 makes the allowed maximum
    vocab_size, so large positive values pass the > check. The cap must
    still reject them."""
    from vllm.exceptions import VLLMValidationError

    # Simulate the exact bypass scenario: max_logprobs=-1 (unlimited)
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=154880)

    with pytest.raises(VLLMValidationError, match="VLLM_MAX_PROMPT_LOGPROBS"):
        sp._validate_logprobs(model_config)


# ---------------------------------------------------------------------------
# C5: same bound applied to sampling logprobs
# ---------------------------------------------------------------------------


def test_sample_logprobs_vocab_size_rejected():
    """C5: logprobs=vocab_size (sampling, not prompt) must also be capped."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(logprobs=154880)

    with pytest.raises(VLLMValidationError, match="VLLM_MAX_LOGPROBS"):
        sp._validate_logprobs(model_config)


def test_sample_logprobs_minus_one_rejected():
    """C5: logprobs=-1 resolves to vocab_size and must be capped."""
    from vllm.exceptions import VLLMValidationError

    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(logprobs=-1)

    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp._validate_logprobs(model_config)


def test_sample_logprobs_at_cap_accepted():
    """C5: logprobs equal to the cap must be accepted."""
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(logprobs=20)
    sp._validate_logprobs(model_config)


# ---------------------------------------------------------------------------
# Custom cap via env var
# ---------------------------------------------------------------------------


def test_prompt_logprobs_custom_cap(monkeypatch):
    """An operator who raises VLLM_MAX_PROMPT_LOGPROBS should be able to
    request more than the default 20."""
    monkeypatch.setenv("VLLM_MAX_PROMPT_LOGPROBS", "100")
    # Need to re-import to pick up the new env var value — envs.py caches
    # via lambda, so the value is read fresh each time.
    model_config = _mock_model_config(vocab_size=154880, max_logprobs=-1)
    sp = SamplingParams(prompt_logprobs=100)
    sp._validate_logprobs(model_config)

    # But 101 should still fail
    from vllm.exceptions import VLLMValidationError

    sp2 = SamplingParams(prompt_logprobs=101)
    with pytest.raises(VLLMValidationError, match="greater than max allowed"):
        sp2._validate_logprobs(model_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
