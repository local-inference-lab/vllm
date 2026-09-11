# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    compute_probs_and_sample_next_token,
)

DEVICE_TYPE = current_platform.device_type


def _seed_default_generator(seed: int) -> None:
    set_random_seed(seed)


def _make_sampling_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32, device=DEVICE_TYPE),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=DEVICE_TYPE),
        presence_penalties=torch.empty(0, device=DEVICE_TYPE),
        repetition_penalties=torch.empty(0, device=DEVICE_TYPE),
        output_token_ids=[[] for _ in range(batch_size)],
        spec_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_compute_probs_and_sample_next_token_uses_fp64_exponential_race():
    batch_size = 4
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(11)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = _make_sampling_metadata(batch_size)

    _seed_default_generator(12345)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty(probs.shape, dtype=torch.float64, device=probs.device)
    q.exponential_()
    expected_ids = q.reciprocal_().mul_(probs).argmax(dim=-1).view(-1)

    _seed_default_generator(12345)
    actual_ids, actual_probs = compute_probs_and_sample_next_token(
        logits.clone(),
        metadata,
        use_fp64_gumbel=True,
    )

    assert torch.equal(actual_ids, expected_ids)
    assert torch.allclose(actual_probs, probs)


def test_compute_probs_and_sample_next_token_returns_constrained_distribution():
    batch_size = 3
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(29)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = dataclasses.replace(
        _make_sampling_metadata(batch_size),
        temperature=torch.tensor([0.7, 1.0, 1.3], device=DEVICE_TYPE),
        top_k=torch.tensor([5, 9, 17], dtype=torch.int32, device=DEVICE_TYPE),
        top_p=torch.tensor([0.6, 0.8, 0.95], device=DEVICE_TYPE),
    )

    expanded_logits = logits / metadata.temperature.unsqueeze(-1)
    expected_probs = apply_top_k_top_p(
        expanded_logits,
        metadata.top_k,
        metadata.top_p,
    ).softmax(dim=-1, dtype=torch.float32)

    token_ids, actual_probs = compute_probs_and_sample_next_token(
        logits.clone(),
        metadata,
    )

    assert torch.equal(actual_probs != 0, expected_probs != 0)
    assert torch.allclose(actual_probs, expected_probs)
    assert torch.all(actual_probs.gather(1, token_ids.unsqueeze(-1)) > 0)


def test_parallel_draft_sampling_repeats_request_constraints_in_request_order():
    batch_size = 2
    positions_per_request = 3
    vocab_size = 32
    generator = torch.Generator(device=DEVICE_TYPE).manual_seed(31)
    logits = torch.randn(
        batch_size * positions_per_request,
        vocab_size,
        dtype=torch.float32,
        device=DEVICE_TYPE,
        generator=generator,
    )
    metadata = dataclasses.replace(
        _make_sampling_metadata(batch_size),
        temperature=torch.tensor([0.7, 1.3], device=DEVICE_TYPE),
        top_k=torch.tensor([5, 11], dtype=torch.int32, device=DEVICE_TYPE),
        top_p=torch.tensor([0.65, 0.9], device=DEVICE_TYPE),
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._enable_probabilistic_draft_probs = True
    proposer.use_fp64_gumbel = False

    token_ids, actual_probs = proposer._sample_from_logits(logits.clone(), metadata)

    temperature = metadata.temperature.repeat_interleave(positions_per_request)
    top_k = metadata.top_k.repeat_interleave(positions_per_request)
    top_p = metadata.top_p.repeat_interleave(positions_per_request)
    expected_probs = apply_top_k_top_p(
        logits / temperature.unsqueeze(-1),
        top_k,
        top_p,
    ).softmax(dim=-1, dtype=torch.float32)
    assert torch.equal(actual_probs != 0, expected_probs != 0)
    assert torch.allclose(actual_probs, expected_probs)
    assert torch.all(actual_probs.gather(1, token_ids.unsqueeze(-1)) > 0)


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("DeepSeekMTPModel", True),
        ("Glm5NextMTPModel", True),
        ("KimiK3MTPModel", True),
        ("MiniMaxM3ForCausalLM", False),
    ],
)
def test_mtp_model_returns_tuple(architecture: str, expected: bool):
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture])
    )

    assert proposer.model_returns_tuple() is expected
