# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import (
    _mtp_draft_decode_only,
    _mtp_draft_uses_cudagraph,
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


def test_mtp_draft_decode_phase_uses_active_prompt_state():
    common = object.__new__(CommonAttentionMetadata)
    common.num_reqs = 2
    common.is_prefilling = torch.tensor([False, False, True])
    assert _mtp_draft_decode_only(common)

    common.is_prefilling[1] = True
    assert not _mtp_draft_decode_only(common)
    common.is_prefilling = None

    assert not _mtp_draft_decode_only(common)
    assert _mtp_draft_decode_only(common, CUDAGraphMode.PIECEWISE)


def test_mtp_draft_cudagraphs_exclude_prompt_batches():
    common = object.__new__(CommonAttentionMetadata)
    common.num_reqs = 2
    common.is_prefilling = torch.tensor([False, False])
    assert _mtp_draft_uses_cudagraph(common)

    common.is_prefilling[1] = True
    assert not _mtp_draft_uses_cudagraph(common)

    common.is_prefilling = None
    assert _mtp_draft_uses_cudagraph(common)


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
