# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend and engine-reasoner checks that are not manager-flow tests."""

from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.parser_engine import ReasoningEnd
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.structured_outputs import (
    StructuredOutputsWorker,
    grammar_invalid_drafts,
)

TOKENIZER = "gpt2"
NUM_SPEC_TOKENS = 4


def _make_manager_and_request(
    backend: Literal["xgrammar", "guidance"], prompt_str: str = '{"a": "b"}'
):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode(prompt_str)

    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
        speculative_config=SpeculativeConfig(
            model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS
        ),
    )
    manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    assert sampling_params.structured_outputs is not None
    sampling_params.structured_outputs._backend = backend
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "mtp_req",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    assert request.structured_output_request is not None
    while not request.structured_output_request._check_grammar_completion():
        continue

    return tokenizer, manager, request, prompt


def test_xgrammar_accept_tokens_stops_at_termination(capfd):
    """Tokens after a terminating EOS do not reach the matcher."""
    tokenizer, _, request, prompt = _make_manager_and_request("xgrammar")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]
    processed_before = grammar.num_processed_tokens

    assert grammar.accept_tokens(request.request_id, [eos, trailing])
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == processed_before + 1
    assert "trying to accept new token" not in capfd.readouterr().err

    processed_after_eos = grammar.num_processed_tokens
    assert grammar.accept_tokens(request.request_id, [trailing])
    assert grammar.num_processed_tokens == processed_after_eos
    assert "trying to accept new token" not in capfd.readouterr().err

    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_xgrammar_validate_tokens_stops_at_termination(capfd):
    """Validation rolls back after reaching a terminating EOS."""
    tokenizer, _, request, prompt = _make_manager_and_request("xgrammar")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]

    assert grammar.validate_tokens([eos, trailing]) == [eos]
    assert "trying to accept new token" not in capfd.readouterr().err
    assert not grammar.matcher.is_terminated()

    assert grammar.accept_tokens(request.request_id, [eos])
    assert grammar.is_terminated()

    assert grammar.validate_tokens([trailing]) == []
    assert "trying to accept new token" not in capfd.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_verified", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("num_valid", "prefix", "ending"),
    [
        (0, '{"a":"b"}', []),
        (1, '{"a":"b"}', []),
        (2, '{"a":"b"', ["}"]),
        (3, '{"a":', ["0", "}"]),
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_gpu_sampler_rejects_drafts_after_grammar_termination(
    num_verified, num_valid, prefix, ending, temperature
):
    """Deferred grammar validation must reach the device rejection sampler."""
    tokenizer, manager, request, prompt = _make_manager_and_request("xgrammar", prefix)
    grammar = request.structured_output_request.grammar
    assert grammar.accept_tokens(request.request_id, prompt)
    eos = tokenizer.eos_token_id
    space = tokenizer.encode(" ")[0]
    drafts = [tokenizer.encode(token)[0] for token in ending]
    if num_valid:
        drafts.append(eos)
    drafts += [space] * (3 - num_valid)
    scheduled = SimpleNamespace(
        has_structured_output_requests=True,
        num_scheduled_tokens={request.request_id: 4},
        scheduled_spec_decode_tokens={request.request_id: [-1] * 3},
    )
    scheduler = SimpleNamespace(
        requests={request.request_id: request}, structured_output_manager=manager
    )
    Scheduler.update_draft_token_ids_in_output(
        scheduler, DraftTokenIds([request.request_id], [drafts.copy()]), scheduled
    )
    grammar_output = Scheduler.get_grammar_bitmask(scheduler, scheduled)
    assert grammar_output is not None

    # The adjacent unstructured request and non-logit input slots must survive
    # compaction and grammar rejection unchanged.
    num_logits = num_verified + 2
    input_ids = torch.tensor([17, 19, 23, 29, *drafts], device="cuda")
    logits_indices = torch.tensor([1, *range(3, 4 + num_verified)], device="cuda")
    batch = SimpleNamespace(
        req_ids=["unstructured", request.request_id],
        num_reqs=2,
        num_draft_tokens=num_verified,
        expanded_local_pos=torch.tensor([0, *range(num_verified + 1)], device="cuda"),
        cu_num_logits_np=np.array([0, 1, num_logits], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 1, num_logits], device="cuda"),
        num_draft_tokens_per_req=np.array([0, 3], dtype=np.int32),
        input_ids=input_ids,
        logits_indices=logits_indices,
    )
    vocab_size = manager.vllm_config.model_config.get_vocab_size()
    logits = torch.full((num_logits, vocab_size), -100.0, device="cuda")
    targets = torch.tensor([space, *drafts, eos][:num_logits], device="cuda")
    logits[torch.arange(num_logits, device="cuda"), targets] = 100.0
    worker = StructuredOutputsWorker(8, vocab_size, torch.device("cuda"), 4, 1)
    worker.apply_grammar_bitmask(
        logits,
        batch,
        grammar_output.structured_output_request_ids,
        grammar_output.grammar_bitmask,
    )
    assert input_ids.tolist() == [17, 19, 23, 29, *drafts]
    invalid_drafts = grammar_invalid_drafts(
        batch,
        grammar_output.structured_output_request_ids,
        grammar_output.num_acceptable_drafts,
    )
    draft_sampled = input_ids[logits_indices]
    if invalid_drafts is not None:
        draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)
    sampled, lengths = rejection_sample(
        target_logits=logits,
        draft_logits=None,
        draft_sampled=draft_sampled,
        cu_num_logits=batch.cu_num_logits,
        pos=torch.arange(num_logits, device="cuda"),
        idx_mapping=torch.tensor([0, 1], device="cuda"),
        expanded_idx_mapping=torch.tensor(
            [0] + [1] * (num_verified + 1), device="cuda"
        ),
        expanded_local_pos=torch.tensor([0, *range(num_verified + 1)], device="cuda"),
        temperature=torch.full((2,), temperature, device="cuda"),
        seed=torch.zeros(2, device="cuda", dtype=torch.int64),
        num_speculative_steps=3,
    )
    accepted = min(num_valid, num_verified)
    assert lengths.tolist() == [1, accepted + 1]
    assert sampled[1, :accepted].tolist() == drafts[:accepted]


class _EngineReasonerStub(ParserEngineReasoningAdapter):
    """Adapter-typed reasoner with a fixed end-token set and no real engine."""

    def __init__(self, end_token_ids):
        self._end_token_ids = frozenset(end_token_ids)
        self.windows: list[list[int]] = []

    @property
    def reasoning_end_token_ids(self):
        return self._end_token_ids

    def find_reasoning_end(self, token_ids):
        self.windows.append(list(token_ids))
        for offset, token in enumerate(token_ids):
            if token in self._end_token_ids:
                return ReasoningEnd(offset, False)
        return ReasoningEnd(len(token_ids), False)

    def is_reasoning_end(self, input_ids):
        return any(token in self._end_token_ids for token in input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        raise AssertionError("engine path must not rescan draft prefixes")


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_engine_reasoner_ends_midwindow_with_padding(backend):
    """Engine reasoners see the draft window once, without -1 padding."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n")[0]
    reasoner = _EngineReasonerStub({marker})
    manager.reasoner_cls = _EngineReasonerStub
    request.structured_output_request.reasoner = reasoner
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode(",")[0]
    drafts = [pre, marker, post, -1]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    assert (bitmask[0] == -1).all()
    assert (bitmask[1] == -1).all()
    assert not (bitmask[2] == -1).all()
    assert reasoner.windows == [[pre, marker, post]]
    assert not grammar.is_terminated()
