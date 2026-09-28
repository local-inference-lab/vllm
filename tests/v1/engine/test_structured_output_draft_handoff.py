# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured output with async speculative decoding must stay constrained
when a request skips a step.

The V2 model runner verifies the drafts it holds itself, while the scheduler
builds the grammar bitmask from -1 placeholders that deferred sampling
back-fills. A request that skipped the previous step (here a compute-share
prefill turn) used to be scheduled with placeholders nobody back-filled, so its
bitmask rows past the first draft allowed every token and the runner accepted
tokens the grammar rejected (issue local-inference-lab/vllm#726).
"""

import itertools
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

from ..core.utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

NUM_SPEC_TOKENS = 3
VOCAB_SIZE = 128
# The only output the grammar accepts, one token per position.
GRAMMAR_TOKENS = list(range(10, 34))
# Token the model prefers at odd positions; the grammar never allows it.
OFF_GRAMMAR_TOKEN = 99


def _preferred_token(position: int) -> int:
    if position % 2 or position >= len(GRAMMAR_TOKENS):
        return OFF_GRAMMAR_TOKEN
    return GRAMMAR_TOKENS[position]


@dataclass
class SequenceGrammar(StructuredOutputGrammar):
    """Accepts exactly GRAMMAR_TOKENS."""

    position: int = 0

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        for token in tokens:
            if self.is_terminated() or token != GRAMMAR_TOKENS[self.position]:
                return False
            self.position += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        valid = list(
            itertools.takewhile(
                lambda item: item[1] == GRAMMAR_TOKENS[self.position + item[0]],
                enumerate(tokens[: len(GRAMMAR_TOKENS) - self.position]),
            )
        )
        return [token for _, token in valid]

    def rollback(self, num_tokens: int) -> None:
        self.position -= num_tokens

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].zero_()
        if not self.is_terminated():
            token = GRAMMAR_TOKENS[self.position]
            # Bit 31 is the int32 sign bit.
            bitmask[batch_index, token // 32] = int(
                np.array(1 << (token % 32), dtype=np.uint32).view(np.int32)
            )

    def is_terminated(self) -> bool:
        return self.position >= len(GRAMMAR_TOKENS)

    def reset(self):
        self.position = 0


class BitmaskBackend:
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full((max_num_seqs, VOCAB_SIZE // 32), -1, dtype=torch.int32)

    def destroy(self):
        pass


def _allowed(row: np.ndarray, token: int) -> bool:
    return bool((int(row[token // 32]) >> (token % 32)) & 1)


@dataclass
class _RunnerRequest:
    prefill_len: int
    num_computed: int
    num_output: int = 0
    # Persistent draft slot, zeroed on admission like RequestState.draft_tokens.
    drafts: list[int] = field(default_factory=lambda: [0] * NUM_SPEC_TOKENS)


class FakeV2Executor:
    """Emulates the V2 runner's draft ownership with a greedy model.

    The model prefers ``_preferred_token``; grammar rows constrain it. Drafts
    are verified from the runner's own slot, never from the scheduler's
    placeholders, and invalidated per ``num_invalid_spec_tokens`` exactly as
    ``_build_grammar_mapping`` does. ``consumed_handoff`` selects what
    ``take_draft_token_ids`` returns: the drafts the latest step consumed, or
    (the previous contract) the drafts the latest sampled step proposed.
    """

    def __init__(self, consumed_handoff: bool):
        self.consumed_handoff = consumed_handoff
        self.requests: dict[str, _RunnerRequest] = {}
        self.pending: tuple[SchedulerOutput, dict[str, int]] | None = None
        self.consumed = DraftTokenIds([], [])
        self.proposed = DraftTokenIds([], [])
        # Per sampled step: request id -> number of accepted drafts.
        self.accepted_per_step: list[dict[str, int]] = []

    @staticmethod
    def _done(value) -> Future:
        future: Future = Future()
        future.set_result(value)
        return future

    def execute_model(self, scheduler_output: SchedulerOutput, non_block=False):
        for req_id in itertools.chain(
            scheduler_output.finished_req_ids, scheduler_output.preempted_req_ids or ()
        ):
            self.requests.pop(req_id, None)
        for new_req in scheduler_output.scheduled_new_reqs:
            assert new_req.prefill_token_ids is not None
            self.requests[new_req.req_id] = _RunnerRequest(
                len(new_req.prefill_token_ids), new_req.num_computed_tokens
            )
        num_drafts = {
            req_id: len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            for req_id in scheduler_output.num_scheduled_tokens
        }
        drafting = [req_id for req_id, n in num_drafts.items() if n]
        self.consumed = DraftTokenIds(
            drafting,
            [self.requests[req_id].drafts[: num_drafts[req_id]] for req_id in drafting],
        )
        if scheduler_output.total_num_scheduled_tokens == 0:
            return self._done(ModelRunnerOutput(req_ids=[], req_id_to_index={}))
        self.pending = (scheduler_output, num_drafts)
        return self._done(None)

    def sample_tokens(self, grammar_output: GrammarOutput | None, non_block=False):
        assert self.pending is not None
        scheduler_output, num_drafts = self.pending
        self.pending = None
        rows: dict[tuple[str, int], np.ndarray] = {}
        num_invalid: dict[str, int] = {}
        if grammar_output is not None:
            row_iter = iter(grammar_output.grammar_bitmask)
            for req_id in grammar_output.structured_output_request_ids:
                for position in range(num_drafts[req_id] + 1):
                    rows[req_id, position] = next(row_iter)
            num_invalid = grammar_output.num_invalid_spec_tokens or {}

        req_ids = list(scheduler_output.num_scheduled_tokens)
        sampled: list[list[int]] = []
        accepted: dict[str, int] = {}
        self.accepted_per_step.append(accepted)
        for req_id in req_ids:
            state = self.requests[req_id]
            end = state.num_computed + scheduler_output.num_scheduled_tokens[req_id]
            if end < state.prefill_len:
                state.num_computed = end
                sampled.append([])
                continue
            n = num_drafts[req_id]
            n_valid = n - num_invalid.get(req_id, 0)
            drafts = state.drafts[:n_valid] + [-1] * (n - n_valid)
            tokens: list[int] = []
            for position in range(n + 1):
                target = _preferred_token(state.num_output + position)
                row = rows.get((req_id, position))
                if row is not None and not _allowed(row, target):
                    target = next(t for t in range(VOCAB_SIZE) if _allowed(row, t))
                tokens.append(target)
                if position == n or drafts[position] != target:
                    break
            accepted[req_id] = len(tokens) - 1
            state.num_computed = end - (n + 1 - len(tokens))
            state.num_output += len(tokens)
            state.drafts = [
                _preferred_token(state.num_output + i) for i in range(NUM_SPEC_TOKENS)
            ]
            sampled.append(tokens)
        self.proposed = DraftTokenIds(
            req_ids,
            [
                self.requests[req_id].drafts
                if scheduler_output.has_structured_output_requests
                else [-1] * NUM_SPEC_TOKENS
                for req_id in req_ids
            ],
        )
        return self._done(
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
        )

    def take_draft_token_ids(self) -> DraftTokenIds:
        return self.consumed if self.consumed_handoff else self.proposed


def _engine(scheduler, executor) -> EngineCore:
    """The real step_with_batch_queue loop around a fake executor."""
    engine = object.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = executor
    engine.batch_queue_size = 2
    engine.batch_queue = deque(maxlen=2)
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.check_for_draft_tokens = True
    engine._last_model_completion_time = None
    engine._should_throttle_prefills = Mock(return_value=False)
    engine.log_error_detail = lambda _: nullcontext()
    engine.capture_iteration_details = lambda _: nullcontext()
    engine._wait_for_boundary_checkpoint_copies = Mock()
    engine._process_aborts_queue = Mock()
    engine._attach_iteration_details = Mock()
    return engine


@pytest.fixture
def opt_model_path(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt",'
        '"max_position_embeddings":32768}'
    )
    return str(tmp_path)


def _structured_request() -> Request:
    params = SamplingParams(
        max_tokens=len(GRAMMAR_TOKENS),
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="structured",
        prompt_token_ids=list(range(8)),
        sampling_params=params,
        pooling_params=None,
    )
    assert request.structured_output_request is not None
    request.structured_output_request.grammar = SequenceGrammar()
    return request


@pytest.mark.parametrize("consumed_handoff", [True, False])
def test_skipped_step_keeps_draft_rows_constrained(opt_model_path, consumed_handoff):
    """consumed_handoff=False is a runner that still hands back the drafts the
    last sampled step proposed: the scheduler must then reject the unresolved
    drafts rather than let the runner accept them against open rows."""
    scheduler = create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        async_scheduling=True,
        use_v2_model_runner=True,
        num_speculative_tokens=NUM_SPEC_TOKENS,
        speculative_method="ngram_gpu",
        prefill_compute_share=0.5,
        max_num_batched_tokens=32,
        max_model_len=512,
    )
    scheduler.structured_output_manager.backend = BitmaskBackend()
    executor = FakeV2Executor(consumed_handoff)
    engine = _engine(scheduler, executor)

    request = _structured_request()
    scheduler.add_request(request)
    # A long plain prompt contends with the structured decode, so prefill
    # turns leave the decode out of some steps.
    (prefill,) = create_requests(
        num_requests=1, num_tokens=320, max_tokens=2, req_ids=["prefill"]
    )

    clock = itertools.count(step=0.001)
    with patch("vllm.v1.engine.core.time.perf_counter", lambda: next(clock)):
        for step in range(200):
            if step == 2:
                scheduler.add_request(prefill)
            if not (scheduler.has_requests() or engine.batch_queue):
                break
            engine.step_with_batch_queue()

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert list(request.output_token_ids) == GRAMMAR_TOKENS

    steps = [i for i, a in enumerate(executor.accepted_per_step) if "structured" in a]
    after_skip = [b for a, b in itertools.pairwise(steps) if b - a > 1]
    assert after_skip, "the prefill turns never skipped the structured decode"
    accepted_after_skip = sum(
        executor.accepted_per_step[i]["structured"] for i in after_skip
    )
    if consumed_handoff:
        # The skipped request's drafts reach the grammar, so it keeps
        # speculating instead of verifying one token per step.
        assert accepted_after_skip > 0
    else:
        assert accepted_after_skip == 0


def test_deferred_step_without_inflight_batch_samples_immediately():
    """A deferred step may have no earlier batch to wait for: a request that
    skipped a step has its output processed already, and only its drafts still
    need the back-fill."""
    done: Future = Future()
    done.set_result(None)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 4
    scheduler_output.pending_structured_output_tokens = True
    scheduler = Mock()
    scheduler.has_requests.return_value = True
    scheduler.schedule.return_value = scheduler_output
    executor = Mock()
    executor.execute_model.return_value = done
    executor.sample_tokens.return_value = done
    engine = _engine(scheduler, executor)

    assert engine.step_with_batch_queue() == (None, True)

    draft_token_ids = executor.take_draft_token_ids.return_value
    scheduler.update_draft_token_ids_in_output.assert_called_once_with(
        draft_token_ids, scheduler_output
    )
    scheduler.get_grammar_bitmask.assert_called_once_with(scheduler_output)
    executor.sample_tokens.assert_called_once_with(
        scheduler.get_grammar_bitmask.return_value, non_block=True
    )
    assert [entry[1] for entry in engine.batch_queue] == [scheduler_output]
