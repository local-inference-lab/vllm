# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The v2 runner consumes a deferred remote proposal in stream order: the
draft tokens are stored for the proposed batch and the step's output copy is
recorded only then."""

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu import model_runner as mrv2


class _Speculator:
    def __init__(self, draft: torch.Tensor | None):
        self._draft = draft
        self.calls = 0

    def resolve_pending(self) -> torch.Tensor | None:
        self.calls += 1
        return self._draft


class _Output:
    def __init__(self) -> None:
        self.recorded: list[tuple[list[str], torch.Tensor]] = []

    def add_draft_token_ids(self, req_ids, draft_tokens):
        self.recorded.append((list(req_ids), draft_tokens.clone()))


def _runner(draft, *, num_draft_tokens, output):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.speculator = _Speculator(draft)
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.full((6, 3), -1, dtype=torch.int64)
    )
    runner._pending_draft = mrv2._PendingDraft(
        idx_mapping=torch.tensor([4, 1]),
        req_ids=["a", "b"],
        num_draft_tokens=num_draft_tokens,
        async_output=output,
    )
    return runner


def test_resolve_stores_the_draft_for_the_proposed_slots_and_records_the_copy():
    output = _Output()
    draft = torch.tensor([[7, 8, 9], [3, 4, 5]])
    runner = _runner(draft, num_draft_tokens=2, output=output)

    mrv2.GPUModelRunner._resolve_pending_draft(runner)

    assert runner.speculator.calls == 1
    assert runner._pending_draft is None
    assert runner.req_states.draft_tokens[4].tolist() == [7, 8, -1]
    assert runner.req_states.draft_tokens[1].tolist() == [3, 4, -1]
    assert runner.req_states.draft_tokens[0].tolist() == [-1, -1, -1]
    ((req_ids, copied),) = output.recorded
    assert req_ids == ["a", "b"]
    assert copied.tolist() == [[7, 8], [3, 4]]


def test_resolve_without_draft_tokens_still_records_an_empty_copy():
    output = _Output()
    runner = _runner(None, num_draft_tokens=0, output=output)

    mrv2.GPUModelRunner._resolve_pending_draft(runner)

    ((req_ids, copied),) = output.recorded
    assert req_ids == ["a", "b"]
    assert copied.shape == (2, 0)
    assert torch.all(runner.req_states.draft_tokens == -1)


def test_resolve_is_a_no_op_without_a_pending_proposal():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner._pending_draft = None
    runner.speculator = _Speculator(torch.zeros(1, 1))

    mrv2.GPUModelRunner._resolve_pending_draft(runner)

    assert runner.speculator.calls == 0


def test_late_input_ids_only_without_pcp_and_capacity_manager():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.pcp_manager = None
    runner.verification_capacity_manager = None
    assert mrv2.GPUModelRunner._late_input_ids.fget(runner)
    runner.verification_capacity_manager = object()
    assert not mrv2.GPUModelRunner._late_input_ids.fget(runner)
