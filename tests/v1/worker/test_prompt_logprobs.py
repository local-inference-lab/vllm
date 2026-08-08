# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import prompt_logprob
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    PromptLogprobsWorker,
    compute_prompt_logprobs_with_chunking,
)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_prompt_logprobs_worker_rejects_invalid_chunk_size(
    monkeypatch: pytest.MonkeyPatch, chunk_size: int
):
    monkeypatch.setenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", str(chunk_size))

    with pytest.raises(ValueError, match="must be greater than zero"):
        PromptLogprobsWorker(max_num_reqs=1)


def test_prompt_logprobs_chunk_size_bounds_logits_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk_rows: list[int] = []

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        chunk_rows.append(hidden_states.shape[0])
        return torch.zeros((hidden_states.shape[0], 8))

    def fake_compute_topk_logprobs(
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
    ) -> SimpleNamespace:
        del num_logprobs
        return SimpleNamespace(
            logprob_token_ids=sampled_token_ids.unsqueeze(-1),
            logprobs=torch.zeros((logits.shape[0], 1)),
            selected_token_ranks=torch.ones(logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_topk_logprobs", fake_compute_topk_logprobs
    )
    prompt_token_ids = torch.arange(5)
    prompt_hidden_states = torch.zeros((5, 4))

    token_ids, logprobs, ranks = compute_prompt_logprobs_with_chunking(
        prompt_token_ids,
        prompt_hidden_states,
        logits_fn,
        num_prompt_logprobs=1,
        chunk_size=2,
    )

    assert chunk_rows == [2, 2, 1]
    torch.testing.assert_close(token_ids.squeeze(-1), prompt_token_ids)
    assert logprobs.shape == (5, 1)
    torch.testing.assert_close(ranks, torch.ones(5, dtype=torch.int64))


def test_prompt_logprobs_profile_uses_full_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", "2")
    worker = PromptLogprobsWorker(max_num_reqs=1)
    helper = Mock()
    monkeypatch.setattr(prompt_logprob, "compute_prompt_logprobs_with_chunking", helper)
    hidden_states = torch.zeros((5, 4))
    logits_fn = Mock()

    worker.profile_run(logits_fn, hidden_states, max_num_logprobs=-1)

    prompt_token_ids, profiled_hidden_states, fn, max_logprobs, chunk_size = (
        helper.call_args.args
    )
    assert prompt_token_ids.shape == (5,)
    assert profiled_hidden_states is hidden_states
    assert fn is logits_fn
    assert max_logprobs == -1
    assert chunk_size == 2


def test_prompt_logprobs_accumulates_chunks_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = PromptLogprobsWorker(max_num_reqs=1)
    worker.uses_prompt_logprobs[0] = True
    worker.num_prompt_logprobs[0] = 1
    worker.in_progress_prompt_logprobs["req"] = None

    monkeypatch.setattr(
        prompt_logprob,
        "get_prompt_logprobs_token_ids",
        lambda *args, **kwargs: torch.arange(args[0]),
    )
    chunks = iter(
        [
            (
                torch.tensor([[10, 11], [20, 21]]),
                torch.tensor([[1.0, 1.1], [2.0, 2.1]]),
                torch.tensor([1, 2]),
            ),
            (
                torch.tensor([[30, 31], [40, 41], [50, 51]]),
                torch.tensor([[3.0, 3.1], [4.0, 4.1], [5.0, 5.1]]),
                torch.tensor([3, 4, 5]),
            ),
        ]
    )
    monkeypatch.setattr(
        prompt_logprob,
        "compute_prompt_logprobs_with_chunking",
        lambda *args, **kwargs: next(chunks),
    )
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    def make_batch(computed: int, scheduled: int) -> SimpleNamespace:
        return SimpleNamespace(
            idx_mapping_np=np.array([0]),
            idx_mapping=torch.tensor([0]),
            num_computed_prefill_tokens_np=np.array([computed]),
            prefill_len_np=np.array([5]),
            num_scheduled_tokens=np.array([scheduled]),
            num_tokens=scheduled,
            query_start_loc=torch.tensor([0, scheduled]),
            query_start_loc_np=np.array([0, scheduled]),
            req_ids=["req"],
        )

    output = worker.compute_prompt_logprobs(
        Mock(),
        torch.zeros((2, 4)),
        make_batch(computed=0, scheduled=2),
        torch.zeros((1, 5), dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        np.array([5]),
    )
    assert output == {}
    in_progress = worker.in_progress_prompt_logprobs["req"]
    assert in_progress is not None
    assert in_progress.logprobs.device.type == "cpu"
    synchronize.assert_not_called()

    output = worker.compute_prompt_logprobs(
        Mock(),
        torch.zeros((3, 4)),
        make_batch(computed=2, scheduled=3),
        torch.zeros((1, 5), dtype=torch.int64),
        torch.full((1,), 2, dtype=torch.int64),
        np.array([5]),
    )
    result = output["req"]
    torch.testing.assert_close(
        result.logprob_token_ids,
        torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.logprobs,
        torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]]),
    )
    torch.testing.assert_close(
        result.selected_token_ranks, torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    )
    assert worker.in_progress_prompt_logprobs["req"] is None
    synchronize.assert_not_called()


def test_model_runner_profiles_prompt_logprobs(
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_states = torch.zeros((8, 4))
    sample_hidden_states = hidden_states[:2]
    logits_fn = Mock()
    runner = object.__new__(GPUModelRunner)
    runner.max_num_tokens = 8
    runner.is_last_pp_rank = True
    runner.pooling_runner = None
    runner.model = SimpleNamespace(compute_logits=logits_fn)
    runner.model_config = SimpleNamespace(max_logprobs=20)
    runner.prompt_logprobs_worker = Mock(chunk_size=256)
    runner._dummy_run = Mock(return_value=(hidden_states, sample_hidden_states))
    runner._dummy_sampler_run = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())

    GPUModelRunner.profile_run(runner)

    runner._dummy_run.assert_called_once_with(8, skip_attn=True, is_profile=True)
    runner._dummy_sampler_run.assert_called_once_with(sample_hidden_states)
    runner.prompt_logprobs_worker.profile_run.assert_called_once_with(
        logits_fn, hidden_states, 20
    )


def test_non_last_pp_rank_does_not_profile_prompt_logprobs(
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_states = torch.zeros((8, 4))
    runner = object.__new__(GPUModelRunner)
    runner.max_num_tokens = 8
    runner.is_last_pp_rank = False
    runner.pooling_runner = None
    runner.prompt_logprobs_worker = Mock()
    runner._dummy_run = Mock(return_value=(hidden_states, None))
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())

    GPUModelRunner.profile_run(runner)

    runner.prompt_logprobs_worker.profile_run.assert_not_called()
