# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as gpu_model_runner_module
from vllm.config.model import LogprobsMode
from vllm.v1.worker.gpu.input_batch import InputBuffers
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


@pytest.mark.parametrize(
    ("logprobs_mode", "uses_logits"),
    [("raw_logprobs", False), ("processed_logits", True)],
)
def test_prompt_logprobs_chunk_size_bounds_logits_rows_and_preserves_mode(
    monkeypatch: pytest.MonkeyPatch,
    logprobs_mode: LogprobsMode,
    uses_logits: bool,
):
    chunk_rows: list[int] = []
    logits_modes: list[bool] = []

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        chunk_rows.append(hidden_states.shape[0])
        return torch.zeros((hidden_states.shape[0], 8))

    def fake_compute_topk_scores(
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        *,
        logits_mode: bool,
    ) -> SimpleNamespace:
        del num_logprobs
        logits_modes.append(logits_mode)
        return SimpleNamespace(
            logprob_token_ids=sampled_token_ids.unsqueeze(-1),
            logprobs=torch.zeros((logits.shape[0], 1)),
            selected_token_ranks=torch.ones(logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_compute_topk_scores)
    prompt_token_ids = torch.arange(5)
    prompt_hidden_states = torch.zeros((5, 4))

    token_ids, logprobs, ranks = compute_prompt_logprobs_with_chunking(
        prompt_token_ids,
        prompt_hidden_states,
        logits_fn,
        num_prompt_logprobs=1,
        logprobs_mode=logprobs_mode,
        chunk_size=2,
    )

    assert chunk_rows == [2, 2, 1]
    assert logits_modes == [uses_logits] * 3
    torch.testing.assert_close(token_ids.squeeze(-1), prompt_token_ids)
    assert logprobs.shape == (5, 1)
    torch.testing.assert_close(ranks, torch.ones(5, dtype=torch.int64))


def test_prompt_logprobs_profile_uses_full_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PROMPT_LOGPROBS_CHUNK_SIZE", "2")
    worker = PromptLogprobsWorker(max_num_reqs=1, logprobs_mode="processed_logits")
    helper = Mock()
    monkeypatch.setattr(prompt_logprob, "compute_prompt_logprobs_with_chunking", helper)
    hidden_states = torch.zeros((5, 4))
    logits_fn = Mock()

    worker.profile_run(logits_fn, hidden_states, max_num_logprobs=-1)

    prompt_token_ids, profiled_hidden_states, fn, max_logprobs = helper.call_args.args
    assert prompt_token_ids.shape == (5,)
    assert profiled_hidden_states is hidden_states
    assert fn is logits_fn
    assert max_logprobs == -1
    assert helper.call_args.kwargs == {
        "logprobs_mode": "processed_logits",
        "chunk_size": 2,
    }


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
    runner.supports_mm_inputs = False
    runner.is_encoder_only = False
    runner.is_last_pp_rank = True
    runner.pooling_runner = None
    runner.model = SimpleNamespace(compute_logits=logits_fn)
    runner.model_config = SimpleNamespace(max_logprobs=20)
    runner.prompt_logprobs_worker = Mock(chunk_size=256)
    runner._dummy_run = Mock(return_value=(hidden_states, sample_hidden_states))
    runner._dummy_sampler_run = Mock()
    runner.reset_encoder_cache = Mock()
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
    runner.supports_mm_inputs = False
    runner.is_encoder_only = False
    runner.is_last_pp_rank = False
    runner.pooling_runner = None
    runner.prompt_logprobs_worker = Mock()
    runner._dummy_run = Mock(return_value=(hidden_states, None))
    runner.reset_encoder_cache = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())

    GPUModelRunner.profile_run(runner)

    runner.prompt_logprobs_worker.profile_run.assert_not_called()


@pytest.mark.parametrize("dummy_run_fails", [False, True])
def test_v2_single_request_prefill_profile_uses_attention_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    dummy_run_fails: bool,
):
    hidden_states = torch.zeros((8, 4))
    sample_hidden_states = hidden_states[-1:]
    runner = object.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner.max_num_tokens = 8192
    runner.is_last_pp_rank = True
    runner.pooling_runner = None
    runner._init_minimal_kv_cache_for_profiling = Mock()
    runner._dummy_sampler_run = Mock()
    runner.reset_encoder_cache = Mock()
    runner._cleanup_cudagraph_memory_profile = Mock()

    if dummy_run_fails:
        runner._dummy_run = Mock(side_effect=RuntimeError("expected prefill failure"))
    else:
        runner._dummy_run = Mock(return_value=(hidden_states, sample_hidden_states))

    monkeypatch.setattr(
        gpu_model_runner_module,
        "set_current_vllm_config",
        lambda _: nullcontext(),
    )
    synchronize = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)

    if dummy_run_fails:
        with pytest.raises(RuntimeError, match="expected prefill failure"):
            runner.profile_single_request_prefill()
    else:
        runner.profile_single_request_prefill()

    runner._init_minimal_kv_cache_for_profiling.assert_called_once_with()
    runner._dummy_run.assert_called_once_with(
        8192,
        skip_eplb=True,
        is_profile=True,
        single_request_prefill=True,
    )
    runner.reset_encoder_cache.assert_called_once_with()
    runner._cleanup_cudagraph_memory_profile.assert_called_once_with()
    if dummy_run_fails:
        synchronize.assert_not_called()
        runner._dummy_sampler_run.assert_not_called()
    else:
        synchronize.assert_called_once_with()
        runner._dummy_sampler_run.assert_called_once_with(sample_hidden_states)


def test_v2_dummy_sampler_profiles_configured_speculative_width():
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_buffers = InputBuffers(8, 48, runner.device)
    runner.num_speculative_steps = 5
    runner.decode_query_len = 6
    runner.model = SimpleNamespace(
        compute_logits=lambda hidden_states: torch.zeros(
            hidden_states.shape[0], 16, dtype=hidden_states.dtype
        )
    )
    runner.sampler = Mock()
    runner.rejection_sampler = Mock()
    draft_logits = torch.zeros(8, 5, 16)
    runner.speculator = SimpleNamespace(draft_logits=draft_logits)

    runner._dummy_sampler_run(torch.zeros(3, 8))

    runner.sampler.assert_called_once()
    verify_logits, input_batch, passed_draft_logits = (
        runner.rejection_sampler.call_args.args
    )
    assert verify_logits.shape == (18, 16)
    assert input_batch.num_draft_tokens == 15
    assert input_batch.num_draft_tokens_per_req.tolist() == [5, 5, 5]
    assert input_batch.cu_num_logits_np.tolist() == [0, 6, 12, 18]
    assert input_batch.expanded_idx_mapping.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    assert input_batch.expanded_local_pos.tolist() == [0, 1, 2, 3, 4, 5] * 3
    assert passed_draft_logits is draft_logits
