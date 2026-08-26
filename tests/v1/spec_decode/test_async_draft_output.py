# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput


def _make_async_output() -> AsyncOutput:
    output = object.__new__(AsyncOutput)
    output.copy_event_recorded = True
    output.copy_event = SimpleNamespace(synchronize=lambda: None)
    output.sampled_token_ids = np.array([[7, -1]], dtype=np.int32)
    output.num_sampled_tokens_np = np.array([1], dtype=np.int32)
    output.num_nans = None
    output.logprobs_tensors = None
    output.routed_experts_cpu = None
    output._has_fault = None
    output.prompt_logprobs_dict = {}
    output.model_runner_output = ModelRunnerOutput(
        req_ids=["req-0"], req_id_to_index={"req-0": 0}
    )
    output.draft_req_ids = None
    output.draft_token_ids_np = None
    return output


def test_async_output_returns_draft_ids_on_same_event() -> None:
    output = _make_async_output()
    output.draft_req_ids = ["req-0"]
    output.draft_token_ids_np = np.array([[11, 12, -1]], dtype=np.int32)

    model_output = output.get_output()

    assert model_output.sampled_token_ids == [[7]]
    assert model_output.draft_token_ids is not None
    assert model_output.draft_token_ids.req_ids == ["req-0"]
    assert model_output.draft_token_ids.draft_token_ids == [[11, 12]]


def test_async_output_returns_counters_mutated_after_construction() -> None:
    output = _make_async_output()
    output.model_runner_output.dspark_suppressed_batches = 1
    output.model_runner_output.dspark_suppressed_rows = 2

    model_output = output.get_output()

    assert model_output.dspark_suppressed_batches == 1
    assert model_output.dspark_suppressed_rows == 2
