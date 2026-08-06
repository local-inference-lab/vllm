# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput


def test_async_output_returns_draft_ids_on_same_event() -> None:
    output = object.__new__(AsyncOutput)
    output.copy_event_recorded = True
    output.copy_event = SimpleNamespace(synchronize=lambda: None)
    output.sampled_token_ids = np.array([[7, -1]], dtype=np.int32)
    output.num_sampled_tokens_np = np.array([1], dtype=np.int32)
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output.model_runner_output = ModelRunnerOutput(
        req_ids=["req-0"], req_id_to_index={"req-0": 0}
    )
    output.draft_req_ids = ["req-0"]
    output.draft_token_ids_np = np.array([[11, 12, -1]], dtype=np.int32)

    model_output = output.get_output()

    assert model_output.sampled_token_ids == [[7]]
    assert model_output.draft_token_ids is not None
    assert model_output.draft_token_ids.req_ids == ["req-0"]
    assert model_output.draft_token_ids.draft_token_ids == [[11, 12]]
