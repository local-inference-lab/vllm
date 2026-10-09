# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState


@pytest.mark.parametrize("fitted_limit", [969_728, 216_576, 4096])
def test_capture_uses_auto_fitted_context_limit(monkeypatch, fitted_limit):
    """Capture bounds must fit attention buffers allocated after KV auto-fit."""
    model_config = SimpleNamespace(
        max_model_len=1_048_576,
        dtype=torch.bfloat16,
        get_inputs_embeds_size=lambda: 4096,
    )
    config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=4096),
    )
    state = object.__new__(DefaultModelState)
    ModelState.__init__(state, config, None, None, torch.device("cpu"))
    assert state.max_model_len == 1_048_576
    # Worker.update_max_model_len updates the shared config before builders
    # allocate context-dependent buffers and graphs are captured.
    model_config.max_model_len = fitted_limit
    captured = {}

    def build_metadata(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.default.build_attn_metadata", build_metadata
    )
    batch = InputBatch.make_dummy(1, 6, InputBuffers(8, 48, torch.device("cpu")))
    state.prepare_attn(batch, CUDAGraphMode.NONE, [], {}, [], None, for_capture=True)
    assert captured["max_seq_len"] == fitted_limit
    assert state.max_model_len == fitted_limit

    state.prepare_attn(batch, CUDAGraphMode.NONE, [], {}, [], None, for_capture=False)
    assert captured["max_seq_len"] == 6
