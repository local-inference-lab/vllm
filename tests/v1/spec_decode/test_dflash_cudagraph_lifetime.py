# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _make_speculator() -> SimpleNamespace:
    hidden_states = torch.randn(2, 8)
    return SimpleNamespace(
        _run_model=Mock(return_value=hidden_states),
        _captured_backbone_outputs=[],
        num_speculative_steps=2,
        sample_indices=torch.tensor([0, 1]),
        sample_pos=torch.tensor([1, 2]),
        sample_idx_mapping=torch.tensor([0, 0]),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        sample_col=torch.tensor([0, 1]),
        draft_logits=None,
        sample_draft=Mock(return_value=torch.tensor([11, 12])),
        draft_tokens=torch.zeros(1, 2, dtype=torch.int64),
    )


def test_dflash_retains_backbone_output_during_cudagraph_capture(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    DFlashSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert len(speculator._captured_backbone_outputs) == 1
    assert (
        speculator._captured_backbone_outputs[0] is speculator._run_model.return_value
    )


def test_dflash_does_not_retain_eager_backbone_output(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DFlashSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert speculator._captured_backbone_outputs == []
