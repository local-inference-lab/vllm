# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_probabilistic_draft_uses_disjoint_philox_offsets(monkeypatch):
    captured_positions = None

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        nonlocal captured_positions
        captured_positions = positions
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(spec_module, "gumbel_sample", fake_gumbel_sample)
    speculator = object.__new__(_TestSpeculator)
    speculator.model = SimpleNamespace(
        compute_logits=lambda hidden_states: torch.zeros(hidden_states.shape[0], 5)
    )
    speculator.active_num_reqs = torch.tensor(2, dtype=torch.int32)
    speculator.use_fp64_gumbel = False
    positions = torch.tensor([12, 99], dtype=torch.int64)

    speculator.sample_draft(
        hidden_states=torch.zeros(2, 3),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
    )

    assert captured_positions is not None
    torch.testing.assert_close(
        captured_positions,
        positions + 1 + (1 << 30),
        rtol=0,
        atol=0,
    )
