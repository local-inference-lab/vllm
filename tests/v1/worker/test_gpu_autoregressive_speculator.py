# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


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


def test_probabilistic_draft_sampler_owns_disjoint_philox_offset(monkeypatch):
    captured = {}

    def fake_gumbel_sample(
        logits,
        idx_mapping,
        temperature,
        seeds,
        positions,
        **kwargs,
    ):
        captured["positions"] = positions
        captured.update(kwargs)
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(base_spec_module, "gumbel_sample", fake_gumbel_sample)
    positions = torch.tensor([12, 99], dtype=torch.int64)
    active_rows = torch.tensor(2, dtype=torch.int32)
    speculator = object.__new__(_TestSpeculator)
    speculator.use_fp64_gumbel = False

    speculator._sample_probabilistic_draft(
        logits=torch.zeros(2, 5),
        positions=positions,
        idx_mapping=torch.arange(2),
        temperature=torch.ones(2),
        seeds=torch.tensor([7, 11], dtype=torch.int64),
        draft_step=torch.tensor(0, dtype=torch.int64),
        draft_logits=torch.empty(2, 5),
        active_rows=active_rows,
    )

    torch.testing.assert_close(
        captured["positions"], positions + 1 + (1 << 30), rtol=0, atol=0
    )
    assert captured["apply_temperature"] is True
    assert captured["output_processed_logits_active_rows"] is active_rows


def test_ar_probabilistic_draft_uses_shared_sampler(monkeypatch):
    captured = {}

    def fake_sample_probabilistic_draft(
        self,
        logits,
        positions,
        idx_mapping,
        temperature,
        seeds,
        draft_step,
        draft_logits,
        active_rows=None,
    ):
        captured["positions"] = positions
        captured["active_rows"] = active_rows
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(
        DraftModelSpeculator,
        "_sample_probabilistic_draft",
        fake_sample_probabilistic_draft,
    )
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

    assert captured["positions"] is positions
    assert captured["active_rows"] is speculator.active_num_reqs


def test_autoregressive_capture_ids_are_deterministic_and_phase_separated():
    def capture_channel_ids() -> list[str]:
        events = []

        class FakeManager:
            use_breakable_cg = False

            def capture(self, *args, **kwargs):
                events.append(kwargs["channel_id"])

        speculator = object.__new__(_TestSpeculator)
        speculator.last_token_indices = torch.zeros(1)
        speculator.prefill_cudagraph_manager = FakeManager()
        speculator.decode_cudagraph_manager = FakeManager()
        speculator.model = object()
        speculator._prefill = object()
        speculator._generate_draft = object()
        speculator.model_state = object()
        speculator.target_input_buffers = object()
        speculator.input_buffers = object()
        speculator.block_tables = object()
        speculator.target_attn_groups = object()
        speculator.attn_groups = object()
        speculator.kv_cache_config = object()
        speculator.num_speculative_steps = 3

        speculator.capture(capture_phase="profile")
        speculator.capture(capture_phase="production")
        return events

    expected = [
        "vllm:draft:prefill:profile",
        "vllm:draft:decode:profile",
        "vllm:draft:prefill:production",
        "vllm:draft:decode:production",
    ]
    assert capture_channel_ids() == capture_channel_ids() == expected


def test_autoregressive_graph_channel_is_bound_at_capture_not_construction(
    monkeypatch,
):
    created = []

    class FakeManager:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            created.append((vllm_config, device, cudagraph_mode, decode_query_len))

    monkeypatch.setattr(spec_module, "SpeculatorCudaGraphManager", FakeManager)
    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3

    AutoRegressiveSpeculator.init_cudagraph_manager(
        speculator,
        CUDAGraphMode.FULL,
    )

    assert created == [
        (
            speculator.vllm_config,
            speculator.device,
            CUDAGraphMode.FULL,
            4,
        ),
        (
            speculator.vllm_config,
            speculator.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            1,
        ),
    ]
