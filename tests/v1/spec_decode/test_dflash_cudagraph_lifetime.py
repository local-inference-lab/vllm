# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as spec_module
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


def test_dflash_capture_uses_phase_specific_draft_channel_ids():
    events = []

    class FakeQueryManager:
        def capture(self, *args, **kwargs):
            events.append(("query", kwargs["channel_id"]))

    class FakeContextManager:
        def capture_context(self, *args, **kwargs):
            events.append(("context", kwargs["channel_id"]))

    speculator = SimpleNamespace(
        _speculator_name="DFlash",
        sample_indices=torch.zeros(1),
        sample_pos=torch.zeros(1),
        sample_idx_mapping=torch.zeros(1),
        query_cudagraph_manager=FakeQueryManager(),
        context_cudagraph_manager=FakeContextManager(),
        _context_slot_mappings=torch.zeros(1, 1),
        context_positions=torch.ones(1),
        _capture_context_kv=object(),
        _generate_draft=object(),
        input_buffers=object(),
        block_tables=object(),
        attn_groups=object(),
        kv_cache_config=object(),
        max_model_len=1,
        _group_causal=True,
    )

    DFlashSpeculator.capture(speculator, capture_phase="profile")
    DFlashSpeculator.capture(speculator, capture_phase="production")

    assert events == [
        ("query", "vllm:draft:dflash:profile"),
        ("context", "vllm:draft:dflash:context:profile"),
        ("query", "vllm:draft:dflash:production"),
        ("context", "vllm:draft:dflash:context:production"),
    ]


def test_dflash_graph_channel_is_bound_at_capture_not_construction(monkeypatch):
    created = []

    class FakeQueryManager:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            created.append(
                ("query", vllm_config, device, cudagraph_mode, decode_query_len)
            )

    class FakeContextManager:
        def __init__(self, vllm_config, device, max_num_context_tokens):
            created.append(("context", vllm_config, device, max_num_context_tokens))

    monkeypatch.setattr(spec_module, "DFlashCudaGraphManager", FakeQueryManager)
    monkeypatch.setattr(
        spec_module,
        "DFlashContextCudaGraphManager",
        FakeContextManager,
    )

    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_query_per_req = 6
    speculator.max_num_tokens = 128
    speculator._speculator_name = "DSpark"
    speculator.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
        min_cg_attn_backend="test",
    )

    DFlashSpeculator.init_cudagraph_manager(speculator, CUDAGraphMode.FULL)

    assert created == [
        (
            "query",
            speculator.vllm_config,
            speculator.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            6,
        ),
        ("context", speculator.vllm_config, speculator.device, 128),
    ]


def test_dflash_context_precompute_replays_full_graph():
    manager = Mock()
    model = Mock()
    speculator = SimpleNamespace(
        context_cudagraph_manager=manager,
        model=model,
        hidden_states=torch.zeros(8, 4),
        context_positions=torch.zeros(8, dtype=torch.int64),
    )
    desc = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)

    DFlashSpeculator._precompute_context_kv(
        speculator,
        num_target_tokens=3,
        batch_desc=desc,
        context_slots=torch.zeros(3, dtype=torch.int64),
    )

    manager.run_fullgraph.assert_called_once_with(desc)
    model.precompute_and_store_context_kv.assert_not_called()


def test_dflash_context_precompute_keeps_eager_fallback():
    model = Mock()
    hidden_states = torch.zeros(8, 4)
    context_positions = torch.zeros(8, dtype=torch.int64)
    context_slots = torch.zeros(3, dtype=torch.int64)
    speculator = SimpleNamespace(
        context_cudagraph_manager=None,
        model=model,
        hidden_states=hidden_states,
        context_positions=context_positions,
    )

    DFlashSpeculator._precompute_context_kv(
        speculator,
        num_target_tokens=3,
        batch_desc=SimpleNamespace(cg_mode=CUDAGraphMode.NONE),
        context_slots=context_slots,
    )

    args = model.precompute_and_store_context_kv.call_args.args
    assert args[0].data_ptr() == hidden_states.data_ptr()
    assert args[0].shape == (3, 4)
    assert args[1].data_ptr() == context_positions.data_ptr()
    assert args[1].shape == (3,)
    assert args[2] is context_slots
