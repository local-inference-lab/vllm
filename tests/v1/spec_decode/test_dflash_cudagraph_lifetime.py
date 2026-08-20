# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as cudagraph_module
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
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


def test_dflash_graph_capture_marks_forward_as_capture_only(monkeypatch):
    forward = Mock()
    manager = object.__new__(DFlashCudaGraphManager)
    manager.max_num_reqs = 4
    manager.dp_size = 1
    desc = SimpleNamespace(
        num_tokens=6,
        num_reqs=2,
        cg_mode=CUDAGraphMode.FULL,
        uniform_token_count=3,
    )

    monkeypatch.setattr(
        cudagraph_module,
        "_prepare_dflash_inputs_to_capture",
        lambda *args, **kwargs: ("attention", "slots"),
    )

    def fake_capture(_manager, create_forward_fn, **kwargs):
        create_forward_fn(desc, warmup=True)(CUDAGraphMode.FULL)

    monkeypatch.setattr(CudaGraphManager, "capture", fake_capture)

    DFlashCudaGraphManager.capture(
        manager,
        forward_fn=forward,
        input_buffers=object(),
        block_tables=object(),
        attn_groups=[],
        kv_cache_config=object(),
        max_model_len=1024,
        causal=False,
        channel_id="draft-query",
    )

    assert forward.call_args.args == (
        2,
        6,
        "attention",
        "slots",
        None,
        CUDAGraphMode.FULL,
    )
    assert forward.call_args.kwargs == {
        "num_query_per_req": 3,
        "capture_only": True,
    }


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
    created: list[tuple[object, ...]] = []

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


def test_dflash_uses_streamed_context_without_copying_full_hidden_state():
    speculator = object.__new__(DFlashSpeculator)
    streamed = torch.randn(1024, 4)
    retained = torch.full((1024, 8), float("nan"))
    model = Mock()
    model.is_streamed_context_states.return_value = True
    speculator.model = model
    speculator.hidden_states = retained
    speculator.dynamic_physical_depth = False
    speculator.num_speculative_steps = 7
    speculator.num_query_per_req = 8
    speculator.sample_from_anchor = False
    speculator.max_model_len = 4096
    speculator.draft_kv_window = None
    speculator.context_positions = torch.arange(1024)
    speculator._prepare_eplb_forward = Mock()
    speculator._generate_draft = Mock()
    speculator.draft_tokens = torch.zeros(1, 7, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1024,
        seq_lens_cpu_upper_bound=torch.tensor([1024]),
    )

    result = DFlashSpeculator.propose(
        speculator,
        input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.empty(1024, 8),
        aux_hidden_states=[streamed],
        num_sampled=torch.zeros(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int32),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int32),
        temperature=torch.zeros(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
    )

    args = model.precompute_and_store_context_kv.call_args.args
    assert args[0] is streamed
    assert torch.isnan(retained).all()
    speculator._generate_draft.assert_called_once()
    assert result.shape == (1, 7)


def test_dflash_streamed_context_bypasses_captured_context_graph():
    manager = Mock()
    manager.dispatch_context.return_value = SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=2048,
    )
    speculator = SimpleNamespace(context_cudagraph_manager=manager)

    result = DFlashSpeculator._dispatch_context_batch(
        speculator,
        1024,
        context_states_are_streamed=True,
        dummy_run=False,
        is_profile=False,
    )

    manager.dispatch_context.assert_not_called()
    assert result.cg_mode == CUDAGraphMode.NONE
    assert result.num_tokens == 1024


def test_dflash_input_warmup_copies_sampling_state(monkeypatch):
    temperature = torch.ones(2)
    seeds = torch.zeros(2, dtype=torch.int64)
    speculator = SimpleNamespace(
        draft_kv_cache_group_id=0,
        draft_kv_cache_group_ids=[0],
        num_query_per_req=2,
        dynamic_physical_depth=False,
        max_num_reqs=2,
        max_num_tokens=2048,
        max_model_len=4096,
        device=torch.device("cpu"),
        input_buffers=object(),
        block_tables=SimpleNamespace(
            slot_mappings=[object()],
            input_block_tables=[object()],
            kernel_block_sizes=[16],
        ),
        context_positions=object(),
        _context_slot_mappings=[object()],
        sample_indices=object(),
        sample_pos=object(),
        sample_idx_mapping=object(),
        temperature=temperature,
        seeds=seeds,
        num_cached_tokens=object(),
        parallel_drafting_token_id=1,
        sample_from_anchor=False,
        _speculative_steps_for_query_len=lambda query_len: query_len - 1,
    )
    prepare_inputs = Mock()
    monkeypatch.setattr(spec_module, "prepare_dflash_inputs", prepare_inputs)

    DFlashSpeculator._warmup_prepare_inputs_kernel(speculator)

    assert prepare_inputs.call_count == 5
    for call in prepare_inputs.call_args_list:
        assert call.args[7] is temperature
        assert call.args[8] is seeds
        assert call.args[14] is temperature
        assert call.args[15] is seeds
