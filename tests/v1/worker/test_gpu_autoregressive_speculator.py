# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    compact_mrope_positions,
    prepare_decode_inputs,
    prepare_prefill_inputs,
    update_draft_inputs,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output
        self.last_kwargs: dict[str, Any] | None = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _QSAIntervalLifecycle:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.interval_start = 7
        self.interval_start_snapshot: int | None = None
        self.skip_topk = False

    def snapshot_qsa_interval_starts(self) -> None:
        self.calls.append("snapshot")
        self.interval_start_snapshot = self.interval_start

    def restore_qsa_interval_starts(self) -> None:
        self.calls.append("restore")
        assert self.interval_start_snapshot is not None
        self.interval_start = self.interval_start_snapshot

    def set_skip_topk(self, skip: bool) -> None:
        self.calls.append(f"skip_topk={skip}")
        self.skip_topk = skip

    def compact_topk_indices(self, last_token_indices: torch.Tensor) -> None:
        self.calls.append("compact_topk")


def test_autoregressive_speculator_uses_sparse_full_draft_prefill_graphs(
    monkeypatch,
) -> None:
    manager_args: list[tuple[CUDAGraphMode, dict[str, Any]]] = []

    def make_manager(_config, _device, mode, *args, **kwargs):
        manager_args.append((mode, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(spec_module, "SpeculatorCudaGraphManager", make_manager)

    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = SimpleNamespace()
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 5
    speculator.max_num_reqs = 32

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_AND_PIECEWISE)

    assert manager_args == [
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            {"full_capture_request_sizes": frozenset({1, 2, 4, 8, 16, 32})},
        ),
        (CUDAGraphMode.FULL_DECODE_ONLY, {"decode_query_len": 1}),
    ]


def test_cudagraph_manager_excludes_uncaptured_full_candidates() -> None:
    manager = object.__new__(CudaGraphManager)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 4, *range(8, 193, 8)],
        max_cudagraph_capture_size=192,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    manager.max_num_reqs = 32
    manager.decode_query_len = 6
    manager.varlen_decode = False
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.lora_capture_cases = [0]
    manager.full_capture_request_sizes = frozenset({1, 2, 4, 8, 16, 32})
    manager._capture_descs = {}
    manager._candidates = {}

    manager._init_candidates()

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    assert {desc.num_tokens for desc in full_descs} == {6, 12, 24, 48, 96, 192}
    assert all(
        desc.cg_mode != CUDAGraphMode.FULL
        or desc.num_tokens in {6, 12, 24, 48, 96, 192}
        for candidates in manager._candidates.values()
        for desc in candidates
    )


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def test_glm5next_mtp_uses_collapsed_hidden_size() -> None:
    draft_model_config = SimpleNamespace(
        get_hidden_size=lambda: 4096,
        get_vocab_size=lambda: 154880,
        hf_config=SimpleNamespace(
            model_type="glm5_next_mtp",
            hc_mult=4,
            mhc=True,
        ),
        uses_mrope=False,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=5,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=1024,
            dtype=torch.bfloat16,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == 4096
    assert speculator.hidden_states.shape == (8, 4096)


def test_model_runner_propagates_auto_fit_length_to_draft_model() -> None:
    draft_model = Mock()
    speculator = object.__new__(_TestSpeculator)
    speculator.model = draft_model
    speculator.max_model_len = 1024
    speculator.draft_max_seq_len = 1024

    target_model = Mock()
    runner = object.__new__(GPUModelRunner)
    runner.model = target_model
    runner.speculator = speculator
    runner.req_states = SimpleNamespace(max_model_len=1024)

    runner.update_max_model_len(216_576)

    assert runner.max_model_len == 216_576
    assert runner.req_states.max_model_len == 216_576
    assert speculator.max_model_len == 216_576
    assert speculator.draft_max_seq_len == 216_576
    target_model.update_max_model_len.assert_called_once_with(216_576)
    draft_model.update_max_model_len.assert_called_once_with(216_576)


def test_mtp_speculator_rolls_back_qsa_anchor_around_lookahead() -> None:
    lifecycle = _QSAIntervalLifecycle()
    speculator = object.__new__(MTPSpeculator)
    speculator.model = SimpleNamespace(model=lifecycle)
    speculator.rollback_qsa_interval_starts = True
    speculator.share_mtp_topk_indices = False

    speculator.on_multi_step_decode_begin(num_reqs=3)
    speculator.on_multi_step_decode_end(num_reqs=3)

    assert lifecycle.calls == ["snapshot", "restore"]


def test_propose_restores_mtp_state_when_draft_decode_raises(monkeypatch) -> None:
    lifecycle = _QSAIntervalLifecycle()
    speculator = object.__new__(MTPSpeculator)
    speculator.model = SimpleNamespace(model=lifecycle)
    speculator.rollback_qsa_interval_starts = True
    speculator.share_mtp_topk_indices = True
    speculator.prefill_outputs_are_compact = False
    speculator.num_speculative_steps = 2
    speculator.max_model_len = 32
    speculator.max_num_reqs = 1
    speculator.hidden_states = torch.zeros(3, 2)
    speculator.last_token_indices = torch.zeros(1, dtype=torch.int64)
    speculator.sample_src_positions = torch.zeros(1, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0, dtype=torch.int64)
    speculator.input_buffers = SimpleNamespace()
    speculator.draft_tokens = torch.zeros((1, 2), dtype=torch.int64)
    speculator.prefill_cudagraph_manager = object()
    speculator.decode_cudagraph_manager = object()
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.use_fused_multi_step_decode = False
    speculator.mrope_positions = None
    speculator._copy_request_inputs = Mock()
    speculator._prepare_eplb_forward = Mock()
    speculator._prefill = Mock()

    def fail_decode(*args, **kwargs) -> None:
        lifecycle.interval_start = 19
        raise RuntimeError("draft decode failed")

    speculator._multi_step_decode = fail_decode

    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", Mock())
    monkeypatch.setattr(spec_module, "prepare_decode_inputs", Mock())
    monkeypatch.setattr(
        spec_module,
        "get_uniform_decode_token_count",
        lambda *args, **kwargs: 3,
    )
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            SimpleNamespace(cg_mode=CUDAGraphMode.NONE, num_tokens=3),
            None,
        ),
    )

    input_batch = SimpleNamespace(
        num_tokens=3,
        num_tokens_after_padding=3,
        num_reqs=1,
        num_scheduled_tokens=torch.tensor([3]),
        seq_lens_cpu_upper_bound=torch.tensor([5]),
        idx_mapping=torch.tensor([0]),
        has_prefill=False,
        seq_lens=torch.tensor([5]),
    )

    with pytest.raises(RuntimeError, match="draft decode failed"):
        speculator.propose(
            input_batch=input_batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.zeros(3, 2),
            aux_hidden_states=None,
            num_sampled=torch.tensor([1]),
            num_rejected=torch.tensor([0]),
            last_sampled=torch.tensor([1]),
            next_prefill_tokens=torch.tensor([0]),
            temperature=torch.tensor([1.0]),
            seeds=torch.tensor([0]),
        )

    assert lifecycle.interval_start == 7
    assert not lifecycle.skip_topk
    assert lifecycle.calls == [
        "skip_topk=False",
        "compact_topk",
        "snapshot",
        "skip_topk=True",
        "restore",
        "skip_topk=False",
    ]


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
    speculator.mrope_positions = None
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = SimpleNamespace(uses_mrope=False)
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


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


def test_run_model_uses_persistent_mrope_positions(monkeypatch):
    hidden = torch.zeros(4, 3)
    speculator = _make_speculator(monkeypatch, hidden)
    speculator.mrope_positions = torch.arange(15).reshape(3, 5)

    speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    positions = speculator.model.last_kwargs["positions"]
    assert positions.shape == (3, 4)
    assert positions.stride() == (5, 1)
    assert positions.data_ptr() == speculator.mrope_positions.data_ptr()


def test_default_model_state_exposes_prepared_model_positions():
    expected = torch.arange(15).reshape(3, 5)
    rope_state = SimpleNamespace(get_positions=Mock(return_value=expected))
    state = object.__new__(DefaultModelState)
    state.rope_state = rope_state
    input_batch = SimpleNamespace(
        positions=torch.arange(4),
        num_tokens_after_padding=4,
    )

    actual = state.get_model_positions(input_batch)

    assert actual is expected
    rope_state.get_positions.assert_called_once_with(4)


def test_mrope_profile_uses_scalar_positions_before_target_state_is_bound(monkeypatch):
    speculator = _make_speculator(monkeypatch, torch.zeros(4, 3))
    speculator.mrope_positions = torch.zeros(3, 5, dtype=torch.int64)
    assert not hasattr(speculator, "model_state")
    input_batch = SimpleNamespace(positions=torch.arange(4, dtype=torch.int64))

    actual = speculator._target_model_positions(input_batch, is_profile=True)

    assert actual is input_batch.positions

    with pytest.raises(RuntimeError, match="target model state is not bound"):
        speculator._target_model_positions(input_batch, is_profile=False)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="accelerator required")
@pytest.mark.parametrize("max_model_len", [108, 4096])
@pytest.mark.parametrize("advance_draft_positions", [False, True])
def test_mrope_prefill_compaction_and_continuation_kernels(
    max_model_len: int, advance_draft_positions: bool
):
    device = torch.device("cuda")
    sentinel = -777
    input_buffers = SimpleNamespace(
        input_ids=torch.full((8,), sentinel, dtype=torch.int32, device=device),
        positions=torch.full((8,), sentinel, dtype=torch.int64, device=device),
        query_start_loc=torch.full((4,), sentinel, dtype=torch.int32, device=device),
        seq_lens=torch.full((3,), sentinel, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        input_ids=torch.arange(1, 8, dtype=torch.int32, device=device),
        positions=torch.arange(100, 107, dtype=torch.int64, device=device),
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 4, 7], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([10, 20], dtype=torch.int32, device=device),
    )
    target_model_positions = torch.stack(
        [
            torch.arange(10, 17, dtype=torch.int64, device=device),
            torch.arange(20, 27, dtype=torch.int64, device=device),
            torch.arange(30, 37, dtype=torch.int64, device=device),
        ]
    )
    mrope_positions = torch.full((3, 9), sentinel, dtype=torch.int64, device=device)
    mrope_positions_scratch = torch.empty((3, 3), dtype=torch.int64, device=device)
    backing_ptr = mrope_positions.data_ptr()
    last_token_indices = torch.zeros(2, dtype=torch.int64, device=device)
    current_draft_step = torch.tensor(9, dtype=torch.int64, device=device)
    num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=device)

    prepare_prefill_inputs(
        last_token_indices,
        current_draft_step,
        input_buffers,
        input_batch,
        num_sampled=torch.ones(2, dtype=torch.int32, device=device),
        num_rejected=num_rejected,
        last_sampled=torch.tensor([90, 91], dtype=torch.int64, device=device),
        next_prefill_tokens=torch.zeros(2, dtype=torch.int32, device=device),
        max_num_reqs=3,
        target_model_positions=target_model_positions,
        draft_mrope_positions=mrope_positions,
    )
    torch.accelerator.synchronize()

    assert last_token_indices.tolist() == [2, 6]
    assert current_draft_step.item() == 0
    assert input_buffers.positions.tolist() == [
        100,
        101,
        102,
        sentinel,
        104,
        105,
        106,
        sentinel,
    ]
    assert mrope_positions[:, 3].tolist() == [sentinel, sentinel, sentinel]
    assert torch.equal(mrope_positions[:, :3], target_model_positions[:, :3])
    assert torch.equal(mrope_positions[:, 4:7], target_model_positions[:, 4:7])

    compact_mrope_positions(
        mrope_positions,
        mrope_positions_scratch,
        last_token_indices,
        num_reqs=2,
    )
    input_buffers.positions[:2] = torch.tensor([102, 106], device=device)
    sample_src_positions = torch.tensor([103, 107], device=device)
    prepare_decode_inputs(
        draft_tokens=torch.tensor([70, 71], dtype=torch.int64, device=device),
        target_seq_lens=input_batch.seq_lens,
        num_rejected=num_rejected,
        input_buffers=input_buffers,
        sample_src_positions=sample_src_positions,
        max_model_len=max_model_len,
        max_num_reqs=3,
        advance_draft_positions=advance_draft_positions,
        mrope_positions=mrope_positions,
    )
    torch.accelerator.synchronize()

    shift = int(advance_draft_positions)
    assert input_buffers.positions[:2].tolist() == [102 + shift, 106 + shift]
    expected_mrope = (
        [[13, 17]] * 3 if advance_draft_positions else [[12, 16], [22, 26], [32, 36]]
    )
    assert mrope_positions[:, :2].tolist() == expected_mrope
    assert sample_src_positions.tolist() == [104, 108]
    assert mrope_positions.data_ptr() == backing_ptr

    overlapping_positions = torch.tensor(
        [
            [10, 11, 12, sentinel],
            [20, 21, 22, sentinel],
            [30, 31, 32, sentinel],
        ],
        dtype=torch.int64,
        device=device,
    )
    compact_mrope_positions(
        overlapping_positions,
        mrope_positions_scratch,
        torch.tensor([1, 2], dtype=torch.int64, device=device),
        num_reqs=2,
    )
    torch.accelerator.synchronize()
    assert overlapping_positions[:, :2].tolist() == [
        [11, 12],
        [21, 22],
        [31, 32],
    ]

    large_num_reqs = 128
    permuted_positions = torch.arange(
        3 * (large_num_reqs + 1), dtype=torch.int64, device=device
    ).reshape(3, large_num_reqs + 1)
    permutation = torch.arange(
        large_num_reqs - 1, -1, -1, dtype=torch.int64, device=device
    )
    expected_permutation = permuted_positions[:, permutation].clone()
    large_scratch = torch.empty((3, large_num_reqs), dtype=torch.int64, device=device)
    compact_mrope_positions(
        permuted_positions,
        large_scratch,
        permutation,
        num_reqs=large_num_reqs,
    )
    torch.accelerator.synchronize()
    assert torch.equal(permuted_positions[:, :large_num_reqs], expected_permutation)

    update_draft_inputs(
        draft_tokens=torch.tensor([80, 81], dtype=torch.int64, device=device),
        current_draft_step=torch.tensor(1, dtype=torch.int64, device=device),
        hidden_states=torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4),
        output_draft_tokens=torch.zeros((2, 3), dtype=torch.int64, device=device),
        next_input_hidden_states=torch.zeros((2, 4), device=device),
        input_buffers=input_buffers,
        sample_src_positions=sample_src_positions,
        num_reqs=2,
        max_model_len=max_model_len,
        num_speculative_steps=3,
        advance_draft_positions=advance_draft_positions,
        mrope_positions=mrope_positions,
    )
    torch.accelerator.synchronize()

    assert input_buffers.positions[:2].tolist() == [
        102 + 2 * shift,
        min(106 + 2 * shift, max_model_len - 1),
    ]
    expected_mrope = (
        [[14, 18]] * 3 if advance_draft_positions else [[12, 16], [22, 26], [32, 36]]
    )
    assert mrope_positions[:, :2].tolist() == expected_mrope
    # Sampling positions must advance even for clamped or Q-only forward positions.
    assert sample_src_positions.tolist() == [105, 109]
    assert mrope_positions.data_ptr() == backing_ptr


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None
