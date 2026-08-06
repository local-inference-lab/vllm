# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.spec_decode.dynamic.acceptance_length import (
    AcceptanceLengthController,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.utils import limit_draft_tokens


def test_controller_starts_at_max_and_waits_for_window():
    controller = AcceptanceLengthController(max_num_spec_tokens=5, observation_window=2)

    assert controller.num_spec_tokens == 5
    assert (
        controller.observe_batch(
            num_drafts=2,
            num_draft_tokens=10,
            num_accepted_tokens=2,
        )
        is None
    )

    update = controller.observe_batch(
        num_drafts=2,
        num_draft_tokens=10,
        num_accepted_tokens=2,
    )

    assert update is not None
    assert update.mean_num_accepted_tokens == 1.0
    assert update.mean_num_draft_tokens == 5.0
    assert update.previous_num_spec_tokens == 5
    assert update.num_spec_tokens == 2


def test_controller_drops_directly_to_lower_target():
    controller = AcceptanceLengthController(max_num_spec_tokens=5, observation_window=1)

    update = controller.observe_batch(
        num_drafts=1,
        num_draft_tokens=5,
        num_accepted_tokens=0,
    )

    assert update is not None
    assert controller.num_spec_tokens == 1


def test_controller_rounds_target_and_creeps_upward():
    controller = AcceptanceLengthController(max_num_spec_tokens=5, observation_window=1)

    controller.observe_batch(
        num_drafts=100,
        num_draft_tokens=500,
        num_accepted_tokens=237,
    )
    assert controller.num_spec_tokens == 3

    controller.observe_batch(
        num_drafts=10,
        num_draft_tokens=30,
        num_accepted_tokens=28,
    )
    assert controller.num_spec_tokens == 4


def test_controller_recovers_after_acceptance_improves():
    controller = AcceptanceLengthController(max_num_spec_tokens=3, observation_window=1)

    controller.observe_batch(
        num_drafts=1,
        num_draft_tokens=3,
        num_accepted_tokens=0,
    )
    assert controller.num_spec_tokens == 1

    for attempted, expected in ((1, 2), (2, 3), (3, 3)):
        controller.observe_batch(
            num_drafts=1,
            num_draft_tokens=attempted,
            num_accepted_tokens=attempted,
        )
        assert controller.num_spec_tokens == expected


def test_controller_ignores_empty_steps_and_validates_counts():
    controller = AcceptanceLengthController(max_num_spec_tokens=3, observation_window=1)

    assert (
        controller.observe_batch(
            num_drafts=0,
            num_draft_tokens=0,
            num_accepted_tokens=0,
        )
        is None
    )
    assert controller.num_spec_tokens == 3

    with pytest.raises(ValueError, match="must not exceed"):
        controller.observe_batch(
            num_drafts=1,
            num_draft_tokens=1,
            num_accepted_tokens=2,
        )
    with pytest.raises(ValueError, match="require at least one draft"):
        controller.observe_batch(
            num_drafts=0,
            num_draft_tokens=1,
            num_accepted_tokens=0,
        )


def _make_scheduler(
    *,
    observation_window: int = 1,
    schedule: list[tuple[int, int, int]] | None = None,
    use_async_scheduler: bool = False,
    log_stats: bool = False,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.adaptive_speculative_tokens_window = observation_window
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    scheduler_cls = AsyncScheduler if use_async_scheduler else Scheduler
    return scheduler_cls(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=log_stats,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _model_output(req_id: str, sampled_token_ids: list[int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[sampled_token_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_scheduler_observes_acceptance_when_stats_are_disabled():
    scheduler = _make_scheduler()
    request = create_requests(num_requests=1, num_tokens=1, ignore_eos=True)[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    assert prefill_output.num_spec_tokens_to_schedule == 3
    scheduler.update_from_output(prefill_output, _model_output(request.request_id, [0]))

    scheduler.update_draft_token_ids(
        DraftTokenIds(req_ids=[request.request_id], draft_token_ids=[[1, 2, 3]])
    )
    verify_output = scheduler.schedule()
    assert len(verify_output.scheduled_spec_decode_tokens[request.request_id]) == 3
    scheduler.update_from_output(verify_output, _model_output(request.request_id, [4]))

    controller = scheduler.acceptance_length_controller
    assert controller is not None
    assert controller.num_spec_tokens == 1
    assert scheduler.schedule().num_spec_tokens_to_schedule == 1


def test_scheduler_stats_report_current_adaptive_depth():
    scheduler = _make_scheduler(log_stats=True)
    request = create_requests(num_requests=1, num_tokens=1, ignore_eos=True)[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(prefill_output, _model_output(request.request_id, [0]))
    scheduler.update_draft_token_ids(
        DraftTokenIds(req_ids=[request.request_id], draft_token_ids=[[1, 2, 3]])
    )

    verify_output = scheduler.schedule()
    engine_outputs = scheduler.update_from_output(
        verify_output, _model_output(request.request_id, [4])
    )

    stats = engine_outputs[0].scheduler_stats
    assert stats is not None
    assert stats.spec_decoding_stats is not None
    assert stats.spec_decoding_stats.current_num_spec_tokens == 1


def test_scheduler_batch_schedule_caps_adaptive_depth():
    scheduler = _make_scheduler(schedule=[(1, 4, 2)])
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert scheduler.acceptance_length_controller is not None
    assert scheduler.acceptance_length_controller.num_spec_tokens == 3
    assert output.num_spec_tokens_to_schedule == 2


def test_async_scheduler_uses_adaptive_depth_for_placeholders():
    scheduler = _make_scheduler(
        schedule=[(1, 4, 2)],
        use_async_scheduler=True,
    )
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_spec_tokens_to_schedule == 2
    assert len(request.spec_token_ids) == 2


def test_adaptive_depth_rejects_non_model_proposers():
    with pytest.raises(ValueError, match="model-backed speculative decoding"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=3,
            adaptive_speculative_tokens_window=2,
        )


def test_runner_v2_limits_drafts_to_adaptive_depth():
    draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    limited = limit_draft_tokens(
        draft_tokens,
        num_speculative_tokens=2,
        max_num_speculative_tokens=3,
    )

    assert limited.tolist() == [[1, 2], [4, 5]]
    assert (
        limited.untyped_storage().data_ptr()
        == draft_tokens.untyped_storage().data_ptr()
    )


def test_synthetic_scheduler_output_uses_default_speculative_depth():
    output = SchedulerOutput.make_empty()

    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 5

    output.num_spec_tokens_to_schedule = 2
    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 2

    output.num_spec_tokens_to_schedule = 0
    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 0


def test_runner_v2_autoregressive_drafter_stops_at_adaptive_depth(monkeypatch):
    monkeypatch.setattr(AutoRegressiveSpeculator, "__abstractmethods__", frozenset())
    speculator = object.__new__(AutoRegressiveSpeculator)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.zeros(1),
        query_start_loc=torch.zeros(2),
    )
    speculator.idx_mapping = torch.zeros(1, dtype=torch.int32)
    speculator.active_num_reqs = torch.zeros(1, dtype=torch.int32)
    speculator.current_draft_step = torch.zeros(1, dtype=torch.int32)
    speculator._generate_draft = Mock()

    AutoRegressiveSpeculator._multi_step_decode(
        speculator,
        num_reqs=1,
        skip_attn=True,
        batch_desc=SimpleNamespace(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=1,
        ),
        num_tokens_across_dp=None,
        num_speculative_tokens=3,
    )

    assert speculator._generate_draft.call_count == 2
    assert speculator.current_draft_step.item() == 2


def test_runner_v2_allows_acceptance_length_adaptation(monkeypatch):
    speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=3,
    )
    # Avoid model downloads while exercising VllmConfig's runner selection.
    speculative_config.method = "mtp"
    speculative_config.adaptive_speculative_tokens_window = 2

    config = VllmConfig(speculative_config=speculative_config)

    unsupported = config._get_v2_model_runner_unsupported_features()
    assert not any("dynamic speculative decoding" in item for item in unsupported)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config._maybe_override_dynamic_sd_cudagraph_mode()
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL

    # The V2 runner supports the batch-size schedule too (the restriction was
    # lifted upstream), so combining both controls stays supported.
    speculative_config.num_speculative_tokens_per_batch_size = [(1, 1, 1)]
    assert not any(
        "dynamic speculative decoding" in item
        for item in config._get_v2_model_runner_unsupported_features()
    )
