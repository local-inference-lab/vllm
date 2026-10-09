# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CUDAGraphMode, SpeculativeConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.spec_decode.dynamic.acceptance_length import (
    AcceptanceLengthController,
    BatchSizeAcceptanceLengthController,
)
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.utils import limit_draft_tokens


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def _make_scheduler_with_adaptive_sd(
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 5,
    observation_window: int = 1,
    initial_num_speculative_tokens: int | None = 3,
    schedule: list[tuple[int, int, int]] | None = None,
    log_stats: bool = True,
    enable_prefix_caching: bool = False,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
        num_speculative_tokens_per_batch_size=schedule,
        adaptive_speculative_tokens_window=observation_window,
        adaptive_speculative_tokens_initial=initial_num_speculative_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.method = "draft_model"
    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=log_stats,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_acceptance_length_controller_drops_directly_and_recovers_one_step():
    controller = AcceptanceLengthController(
        max_num_spec_tokens=5,
        observation_window=1,
        initial_num_spec_tokens=3,
    )

    update = controller.observe_batch(
        num_drafts=2,
        num_draft_tokens=6,
        num_accepted_tokens=0,
    )
    assert update is not None
    assert update.num_spec_tokens == 1

    update = controller.observe_batch(
        num_drafts=2,
        num_draft_tokens=2,
        num_accepted_tokens=2,
    )
    assert update is not None
    assert update.num_spec_tokens == 2


def test_acceptance_length_controller_keeps_batch_cap_bands_independent():
    controller = BatchSizeAcceptanceLengthController(
        max_num_spec_tokens=5,
        observation_window=1,
        initial_num_spec_tokens=3,
        num_spec_tokens_by_batch_size=[0, 5, 5, 3, 3, 0],
    )

    controller.observe_batch(
        batch_size=1,
        num_drafts=1,
        num_draft_tokens=3,
        num_accepted_tokens=0,
    )

    assert controller.num_spec_tokens_for_batch_size(1) == 1
    assert controller.num_spec_tokens_for_batch_size(2) == 1
    assert controller.num_spec_tokens_for_batch_size(3) == 3
    assert controller.num_spec_tokens_for_batch_size(5) == 0


def test_acceptance_length_controller_validates_initial_depth():
    with pytest.raises(ValueError, match="initial_num_spec_tokens"):
        AcceptanceLengthController(
            max_num_spec_tokens=3,
            observation_window=1,
            initial_num_spec_tokens=4,
        )


def test_acceptance_length_adaptation_rejects_non_model_proposer():
    with pytest.raises(ValueError, match="model-backed speculative decoding"):
        SpeculativeConfig(
            model="ngram",
            num_speculative_tokens=3,
            adaptive_speculative_tokens_window=1,
        )


def test_adaptive_initial_depth_requires_window():
    with pytest.raises(
        ValueError, match="adaptive_speculative_tokens_initial requires"
    ):
        SpeculativeConfig(
            model="ngram",
            num_speculative_tokens=3,
            adaptive_speculative_tokens_initial=2,
        )


def test_acceptance_length_adaptation_updates_without_metrics():
    scheduler = _make_scheduler_with_adaptive_sd(log_stats=False)
    [request] = create_requests(num_requests=1, num_tokens=1)
    scheduler.add_request(request)
    request_id = request.request_id

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[request_id],
            req_id_to_index={request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(DraftTokenIds([request_id], [[1, 2, 3]]))
    verify_output = scheduler.schedule()
    scheduler.update_from_output(
        verify_output,
        ModelRunnerOutput(
            req_ids=[request_id],
            req_id_to_index={request_id: 0},
            sampled_token_ids=[[4]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    assert scheduler.schedule().num_spec_tokens_to_schedule == 1


def test_synthetic_scheduler_output_uses_default_speculative_depth():
    output = SchedulerOutput.make_empty()

    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 5

    output.num_spec_tokens_to_schedule = 2
    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 2

    output.num_spec_tokens_to_schedule = 0
    assert output.resolve_num_spec_tokens_to_schedule(default=5) == 0


def test_runner_v2_limits_drafts_to_selected_depth():
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


@pytest.mark.parametrize("num_speculative_tokens", range(1, 6))
def test_runner_v2_autoregressive_drafter_stops_at_selected_depth(
    monkeypatch, num_speculative_tokens
):
    speculator = SimpleNamespace()
    speculator.input_buffers = SimpleNamespace(
        positions=torch.zeros(1),
        query_start_loc=torch.zeros(2),
    )
    speculator.idx_mapping = torch.zeros(1, dtype=torch.int32)
    speculator.current_draft_step = torch.zeros(1, dtype=torch.int32)
    speculator._generate_draft = Mock()

    TargetDependentARSpeculator._multi_step_decode(
        speculator,
        num_reqs=1,
        skip_attn=True,
        batch_desc=SimpleNamespace(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=1,
        ),
        num_tokens_across_dp=None,
        seq_lens_cpu_upper_bound=torch.ones(1, dtype=torch.int32),
        num_speculative_steps=num_speculative_tokens,
    )

    assert speculator._generate_draft.call_count == num_speculative_tokens - 1
    assert speculator.current_draft_step.item() == num_speculative_tokens - 1


def test_acceptance_length_adaptation_uses_batch_size_caps():
    scheduler = _make_scheduler_with_adaptive_sd(
        max_num_seqs=4,
        schedule=[(1, 2, 5), (3, 4, 3)],
    )
    controller = scheduler.acceptance_length_controller
    assert controller is not None
    controller.observe_batch(
        batch_size=1,
        num_drafts=1,
        num_draft_tokens=3,
        num_accepted_tokens=0,
    )

    output = _add_requests_and_schedule(scheduler, 1)
    assert output.num_spec_tokens_to_schedule == 1
    assert controller.num_spec_tokens_for_batch_size(3) == 3


def test_acceptance_length_adaptation_does_not_pad_to_static_max_depth():
    scheduler = _make_scheduler_with_adaptive_sd(
        enable_prefix_caching=True,
    )
    first_request, new_request = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=16,
    )
    scheduler.add_request(first_request)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[first_request.request_id],
            req_id_to_index={first_request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds([first_request.request_id], [[1, 2, 3]])
    )

    scheduler.add_request(new_request)
    output = scheduler.schedule()

    assert output.scheduled_spec_decode_tokens[first_request.request_id] == [
        1,
        2,
        3,
    ]
    assert output.num_scheduled_tokens[new_request.request_id] == 1
    assert new_request.request_id not in output.scheduled_spec_decode_tokens


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            adaptive_speculative_tokens_window=8,
            adaptive_speculative_tokens_initial=2,
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert speculative_config.adaptive_speculative_tokens_window is None
    assert speculative_config.adaptive_speculative_tokens_initial is None
    assert scheduler.dynamic_sd_lookup is None
    assert scheduler.acceptance_length_controller is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3
