# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.config import DeviceConfig, VllmConfig
from vllm.v1.core.sched.output import ScheduledEncoderInputStats, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.loggers import (
    AggregatedLoggingStatLogger,
    LoggingStatLogger,
    PrometheusStatLogger,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import (
    IterationStats,
    PrefillStats,
    PromptTokenStats,
    RequestStateStats,
    SchedulerIterationDetails,
    SchedulerStats,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats
from vllm.v1.utils import compute_iteration_details


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    assert repr(iteration_stats).startswith("IterationStats(")


def test_spec_decoding_logging_uses_verified_position_counts():
    spec_logging = SpecDecodingLogging()
    one_verified = SpecDecodingStats.new(num_spec_tokens=3)
    one_verified.observe_draft(num_draft_tokens=1, num_accepted_tokens=1)
    two_verified = SpecDecodingStats.new(num_spec_tokens=3)
    two_verified.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
    spec_logging.observe(one_verified)
    spec_logging.observe(two_verified)

    messages = []
    spec_logging.log(lambda fmt, *args: messages.append(fmt % args))

    assert len(messages) == 1
    assert "Mean verification depth: 1.50/3" in messages[0]
    assert "Drafted: 3 tokens" in messages[0]
    assert "Per-position acceptance rate: 1.000, 0.000, -----" in messages[0]
    assert "Per-position verification coverage: 1.000, 0.500, 0.000" in messages[0]
    assert "Avg Draft acceptance rate: 66.7%" in messages[0]


def test_scheduler_iteration_details_serialization():
    iteration_details = SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
        num_encoder_inputs=2,
        num_encoder_output_tokens=392,
    )
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            kv_cache_usage=0.5,
            iteration_details=iteration_details,
            num_computed_prefill_tokens=4096,
        )
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.kv_cache_usage == 0.5
    assert decoded.scheduler_stats.iteration_details == iteration_details
    assert decoded.scheduler_stats.num_computed_prefill_tokens == 4096


@pytest.mark.parametrize("aggregated", [False, True])
def test_prefill_counters_advance_before_first_output_without_double_count(aggregated):
    """Chunk completion credits compute; first output credits only cache reuse."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = Mock(
        served_model_name="test-model", max_model_len=16384, is_diffusion=False
    )
    with patch("vllm.v1.metrics.loggers.time.monotonic", return_value=0.0):
        logs = (
            AggregatedLoggingStatLogger(config, [0])
            if aggregated
            else LoggingStatLogger(config)
        )
    try:
        metrics = PrometheusStatLogger(config)
        sources = metrics.counter_prompt_tokens_by_source
        for count in (4096, 2048):
            step = SchedulerStats(num_computed_prefill_tokens=count)
            metrics.record(step, None)
            logs.record(step, None)
        assert metrics.counter_prompt_tokens[0]._value.get() == 6144
        assert sources["local_compute"][0]._value.get() == 6144
        assert metrics.counter_prompt_tokens_cached[0]._value.get() == 0
        with patch("vllm.v1.metrics.loggers.time.monotonic", return_value=10.0):
            logs.log()
        assert logs.last_prompt_throughput == 614.4

        prefill = PrefillStats()
        prefill.set(
            num_prompt_tokens=10000,
            num_local_cached_tokens=2000,
            num_external_cached_tokens=1000,
        )
        first_output = IterationStats()
        first_output.prompt_token_stats.update_from_output(prefill)
        first_output.num_generation_tokens = 1
        last_chunk = SchedulerStats(num_computed_prefill_tokens=856)
        metrics.record(last_chunk, first_output)
        logs.record(last_chunk, first_output)
        assert metrics.counter_prompt_tokens[0]._value.get() == 10000
        assert {
            source: counters[0]._value.get() for source, counters in sources.items()
        } == {
            "local_compute": 7000,
            "local_cache_hit": 2000,
            "external_kv_transfer": 1000,
        }
        assert metrics.counter_prompt_tokens_cached[0]._value.get() == 3000
        assert metrics.counter_generation_tokens[0]._value.get() == 1
        assert first_output.num_prompt_tokens == 10000
        assert first_output.prompt_token_stats.computed == 7000
        with patch("vllm.v1.metrics.loggers.time.monotonic", return_value=20.0):
            logs.log()
        assert logs.last_prompt_throughput == 85.6
    finally:
        unregister_vllm_metrics()


def test_saved_logits_cache_hit_has_no_computed_prefill_counter():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = Mock(
        served_model_name="test-model", max_model_len=16384, is_diffusion=False
    )
    try:
        metrics = PrometheusStatLogger(config)
        prefill = PrefillStats()
        prefill.set(
            num_prompt_tokens=10000,
            num_local_cached_tokens=10000,
            num_external_cached_tokens=0,
        )
        first_output = IterationStats()
        first_output.prompt_token_stats.update_from_output(prefill)
        metrics.record(SchedulerStats(), first_output)
        assert metrics.counter_prompt_tokens[0]._value.get() == 10000
        assert (
            metrics.counter_prompt_tokens_by_source["local_compute"][0]._value.get()
            == 0
        )
        assert metrics.counter_prompt_tokens_cached[0]._value.get() == 10000
    finally:
        unregister_vllm_metrics()


def test_synchronous_engine_logs_partial_prefill_without_request_output():
    engine = LLMEngine.__new__(LLMEngine)
    engine.should_execute_dummy_batch = False
    engine.log_stats = True
    engine.engine_core = Mock()
    engine.engine_core.get_output.return_value = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(num_computed_prefill_tokens=16)
    )
    engine.output_processor = Mock()
    engine.output_processor.process_outputs.return_value = SimpleNamespace(
        reqs_to_abort=[], request_outputs=[]
    )
    engine.renderer = Mock()
    engine.logger_manager = Mock()
    engine.do_log_stats_with_interval = Mock()
    assert engine.step() == []
    engine.logger_manager.record.assert_called_once()
    engine.do_log_stats_with_interval.assert_called_once()


def test_compute_iteration_details_includes_encoder_stats():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_encoder_input_stats = ScheduledEncoderInputStats(
        num_inputs=2,
        output_tokens=392,
    )

    iteration_details = compute_iteration_details(scheduler_output)

    assert iteration_details.num_encoder_inputs == 2
    assert iteration_details.num_encoder_output_tokens == 392


def test_prefill_kv_computed_with_cache():
    """Test that prefill KV compute correctly excludes cached tokens."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 5.0
    req_stats.num_generation_tokens = 50

    # Case 1: With prefix cache (1200 tokens cached)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-001",
        num_prompt_tokens=10000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=1200,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 10000
    assert finished_req.num_cached_tokens == 1200
    assert finished_req.request_id == "test-req-001"

    # Verify calculation: prefill KV = prompt tokens - cached tokens
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 8800  # 10000 - 1200


def test_prefill_kv_computed_no_cache():
    """Test prefill KV compute without prefix caching."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 10

    # Case 2: No prefix cache
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-002",
        num_prompt_tokens=2000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=0,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 2000
    assert finished_req.num_cached_tokens == 0
    assert finished_req.request_id == "test-req-002"

    # Verify calculation: prefill KV = full prompt when no cache
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 2000


def test_prefill_kv_computed_edge_cases():
    """Test edge cases for prefill KV compute calculation."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 1.0
    req_stats.num_generation_tokens = 1

    # Case 3: Negative num_cached_tokens (shouldn't happen, but handle gracefully)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-003",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=-1,
    )

    finished_req = iteration_stats.finished_requests[0]
    # max() should handle negative values
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 100  # Should treat negative as 0
    assert finished_req.request_id == "test-req-003"

    # Case 4: All tokens cached (shouldn't happen in practice)
    iteration_stats2 = IterationStats()
    iteration_stats2.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-004",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=100,
    )

    finished_req2 = iteration_stats2.finished_requests[0]
    prefill_kv_computed2 = finished_req2.num_prompt_tokens - max(
        finished_req2.num_cached_tokens, 0
    )
    assert prefill_kv_computed2 == 0  # All cached, nothing computed
    assert finished_req2.request_id == "test-req-004"


def test_prompt_token_stats_all_computed():
    """Test all tokens computed locally, no caching."""
    stats = PromptTokenStats()

    # Case 1: No caching (All tokens computed locally)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1000
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 0
    assert stats.total == 1000


def test_prompt_token_stats_partial_local_cache():
    """Test partial local prefix cache hit."""
    stats = PromptTokenStats()

    # Case 2: Partial local cache
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=300,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 700
    assert stats.local_cache_hit == 300
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 300
    assert stats.total == 1000


def test_prompt_token_stats_partial_external_transfer():
    """Test partial external KV transfer."""
    stats = PromptTokenStats()

    # Case 3: Partial external transfer
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=500,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 500
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 500
    assert stats.cached_tokens == 500
    assert stats.total == 1000


def test_prompt_token_stats_mixed_sources():
    """Test mix of local cache and external transfer."""
    stats = PromptTokenStats()

    # Case 4: Mixed sources
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=400,
        num_external_cached_tokens=200,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200
    assert stats.cached_tokens == 600
    assert stats.total == 1000


def test_prompt_token_stats_full_local_cache_recompute():
    """Test full local cache triggers last token recomputation.

    When all tokens are cached, the scheduler forces the model to recompute
    the last token (num_computed_tokens=1), with the rest from cache.
    """
    stats = PromptTokenStats()

    # Case 5: Full local cache (999 cached, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=999,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 999
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 999
    assert stats.total == 1000


def test_prompt_token_stats_full_external_transfer_recompute():
    """Test full external transfer triggers last token recomputation."""
    stats = PromptTokenStats()

    # Case 6: Full external transfer (999 from external, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=999,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 999
    assert stats.cached_tokens == 999
    assert stats.total == 1000
