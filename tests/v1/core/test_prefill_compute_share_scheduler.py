# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def opt_model_path(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt",'
        '"max_position_embeddings":32768}'
    )
    return str(tmp_path)


def _create_fair_scheduler(opt_model_path, **kwargs):
    return create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        prefill_compute_share=0.5,
        device="cpu",
        **kwargs,
    )


def _update(scheduler, output) -> None:
    req_ids = list(output.num_scheduled_tokens)
    sampled_token_ids = [
        [] if scheduler.requests[req_id].is_prefill_chunk else [0] for req_id in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _establish_decode(scheduler, req_id: str = "decode"):
    (request,) = create_requests(
        num_requests=1,
        num_tokens=4,
        max_tokens=100,
        req_ids=[req_id],
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.compute_service_class == "prefill"
    assert not output.compute_contention
    _update(scheduler, output)
    assert not request.is_prefill_chunk
    return request


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.1])
def test_prefill_compute_share_rejects_invalid_values(share: float):
    with pytest.raises(ValueError, match="prefill_compute_share"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share=share,
        )


def test_prefill_compute_share_cli_contract():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    namespace = parser.parse_args(["--prefill-compute-share", "0.65"])
    engine_args = EngineArgs.from_cli_args(namespace)

    assert engine_args.prefill_compute_share == pytest.approx(0.65)


def test_prefill_compute_share_rejects_fixed_cadence():
    with pytest.raises(ValueError, match="prefill_compute_share"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share=0.5,
            prefill_schedule_interval=2,
        )


def test_prefill_compute_share_rejects_unsynchronized_dp(opt_model_path):
    with pytest.raises(ValueError, match="synchronized service-class decision"):
        _create_fair_scheduler(opt_model_path, data_parallel_size=2)


def test_disabled_scheduler_emits_no_compute_policy(opt_model_path):
    scheduler = create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
    )
    (request,) = create_requests(num_requests=1, req_ids=["legacy"])
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert scheduler.compute_share_controller is None
    assert output.compute_service_class is None
    assert not output.compute_contention


def test_decode_tie_then_prefill_and_work_conserving_fallback(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=4,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    assert decode_output.compute_service_class == "decode"
    assert decode_output.compute_contention
    assert set(decode_output.num_scheduled_tokens) == {"decode"}
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)

    prefill_output = scheduler.schedule()
    assert prefill_output.compute_service_class == "prefill"
    assert prefill_output.compute_contention
    assert "prefill" in prefill_output.num_scheduled_tokens
    # The selected short prefill cannot fill the batch, so decode consumes the
    # genuinely remaining capacity in the same model step.
    assert "decode" in prefill_output.num_scheduled_tokens


def test_prefill_fcfs_and_capacity_bound_does_not_bypass_share(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=128,
    )
    _establish_decode(scheduler)
    first, second = create_requests(
        num_requests=2,
        num_tokens=16,
        req_ids=["first", "second"],
    )
    scheduler.add_request(first)
    scheduler.add_request(second)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)

    # The legacy cadence escape must have no effect in adaptive mode.
    scheduler.prefill_capacity_bound = True
    prefill_output = scheduler.schedule()
    assert prefill_output.compute_service_class == "prefill"
    assert prefill_output.num_scheduled_tokens == {"first": 8}
    assert second in scheduler.waiting


def test_running_prefills_retain_fcfs_order(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=128,
        long_prefill_token_threshold=8,
    )
    _establish_decode(scheduler)
    first, second = create_requests(
        num_requests=2,
        num_tokens=32,
        req_ids=["first", "second"],
    )
    scheduler.add_request(first)
    scheduler.add_request(second)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)
    first_prefill_output = scheduler.schedule()
    assert list(first_prefill_output.num_scheduled_tokens) == ["first", "second"]
    scheduler.record_compute_time("prefill", 0.01, contended=True)
    _update(scheduler, first_prefill_output)

    second_decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, second_decode_output)
    second_prefill_output = scheduler.schedule()

    assert list(second_prefill_output.num_scheduled_tokens) == ["first", "second"]


def test_prefill_turn_falls_back_when_prefill_is_alignment_blocked(
    opt_model_path, monkeypatch
):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["blocked-prefill"],
    )
    scheduler.add_request(prefill)
    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)

    scheduler.need_mamba_block_aligned_split = True

    def block_prefill(request, num_new_tokens, *_args):
        return 0 if request.request_id == prefill.request_id else num_new_tokens

    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", block_prefill)
    output = scheduler.schedule()

    assert output.compute_service_class == "decode"
    assert output.num_scheduled_tokens == {"decode": 1}


def test_decode_turn_falls_back_when_decode_is_no_longer_runnable(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    decode = _establish_decode(scheduler)
    # Model the async guard that can make a coarsely eligible decode unable to
    # consume another step after the class decision has already been made.
    decode.num_output_placeholders = 1
    decode.max_tokens = 0
    decode.sampling_params.max_tokens = 0
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["fallback-prefill"],
    )
    scheduler.add_request(prefill)

    output = scheduler.schedule()

    assert output.compute_service_class == "prefill"
    assert output.num_scheduled_tokens == {prefill.request_id: 16}


def test_decode_turn_falls_back_to_running_prefill(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=16,
        max_model_len=128,
        long_prefill_token_threshold=8,
    )
    decode = _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["running-prefill"],
    )
    scheduler.add_request(prefill)
    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)
    prefill_output = scheduler.schedule()
    scheduler.record_compute_time("prefill", 0.01, contended=True)
    _update(scheduler, prefill_output)
    assert prefill.is_prefill_chunk

    decode.num_output_placeholders = 1
    decode.max_tokens = 0
    decode.sampling_params.max_tokens = 0
    fallback_output = scheduler.schedule()

    assert fallback_output.compute_service_class == "prefill"
    assert fallback_output.num_scheduled_tokens == {prefill.request_id: 8}


def test_full_apc_hit_is_decode_not_prefill(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=128,
    )
    warm, hit = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=1,
        req_ids=["warm", "hit"],
    )
    scheduler.add_request(warm)
    warm_output = scheduler.schedule()
    _update(scheduler, warm_output)
    assert warm.request_id in scheduler.finished_req_ids

    _establish_decode(scheduler)
    scheduler.add_request(hit)
    hit_output = scheduler.schedule()
    assert hit_output.compute_service_class == "decode"
    assert hit_output.num_scheduled_tokens["hit"] == 1


def test_async_external_load_does_not_consume_service_credit(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=128,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["restore"],
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert output.compute_service_class is None
    assert not output.compute_contention
    controller = scheduler.compute_share_controller
    assert controller is not None
    assert controller.decode_virtual_runtime == 0.0
    assert controller.prefill_virtual_runtime == 0.0


def test_partial_external_hit_is_prefill_compute(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=128,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=False),
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["partial-restore"],
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.compute_service_class == "prefill"
    assert output.num_scheduled_tokens[request.request_id] == 32


def test_only_contended_compute_is_exported_and_stats_drain(opt_model_path):
    scheduler = _create_fair_scheduler(opt_model_path)

    scheduler.record_compute_time("decode", 1.0, contended=False)
    scheduler.record_compute_time("decode", 0.25, contended=True)
    scheduler.record_compute_time("prefill", 0.75, contended=True)
    stats = scheduler.make_stats()

    assert stats is not None
    assert stats.decode_compute_seconds == pytest.approx(0.25)
    assert stats.prefill_compute_seconds == pytest.approx(0.75)
    assert stats.prefill_compute_share == pytest.approx(0.5)

    drained = scheduler.make_stats()
    assert drained is not None
    assert drained.decode_compute_seconds == 0.0
    assert drained.prefill_compute_seconds == 0.0


def test_speculative_decode_preserves_selected_prefill_quantum(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=16,
        max_model_len=128,
        num_speculative_tokens=3,
    )
    _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["spec-prefill"],
    )
    scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)
    prefill_output = scheduler.schedule()

    assert prefill_output.compute_service_class == "prefill"
    assert prefill_output.num_scheduled_tokens[prefill.request_id] == 16
    assert "decode" not in prefill_output.num_scheduled_tokens


def test_async_scheduler_preserves_compute_class_selection(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        async_scheduling=True,
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["async-prefill"],
    )
    scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    assert decode_output.compute_service_class == "decode"
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)
    prefill_output = scheduler.schedule()

    assert prefill_output.compute_service_class == "prefill"
    assert prefill_output.num_scheduled_tokens[prefill.request_id] == 16
