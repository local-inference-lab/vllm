# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

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
    prefill_compute_share = kwargs.pop("prefill_compute_share", 0.5)
    prefill_compute_half_life = kwargs.pop("prefill_compute_half_life", None)
    return create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        prefill_compute_share=prefill_compute_share,
        prefill_compute_half_life=prefill_compute_half_life,
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
    assert output.compute_service_class is None
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


def test_prefill_compute_share_cli_accepts_auto():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    namespace = parser.parse_args(["--prefill-compute-share", "auto"])
    engine_args = EngineArgs.from_cli_args(namespace)

    assert engine_args.prefill_compute_share == "auto"


def test_prefill_interleave_cli_contract():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    namespace = parser.parse_args(
        [
            "--max-parallel-prefills",
            "auto",
            "--prefill-policy",
            "decode-aware",
            "--decode-refill-target",
            "8",
        ]
    )
    engine_args = EngineArgs.from_cli_args(namespace)

    assert engine_args.max_parallel_prefills == "auto"
    assert engine_args.prefill_policy == "decode-aware"
    assert engine_args.decode_refill_target == 8


def test_prefill_interleave_cli_accepts_numeric_parallel_prefills():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    namespace = parser.parse_args(["--max-parallel-prefills", "4"])
    engine_args = EngineArgs.from_cli_args(namespace)

    assert engine_args.max_parallel_prefills == 4


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("smooth", "smooth"), ("responsive", "responsive"), ("5", 5.0), ("0.2", 0.2)],
)
def test_prefill_compute_half_life_cli_contract(setting, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    namespace = parser.parse_args(
        [
            "--prefill-compute-share",
            "auto",
            "--prefill-compute-half-life",
            setting,
        ]
    )
    engine_args = EngineArgs.from_cli_args(namespace)

    assert engine_args.prefill_compute_half_life == expected


@pytest.mark.parametrize("setting", ["0", "-1", "nan", "inf", "fast"])
def test_prefill_compute_half_life_cli_rejects_invalid_values(setting):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--prefill-compute-share",
                "auto",
                "--prefill-compute-half-life",
                setting,
            ]
        )


def test_prefill_compute_share_rejects_unknown_mode():
    with pytest.raises(ValueError, match="prefill_compute_share"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share="fast",
        )


@pytest.mark.parametrize("half_life", [0.0, -1.0, float("inf"), float("nan")])
def test_prefill_compute_half_life_rejects_invalid_values(half_life):
    with pytest.raises(ValueError, match="prefill_compute_half_life"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share="auto",
            prefill_compute_half_life=half_life,
        )


@pytest.mark.parametrize("share", [None, 0.4])
def test_prefill_compute_half_life_requires_auto(share):
    with pytest.raises(ValueError, match="requires prefill_compute_share='auto'"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share=share,
            prefill_compute_half_life="responsive",
        )


def test_auto_mode_receives_scheduler_cost_feedback(opt_model_path):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        prefill_compute_share="auto",
        prefill_compute_half_life="smooth",
        max_num_batched_tokens=16,
        max_model_len=128,
    )
    _establish_decode(scheduler)
    (prefill,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["auto-prefill"],
    )
    scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time(
        "decode",
        0.01,
        contended=True,
        scheduled_tokens=decode_output.compute_service_tokens,
    )
    _update(scheduler, decode_output)
    prefill_output = scheduler.schedule()
    scheduler.record_compute_time(
        "prefill",
        0.16,
        contended=True,
        scheduled_tokens=prefill_output.compute_service_tokens,
    )

    controller = scheduler.compute_share_controller
    assert controller is not None
    assert prefill_output.compute_service_tokens == 16
    assert controller.prefill_seconds_per_token == pytest.approx(0.01)
    config = scheduler.get_prefill_fairness()
    assert config["prefill_compute_share"] == "auto"
    assert config["prefill_compute_half_life"] == "smooth"
    assert config["effective_prefill_compute_half_life_seconds"] == pytest.approx(2.0)
    assert config["effective_prefill_compute_share"] == pytest.approx(0.4)


def test_prefill_compute_share_rejects_fixed_cadence():
    with pytest.raises(ValueError, match="prefill_compute_share"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share=0.5,
            prefill_schedule_interval=2,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_parallel_prefills": 5, "max_num_seqs": 4},
        {"max_parallel_prefills": "auto", "enable_chunked_prefill": False},
        {"max_parallel_prefills": 1, "prefill_policy": "decode-aware"},
        {"decode_refill_target": 2},
        {
            "max_parallel_prefills": 2,
            "prefill_policy": "decode-aware",
            "decode_refill_target": 5,
            "max_num_seqs": 4,
        },
    ],
)
def test_prefill_interleave_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            **kwargs,
        )


def test_prefill_compute_share_rejects_unsynchronized_dp(opt_model_path):
    with pytest.raises(ValueError, match="synchronized fairness decision"):
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
    assert not output.compute_timing_enabled
    assert not output.compute_contention


def test_auto_compute_share_does_not_scan_or_time_uncontended_decode(
    opt_model_path, monkeypatch
):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        prefill_compute_share="auto",
        max_parallel_prefills="auto",
    )
    decode = _establish_decode(scheduler)
    decode.num_output_placeholders = 1
    monkeypatch.setattr(
        scheduler,
        "_request_is_runnable_decode",
        lambda _request: pytest.fail("decode-only path scanned runnable decodes"),
    )
    monkeypatch.setattr(
        scheduler.prefill_interleave_controller,
        "begin_step",
        lambda **_kwargs: pytest.fail("decode-only path entered prefill policy"),
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {decode.request_id: 2}
    assert output.compute_service_class is None
    assert not output.compute_timing_enabled
    assert not output.compute_contention


def test_runtime_switch_is_live_and_preserves_inflight_accounting(opt_model_path):
    scheduler = create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        prefill_compute_share=0.4,
    )
    controller = scheduler.compute_share_controller
    assert controller is not None
    controller.last_prefill_seconds = 1.0
    controller.dispatch("prefill", contended=True)

    fixed = scheduler.set_prefill_fairness({"prefill_compute_share": 0.9})

    assert fixed["prefill_compute_share"] == pytest.approx(0.9)
    assert fixed["effective_prefill_compute_share"] == pytest.approx(0.9)
    assert scheduler.compute_share_controller is controller
    assert list(controller.prefill_reservations) == [(2.5, 0.4)]

    smooth = scheduler.set_prefill_fairness(
        {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "smooth",
        }
    )
    assert smooth["prefill_compute_share"] == "auto"
    assert smooth["prefill_compute_half_life"] == "smooth"
    assert smooth["effective_prefill_compute_half_life_seconds"] == pytest.approx(2.0)
    assert smooth["effective_prefill_compute_share"] == pytest.approx(0.8)

    responsive = scheduler.set_prefill_fairness(
        {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "responsive",
        }
    )
    assert responsive["prefill_compute_share"] == "auto"
    assert responsive["prefill_compute_half_life"] == "responsive"
    assert responsive["effective_prefill_compute_half_life_seconds"] == pytest.approx(
        0.5
    )
    assert responsive["effective_prefill_compute_share"] == pytest.approx(0.8)

    numeric_half_life = scheduler.set_prefill_fairness(
        {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": 0.2,
        }
    )
    assert numeric_half_life["prefill_compute_half_life"] == pytest.approx(0.2)
    assert numeric_half_life[
        "effective_prefill_compute_half_life_seconds"
    ] == pytest.approx(0.2)
    assert scheduler.compute_share_controller is controller
    assert list(controller.prefill_reservations) == [(2.5, 0.4)]

    disabled = scheduler.set_prefill_fairness({"prefill_compute_share": None})

    assert disabled["prefill_compute_share"] is None
    assert disabled["prefill_compute_half_life"] is None
    assert disabled["effective_prefill_compute_half_life_seconds"] is None
    assert disabled["effective_prefill_compute_share"] is None
    assert scheduler.compute_share_controller is None


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
        prefill_compute_share="auto",
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
    assert not output.compute_timing_enabled
    assert not output.compute_contention
    controller = scheduler.compute_share_controller
    assert controller is not None
    assert controller.decode_virtual_runtime == 0.0
    assert controller.prefill_virtual_runtime == 0.0
    assert controller.local_prefill_backlog_tokens == 0


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

    # The local remainder is prefill work, but there is no competing decode,
    # so the controller does not install timing on this model step.
    assert output.compute_service_class is None
    assert not output.compute_timing_enabled
    assert not output.compute_contention
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


@pytest.mark.parametrize(
    ("method", "max_num_batched_tokens", "num_speculative_tokens"),
    [("dspark", 8, 4), ("dflash", 20, 7)],
)
def test_separate_draft_input_budget_limits_prefill_fairness_turn(
    opt_model_path,
    method,
    max_num_batched_tokens,
    num_speculative_tokens,
):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=128,
        num_speculative_tokens=num_speculative_tokens,
    )
    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.method = method
    _establish_decode(scheduler)

    prefills = create_requests(
        num_requests=3,
        num_tokens=2,
        req_ids=["prefill-0", "prefill-1", "prefill-2"],
    )
    for prefill in prefills:
        scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)
    prefill_output = scheduler.schedule()

    assert prefill_output.compute_service_class == "prefill"
    assert prefill_output.num_scheduled_tokens == {
        "prefill-0": 2,
        "prefill-1": 2,
    }
    assert prefills[2] in scheduler.waiting


def test_dflash_budget_covers_running_prefills_before_waiting_admission(
    opt_model_path,
):
    scheduler = _create_fair_scheduler(
        opt_model_path,
        max_num_batched_tokens=20,
        max_model_len=128,
        num_speculative_tokens=7,
    )
    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.method = "dflash"
    _establish_decode(scheduler)
    scheduler.scheduler_config.long_prefill_token_threshold = 2

    prefills = create_requests(
        num_requests=3,
        num_tokens=32,
        req_ids=["prefill-0", "prefill-1", "prefill-2"],
    )
    for prefill in prefills:
        scheduler.add_request(prefill)

    decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, decode_output)

    first_prefill_output = scheduler.schedule()
    assert first_prefill_output.num_scheduled_tokens == {
        "prefill-0": 2,
        "prefill-1": 2,
    }
    scheduler.record_compute_time("prefill", 0.01, contended=True)
    _update(scheduler, first_prefill_output)

    second_decode_output = scheduler.schedule()
    scheduler.record_compute_time("decode", 0.01, contended=True)
    _update(scheduler, second_decode_output)
    second_prefill_output = scheduler.schedule()

    assert second_prefill_output.compute_service_class == "prefill"
    assert second_prefill_output.num_scheduled_tokens == {
        "prefill-0": 2,
        "prefill-1": 2,
    }
    assert prefills[2] in scheduler.waiting


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


def _create_interleaving_scheduler(opt_model_path, **kwargs):
    max_parallel_prefills = kwargs.pop("max_parallel_prefills", "auto")
    block_size = kwargs.pop("block_size", 4)
    max_num_seqs = kwargs.pop("max_num_seqs", 8)
    max_num_batched_tokens = kwargs.pop("max_num_batched_tokens", 16)
    max_model_len = kwargs.pop("max_model_len", 128)
    return create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        max_parallel_prefills=max_parallel_prefills,
        block_size=block_size,
        **kwargs,
    )


def test_parallel_prefill_preserves_solo_prefill_budget(opt_model_path):
    scheduler = _create_interleaving_scheduler(opt_model_path)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["solo"],
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"solo": 16}


def test_parallel_prefill_splits_budget_across_four_requests(opt_model_path):
    scheduler = _create_interleaving_scheduler(opt_model_path)
    requests = create_requests(
        num_requests=4,
        num_tokens=64,
        req_ids=["p0", "p1", "p2", "p3"],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        "p0": 4,
        "p1": 4,
        "p2": 4,
        "p3": 4,
    }


def test_parallel_prefill_redistributes_unused_short_share(opt_model_path):
    scheduler = _create_interleaving_scheduler(opt_model_path)
    (short,) = create_requests(num_requests=1, num_tokens=2, req_ids=["short"])
    long_requests = create_requests(
        num_requests=3,
        num_tokens=64,
        req_ids=["long0", "long1", "long2"],
    )
    for request in (short, *long_requests):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["short"] == 2
    assert sum(output.num_scheduled_tokens.values()) == 16
    assert sorted(
        output.num_scheduled_tokens[request.request_id] for request in long_requests
    ) == [4, 5, 5]


def test_round_robin_reaches_prefill_outside_first_batch(opt_model_path):
    scheduler = _create_interleaving_scheduler(opt_model_path)
    requests = create_requests(
        num_requests=5,
        num_tokens=64,
        req_ids=["p0", "p1", "p2", "p3", "p4"],
    )
    for request in requests:
        scheduler.add_request(request)

    first_output = scheduler.schedule()
    assert set(first_output.num_scheduled_tokens) == {"p0", "p1", "p2", "p3"}
    _update(scheduler, first_output)

    second_output = scheduler.schedule()

    assert "p4" in second_output.num_scheduled_tokens
    assert len(second_output.num_scheduled_tokens) == 4


def test_round_robin_prioritizes_never_scheduled_prefills(opt_model_path):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
    )
    (long_request,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["long"],
    )
    scheduler.add_request(long_request)
    first_output = scheduler.schedule()
    _update(scheduler, first_output)

    (short_request,) = create_requests(
        num_requests=1,
        num_tokens=8,
        req_ids=["short"],
    )
    (medium_request,) = create_requests(
        num_requests=1,
        num_tokens=24,
        req_ids=["medium"],
    )
    scheduler.add_request(short_request)
    scheduler.add_request(medium_request)

    output = scheduler.schedule()

    assert "long" not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens == {"short": 8, "medium": 8}


def test_decode_aware_reserves_one_lane_for_nearest_decode(opt_model_path):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
        prefill_policy="decode-aware",
    )
    (long_request,) = create_requests(num_requests=1, num_tokens=64, req_ids=["long"])
    (medium_request,) = create_requests(
        num_requests=1, num_tokens=24, req_ids=["medium"]
    )
    (short_request,) = create_requests(num_requests=1, num_tokens=8, req_ids=["short"])
    for request in (long_request, medium_request, short_request):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"short": 8, "long": 8}
    assert scheduler.get_prefill_fairness()["prefill_policy"] == "decode-aware"
    assert scheduler.get_prefill_fairness()["effective_decode_refill_target"] == 2


def test_decode_aware_uses_round_robin_with_healthy_reservoir(opt_model_path):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
        prefill_policy="decode-aware",
    )
    decodes = create_requests(
        num_requests=2,
        num_tokens=2,
        max_tokens=100,
        req_ids=["decode0", "decode1"],
    )
    for request in decodes:
        scheduler.add_request(request)
    initial = scheduler.schedule()
    _update(scheduler, initial)

    (long_request,) = create_requests(num_requests=1, num_tokens=64, req_ids=["long"])
    (medium_request,) = create_requests(
        num_requests=1, num_tokens=24, req_ids=["medium"]
    )
    (short_request,) = create_requests(num_requests=1, num_tokens=8, req_ids=["short"])
    for request in (long_request, medium_request, short_request):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert {"long", "medium"} <= output.num_scheduled_tokens.keys()
    assert "short" not in output.num_scheduled_tokens


def test_decode_aware_continues_when_no_prefill_needs_service(opt_model_path):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
        prefill_policy="decode-aware",
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=2,
        max_tokens=100,
        req_ids=["decode"],
    )
    scheduler.add_request(request)
    initial = scheduler.schedule()
    _update(scheduler, initial)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1}


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "block_size", "max_num_seqs", "expected"),
    [
        (4096, 256, 64, 4),
        (512, 256, 64, 4),
        (128, 256, 64, 4),
        (4096, 256, 2, 2),
    ],
)
def test_auto_parallel_prefills_ignore_token_geometry(
    opt_model_path,
    max_num_batched_tokens,
    block_size,
    max_num_seqs,
    expected,
):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max(max_num_batched_tokens, 4096),
    )

    assert scheduler.max_parallel_prefills == expected
    assert (
        scheduler.get_prefill_fairness()["effective_max_parallel_prefills"] == expected
    )


def test_default_path_skips_prefill_policy_scans(opt_model_path, monkeypatch):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=1,
        prefill_compute_share=None,
    )
    (request,) = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(request)

    monkeypatch.setattr(
        scheduler,
        "_request_is_runnable_decode",
        lambda _request: pytest.fail("default path scanned runnable decodes"),
    )
    monkeypatch.setattr(
        scheduler.prefill_interleave_controller,
        "begin_step",
        lambda **_kwargs: pytest.fail("default path entered prefill policy"),
    )

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 16


def test_prefill_interleave_preserves_request_priority(opt_model_path):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
        scheduling_policy="priority",
    )
    (high_priority_long,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["high-priority-long"],
    )
    high_priority_long.priority = 0
    (normal_priority_long,) = create_requests(
        num_requests=1,
        num_tokens=64,
        req_ids=["normal-priority-long"],
    )
    normal_priority_long.priority = 0
    (low_priority_short,) = create_requests(
        num_requests=1,
        num_tokens=8,
        req_ids=["low-priority-short"],
    )
    low_priority_short.priority = 10
    scheduler.add_request(low_priority_short)
    scheduler.add_request(high_priority_long)
    scheduler.add_request(normal_priority_long)

    output = scheduler.schedule()

    assert set(output.num_scheduled_tokens) == {
        "high-priority-long",
        "normal-priority-long",
    }


def test_async_restore_does_not_consume_parallel_prefill_lane(
    opt_model_path, monkeypatch
):
    scheduler = _create_interleaving_scheduler(
        opt_model_path,
        max_parallel_prefills=2,
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
    )
    assert scheduler.connector is not None

    def matched_tokens(request, _num_computed_tokens):
        return (32, True) if request.request_id == "restore" else (0, False)

    monkeypatch.setattr(
        scheduler.connector, "get_num_new_matched_tokens", matched_tokens
    )
    (restore,) = create_requests(
        num_requests=1,
        num_tokens=32,
        req_ids=["restore"],
    )
    local_requests = create_requests(
        num_requests=2,
        num_tokens=64,
        req_ids=["local0", "local1"],
    )
    scheduler.add_request(restore)
    for local in local_requests:
        scheduler.add_request(local)

    output = scheduler.schedule()

    assert restore.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert output.num_scheduled_tokens == {"local0": 8, "local1": 8}


def test_prefill_fairness_hot_switch_is_atomic(opt_model_path):
    scheduler = _create_fair_scheduler(opt_model_path)

    updated = scheduler.set_prefill_fairness(
        {
            "prefill_compute_share": "auto",
            "prefill_compute_half_life": "responsive",
        }
    )

    assert updated["prefill_compute_share"] == "auto"
    assert updated["prefill_compute_half_life"] == "responsive"
    with pytest.raises(ValueError, match="prefill_compute_share"):
        scheduler.set_prefill_fairness(
            {
                "prefill_compute_share": 1.0,
            }
        )

    unchanged = scheduler.get_prefill_fairness()
    assert unchanged["prefill_compute_share"] == "auto"
    assert unchanged["prefill_compute_half_life"] == "responsive"
