# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.micro_slicing import (
    MicroSlicingController,
    MicroSlicingSettings,
)
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


def _settings(**kwargs) -> MicroSlicingSettings:
    defaults = {"max_num_prefill_tokens_per_step": 4096}
    defaults.update(kwargs)
    return MicroSlicingSettings(**defaults)


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
        num_tokens=8,
        max_tokens=200,
        req_ids=[req_id],
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    _update(scheduler, output)
    assert not request.is_prefill_chunk
    return request


def _create_micro_scheduler(opt_model_path, **kwargs):
    scheduler_kwargs: dict[str, Any] = {
        "max_num_batched_tokens": 8192,
        "max_model_len": 32768,
        "max_num_prefill_tokens_per_step": 4096,
    }
    scheduler_kwargs.update(kwargs)
    return create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        fairness_engine="micro_slicing",
        **scheduler_kwargs,
    )


def test_selector_requires_only_selected_engine_options():
    with pytest.raises(ValueError, match="fairness_engine must be selected"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            prefill_compute_share=0.5,
        )
    with pytest.raises(ValueError, match="micro-slicing options"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            fairness_engine="compute_share",
            prefill_compute_share=0.5,
            max_num_prefill_tokens_per_step=64,
        )
    with pytest.raises(ValueError, match="prefill_compute_share cannot"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            fairness_engine="micro_slicing",
            prefill_compute_share=0.5,
            max_num_prefill_tokens_per_step=64,
        )


def test_micro_slicing_validation_and_cli_contract():
    with pytest.raises(ValueError, match="max_num_prefill_tokens_per_step"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            fairness_engine="micro_slicing",
        )
    with pytest.raises(ValueError, match="decode_prefill_max_wait_ms"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            fairness_engine="micro_slicing",
            max_num_prefill_tokens_per_step=64,
            decode_prefill_max_wait_ms=100,
        )

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    namespace = parser.parse_args(
        [
            "--fairness-engine",
            "micro_slicing",
            "--max-num-prefill-tokens-per-step",
            "4096",
            "--max-num-partial-prefills",
            "2",
            "--decode-prefill-min-decode-steps",
            "4",
            "--decode-prefill-max-wait-ms",
            "250",
        ]
    )
    args = EngineArgs.from_cli_args(namespace)
    assert args.fairness_engine == "micro_slicing"
    assert args.max_num_prefill_tokens_per_step == 4096
    assert args.max_num_partial_prefills == 2
    assert args.decode_prefill_min_decode_steps == 4
    assert args.decode_prefill_max_wait_ms == 250


def test_controller_distributes_full_budget_and_rotates():
    controller = MicroSlicingController(_settings(max_num_prefill_tokens_per_step=8))
    assert controller.select_running_limits(["a", "b", "c"], 8) == {
        "a": 3,
        "b": 3,
        "c": 2,
    }
    assert controller.select_running_limits(["a", "b", "c"], 2) == {
        "c": 1,
        "a": 1,
    }


def test_controller_counts_decode_before_waiter_arrives():
    controller = MicroSlicingController(_settings(decode_prefill_min_decode_steps=3))
    for _ in range(3):
        controller.observe_step(
            has_eligible_decode=True,
            had_pending_prefill=False,
            scheduled_decode_tokens=1,
            scheduled_prefill_tokens=0,
            deadline_bypass_candidate=False,
        )
    defer, bypass = controller.should_defer_prefill(
        has_eligible_decode=True,
        has_pending_prefill=True,
        oldest_waiter_age_ms=0,
    )
    assert not defer
    assert not bypass


def test_deadline_metric_requires_actual_prefill_service():
    controller = MicroSlicingController(
        _settings(
            decode_prefill_min_decode_steps=4,
            decode_prefill_max_wait_ms=100,
        )
    )
    defer, bypass = controller.should_defer_prefill(
        has_eligible_decode=True,
        has_pending_prefill=True,
        oldest_waiter_age_ms=101,
    )
    assert not defer and bypass
    controller.observe_step(
        has_eligible_decode=True,
        had_pending_prefill=True,
        scheduled_decode_tokens=1,
        scheduled_prefill_tokens=0,
        deadline_bypass_candidate=bypass,
    )
    assert controller.fairness_bypasses == 0


def test_blocked_waiter_budget_is_reclaimed_by_running_prefills(opt_model_path):
    scheduler = _create_micro_scheduler(
        opt_model_path,
        max_num_partial_prefills=2,
    )
    _establish_decode(scheduler)
    prefills = create_requests(
        num_requests=2,
        num_tokens=12000,
        req_ids=["p0", "p1"],
    )
    for request in prefills:
        scheduler.add_request(request)
    first = scheduler.schedule()
    _update(scheduler, first)
    second = scheduler.schedule()
    _update(scheduler, second)
    assert scheduler.num_active_local_partial_prefills == 2

    (waiter,) = create_requests(
        num_requests=1,
        num_tokens=12000,
        req_ids=["waiter"],
    )
    scheduler.add_request(waiter)
    output = scheduler.schedule()

    assert waiter in scheduler.waiting
    assert sum(output.num_scheduled_tokens[req.request_id] for req in prefills) == 4096
    assert output.num_scheduled_tokens["decode"] > 0


def test_partial_cap_allows_waiter_that_finishes_in_quantum(opt_model_path):
    scheduler = _create_micro_scheduler(
        opt_model_path,
        max_num_partial_prefills=1,
    )
    _establish_decode(scheduler)
    (partial,) = create_requests(
        num_requests=1,
        num_tokens=12000,
        req_ids=["partial"],
    )
    scheduler.add_request(partial)
    first = scheduler.schedule()
    _update(scheduler, first)
    assert scheduler.num_active_local_partial_prefills == 1

    (finisher,) = create_requests(
        num_requests=1,
        num_tokens=1024,
        req_ids=["finisher"],
    )
    scheduler.add_request(finisher)
    output = scheduler.schedule()

    assert finisher in scheduler.running
    assert output.num_scheduled_tokens[finisher.request_id] == 1024


def test_decode_burst_and_deadline_use_actual_bypass(opt_model_path):
    scheduler = _create_micro_scheduler(
        opt_model_path,
        decode_prefill_min_decode_steps=2,
        decode_prefill_max_wait_ms=100,
    )
    _establish_decode(scheduler)
    (waiter,) = create_requests(
        num_requests=1,
        num_tokens=12000,
        req_ids=["waiter"],
    )
    waiter.arrival_time = time.time() - 1.0
    scheduler.add_request(waiter)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[waiter.request_id] == 4096
    assert scheduler.micro_slicing_controller is not None
    assert scheduler.micro_slicing_controller.fairness_bypasses == 1


def test_idle_policy_switch_replaces_controller_without_touching_cache(opt_model_path):
    scheduler = create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        max_num_batched_tokens=8192,
        max_model_len=32768,
    )
    cache_manager = scheduler.kv_cache_manager

    compute = scheduler.set_prefill_fairness(
        {
            "fairness_engine": "compute_share",
            "prefill_compute_share": 0.65,
        }
    )
    assert compute["fairness_engine"] == "compute_share"
    assert scheduler.compute_share_controller is not None
    assert scheduler.micro_slicing_controller is None

    micro = scheduler.set_prefill_fairness(
        {
            "fairness_engine": "micro_slicing",
            "max_num_prefill_tokens_per_step": 4096,
            "max_num_partial_prefills": 2,
            "decode_prefill_min_decode_steps": 3,
            "decode_prefill_max_wait_ms": 250,
        }
    )
    assert micro["fairness_engine"] == "micro_slicing"
    assert scheduler.compute_share_controller is None
    assert scheduler.micro_slicing_controller is not None

    disabled = scheduler.set_prefill_fairness({"fairness_engine": None})
    assert disabled["fairness_engine"] is None
    assert scheduler.compute_share_controller is None
    assert scheduler.micro_slicing_controller is None
    assert scheduler.kv_cache_manager is cache_manager


def test_invalid_live_policy_does_not_replace_current_controller(opt_model_path):
    scheduler = _create_micro_scheduler(opt_model_path)
    controller = scheduler.micro_slicing_controller
    scheduler.prefill_fairness_quantum = 4096
    with pytest.raises(ValueError, match="prefill quantum"):
        scheduler.set_prefill_fairness(
            {
                "fairness_engine": "micro_slicing",
                "max_num_prefill_tokens_per_step": 4095,
            }
        )
    assert scheduler.micro_slicing_controller is controller


def test_async_external_restore_does_not_consume_local_partial_slot(opt_model_path):
    scheduler = _create_micro_scheduler(
        opt_model_path,
        max_num_batched_tokens=128,
        max_model_len=128,
        max_num_prefill_tokens_per_step=64,
        max_num_partial_prefills=1,
        enable_prefix_caching=True,
        block_size=16,
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
    assert scheduler.num_active_local_partial_prefills == 0
    assert request in scheduler._inflight_prefills


@pytest.mark.parametrize("prefill_budget", [2048, 4096])
def test_split_cache_geometry_allocates_usable_prefill_slices(
    opt_model_path, monkeypatch, prefill_budget
):
    """Three prefills must not divide a usable budget into sub-block slices."""
    from functools import partial

    import torch

    from vllm.config import CacheConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import MambaSpec

    from . import utils

    monkeypatch.setattr(
        utils, "CacheConfig", partial(CacheConfig, mamba_cache_mode="align")
    )
    scheduler = _create_micro_scheduler(
        opt_model_path,
        block_size=2048,
        kv_cache_spec=MambaSpec(
            block_size=256,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        max_num_prefill_tokens_per_step=prefill_budget,
    )
    requests = create_requests(3, num_tokens=16384, block_size=256)
    controller = scheduler.micro_slicing_controller
    assert controller is not None
    limits = controller.select_running_limits(
        [request.request_id for request in requests], prefill_budget
    )
    actual = [
        Scheduler._mamba_block_aligned_split(
            scheduler, request, limits[request.request_id]
        )
        for request in requests
        if request.request_id in limits
    ]
    assert actual and all(tokens > 0 for tokens in actual)
    assert sum(actual) == prefill_budget


def test_split_cache_geometry_rejects_unusable_micro_budget(
    opt_model_path, monkeypatch
):
    from functools import partial

    import torch

    from vllm.config import CacheConfig
    from vllm.v1.kv_cache_interface import MambaSpec

    from . import utils

    monkeypatch.setattr(
        utils, "CacheConfig", partial(CacheConfig, mamba_cache_mode="align")
    )
    with pytest.raises(ValueError, match="scheduler prefill quantum"):
        _create_micro_scheduler(
            opt_model_path,
            block_size=2048,
            kv_cache_spec=MambaSpec(
                block_size=256,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            max_num_prefill_tokens_per_step=2304,
        )


@pytest.mark.parametrize("global_budget,long_threshold", [(1024, 0), (8192, 1024)])
def test_split_cache_geometry_preserves_global_sub_block_progress(
    opt_model_path, monkeypatch, global_budget, long_threshold
):
    from functools import partial

    import torch

    from vllm.config import CacheConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import MambaSpec

    from . import utils

    monkeypatch.setattr(
        utils, "CacheConfig", partial(CacheConfig, mamba_cache_mode="align")
    )
    scheduler = _create_micro_scheduler(
        opt_model_path,
        block_size=2048,
        kv_cache_spec=MambaSpec(
            block_size=256,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        max_num_batched_tokens=global_budget,
        long_prefill_token_threshold=long_threshold,
        max_num_prefill_tokens_per_step=1024,
    )
    (request,) = create_requests(1, num_tokens=16384, block_size=256)
    controller = scheduler.micro_slicing_controller
    assert controller is not None
    limits = controller.select_running_limits([request.request_id], 1024)
    assert (
        Scheduler._mamba_block_aligned_split(
            scheduler, request, limits[request.request_id]
        )
        == 1024
    )
