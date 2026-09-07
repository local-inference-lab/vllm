# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.scheduler import (
    should_defer_draft_for_partial_packed_dcp_resume,
)
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("cache_dtype", "cached_tokens", "num_scheduled_tokens", "expected"),
    [
        ("fp8_ds_mla", 4608, 440, True),
        ("fp8_ds_mla", 12_288, 440, False),
        ("fp8_ds_mla", 0, 440, False),
        ("fp8", 4608, 440, False),
        ("fp8_ds_mla", 4608, 439, False),
    ],
)
def test_partial_packed_dcp_resume_draft_deferral(
    cache_dtype: str,
    cached_tokens: int,
    num_scheduled_tokens: int,
    expected: bool,
) -> None:
    assert (
        should_defer_draft_for_partial_packed_dcp_resume(
            cache_dtype=cache_dtype,
            dcp_size=8,
            block_size=1536,
            cached_tokens=cached_tokens,
            num_computed_tokens=4608,
            num_scheduled_tokens=num_scheduled_tokens,
            num_prompt_tokens=5048,
        )
        is expected
    )


def test_async_scheduler_emits_no_placeholder_after_partial_dcp_hit() -> None:
    scheduler = create_scheduler(
        model="/model",
        skip_tokenizer_init=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        async_scheduling=True,
        use_v2_model_runner=True,
        enable_prefix_caching=True,
        block_size=16,
        max_num_batched_tokens=128,
    )
    scheduler.cache_config.cache_dtype = "fp8_ds_mla"
    scheduler.dcp_world_size = 8
    warm, resumed = create_requests(
        num_requests=2,
        num_tokens=65,
        same_prompt=True,
        max_tokens=1,
        block_size=16,
    )

    scheduler.add_request(warm)
    warm_output = scheduler.schedule()
    scheduler.update_from_output(
        warm_output,
        ModelRunnerOutput(
            req_ids=[warm.request_id],
            req_id_to_index={warm.request_id: 0},
            sampled_token_ids=[[100]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    scheduler.add_request(resumed)
    resumed_output = scheduler.schedule()

    assert resumed_output.num_scheduled_tokens[resumed.request_id] == 1
    assert resumed_output.num_spec_tokens_to_schedule == 0
    assert resumed.spec_token_ids == []
