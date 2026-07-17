# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.warmup import (
    _derive_dspark_draft_token_budget,
    _stable_sps_step_ms,
    run_mixed_prefill_decode_warmup,
)


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


def test_sps_profile_median_rejects_single_stall():
    assert _stable_sps_step_ms([10.0, 10.2, 75.0, 9.9, 10.1]) == pytest.approx(10.1)


def test_dspark_dynamic_budget_uses_upper_load_sps_knee():
    curve = [
        (6, 200.0),
        (12, 80.0),
        (24, 67.71),
        (48, 58.52),
        (96, 51.29),
        (192, 40.33),
        (384, 23.72),
    ]

    assert _derive_dspark_draft_token_budget(curve, max_draft_depth=5) == 160


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )
