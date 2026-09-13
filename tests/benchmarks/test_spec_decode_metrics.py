# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.serve import (
    SpecDecodeMetrics,
    calculate_spec_decode_stats,
    fetch_spec_decode_metrics,
)


class _MetricsResponse:
    status = 200

    def __init__(self, body: str):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body


class _MetricsSession:
    def __init__(self, body: str):
        self.body = body

    def get(self, _url: str):
        return _MetricsResponse(self.body)


@pytest.mark.asyncio
async def test_fetch_spec_decode_metrics_separates_position_counters():
    body = """
vllm:spec_decode_num_drafts_total 4
vllm:spec_decode_num_draft_tokens_total 6
vllm:spec_decode_num_accepted_tokens_total 3
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 1
vllm:spec_decode_num_draft_tokens_per_pos_total{position="0"} 4
vllm:spec_decode_num_draft_tokens_per_pos_total{position="1"} 2
vllm:spec_decode_num_draft_tokens_per_pos_total{position="2"} 0
"""

    metrics = await fetch_spec_decode_metrics(
        "http://localhost:8000",
        _MetricsSession(body),  # type: ignore[arg-type]
    )

    assert metrics == SpecDecodeMetrics(
        num_drafts=4,
        num_draft_tokens=6,
        num_accepted_tokens=3,
        accepted_per_pos={0: 2, 1: 1},
        drafted_per_pos={0: 4, 1: 2, 2: 0},
    )


def test_adaptive_verification_uses_per_position_denominators():
    before = SpecDecodeMetrics(
        num_drafts=10,
        num_draft_tokens=20,
        num_accepted_tokens=10,
        accepted_per_pos={0: 8, 1: 2, 2: 0},
        drafted_per_pos={0: 10, 1: 5, 2: 0},
    )
    after = SpecDecodeMetrics(
        num_drafts=14,
        num_draft_tokens=26,
        num_accepted_tokens=13,
        accepted_per_pos={0: 10, 1: 3, 2: 0},
        drafted_per_pos={0: 14, 1: 7, 2: 0},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    assert stats["acceptance_rate"] == pytest.approx(50.0)
    assert stats["per_position_acceptance_rates"] == [0.5, 0.5, None]
