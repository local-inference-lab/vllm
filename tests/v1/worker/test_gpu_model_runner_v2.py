# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for V2 structured-output/speculative-decode transport."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, _get_invalid_draft_counts


def _grammar_output() -> GrammarOutput:
    return GrammarOutput(
        structured_output_request_ids=["structured-b", "structured-a"],
        grammar_bitmask=np.empty((0, 0), dtype=np.int32),
        num_spec_tokens=[4, 3],
        has_bonus_token=[True, True],
        num_invalid_spec_tokens=[3, 1],
    )


def _grammar_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["plain", "structured-a", "structured-b"],
        num_draft_tokens_per_req=np.array([0, 2, 3], dtype=np.int32),
        cu_num_logits_np=np.array([0, 1, 4, 8], dtype=np.int32),
    )


def test_invalid_grammar_suffix_counts_follow_reorder_and_compaction():
    assert _get_invalid_draft_counts(_grammar_output(), _grammar_input_batch()) == [
        0,
        0,
        2,
    ]
    new_request_batch = SimpleNamespace(
        req_ids=["new"],
        num_draft_tokens_per_req=np.array([2], dtype=np.int32),
    )
    assert _get_invalid_draft_counts(_grammar_output(), new_request_batch) is None


@pytest.mark.parametrize(
    "source_width,source_invalid,active_width,expected",
    [
        (4, 0, 4, None),
        (4, 4, 4, [4]),
        (4, 3, 2, [1]),
        (4, 3, 1, None),
    ],
)
def test_invalid_grammar_suffix_counts_project_active_width(
    source_width, source_invalid, active_width, expected
):
    grammar_output = GrammarOutput(
        structured_output_request_ids=["structured"],
        grammar_bitmask=np.empty((0, 0), dtype=np.int32),
        num_spec_tokens=[source_width],
        has_bonus_token=[True],
        num_invalid_spec_tokens=[source_invalid],
    )
    input_batch = SimpleNamespace(
        req_ids=["structured"],
        num_draft_tokens_per_req=np.array([active_width], dtype=np.int32),
    )

    assert _get_invalid_draft_counts(grammar_output, input_batch) == expected


@pytest.mark.parametrize(
    "request_ids,widths,invalid,active_widths,error",
    [
        (["a", "a"], [1, 1], [0, 0], [1], "duplicate"),
        (["a"], [1, 2], [0], [1], "align"),
        (["a"], [1], None, [1], "missing"),
        (["a"], [1], [2], [1], "outside"),
        (["a"], [1], [0], [2], "active"),
    ],
)
def test_invalid_grammar_suffix_counts_validate_metadata(
    request_ids, widths, invalid, active_widths, error
):
    grammar_output = GrammarOutput(
        structured_output_request_ids=request_ids,
        grammar_bitmask=np.empty((0, 0), dtype=np.int32),
        num_spec_tokens=widths,
        has_bonus_token=[True] * len(request_ids),
        num_invalid_spec_tokens=invalid,
    )
    input_batch = SimpleNamespace(
        req_ids=["a"],
        num_draft_tokens_per_req=np.array(active_widths, dtype=np.int32),
    )

    with pytest.raises(ValueError, match=error):
        _get_invalid_draft_counts(grammar_output, input_batch)


def test_sample_forwards_projected_grammar_rejections():
    input_batch = _grammar_input_batch()
    input_batch.num_draft_tokens = 5
    input_batch.logits_indices = torch.arange(8)
    input_batch.max_query_len = 4
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = SimpleNamespace(compute_logits=lambda _: torch.zeros(8, 4))
    runner.structured_outputs_worker = SimpleNamespace(
        apply_grammar_bitmask=lambda *_: None
    )
    runner.rejection_sampler = Mock(
        return_value=SimpleNamespace(num_sampled=None, num_rejected=None)
    )
    runner.speculator = SimpleNamespace(draft_logits=None, online_sts=None)
    runner.verification_capacity_manager = None

    GPUModelRunner.sample(runner, torch.zeros(8, 1), input_batch, _grammar_output())

    assert runner.rejection_sampler.call_args.args[3] == [0, 0, 2]


def test_sample_without_grammar_forwards_no_invalid_counts():
    input_batch = _grammar_input_batch()
    input_batch.num_draft_tokens = 5
    input_batch.logits_indices = torch.arange(8)
    input_batch.max_query_len = 4
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = SimpleNamespace(compute_logits=lambda _: torch.zeros(8, 4))
    runner.rejection_sampler = Mock(
        return_value=SimpleNamespace(num_sampled=None, num_rejected=None)
    )
    runner.speculator = SimpleNamespace(draft_logits=None, online_sts=None)
    runner.verification_capacity_manager = None

    GPUModelRunner.sample(runner, torch.zeros(8, 1), input_batch, None)

    assert runner.rejection_sampler.call_args.args[3] is None
