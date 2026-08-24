# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.structured_outputs import _build_grammar_row_mapping


def test_grammar_mapping_preserves_bonus_rows_after_zero_draft_budget():
    """Zero draft capacity retains each request's scheduled bonus mask."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["low", "high", "prefill"],
        grammar_req_ids=["low", "high", "prefill"],
        grammar_num_spec_tokens=[2, 2, 0],
        grammar_has_bonus_token=[True, True, True],
        cu_num_logits_np=np.array([0, 1, 2, 3], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 0, 0], dtype=np.int32),
        num_bonus_tokens=1,
    )

    assert source_indices == [2, 5, 6]
    assert logits_indices == [0, 1, 2]


def test_grammar_mapping_selects_active_drafts_from_each_source_group():
    """Compaction preserves per-request draft rows and the final bonus row."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["plain", "trimmed", "full"],
        grammar_req_ids=["trimmed", "full"],
        grammar_num_spec_tokens=[3, 3],
        grammar_has_bonus_token=[True, True],
        cu_num_logits_np=np.array([0, 1, 3, 7], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 1, 3], dtype=np.int32),
        num_bonus_tokens=1,
    )

    assert source_indices == [0, 3, 4, 5, 6, 7]
    assert logits_indices == [1, 2, 3, 4, 5, 6]


def test_grammar_mapping_supports_non_speculative_batches():
    """A batch without draft tokens maps one bonus row per grammar request."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["plain", "grammar"],
        grammar_req_ids=["grammar"],
        grammar_num_spec_tokens=[0],
        grammar_has_bonus_token=[True],
        cu_num_logits_np=np.array([0, 1, 2], dtype=np.int32),
        num_draft_tokens_per_req=None,
        num_bonus_tokens=1,
    )

    assert source_indices == [0]
    assert logits_indices == [1]


def test_grammar_mapping_advances_past_unmapped_diffusion_bonus_rows():
    """Serialized diffusion bonus rows advance source offsets without logits."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["diffusion", "no-canvas", "later"],
        grammar_req_ids=["diffusion", "no-canvas", "later"],
        grammar_num_spec_tokens=[1, 0, 2],
        grammar_has_bonus_token=[False, True, False],
        cu_num_logits_np=np.array([0, 1, 1, 3], dtype=np.int32),
        num_draft_tokens_per_req=np.array([1, 0, 2], dtype=np.int32),
        num_bonus_tokens=0,
    )

    assert source_indices == [0, 2, 3]
    assert logits_indices == [0, 1, 2]
