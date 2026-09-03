# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.structured_outputs import _build_grammar_row_mapping


def test_grammar_mapping_uses_active_width_row_for_bonus_after_zero_drafts():
    """A zero-draft request masks its bonus with the state after zero drafts."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["low", "high", "prefill"],
        grammar_req_ids=["low", "high", "prefill"],
        grammar_num_spec_tokens=[2, 2, 0],
        grammar_has_bonus_token=[True, True, True],
        cu_num_logits_np=np.array([0, 1, 2, 3], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 0, 0], dtype=np.int32),
        num_bonus_tokens=1,
    )

    assert source_indices == [0, 3, 6]
    assert logits_indices == [0, 1, 2]


def test_grammar_mapping_selects_active_drafts_from_each_source_group():
    """The bonus row tracks the active width, not the full serialized window."""
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["plain", "trimmed", "full"],
        grammar_req_ids=["trimmed", "full"],
        grammar_num_spec_tokens=[3, 3],
        grammar_has_bonus_token=[True, True],
        cu_num_logits_np=np.array([0, 1, 3, 7], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 1, 3], dtype=np.int32),
        num_bonus_tokens=1,
    )

    assert source_indices == [0, 1, 4, 5, 6, 7]
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


def test_grammar_mapping_mixed_active_widths_and_reordering():
    """k=0/k=1/k=K requests select their bonus row at the active width.

    Grammar groups arrive reordered relative to the batch and include the
    last request; the last group must not read past its source rows.
    """
    source_indices, logits_indices = _build_grammar_row_mapping(
        req_ids=["zero", "one", "full"],
        grammar_req_ids=["full", "zero", "one"],
        grammar_num_spec_tokens=[7, 7, 7],
        grammar_has_bonus_token=[True, True, True],
        cu_num_logits_np=np.array([0, 1, 3, 11], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 1, 7], dtype=np.int32),
        num_bonus_tokens=1,
    )

    # full: rows 0..6 drafts, bonus row 7; zero: bonus row 8; one: draft row 16,
    # bonus row 17.
    assert source_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17]
    assert logits_indices == [3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2]
