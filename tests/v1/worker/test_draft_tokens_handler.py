# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft hand-off must return the drafts a step's inputs consumed.

Under async scheduling the scheduler back-fills a step's draft placeholders
after that step's inputs were prepared. A request that skipped the step whose
drafts were proposed last must still get its own drafts back.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DraftTokensHandler copies on CUDA streams"
)


def _batch(req_ids, slots, num_drafts, structured=True):
    return SimpleNamespace(
        req_ids=req_ids,
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        num_draft_tokens_per_req=(
            None if num_drafts is None else np.array(num_drafts, dtype=np.int32)
        ),
        has_structured_output_reqs=structured,
    )


def _draft_slots():
    # One row of drafts per persistent request slot.
    return torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64, device="cuda"
    )


def test_consumed_drafts_include_requests_that_skipped_the_proposing_step():
    handler = DraftTokensHandler(torch.device("cuda"), track_consumed_drafts=True)
    drafts = _draft_slots()

    # The latest sampled step proposed drafts for slot 0 only.
    handler.set_draft_tokens(_batch(["a"], [0], [3]), drafts[:1])
    # The next step also verifies slot 2, which skipped the proposing step.
    handler.set_consumed_draft_tokens(_batch(["a", "b"], [0, 2], [3, 2]), drafts)
    # Proposals of the step being sampled must not replace the hand-off.
    handler.set_draft_tokens(_batch(["a", "b"], [0, 2], [3, 2]), drafts[:2])

    draft_token_ids = handler.get_draft_tokens()
    assert draft_token_ids.req_ids == ["a", "b"]
    assert draft_token_ids.draft_token_ids == [[1, 2, 3], [7, 8, 9]]


def test_step_without_drafts_clears_the_hand_off():
    handler = DraftTokensHandler(torch.device("cuda"), track_consumed_drafts=True)
    drafts = _draft_slots()
    handler.set_consumed_draft_tokens(_batch(["a"], [0], [3]), drafts)

    handler.set_consumed_draft_tokens(_batch(["b"], [1], None), drafts)

    draft_token_ids = handler.get_draft_tokens()
    assert draft_token_ids.req_ids == ["b"]
    assert draft_token_ids.draft_token_ids == [[]]


def test_synchronous_scheduling_hands_back_proposals():
    handler = DraftTokensHandler(torch.device("cuda"))
    drafts = _draft_slots()

    handler.set_consumed_draft_tokens(_batch(["a", "b"], [0, 2], [3, 3]), drafts)
    handler.set_draft_tokens(_batch(["c"], [1], [3]), drafts[1:2])

    draft_token_ids = handler.get_draft_tokens()
    assert draft_token_ids.req_ids == ["c"]
    assert draft_token_ids.draft_token_ids == [[4, 5, 6]]
