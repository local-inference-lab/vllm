# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def _handler(*, needs_real_draft_tokens: bool) -> DraftTokensHandler:
    handler = object.__new__(DraftTokensHandler)
    handler.needs_real_draft_tokens = needs_real_draft_tokens
    handler.req_ids = []
    handler.draft_tokens_np = None
    handler.num_draft_tokens = 0
    return handler


def test_fixed_width_drafts_skip_host_copy() -> None:
    handler = _handler(needs_real_draft_tokens=False)
    input_batch = SimpleNamespace(
        req_ids=["req-0", "req-1"], has_structured_output_reqs=False
    )
    draft_tokens = np.zeros((2, 2), dtype=np.int32)

    handler.set_draft_tokens(input_batch, draft_tokens)  # type: ignore[arg-type]
    output = handler.get_draft_tokens()

    assert handler.draft_tokens_np is None
    assert output is not None
    assert output.req_ids == ["req-0", "req-1"]
    assert output.draft_token_ids == [[-1, -1], [-1, -1]]


def test_host_copy_requirement() -> None:
    fixed = _handler(needs_real_draft_tokens=False)
    variable = _handler(needs_real_draft_tokens=True)
    plain_batch = SimpleNamespace(has_structured_output_reqs=False)
    structured_batch = SimpleNamespace(has_structured_output_reqs=True)

    assert not fixed.needs_host_copy(plain_batch)  # type: ignore[arg-type]
    assert fixed.needs_host_copy(structured_batch)  # type: ignore[arg-type]
    assert variable.needs_host_copy(plain_batch)  # type: ignore[arg-type]


def test_host_draft_ids_trim_negative_suffix() -> None:
    handler = _handler(needs_real_draft_tokens=True)
    handler.req_ids = ["req-0", "req-1"]
    handler.draft_tokens_np = np.array([[11, 12], [21, -1]], dtype=np.int32)
    handler.copy_event = SimpleNamespace(synchronize=lambda: None)

    output = handler.get_draft_tokens()

    assert output is not None
    assert output.draft_token_ids == [[11, 12], [21]]
