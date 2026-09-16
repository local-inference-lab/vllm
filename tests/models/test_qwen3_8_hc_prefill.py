# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep partial HC ownership out of decode, graphs and incompatible metadata."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen3_8_flash_next import hc_prefill


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, True),
        ({"num_prefill_tokens": 7519}, False),
        ({"num_decodes": 1}, False),
        ({"num_spec_decodes": 1}, False),
        ({"num_prefills": 0}, False),
    ],
)
def test_ownership_requires_complete_pure_prefill_metadata(
    monkeypatch, change, expected
):
    counts = dict(
        num_prefill_tokens=7520, num_prefills=1, num_decodes=0, num_spec_decodes=0
    )
    first = SimpleNamespace(**counts)
    second = SimpleNamespace(**(counts | change))
    context = SimpleNamespace(
        is_dummy_run=False,
        cudagraph_runtime_mode=SimpleNamespace(name="NONE"),
        ubatch_slices=None,
        attn_metadata={"gdn0": first, "gdn1": second},
    )
    monkeypatch.setattr(hc_prefill, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(hc_prefill, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    model = SimpleNamespace(hc_prefill_mode="shard")
    assert hc_prefill.eligible(model, 7520) is expected
    assert not hc_prefill.eligible(model, 674)
    context.cudagraph_runtime_mode.name = "PIECEWISE"
    assert hc_prefill.eligible(model, 7520) is expected
    context.cudagraph_runtime_mode.name = "FULL"
    assert not hc_prefill.eligible(model, 7520)


def test_owned_rows_preserve_global_token_order_and_cover_the_tail():
    full = torch.arange(2256 * 3).reshape(2256, 3)
    shards = [
        hc_prefill.RowOwnership(2256, rank, None).local(full) for rank in range(4)
    ]
    torch.testing.assert_close(torch.cat(shards), full, rtol=0, atol=0)
    assert all(shard.shape == (564, 3) for shard in shards)
    with pytest.raises(ValueError, match="row count"):
        hc_prefill.RowOwnership(2256, 0, None).local(full[:-1])


def test_collectives_reject_full_and_owned_row_confusion():
    owner = hc_prefill.RowOwnership(7520, 0, None)
    with pytest.raises(ValueError, match="owned tensor"):
        owner.gather(torch.empty(7520, 1))
    with pytest.raises(ValueError, match="full TP partial"):
        owner.reduce(torch.empty(1880, 1))
