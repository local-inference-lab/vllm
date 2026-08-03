"""Persistent FlashInfer decode wrappers may only be reused when the runtime
q_len_per_req equals the value frozen during wrapper planning (1 + K).

The decode-classification ceiling (reorder_batch_threshold = 1 + 2K under
parallel drafting) deliberately admits reduced-depth lone steps as decodes —
spec truncation near max_tokens, short chunked-prefill tails fused with the
spec step. Those steps must take the dynamic wrapper; routing them to a
captured wrapper raises inside flashinfer's fast_decode_plan ("q_len_per_req
is part of the frozen cudagraph shape: this wrapper was planned with 6,
got 8|5") and kills the engine.

These tests pin the invariant and, deliberately, the distinction between the
classification ceiling (1 + 2K) and the planned shape (1 + K): substituting
reorder_batch_threshold for the planned length reintroduces the crash.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.flashinfer import (
    persistent_decode_wrapper_eligible,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

K = 5
PLANNED = 1 + K
CEILING = 1 + 2 * K
MAX_BS = 96


def _threshold_for(parallel_drafting: bool) -> int:
    stub = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=K,
                parallel_drafting=parallel_drafting,
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        )
    )
    AttentionMetadataBuilder._init_reorder_batch_threshold(
        stub, 1, supports_spec_as_decode=True
    )
    return stub.reorder_batch_threshold


def _eligible(q_len, *, num_reqs=1, pure_decode=True, max_bs=MAX_BS,
              planned=PLANNED):
    return persistent_decode_wrapper_eligible(
        pure_decode=pure_decode,
        num_decode_tokens=q_len * num_reqs,
        decode_cudagraph_max_bs=max_bs,
        decode_q_len=q_len,
        planned_decode_q_len=planned,
    )


def test_classification_ceiling_is_not_planned_shape():
    assert _threshold_for(parallel_drafting=True) == CEILING
    assert _threshold_for(parallel_drafting=False) == PLANNED
    assert CEILING != PLANNED


def test_planned_shape_selects_persistent_wrapper():
    assert _eligible(PLANNED)
    assert _eligible(PLANNED, num_reqs=8)


@pytest.mark.parametrize(
    "q_len", [q for q in range(1, CEILING + 1) if q != PLANNED]
)
def test_reduced_or_extended_depth_falls_back_to_dynamic(q_len):
    # Covers both observed crash signatures: q_len=5 (spec truncation near
    # max_tokens) and q_len=8 (chunked-prefill tail fused with spec step).
    assert not _eligible(q_len)


def test_above_ceiling_classifies_as_prefill():
    meta = SimpleNamespace(
        max_query_len=CEILING + 1,
        num_reqs=1,
        num_actual_tokens=CEILING + 1,
        query_start_loc_cpu=torch.tensor([0, CEILING + 1], dtype=torch.int32),
    )
    nd, npf, ndt, npt = split_decodes_and_prefills(
        meta, decode_threshold=CEILING, require_uniform=True
    )
    assert (nd, npf, ndt, npt) == (0, 1, 0, CEILING + 1)


def test_other_predicate_terms_still_gate():
    assert not _eligible(PLANNED, pure_decode=False)
    assert not _eligible(PLANNED, num_reqs=32)  # over capacity


def test_no_spec_planned_length_is_one():
    assert _eligible(1, planned=1)
    assert not _eligible(2, planned=1)
