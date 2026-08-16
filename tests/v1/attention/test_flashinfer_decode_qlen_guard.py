"""Persistent FlashInfer decode wrappers may only be reused for the exact
``(num_decodes, q_len_per_req)`` shape a CUDA-graph capture pass planned them
with.

Two independent hazards make the shape load-bearing:

* FlashInfer freezes ``q_len_per_req`` into a CUDA-graph wrapper and raises
  inside ``fast_decode_plan`` when asked to replan for another one
  ("q_len_per_req is part of the frozen cudagraph shape: this wrapper was
  planned with 6, got 8|5").
* A captured graph bakes in the *addresses* of its wrapper's plan buffers. The
  dynamic wrapper rebinds ``_paged_kv_indptr_buf`` /
  ``_paged_kv_last_page_len_buf`` and reallocates ``_qo_indptr_buf`` on every
  plan, so a graph captured around it replays against freed memory
  (cudaErrorIllegalAddress).

Hence the planned shapes are recorded by capture, not derived from
``1 + num_spec_tokens``: the V2 speculator's draft-decode graphs are captured
at ``q_len == 1`` and a per-batch-size speculative depth schedule captures one
verify shape per depth. These tests pin that recorded-shape contract and,
deliberately, the distinction between the decode-classification ceiling
(``reorder_batch_threshold == 1 + 2K`` under parallel drafting) and the planned
shapes: substituting the ceiling reintroduces the crash.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.flashinfer import (
    decode_q_len_from_indptr,
    persistent_decode_wrapper_eligible,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

K = 5
PLANNED = 1 + K
CEILING = 1 + 2 * K
MAX_BS = 96
# What a V2 + MTP-K capture pass plans: target verify graphs at q_len 1 + K and
# the speculator's draft-decode graphs at q_len 1, for every captured size.
CAPTURED = {(bs, q) for bs in (1, 8) for q in (1, PLANNED)}


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
              planned=CAPTURED, capturing=False):
    return persistent_decode_wrapper_eligible(
        pure_decode=pure_decode,
        num_decode_tokens=q_len * num_reqs,
        decode_cudagraph_max_bs=max_bs,
        decode_shape=(num_reqs, q_len),
        planned_decode_shapes=planned,
        planning_for_capture=capturing,
    )


def test_classification_ceiling_is_not_planned_shape():
    assert _threshold_for(parallel_drafting=True) == CEILING
    assert _threshold_for(parallel_drafting=False) == PLANNED
    assert CEILING != PLANNED


def test_planned_verify_shape_selects_persistent_wrapper():
    assert _eligible(PLANNED)
    assert _eligible(PLANNED, num_reqs=8)


def test_planned_draft_decode_shape_selects_persistent_wrapper():
    # The V2 speculator drafts one token per request per step, so its captured
    # draft-decode graphs run q_len == 1. Demoting them to the dynamic wrapper
    # is what made their replay fault with cudaErrorIllegalAddress.
    assert _eligible(1)
    assert _eligible(1, num_reqs=8)


def test_capture_build_is_always_eligible():
    # The capture-time build is what creates, plans and records the wrapper,
    # so it must be admitted before its shape is in the recorded set.
    assert _eligible(3, planned=set(), capturing=True)
    assert _eligible(1, num_reqs=4, planned=set(), capturing=True)


@pytest.mark.parametrize(
    "q_len", [q for q in range(1, CEILING + 1) if q not in (1, PLANNED)]
)
def test_uncaptured_depth_falls_back_to_dynamic(q_len):
    # Covers both observed crash signatures: q_len=5 (spec truncation near
    # max_tokens) and q_len=8 (chunked-prefill tail fused with spec step).
    # Nothing replays a graph for these shapes, so replanning per call is safe.
    assert not _eligible(q_len)


def test_shape_is_keyed_on_batch_size_too():
    # A wrapper planned for another batch size has the wrong fixed_batch_size
    # and the wrong buffer slices; FlashInfer would reject the replan.
    assert not _eligible(PLANNED, num_reqs=4)
    assert not _eligible(1, num_reqs=4)


def test_depth_schedule_captures_one_verify_shape_per_depth():
    # num_speculative_tokens_per_batch_size captures a verify graph per depth;
    # every captured depth must reach its own persistent wrapper, and only the
    # captured ones.
    scheduled = {(1, 4), (2, 4), (4, 2), (8, 2)}
    assert _eligible(4, num_reqs=2, planned=scheduled)
    assert _eligible(2, num_reqs=8, planned=scheduled)
    assert not _eligible(4, num_reqs=8, planned=scheduled)


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
    assert not _eligible(PLANNED, num_reqs=8, max_bs=8)  # over capacity
    # Capacity and purity gate the capture build as well.
    assert not _eligible(PLANNED, pure_decode=False, capturing=True)


def test_no_spec_planned_length_is_one():
    assert _eligible(1, planned={(1, 1)})
    assert not _eligible(2, planned={(1, 1)})


def _indptr(*lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


@pytest.mark.parametrize(
    ("lens", "expected"),
    [
        ((PLANNED,), PLANNED),
        ((PLANNED, 0), PLANNED),          # one active + one padding row
        ((PLANNED, PLANNED, 0), PLANNED),
        ((PLANNED,) * 3 + (0,), PLANNED), # total not divisible by num rows
        ((0, 0), 0),                      # all-padding
        ((5,), 5),                        # reduced-depth lone step
        ((1, 1, 0), 1),                   # draft decode step with padding
    ],
)
def test_decode_q_len_ignores_zero_padding(lens, expected):
    assert decode_q_len_from_indptr(_indptr(*lens), len(lens)) == expected


def test_zero_padded_planned_batch_keeps_persistent_wrapper():
    # Regression: uniform CUDA-graph batches may carry zero-length padding
    # rows; averaging tokens over rows understated the active q_len and
    # demoted a planned-shape batch to the dynamic wrapper.
    lens = (PLANNED, 0)
    q_len = decode_q_len_from_indptr(_indptr(*lens), len(lens))
    assert persistent_decode_wrapper_eligible(
        pure_decode=True,
        num_decode_tokens=sum(lens),
        decode_cudagraph_max_bs=MAX_BS,
        decode_shape=(len(lens), q_len),
        planned_decode_shapes={(len(lens), PLANNED)},
        planning_for_capture=False,
    )
