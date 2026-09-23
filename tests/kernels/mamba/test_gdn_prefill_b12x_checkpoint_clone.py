# SPDX-License-Identifier: Apache-2.0
# The b12x GDN prefill kernel must clone state at
# checkpoint_offset == query_len (a step-end boundary crossing) into the
# designated checkpoint slot, matching the reference oracle. The
# strictly-interior offset runs as the control.
import pytest
import torch


def _setup(lengths, offsets, ckpt_slots, max_seqs):
    from b12x.testing.delta_prefill_cases import PrefillCase, make_inputs

    device = torch.device("cuda")
    case = PrefillCase("gdn", 2, 6, lengths)
    max_tokens = sum(lengths)
    tensors = make_inputs(
        case, device=device, max_tokens=max_tokens, max_seqs=max_seqs
    )
    # Distinct non-final, non-initial checkpoint slots (pool has 3*max_seqs+1).
    tensors["checkpoint_state_indices"][: len(lengths)] = torch.tensor(
        ckpt_slots, dtype=torch.int32, device=device
    )
    tensors["checkpoint_offsets"][: len(lengths)] = torch.tensor(
        offsets, dtype=torch.int32, device=device
    )
    return case, tensors


def _run_case(lengths, offsets, ckpt_slots, max_seqs=2):
    from b12x.testing.delta_prefill_cases import (
        assert_close,
        oracle,
        prepared_binding,
        run_binding,
    )

    case, tensors = _setup(lengths, offsets, ckpt_slots, max_seqs)
    with prepared_binding(
        case,
        tensors,
        max_tokens=sum(lengths),
        max_seqs=max_seqs,
        checkpoint_export=True,
        null_state_index=0,
    ) as binding:
        run_binding("gdn", binding)
        torch.cuda.synchronize()
        expected_output, expected_pool = oracle(case, tensors, null_state_index=0)
        assert_close(
            "output",
            binding.output[: sum(lengths)],
            expected_output[: sum(lengths)],
            ratio=1e-2,
        )
        for slot, offset in zip(ckpt_slots, offsets):
            assert_close(
                f"state[{slot}]@offset={offset}",
                binding.recurrent_state[slot],
                expected_pool[slot],
                ratio=5e-3,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v1_checkpoint_clone_interior_offset():
    """Control: interior offset (32 < query_len 64) exports state@32."""
    _run_case(lengths=(64,), offsets=(32,), ckpt_slots=(4,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v1_checkpoint_clone_at_query_end():
    """The design's step-end case: offset == query_len (64) must export
    state@64 into the checkpoint slot (distinct from the final slot)."""
    _run_case(lengths=(64,), offsets=(64,), ckpt_slots=(4,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v1_checkpoint_clone_at_query_end_multi_seq():
    """Step-end clone for every sequence in a packed query."""
    _run_case(lengths=(64, 48), offsets=(64, 48), ckpt_slots=(4, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v1_checkpoint_clone_offset_exceeds_query_len_no_write():
    """offset > query_len: reference writes nothing; kernel must leave the
    slot untouched (design gate never emits this, guard against corruption)."""
    from b12x.testing.delta_prefill_cases import prepared_binding, run_binding

    case, tensors = _setup((64,), (96,), (4,), max_seqs=2)
    with prepared_binding(
        case,
        tensors,
        max_tokens=64,
        max_seqs=2,
        checkpoint_export=True,
        null_state_index=0,
    ) as binding:
        saved = binding.recurrent_state[4].clone()
        run_binding("gdn", binding)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            binding.recurrent_state[4], saved, rtol=0, atol=0
        )
