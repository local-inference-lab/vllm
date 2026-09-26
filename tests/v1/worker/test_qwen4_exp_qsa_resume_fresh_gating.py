# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA pool fresh-reset must key on a real selector-pool restore.

The QSA selector pools (raw ring / logical tags / RoPE / anchor) are per-slot
module buffers, reset when ``qsa_state_is_fresh_gpu`` is set for a slot
(b12x_qsa.py).  Only the boundary-checkpoint rail restores them: it collects
``get_recurrent_checkpoint_tensors`` and copies the slot's rows, so it alone
may clear the flag -- and it does so itself, or the next forward would
re-zero what it just copied.

A connector external-resume restores KV pages, not these pools.  The ring is
a bounded speculative-interval window (``raw_ring_capacity`` rows); the
committed prefix's selector coverage is the compressed-K page tail, which the
padded connector transport does restore.  So the reset discards no prefix
state, and it re-derives the anchor as ``first_position - accepted`` =
``prefix - 1``, the value the checkpoint path seeds by hand.  A resumed
prefix without a checkpoint therefore keeps the fresh flag, which clears the
prior slot owner's rows.

Regression shape: ``add_request`` keyed the clear on ``num_computed_tokens >
0``, so a recycled slot still holding another request's rows was read as if it
were this request's committed selector state.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

MAX_REQS = 8


def _bare_qsa_state() -> Qwen4ExpModelState:
    """A Qwen4ExpModelState with only the fields add_request touches."""
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.max_num_reqs = MAX_REQS
    state.uses_qsa = True
    state.qsa_state_slot_ids = torch.arange(MAX_REQS, dtype=torch.int32)
    state._qsa_default_slot_ids = state.qsa_state_slot_ids.clone()
    state.qsa_state_is_fresh = torch.ones(MAX_REQS, dtype=torch.bool)
    state.qsa_state_is_fresh_gpu = torch.ones(MAX_REQS, dtype=torch.bool)
    state.qsa_num_accepted_tokens = torch.ones(MAX_REQS, dtype=torch.int32)
    state.qsa_committed_num_accepted_tokens_gpu = torch.ones(
        MAX_REQS, dtype=torch.int32
    )
    state._qsa_draft_is_prefilling = torch.zeros(MAX_REQS, dtype=torch.bool)
    state._qsa_draft_is_prefilling_gpu = torch.zeros(MAX_REQS, dtype=torch.bool)
    return state


class _FakeQSALayer:
    """Minimal stand-in for the QSA attention module surface add_request
    consumes: the anchor pool + the setter."""

    def __init__(self, max_seqs: int = MAX_REQS):
        self._raw_interval_start_positions = torch.full(
            (max_seqs,), -1, dtype=torch.int64
        )

    def set_recurrent_checkpoint_anchor(self, slot: int, anchor) -> None:
        self._raw_interval_start_positions[slot] = anchor


def _attach_fake_layers(state: Qwen4ExpModelState) -> list[_FakeQSALayer]:
    layers = [_FakeQSALayer(), _FakeQSALayer()]
    state.model = SimpleNamespace(
        modules=lambda: iter(layers),
    )
    return layers


def _new_req(
    num_computed_tokens: int,
    boundary_checkpoint=None,
    boundary_checkpoint_blocks=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens,
        boundary_checkpoint=boundary_checkpoint,
        boundary_checkpoint_blocks=boundary_checkpoint_blocks,
    )


def test_cold_add_keeps_fresh_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a cold add (no restored prefix) owns a recycled slot."""
    state = _bare_qsa_state()
    layers = _attach_fake_layers(state)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request", lambda self, i, r: None
    )

    state.add_request(3, _new_req(0))

    assert bool(state.qsa_state_is_fresh_gpu[3])
    assert int(state.qsa_committed_num_accepted_tokens_gpu[3]) == 1
    # No anchor seeding for a cold add.
    for layer in layers:
        assert int(layer._raw_interval_start_positions[3]) == -1


def test_external_resume_add_keeps_fresh_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connector external-resume restores KV pages only.  Its slot may be
    recycled from another request, so the pools must be reset; the reset
    re-derives the same anchor a checkpoint restore would have seeded."""
    state = _bare_qsa_state()
    state.qsa_state_is_fresh_gpu[5].fill_(False)  # prior owner cleared
    layers = _attach_fake_layers(state)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request", lambda self, i, r: None
    )

    state.add_request(5, _new_req(9024))

    assert bool(state.qsa_state_is_fresh_gpu[5])
    assert int(state.qsa_committed_num_accepted_tokens_gpu[5]) == 1
    # No anchor seeding: the fresh reset owns that write.
    for layer in layers:
        assert int(layer._raw_interval_start_positions[5]) == -1


def test_local_hit_add_keeps_fresh_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local prefix-cache hit without a checkpoint has the same gap: the
    producer's raw rows lived in the producer's slot, not in the restored
    pages, so the resumed slot stays fresh."""
    state = _bare_qsa_state()
    _attach_fake_layers(state)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request", lambda self, i, r: None
    )

    state.add_request(0, _new_req(6016))

    assert bool(state.qsa_state_is_fresh_gpu[0])


def test_boundary_checkpoint_add_clears_fresh_and_seeds_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one resume with selector-pool continuity: the checkpoint restore
    writes this slot's pools, so the flag stays clear and the anchor seeds to
    the checkpoint boundary - 1, the value the skipped fresh-reset would have
    produced (the rail seeds the same value; this keeps the pre-forward
    state right on its own)."""
    state = _bare_qsa_state()
    state.qsa_state_is_fresh_gpu[5].fill_(True)
    layers = _attach_fake_layers(state)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request", lambda self, i, r: None
    )

    checkpoint = SimpleNamespace(num_tokens=9024)
    state.add_request(
        5,
        _new_req(
            9024,
            boundary_checkpoint=checkpoint,
            boundary_checkpoint_blocks=((1,),),
        ),
    )

    assert not bool(state.qsa_state_is_fresh_gpu[5])
    assert int(state.qsa_committed_num_accepted_tokens_gpu[5]) == 1
    for layer in layers:
        assert int(layer._raw_interval_start_positions[5]) == 9023


def test_checkpoint_without_allocation_keeps_fresh_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rail restores only when both fields are present: without an
    allocation its add_request returns before the pool copy, so the gate must
    not clear a flag the restore will not re-clear."""
    state = _bare_qsa_state()
    state.qsa_state_is_fresh_gpu[5].fill_(False)
    _attach_fake_layers(state)
    monkeypatch.setattr(
        MambaHybridModelState, "add_request", lambda self, i, r: None
    )

    state.add_request(
        5,
        _new_req(9024, boundary_checkpoint=SimpleNamespace(num_tokens=9024)),
    )

    assert bool(state.qsa_state_is_fresh_gpu[5])


def test_fresh_hook_returns_the_gpu_flag_tensor() -> None:
    """The boundary-restore path consumes the flag through the interface
    hook; it must be the live GPU tensor, not a copy."""
    state = _bare_qsa_state()
    assert state.get_recurrent_checkpoint_fresh() is state.qsa_state_is_fresh_gpu


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_boundary_restore_clears_fresh_flag() -> None:
    """The restore branch of BoundaryCheckpointState.add_request must clear
    the fresh flag.  Drives the real method with a stub state carrying only
    the attributes the restore branch reads."""
    from vllm.v1.worker.gpu.boundary_checkpoint import BoundaryCheckpointState

    device = torch.device("cuda")
    slot = 2

    fresh_gpu = torch.ones(MAX_REQS, dtype=torch.bool, device=device)
    acceptance_gpu = torch.ones(MAX_REQS, dtype=torch.int32, device=device)
    model_state = SimpleNamespace(
        get_recurrent_checkpoint_fresh=lambda: fresh_gpu,
        get_recurrent_checkpoint_acceptance=lambda: acceptance_gpu,
        model_config=SimpleNamespace(max_model_len=16384),
    )

    # One auxiliary tensor restored from pool block 1 into the slot row.
    aux = torch.zeros(MAX_REQS, 8, dtype=torch.uint8, device=device)
    pool = torch.zeros(4, 64, dtype=torch.uint8, device=device)
    pool[1, 0:8] = 7
    aux_metadata = torch.tensor(
        [[aux.data_ptr(), aux.stride(0) * aux.element_size(), 8, 0]],
        dtype=torch.int64,
        device=device,
    )

    checkpoint = SimpleNamespace(
        num_tokens=9024,
        auxiliary_block_ids=(1,),
    )
    st = BoundaryCheckpointState.__new__(BoundaryCheckpointState)
    st.device = device
    st.model_state = model_state
    st.seen = torch.zeros(MAX_REQS, 4, dtype=torch.int32, device=device)
    st.metadata = torch.zeros(MAX_REQS, 7, dtype=torch.int32, device=device)
    st.blocks = torch.zeros(MAX_REQS, 4, 8, dtype=torch.int32, device=device)
    st.mamba_blocks = torch.zeros(MAX_REQS, 4, 1, dtype=torch.int32, device=device)
    st.stop_tokens = torch.zeros(MAX_REQS, 128, dtype=torch.int32, device=device)
    st.mamba_group_ids = [0]
    st.auxiliary_metadata = aux_metadata
    st.pool = pool
    st.target_modules = []

    st.add_request(
        slot,
        SimpleNamespace(
            boundary_checkpoint_blocks=((1, 0, 0, 0, 0, 0, 0, 0),),
            sampling_params=SimpleNamespace(
                max_tokens=64, min_tokens=1, stop_token_ids=[], eos_token_id=None
            ),
            prompt_len=9059,
            recurrent_instruction_boundary=None,
            recurrent_prefill_tail_boundary=None,
            boundary_checkpoint=checkpoint,
        ),
    )

    assert not bool(fresh_gpu[slot])
    # The restore still performs its other writes (acceptance reset).
    assert int(acceptance_gpu[slot]) == 1
