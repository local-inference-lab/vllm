# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.glm5next.model_state import (
    Glm5NextAttnMetadata,
    Glm5NextModelState,
)
from vllm.platforms import current_platform
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _bare_model_state() -> Glm5NextModelState:
    state = Glm5NextModelState.__new__(Glm5NextModelState)
    state.max_num_reqs = 8
    state.uses_pooled_selector = True
    state.selector_pool_size = 4
    state.selector_state_slot_ids = torch.full((8,), -1, dtype=torch.int32)
    state.selector_state_is_fresh = torch.ones(8, dtype=torch.bool)
    state.selector_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.mamba_num_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.selector_state_is_fresh_gpu = torch.tensor(
        [True, False, True, True, True, False, True, True]
    )
    state.selector_committed_num_accepted_tokens_gpu = torch.tensor(
        [1, 2, 1, 1, 1, 4, 1, 1], dtype=torch.int32
    )
    state.num_accepted_tokens_gpu = torch.ones(8, dtype=torch.int32)
    state._selector_draft_is_prefilling = torch.zeros(8, dtype=torch.bool)
    state._selector_draft_is_prefilling_gpu = torch.zeros(8, dtype=torch.bool)
    return state


def test_glm5next_metadata_targets_only_selector_builder() -> None:
    metadata = Glm5NextAttnMetadata(
        is_prefilling=torch.zeros(4, dtype=torch.bool),
        selector_state_slot_ids=torch.tensor([5, 1, -1, -1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, False, True, True]),
        selector_num_accepted_tokens=torch.tensor([4, 2, 1, 1], dtype=torch.int32),
        selector_is_prefilling=torch.tensor([False, True, False, False]),
    )
    selector_builder = SimpleNamespace(requires_glm_next_selector_metadata=True)

    kwargs = metadata.get_extra_attn_kwargs(selector_builder, 2)

    assert set(kwargs) == {
        "selector_state_slot_ids",
        "selector_state_is_fresh",
        "selector_num_accepted_tokens",
        "selector_is_prefilling",
    }
    assert torch.equal(
        kwargs["selector_state_slot_ids"], torch.tensor([5, 1], dtype=torch.int32)
    )
    assert metadata.get_extra_attn_kwargs(SimpleNamespace(), 2) == {}


def test_glm5next_selector_state_tracks_reordering_and_invalidates_padding() -> None:
    state = _bare_model_state()
    first_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([5, 1], dtype=torch.int64),
    )

    slots, fresh, accepted = state._prepare_selector_state(first_batch, 4)
    pointers = slots.data_ptr(), fresh.data_ptr(), accepted.data_ptr()
    assert torch.equal(slots, torch.tensor([5, 1, -1, -1], dtype=torch.int32))
    assert torch.equal(fresh, torch.tensor([False, False, True, True]))
    assert torch.equal(accepted, torch.tensor([4, 2, 1, 1], dtype=torch.int32))

    reordered = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([1, 5], dtype=torch.int64),
    )
    slots, fresh, accepted = state._prepare_selector_state(reordered, 4)
    assert (slots.data_ptr(), fresh.data_ptr(), accepted.data_ptr()) == pointers
    assert torch.equal(slots, torch.tensor([1, 5, -1, -1], dtype=torch.int32))
    assert torch.equal(accepted, torch.tensor([2, 4, 1, 1], dtype=torch.int32))


def test_glm5next_selector_and_mamba_use_independent_acceptance_after_alignment() -> (
    None
):
    state = _bare_model_state()
    state.selector_committed_num_accepted_tokens_gpu[5] = 4
    state.num_accepted_tokens_gpu[5] = 1
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([5], dtype=torch.int32),
    )

    _, _, selector_accepted = state._prepare_selector_state(input_batch, num_reqs=1)
    mamba_accepted = state._prepare_mamba_acceptance(input_batch, num_reqs=1)

    assert selector_accepted.data_ptr() != mamba_accepted.data_ptr()
    assert torch.equal(selector_accepted, torch.tensor([4], dtype=torch.int32))
    assert torch.equal(mamba_accepted, torch.tensor([1], dtype=torch.int32))


def test_glm5next_kda_reuses_captured_metadata_after_reordering(monkeypatch) -> None:
    from tests.v1.attention.test_gdn_metadata_builder import _create_gdn_builder

    monkeypatch.setenv("VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH", "1")
    builders = [
        _create_gdn_builder(3, full_cuda_graph=True, num_prefill_checkpoint_blocks=1)
        for _ in range(2)
    ]
    for builder in builders:
        builder.vllm_config.cache_config.mamba_cache_mode = "align"
    state = _bare_model_state()
    state.vllm_config = builders[0].vllm_config
    state.max_model_len = 4096
    state.selector_is_prefilling = CpuGpuBuffer(
        8, dtype=torch.bool, device=torch.device("cpu"), pin_memory=False
    )
    state._align_mode = True
    state._aligned_metadata_groups = state._aligned_metadata_ctx = None
    state._aligned_metadata_builders = []
    state._gdn_spec_accepted_tokens = torch.ones(8, dtype=torch.int32)
    state.recoverssm = None
    state._get_mamba_group_info = lambda _: ([0, 1], None)
    indices = torch.arange(64, dtype=torch.int32).reshape(2, 8, 4)
    ctx = SimpleNamespace(
        aligned_state_indices=indices, compute_aligned_state_indices=Mock()
    )
    state._ensure_align_ctx = lambda *_: ctx
    groups = [
        [SimpleNamespace(get_metadata_builder=lambda _, b=b: b, layer_names=[str(i)])]
        for i, b in enumerate(builders)
    ]
    kv_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=b.kv_cache_spec) for b in builders
        ]
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=4,
        num_tokens_after_padding=4,
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_draft_tokens_per_req=np.array([3], dtype=np.int32),
        is_prefilling_np=np.array([False]),
        seq_lens_cpu_upper_bound=torch.tensor([64], dtype=torch.int32),
        seq_lens=torch.tensor([64], dtype=torch.int32),
        idx_mapping=torch.tensor([5], dtype=torch.int32),
        prompt_lens=torch.tensor([32]),
        dcp_local_seq_lens=None,
    )
    block_tables = (torch.zeros(1, 8, dtype=torch.int32),) * 2
    slot_mappings = torch.zeros(2, 4, dtype=torch.int64)

    def prepare(for_capture=False):
        return state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            block_tables,
            slot_mappings,
            groups,
            kv_config,
            for_capture=for_capture,
        )

    captured = prepare(for_capture=True)
    for slot, accepted in ((5, 4), (1, 2), (5, 1)):
        batch.idx_mapping.fill_(slot)
        state.num_accepted_tokens_gpu[slot] = accepted
        indices.add_(8)
        current = prepare()
        for i in range(2):
            assert current[str(i)].is_uniform_spec_decode
            torch.testing.assert_close(
                captured[str(i)].spec_state_indices_tensor, indices[i, :1]
            )
            assert captured[str(i)].num_accepted_tokens.item() == accepted
        assert (
            current["0"].num_accepted_tokens.data_ptr()
            == current["1"].num_accepted_tokens.data_ptr()
        )


def test_glm5next_draft_metadata_preserves_first_step_acceptance() -> None:
    state = _bare_model_state()
    idx_mapping = torch.tensor([5, 1], dtype=torch.int32)

    first = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=1,
    )
    assert first is not None
    assert torch.equal(
        first.selector_state_slot_ids,
        torch.tensor([5, 1, -1, -1], dtype=torch.int32),
    )
    assert torch.equal(
        first.selector_state_is_fresh,
        torch.tensor([False, False, True, True]),
    )
    assert torch.equal(
        first.selector_num_accepted_tokens,
        torch.tensor([4, 2, 1, 1], dtype=torch.int32),
    )

    later = state.prepare_draft_attn_metadata(
        idx_mapping=idx_mapping,
        num_reqs=2,
        num_reqs_padded=4,
        draft_index=2,
    )
    assert later is not None
    assert torch.equal(
        later.selector_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_glm5next_postprocess_commits_selector_before_mamba_alignment_reset() -> None:
    class ResetAcceptedTokens:
        def run_fused_postprocess_align(
            self,
            num_reqs: int,
            num_accepted_tokens_gpu: torch.Tensor,
            state_idx_gpu: torch.Tensor,
            num_computed_tokens: torch.Tensor,
            idx_mapping: torch.Tensor,
        ) -> None:
            num_accepted_tokens_gpu.fill_(1)

    state = Glm5NextModelState.__new__(Glm5NextModelState)
    state.uses_pooled_selector = True
    state.selector_committed_num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state.selector_state_is_fresh_gpu = torch.ones(5, dtype=torch.bool, device="cuda")
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = True
    state._mamba_ctx = ResetAcceptedTokens()
    state._mamba_state_idx_gpu = torch.zeros(5, dtype=torch.int32, device="cuda")
    state.recoverssm = None

    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int32, device="cuda")
    num_sampled = torch.tensor([4, 2, 3], dtype=torch.int32, device="cuda")
    num_computed_tokens = torch.zeros(5, dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    assert state.num_accepted_tokens_gpu.tolist() == [1, 1, 1, 1, 1]
    assert state.selector_committed_num_accepted_tokens_gpu.tolist() == [
        9,
        3,
        9,
        4,
        9,
    ]
    assert state.selector_state_is_fresh_gpu.tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_glm5next_recycled_and_rebound_state_is_fresh(monkeypatch) -> None:
    state = _bare_model_state()
    state.selector_state_is_fresh_gpu.fill_(False)
    state.selector_committed_num_accepted_tokens_gpu.fill_(7)
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: None,
    )
    monkeypatch.setattr(
        MambaHybridModelState,
        "reset_kv_cache_state",
        lambda self: None,
    )

    state.add_request(5, SimpleNamespace(num_computed_tokens=0))

    assert state.selector_state_is_fresh_gpu[5]
    assert state.selector_committed_num_accepted_tokens_gpu[5] == 1
    assert not torch.any(state.selector_state_is_fresh_gpu[:5])
    state.reset_kv_cache_state()
    assert torch.all(state.selector_state_is_fresh_gpu)
    assert torch.all(state.selector_committed_num_accepted_tokens_gpu == 1)


@pytest.mark.parametrize(
    "prefix_length",
    [
        pytest.param(1, id="prefix-match-unit-1"),
        pytest.param(2, id="prefix-match-unit-2"),
        pytest.param(5, id="odd-connector-hit"),
    ],
)
def test_glm5next_rejects_unaligned_fresh_prefix(
    monkeypatch,
    prefix_length: int,
) -> None:
    state = _bare_model_state()
    calls = []
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: calls.append(req_index),
    )

    with pytest.raises(
        ValueError,
        match=rf"num_computed_tokens={prefix_length}.*divisible by index_kpool=4",
    ):
        state.add_request(
            3,
            SimpleNamespace(num_computed_tokens=prefix_length),
        )

    assert calls == []


def test_glm5next_accepts_pool_aligned_fresh_prefix(monkeypatch) -> None:
    state = _bare_model_state()
    calls = []
    monkeypatch.setattr(
        MambaHybridModelState,
        "add_request",
        lambda self, req_index, new_req_data: calls.append(req_index),
    )

    state.add_request(3, SimpleNamespace(num_computed_tokens=4))

    assert calls == [3]
    assert state.selector_state_is_fresh_gpu[3]
