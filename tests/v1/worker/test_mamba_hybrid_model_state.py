# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


def test_reset_kv_cache_state_recreates_align_context(monkeypatch) -> None:
    import vllm.v1.worker.gpu.model_states.mamba_hybrid as state_module

    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = object()
    state._mamba_copy_funcs_by_type = object()
    state._mamba_group_ids = [1]
    state._mamba_spec = object()
    state.recoverssm = RecoverSSMState()
    state.recoverssm._step = (object(),)
    state.model = object()
    state.max_num_reqs = 2
    state.device = torch.device("cpu")
    new_cache = torch.empty(2, 1)
    forward_context = {"layer": SimpleNamespace(kv_cache=(new_cache,))}
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=forward_context)
    )
    kv_cache_config = object()
    unused_block_table = torch.empty(2, 1, dtype=torch.int32)
    new_block_table = torch.empty(2, 1, dtype=torch.int32)
    copy_funcs = object()
    initialized_with: list[tuple[object, object, object]] = []

    class _Context:
        is_initialized = False

        def initialize_from_forward_context(
            self, config, context, funcs, block_tables
        ) -> None:
            initialized_with.append((config, context, block_tables[0]))
            assert funcs is copy_funcs
            assert context["layer"].kv_cache[0] is new_cache
            self.is_initialized = True

    created_context = _Context()
    monkeypatch.setattr(
        state_module,
        "resolve_mamba_state_copy_funcs",
        lambda model, config: copy_funcs,
    )
    monkeypatch.setattr(
        state_module.MambaSpecDecodeGPUContext,
        "create",
        lambda **_kwargs: created_context,
    )

    previous_group_ids = state._mamba_group_ids
    previous_spec = state._mamba_spec
    state.reset_kv_cache_state()

    assert state._mamba_ctx is None
    assert state._mamba_copy_funcs_by_type is None
    assert state._aligned_metadata_ctx is None
    assert state._aligned_metadata_groups is None
    assert state._mamba_group_ids is previous_group_ids
    assert state._mamba_spec is previous_spec
    assert state.recoverssm._step is None

    result = state._ensure_align_ctx(
        kv_cache_config,
        state._mamba_group_ids,
        (unused_block_table, new_block_table),
    )

    assert result is created_context
    assert initialized_with == [(kv_cache_config, forward_context, new_block_table)]


def test_aligned_metadata_reuses_views_and_rebinds_with_the_cache() -> None:
    state = object.__new__(MambaHybridModelState)
    state._aligned_metadata_groups = None
    state._aligned_metadata_ctx = None
    state._aligned_metadata_builders = []
    state._get_mamba_group_info = lambda _: ([1, 2], None)
    builders = [SimpleNamespace(mamba_aligned_state_indices=None) for _ in range(2)]
    groups = [
        [SimpleNamespace(get_metadata_builder=Mock(return_value=builder))]
        for builder in builders
    ]
    attn_groups = [[], *groups]
    indices = torch.arange(6, dtype=torch.int32).reshape(2, 3, 1)
    ctx = SimpleNamespace(
        aligned_state_indices=indices,
        compute_aligned_state_indices=Mock(),
    )
    state._ensure_align_ctx = lambda *_: ctx
    seq_lens = torch.tensor([10, 20, 0], dtype=torch.int32)
    for num_reqs in (3, 2):
        state._prepare_aligned_state_indices(seq_lens, num_reqs, attn_groups, None, ())
        if num_reqs == 3:
            views = [builder.mamba_aligned_state_indices for builder in builders]
        indices.add_(10)
        for index, builder in enumerate(builders):
            assert builder.mamba_aligned_state_indices is views[index]
            torch.testing.assert_close(
                builder.mamba_aligned_state_indices, indices[index]
            )
    for group in groups:
        group[0].get_metadata_builder.assert_called_once_with(0)
    assert ctx.compute_aligned_state_indices.call_count == 2

    replacement = torch.full_like(indices, 42)
    ctx = SimpleNamespace(
        aligned_state_indices=replacement,
        compute_aligned_state_indices=Mock(),
    )
    state._prepare_aligned_state_indices(seq_lens, 3, attn_groups, None, ())
    for index, builder in enumerate(builders):
        assert builder.mamba_aligned_state_indices is not views[index]
        torch.testing.assert_close(
            builder.mamba_aligned_state_indices, replacement[index]
        )
        torch.testing.assert_close(views[index], indices[index])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
