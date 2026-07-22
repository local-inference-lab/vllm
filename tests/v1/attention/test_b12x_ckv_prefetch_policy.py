# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseImpl,
    _ckv_prefetch_ring_slots,
    _ckv_prefetch_supports_format,
    _ckv_prefetch_target_indices,
    _ckv_workspace_identity,
    _CKVPrefetchStateRegistry,
)


@pytest.mark.parametrize(
    ("depth", "expected_slots", "expected_targets"),
    [
        (0, 1, []),
        (1, 2, [2]),
        (3, 4, [2, 3, 4]),
    ],
)
def test_ckv_prefetch_depth_controls_ring_and_targets(
    depth, expected_slots, expected_targets
):
    caches = [torch.empty(0) for _ in range(6)]

    assert _ckv_prefetch_ring_slots(depth) == expected_slots
    assert _ckv_prefetch_target_indices(1, depth, caches, {}) == expected_targets


def test_ckv_prefetch_supports_native_full_record_formats():
    assert _ckv_prefetch_supports_format("nvfp4_ds_mla")
    assert _ckv_prefetch_supports_format("fp8_ds_mla")
    assert not _ckv_prefetch_supports_format("auto")


def test_ckv_prefetch_targets_stop_at_first_unregistered_layer():
    caches = [torch.empty(0), torch.empty(0), torch.empty(0), None, torch.empty(0)]
    pending = {2: (object(), 0)}

    assert _ckv_prefetch_target_indices(1, 3, caches, pending) == []


@pytest.mark.parametrize("record_bytes", [368, 432])
def test_ckv_workspace_reuses_local_staging_across_ring_slots(record_bytes):
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_gather_enabled = True
    impl._ckv_workspace_slots = 4
    impl._ckv_local_capacity = 8
    impl._kv_record_bytes = record_bytes
    impl.dcp_world_size = 4
    impl.device = torch.device("cpu")
    impl._ckv_workspace_nbytes = (
        (1 + impl._ckv_workspace_slots * impl.dcp_world_size)
        * impl._ckv_local_capacity
        * impl._kv_record_bytes
    )
    workspace = torch.empty(impl._ckv_workspace_nbytes, dtype=torch.uint8)

    local_0, gathered_0 = impl._ckv_workspace_views(workspace, 0)
    local_3, gathered_3 = impl._ckv_workspace_views(workspace, 3)

    assert local_0.data_ptr() == local_3.data_ptr()
    assert gathered_0.shape == gathered_3.shape == (32, record_bytes)
    assert gathered_0.data_ptr() != gathered_3.data_ptr()


def test_ckv_workspace_rejects_ring_slot_outside_depth():
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_gather_enabled = True
    impl._ckv_workspace_slots = 2
    impl._ckv_local_capacity = 1
    impl._kv_record_bytes = 432
    impl.dcp_world_size = 2
    impl.device = torch.device("cpu")
    impl._ckv_workspace_nbytes = 5 * impl._kv_record_bytes
    workspace = torch.empty(impl._ckv_workspace_nbytes, dtype=torch.uint8)

    with pytest.raises(ValueError, match="outside"):
        impl._ckv_workspace_views(workspace, 2)


class _FakeEvent:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


def test_ckv_prefetch_incomplete_step_recovers_with_stream_wait():
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(torch.empty(16, dtype=torch.uint8))
    cache = torch.empty(0)
    event = _FakeEvent()
    state.register_cache(3, cache)
    state.pending_layers[3] = (event, 0)
    state.last_layer_idx = 2

    state.enter_layer(0)

    assert event.wait_calls == 1
    assert state.pending_layers == {}
    assert state.layer_caches[3] is cache
    assert state.last_layer_idx == 0


def test_ckv_prefetch_first_request_discovers_caches_without_lookahead():
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(torch.empty(16, dtype=torch.uint8))
    caches = [torch.empty(0) for _ in range(4)]

    for layer_idx, cache in enumerate(caches):
        state.enter_layer(layer_idx)
        state.register_cache(layer_idx, cache)

        assert (
            _ckv_prefetch_target_indices(
                layer_idx, 3, state.layer_caches, state.pending_layers
            )
            == []
        )
        assert state.gather_stream is None

    state.enter_layer(0)

    assert _ckv_prefetch_target_indices(0, 3, state.layer_caches, {}) == [1, 2, 3]


def test_ckv_prefetch_target_and_draft_lifecycles_are_isolated(monkeypatch):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    (target_workspace,) = manager.get_simultaneous(((16,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft_workspace,) = manager.get_simultaneous(((16,), torch.uint8))

    registry = _CKVPrefetchStateRegistry()
    target_state = registry.for_workspace(target_workspace)
    draft_state = registry.for_workspace(draft_workspace)
    target_cache = torch.empty(0)
    draft_cache = torch.empty(0)
    target_event = _FakeEvent()
    target_ring = target_state.get_ckv_workspace(64)
    draft_ring = draft_state.get_ckv_workspace(64)

    target_state.register_cache(1, target_cache)
    target_state.pending_layers[1] = (target_event, 1)
    draft_state.register_cache(1, draft_cache)

    assert target_state is not draft_state
    assert target_ring.untyped_storage().data_ptr() != (
        draft_ring.untyped_storage().data_ptr()
    )
    assert target_state.layer_caches[1] is target_cache
    assert target_state.pending_layers[1] == (target_event, 1)
    assert draft_state.layer_caches[1] is draft_cache
    assert draft_state.pending_layers == {}


def test_ckv_prefetch_lazily_owns_one_stream_per_workspace_lane(monkeypatch):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    (target_workspace,) = manager.get_simultaneous(((16,), torch.uint8))
    (target_workspace_reused,) = manager.get_simultaneous(((16,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft_workspace,) = manager.get_simultaneous(((16,), torch.uint8))

    created_streams = []

    def create_stream(*, device):
        stream = SimpleNamespace(device=device)
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", create_stream)
    registry = _CKVPrefetchStateRegistry()
    target_state = registry.for_workspace(target_workspace)
    target_state_reused = registry.for_workspace(target_workspace_reused)
    draft_state = registry.for_workspace(draft_workspace)

    assert created_streams == []
    assert target_state_reused is target_state
    assert target_state.get_gather_stream() is target_state.get_gather_stream()
    assert draft_state.get_gather_stream() is draft_state.get_gather_stream()
    assert target_state.gather_stream is not draft_state.gather_stream
    assert len(created_streams) == 2


def test_ckv_prefetch_ring_survives_intervening_workspace_borrow(monkeypatch):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)
    (lane_workspace,) = manager.get_simultaneous(((256,), torch.uint8))
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(lane_workspace)

    assert state.ckv_workspace is None
    ring = state.get_ckv_workspace(64)
    ring.fill_(0xA5)

    # WorkspaceManager callers all borrow from offset zero. An intervening
    # indexer/MoE scratch allocation must not alias cross-layer CKV state.
    (intervening_workspace,) = manager.get_simultaneous(((128,), torch.uint8))
    intervening_workspace.zero_()

    assert ring.untyped_storage().data_ptr() != (
        intervening_workspace.untyped_storage().data_ptr()
    )
    assert torch.all(ring == 0xA5)


def test_ckv_prefetch_ring_resize_drains_pending_generation():
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(torch.empty(16, dtype=torch.uint8))

    first_ring = state.get_ckv_workspace(64)
    assert state.get_ckv_workspace(64) is first_ring
    assert state.ckv_workspace_generation == 1

    event = _FakeEvent()
    state.pending_layers[1] = (event, 0)
    resized_ring = state.get_ckv_workspace(128)

    assert resized_ring is not first_ring
    assert resized_ring.numel() == 128
    assert state.ckv_workspace_generation == 2
    assert event.wait_calls == 1
    assert state.pending_layers == {}


def test_ckv_prefetch_workspace_identity_invalidates_changed_geometry():
    registry = _CKVPrefetchStateRegistry()
    workspace_buffer = torch.empty(16, dtype=torch.uint8)
    same_geometry = workspace_buffer.view_as(workspace_buffer)
    changed_geometry = workspace_buffer[:8]
    old_state = registry.for_workspace(workspace_buffer)
    event = _FakeEvent()
    old_state.pending_layers[1] = (event, 0)

    assert registry.for_workspace(same_geometry) is old_state

    resized_state = registry.for_workspace(changed_geometry)

    assert resized_state is not old_state
    assert event.wait_calls == 1
    assert len(registry.states) == 1


def test_ckv_prefetch_workspace_identity_tracks_manager_resize(monkeypatch):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    (first_workspace,) = manager.get_simultaneous(((16,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft_workspace,) = manager.get_simultaneous(((16,), torch.uint8))
    registry = _CKVPrefetchStateRegistry()
    cache = torch.empty(0)
    first_state = registry.for_workspace(first_workspace, 0, cache)
    first_state.register_cache(0, cache)
    draft_cache = torch.empty(0)
    draft_state = registry.for_workspace(draft_workspace, 0, draft_cache)
    draft_state.register_cache(0, draft_cache)
    event = _FakeEvent()
    first_state.pending_layers[1] = (event, 0)

    (resized_workspace,) = manager.get_simultaneous(((257,), torch.uint8))
    resized_state = registry.for_workspace(resized_workspace, 0, cache)

    assert (
        _ckv_workspace_identity(first_workspace).storage_generation
        != _ckv_workspace_identity(resized_workspace).storage_generation
    )
    assert resized_state is not first_state
    assert resized_state.layer_caches == []
    assert event.wait_calls == 1
    assert registry.for_workspace(draft_workspace) is draft_state
    assert len(registry.states) == 2


def test_ckv_prefetch_registry_retires_released_profile_workspace():
    registry = _CKVPrefetchStateRegistry()
    profile_workspace = torch.empty(16, dtype=torch.uint8)
    state = registry.for_workspace(profile_workspace)
    event = _FakeEvent()
    state.pending_layers[1] = (event, 0)

    del profile_workspace
    gc.collect()
    registry.begin_step()

    assert event.wait_calls == 1
    assert registry.states == {}


def test_ckv_gather_uses_capture_fallback_without_reading_prefetch_state(
    monkeypatch,
):
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_gather_enabled = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert not impl.dcp_prefill_ckv_gather_eligible(SimpleNamespace(), 128)
