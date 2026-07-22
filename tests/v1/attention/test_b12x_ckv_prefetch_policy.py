# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.v1.attention.backends.mla import b12x_sparse_ckv_decode
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


class _FakeStream:
    def __init__(self):
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


def test_ckv_bulk_prefetch_ticket_waits_once_for_three_shared_layers():
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(torch.empty(16, dtype=torch.uint8))
    event = _FakeEvent()
    stream = _FakeStream()

    ticket = state.register_pending_group({3: 0, 4: 1, 5: 2}, event)
    pending = [state.pop_pending_layer(layer) for layer in (3, 4, 5)]

    assert all(item is not None for item in pending)
    assert all(item.ticket is ticket for item in pending if item is not None)
    for item in pending:
        assert item is not None
        item.ticket.wait_on_stream_once(stream)

    assert stream.waited_events == [event]
    assert state.pending_layers == {}


def test_ckv_prefetch_incomplete_step_recovers_with_stream_wait():
    registry = _CKVPrefetchStateRegistry()
    state = registry.for_workspace(torch.empty(16, dtype=torch.uint8))
    cache = torch.empty(0)
    event = _FakeEvent()
    state.register_cache(3, cache)
    state.register_pending_group({3: 0}, event)
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
    shared_workspace = torch.empty(16, dtype=torch.uint8)
    target_lane = ("target", 1)
    draft_lane = ("draft", 2)
    registry = _CKVPrefetchStateRegistry()
    target_state = registry.for_workspace(
        shared_workspace, execution_lane_key=target_lane
    )
    draft_state = registry.for_workspace(
        shared_workspace, execution_lane_key=draft_lane
    )
    target_cache = torch.empty(0)
    draft_cache = torch.empty(0)
    target_event = _FakeEvent()
    target_ring = target_state.get_ckv_workspace(64)
    draft_ring = draft_state.get_ckv_workspace(64)

    target_state.register_cache(1, target_cache)
    target_pending = target_state.register_pending_group({1: 1}, target_event)
    draft_state.register_cache(1, draft_cache)

    class SparseState:
        def __init__(self, *, layout, device, exchange):
            self.layout = layout
            self.device = device
            self.exchange = exchange

    monkeypatch.setattr(b12x_sparse_ckv_decode, "SparseCKVDecodeState", SparseState)
    target_exchange = object()
    draft_exchange = object()
    layout = object()
    target_sparse = target_state.get_sparse_decode_state(layout, target_exchange)
    draft_sparse = draft_state.get_sparse_decode_state(layout, draft_exchange)

    assert target_state is not draft_state
    assert target_ring.untyped_storage().data_ptr() != (
        draft_ring.untyped_storage().data_ptr()
    )
    assert target_state.layer_caches[1] is target_cache
    assert target_state.pending_layers[1].ticket is target_pending
    assert target_state.pending_layers[1].buf_idx == 1
    assert draft_state.layer_caches[1] is draft_cache
    assert draft_state.pending_layers == {}
    assert target_sparse.exchange is target_exchange
    assert draft_sparse.exchange is draft_exchange


def test_ckv_prefetch_lazily_owns_one_stream_per_workspace_lane(monkeypatch):
    current_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: current_ubatch[0])
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    (target_workspace,) = manager.get_simultaneous(((16,), torch.uint8))
    (target_workspace_reused,) = manager.get_simultaneous(((16,), torch.uint8))
    current_ubatch[0] = 1
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
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_ubatches=1)
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
    state.register_pending_group({1: 0}, event)
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
    old_state.register_pending_group({1: 0}, event)

    assert registry.for_workspace(same_geometry) is old_state

    resized_state = registry.for_workspace(changed_geometry)

    assert resized_state is not old_state
    assert event.wait_calls == 1
    assert len(registry.states) == 1


def test_ckv_prefetch_resize_retires_only_matching_execution_lane():
    workspace_buffer = torch.empty(257, dtype=torch.uint8)
    first_workspace = workspace_buffer[:16]
    draft_workspace = first_workspace
    resized_workspace = workspace_buffer
    target_lane = ("target", 1)
    draft_lane = ("draft", 2)
    registry = _CKVPrefetchStateRegistry()
    cache = torch.empty(0)
    first_state = registry.for_workspace(
        first_workspace,
        0,
        cache,
        execution_lane_key=target_lane,
    )
    first_state.register_cache(0, cache)
    draft_cache = torch.empty(0)
    draft_state = registry.for_workspace(
        draft_workspace,
        0,
        draft_cache,
        execution_lane_key=draft_lane,
    )
    draft_state.register_cache(0, draft_cache)
    event = _FakeEvent()
    first_state.register_pending_group({1: 0}, event)

    resized_state = registry.for_workspace(
        resized_workspace,
        0,
        cache,
        execution_lane_key=target_lane,
    )

    assert _ckv_workspace_identity(first_workspace) != _ckv_workspace_identity(
        resized_workspace
    )
    assert resized_state is not first_state
    assert resized_state.layer_caches == []
    assert event.wait_calls == 1
    assert (
        registry.for_workspace(draft_workspace, execution_lane_key=draft_lane)
        is draft_state
    )
    assert len(registry.states) == 2


def test_ckv_prefetch_registry_retires_released_profile_workspace():
    registry = _CKVPrefetchStateRegistry()
    profile_workspace = torch.empty(16, dtype=torch.uint8)
    state = registry.for_workspace(profile_workspace)
    event = _FakeEvent()
    state.register_pending_group({1: 0}, event)

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
