# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 2),
        (False, _SpecConfig(True), 1),
        (True, _SpecConfig(False), 1),
        (True, None, 1),
    ],
)
def test_workspace_lane_count_is_dspark_only(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]
    assert manager.available_bytes() == 0

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    assert manager.available_bytes() == 512
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
        assert manager.available_bytes() == 256
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    assert manager.available_bytes() == 512

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_preallocated_workspace_view_restores_context() -> None:
    outer = torch.empty(512, dtype=torch.uint8)
    inner = torch.empty(256, dtype=torch.uint8)

    assert workspace.current_preallocated_workspace() is None
    with workspace.use_preallocated_workspace(outer):
        assert workspace.current_preallocated_workspace() is outer
        with workspace.use_preallocated_workspace(inner):
            assert workspace.current_preallocated_workspace() is inner
        assert workspace.current_preallocated_workspace() is outer
    assert workspace.current_preallocated_workspace() is None


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lock_blocks_growth_and_unlock_restores(monkeypatch) -> None:
    """Once locked, oversized requests fail loudly instead of reallocating the
    buffer that captured CUDA graphs point at; unlock restores growth."""
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    (buf,) = manager.get_simultaneous(((256,), torch.uint8))
    manager.lock()
    assert manager.is_locked()

    # Requests within the reserved size still reuse the same buffer.
    (same,) = manager.get_simultaneous(((256,), torch.uint8))
    (smaller,) = manager.get_simultaneous(((8,), torch.uint8))
    assert same.data_ptr() == buf.data_ptr()
    assert smaller.data_ptr() == buf.data_ptr()

    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((512,), torch.uint8))

    manager.unlock()
    (grown,) = manager.get_simultaneous(((512,), torch.uint8))
    assert grown.numel() == 512


def test_workspace_reservation_covers_every_execution_slot(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    manager.reserve_all(((257,), torch.uint8), ((1,), torch.float32))

    assert [
        buffer.numel() if buffer is not None else 0
        for buffer in manager._current_workspaces
    ] == [768, 768, 768, 768]
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            workspace_id = ubatch_id * 2 + lane
            with workspace.use_workspace_lane(lane):
                (view,) = manager.get_simultaneous(((8,), torch.uint8))
            reserved = manager._current_workspaces[workspace_id]
            assert reserved is not None
            assert view.data_ptr() == reserved.data_ptr()

    manager.lock()
    with pytest.raises(AssertionError, match="reserve_all"):
        manager.reserve_all(((1024,), torch.uint8))


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


def test_profile_reserves_model_lanes_without_replicating_target_capacity(
    monkeypatch,
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    monkeypatch.setattr(workspace, "_manager", manager)
    calls: list[tuple[int, int]] = []

    class ScratchOwner(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size

        def reserve_profile_scratch(self):
            calls.append((self.size, workspace._workspace_lane.get()))
            manager.get_simultaneous(((self.size,), torch.uint8))

    target = ScratchOwner(4096)
    draft = ScratchOwner(512)
    shared = ScratchOwner(256)
    target.add_module("shared", shared)
    draft.add_module("shared", shared)
    runner = SimpleNamespace(
        get_model=lambda: target,
        get_draft_model=lambda: draft,
        _draft_workspace_lane=1,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "target": target,
                "alias": target,
                "draft": draft,
                "shared": shared,
            }
        ),
    )
    GPUModelRunner._reserve_profile_scratch(runner)

    assert calls == [(4096, 0), (512, 1), (256, 0), (256, 1)]
    buffers = [buffer for buffer in manager._current_workspaces if buffer is not None]
    assert [buffer.numel() for buffer in buffers] == [4096, 512, 4096, 512]
    pointers = [buffer.data_ptr() for buffer in buffers]
    assert len(set(pointers)) == 4
    manager.lock()
    manager.reserve_by_lane()
    for ubatch in range(2):
        active_ubatch[0] = ubatch
        for lane, size in enumerate((4096, 512)):
            with workspace.use_workspace_lane(lane):
                (view,) = manager.get_simultaneous(((size,), torch.uint8))
            assert view.data_ptr() == pointers[ubatch * 2 + lane]


def test_lane_reservation_rejects_unreserved_microbatch_when_locked(monkeypatch):
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    manager.get_simultaneous(((256,), torch.uint8))
    manager.lock()
    with pytest.raises(AssertionError, match="reserve_by_lane"):
        manager.reserve_by_lane()


@pytest.mark.parametrize("scoped", [False, True])
def test_preparation_reservation_preserves_per_lane_profile_capacity(
    monkeypatch, scoped
):
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    manager.get_simultaneous(((4096,), torch.uint8))
    with workspace.use_workspace_lane(1):
        manager.get_simultaneous(((256,), torch.uint8))
    requests = []
    for name, size in (("target", 1024), ("draft", 512), ("shared", 768)):
        spec = SimpleNamespace(shape=(size,), dtype=torch.uint8)
        requests.append(
            SimpleNamespace(
                name=name,
                plan=SimpleNamespace(scratch_specs=lambda spec=spec: (spec,)),
            )
        )
    job = SimpleNamespace(
        advance=lambda **kwargs: SimpleNamespace(done=True, pending_compilation=False),
        result=lambda: SimpleNamespace(close=lambda: None),
        close=lambda: None,
    )
    coordinator = B12xPreparationCoordinator(
        SimpleNamespace(begin=lambda *args, **kwargs: job),
        [(tuple(requests), False)],
        global_rank=0,
        world_group=None,
        process_local_only=True,
        workspace=manager,
        request_workspace_lanes=(
            {"target": (0,), "draft": (1,), "shared": (0, 1)} if scoped else None
        ),
    )
    result = coordinator.advance()
    assert result["done"] and result["error"] is None
    buffers = [buffer for buffer in manager._current_workspaces if buffer is not None]
    expected = [4096, 768, 4096, 768] if scoped else [4096] * 4
    assert [buffer.numel() for buffer in buffers] == expected
    assert len({buffer.data_ptr() for buffer in buffers}) == 4
    manager.lock()
    manager.reserve_by_lane()


def test_cuda_graph_capture_resources_are_scoped_to_collector() -> None:
    outside = object()
    first = object()
    nested = object()
    second = object()

    assert not workspace.retain_cuda_graph_capture_resource(outside)
    with workspace.collect_cuda_graph_capture_resources() as resources:
        assert workspace.retain_cuda_graph_capture_resource(first)
        with workspace.collect_cuda_graph_capture_resources() as nested_resources:
            assert workspace.retain_cuda_graph_capture_resource(nested)
        assert workspace.retain_cuda_graph_capture_resource(second)

    assert resources == [first, second]
    assert nested_resources == [nested]
    assert not workspace.retain_cuda_graph_capture_resource(outside)


def test_suspended_graph_resources_restore_collector_after_failure() -> None:
    owner = object()
    with workspace.collect_cuda_graph_capture_resources() as resources:
        with (
            pytest.raises(ValueError),
            workspace.suspend_cuda_graph_capture_resources(),
        ):
            assert not workspace.retain_cuda_graph_capture_resource(object())
            raise ValueError("eager operation failed")
        assert workspace.retain_cuda_graph_capture_resource(owner)
    assert resources == [owner]


@pytest.mark.parametrize("image_tokens", [0, 384])
def test_dsv4_metadata_free_profile_does_not_reserve_split_attention(
    monkeypatch, image_tokens
) -> None:
    """Prepared state declarations own attention scratch before KV admission."""
    from b12x.attention import compressed_sparse_mla

    from vllm.models.deepseek_v4.nvidia import b12x as dsv4

    requested = []
    monkeypatch.setattr(
        workspace,
        "_manager",
        SimpleNamespace(get_simultaneous=lambda *specs: requested.append(specs)),
    )
    monkeypatch.setattr(
        dsv4, "_require_b12x_compressed_sparse_mla", lambda: compressed_sparse_mla
    )
    monkeypatch.setattr(
        dsv4, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    layer = dsv4.DeepseekV4B12xAttention.__new__(dsv4.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = 128
    layer.max_model_len = 1048576
    layer.max_num_batched_tokens = 4096
    layer.window_size = 128
    layer.max_image_tokens = image_tokens
    layer.swa_cache_layer = SimpleNamespace(block_size=256)
    layer.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dspark=lambda: True, num_speculative_tokens=5
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=8),
    )
    q = torch.randn((3, 32, 512), dtype=torch.bfloat16)
    original_q = q.clone()
    output = torch.full_like(q, float("nan"))
    layer.forward_mqa(q, torch.empty(0), torch.arange(3), output)

    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    assert torch.count_nonzero(output) == 0
    assert requested == []


def test_dsv4_profile_retains_padded_query_reservation() -> None:
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4.nvidia.b12x import DeepseekV4B12xAttention

    assert (
        DeepseekV4B12xAttention.reserve_profile_scratch
        is DeepseekV4Attention.reserve_profile_scratch
    )
    layer = DeepseekV4B12xAttention.__new__(DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_torch_dtype = torch.uint8
    device = torch.device("cuda:0")
    layer.q_norm = SimpleNamespace(weight=SimpleNamespace(device=device))
    layer._q_padded_scratch_num_ubatches = 2
    layer.max_num_batched_tokens = 4096
    layer.padded_heads = 32
    layer.head_dim = 512
    layer._q_padded_scratch_dtype = torch.bfloat16
    reserved = []
    layer._reserve_q_padded_scratch_buffer = lambda *args: reserved.append(args)

    layer.reserve_profile_scratch()

    assert reserved == [
        (4096, 32, 512, torch.bfloat16, device, ubatch) for ubatch in range(2)
    ]
