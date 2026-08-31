# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


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
