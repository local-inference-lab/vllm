# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.v1.worker.workspace as workspace


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    ubatch_id = 0
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: ubatch_id)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    (target,) = manager.get_simultaneous(((16,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((16,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

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


def test_workspace_lane_validation() -> None:
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
