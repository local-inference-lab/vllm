# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Groups over the same ranks share one PyNCCL communicator on request."""

import pytest
import torch

from vllm.distributed.device_communicators import cuda_communicator as cc


class _FakeComm:
    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


@pytest.fixture
def fake_pynccl(monkeypatch):
    import torch.distributed as dist

    from vllm.distributed.device_communicators import pynccl

    monkeypatch.setattr(pynccl, "PyNcclCommunicator", _FakeComm)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group)
    monkeypatch.setattr(cc, "_SHARED_PYNCCL", {})


def test_groups_over_the_same_ranks_share_one_communicator(fake_pynccl, monkeypatch):
    monkeypatch.setenv("VLLM_SHARE_PYNCCL_COMMS", "1")
    device = torch.device("cuda:0")
    tp, tp_key, tp_created = cc._acquire_pynccl([0, 1], device)
    dcp, dcp_key, dcp_created = cc._acquire_pynccl([0, 1], device)
    other, _, other_created = cc._acquire_pynccl([0, 2], device)

    assert tp is dcp and tp_key == dcp_key
    assert tp_created and not dcp_created
    assert other is not tp and other_created

    cc._release_pynccl(dcp, dcp_key)
    assert tp.destroyed == 0
    cc._release_pynccl(tp, tp_key)
    assert tp.destroyed == 1
    assert tp_key not in cc._SHARED_PYNCCL


def test_communicators_are_not_shared_by_default(fake_pynccl, monkeypatch):
    monkeypatch.delenv("VLLM_SHARE_PYNCCL_COMMS", raising=False)
    device = torch.device("cuda:0")
    first, first_key, _ = cc._acquire_pynccl([0, 1], device)
    second, second_key, _ = cc._acquire_pynccl([0, 1], device)

    assert first is not second
    assert first_key is None and second_key is None
    cc._release_pynccl(first, first_key)
    assert first.destroyed == 1 and second.destroyed == 0
