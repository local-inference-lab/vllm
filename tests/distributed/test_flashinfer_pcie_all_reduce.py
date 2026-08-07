# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce
from vllm.distributed.device_communicators import flashinfer_pcie_all_reduce as fi_pcie


class FakeWorkspace:
    instances: list[FakeWorkspace] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.destroyed = False
        self.last_input: torch.Tensor | None = None
        FakeWorkspace.instances.append(self)

    def supports(self, inp: torch.Tensor) -> bool:
        return inp.numel() <= self.kwargs["max_numel"]

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.last_input = inp
        if out is None:
            return inp.clone()
        out.copy_(inp)
        return out

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeWorkspace.instances.clear()
    monkeypatch.setattr(fi_pcie, "_load_workspace_class", lambda: FakeWorkspace)
    monkeypatch.setattr(fi_pcie.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(fi_pcie.dist, "get_world_size", lambda group=None: 2)

    def all_gather_object(gathered, value, *, group):
        del group
        gathered[:] = [value, value]

    monkeypatch.setattr(fi_pcie.dist, "all_gather_object", all_gather_object)


def make_pool() -> fi_pcie.FlashInferPcieIpcAllReducePool:
    return fi_pcie.FlashInferPcieIpcAllReducePool(
        exchange_group=object(),  # type: ignore[arg-type]
        device="cpu",
        max_size=64,
    )


def test_each_semantic_channel_gets_an_independent_workspace() -> None:
    pool = make_pool()
    assert FakeWorkspace.instances == []
    checkpoint = pool.checkpoint_channels()

    pool.prepare_channels(("vllm:target:profile",))
    assert len(FakeWorkspace.instances) == 1
    assert pool.for_stream(channel_id="vllm:target:profile").should_allreduce(
        torch.ones(4)
    )

    profile_workspace = FakeWorkspace.instances[-1]
    pool.rollback_channels(checkpoint)
    assert profile_workspace.destroyed
    assert pool.checkpoint_channels() == ()

    inp = torch.ones(4)
    assert torch.equal(pool.all_reduce(inp), inp)
    eager = FakeWorkspace.instances[-1]
    pool.close()
    assert eager.destroyed


def test_capture_routes_graph_calls_without_reusing_eager_state() -> None:
    pool = make_pool()
    inp = torch.arange(4, dtype=torch.float32)
    out = torch.empty_like(inp)

    with pool.capture(channel_id="vllm:target:production"):
        pool.prepare_graph_all_reduce(inp)
        actual = pool.all_reduce(
            inp,
            out=out,
            channel_id="vllm:target:production",
        )

    assert actual is out
    assert torch.equal(actual, inp)
    assert len(FakeWorkspace.instances) == 1
    assert FakeWorkspace.instances[0].last_input is inp
    pool.close()


def test_channel_creation_fails_closed_when_rank_ids_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = make_pool()

    def divergent(gathered, value, *, group):
        del group
        gathered[:] = [value, "different-channel"]

    monkeypatch.setattr(fi_pcie.dist, "all_gather_object", divergent)
    with pytest.raises(RuntimeError, match="rank-stable"):
        pool.prepare_channels(("vllm:target:production",))
    pool.close()


def test_flashinfer_ipc_is_an_explicit_integrated_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ENABLE_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_PCIE_ALLREDUCE_BACKEND", "flashinfer-ipc")

    assert custom_all_reduce._flashinfer_pcie_allreduce_requested()
    assert not custom_all_reduce._b12x_pcie_allreduce_requested()
    assert custom_all_reduce._get_pcie_allreduce_backend() == "flashinfer-ipc"


def test_custom_allreduce_constructs_flashinfer_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    runtime = MagicMock()

    class FakePool:
        @classmethod
        def from_exchange_group(cls, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return runtime

    def fake_all_gather(gather_list, tensor, *, group) -> None:
        del tensor, group
        for index, slot in enumerate(gather_list):
            slot.fill_(index)

    fake_config = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 4096),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
    )
    monkeypatch.setattr(custom_all_reduce, "custom_ar", True)
    monkeypatch.setattr(custom_all_reduce.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        custom_all_reduce,
        "in_the_same_node_as",
        lambda group, source_rank=0: [True, True],
    )
    monkeypatch.setattr(custom_all_reduce.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(custom_all_reduce.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(custom_all_reduce.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        custom_all_reduce.dist, "all_reduce", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        custom_all_reduce, "_b12x_pcie_allreduce_requested", lambda: False
    )
    monkeypatch.setattr(
        custom_all_reduce, "_flashinfer_pcie_allreduce_requested", lambda: True
    )
    monkeypatch.setattr(
        custom_all_reduce, "_get_pcie_allreduce_backend", lambda: "flashinfer-ipc"
    )
    monkeypatch.setattr(custom_all_reduce, "_can_p2p", lambda *args: True)
    monkeypatch.setattr(custom_all_reduce, "_is_cross_numa_topology", lambda ids: False)
    monkeypatch.setattr(
        custom_all_reduce, "_load_flashinfer_pcie_oneshot_pool", lambda: FakePool
    )
    monkeypatch.setattr(
        custom_all_reduce.current_platform, "get_device_capability", lambda: None
    )
    monkeypatch.setattr(
        custom_all_reduce.current_platform,
        "visible_device_id_to_physical_device_id",
        lambda index: index,
    )
    monkeypatch.setattr(
        custom_all_reduce.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.setattr(
        custom_all_reduce.current_platform,
        "is_fully_connected",
        lambda ids: False,
        raising=False,
    )
    monkeypatch.setattr(custom_all_reduce.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(custom_all_reduce.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: fake_config)
    monkeypatch.setattr(
        custom_all_reduce.envs,
        "VLLM_PCIE_ONESHOT_SINGLE_CHANNEL",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        custom_all_reduce.envs,
        "VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE",
        False,
        raising=False,
    )

    built = custom_all_reduce.CustomAllreduce(
        object(),  # type: ignore[arg-type]
        torch.device("cuda:0"),
        nccl_group=object(),  # type: ignore[arg-type]
    )

    assert not built.disabled
    assert built.backend_name() == "FLASHINFER_PCIE_IPC"
    assert captured["max_size"] > 0
    runtime.prepare_channels.assert_called_once()
    runtime.for_stream.assert_called_once()
