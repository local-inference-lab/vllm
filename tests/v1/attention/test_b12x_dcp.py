# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops import b12x_dcp
from vllm.v1.attention.ops.dcp import MLADCPManager


class _FakeGroup:
    world_size = 16

    def __init__(self) -> None:
        self.device_group = object()


class _RecordingPool:
    def __init__(self) -> None:
        self.calls = []

    def all_gather_heads(self, value, *, channel_id):
        self.calls.append(("gather", value.shape, channel_id))
        return value.repeat(1, 16, 1)

    def lse_reduce_scatter(
        self,
        output,
        lse,
        *,
        out,
        is_lse_base_on_e,
        channel_id,
    ):
        self.calls.append(
            (
                "lse",
                output.shape,
                lse.shape,
                out.shape,
                is_lse_base_on_e,
                channel_id,
            )
        )
        out.zero_()
        return out

    def all_gather_pair(self, first, second, *, channel_id):
        self.calls.append(("pair", first.shape, second.shape, channel_id))
        return first.repeat(1, 16), second.repeat(1, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_query_gather_reuses_dense_mla_pool_geometry(monkeypatch):
    group = _FakeGroup()
    pool = _RecordingPool()
    captured = {}

    def get_pool(*args, **kwargs):
        captured.update(kwargs)
        return pool

    monkeypatch.setattr(b12x_dcp, "_get_pool", get_pool)
    query = torch.zeros((1, 6, 576), device="cuda", dtype=torch.bfloat16)
    result = b12x_dcp.try_b12x_query_gather(
        query,
        group,
        output_head_dim=512,
    )

    assert result is not None
    assert result.shape == (1, 96, 576)
    assert captured["total_heads"] == 96
    assert captured["query_head_dim"] == 576
    assert captured["output_head_dim"] == 512
    assert pool.calls == [("gather", torch.Size((1, 6, 576)), "vllm:eager:dcp")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lse_reduce_scatter_uses_head_major_output(monkeypatch):
    group = _FakeGroup()
    pool = _RecordingPool()
    monkeypatch.setattr(b12x_dcp, "_get_pool", lambda *args, **kwargs: pool)
    output = torch.zeros((1, 96, 512), device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros((1, 96), device="cuda", dtype=torch.float32)

    result = b12x_dcp.try_b12x_lse_reduce_scatter(
        output,
        lse,
        group,
        is_lse_base_on_e=True,
        query_head_dim=576,
    )

    assert result is not None
    assert result.shape == (1, 6, 512)
    assert result.stride() == (512, 512, 1)
    assert pool.calls == [
        (
            "lse",
            torch.Size((1, 96, 512)),
            torch.Size((1, 96)),
            torch.Size((1, 6, 512)),
            True,
            "vllm:eager:dcp",
        )
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_projection_pair_uses_one_pool_operation(monkeypatch):
    group = _FakeGroup()
    pool = _RecordingPool()
    captured = {}

    def get_pool(*args, **kwargs):
        captured.update(kwargs)
        return pool

    monkeypatch.setattr(b12x_dcp, "_get_pool", get_pool)
    down = torch.zeros((1, 224), device="cuda", dtype=torch.bfloat16)
    router = torch.zeros((1, 56), device="cuda", dtype=torch.float32)
    result = b12x_dcp.try_b12x_projection_pair_gather(down, router, group)

    assert result is not None
    assert result[0].shape == (1, 3584)
    assert result[1].shape == (1, 896)
    assert captured["total_heads"] == 16
    assert captured["query_head_dim"] == 672
    assert captured["output_head_dim"] == 672
    assert pool.calls == [
        (
            "pair",
            torch.Size((1, 224)),
            torch.Size((1, 56)),
            "vllm:eager:dcp",
        )
    ]


def test_pool_created_during_capture_joins_graph_owner(monkeypatch):
    group = _FakeGroup()
    events = []

    class Pool:
        def prepare_channels(self, names):
            events.append(("prepare", tuple(names)))

        def for_stream(self, *, channel_id):
            events.append(("eager", channel_id))

        @contextmanager
        def capture(self, *, stream, channel_id):
            events.append(("enter", stream, channel_id))
            try:
                yield self
            finally:
                events.append(("exit", stream, channel_id))

    pool = Pool()
    pool_type = SimpleNamespace(from_exchange_group=lambda **kwargs: pool)
    monkeypatch.setattr(b12x_dcp, "_load_pool_type", lambda: pool_type)
    monkeypatch.setattr(b12x_dcp, "_pool_init_failed", lambda *args: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(b12x_dcp, "_POOLS", {})
    monkeypatch.setattr(b12x_dcp, "_DISABLED", set())
    monkeypatch.setattr(b12x_dcp, "_ACTIVE_CAPTURE", {})
    stream = object()

    with b12x_dcp.capture_b12x_dcp_pools(group, stream, channel_id="vllm:target:test"):
        result = b12x_dcp._get_pool(
            group,
            device=torch.device("cuda", 0),
            total_heads=96,
            output_head_dim=512,
            query_head_dim=576,
            max_batch_size=8,
        )
        assert result is pool
        assert events[-1] == ("enter", stream, "vllm:target:test")

    assert events == [
        ("prepare", ("vllm:eager:dcp", "vllm:target:test")),
        ("enter", stream, "vllm:target:test"),
        ("exit", stream, "vllm:target:test"),
    ]


def test_explicit_b12x_precedes_symmetric_memory(monkeypatch):
    manager = MLADCPManager.__new__(MLADCPManager)
    manager.use_a2a = True
    manager.use_b12x = True
    manager.group = _FakeGroup()
    manager.device = torch.device("cuda", 0)
    manager.max_num_tokens = 4096
    manager.num_ubatches = 1
    manager.padded_num_heads = None
    manager.query_head_dim = 576
    manager.output_head_dim = 512
    monkeypatch.setattr(
        "vllm.v1.attention.ops.dcp.get_direct_dcp_a2a_workspace",
        lambda *args, **kwargs: pytest.fail("direct A2A must not initialize"),
    )
    monkeypatch.setattr(
        "vllm.v1.attention.ops.dcp.get_direct_dcp_q_gather_workspace",
        lambda *args, **kwargs: pytest.fail("direct gather must not initialize"),
    )

    combine = manager._init_combine(6, 512, torch.bfloat16, True, False)
    query_gather = manager._init_query_gather(6, 576, torch.bfloat16)

    assert combine.func.__func__ is MLADCPManager._b12x_combine
    assert query_gather.__func__ is MLADCPManager._b12x_query_gather
