# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia import tp_projection


def test_full_width_projection_reduces_large_prefill_in_place(monkeypatch):
    output_parallel = torch.empty(4096, 4)
    calls: list[tuple[str, torch.Tensor]] = []

    def reduce_in_place(value: torch.Tensor) -> torch.Tensor:
        calls.append(("in_place", value))
        return value

    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        reduce_in_place,
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        lambda value: pytest.fail("large prefill must not allocate collective output"),
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 16)

    assert result is output_parallel
    assert calls == [("in_place", output_parallel)]


@pytest.mark.parametrize("num_tokens", [1, 8, 1023])
def test_full_width_projection_preserves_decode_collective(monkeypatch, num_tokens):
    output_parallel = torch.empty(num_tokens, 4)
    reduced = torch.empty_like(output_parallel)
    calls: list[torch.Tensor] = []

    def reduce(value: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return reduced

    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        reduce,
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        lambda value: pytest.fail("decode must retain its functional collective"),
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 16)

    assert result is reduced
    assert calls == [output_parallel]


def test_full_width_projection_is_identity_at_tp1(monkeypatch):
    output_parallel = torch.empty(4096, 4)
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce",
        lambda value: pytest.fail("TP1 must not enter a collective"),
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        lambda value: pytest.fail("TP1 must not enter a collective"),
    )

    result = tp_projection.reduce_kimi_full_width_projection(output_parallel, 1)

    assert result is output_parallel


def test_projection_group_uses_matching_dcp_coordinator(monkeypatch):
    tp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    dcp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(tp_projection, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(tp_projection, "get_dcp_group", lambda: dcp_group)

    assert tp_projection._get_kimi_projection_group() is dcp_group


def test_projection_group_uses_tp_for_different_dcp_ranks(monkeypatch):
    tp_group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    dcp_group = SimpleNamespace(world_size=4, ranks=list(range(4)))
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(tp_projection, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(tp_projection, "get_dcp_group", lambda: dcp_group)

    assert tp_projection._get_kimi_projection_group() is tp_group


def test_projection_group_rejects_incomplete_tp_coordinator(monkeypatch):
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(
        tp_projection,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, ranks=list(range(4))),
    )
    monkeypatch.setattr(
        tp_projection,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=4, ranks=list(range(4))),
    )

    with pytest.raises(RuntimeError, match="does not span"):
        tp_projection._get_kimi_projection_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_removes_each_rank_padding(monkeypatch):
    tp_size = 2
    local_width = 132
    local = torch.arange(local_width, dtype=torch.bfloat16, device="cuda").view(1, -1)
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    received: dict[str, object] = {}
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        received.update(
            transport=transport,
            projection_group=projection_group,
            max_batch_size=max_batch_size,
        )
        padded_width = transport.shape[-1]
        result = torch.full(
            (1, tp_size, padded_width),
            -1,
            dtype=transport.dtype,
            device=transport.device,
        )
        result[0, 0, :local_width] = torch.arange(
            local_width, dtype=transport.dtype, device=transport.device
        )
        result[0, 1, :local_width] = torch.arange(
            local_width,
            2 * local_width,
            dtype=transport.dtype,
            device=transport.device,
        )
        return result

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    expected = torch.arange(
        2 * local_width, dtype=local.dtype, device=local.device
    ).view(1, -1)
    torch.testing.assert_close(actual, expected)
    assert received["projection_group"] is group
    assert received["max_batch_size"] == 1
    transport = received["transport"]
    assert isinstance(transport, torch.Tensor)
    assert transport.shape == (1, 1, 136)


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_preserves_fp32_payload_bits(monkeypatch):
    tp_size = 2
    local = torch.tensor([[1.25, -2.5]], dtype=torch.float32, device="cuda")
    other = torch.tensor([[3.75, -4.5]], dtype=torch.float32, device="cuda")
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    received: dict[str, object] = {}
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        received["transport"] = transport
        result = torch.empty(
            (1, tp_size, transport.shape[-1]),
            dtype=transport.dtype,
            device=transport.device,
        )
        result[0, 0].copy_(transport[0, 0])
        result[0, 1].copy_(other.view(torch.float8_e4m3fn).flatten())
        return result

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    torch.testing.assert_close(actual, torch.cat((local, other), dim=-1))
    assert actual.dtype == torch.float32
    transport = received["transport"]
    assert isinstance(transport, torch.Tensor)
    assert transport.dtype == torch.float8_e4m3fn
    assert transport.shape == (1, 1, 8)


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_b12x_projection_gather_preserves_fp8_payload(monkeypatch):
    tp_size = 2
    local = (
        torch.arange(16, dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(1, -1)
    )
    other = (
        torch.arange(16, 32, dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(1, -1)
    )
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    monkeypatch.setattr(tp_projection.envs, "VLLM_USE_B12X_DCP_A2A", True)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(transport, projection_group, *, max_batch_size):
        return torch.stack((transport[:, 0], other), dim=1)

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_heads", gather)

    actual = tp_projection.gather_kimi_sharded_projection(local)

    torch.testing.assert_close(
        actual.float(),
        torch.cat((local, other), dim=-1).float(),
    )
    assert actual.dtype == torch.float8_e4m3fn


def test_projection_gather_uses_standard_collective_outside_decode(monkeypatch):
    local = torch.arange(12).view(3, 4)
    expected = torch.cat((local, local), dim=-1)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_gather",
        lambda output, dim: expected,
    )

    actual = tp_projection.gather_kimi_sharded_projection(local)

    assert actual is expected


def test_projection_pair_is_identity_at_tp1(monkeypatch):
    first = torch.empty(2, 8)
    second = torch.empty(2, 4)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 1
    )

    actual = tp_projection.gather_kimi_sharded_projection_pair(first, second)

    assert actual[0] is first
    assert actual[1] is second


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
def test_projection_pair_uses_one_b12x_barrier(monkeypatch):
    tp_size = 2
    first = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    second = torch.empty(2, 4, dtype=torch.float32, device="cuda")
    expected = (
        torch.empty(2, 16, dtype=first.dtype, device=first.device),
        torch.empty(2, 8, dtype=second.dtype, device=second.device),
    )
    group = SimpleNamespace(world_size=tp_size, ranks=list(range(tp_size)))
    received: dict[str, object] = {}
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather_pair(local_first, local_second, projection_group, *, max_batch_size):
        received.update(
            local_first=local_first,
            local_second=local_second,
            projection_group=projection_group,
            max_batch_size=max_batch_size,
        )
        return expected

    monkeypatch.setattr(tp_projection, "dcp_b12x_all_gather_pair", gather_pair)

    actual = tp_projection.gather_kimi_sharded_projection_pair(first, second)

    assert actual is expected
    assert received["local_first"] is first
    assert received["local_second"] is second
    assert received["projection_group"] is group
    assert received["max_batch_size"] == 8


def test_projection_pair_uses_exact_separate_fallback(monkeypatch):
    first = torch.empty(9, 8)
    second = torch.empty(9, 4)
    gathered_first = torch.empty(9, 16)
    gathered_second = torch.empty(9, 8)
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )

    def gather_one(local: torch.Tensor) -> torch.Tensor:
        calls.append(local)
        return gathered_first if local is first else gathered_second

    monkeypatch.setattr(tp_projection, "gather_kimi_sharded_projection", gather_one)

    actual = tp_projection.gather_kimi_sharded_projection_pair(first, second)

    assert actual[0] is gathered_first
    assert actual[1] is gathered_second
    assert calls[0] is first
    assert calls[1] is second


def test_projection_pair_topk_uses_available_b12x_binding(monkeypatch):
    local_down = torch.empty(1, 448)
    local_router = torch.empty(1, 112)
    correction_bias = torch.empty(896)
    group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    expected = (torch.empty(1, 3584), torch.empty(2, 16))
    received: dict[str, object] = {}
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def pair_topk(local_down_arg, local_router_arg, bias_arg, group_arg):
        received.update(
            local_down=local_down_arg,
            local_router=local_router_arg,
            correction_bias=bias_arg,
            group=group_arg,
        )
        return expected

    monkeypatch.setattr(
        tp_projection.dcp_alltoall,
        "try_dcp_b12x_all_gather_pair_kimi_topk",
        pair_topk,
        raising=False,
    )

    actual = tp_projection.try_gather_kimi_sharded_projection_pair_topk(
        local_down,
        local_router,
        correction_bias,
    )

    assert actual is expected
    assert received == {
        "local_down": local_down,
        "local_router": local_router,
        "correction_bias": correction_bias,
        "group": group,
    }


def test_projection_pair_topk_returns_none_without_b12x_binding(monkeypatch):
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.delattr(
        tp_projection.dcp_alltoall,
        "try_dcp_b12x_all_gather_pair_kimi_topk",
        raising=False,
    )
    monkeypatch.setattr(
        tp_projection,
        "_get_kimi_projection_group",
        lambda: pytest.fail("an unavailable binding must not resolve a group"),
    )

    actual = tp_projection.try_gather_kimi_sharded_projection_pair_topk(
        torch.empty(1, 448),
        torch.empty(1, 112),
        torch.empty(896),
    )

    assert actual is None
