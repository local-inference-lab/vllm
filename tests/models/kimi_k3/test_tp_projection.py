# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.kimi_k3.nvidia import tp_projection


def test_projection_group_prefers_identical_dcp_ranks(monkeypatch) -> None:
    tp_group = SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3])
    dcp_group = SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3])
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(tp_projection, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(tp_projection, "get_dcp_group", lambda: dcp_group)

    assert tp_projection._get_kimi_projection_group() is dcp_group

    dcp_group.ranks = [3, 2, 1, 0]
    assert tp_projection._get_kimi_projection_group() is tp_group


def test_projection_group_rejects_incomplete_tp_coordinator(monkeypatch) -> None:
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        tp_projection,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, ranks=[0, 1]),
    )

    with pytest.raises(RuntimeError, match="does not span tensor-parallel ranks"):
        tp_projection._get_kimi_projection_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b12x_single_gather_strips_each_bf16_rank_padding(monkeypatch) -> None:
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)
    captured = {}

    def gather(value, actual_group, *, output_head_dim):
        captured["shape"] = tuple(value.shape)
        captured["output_head_dim"] = output_head_dim
        assert actual_group is group
        return torch.cat((value, value + 10), dim=1)

    monkeypatch.setattr(tp_projection, "try_b12x_query_gather", gather)
    local = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.bfloat16)

    output = tp_projection._try_b12x_kimi_projection_gather(local)

    assert output is not None
    assert captured == {"shape": (1, 1, 8), "output_head_dim": 8}
    torch.testing.assert_close(
        output,
        torch.tensor([[1, 2, 3, 11, 12, 13]], device="cuda", dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b12x_single_gather_preserves_fp32_payload_bits(monkeypatch) -> None:
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)

    def gather(value, actual_group, *, output_head_dim):
        assert actual_group is group
        assert value.dtype == torch.float8_e4m3fn
        assert tuple(value.shape) == (1, 1, 16)
        assert output_head_dim == 16
        return value.repeat(1, 2, 1)

    monkeypatch.setattr(tp_projection, "try_b12x_query_gather", gather)
    local = torch.tensor(
        [[-3.25, 0.0, 1.5, float("inf")]], device="cuda", dtype=torch.float32
    )

    output = tp_projection._try_b12x_kimi_projection_gather(local)

    assert output is not None
    torch.testing.assert_close(output, local.repeat(1, 2), equal_nan=True)


def test_projection_pair_uses_one_b12x_operation(monkeypatch) -> None:
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)
    first = torch.zeros((1, 4))
    second = torch.ones((1, 2))
    expected = (torch.full((1, 8), 2.0), torch.full((1, 4), 3.0))
    fast_gather = Mock(return_value=expected)
    monkeypatch.setattr(tp_projection, "try_b12x_projection_pair_gather", fast_gather)
    generic_gather = Mock(side_effect=AssertionError("generic gather was used"))
    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_all_gather", generic_gather
    )

    actual = tp_projection.gather_kimi_sharded_projection_pair(
        first,
        second,
        use_b12x=True,
    )

    assert actual is expected
    fast_gather.assert_called_once_with(first, second, group)
    generic_gather.assert_not_called()


def test_projection_pair_falls_back_to_exact_tp_gathers(monkeypatch) -> None:
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(tp_projection, "_get_kimi_projection_group", lambda: group)
    monkeypatch.setattr(
        tp_projection, "try_b12x_projection_pair_gather", lambda *args: None
    )
    generic_gather = Mock(side_effect=lambda value, dim: value.repeat(1, 2))
    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_all_gather", generic_gather
    )
    first = torch.zeros((9, 4))
    second = torch.ones((9, 2))

    actual_first, actual_second = tp_projection.gather_kimi_sharded_projection_pair(
        first,
        second,
        use_b12x=True,
    )

    torch.testing.assert_close(actual_first, first.repeat(1, 2))
    torch.testing.assert_close(actual_second, second.repeat(1, 2))
    assert generic_gather.call_count == 2
