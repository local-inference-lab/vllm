# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.qwen3_dflash2 as dflash2_module
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The fused grouped convolution requires CUDA",
)


def _reference_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    *,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    position = position & (block_size - 1)
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 16, 17, 64])
def test_dflash2_grouped_conv_taps2_matches_bf16_reference(rows: int) -> None:
    torch.manual_seed(20260901 + rows)
    block_size = 8
    num_groups = 256
    group_size = 16
    taps = 2
    hidden_size = num_groups * group_size
    hidden_states = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    delta = torch.randn(
        rows, taps, num_groups, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    base = torch.randn(
        taps, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()

    actual = _grouped_conv(
        hidden_states,
        delta,
        base,
        block_size,
        num_groups,
        group_size,
        taps,
    )
    expected = _reference_grouped_conv(
        hidden_states,
        delta,
        base,
        block_size=block_size,
        num_groups=num_groups,
        group_size=group_size,
        taps=taps,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("broadcast_axis", ["row", "tap", "group"])
def test_dflash2_grouped_conv_broadcast_delta_uses_generic_path(
    broadcast_axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(20260901)
    rows = 8
    block_size = 8
    num_groups = 256
    group_size = 16
    taps = 2
    hidden_size = num_groups * group_size
    delta_shapes = {
        "row": (1, taps, num_groups),
        "tap": (rows, 1, num_groups),
        "group": (rows, taps, 1),
    }
    hidden_states = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    delta = torch.randn(
        delta_shapes[broadcast_axis], device="cuda", dtype=torch.bfloat16
    ).contiguous()
    base = torch.randn(
        taps, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    expected = _reference_grouped_conv(
        hidden_states,
        delta,
        base,
        block_size=block_size,
        num_groups=num_groups,
        group_size=group_size,
        taps=taps,
    )

    def reject_fused_path(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("broadcast delta must use the generic grouped convolution")

    monkeypatch.setattr(
        dflash2_module, "_dflash2_grouped_conv_taps2", reject_fused_path
    )
    actual = _grouped_conv(
        hidden_states,
        delta,
        base,
        block_size,
        num_groups,
        group_size,
        taps,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dflash2_grouped_conv_taps2_replays_under_cuda_graph() -> None:
    torch.manual_seed(20260901)
    rows = 8
    block_size = 8
    num_groups = 256
    group_size = 16
    taps = 2
    hidden_size = num_groups * group_size
    hidden_states = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    delta = torch.randn(
        rows, taps, num_groups, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    base = torch.randn(
        taps, hidden_size, device="cuda", dtype=torch.bfloat16
    ).contiguous()

    # Compile the Triton specialization before entering CUDA graph capture.
    _grouped_conv(
        hidden_states,
        delta,
        base,
        block_size,
        num_groups,
        group_size,
        taps,
    )
    torch.accelerator.synchronize()

    device_module = torch.get_device_module(hidden_states.device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        captured = _grouped_conv(
            hidden_states,
            delta,
            base,
            block_size,
            num_groups,
            group_size,
            taps,
        )
    graph.replay()
    torch.accelerator.synchronize()

    expected = _reference_grouped_conv(
        hidden_states,
        delta,
        base,
        block_size=block_size,
        num_groups=num_groups,
        group_size=group_size,
        taps=taps,
    )
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
