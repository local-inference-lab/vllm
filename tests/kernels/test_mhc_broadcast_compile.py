# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_broadcast_tilelang


def _inputs(device: str = "meta") -> tuple[torch.Tensor, ...]:
    num_tokens = 6
    hidden_size = 128
    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    return (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device),
        torch.empty(
            hc_mult3,
            hc_mult * hidden_size,
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(3, dtype=torch.float32, device=device),
        torch.empty(hc_mult3, dtype=torch.float32, device=device),
        torch.empty(hidden_size, dtype=torch.bfloat16, device=device),
        torch.empty(
            hc_mult3,
            hidden_size,
            dtype=torch.float32,
            device=device,
        ),
    )


def _run_broadcast_mhc(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    fn_broadcast: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return mhc_pre_broadcast_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        norm_weight=norm_weight,
        fn_broadcast=fn_broadcast,
    )


def _assert_output_contract(outputs: tuple[torch.Tensor, ...]) -> None:
    residual, post_mix, res_mix, layer_input = outputs
    assert residual.shape == (6, 4, 128)
    assert residual.dtype == torch.bfloat16
    assert post_mix.shape == (6, 4, 1)
    assert post_mix.dtype == torch.float32
    assert res_mix.shape == (6, 4, 4)
    assert res_mix.dtype == torch.float32
    assert layer_input.shape == (6, 128)
    assert layer_input.dtype == torch.bfloat16


def test_mhc_pre_broadcast_fake_output_contract() -> None:
    _assert_output_contract(_run_broadcast_mhc(*_inputs()))


def test_mhc_pre_broadcast_is_fullgraph_compile_boundary() -> None:
    compiled = torch.compile(_run_broadcast_mhc, backend="eager", fullgraph=True)
    _assert_output_contract(compiled(*_inputs()))
