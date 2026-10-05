# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import torch

from vllm.models.common.ops import fused_allreduce_rms_norm as fused_module


def _operands() -> tuple[torch.Tensor, torch.Tensor, MagicMock]:
    hidden_states = torch.randn(4, 16)
    residual = torch.randn_like(hidden_states)
    norm = MagicMock()
    norm.weight = torch.randn(16)
    norm.variance_epsilon = 1e-5
    return hidden_states, residual, norm


def test_b12x_fused_dispatch_precedes_other_backends(
    monkeypatch,
) -> None:
    hidden_states, residual, norm = _operands()
    communicator = MagicMock()
    communicator.try_fused_add_rms_norm.return_value = True
    monkeypatch.setattr(fused_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(fused_module, "get_b12x_pcie_allreduce", lambda: communicator)
    monkeypatch.setattr(
        fused_module,
        "flashinfer_trtllm_fused_allreduce_norm",
        MagicMock(),
    )
    fallback = MagicMock()
    monkeypatch.setattr(fused_module, "tensor_model_parallel_all_reduce", fallback)

    output, residual_output = fused_module.fused_allreduce_rms_norm(
        hidden_states, residual, norm
    )

    assert output is hidden_states
    assert residual_output is residual
    communicator.try_fused_add_rms_norm.assert_called_once_with(
        hidden_states,
        residual,
        norm.weight,
        norm.variance_epsilon,
    )
    fused_module.flashinfer_trtllm_fused_allreduce_norm.assert_not_called()
    fallback.assert_not_called()
    norm.assert_not_called()


def test_b12x_rejection_preserves_explicit_fallback(monkeypatch) -> None:
    hidden_states, residual, norm = _operands()
    communicator = MagicMock()
    communicator.try_fused_add_rms_norm.return_value = False
    reduced = torch.randn_like(hidden_states)
    expected_output = torch.randn_like(hidden_states)
    expected_residual = torch.randn_like(residual)
    norm.return_value = expected_output, expected_residual
    monkeypatch.setattr(fused_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(fused_module, "get_b12x_pcie_allreduce", lambda: communicator)
    monkeypatch.setattr(fused_module, "flashinfer_trtllm_fused_allreduce_norm", None)
    fallback = MagicMock(return_value=reduced)
    monkeypatch.setattr(fused_module, "tensor_model_parallel_all_reduce", fallback)

    output, residual_output = fused_module.fused_allreduce_rms_norm(
        hidden_states, residual, norm
    )

    assert output is expected_output
    assert residual_output is expected_residual
    fallback.assert_called_once_with(hidden_states)
    norm.assert_called_once_with(reduced, residual)


def test_tensor_parallel_one_bypasses_communicators(monkeypatch) -> None:
    hidden_states, residual, norm = _operands()
    expected_output = torch.randn_like(hidden_states)
    expected_residual = torch.randn_like(residual)
    norm.return_value = expected_output, expected_residual
    get_b12x = MagicMock()
    monkeypatch.setattr(fused_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(fused_module, "get_b12x_pcie_allreduce", get_b12x)

    output, residual_output = fused_module.fused_allreduce_rms_norm(
        hidden_states, residual, norm
    )

    assert output is expected_output
    assert residual_output is expected_residual
    norm.assert_called_once_with(hidden_states, residual)
    get_b12x.assert_not_called()
