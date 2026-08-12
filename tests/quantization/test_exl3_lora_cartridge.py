# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for EXL3 LoRA cartridge support.

Tests the cartridge data structures, the _apply_expert patching mechanism,
and the adapter loading logic. Uses mocks for the EXL3 GEMM kernel.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.quantization.exl3_lora_cartridge import (
    Exl3LoraCartridge,
    apply_exl3_cartridge,
    exl3_get_supported_lora_modules,
    load_cartridge_from_adapter,
)


def test_get_supported_lora_modules():
    """Test that EXL3 returns the correct LoRA-capable module names."""
    modules = exl3_get_supported_lora_modules()
    assert "gate_proj" in modules
    assert "up_proj" in modules
    assert "down_proj" in modules
    assert len(modules) == 3


def test_cartridge_init():
    """Test Exl3LoraCartridge initialization."""
    cart = Exl3LoraCartridge(num_stages=2, num_experts=256, device=torch.device("cpu"))
    assert cart.num_stages == 2
    assert cart.num_experts == 256
    assert not cart.active
    assert len(cart.stages) == 2
    assert all(isinstance(s, dict) for s in cart.stages)


def test_cartridge_set_get_tensors():
    """Test setting and getting cartridge tensors."""
    cart = Exl3LoraCartridge(num_stages=1, num_experts=4, device=torch.device("cpu"))
    trellis = torch.zeros(8, 16, 32, dtype=torch.int16)
    suh = torch.ones(128, dtype=torch.float16)
    svh = torch.ones(64, dtype=torch.float16)
    scale = 2.5

    cart.set_stage_tensors(0, 0, "w1", trellis, suh, svh, scale)
    result = cart.get_stage_tensors(0, 0, "w1")
    assert result is not None
    assert torch.equal(result["trellis"], trellis)
    assert result["scale"] == 2.5

    # Non-existent expert
    assert cart.get_stage_tensors(0, 1, "w1") is None


def test_cartridge_clear():
    """Test clearing cartridge tensors."""
    cart = Exl3LoraCartridge(num_stages=1, num_experts=4, device=torch.device("cpu"))
    cart.set_stage_tensors(0, 0, "w1", torch.zeros(8, 16, 32, dtype=torch.int16),
                           torch.ones(128, dtype=torch.float16),
                           torch.ones(64, dtype=torch.float16), 1.0)
    cart.active = True
    cart.clear()
    assert not cart.active
    assert len(cart.stages[0]) == 0


def test_apply_cartridge_noop():
    """Test that apply_exl3_cartridge is a no-op when cartridge is None or inactive."""
    base_output = torch.randn(4, 64, dtype=torch.float16)
    x = torch.randn(4, 128, dtype=torch.float16)
    layer = MagicMock()

    # No cartridge
    result = apply_exl3_cartridge(base_output, x, layer, "w13", 0, "w1", None)
    assert torch.equal(result, base_output)

    # Inactive cartridge
    cart = Exl3LoraCartridge(1, 4, torch.device("cpu"))
    result = apply_exl3_cartridge(base_output, x, layer, "w13", 0, "w1", cart)
    assert torch.equal(result, base_output)


def test_apply_cartridge_with_stage():
    """Test that apply_exl3_cartridge applies correction when cartridge is active."""
    base_output = torch.randn(4, 64, dtype=torch.float16)
    x = torch.randn(4, 128, dtype=torch.float16)
    layer = MagicMock()
    cart = Exl3LoraCartridge(1, 4, torch.device("cpu"))
    cart.active = True
    cart.set_stage_tensors(0, 0, "w1",
                           torch.zeros(8, 16, 32, dtype=torch.int16),
                           torch.ones(128, dtype=torch.float16),
                           torch.ones(64, dtype=torch.float16),
                           1.0)

    with patch("vllm.model_executor.layers.quantization.exl3_lora_cartridge._exl3_gemm") as mock_gemm:
        mock_gemm.return_value = torch.ones(4, 64, dtype=torch.float16)
        result = apply_exl3_cartridge(base_output, x, layer, "w13", 0, "w1", cart)
        # Should be base + cartridge_output * (1/scale) = base + ones * 1.0
        expected = base_output + torch.ones(4, 64, dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-5)
        mock_gemm.assert_called_once()


def test_cartridge_scale_correction():
    """Test that rescaling factor is applied correctly."""
    base_output = torch.zeros(4, 64, dtype=torch.float16)
    x = torch.randn(4, 128, dtype=torch.float16)
    layer = MagicMock()
    cart = Exl3LoraCartridge(1, 4, torch.device("cpu"))
    cart.active = True
    scale = 3.0  # output should be cartridge_output / 3.0
    cart.set_stage_tensors(0, 0, "w1",
                           torch.zeros(8, 16, 32, dtype=torch.int16),
                           torch.ones(128, dtype=torch.float16),
                           torch.ones(64, dtype=torch.float16),
                           scale)

    with patch("vllm.model_executor.layers.quantization.exl3_lora_cartridge._exl3_gemm") as mock_gemm:
        mock_gemm.return_value = torch.full((4, 64), 9.0, dtype=torch.float16)
        result = apply_exl3_cartridge(base_output, x, layer, "w13", 0, "w1", cart)
        # Should be 0 + 9.0 / 3.0 = 3.0
        expected = torch.full((4, 64), 3.0, dtype=torch.float16)
        assert torch.allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
