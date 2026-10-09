# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class NvFp4LinearLayerConfig:
    """Configuration for an NVFP4 linear layer.

    Weights are packed uint8 NVFP4 with E4M3 block scales and a scalar global
    weight scale. W4A16 checkpoints consume floating-point activations directly; W4A4
    checkpoints also carry activation quantization scales.
    """

    use_a16: bool = False


class NvFp4LinearKernel(ABC):
    """Base class for NVFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order,calling ``is_supported`` and ``can_implement`` to find the best
    match for the current hardware.
    """

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    def input_quant_key(self) -> QuantKey | None:
        """Return the input quantization key supported by this kernel. If the kernel
        does not support input quantization outside of the kernel, return None.
        """
        return None

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Return whether this kernel can run on the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        """Return whether this kernel can handle *config*."""
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transform weights into the format required by this kernel.

        Called once after checkpoint weights have been loaded onto the
        device.  Implementations should repack / swizzle / pad weights
        and scales in-place on *layer*.
        """
        raise NotImplementedError

    def get_workspace_size(self, layer: torch.nn.Module, rows: int) -> int:
        """Scratch bytes one call at ``rows`` needs from a caller-reserved view."""
        del layer, rows
        return 0

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the quantized GEMM."""
        raise NotImplementedError
