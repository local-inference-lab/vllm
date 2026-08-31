# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from vllm.v1.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    def unbind_kv_cache(self) -> None:
        """Release cache tensors and views retained by this layer.

        This is the lifecycle inverse of :meth:`bind_kv_cache`. Subclasses
        that derive cache views, bindings, or plans must clear those references
        before delegating here.
        """
        self.kv_cache = torch.tensor([])
        impl = getattr(self, "impl", None)
        if impl is not None:
            # Quantized Triton attention derives scale views from the cache.
            if hasattr(impl, "_k_scale_cache"):
                impl._k_scale_cache = None
            if hasattr(impl, "_v_scale_cache"):
                impl._v_scale_cache = None

    def finalize_kv_cache_geometry(self, vllm_config: VllmConfig) -> None:
        """Apply the resolved cache geometry to a loaded backend.

        Hybrid cache alignment runs after model construction because it needs
        the selected attention backend. Implementations that derive kernel
        plans or persistent workspace sizes from the page geometry receive the
        resolved value here, before device-memory profiling assigns the
        remaining capacity to KV cache.

        Args:
            vllm_config: Configuration containing the resolved cache geometry.
        """
        impl = getattr(self, "impl", None)
        finalize = getattr(impl, "finalize_kv_cache_geometry", None)
        if callable(finalize):
            finalize(vllm_config.cache_config.block_size)

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
