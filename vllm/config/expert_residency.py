# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for b12x expert residency on SM103.

Expert residency serves native MXFP4 routed experts from two tiers: frequently
routed experts stay in HBM and the rest stay in coherent Grace memory, which
the GPU reads directly. "Expert residency" is a working name.
"""

from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig
    from vllm.config.load import LoadConfig
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

_GIB = 1 << 30
# Loaders that write expert tensors through parameter weight callbacks, so the
# routed experts can be staged in host memory instead of HBM.
_HOST_STAGED_LOAD_FORMATS = ("auto", "safetensors")


@config
class ExpertResidencyConfig:
    """Serve routed MXFP4 experts from HBM and coherent Grace memory.

    Static placement only: a validated b12x placement profile, or a placement
    balanced across layers from the memory budget. Routing calibration and
    live placement changes are not enabled by this configuration.
    """

    profile_path: str | None = None
    """b12x placement profile to serve. It must match this checkpoint, layer
    geometry and memory budget, or startup fails. Without a profile, each
    layer keeps a budget-balanced set of experts in HBM."""

    workload: str = "default"
    """Workload name used to look up stored b12x placement profiles."""

    profile_directory: str | None = None
    """Directory of stored b12x placement profiles. Defaults to the b12x
    profile store location."""

    hbm_reserve_gb: float = Field(default=16.0, ge=0)
    """HBM kept free for activations, CUDA graphs and other prepared
    operators, beyond the KV cache. Expert rows never use it."""

    grace_gb: float | None = Field(default=None, gt=0)
    """Upper bound on Grace memory for cold experts. Defaults to the host
    memory available when experts are placed."""

    grace_reserve_gb: float = Field(default=16.0, ge=0)
    """Host memory kept free after placing cold experts."""

    min_hot_experts: int | None = Field(default=None, ge=0)
    """Minimum experts per layer kept in HBM."""

    max_hot_experts: int | None = Field(default=None, ge=0)
    """Maximum experts per layer kept in HBM."""

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if (
            self.min_hot_experts is not None
            and self.max_hot_experts is not None
            and self.min_hot_experts > self.max_hot_experts
        ):
            raise ValueError("min_hot_experts exceeds max_hot_experts")
        return self

    @property
    def hbm_reserve_bytes(self) -> int:
        return int(self.hbm_reserve_gb * _GIB)

    @property
    def grace_reserve_bytes(self) -> int:
        return int(self.grace_reserve_gb * _GIB)

    @property
    def grace_bytes(self) -> int | None:
        return None if self.grace_gb is None else int(self.grace_gb * _GIB)

    def verify(
        self,
        *,
        model_config: "ModelConfig | None",
        parallel_config: "ParallelConfig",
        cache_config: "CacheConfig",
        load_config: "LoadConfig",
        moe_backend: str | None,
    ) -> None:
        """Reject deployments this static integration does not implement."""
        import vllm.envs as envs
        from vllm.platforms import current_platform

        if not current_platform.is_cuda() or not current_platform.is_device_capability(
            (10, 3)
        ):
            raise ValueError("expert residency requires an SM103 GPU")
        if not envs.VLLM_B12X_SM103:
            raise ValueError("expert residency requires VLLM_B12X_SM103=1")
        if model_config is None:
            raise ValueError("expert residency requires a model configuration")
        if moe_backend != "b12x":
            raise ValueError("expert residency requires --moe-backend b12x")
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.data_parallel_size != 1
            or parallel_config.enable_expert_parallel
        ):
            raise ValueError(
                "expert residency currently supports one GPU without tensor, "
                "pipeline, data or expert parallelism"
            )
        if cache_config.kv_cache_memory_bytes is None:
            raise ValueError(
                "expert residency requires --kv-cache-memory-bytes so the KV cache "
                "is reserved before experts are placed in HBM"
            )
        if load_config.load_format not in _HOST_STAGED_LOAD_FORMATS:
            raise ValueError(
                "expert residency stages routed experts in host memory and requires "
                f"load_format {' or '.join(map(repr, _HOST_STAGED_LOAD_FORMATS))}, "
                f"got {load_config.load_format!r}"
            )

    def compute_hash(self) -> str:
        """Placement changes which expert storage each prepared plan reads."""
        return hash_factors(get_hash_factors(self, set()))
