# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
B12x MoE expert backend for NVFP4 on SM120 (Blackwell) GPUs.

This backend replaces FlashInfer CUTLASS for fused MoE FP4 operations,
providing significantly higher throughput on Blackwell-class hardware.
It reuses the same weight preparation path as FLASHINFER_CUTLASS
(swizzled block-scales, [w3,w1] ordering for gated activations) and
delegates the actual computation to ``b12x.integration.tp_moe.b12x_moe_fp4``.
"""

import os
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.b12x_contract import (
    b12x_sparse_mla_active_for_config,
)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)
_B12X_MOE_A16_MIN_LAYER = int(
    os.getenv("VLLM_B12X_MOE_A16_MIN_LAYER", "-1")
)
_B12X_MOE_A16_MAX_LAYER = int(
    os.getenv("VLLM_B12X_MOE_A16_MAX_LAYER", "-1")
)

# ---------------------------------------------------------------------------
# Per-device workspace pool (mirrors SGLang's pattern).
# The pool is stream-aware inside b12x, so one pool per device suffices.
# ---------------------------------------------------------------------------
_B12X_MOE_WORKSPACE_POOLS: dict[int, Any] = {}


# Resolve the new shared-arena workspace API ONCE at module load. Repeating the
# `from b12x.integration import get_b12x_moe_workspace_pool` inside the
# per-layer call introduces a per-MoE-call import-lookup roundtrip on every
# decode/prefill step; even when the import resolves to ImportError the
# failure path interacts badly with the ProcessGroupNCCL all-rank coordination
# vLLM does immediately around `_get_b12x_workspace_pool` and inflates rank-0
# resident memory by ~33 GiB during `profile_run` (see task #41 bisect:
# `handoff_work/regression_23_rootcause.md`).
try:
    from b12x.integration import (
        get_b12x_moe_workspace_pool as _b12x_shared_arena_pool,
    )
except ImportError:
    _b12x_shared_arena_pool = None


def _active_vllm_config_for_b12x_moe():
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None and is_forward_context_available():
        vllm_config = get_forward_context().vllm_config
    return vllm_config


def _requires_b12x_joint_moe_pool() -> bool:
    vllm_config = _active_vllm_config_for_b12x_moe()
    return b12x_sparse_mla_active_for_config(vllm_config)


def _run_b12x_moe_fp4(
    *,
    a: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    **kwargs,
) -> None:
    """Call b12x MoE with the live row count.

    b12x owns capacity-based launch and workspace selection; vLLM should not
    pad, split, or otherwise bucket rows before dispatch.
    """
    from b12x.integration.tp_moe import b12x_moe_fp4

    b12x_moe_fp4(
        a=a,
        output=output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        **kwargs,
    )


def _get_b12x_workspace_pool(device: torch.device):
    """Return the b12x MoE pool for the active execution lane.

    When sparse MLA is active, attention and MoE must share one execution-lane
    arena. Otherwise this may use the standalone TP-MoE pool.
    """
    requires_joint = _requires_b12x_joint_moe_pool()
    if _b12x_shared_arena_pool is not None:
        try:
            if requires_joint:
                from b12x.integration.arena import get_b12x_execution_lane

                lane = get_b12x_execution_lane(
                    device,
                    create_standalone_moe_pool=False,
                )
                if lane is None or getattr(lane, "arena", None) is None:
                    raise RuntimeError(
                        "shared execution-lane arena is not installed"
                    )
            return _b12x_shared_arena_pool(device)
        except (AttributeError, RuntimeError, TypeError) as exc:
            if requires_joint:
                raise RuntimeError(
                    "B12X MoE cannot fall back to a standalone workspace pool "
                    "while B12X sparse MLA attention is active"
                ) from exc
            logger.warning_once(
                "Falling back to standalone b12x MoE workspace pool: %s", exc
            )
    elif requires_joint:
        raise RuntimeError(
            "B12X sparse MLA attention with B12X MoE requires a b12x build "
            "with shared execution-lane arena support"
        )

    from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool

    device_idx = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    pool = _B12X_MOE_WORKSPACE_POOLS.get(device_idx)
    if pool is None:
        pool = allocate_tp_moe_workspace_pool()
        _B12X_MOE_WORKSPACE_POOLS[device_idx] = pool
    return pool


def _has_b12x() -> bool:
    """Return True when the b12x MoE kernel is importable."""
    try:
        from b12x.integration.tp_moe import b12x_moe_fp4  # noqa: F401

        return True
    except ImportError:
        return False


def _b12x_env_force_moe_a16() -> bool:
    return envs.VLLM_B12X_FORCE_MOE_A16


def _b12x_quant_mode_uses_w4a16(quant_mode: str | None) -> bool:
    if quant_mode is None:
        return _b12x_env_force_moe_a16()
    return quant_mode == "w4a16"


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        return "silu"
    raise ValueError(f"B12xExperts does not support {activation} activation")


def _iter_attn_metadata(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        yield from attn_metadata.values()
        return
    if isinstance(attn_metadata, list):
        for item in attn_metadata:
            yield from _iter_attn_metadata(item)
        return
    if attn_metadata is not None:
        yield attn_metadata


def _current_forward_is_decode_only() -> bool:
    if not is_forward_context_available():
        return False

    forward_context = get_forward_context()
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    if (
        getattr(
            forward_context,
            "cudagraph_runtime_mode",
            CUDAGraphMode.NONE,
        )
        != CUDAGraphMode.NONE
        and batch_descriptor is not None
        and batch_descriptor.uniform
    ):
        return True

    saw_decode = False
    for metadata in _iter_attn_metadata(forward_context.attn_metadata):
        num_prefills = getattr(metadata, "num_prefills", None)
        num_decodes = getattr(metadata, "num_decodes", None)
        if num_prefills is not None or num_decodes is not None:
            if int(num_prefills or 0) > 0:
                return False
            saw_decode = saw_decode or int(num_decodes or 0) > 0
            continue

        max_query_len = getattr(metadata, "max_query_len", None)
        if max_query_len is None:
            continue
        if int(max_query_len) > 1:
            return False
        saw_decode = True

    return saw_decode


# ---------------------------------------------------------------------------
# Expert implementation
# ---------------------------------------------------------------------------


class B12xExperts(mk.FusedMoEExpertsModular):
    """
    NVFP4 fused-MoE expert backend powered by b12x kernels.

    Weight contract
    ~~~~~~~~~~~~~~~
    This class relies on the *same* weight preparation that
    ``FLASHINFER_CUTLASS`` uses (handled by
    ``prepare_nvfp4_moe_layer_for_fi_or_cutlass`` in the quantisation
    layer):

    * Weights are packed uint8 (two FP4 values per byte).
    * Block-scales are swizzled via ``swizzle_blockscale``.
    * For gated activations the W13 tensor is reordered to ``[w3, w1]``.

    The ``FusedMoEQuantConfig`` supplies the following tensors consumed
    here (via the property helpers inherited from ``FusedMoEExperts``):

    ==================  ==========================================
    quant_config attr   b12x parameter
    ==================  ==========================================
    ``a1_gscale``       ``a1_gscale``   (reciprocal input scale)
    ``g1_alphas``       ``w1_alphas``   (alpha = input_scale * weight_scale_2)
    ``w1_scale``        ``w1_blockscale`` (swizzled, viewed as int32 for FI; raw fp8 for b12x)
    ``a2_gscale``       ``a2_gscale``
    ``g2_alphas``       ``w2_alphas``
    ``w2_scale``        ``w2_blockscale``
    ==================  ==========================================
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype == "nvfp4", (
            f"B12xExperts only supports nvfp4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.out_dtype = moe_config.in_dtype
        self.layer_name = moe_config.layer_name
        self.layer_idx = moe_config.layer_idx
        self._warned_global_a16 = False
        self._prepared_w4a16: Any | None = None

    # ------------------------------------------------------------------
    # process_weights_after_loading: fuse input scales into g{1,2}_alphas
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Fuse per-expert activation scales into the alpha tensors.

        This matches the FlashInferExperts behaviour: after this call
        ``g1_alphas == w13_weight_scale_2 * w13_input_scale`` and
        ``g2_alphas == w2_weight_scale_2 * w2_input_scale``.
        The quant_config tensors are updated in-place so that the EPLB
        rearrangement pathway stays in sync.
        """
        if self.quant_config.use_nvfp4_w4a4:
            w13_input_scale = layer.w13_input_scale
            if layer.w13_weight_scale_2.ndim == 2 and w13_input_scale.ndim == 1:
                w13_input_scale = w13_input_scale[:, None]
            layer.w13_weight_scale_2.data.mul_(w13_input_scale)
            layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

        self._prepared_w4a16 = None
        if not self._should_prepare_w4a16_weights():
            return

        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for B12xExperts"
        )
        assert self.a1_gscale is not None and self.a2_gscale is not None, (
            "a1_gscale and a2_gscale must not be None for B12xExperts"
        )

        from b12x.integration.tp_moe import prepare_b12x_w4a16_packed_weights

        self._prepared_w4a16 = prepare_b12x_w4a16_packed_weights(
            layer.w13_weight,
            self.w1_scale,
            self.g1_alphas,
            self.a1_gscale,
            layer.w2_weight,
            self.w2_scale,
            self.g2_alphas,
            self.a2_gscale,
            activation=_b12x_activation_name(layer.activation),
            params_dtype=self.out_dtype,
            quant_mode="w4a16",
            source_format="modelopt",
            reuse_input_storage=True,
        )

    # ------------------------------------------------------------------
    # Static capabilities
    # ------------------------------------------------------------------
    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and _has_b12x()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # b12x supports SiLU-gated (SwiGLU) activations.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size <= 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # b12x performs its own FP4 quantisation internally.
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x fuses topk-weight application and expert reduction internally.
        return TopKWeightAndReduceNoOP()

    def _select_quant_mode(self) -> str | None:
        """Return the explicit b12x quant_mode override for this layer.

        ``None`` preserves b12x's default behavior, normally the NVFP4 A4 path.
        The normal vLLM W4A16 path prepares weights at load time and ``apply``
        then passes ``quant_mode="w4a16"`` with those prepared weights. Passing
        ``"nvfp4"`` is used only for layer-gated or prefill/mixed-forwards
        cases that intentionally stay on NVFP4.

        W4A16 bypasses NVFP4 activation quant/dequant. The optional layer gate
        lets us keep selected layers on NVFP4 while isolating A16 numerics to a
        specific layer range during GLM-5.1 quality/debug runs.
        """
        if envs.VLLM_B12X_MOE_DECODE_A16:
            if _current_forward_is_decode_only():
                return "w4a16"
            return "nvfp4"

        if _B12X_MOE_A16_MIN_LAYER >= 0 and _b12x_env_force_moe_a16():
            if self.layer_idx is None:
                logger.warning_once(
                    "VLLM_B12X_MOE_A16_MIN_LAYER is set, but layer index "
                    "is unavailable for %s; keeping b12x default MoE mode.",
                    self.layer_name,
                )
                return None
            if self.layer_idx < _B12X_MOE_A16_MIN_LAYER:
                return "nvfp4"
        if _B12X_MOE_A16_MAX_LAYER >= 0 and _b12x_env_force_moe_a16():
            if self.layer_idx is None:
                logger.warning_once(
                    "VLLM_B12X_MOE_A16_MAX_LAYER is set, but layer index "
                    "is unavailable for %s; keeping b12x default MoE mode.",
                    self.layer_name,
                )
                return None
            if self.layer_idx > _B12X_MOE_A16_MAX_LAYER:
                return "nvfp4"

        if _b12x_env_force_moe_a16() and not self._warned_global_a16:
            self._warned_global_a16 = True
            logger.warning_once(
                "VLLM_B12X_FORCE_MOE_A16=1 changes B12X MoE activation "
                "numerics. "
                "For GLM-5.1 NVFP4 quality checks, leave it unset or gate "
                "A16 with VLLM_B12X_MOE_A16_MIN_LAYER."
            )

        return None

    def _should_prepare_w4a16_weights(self) -> bool:
        if envs.VLLM_B12X_MOE_DECODE_A16:
            return False
        if not _b12x_env_force_moe_a16():
            return False
        if self.layer_idx is None:
            return True
        if (
            _B12X_MOE_A16_MIN_LAYER >= 0
            and self.layer_idx < _B12X_MOE_A16_MIN_LAYER
        ):
            return False
        if (
            _B12X_MOE_A16_MAX_LAYER >= 0
            and self.layer_idx > _B12X_MOE_A16_MAX_LAYER
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # workspace_shapes
    # ------------------------------------------------------------------
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x manages its own internal workspace via the pool.
        # We only need to declare the output shape for the framework.
        workspace1 = (M, K)
        workspace2 = (0,)
        # nvfp4 output is packed int8 -> hidden dim is 2 * K
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    # ------------------------------------------------------------------
    # apply -- the hot path
    # ------------------------------------------------------------------
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for B12xExperts"
        )
        assert self.a1_gscale is not None and self.a2_gscale is not None, (
            "a1_gscale and a2_gscale must not be None for B12xExperts"
        )

        workspace_pool = _get_b12x_workspace_pool(hidden_states.device)
        quant_mode = self._select_quant_mode()
        prepared_w4a16 = None
        if self._prepared_w4a16 is not None:
            if not _b12x_quant_mode_uses_w4a16(quant_mode):
                raise RuntimeError(
                    "B12x W4A16 weights were prepared in-place, but the "
                    "current b12x MoE quant mode is not W4A16."
                )
            quant_mode = "w4a16"
            prepared_w4a16 = self._prepared_w4a16

        _run_b12x_moe_fp4(
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=(
                apply_router_weight_on_input
                if apply_router_weight_on_input is not None
                else False
            ),
            workspace=workspace_pool,
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            quant_mode=quant_mode,
            prepared_w4a16=prepared_w4a16,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
