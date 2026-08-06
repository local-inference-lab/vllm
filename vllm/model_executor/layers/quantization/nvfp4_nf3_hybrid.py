# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed NVFP4/MXFP4 + NF3 (3-bit) MoE quantization ("nvfp4_nf3_hybrid").

Serves checkpoints whose routed experts are per-layer mixed precision: a
high-saliency "kept" tier stored as NVFP4 (e2m1 values + e4m3 group scales)
or MXFP4 (e2m1 + ue8m0 scales per 32 group, checkpoint key
``kept_format = "mxfp4_e8m0k32"``), and a low-saliency tier stored as NF3
(3-bit codebook packed 8 codes per 3 bytes, e4m3 scales per 32 group).

The tier assignment is carried by the ``hybrid_bit_map`` key of the
checkpoint quantization config: a dict mapping decoder-layer index (as a
string) to a per-expert list of bit widths (4 = kept, 3 = NF3). MoE layers
absent from the map (e.g. an MTP head) are treated as uniform NVFP4 and run
through the same path as an all-kept layer. Non-expert linear layers are
excluded by the checkpoint config and handled by the regular machinery.

The retained tier executes through its ordinary MXFP4 W4A16 kernel.  A TP12
E4M3 trellis tier (MUL1 or SQG) can use a direct W4A8 route-major decode for
the small decode buckets where it wins, and retains the full-rotation W4A16
path for prefill and larger batches.  Both paths consume the same compact
P24/P33 payload and global-to-local expert map.  All scratch is allocated and
all kernels are prewarmed during vLLM's eager profile run, so both choices
remain CUDA-graph safe.
"""

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch

from vllm import envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts

logger = init_logger(__name__)

_qsrt_repeat_check_reports = 0

# Pinned CTA tiles (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n): the NF3
# flat-span weight layout is packed for a SPECIFIC tile_n, but the kernel's
# auto tile selection is m-dependent (fc1_tile_n flips 128<->256 across m).
# (64, 256, 64, 256) is what auto-selection picks for the max-m prefill, and
# its shared-memory/register footprint fits both moe_block_size 8 (decode)
# and 64 (prefill) at both scale formats.
_B12X_TILES = (64, 256, 64, 256)
# Batches of at most this many tokens take the preplanned TC-decode launch.
_B12X_DECODE_M = 8
# Production-shape microbenchmarks favor the native E4M3 W4A8 path for M=1..4.
# At M=8 the current route-major FC2 loses its decode advantage, so retain the
# mature W4A16 path there until same-expert route grouping lands.
_TRELLIS_W4A8_DECODE_M = 4
# Global scale the NF3 prepare path expects (scales are stored pre-divided).
_NF3_GLOBAL_SCALE = 2.0**116
# Expert-chunk size for NF3 unpack/repack. Kimi K3 TP16 runs with less than
# 4 GiB of headroom after loading, so keep the int32 decode temporary small;
# final packed planes are written directly into their resident buffers below.
_NF3_PACK_CHUNK = 4
# Exact one-grid decode specialization published for the TP4 hybrid checkpoint.
_GRID188_M = 4
_GRID188_TOPK = 8
_GRID188_HIDDEN = 6144
_GRID188_INTERMEDIATE = 512
_GRID188_NUM_KEPT = 64
_GRID188_NUM_NF3 = 192
# Kimi K3 TP16 decode geometry.  Its tier sizes vary by layer, so only the
# global geometry is pinned here; each layer compiles against its exact split.
_K3_HYBRID_M = 1
_K3_HYBRID_TOPK = 16
_K3_HYBRID_HIDDEN = 3584
_K3_HYBRID_INTERMEDIATE = 192
_K3_HYBRID_EXPERTS = 896
# Fixed X4T/W4A16 TP12 ABI.  QSRT fuses checkpoint-order [w1; w3] rows,
# which are [gate; up] for SiTU; B12X names that physical order ``w31``.
_QSRT_X4T_W13_LAYOUT = "w31"
_QSRT_X4T_W13_EXCEPTION_TASK_ROWS = 128
_QSRT_X4T_W2_EXCEPTION_TASK_ROWS = 896
_QSRT_X4T_W13_EXCEPTION_ROW_ROTATION = 0


def _qsrt_w4a8_requested() -> bool:
    """Return the opt-in runtime toggle for native QSRT W4A8.

    The direct route-major port is numerically closed but remains behind the
    mature W4A16 fused path in CUDA-graph decode benchmarks.  Keep it off by
    default until the fused-scheduler implementation passes the performance
    gate; this switch exists for explicit kernel evaluation only.
    """

    return os.getenv("VLLM_KQUANT_TRELLIS_W4A8", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _stack_exl3_intermediate_rotations(
    w13_svh: torch.Tensor,
    w2_suh: torch.Tensor,
) -> torch.Tensor:
    """Build B12X's ``[gate_svh, up_svh, down_suh]`` rotation bundle."""

    if w13_svh.ndim != 3 or int(w13_svh.shape[1]) != 2:
        raise ValueError("EXL3 w13_svh must have shape [experts, 2, intermediate]")
    if (
        w2_suh.ndim != 2
        or int(w2_suh.shape[0]) != int(w13_svh.shape[0])
        or int(w2_suh.shape[1]) != int(w13_svh.shape[2])
    ):
        raise ValueError("EXL3 w2_suh must have shape [experts, intermediate]")
    return torch.cat(
        [w13_svh[:, 0], w13_svh[:, 1], w2_suh],
        dim=1,
    ).contiguous()


def _require_rank_local_kept_kernel(kernel: Any) -> None:
    """Reject a kept-tier kernel that would bypass the outer TP reduction.

    Mixed K3 weights are already sharded over the intermediate axis. The
    compact MXFP4 tier is therefore built with a no-parallel MoE config and
    must return the same kind of rank-local latent partial as the trellis
    tier. The outer FusedMoE runner owns the reduction after the optional
    Kimi routed-output transform.
    """

    if kernel.output_is_reduced():
        raise RuntimeError(
            "nvfp4_nf3_hybrid kept kernel must return an unreduced rank-local partial"
        )


def _is_dense_layer_ignored(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: dict[str, list[str]],
) -> bool:
    """Resolve dense-format exclusions from full paths or module names.

    kquant artifacts use leaf/component names such as ``g_proj`` and
    ``vision_tower`` because the same exclusion applies throughout the model.
    ``is_layer_skipped`` otherwise treats entries as exact full prefixes, which
    silently quantizes those BF16-only modules and leaves their nonexistent
    MXFP8 scales uninitialized.

    Expand component-only entries to the concrete prefix (and to each logical
    child of a fused linear) before delegating to the standard matcher. This
    preserves its validation that all shards of a fused linear use one format,
    while avoiding substring matches such as ``b_proj`` matching ``q_b_proj``.
    """
    expanded = list(ignored_layers)
    candidates = [prefix]
    base, separator, projection = prefix.rpartition(".")
    if projection in fused_mapping:
        candidates.extend(
            f"{base}{separator}{shard}" for shard in fused_mapping[projection]
        )

    for ignored in ignored_layers:
        if not ignored or "." in ignored:
            continue
        expanded.extend(
            candidate for candidate in candidates if ignored in candidate.split(".")
        )

    return is_layer_skipped(
        prefix=prefix,
        ignored_layers=expanded,
        fused_mapping=fused_mapping,
    )


def _combined_tier_local_descriptors(
    remap: dict[int, tuple[int, int]],
    *,
    num_experts: int = _GRID188_NUM_KEPT + _GRID188_NUM_NF3,
    num_kept: int = _GRID188_NUM_KEPT,
    num_nf3: int = _GRID188_NUM_NF3,
) -> list[int]:
    """Encode an exact two-tier partition for the hybrid one-grid kernel."""
    descriptors = [-1] * num_experts
    seen_local: tuple[set[int], set[int]] = (set(), set())
    for global_id, tier_local in remap.items():
        try:
            global_id_i = int(global_id)
            tier, local_id = tier_local
            tier_i, local_id_i = int(tier), int(local_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid heterogeneous expert remap entry") from exc
        if global_id_i != global_id or not 0 <= global_id_i < len(descriptors):
            raise ValueError(f"invalid global expert ID {global_id!r}")
        if descriptors[global_id_i] != -1:
            raise ValueError(f"duplicate global expert ID {global_id_i}")
        local_limit = num_kept if tier_i == 0 else num_nf3 if tier_i == 1 else 0
        if (
            tier_i != tier
            or local_id_i != local_id
            or local_limit == 0
            or not 0 <= local_id_i < local_limit
        ):
            raise ValueError(f"invalid tier/local expert descriptor {tier_local!r}")
        if local_id_i in seen_local[tier_i]:
            raise ValueError(
                f"duplicate tier/local expert descriptor {(tier_i, local_id_i)!r}"
            )
        seen_local[tier_i].add(local_id_i)
        descriptors[global_id_i] = local_id_i if tier_i == 0 else 0x10000 | local_id_i
    if any(descriptor < 0 for descriptor in descriptors):
        raise ValueError(
            f"heterogeneous remap does not cover all {num_experts} global experts"
        )
    if seen_local[0] != set(range(num_kept)) or seen_local[1] != set(range(num_nf3)):
        raise ValueError("heterogeneous remap is not a complete two-tier partition")
    return descriptors


def _is_grid188_geometry(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_kept: int,
    num_nf3: int,
    topk: int,
    kept_mx: bool,
) -> bool:
    return (
        envs.VLLM_NF3_GRID188_DECODE
        and not kept_mx
        and hidden_size == _GRID188_HIDDEN
        and intermediate_size == _GRID188_INTERMEDIATE
        and num_experts == _GRID188_NUM_KEPT + _GRID188_NUM_NF3
        and num_kept == _GRID188_NUM_KEPT
        and num_nf3 == _GRID188_NUM_NF3
        and topk == _GRID188_TOPK
    )


def _is_k3_hybrid_geometry(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_kept: int,
    num_nf3: int,
    topk: int,
    kept_mx: bool,
) -> bool:
    return (
        envs.VLLM_NF3_GRID188_DECODE
        and bool(int(os.getenv("VLLM_K3_HYBRID_DECODE", "1")))
        and kept_mx
        and hidden_size == _K3_HYBRID_HIDDEN
        and intermediate_size == _K3_HYBRID_INTERMEDIATE
        and num_experts == _K3_HYBRID_EXPERTS
        and num_kept > 0
        and num_nf3 > 0
        and num_kept + num_nf3 == num_experts
        and topk == _K3_HYBRID_TOPK
    )


def _read_hybrid_keys(config: Any) -> tuple[dict[str, list[int]] | None, str | None]:
    """Read ``hybrid_bit_map``/``kept_format`` from a quantization config dict.

    Both config layouts are supported: keys at the top level (config.json
    ``quantization_config``) or nested under ``"quantization"``
    (hf_quant_config.json).
    """
    if not isinstance(config, dict):
        return None, None
    hybrid_bit_map = config.get("hybrid_bit_map")
    kept_format = config.get("kept_format")
    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        hybrid_bit_map = hybrid_bit_map or quantization.get("hybrid_bit_map")
        kept_format = kept_format or quantization.get("kept_format")
    return hybrid_bit_map, kept_format


def _apply_nf3_codebook_override(levels: list[float]) -> None:
    """Install a checkpoint-specific NF3 codebook process-wide.

    The b12x execution path crosses the ``torch.ops.sparkinfer`` custom-op
    boundary with primitive args only; the op re-resolves its kernel from
    module globals, so a per-kernel codebook argument cannot reach the
    silicon. One process serves one model, so a global is correct here. The
    env stamp folds the codebook into sparkinfer's compile disk-cache key
    (SPARKINFER_* env vars are part of its cache context), preventing stale
    cubins when a different-codebook model is served later.
    """
    import os

    from sparkinfer.moe._shared.kernels.w4a16 import kernel as _sk
    from sparkinfer.moe._shared.kernels.w4a16 import prepare as _sp

    t = tuple(float(v) for v in levels)
    if tuple(_sk._NF3_CODEBOOK) != t:
        logger.info_once("Overriding b12x NF3 codebook with checkpoint levels: %s", t)
        _sk._NF3_CODEBOOK = t
        _sp._NF3_CODEBOOK = t
    os.environ["SPARKINFER_NF3_CODEBOOK"] = ",".join(f"{v:.10g}" for v in t)


def _unpack_nf3_codes(packed: torch.Tensor, size_k: int) -> torch.Tensor:
    """Unpack NF3 codes stored 8-per-3-bytes: uint8 [E, N, K//8*3] -> int32
    [E, N, K] codes in 0..7."""
    num_experts, rows, _ = packed.shape
    triplets = packed.reshape(num_experts, rows, size_k // 8, 3).to(torch.int32)
    word = triplets[..., 0] | (triplets[..., 1] << 8) | (triplets[..., 2] << 16)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 3
    codes = (word.unsqueeze(-1) >> shifts) & 7
    return codes.reshape(num_experts, rows, size_k)


def _b12x_tiles_for_geometry(
    hidden_size: int, intermediate_size: int
) -> tuple[int, int, int, int]:
    """Select one fixed b12x tile pair that exactly divides both GEMMs.

    GLM's TP-local expert width uses N=256 tiles. Kimi K3 at TP16 has a
    local expert width of 192, so FC1 has N=384 and needs an N=128 tile.
    Both GEMMs use the measured 128-thread K64/N128 SM121 specialization.
    """
    # TP16 K3 M=1 tuning on SM121 favors a narrower FC1 N tile with twice the
    # K depth. Across the checkpoint's real mixed-tier splits this cuts the
    # one-grid kernel by roughly 1--3% in eager and 24--28% in graph replay.
    # Keep the specialization local to K3 so existing GLM/Grid188 packing is
    # byte-for-byte unchanged.
    if (
        hidden_size == _K3_HYBRID_HIDDEN
        and intermediate_size == _K3_HYBRID_INTERMEDIATE
    ):
        return (128, 64, 64, 128)
    candidates = (_B12X_TILES, (64, 128, 64, 128))
    for fc1_k, fc1_n, fc2_k, fc2_n in candidates:
        if (
            hidden_size % fc1_k == 0
            and (2 * intermediate_size) % fc1_n == 0
            and intermediate_size % fc2_k == 0
            and hidden_size % fc2_n == 0
        ):
            return (fc1_k, fc1_n, fc2_k, fc2_n)
    raise ValueError(
        "nvfp4_nf3_hybrid has no fixed b12x tile configuration for "
        f"hidden={hidden_size}, intermediate={intermediate_size}"
    )


def _decode_kquant_nf3_scale(raw: torch.Tensor) -> torch.Tensor:
    """Interpret kquant's uint8 payload as biased E4M3 scale values.

    kquant stores the raw E4M3 bits of ``scale * 16``. Keeping the biased
    FP8 value resident avoids expanding the checkpoint scales to FP32; the
    bias is removed immediately before SparkInfer's scale packer runs.
    """
    if raw.dtype == torch.uint8:
        return raw.contiguous().view(torch.float8_e4m3fn)
    if raw.dtype == torch.float8_e4m3fn:
        return raw
    raise TypeError(f"NF3 scale must be uint8/E4M3, got {raw.dtype}")


class _HybridSharedRuntime:
    """Process-wide b12x W4A16 runtime shared by every hybrid MoE layer.

    One preplanned-launch cache and one scratch/route buffer set serve all
    layers: launches on a single stream never overlap and every
    ``run_w4a16_moe`` call fully overwrites the buffers it uses.
    """

    def __init__(self) -> None:
        self.max_m: int | None = None
        self.topk: int | None = None
        # (num_experts, weight_layout, scale_format, topk, max_m, H, I)
        #   -> (decode_launch, prefill_launch)
        self.launches: dict[tuple, Any] = {}
        self.buffers: Any = None
        self.out_kept: torch.Tensor | None = None
        self.out_nf3: torch.Tensor | None = None
        self.grid188_launch: Any = None
        self.grid188_scratch: dict[str, torch.Tensor] | None = None
        self.grid188_sms: int | None = None
        self.grid188_max_shared_mem: int | None = None
        self.grid188_disabled_reason: str | None = None
        self.k3_hybrid_scratch: dict[str, torch.Tensor] | None = None
        self.k3_hybrid_sms: int | None = None
        self.k3_hybrid_max_shared_mem: int | None = None
        self.k3_hybrid_disabled_reason: str | None = None
        # Capture-only route-major canonical pre-w2 scratch. EXL3 cache2 is
        # H128(h * down_suh); this stable buffer receives its inverse transform.
        self.kquant_logical_mid: torch.Tensor | None = None
        # Native E4M3 W4A8 decode buffers, one allocation per captured M.
        # They are shared across layers because execution is single-stream.
        self.trellis_w4a8_scratch: dict[int, Any] = {}
        self.trellis_scratch: torch.Tensor | None = None
        # X4T expands only routed scale rows immediately before W4A16. The
        # output grids are shared across layers on the same CUDA stream.
        self.x4t_w13_scale_scratch: torch.Tensor | None = None
        self.x4t_w2_scale_scratch: torch.Tensor | None = None


class _HybridLayerState:
    """Per-layer tier bookkeeping, filled in across ``create_weights`` ->
    ``process_weights_after_loading`` -> first ``apply``."""

    def __init__(
        self,
        remap: dict[int, tuple[int, int]],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        kept_mx: bool,
    ) -> None:
        # global expert id -> (tier, local index); tier 0 = kept, 1 = NF3.
        self.remap = remap
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.kept_mx = kept_mx
        self.num_kept = sum(1 for tier, _ in remap.values() if tier == 0)
        self.num_nf3 = sum(1 for tier, _ in remap.values() if tier == 1)
        self.tiles = _b12x_tiles_for_geometry(hidden_size, intermediate_size)
        # b12x prepared weights (W4A16PackedWeights / PreparedNF3MoeWeights).
        self.prep_kept: Any = None
        # Native MXFP4 W4A16 representation already owned by kept_kernel.
        # This is a view/metadata bundle, not a second resident weight copy.
        self.prep_kept_hybrid: Any = None
        self.prep_nf3: Any = None
        # Global -> local id maps, -1 for experts outside the tier.
        self.emap_kept: torch.Tensor | None = None
        self.emap_nf3: torch.Tensor | None = None
        # (decode_launch, prefill_launch) per tier, set at first apply.
        self.launch_kept: tuple[Any, Any] | None = None
        self.launch_nf3: tuple[Any, Any] | None = None
        # MXFP4 kept tier: modular kernel + its weight-holder module and a
        # global -> local map; -1 is the inactive-route sentinel.
        self.kept_kernel: Any = None
        self.kept_module: torch.nn.Module | None = None
        self.kept_remap: torch.Tensor | None = None
        # Exact TP4 E64-NVFP4/E192-NF3 one-grid decode resources.
        self.grid188_weight_views: tuple[torch.Tensor, ...] | None = None
        self.grid188_tier_map: torch.Tensor | None = None
        self.grid188_output: torch.Tensor | None = None
        self.grid188_ready = False
        # K3 TP16 one-grid MXFP4/NF3 decode resources.
        self.k3_hybrid_launch: Any = None
        self.k3_hybrid_weight_views: tuple[torch.Tensor, ...] | None = None
        self.k3_hybrid_tier_map: torch.Tensor | None = None
        self.k3_hybrid_output: torch.Tensor | None = None
        self.k3_hybrid_ready = False
        # Keeps kernel-format tensors alive: b12x prepared weights VIEW the
        # converted tensors, so dropping them would dangle the views.
        self.keepalive: Any = None
        # This mapped K3 layer owns its experts through one fixed TP12 slab,
        # rather than ordinary safetensors expert parameters.
        self.uses_mixed_tp12_slab = False
        self.trellis_w4a8_ready = False
        self.trellis_w4a8_prewarmed = False
        self.trellis_weights: Any = None
        self.trellis_plan: Any = None
        self.runtime_ready = False


class NvFp4Nf3HybridConfig(ModelOptNvFp4Config):
    """Config for mixed NVFP4/MXFP4 + NF3 checkpoints.

    Extends :class:`ModelOptNvFp4Config` with the two hybrid checkpoint
    keys: ``hybrid_bit_map`` (required; per-layer, per-expert bit widths)
    and ``kept_format`` (optional; ``"mxfp4_e8m0k32"`` switches the kept
    tier from NVFP4 to MXFP4).
    """

    def __init__(
        self,
        quant_method: str = "NVFP4",
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str | None = None,
        exclude_modules: list[str] | None = None,
        group_size: int = 16,
        hybrid_bit_map: dict[str, list[int]] | None = None,
        kept_format: str | None = None,
    ) -> None:
        super().__init__(
            quant_method,
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo,
            exclude_modules,
            group_size,
        )
        self.hybrid_bit_map: dict[str, list[int]] = hybrid_bit_map or {}
        self.kept_format = kept_format
        self.kept_storage: str = "inline-mxfp4"
        self.nf3_levels: list[float] | None = None
        # "nf3_2p1" (default) or "exl3_3": how demoted (bit-3) experts are
        # stored and executed. exl3_3 = native EXL3 trellis tensors run via
        # sparkinfer trellis_moe.
        self.demoted_format: str = "nf3_2p1"
        self.mixed_exl3_tp12: dict[str, Any] | None = None
        self.qsrt_tp12: dict[str, Any] | None = None
        self.trellis_codebook: str = "mcg"
        self.trellis_mcg: int = 0
        self.trellis_mul1_e4m3: int = 0
        self.trellis_shared_su: bool = False
        # "mxfp8" routes non-ignored dense linears to the serialized loader
        # (offline-baked fp8 weights + e8m0 scales in the checkpoint).
        self.dense_format: str | None = None
        self.dense_ignored_layers: list[str] = []
        self.shared_runtime = _HybridSharedRuntime()

    def get_name(self) -> QuantizationMethods:
        return "nvfp4_nf3_hybrid"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Quantize only routed experts; K3's remaining tensors are BF16.

        The source checkpoint has no serialized ModelOpt tensors for dense
        linears. Inheriting ModelOpt's default selection would nevertheless
        allocate FP4 parameters for every such layer and make them unloadable.
        """
        if isinstance(layer, RoutedExperts):
            return self.FusedMoEMethodCls(
                quant_config=self, moe_config=layer.moe_config
            )
        if isinstance(layer, LinearBase):
            # Serialized-MXFP8 dense linears (kquant offline bake): same
            # dense_format convention Fp8Config uses. Modules in
            # dense_ignored_layers stay BF16 (kv_b_proj, KDA gate heads...).
            if self.dense_format == "mxfp8":
                from vllm.model_executor.layers.quantization.fp8 import (
                    Mxfp8SerializedLinearMethod,
                )

                if not _is_dense_layer_ignored(
                    prefix=prefix,
                    ignored_layers=self.dense_ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
                ):
                    return Mxfp8SerializedLinearMethod()
                return UnquantizedLinearMethod()
            # Honor the --quantization-config online overlay (MXFP8 on BF16
            # attention/shared-expert linears): at K3 TP6/TP12 the per-GPU
            # ledger needs the halved non-expert footprint; without an overlay
            # spec this falls through to plain BF16.
            online = self._get_shared_expert_online_method(
                layer, prefix
            ) or self._get_dense_linear_online_method(layer, prefix)
            return online or UnquantizedLinearMethod()
        # In particular, do not inherit ModelOpt's serialized-NVFP4 method for
        # ParallelLMHead/VocabParallelEmbedding. K3 stores both as BF16; using
        # the parent method allocates a packed [vocab, hidden/2] parameter and
        # then fails when the [vocab, hidden] checkpoint tensor is loaded.
        return None

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant is not None and user_quant != "nvfp4_nf3_hybrid":
            # Respect an explicit --quantization choice.
            return None
        hybrid_bit_map, _ = _read_hybrid_keys(hf_quant_cfg)
        if hybrid_bit_map:
            return "nvfp4_nf3_hybrid"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> "NvFp4Nf3HybridConfig":
        hybrid_bit_map, kept_format = _read_hybrid_keys(original_config)
        if not isinstance(hybrid_bit_map, dict) or not hybrid_bit_map:
            raise ValueError(
                "nvfp4_nf3_hybrid requires a non-empty 'hybrid_bit_map' dict "
                "in the checkpoint quantization config."
            )
        config = super()._from_config(
            quant_method=quant_method,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            original_config=original_config,
            group_size=group_size,
            **kwargs,
        )
        assert isinstance(config, NvFp4Nf3HybridConfig)
        config.hybrid_bit_map = hybrid_bit_map
        config.kept_format = kept_format
        nf3_levels = original_config.get("nf3_levels")
        quantization = original_config.get("quantization")
        if nf3_levels is None and isinstance(quantization, dict):
            nf3_levels = quantization.get("nf3_levels")
        if nf3_levels is not None:
            if len(nf3_levels) != 8:
                raise ValueError("nf3_levels must contain exactly 8 dequant levels")
            config.nf3_levels = [float(v) for v in nf3_levels]
        demoted_format = original_config.get("demoted_format")
        if demoted_format is None and isinstance(quantization, dict):
            demoted_format = quantization.get("demoted_format")
        if demoted_format is not None:
            if demoted_format not in (
                "nf3_2p1",
                "exl3_3",
                "mixed_exl3_tp12",
                "qsrt_tp12",
            ):
                raise ValueError(f"unsupported demoted_format {demoted_format!r}")
            config.demoted_format = demoted_format
        mixed_exl3_tp12 = original_config.get("mixed_exl3_tp12")
        if mixed_exl3_tp12 is None and isinstance(quantization, dict):
            mixed_exl3_tp12 = quantization.get("mixed_exl3_tp12")
        if demoted_format == "mixed_exl3_tp12":
            if not isinstance(mixed_exl3_tp12, dict):
                raise ValueError(
                    "mixed_exl3_tp12 demotion requires a mixed_exl3_tp12 "
                    "format descriptor"
                )
            if mixed_exl3_tp12.get("schema") != "kquant_mixed_exl3_tp12_proto_v3":
                raise ValueError("unsupported mixed_exl3_tp12 schema")
            if mixed_exl3_tp12.get("tp_size") != 12:
                raise ValueError("mixed_exl3_tp12 format requires tp_size=12")
            kept_storage = original_config.get("kept_storage")
            if kept_storage is None and isinstance(quantization, dict):
                kept_storage = quantization.get("kept_storage")
            if kept_storage is None:
                kept_storage = "inline-mxfp4"
            if kept_storage not in {"inline-mxfp4", "external-x4t"}:
                raise ValueError(
                    f"unsupported mixed_exl3_tp12 kept_storage {kept_storage!r}"
                )
            if kept_storage == "external-x4t":
                expected_external = {
                    "layer_header_version": 5,
                    "x4t_tp12_rank_file_pattern": (
                        "x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"
                    ),
                    "x4t_tp12_version": 1,
                }
                for name, expected in expected_external.items():
                    if mixed_exl3_tp12.get(name) != expected:
                        raise ValueError(
                            f"external-X4T mixed_exl3_tp12 {name} must be {expected!r}"
                        )
            elif mixed_exl3_tp12.get("layer_header_version") not in (None, 3, 4):
                raise ValueError(
                    "inline mixed_exl3_tp12 requires layer header version 3 or 4"
                )
            config.kept_storage = kept_storage
            config.mixed_exl3_tp12 = dict(mixed_exl3_tp12)
        qsrt_tp12 = original_config.get("qsrt_tp12")
        if qsrt_tp12 is None and isinstance(quantization, dict):
            qsrt_tp12 = quantization.get("qsrt_tp12")
        if demoted_format == "qsrt_tp12":
            if not isinstance(qsrt_tp12, dict):
                raise ValueError(
                    "qsrt_tp12 demotion requires a qsrt_tp12 format descriptor"
                )
            expected_qsrt = {
                "schema": "kquant_kimi_k3_qsrt_tp12_v1",
                "layer_header_version": 5,
                "tp_size": 12,
                "layer_file_pattern": "qsrt-tp12-layer-{layer:05d}.bin",
                "x4t_tp12_rank_file_pattern": (
                    "x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"
                ),
                "x4t_tp12_version": 1,
            }
            for name, expected in expected_qsrt.items():
                if qsrt_tp12.get(name) != expected:
                    raise ValueError(
                        f"QSRT TP12 {name} must be {expected!r}, got "
                        f"{qsrt_tp12.get(name)!r}"
                    )
            kept_storage = original_config.get("kept_storage")
            if kept_storage is None and isinstance(quantization, dict):
                kept_storage = quantization.get("kept_storage")
            if kept_storage != "external-x4t":
                raise ValueError("QSRT TP12 requires kept_storage='external-x4t'")
            config.kept_storage = kept_storage
            config.qsrt_tp12 = dict(qsrt_tp12)
        trellis = original_config.get("trellis")
        if trellis is None and isinstance(quantization, dict):
            trellis = quantization.get("trellis")
        if isinstance(trellis, dict):
            codebook = str(trellis.get("codebook", "mcg")).lower()
            if codebook not in {
                "mcg",
                "mul1-e4m3",
                "sqg-normal-e4m3",
                "sqg-cheb-normal-e4m3",
                "sqg-cheb-normal-k2-q8h4-w2-e4m3",
            }:
                raise ValueError(f"unsupported EXL3 trellis codebook {codebook!r}")
            config.trellis_codebook = codebook
            if codebook == "mcg" and "mcg_mult" in trellis:
                config.trellis_mcg = int(
                    torch.tensor(int(trellis["mcg_mult"]), dtype=torch.uint32).view(
                        torch.int32
                    )
                )
            if codebook == "mul1-e4m3":
                if trellis.get("reconstruction_dtype") != "e4m3":
                    raise ValueError(
                        "mul1-e4m3 trellis requires reconstruction_dtype='e4m3'"
                    )
                if "mul1_mult" not in trellis:
                    raise ValueError("mul1-e4m3 trellis requires the mul1_mult marker")
                config.trellis_mul1_e4m3 = int(
                    torch.tensor(int(trellis["mul1_mult"]), dtype=torch.uint32).view(
                        torch.int32
                    )
                )
            if codebook == "sqg-normal-e4m3":
                expected_sqg = {
                    "labelling": "sqg-l16-normal-r44-v1",
                    "reconstruction_dtype": "e4m3",
                    "rate_dependent_reconstruction": True,
                    "mode_ids": [0, 1, 2],
                    "separate_r13_r2": True,
                }
                for name, expected in expected_sqg.items():
                    if trellis.get(name) != expected:
                        raise ValueError(
                            f"sqg-normal-e4m3 trellis {name} must be {expected!r}"
                        )
            if codebook == "sqg-cheb-normal-e4m3":
                expected_sqg_cheb = {
                    "labelling": "sqg-l16-normal-cheb-v1",
                    "reconstruction_dtype": "e4m3",
                    "rate_dependent_reconstruction": True,
                    "mode_ids": [0, 1, 2],
                    "separate_r13_r2": True,
                }
                for name, expected in expected_sqg_cheb.items():
                    if trellis.get(name) != expected:
                        raise ValueError(
                            f"sqg-cheb-normal-e4m3 trellis {name} must be {expected!r}"
                        )
            if codebook == "sqg-cheb-normal-k2-q8h4-w2-e4m3":
                expected_sqg_cheb_q8h4 = {
                    "labelling": "sqg-l16-normal-cheb-k2-q8h4-w2-v1",
                    "reconstruction_dtype": "e4m3",
                    "rate_dependent_reconstruction": True,
                    "mode_ids": [0, 1, 2],
                    "separate_r13_r2": True,
                }
                for name, expected in expected_sqg_cheb_q8h4.items():
                    if trellis.get(name) != expected:
                        raise ValueError(
                            f"SQG-Cheb Q8H4 trellis {name} must be {expected!r}"
                        )
            # shared-su artifacts store one H-side rotation row per
            # (layer, matrix); register [1, ...] params and let the kernels
            # broadcast (zero expert stride).
            config.trellis_shared_su = bool(trellis.get("shared_su", False))
        dense_format = original_config.get("dense_format")
        if dense_format is not None:
            if dense_format != "mxfp8":
                raise ValueError(f"unsupported dense_format {dense_format!r}")
            config.dense_format = dense_format
            config.dense_ignored_layers = list(
                original_config.get("ignored_layers") or []
            )
        return config


class NvFp4Nf3HybridMoEMethod(FusedMoEMethodBase):
    """Fused-MoE method serving both hybrid tiers via the b12x W4A16 kernel.

    Weight storage is compact two-group: per layer, kept experts and NF3
    experts are stored in separate stacked tensors, and a custom per-param
    weight loader demultiplexes each checkpoint expert into its tier slot
    (TP-sharding gate/up along dim 0 and down along dim 1). ``apply``
    returns the routed-experts output only; routing and shared experts are
    handled upstream by the MoE runner.
    """

    def __init__(
        self,
        quant_config: NvFp4Nf3HybridConfig,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> "mk.FusedMoEPrepareAndFinalizeModular | None":
        # The hybrid forward is self-contained (preplanned b12x launches);
        # the MXFP4 kept-tier modular kernel, when built, owns its own
        # prepare/finalize.
        return None

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        # Quant params are consumed directly by the b12x prepare/launch path.
        return None

    def _layer_bits(self, layer: "RoutedExperts") -> list[int] | None:
        """Per-expert bit widths for this layer, or None if unmapped."""
        match = re.search(r"layers\.(\d+)\b", layer.layer_name)
        if match is None:
            return None
        return self.quant_config.hybrid_bit_map.get(match.group(1))

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert self.quant_config.is_checkpoint_nvfp4_serialized
        if layer.activation not in (MoEActivation.SILU, MoEActivation.SITU):
            raise NotImplementedError(
                "nvfp4_nf3_hybrid only supports SiLU/SiTU-gated MoE layers, got "
                f"{layer.activation}."
            )
        if self.quant_config.nf3_levels is not None:
            _apply_nf3_codebook_override(self.quant_config.nf3_levels)
        bits = self._layer_bits(layer)
        mapped_layer = bits is not None
        kept_mx = mapped_layer and self.quant_config.kept_format == "mxfp4_e8m0k32"
        if bits is None:
            # MoE layer absent from hybrid_bit_map (e.g. an MTP head): its
            # experts are uniform NVFP4; run it through the hybrid path as
            # all-kept so it shares this loader and kernel.
            bits = [4] * num_experts
        if len(bits) != num_experts:
            raise ValueError(
                f"hybrid_bit_map entry for {layer.layer_name} has {len(bits)} "
                f"experts, expected {num_experts}."
            )
        hidden = hidden_size
        inter = intermediate_size_per_partition
        group_size = self.quant_config.group_size
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        kept = [e for e, b in enumerate(bits) if b == 4]
        demoted = [e for e, b in enumerate(bits) if b == 3]
        if len(kept) + len(demoted) != num_experts:
            raise ValueError(
                f"hybrid_bit_map entry for {layer.layer_name} contains bit "
                "widths other than 4 (kept) and 3 (NF3)."
            )
        remap = {
            **{e: (0, i) for i, e in enumerate(kept)},
            **{e: (1, i) for i, e in enumerate(demoted)},
        }
        state = _HybridLayerState(remap, hidden, inter, num_experts, kept_mx)
        state.uses_mixed_tp12_slab = bool(
            mapped_layer
            and self.quant_config.demoted_format in {"mixed_exl3_tp12", "qsrt_tp12"}
        )
        layer.hybrid_state = state

        if state.uses_mixed_tp12_slab:
            if tp_size != 12:
                raise ValueError(
                    f"mixed_exl3_tp12 serving requires TP=12, got TP={tp_size}"
                )
            if hidden != 3584 or inter != 256 or num_experts != 896:
                raise ValueError(
                    "mixed_exl3_tp12 serving requires K3's rank-local "
                    "H=3584, I=256, E=896 geometry"
                )
            if not kept_mx:
                raise ValueError(
                    "mixed_exl3_tp12 serving requires kept_format='mxfp4_e8m0k32'"
                )
            placeholder = torch.nn.Parameter(
                torch.empty(
                    (0,),
                    dtype=torch.uint8,
                    device=torch.accelerator.current_device_index(),
                ),
                requires_grad=False,
            )
            layer.register_parameter("mixed_exl3_tp12_placeholder", placeholder)
            return

        def hybrid_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            name_mapped: str | None = None,
            *,
            weight_name: str | None = None,
            shard_id: str | None = None,
            expert_id: int | None = None,
            return_success: bool = False,
            **kwargs,
        ) -> bool:
            """Demux one checkpoint expert tensor into its tier storage.

            The registered params under the stock expert-mapping names are
            dispatchers; the real block-scale storage is selected here by
            the expert's tier. Always returns True (success).
            """
            name = name_mapped or weight_name or ""
            if "input_scale" in name:  # W4A16: activation scales are unused
                return True
            if expert_id is None:
                raise ValueError(f"expert tensor {name!r} is missing expert_id")
            tier, local_id = state.remap[int(expert_id)]
            if "exl3_" in name:
                # Native EXL3 tier tensors. TP sharding slices only the
                # intermediate axis (whole 16-tiles / whole 128-Hadamard
                # blocks, so slicing is exact).
                assert tier == 1, f"exl3 tensor for kept expert: {name}"
                family = "w13" if "w13_" in name else "w2"
                part = name.rsplit("exl3_", 1)[1]
                target = getattr(layer, f"{family}_exl3_{part}")
                lw = loaded_weight
                if tp_size > 1:
                    if family == "w13":
                        if part == "trellis":
                            lw = lw.chunk(tp_size, 1)[tp_rank]  # n-tiles (I)
                        elif part == "svh":
                            lw = lw.chunk(tp_size, 0)[tp_rank]  # I axis
                        # suh spans H: replicated
                    else:
                        if part == "trellis":
                            lw = lw.chunk(tp_size, 0)[tp_rank]  # k-tiles (I)
                        elif part == "suh":
                            lw = lw.chunk(tp_size, 0)[tp_rank]  # I axis
                        # svh spans H: replicated
                # shared-su artifacts register one broadcast row for the
                # H-side vectors; every expert carries an identical copy, so
                # writes to row 0 are idempotent.
                if target.data.shape[0] == 1:
                    local_id = 0
                if family == "w13":
                    widx = 0 if shard_id == "w1" else 1
                    dst = target.data[local_id, widx]
                else:
                    dst = target.data[local_id]
                dst.copy_(lw.reshape(dst.shape).to(dst.dtype))
                return True
            family = "w13" if "w13_" in name else "w2"
            if "weight_scale_2" in name:  # NVFP4 per-tensor global (kept only)
                target = getattr(layer, f"{family}_weight_scale_2")
                if family == "w13":
                    col = 0 if shard_id == "w1" else 1
                    target.data[local_id, col] = loaded_weight.reshape(()).to(
                        target.dtype
                    )
                else:
                    target.data[local_id] = loaded_weight.reshape(()).to(target.dtype)
                return True
            # TP-shard the block-quantized 2D tensor (gate/up dim 0, down dim 1).
            if tp_size > 1 and loaded_weight.ndim >= 2:
                if shard_id in ("w1", "w3"):
                    loaded_weight = loaded_weight.chunk(tp_size, 0)[tp_rank]
                elif shard_id == "w2":
                    loaded_weight = loaded_weight.chunk(tp_size, 1)[tp_rank]
            if "weight_scale" in name:  # block scale: demux by tier
                suffix = "_nv_scale" if tier == 0 else "_nf3_scale"
                target = getattr(layer, f"{family}{suffix}")
            elif tier == 1:  # NF3 packed codes (serialized as nf3_packed)
                target = getattr(layer, f"{family}_weight_packed")
            else:  # plain NVFP4/MXFP4 weight
                target = getattr(layer, f"{family}_weight")
            dst = target.data[local_id]
            if family == "w13" and shard_id in ("w1", "w3"):
                # gate -> top half, up -> bottom half of the fused rows.
                half = dst.shape[0] // 2
                dst = dst[:half] if shard_id == "w1" else dst[half:]
            if loaded_weight.numel() != dst.numel():
                raise RuntimeError(
                    "hybrid expert tensor shape mismatch: "
                    f"layer={layer.layer_name}, expert={expert_id}, tier={tier}, "
                    f"shard={shard_id}, mapped_name={name}, "
                    f"checkpoint_shape={tuple(loaded_weight.shape)}, "
                    f"destination_shape={tuple(dst.shape)}"
                )
            loaded_weight = loaded_weight.reshape(dst.shape)
            if tier == 1 and "weight_scale" in name:
                loaded_weight = _decode_kquant_nf3_scale(loaded_weight)
            dst.copy_(loaded_weight.to(dst.dtype))
            return True

        def register(name: str, shape: tuple[int, ...], dtype=torch.uint8) -> None:
            param = torch.nn.Parameter(
                torch.zeros(
                    shape,
                    dtype=dtype,
                    device=torch.accelerator.current_device_index(),
                ),
                requires_grad=False,
            )
            set_weight_attrs(param, {"weight_loader": hybrid_weight_loader})
            layer.register_parameter(name, param)

        num_kept = max(state.num_kept, 1)
        num_nf3 = max(state.num_nf3, 1)
        # Names the stock prefix-based expert mapping produces; the scalar
        # *_weight_scale / *_input_scale entries are dispatchers whose loads
        # are routed (or dropped) by hybrid_weight_loader above.
        exl3 = self.quant_config.demoted_format == "exl3_3"
        if exl3:
            # Native EXL3 trellis tensors (16x16 tiles, K=3). w13 stacks
            # gate (idx 0) and up (idx 1); `inter` is already the TP-local
            # intermediate. suh spans the unsharded input axis, svh the
            # unsharded output axis; the sharded counterparts slice along
            # the intermediate axis in the loader.
            tb = 48  # 16 * 3 bits
            register(
                "w13_exl3_trellis",
                (num_nf3, 2, hidden // 16, inter // 16, tb),
                torch.int16,
            )
            # shared-su artifacts: one broadcast row instead of per-expert
            # H-side vectors (saves ~1.4 GiB/rank at K3 scale).
            h_rows = (
                1 if getattr(self.quant_config, "trellis_shared_su", False) else num_nf3
            )
            register("w13_exl3_suh", (h_rows, 2, hidden), torch.float16)
            register("w13_exl3_svh", (num_nf3, 2, inter), torch.float16)
            register(
                "w2_exl3_trellis",
                (num_nf3, inter // 16, hidden // 16, tb),
                torch.int16,
            )
            register("w2_exl3_suh", (num_nf3, inter), torch.float16)
            register("w2_exl3_svh", (h_rows, hidden), torch.float16)
        register("w13_weight", (num_kept, 2 * inter, hidden // 2))
        register(
            "w13_weight_packed",
            (1 if exl3 else num_nf3, 2 * inter, hidden // 8 * 3),
        )
        register("w13_weight_scale", (1,))
        register("w13_weight_scale_2", (num_kept, 2), torch.float32)
        register("w13_input_scale", (1,), torch.float32)
        register("w2_weight", (num_kept, hidden, inter // 2))
        register(
            "w2_weight_packed",
            (1 if exl3 else num_nf3, hidden, inter // 8 * 3),
        )
        register("w2_weight_scale", (1,))
        register("w2_weight_scale_2", (num_kept,), torch.float32)
        register("w2_input_scale", (1,), torch.float32)
        # Real block-scale storage, filled by the dispatcher (not routed by
        # the expert mapping). MXFP4 kept tier stores ue8m0 scales per 32
        # group (uint8) instead of e4m3 per group_size.
        nv_group = 32 if kept_mx else group_size
        nv_dtype = torch.uint8 if kept_mx else torch.float8_e4m3fn
        # NF3 scale storage is CPU-staged: at 86 GiB/GPU of expert codes
        # (K3 TP6/TP12) the extra ~7 GiB of checkpoint-layout scales does not
        # fit alongside BF16 non-expert weights during load. The repack
        # streams them through the GPU per chunk; _vllm_keep_on_cpu stops
        # device_loading_context from bulk-moving them first.
        for name, shape, dtype, dev in (
            ("w13_nv_scale", (num_kept, 2 * inter, hidden // nv_group), nv_dtype, None),
            (
                "w13_nf3_scale",
                (num_nf3, 2 * inter, hidden // 32),
                torch.float8_e4m3fn,
                "cpu",
            ),
            ("w2_nv_scale", (num_kept, hidden, inter // nv_group), nv_dtype, None),
            (
                "w2_nf3_scale",
                (num_nf3, hidden, inter // 32),
                torch.float8_e4m3fn,
                "cpu",
            ),
        ):
            scale_param = torch.nn.Parameter(
                torch.zeros(
                    shape,
                    dtype=dtype,
                    device=dev or torch.accelerator.current_device_index(),
                ),
                requires_grad=False,
            )
            if dev == "cpu":
                scale_param._vllm_keep_on_cpu = True
            layer.register_parameter(name, scale_param)

    def _load_mixed_exl3_tp12_slab(
        self,
        layer: "RoutedExperts",
        *,
        device: torch.device,
    ) -> None:
        """Load this process's rank section from the fixed K3 TP12 slab."""

        from sparkinfer.moe import fused_moe

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.quantization.kquant_kimi_k3_qsrt_tp12 import (
            MCG_MULT,
            MUL1_MULT,
            read_tp12_rank_payload,
        )

        state: _HybridLayerState = layer.hybrid_state
        match = re.search(r"layers\.(\d+)\b", layer.layer_name)
        if match is None:
            raise ValueError(
                f"cannot resolve a mixed_exl3_tp12 layer from {layer.layer_name!r}"
            )
        layer_index = int(match.group(1))
        bits = self._layer_bits(layer)
        if bits is None:
            raise ValueError("mixed_exl3_tp12 layer is absent from hybrid_bit_map")
        tp_rank = get_tensor_model_parallel_rank()
        model_root = Path(get_current_vllm_config().model_config.model)
        if not model_root.is_dir():
            raise ValueError(
                "mixed_exl3_tp12 serving requires a local model directory, got "
                f"{model_root}"
            )
        descriptor = (
            self.quant_config.qsrt_tp12
            if self.quant_config.demoted_format == "qsrt_tp12"
            else self.quant_config.mixed_exl3_tp12
        )
        assert descriptor is not None
        slab_pattern = descriptor.get(
            "layer_file_pattern", "mixed-exl3-tp12-layer-{layer:05d}.bin"
        )
        slab_path = model_root / slab_pattern.format(layer=layer_index)
        x4t_tp12_path = None
        if self.quant_config.kept_storage == "external-x4t":
            pattern = descriptor["x4t_tp12_rank_file_pattern"]
            candidate = model_root / pattern.format(
                layer=layer_index,
                rank=tp_rank,
            )
            if not candidate.is_file():
                raise FileNotFoundError(
                    "external-X4T TP12 serving requires its persistent rank "
                    f"checkpoint shard: {candidate}"
                )
            x4t_tp12_path = candidate
        payload = read_tp12_rank_payload(
            slab_path,
            layer=layer_index,
            rank=tp_rank,
            x4t_tp12_path=x4t_tp12_path,
            expected_bits=bits,
            expected_codebook=self.quant_config.trellis_codebook,
        )
        expected_compressed = tuple(
            global_id
            for global_id, (tier, _local_id) in sorted(
                state.remap.items(), key=lambda item: item[1][1]
            )
            if tier == 1
        )
        expected_kept = tuple(
            global_id
            for global_id, (tier, _local_id) in sorted(
                state.remap.items(), key=lambda item: item[1][1]
            )
            if tier == 0
        )
        if tuple(payload.compressed_expert_ids.tolist()) != expected_compressed:
            raise ValueError(
                "mixed_exl3_tp12 compressed tier order disagrees with remap"
            )
        if tuple(payload.kept_expert_ids.tolist()) != expected_kept:
            raise ValueError("mixed_exl3_tp12 kept tier order disagrees with remap")

        def on_device(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=device, non_blocking=False).contiguous()

        if state.num_nf3:
            w13 = on_device(payload.w13_trellis)
            w2 = on_device(payload.w2_trellis)
            codebook = self.quant_config.trellis_codebook
            source_formats = {
                "mcg": "exl3_trellis_mcg",
                "mul1-e4m3": "exl3_trellis_mul1_e4m3",
                "sqg-normal-e4m3": "exl3_trellis_sqg_e4m3",
                "sqg-cheb-normal-e4m3": "exl3_trellis_sqg_cheb_e4m3",
                "sqg-cheb-normal-k2-q8h4-w2-e4m3": (
                    "exl3_trellis_sqg_cheb_k2_q8h4_w2_e4m3"
                ),
            }
            source_format = source_formats[codebook]
            weight_plan = fused_moe.plan_weights(
                quant_modes="w4a16",
                source_format=source_format,
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                num_experts=state.num_nf3,
                hidden_size=state.hidden_size,
                intermediate_size=state.intermediate_size,
                w13_layout="w13",
                trellis_bits=3,
                trellis_tile_config=state.tiles,
                trellis_pair_format="tp12_p24_p33",
            )
            if codebook == "mcg":
                marker = self.quant_config.trellis_mcg
                expected_marker = MCG_MULT
                marker_args = {"trellis_mcg": marker}
            elif codebook == "mul1-e4m3":
                marker = self.quant_config.trellis_mul1_e4m3
                expected_marker = MUL1_MULT
                marker_args = {"trellis_mul1_e4m3": marker}
            else:
                marker = 0
                expected_marker = 0
                marker_args = {}
            if marker & 0xFFFFFFFF != expected_marker:
                raise ValueError(
                    f"mixed_exl3_tp12 checkpoint has the wrong {codebook} marker: "
                    f"{marker & 0xFFFFFFFF:#010x}"
                )
            state.trellis_weights = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=self.moe.in_dtype,
                w1_fp4=w13,
                w2_fp4=w2,
                gate_suh=on_device(payload.gate_suh),
                up_suh=on_device(payload.up_suh),
                intermediate_rotations=on_device(payload.intermediate_rotations),
                down_svh=on_device(payload.down_svh),
                trellis_fc1_pair_modes=on_device(payload.fc1_pair_modes),
                trellis_fc2_pair_modes=on_device(payload.fc2_pair_modes),
                # The payload was freshly loaded for this layer and has no
                # other owner.  Swizzle it in bounded expert chunks instead
                # of allocating a second full FC1 slab during model startup.
                # This preparation choice is independent of whether serving
                # later dispatches the W4A8 or W4A16 execution path.
                trellis_inplace_fc1_pair_swizzle=True,
                **marker_args,
            )

        def register_loaded(name: str, value: torch.Tensor) -> None:
            layer.register_parameter(
                name,
                torch.nn.Parameter(on_device(value), requires_grad=False),
            )

        if state.num_kept and self.quant_config.kept_storage == "external-x4t":
            from sparkinfer._lib.quant.x4t_scales import make_x4t_scale_batch
            from sparkinfer.moe._shared.kernels.w4a16.prepare import (
                prepare_w4a16_x4t_tp12_weights,
            )

            if (
                payload.w13_x4t_scale_components is None
                or payload.w2_x4t_scale_components is None
            ):
                raise ValueError(
                    "external X4T serving payload omitted its stored scale streams"
                )
            w13_components = payload.w13_x4t_scale_components
            w2_components = payload.w2_x4t_scale_components
            w13_x4t = make_x4t_scale_batch(
                [component.fixed for component in w13_components],
                [component.exceptions for component in w13_components],
                rows=512,
                columns=112,
                device=device,
                exception_task_rows=_QSRT_X4T_W13_EXCEPTION_TASK_ROWS,
                # The sidecar's fused FC1 rows are already [gate; up]
                # (B12X ``w31``), so neither the sparse exception ranges nor
                # the packed W4A16 rows need a half rotation.
                exception_row_rotation=_QSRT_X4T_W13_EXCEPTION_ROW_ROTATION,
            )
            w2_x4t = make_x4t_scale_batch(
                [component.fixed for component in w2_components],
                [component.exceptions for component in w2_components],
                rows=3584,
                columns=8,
                device=device,
                exception_task_rows=_QSRT_X4T_W2_EXCEPTION_TASK_ROWS,
            )
            runtime = self.quant_config.shared_runtime
            if runtime.x4t_w13_scale_scratch is None:
                runtime.x4t_w13_scale_scratch = torch.empty(
                    (896, 112, 512), dtype=torch.uint8, device=device
                )
                runtime.x4t_w2_scale_scratch = torch.empty(
                    (896, 8, 3584), dtype=torch.uint8, device=device
                )
            assert runtime.x4t_w2_scale_scratch is not None
            global_scale = torch.ones(
                state.num_kept, dtype=torch.float32, device=device
            )
            state.prep_kept = prepare_w4a16_x4t_tp12_weights(
                on_device(payload.w13_mxfp4),
                w13_x4t,
                global_scale,
                on_device(payload.w2_mxfp4),
                w2_x4t,
                global_scale.clone(),
                runtime.x4t_w13_scale_scratch,
                runtime.x4t_w2_scale_scratch,
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                # The QSRT sidecar is fused in checkpoint order [w1; w3],
                # i.e. [gate; up].  B12X calls that physical order ``w31``;
                # ``w13`` means [up; gate] and would silently swap SiTU's
                # two inputs during the W4A16 repack.
                w13_layout=_QSRT_X4T_W13_LAYOUT,
            )
        elif state.num_kept:
            register_loaded("w13_weight", payload.w13_mxfp4)
            register_loaded("w13_nv_scale", payload.w13_mxfp4_scale)
            register_loaded("w2_weight", payload.w2_mxfp4)
            register_loaded("w2_nv_scale", payload.w2_mxfp4_scale)
        layer.mixed_exl3_tp12_placeholder.data = (
            layer.mixed_exl3_tp12_placeholder.data.new_empty((0,))
        )
        kept_label = (
            "external X4T"
            if self.quant_config.kept_storage == "external-x4t"
            else "MXFP4"
        )
        format_name = (
            "qsrt_tp12"
            if self.quant_config.demoted_format == "qsrt_tp12"
            else "mixed_exl3_tp12"
        )
        logger.info(
            "Loaded %s layer %d rank %d: %d compressed, %d %s via W4A16FusedMoeKernel",
            format_name,
            layer_index,
            tp_rank,
            state.num_nf3,
            state.num_kept,
            kept_label,
        )

    def _build_kept_mxfp4(self, layer: "RoutedExperts") -> None:
        """Build the MXFP4 kept tier as a modular kernel over the kept
        experts via the stock mxfp4 oracle chain (W4A16 activations).

        The kernel is built over a no-parallel clone of the MoE config with
        the per-rank intermediate size: the weights are already TP-sharded
        by the weight loader, so the kernel must see tp=1 (the layer's
        post-apply all-reduce handles TP). The b12x W4A16 kernel consumes the
        global->local table directly, so routing remains in global-id space.
        """
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            convert_weight_to_mxfp4_moe_kernel_format,
            make_mxfp4_moe_kernel,
            make_mxfp4_moe_quant_config,
            select_mxfp4_moe_backend,
        )

        state: _HybridLayerState = layer.hybrid_state
        device = layer.w13_weight.device
        num_kept = state.num_kept
        kept_moe = dataclasses.replace(
            self.moe,
            num_experts=num_kept,
            num_local_experts=num_kept,
            num_logical_experts=num_kept,
            intermediate_size=self.moe.intermediate_size_per_partition,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        )
        backend, experts_cls = select_mxfp4_moe_backend(kept_moe, activation_key=None)
        if experts_cls is None:
            raise RuntimeError("MXFP4 backend did not provide an experts class")
        kept_module = torch.nn.Module()
        kept_module.activation = layer.activation
        kept_module.moe_config = kept_moe
        kept_module.local_num_experts = num_kept
        # The compact kept tier is registered under its global parent layer;
        # suppress the ordinary B12X collector's local-expert registration.
        kept_module._kquant_capture_parent_managed = True
        w13, w2, w13_scale, w2_scale, _bias13, _bias2 = (
            convert_weight_to_mxfp4_moe_kernel_format(
                backend,
                kept_module,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_nv_scale,
                layer.w2_nv_scale,
            )
        )
        for name, value in (
            ("w13_weight", w13),
            ("w13_weight_scale", w13_scale),
            ("w2_weight", w2),
            ("w2_weight_scale", w2_scale),
        ):
            setattr(kept_module, name, value)
        quant_config = make_mxfp4_moe_quant_config(
            backend, w13_scale, w2_scale, layer=kept_module
        )
        if quant_config is None:
            raise RuntimeError("MXFP4 backend did not provide a quantization config")
        kernel = make_mxfp4_moe_kernel(
            quant_config,
            kept_moe,
            experts_cls,
            mxfp4_backend=backend,
            routing_tables=None,
        )
        _require_rank_local_kept_kernel(kernel)
        kernel.fused_experts.process_weights_after_loading(kept_module)
        prepared_experts = cast(Any, kernel.fused_experts)._lookup_prepared_experts()
        if prepared_experts is None:
            raise RuntimeError("MXFP4 modular kernel did not publish prepared weights")
        state.prep_kept_hybrid = prepared_experts.representation_for("w4a16")
        # Owning a modular kernel makes supports_internal_mk True, so vLLM's
        # post-load maybe_init_modular_kernel() returns early instead of
        # rebuilding a kernel from the (freed) standard weight attrs.
        self.moe_kernel = kernel
        # Global routes not owned by this compact tier remain -1. Both the
        # packed prefill route builder and direct TC-decode resolve this map
        # before any weight access.
        kept_remap = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            if tier == 0:
                kept_remap[global_id] = local_id
        state.kept_kernel = kernel
        state.kept_module = kept_module
        state.kept_remap = kept_remap
        state.keepalive = (w13, w2, w13_scale, w2_scale)
        # Free the compact kept originals (kept_module holds the converted
        # copies) so resident VRAM stays flat.
        for name in ("w13_weight", "w2_weight", "w13_nv_scale", "w2_nv_scale"):
            delattr(layer, name)
        # The prepared representation carries its own packed scale grids; the
        # pre-prepare scale tensors are then dead weight (38.5+19.3 MiB per K3
        # TP16 layer, ~5 GiB per rank over 92 layers). Release their storage
        # in place (every reference — keepalive, kept_module, quant config —
        # observes the swap) unless the prepared grids alias them.
        prep = state.prep_kept_hybrid
        prep_ptrs = set()
        for field in (
            "w13",
            "w2",
            "w13_scale",
            "w2_scale",
            "micro_w13_scale",
            "micro_w2_scale",
            "w13_global_scale",
            "w2_global_scale",
            "micro_w13_global_scale",
            "micro_w2_global_scale",
        ):
            value = getattr(prep, field, None)
            if isinstance(value, torch.Tensor):
                prep_ptrs.add(value.untyped_storage().data_ptr())
        freed = 0
        for tensor in (w13_scale, w2_scale):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.numel() > 0
                and tensor.untyped_storage().data_ptr() not in prep_ptrs
            ):
                freed += tensor.numel() * tensor.element_size()
                tensor.data = tensor.data.new_empty((0,))
        if freed:
            logger.info_once(
                "nvfp4_nf3_hybrid: released %.1f MiB/layer of pre-prepare kept "
                "scale storage (prepared grids are self-contained)",
                freed / 2**20,
            )

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        """Repack both tiers into b12x W4A16 kernel formats.

        NF3 tier first (the kept-tier builders free the originals): unpack
        the checkpoint's 8-per-3-byte codes and pack them into the
        ``nf3_2p1`` flat-span layout with ``e4m3_k32`` scales. Kept tier:
        MXFP4 goes through the production mxfp4 oracle chain
        (:meth:`_build_kept_mxfp4`); NVFP4 is repacked into the
        ``packed``/``e4m3_k16`` W4A16 layout. Launches and scratch buffers
        are built lazily at first apply (top-k and the real max batch size
        are known there, and the first forward is vLLM's eager profile run,
        so nothing compiles inside CUDA-graph capture).
        """
        from sparkinfer.moe._shared.kernels.w4a16.prepare import (
            PreparedNF3MoeWeights,
            W4A16PackedWeights,
            _make_workspace,
            _nf3_pack_code_experts,
            _nf3_pack_scale_experts,
            _permute_nvfp4_scales,
            _repack_weight,
        )

        state: _HybridLayerState = layer.hybrid_state
        hidden, inter = state.hidden_size, state.intermediate_size
        device = (
            torch.device("cuda", torch.accelerator.current_device_index())
            if state.uses_mixed_tp12_slab
            else layer.w13_weight.device
        )
        num_kept, num_nf3 = state.num_kept, state.num_nf3
        emap_kept = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        emap_nf3 = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            (emap_kept if tier == 0 else emap_nf3)[global_id] = local_id
        state.emap_kept, state.emap_nf3 = emap_kept, emap_nf3
        fc1_tile_n, fc2_tile_n = state.tiles[1], state.tiles[3]

        if state.uses_mixed_tp12_slab:
            self._load_mixed_exl3_tp12_slab(layer, device=device)
        elif num_nf3 > 0 and self.quant_config.demoted_format == "exl3_3":
            from sparkinfer.moe import fused_moe

            # Projection-major native stacks; prepare_weights wraps zero-copy.
            w13 = layer.w13_exl3_trellis.data.permute(1, 0, 2, 3, 4).contiguous()
            w2t = layer.w2_exl3_trellis.data.contiguous()
            # B12X consumes the two FC1 output scales before the FC2 input
            # scale.  This order is observable for SiTU because its up branch
            # is nonlinear; the latter two blocks cannot be commuted.
            inter_rot = _stack_exl3_intermediate_rotations(
                layer.w13_exl3_svh.data,
                layer.w2_exl3_suh.data,
            )
            wplan = fused_moe.plan_weights(
                quant_modes="w4a16",
                source_format="exl3_trellis_mcg",
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                num_experts=num_nf3,
                hidden_size=hidden,
                intermediate_size=inter,
                w13_layout="w13",
                trellis_bits=3,
            )
            state.trellis_weights = fused_moe.prepare_weights(
                plan=wplan,
                params_dtype=self.moe.in_dtype,
                w1_fp4=w13,
                w2_fp4=w2t,
                gate_suh=layer.w13_exl3_suh.data[:, 0].contiguous(),
                up_suh=layer.w13_exl3_suh.data[:, 1].contiguous(),
                intermediate_rotations=inter_rot,
                down_svh=layer.w2_exl3_svh.data.contiguous(),
                trellis_mcg=self.quant_config.trellis_mcg,
            )
            for pname in (
                "w13_exl3_trellis",
                "w13_exl3_suh",
                "w13_exl3_svh",
                "w2_exl3_trellis",
                "w2_exl3_suh",
                "w2_exl3_svh",
            ):
                p = getattr(layer, pname)
                p.data = p.data.new_empty((0,))
            torch.accelerator.empty_cache()
        elif num_nf3 > 0:

            def drop_parameter_data(name: str) -> None:
                param = getattr(layer, name)
                param.data = param.data.new_empty((0,))

            # Allocate each resident plane once and fill it by expert chunks.
            # The previous list+cat path retained every chunk and then allocated
            # a second full-size tensor (429 MiB for K3 TP16 w13), leaving more
            # than 1 GiB of unusable holes in expandable CUDA allocator
            # segments. Direct copies both lower peak VRAM and keep the final
            # allocation topology stable for the subsequent 1M-token KV cache.
            w13_words = 3 * (2 * inter) * hidden // 32
            w13_nf3 = torch.empty(
                (num_nf3, w13_words), dtype=torch.int32, device=device
            )
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                end = min(start + _NF3_PACK_CHUNK, num_nf3)
                codes = _unpack_nf3_codes(layer.w13_weight_packed[start:end], hidden)
                packed = _nf3_pack_code_experts(
                    codes, size_k=hidden, size_n=2 * inter, tile_n=fc1_tile_n
                )
                w13_nf3[start:end].copy_(packed)
                del codes, packed
            drop_parameter_data("w13_weight_packed")

            w2_words = 3 * hidden * inter // 32
            w2_nf3 = torch.empty((num_nf3, w2_words), dtype=torch.int32, device=device)
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                end = min(start + _NF3_PACK_CHUNK, num_nf3)
                codes = _unpack_nf3_codes(layer.w2_weight_packed[start:end], inter)
                packed = _nf3_pack_code_experts(
                    codes, size_k=inter, size_n=hidden, tile_n=fc2_tile_n
                )
                w2_nf3[start:end].copy_(packed)
                del codes, packed
            drop_parameter_data("w2_weight_packed")

            w13_nf3_scale = torch.empty(
                (num_nf3, hidden // 32, 2 * inter),
                dtype=torch.uint8,
                device=device,
            )
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                end = min(start + _NF3_PACK_CHUNK, num_nf3)
                packed = _nf3_pack_scale_experts(
                    layer.w13_nf3_scale[start:end].to(device).float() * (2.0**-4),
                    size_k=hidden,
                    size_n=2 * inter,
                )
                w13_nf3_scale[start:end].copy_(packed)
                del packed
            drop_parameter_data("w13_nf3_scale")

            w2_nf3_scale = torch.empty(
                (num_nf3, inter // 32, hidden),
                dtype=torch.uint8,
                device=device,
            )
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                end = min(start + _NF3_PACK_CHUNK, num_nf3)
                packed = _nf3_pack_scale_experts(
                    layer.w2_nf3_scale[start:end].to(device).float() * (2.0**-4),
                    size_k=inter,
                    size_n=hidden,
                )
                w2_nf3_scale[start:end].copy_(packed)
                del packed
            drop_parameter_data("w2_nf3_scale")

            nf3_global = torch.full(
                (num_nf3,), _NF3_GLOBAL_SCALE, dtype=torch.float32, device=device
            )
            state.prep_nf3 = PreparedNF3MoeWeights(
                w13=w13_nf3,
                w13_scale=w13_nf3_scale,
                w13_global_scale=nf3_global,
                w2=w2_nf3,
                w2_scale=w2_nf3_scale,
                w2_global_scale=nf3_global.clone(),
                workspace=_make_workspace(device),
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=num_nf3,
                is_gated=True,
                params_dtype=torch.bfloat16,
                fc1_tile_n=fc1_tile_n,
                fc2_tile_n=fc2_tile_n,
            )

        if num_kept > 0 and state.kept_mx and state.prep_kept is None:
            self._build_kept_mxfp4(layer)
        elif num_kept > 0 and state.prep_kept is None:
            # Kept NVFP4 through the "packed"/e4m3_k16 W4A16 layout. This is
            # byte-identical to the kernel's own prepare entry and lets the
            # TC-decode launches compile; no modular kernel is involved.
            g13 = layer.w13_weight_scale_2[:num_kept, 0].contiguous()
            g2 = layer.w2_weight_scale_2[:num_kept].contiguous()
            w13_packed = _repack_weight(
                layer.w13_weight.contiguous(), size_k=hidden, size_n=2 * inter
            )
            w2_packed = _repack_weight(
                layer.w2_weight.contiguous(), size_k=inter, size_n=hidden
            )
            w13_pscale, w13_pglobal = _permute_nvfp4_scales(
                layer.w13_nv_scale,
                g13,
                size_k=hidden,
                size_n=2 * inter,
                a_dtype=torch.bfloat16,
            )
            w2_pscale, w2_pglobal = _permute_nvfp4_scales(
                layer.w2_nv_scale,
                g2,
                size_k=inter,
                size_n=hidden,
                a_dtype=torch.bfloat16,
            )
            state.prep_kept = W4A16PackedWeights(
                w13=w13_packed,
                w13_scale=w13_pscale,
                w13_global_scale=w13_pglobal,
                w2=w2_packed,
                w2_scale=w2_pscale,
                w2_global_scale=w2_pglobal,
                workspace=_make_workspace(device),
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=num_kept,
                is_gated=True,
                params_dtype=torch.bfloat16,
                source_format="modelopt_nvfp4",
                w13_layout="w13",
                weight_layout="packed",
                scale_format="e4m3_k16",
            )
            for name in ("w13_weight", "w2_weight", "w13_nv_scale", "w2_nv_scale"):
                param = getattr(layer, name)
                param.data = param.data.new_empty((0,))

        if os.getenv("VLLM_KQUANT_CAPTURE_DIR"):
            from vllm.model_executor.layers.fused_moe.kquant_capture import (
                register_kquant_capture_layer,
            )

            prefix = str(layer.layer_name)
            register_kquant_capture_layer(
                prefix=prefix,
                device=device,
                hidden_size=hidden,
                local_intermediate_size=inter,
                num_experts=state.num_experts,
                topk=int(self.moe.experts_per_token),
                quant_mode="hybrid_exl3_3",
            )
            if state.kept_kernel is not None:
                state.kept_kernel.fused_experts._kquant_capture_prefix = prefix

    def _get_launch_pair(
        self, prepared: Any, state: _HybridLayerState
    ) -> tuple[Any, Any]:
        """Compile (or fetch cached) preplanned launches for one tier.

        The prefill launch covers ALL m in [1, max_m]: packed block-64
        routes + expert_map + ``zero_fc2_output=True``. The decode launch
        (m <= 8) compiles at forced pin tiles with block-8 direct top-k
        routing and a fused top-k sum; if that compile is unavailable the
        packed launch also serves decode.
        """
        from sparkinfer.moe._shared.kernels.w4a16.host import (
            max_packed_route_slots,
        )
        from sparkinfer.moe._shared.kernels.w4a16.kernel import (
            compile_w4a16_fused_moe,
        )

        runtime = self.quant_config.shared_runtime
        assert runtime.max_m is not None
        assert runtime.topk is not None
        max_m = runtime.max_m
        topk = runtime.topk
        hidden = self.moe.hidden_dim
        inter = self.moe.intermediate_size_per_partition
        key = (
            prepared.num_experts,
            prepared.weight_layout,
            prepared.scale_format,
            topk,
            max_m,
            hidden,
            inter,
            layer_activation := self.moe.activation.value,
            state.tiles,
        )
        cached = runtime.launches.get(key)
        if cached is not None:
            return cached
        props = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        )
        common = dict(
            hidden_size=hidden,
            intermediate_size=inter,
            num_experts=prepared.num_experts,
            top_k=topk,
            activation=layer_activation,
            apply_router_weight_on_input=False,
            element_dtype="bf16",
            fast_math=True,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            ),
            weight_layout=prepared.weight_layout,
            scale_format=prepared.scale_format,
            force_tile_config=state.tiles,
        )
        cap_slots = max_packed_route_slots(max_m * topk, 64, self.moe.num_experts)
        prefill = compile_w4a16_fused_moe(
            size_m=max_m,
            zero_fc2_output=True,
            moe_block_size=64,
            max_m_blocks=(cap_slots + 63) // 64,
            direct_topk_routes=False,
            tc_decode_fused_sum=False,
            **common,
        )
        assert (int(prefill.fc1_tile_n), int(prefill.fc2_tile_n)) == (
            state.tiles[1],
            state.tiles[3],
        ), "b12x tile pin failed"
        decode = prefill
        try:
            candidate = compile_w4a16_fused_moe(
                size_m=_B12X_DECODE_M,
                zero_fc2_output=False,
                moe_block_size=8,
                max_m_blocks=_B12X_DECODE_M * topk,
                direct_topk_routes=True,
                tc_decode_fused_sum=True,
                **common,
            )
            assert (int(candidate.fc1_tile_n), int(candidate.fc2_tile_n)) == (
                state.tiles[1],
                state.tiles[3],
            ), "b12x TC-decode tile pin failed"
            decode = candidate
        except Exception as exc:
            logger.warning_once(
                "nvfp4_nf3_hybrid: TC-decode launch compile failed (%s); "
                "decode steps fall back to the packed-route launch.",
                exc,
            )
        runtime.launches[key] = (decode, prefill)
        return runtime.launches[key]

    @staticmethod
    def _hybrid_prepared_views(prepared: Any) -> tuple[torch.Tensor, ...]:
        weight_dtype = (
            torch.uint8 if prepared.weight_layout == "modelopt" else torch.int32
        )
        return (
            prepared.w13.view(weight_dtype).view(-1),
            prepared.w2.view(weight_dtype).view(-1),
            prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
            prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
            prepared.w13_global_scale.view(-1),
            prepared.w2_global_scale.view(-1),
        )

    @staticmethod
    def _borrow_hybrid_scratch(
        buffers: Any,
        *,
        device: torch.device,
        routed_rows: int,
        fc1_cols: int,
        intermediate_size: int,
        scratch_elements: int,
        workspace_words: int,
    ) -> dict[str, torch.Tensor]:
        """Borrow serial-path buffers; the two decode paths never overlap."""
        specs = (
            # Flat 1-D views: the unified hybrid op compiles against flat
            # intermediate buffers ([m*topk rows] x fc1_cols / intermediate).
            (
                "fc1",
                "intermediate_cache13",
                torch.bfloat16,
                (routed_rows * fc1_cols,),
            ),
            (
                "activated",
                "intermediate_cache2",
                torch.bfloat16,
                (routed_rows * intermediate_size,),
            ),
            ("fc1_c_tmp", "fc1_c_tmp", torch.float32, (scratch_elements,)),
            ("fc2_c_tmp", "fc2_c_tmp", torch.float32, (scratch_elements,)),
        )
        borrowed: dict[str, torch.Tensor] = {}
        storage_ids: set[int] = set()
        for target_name, source_name, dtype, shape in specs:
            source = getattr(buffers, source_name, None)
            elements = 1
            for extent in shape:
                elements *= int(extent)
            if (
                source is None
                or source.dtype != dtype
                or source.device != device
                or not source.is_contiguous()
                or source.numel() < elements
                or source.data_ptr() == 0
                or source.data_ptr() % 16
            ):
                raise RuntimeError(
                    f"hybrid scratch source {source_name} failed admission"
                )
            storage_id = int(source.untyped_storage().data_ptr())
            if storage_id in storage_ids:
                raise RuntimeError("hybrid scratch sources alias each other")
            storage_ids.add(storage_id)
            borrowed[target_name] = source.view(-1)[:elements].view(shape)
        borrowed["workspace"] = torch.zeros(
            (workspace_words,), dtype=torch.int32, device=device
        )
        return borrowed

    def _prepare_grid188(self, layer: "RoutedExperts", topk: int) -> None:
        """Arm the b12x hybrid one-grid decode during the eager profile forward."""
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if state.grid188_ready or runtime.grid188_disabled_reason is not None:
            return
        if not _is_grid188_geometry(
            hidden_size=state.hidden_size,
            intermediate_size=state.intermediate_size,
            num_experts=state.num_experts,
            num_kept=state.num_kept,
            num_nf3=state.num_nf3,
            topk=topk,
            kept_mx=state.kept_mx,
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            runtime.grid188_disabled_reason = (
                "resources were not prepared before capture"
            )
            return
        try:
            prep_kept, prep_nf3 = state.prep_kept, state.prep_nf3
            if prep_kept is None or prep_nf3 is None:
                raise RuntimeError("both prepared tiers are required")
            prepared_contract = (
                prep_kept.weight_layout == "packed"
                and prep_kept.scale_format == "e4m3_k16"
                and int(prep_kept.num_experts) == _GRID188_NUM_KEPT
                and prep_nf3.weight_layout == "nf3_2p1"
                and prep_nf3.scale_format == "e4m3_k32"
                and int(prep_nf3.num_experts) == _GRID188_NUM_NF3
            )
            if not prepared_contract:
                raise RuntimeError("prepared tier layouts do not match Grid188 ABI")

            props = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            )
            if runtime.grid188_launch is None:
                from sparkinfer.moe._shared.kernels.w4a16.host import (
                    packed_gemm_scratch_elements,
                )
                from sparkinfer.moe._shared.kernels.w4a16.kernel import (
                    compile_w4a16_fused_moe_hybrid,
                )

                launch = compile_w4a16_fused_moe_hybrid(
                    size_m=_GRID188_M,
                    hidden_size=_GRID188_HIDDEN,
                    intermediate_size=_GRID188_INTERMEDIATE,
                    tier0_num_experts=_GRID188_NUM_KEPT,
                    tier1_num_experts=_GRID188_NUM_NF3,
                    top_k=_GRID188_TOPK,
                    activation="silu",
                    map_slots=_GRID188_NUM_KEPT + _GRID188_NUM_NF3,
                    element_dtype="bf16",
                    fast_math=True,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    force_tile_config=_B12X_TILES,
                    schedule_whole_tiles=True,
                )
                # Grid sizing is b12x launch policy now; admission here is
                # geometry plus the spill-free codegen contract (b12x already
                # refuses spilling kernels at compile, -1 = cache reload).
                if (
                    int(launch.size_m) != _GRID188_M
                    or int(launch.blocks_per_sm) != 1
                    or int(launch.shared_memory_bytes) != 45_184
                    or int(launch.map_slots) != _GRID188_NUM_KEPT + _GRID188_NUM_NF3
                    or int(launch.local_memory_bytes) > 0
                ):
                    raise RuntimeError("compiled hybrid launch failed admission")
                if not hasattr(
                    torch.ops.sparkinfer,
                    "w4a16_fused_moe_hybrid_launch",
                ):
                    raise RuntimeError("hybrid one-grid custom op is unavailable")
                runtime.grid188_scratch = self._borrow_hybrid_scratch(
                    runtime.buffers,
                    device=prep_kept.w13.device,
                    routed_rows=_GRID188_M * _GRID188_TOPK,
                    fc1_cols=2 * _GRID188_INTERMEDIATE,
                    intermediate_size=_GRID188_INTERMEDIATE,
                    scratch_elements=packed_gemm_scratch_elements(
                        size_n=max(2 * _GRID188_INTERMEDIATE, _GRID188_HIDDEN),
                        route_slots=_GRID188_M * _GRID188_TOPK,
                        moe_block_size=8,
                        sms=sms,
                    ),
                    workspace_words=sms * 4 + 2,
                )
                runtime.grid188_sms = sms
                runtime.grid188_max_shared_mem = max_shared_mem
                runtime.grid188_launch = launch

            weight_views = (
                *self._hybrid_prepared_views(prep_kept),
                *self._hybrid_prepared_views(prep_nf3),
            )
            tier_map = torch.tensor(
                _combined_tier_local_descriptors(state.remap),
                dtype=torch.int32,
                device=prep_kept.w13.device,
            ).contiguous()
            output = torch.empty(
                (_GRID188_M, _GRID188_HIDDEN),
                dtype=torch.bfloat16,
                device=prep_kept.w13.device,
            )
            # Publish only after every allocation and validation has succeeded.
            state.grid188_weight_views = weight_views
            state.grid188_tier_map = tier_map
            state.grid188_output = output
            state.grid188_ready = True
            logger.info_once(
                "nvfp4_nf3_hybrid: armed hybrid one-grid decode (m<=%d)",
                _GRID188_M,
            )
        except Exception as exc:
            runtime.grid188_disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "nvfp4_nf3_hybrid: Grid188 unavailable; using serial decode: %s",
                runtime.grid188_disabled_reason,
            )

    def _prepare_k3_hybrid(self, layer: "RoutedExperts", topk: int) -> None:
        """Arm K3 TP16's single-token MXFP4+NF3 one-grid decode path."""
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if state.k3_hybrid_ready or runtime.k3_hybrid_disabled_reason is not None:
            return
        if not _is_k3_hybrid_geometry(
            hidden_size=state.hidden_size,
            intermediate_size=state.intermediate_size,
            num_experts=state.num_experts,
            num_kept=state.num_kept,
            num_nf3=state.num_nf3,
            topk=topk,
            kept_mx=state.kept_mx,
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            runtime.k3_hybrid_disabled_reason = (
                "resources were not prepared before capture"
            )
            return
        try:
            prep_kept, prep_nf3 = state.prep_kept_hybrid, state.prep_nf3
            if prep_kept is None or prep_nf3 is None:
                raise RuntimeError("both prepared K3 tiers are required")
            prepared_contract = (
                prep_kept.weight_layout == "modelopt"
                and prep_kept.scale_format == "e8m0_k32"
                and prep_kept.w13_layout == "w31"
                and int(prep_kept.num_experts) == state.num_kept
                and prep_nf3.weight_layout == "nf3_2p1"
                and prep_nf3.scale_format == "e4m3_k32"
                and int(prep_nf3.num_experts) == state.num_nf3
            )
            if not prepared_contract:
                raise RuntimeError("prepared tier layouts do not match K3 hybrid ABI")

            from sparkinfer.moe._shared.kernels.w4a16.host import (
                packed_gemm_scratch_elements,
            )
            from sparkinfer.moe._shared.kernels.w4a16.kernel import (
                compile_w4a16_fused_moe_hybrid,
            )

            props = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            )
            launch = compile_w4a16_fused_moe_hybrid(
                size_m=_K3_HYBRID_M,
                hidden_size=_K3_HYBRID_HIDDEN,
                intermediate_size=_K3_HYBRID_INTERMEDIATE,
                tier0_num_experts=state.num_kept,
                tier1_num_experts=state.num_nf3,
                top_k=_K3_HYBRID_TOPK,
                activation="situ",
                map_slots=_K3_HYBRID_EXPERTS,
                element_dtype="bf16",
                fast_math=True,
                sms=sms,
                max_shared_mem=max_shared_mem,
                tier0_weight_layout=prep_kept.weight_layout,
                tier0_scale_format=prep_kept.scale_format,
                tier0_w13_layout=prep_kept.w13_layout,
                tier1_weight_layout=prep_nf3.weight_layout,
                tier1_scale_format=prep_nf3.scale_format,
                tier1_w13_layout=prep_nf3.w13_layout,
                force_tile_config=state.tiles,
                schedule_whole_tiles=True,
            )
            if (
                int(launch.size_m) != _K3_HYBRID_M
                or int(launch.blocks_per_sm) != 1
                or int(launch.map_slots) != _K3_HYBRID_EXPERTS
                or int(launch.local_memory_bytes) > 0
            ):
                raise RuntimeError("compiled K3 hybrid launch failed admission")
            if not hasattr(torch.ops.sparkinfer, "w4a16_fused_moe_hybrid_launch"):
                raise RuntimeError("hybrid one-grid custom op is unavailable")
            if runtime.k3_hybrid_scratch is None:
                runtime.k3_hybrid_scratch = self._borrow_hybrid_scratch(
                    runtime.buffers,
                    device=prep_kept.w13.device,
                    routed_rows=_K3_HYBRID_M * _K3_HYBRID_TOPK,
                    fc1_cols=2 * _K3_HYBRID_INTERMEDIATE,
                    intermediate_size=_K3_HYBRID_INTERMEDIATE,
                    scratch_elements=packed_gemm_scratch_elements(
                        size_n=max(2 * _K3_HYBRID_INTERMEDIATE, _K3_HYBRID_HIDDEN),
                        route_slots=_K3_HYBRID_M * _K3_HYBRID_TOPK,
                        moe_block_size=8,
                        sms=sms,
                    ),
                    workspace_words=sms * 4 + 2,
                )
                runtime.k3_hybrid_sms = sms
                runtime.k3_hybrid_max_shared_mem = max_shared_mem

            state.k3_hybrid_launch = launch
            state.k3_hybrid_weight_views = (
                *self._hybrid_prepared_views(prep_kept),
                *self._hybrid_prepared_views(prep_nf3),
            )
            state.k3_hybrid_tier_map = torch.tensor(
                _combined_tier_local_descriptors(
                    state.remap,
                    num_experts=state.num_experts,
                    num_kept=state.num_kept,
                    num_nf3=state.num_nf3,
                ),
                dtype=torch.int32,
                device=prep_kept.w13.device,
            ).contiguous()
            # The output escapes this layer and is consumed by later residual
            # operations. Give each layer distinct storage so full-decode CUDA
            # graph capture cannot observe all mixed layers as aliases of the
            # same external buffer. This costs only 7 KiB per mixed K3 layer.
            state.k3_hybrid_output = torch.empty(
                (_K3_HYBRID_M, _K3_HYBRID_HIDDEN),
                dtype=torch.bfloat16,
                device=prep_kept.w13.device,
            )
            state.k3_hybrid_ready = True
            logger.info_once(
                "nvfp4_nf3_hybrid: armed K3 TP16 one-grid decode (MXFP4=%d, NF3=%d)",
                state.num_kept,
                state.num_nf3,
            )
        except Exception as exc:
            runtime.k3_hybrid_disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "nvfp4_nf3_hybrid: K3 one-grid unavailable; using serial decode: %s",
                runtime.k3_hybrid_disabled_reason,
            )

    def _run_grid188(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        launch = runtime.grid188_launch
        scratch = runtime.grid188_scratch
        assert launch is not None and scratch is not None
        assert runtime.grid188_sms is not None
        assert runtime.grid188_max_shared_mem is not None
        assert state.grid188_weight_views is not None
        assert state.grid188_tier_map is not None
        assert state.grid188_output is not None
        m = int(x.shape[0])
        torch.ops.sparkinfer.w4a16_fused_moe_hybrid_launch(
            x,
            *state.grid188_weight_views,
            topk_ids.view(-1),
            state.grid188_tier_map,
            scratch["fc1"],
            scratch["activated"],
            state.grid188_output.view(-1),
            topk_weights,
            scratch["fc1_c_tmp"],
            scratch["fc2_c_tmp"],
            scratch["workspace"],
            m,
            int(launch.size_m),
            int(launch.hidden_size),
            int(launch.intermediate_size),
            int(launch.tier0_num_experts),
            int(launch.tier1_num_experts),
            int(launch.top_k),
            launch.activation,
            int(launch.map_slots),
            int(launch.moe_block_size),
            launch.element_dtype,
            bool(launch.fast_math),
            runtime.grid188_sms,
            runtime.grid188_max_shared_mem,
            launch.tier0_weight_layout,
            launch.tier0_scale_format,
            launch.tier0_w13_layout,
            launch.tier1_weight_layout,
            launch.tier1_scale_format,
            launch.tier1_w13_layout,
            int(launch.fc1_tile_k),
            int(launch.fc1_tile_n),
            int(launch.fc2_tile_k),
            int(launch.fc2_tile_n),
            bool(launch.schedule_whole_tiles),
            int(torch.cuda.current_stream(x.device).cuda_stream),
        )
        return state.grid188_output[:m]

    def _run_k3_hybrid(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        launch = state.k3_hybrid_launch
        scratch = runtime.k3_hybrid_scratch
        assert launch is not None and scratch is not None
        assert runtime.k3_hybrid_sms is not None
        assert runtime.k3_hybrid_max_shared_mem is not None
        assert state.k3_hybrid_weight_views is not None
        assert state.k3_hybrid_tier_map is not None
        assert state.k3_hybrid_output is not None
        torch.ops.sparkinfer.w4a16_fused_moe_hybrid_launch(
            x,
            *state.k3_hybrid_weight_views,
            topk_ids.view(-1),
            state.k3_hybrid_tier_map,
            scratch["fc1"],
            scratch["activated"],
            state.k3_hybrid_output.view(-1),
            topk_weights,
            scratch["fc1_c_tmp"],
            scratch["fc2_c_tmp"],
            scratch["workspace"],
            int(x.shape[0]),
            int(launch.size_m),
            int(launch.hidden_size),
            int(launch.intermediate_size),
            int(launch.tier0_num_experts),
            int(launch.tier1_num_experts),
            int(launch.top_k),
            launch.activation,
            int(launch.map_slots),
            int(launch.moe_block_size),
            launch.element_dtype,
            bool(launch.fast_math),
            runtime.k3_hybrid_sms,
            runtime.k3_hybrid_max_shared_mem,
            launch.tier0_weight_layout,
            launch.tier0_scale_format,
            launch.tier0_w13_layout,
            launch.tier1_weight_layout,
            launch.tier1_scale_format,
            launch.tier1_w13_layout,
            int(launch.fc1_tile_k),
            int(launch.fc1_tile_n),
            int(launch.fc2_tile_k),
            int(launch.fc2_tile_n),
            bool(launch.schedule_whole_tiles),
            int(torch.cuda.current_stream(x.device).cuda_stream),
        )
        return state.k3_hybrid_output

    def _can_use_trellis_w4a8(self, state: _HybridLayerState, *, topk: int) -> bool:
        """Identify the exact TP12 E4M3 trellis contract implemented by B12X."""

        prepared = (
            state.trellis_weights.representation.value
            if getattr(state, "trellis_weights", None) is not None
            else None
        )

        return bool(
            _qsrt_w4a8_requested()
            and not os.getenv("VLLM_KQUANT_CAPTURE_DIR")
            and state.uses_mixed_tp12_slab
            and self.quant_config.trellis_codebook
            in {
                "mul1-e4m3",
                "sqg-normal-e4m3",
                "sqg-cheb-normal-e4m3",
                "sqg-cheb-normal-k2-q8h4-w2-e4m3",
            }
            and get_tensor_model_parallel_world_size() == 12
            and state.hidden_size == 3584
            and state.intermediate_size == 256
            and int(topk) == 16
            and self.moe.activation.value == "situ"
            and prepared is not None
            and int(prepared.gate_suh.shape[0]) == 1
            and int(prepared.up_suh.shape[0]) == 1
        )

    def _run_trellis_w4a8(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the compact compressed tier through native E4M3 W4A8 MMA."""

        from sparkinfer.moe._shared.kernels.trellis_w4a8 import (
            run_trellis_w4a8_moe,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        m = int(x.shape[0])
        scratch = runtime.trellis_w4a8_scratch.get(m)
        if scratch is None:
            raise RuntimeError(
                f"native QSRT W4A8 scratch for m={m} was not preallocated"
            )
        prepared = state.trellis_weights.representation.value
        return run_trellis_w4a8_moe(
            x if x.is_contiguous() else x.contiguous(),
            prepared,
            topk_weights,
            topk_ids,
            scratch,
            expert_map=state.emap_nf3,
            fast_math=True,
        )

    def _ensure_runtime(self, layer: "RoutedExperts", m: int, topk: int) -> None:
        """First-apply init: per-tier preplanned launches plus ONE shared
        scratch/buffer set. The first apply is vLLM's eager profile run at
        max_num_batched_tokens, so max_m sizes itself to the serving
        ceiling and nothing compiles during CUDA-graph capture."""
        from sparkinfer.moe._shared.kernels.w4a16.host import (
            make_w4a16_packed_buffers,
            max_packed_route_slots,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if runtime.max_m is None:
            runtime.max_m = max(int(self.moe.max_num_tokens), int(m))
            runtime.topk = int(topk)
        if int(topk) != runtime.topk:
            raise RuntimeError(
                f"nvfp4_nf3_hybrid: topk changed {runtime.topk} -> {topk}"
            )
        if state.prep_kept is not None:
            state.launch_kept = self._get_launch_pair(state.prep_kept, state)
        if getattr(state, "trellis_weights", None) is not None:
            from sparkinfer.moe import fused_moe

            key = (
                "trellis",
                state.num_nf3,
                state.hidden_size,
                state.intermediate_size,
                runtime.topk,
                runtime.max_m,
            )
            plan = runtime.launches.get(key)
            if plan is None:
                caps = fused_moe.Caps(
                    max_tokens=runtime.max_m,
                    num_topk=runtime.topk,
                    device=torch.accelerator.current_device_index(),
                    weight_plan=state.trellis_weights.plan,
                    quant_mode="w4a16",
                    route_num_experts=self.moe.num_experts,
                    # Full-rotation Trellis owns one immutable route geometry
                    # for prewarm, eager execution, and CUDA-graph replay.
                    # This is the TP12 geometry validated by the B12X closure
                    # and pair-container benchmark paths.
                    w4a16_block_size_m=8,
                )
                plan = fused_moe.plan(caps)
                runtime.launches[key] = plan
            state.trellis_plan = plan
            spec = plan.scratch_specs()[0]
            need = int(torch.Size(spec.shape).numel())
            trellis_scratch = runtime.trellis_scratch
            if trellis_scratch is None or (
                trellis_scratch.numel() < need or trellis_scratch.dtype != spec.dtype
            ):
                runtime.trellis_scratch = torch.empty(
                    spec.shape,
                    dtype=spec.dtype,
                    device=torch.accelerator.current_device_index(),
                )
            if (
                os.getenv("VLLM_KQUANT_CAPTURE_DIR")
                and runtime.kquant_logical_mid is None
            ):
                runtime.kquant_logical_mid = torch.empty(
                    (runtime.max_m * runtime.topk, state.intermediate_size),
                    dtype=torch.float16,
                    device=torch.accelerator.current_device_index(),
                )
            if self._can_use_trellis_w4a8(state, topk=topk):
                from sparkinfer.moe._shared.kernels.trellis_w4a8 import (
                    make_trellis_w4a8_moe_scratch,
                )

                prepared = state.trellis_weights.representation.value
                device = prepared.w13.device
                for decode_m in range(1, _TRELLIS_W4A8_DECODE_M + 1):
                    if decode_m not in runtime.trellis_w4a8_scratch:
                        runtime.trellis_w4a8_scratch[decode_m] = (
                            make_trellis_w4a8_moe_scratch(
                                m=decode_m,
                                topk=topk,
                                hidden_size=state.hidden_size,
                                intermediate_size=state.intermediate_size,
                                device=device,
                            )
                        )
                state.trellis_w4a8_ready = True
        if state.prep_nf3 is not None:
            state.launch_nf3 = self._get_launch_pair(state.prep_nf3, state)
        if runtime.buffers is None:
            prep_any = state.prep_kept or state.prep_nf3
            if prep_any is None:
                # MXFP4-kept layer with no NF3 tier: the kept modular kernel
                # manages its own workspace, no shared buffers needed yet.
                state.runtime_ready = True
                return
            device = prep_any.w13.device
            buffers = make_w4a16_packed_buffers(
                prep_any,
                m=runtime.max_m,
                topk=runtime.topk,
                dtype=torch.bfloat16,
                device=device,
                route_num_experts=self.moe.num_experts,
            )
            # The preplanned prefill launch validates route capacity at
            # moe_block_size=64; the plan's own block choice can be smaller
            # for small max_m, so upsize the route buffers if needed.
            need_slots = max_packed_route_slots(
                runtime.max_m * runtime.topk, 64, self.moe.num_experts
            )
            need_blocks = (need_slots + 63) // 64
            if (
                buffers.packed_route_indices.numel() < need_slots
                or buffers.block_expert_ids.numel() < need_blocks
            ):
                buffers = dataclasses.replace(
                    buffers,
                    packed_route_indices=torch.empty(
                        (need_slots,), dtype=torch.int32, device=device
                    ),
                    block_expert_ids=torch.empty(
                        (need_blocks,), dtype=torch.int32, device=device
                    ),
                )
            runtime.buffers = buffers
            # Per-tier outputs; fully overwritten by every launch that uses
            # them, so sharing them across layers is safe. MXFP4 kept experts
            # run through their modular kernel and return a separate output;
            # in that case (and for an all-NF3 layer) the packed buffer's own
            # output can serve NF3 directly. Avoiding a redundant
            # [max_m, hidden] BF16 tensor saves 7 MiB for K3 at max_m=1024,
            # which is material when the 1M-token cache fits by only a few MiB.
            runtime.out_kept = buffers.output
            runtime.out_nf3 = (
                buffers.output
                if state.prep_kept is None
                else torch.empty_like(buffers.output)
            )
        self._prepare_grid188(layer, topk)
        self._prepare_k3_hybrid(layer, topk)
        state.runtime_ready = True

    def _run_tier(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        prepared: Any,
        launch_pair: tuple[Any, Any],
        expert_map: torch.Tensor,
        output: torch.Tensor,
        decode: bool,
    ) -> torch.Tensor:
        """Run one tier through its preplanned b12x launch."""
        from sparkinfer.moe._shared.kernels.w4a16.kernel import run_w4a16_moe

        runtime = self.quant_config.shared_runtime
        use_decode = decode and launch_pair[0] is not launch_pair[1]
        launch = launch_pair[0] if use_decode else launch_pair[1]
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if use_decode:
            # Direct top-k path: the kernel reads flat LOCAL ids.  Unlike the
            # packed route builder, SparkInfer's direct launcher cannot safely
            # consume an all-negative tier (and some versions dereference
            # negative routes before applying the router weight).  Replace
            # inactive routes with expert zero and give them an exact-zero
            # weight.  This remains graph-safe and avoids a host-side
            # ``any().item()`` synchronization on every decode token.
            ids = expert_map[ids.long()].to(torch.int32).contiguous()
            active = ids >= 0
            topk_weights = topk_weights.masked_fill(~active, 0.0).contiguous()
            ids.clamp_min_(0)
            launch_expert_map = None
            # Keep the explicit clear as a backstop for launch variants which
            # do not overwrite an output row whose router weights are all zero.
            output.zero_()
        else:
            # Packed path: the kernel translates global -> local and drops
            # the -1 entries of the other tier.
            launch_expert_map = expert_map
        buffers = runtime.buffers
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            ids,
            activation=self.moe.activation.value,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_map=launch_expert_map,
            fused_launch=launch,
        )

    def _run_kept(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        decode: bool,
    ) -> torch.Tensor:
        """Kept tier: NVFP4 through the preplanned launcher, MXFP4 through
        the production modular kernel with in-kernel global route mapping."""
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if state.prep_kept is not None:
            m = x.shape[0]
            assert state.launch_kept is not None
            assert state.emap_kept is not None
            assert runtime.out_kept is not None
            return self._run_tier(
                x,
                topk_weights,
                topk_ids,
                state.prep_kept,
                state.launch_kept,
                state.emap_kept,
                runtime.out_kept[:m],
                decode,
            )
        result_m = int(x.shape[0])
        if (
            decode
            and state.kept_mx
            and result_m != 1
            and not os.getenv("VLLM_KQUANT_CAPTURE_DIR")
        ):
            # The native MXFP4 microkernel is valuable for single-sequence
            # decode, where M is exactly one.  Chunked-prefill tails can also
            # land in the nominal decode range (M=2..8); specializing the
            # microkernel for every tail M and every per-layer expert count
            # creates thousands of surprise JIT compiles.  Keep those tails on
            # the numerically-safe packed route while NF3 retains its direct
            # launch at the original M.
            packed_m = _B12X_DECODE_M + 1
            if runtime.max_m is None:
                raise RuntimeError("hybrid runtime was not initialized")
            if runtime.max_m < packed_m:
                raise RuntimeError(
                    "nvfp4_nf3_hybrid requires max_num_batched_tokens >= "
                    f"{packed_m} for safe hybrid prefill tails"
                )
            pad_m = packed_m - result_m
            x = torch.cat((x, x.new_zeros((pad_m, x.shape[1]))), dim=0)
            topk_weights = torch.cat(
                (
                    topk_weights,
                    topk_weights.new_zeros((pad_m, topk_weights.shape[1])),
                ),
                dim=0,
            )
            topk_ids = torch.cat(
                (
                    topk_ids,
                    topk_ids.new_zeros((pad_m, topk_ids.shape[1])),
                ),
                dim=0,
            )
        kept_module = state.kept_module
        if kept_module is None or state.kept_kernel is None:
            raise RuntimeError("MXFP4 kept tier was not prepared")
        return state.kept_kernel.apply(
            x,
            kept_module.w13_weight,
            kept_module.w2_weight,
            topk_weights,
            topk_ids,
            activation=kept_module.activation,
            global_num_experts=state.num_experts,
            expert_map=state.kept_remap,
            apply_router_weight_on_input=False,
            shared_experts=None,
            shared_experts_input=None,
        )[:result_m]

    def _apply_once(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # Routing runs upstream and shared experts are executed by the MoE
        # runner; this method returns the routed-experts output only.
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        m = int(x.shape[0])
        if not state.runtime_ready:
            self._ensure_runtime(layer, m, int(topk_ids.shape[1]))
        if runtime.max_m is None:
            raise RuntimeError("hybrid runtime was not initialized")
        if m > runtime.max_m:
            raise RuntimeError(
                f"nvfp4_nf3_hybrid: m={m} exceeds the planned launch "
                f"capacity {runtime.max_m} (max_num_batched_tokens)."
            )
        decode = m <= _B12X_DECODE_M
        weights = (
            topk_weights
            if topk_weights.dtype == torch.float32
            else topk_weights.float()
        )
        if not weights.is_contiguous():
            weights = weights.contiguous()
        if (
            state.grid188_ready
            and 1 <= m <= _GRID188_M
            and not os.getenv("VLLM_KQUANT_CAPTURE_DIR")
        ):
            # The unified hybrid launch takes dynamic m up to the compiled
            # capacity, so every small decode bucket (not just the exact MTP
            # batch) rides the one-grid path.
            grid_ids = (
                topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            )
            if not grid_ids.is_contiguous():
                grid_ids = grid_ids.contiguous()
            if (
                x.dtype == torch.bfloat16
                and x.is_contiguous()
                and grid_ids.numel() == m * _GRID188_TOPK
                and grid_ids.is_cuda
                and grid_ids.device == x.device
                and grid_ids.data_ptr() % 16 == 0
                and weights.numel() == m * _GRID188_TOPK
            ):
                logger.info_once("nvfp4_nf3_hybrid: executing hybrid one-grid decode")
                return self._run_grid188(layer, x, weights, grid_ids)
        if (
            state.k3_hybrid_ready
            and m == _K3_HYBRID_M
            and not os.getenv("VLLM_KQUANT_CAPTURE_DIR")
        ):
            hybrid_ids = (
                topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            )
            if not hybrid_ids.is_contiguous():
                hybrid_ids = hybrid_ids.contiguous()
            if (
                x.dtype == torch.bfloat16
                and x.is_contiguous()
                and hybrid_ids.numel() == _K3_HYBRID_TOPK
                and hybrid_ids.is_cuda
                and hybrid_ids.device == x.device
                and hybrid_ids.data_ptr() % 16 == 0
                and weights.numel() == _K3_HYBRID_TOPK
            ):
                logger.info_once("nvfp4_nf3_hybrid: executing K3 TP16 one-grid decode")
                return self._run_k3_hybrid(layer, x, weights, hybrid_ids)
        if state.num_nf3 == 0:
            # Uniform kept layer (including all-MXFP4 decoder layers and an
            # unmapped NVFP4 MTP head): single-tier launch.
            return self._run_kept(layer, x, weights, topk_ids, decode)
        if getattr(state, "trellis_weights", None) is not None:
            from sparkinfer.moe import fused_moe

            tids = (
                topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            )
            if not tids.is_contiguous():
                tids = tids.contiguous()
            if state.trellis_w4a8_ready:
                # The first call for every layer happens in vLLM's eager
                # profile pass.  Exercise M=1 there so no CuTe resolution or
                # compilation can occur inside a later CUDA-graph capture.
                if not state.trellis_w4a8_prewarmed:
                    self._run_trellis_w4a8(
                        layer,
                        x[:1],
                        weights[:1],
                        tids[:1],
                    )
                    state.trellis_w4a8_prewarmed = True
                if m <= _TRELLIS_W4A8_DECODE_M:
                    logger.info_once(
                        "nvfp4_nf3_hybrid: executing TP12 E4M3 trellis W4A8 decode"
                    )
                    out_trellis = self._run_trellis_w4a8(
                        layer,
                        x,
                        weights,
                        tids,
                    ).to(x.dtype)
                    if state.num_kept == 0:
                        return out_trellis
                    return (
                        self._run_kept(layer, x, weights, topk_ids, decode)[:m]
                        + out_trellis
                    )
            binding = fused_moe.bind(
                state.trellis_plan,
                scratch=runtime.trellis_scratch,
                a=x if x.is_contiguous() else x.contiguous(),
                experts=state.trellis_weights,
                topk_weights=weights,
                topk_ids=tids,
                route_expert_map=state.emap_nf3,
            )
            # The unified full-rotation top-k sum emits fp32; downstream
            # layers expect the model dtype.
            out_trellis = fused_moe.run(binding=binding)[:m].to(x.dtype)
            if os.getenv("VLLM_KQUANT_CAPTURE_DIR"):
                from vllm.model_executor.layers.fused_moe.kquant_capture import (
                    collect_kquant_exl3_mid,
                )

                prepared = state.trellis_weights.representation.value
                intermediate_rotations = prepared.intermediate_rotations
                logical_scratch = runtime.kquant_logical_mid
                if intermediate_rotations is None or logical_scratch is None:
                    raise RuntimeError(
                        "EXL3 KQuant capture resources were not prepared eagerly"
                    )
                collect_kquant_exl3_mid(
                    prefix=str(layer.layer_name),
                    binding=binding,
                    topk_weights=weights,
                    topk_ids=tids,
                    expert_map=state.emap_nf3,
                    intermediate_rotations=intermediate_rotations,
                    logical_scratch=logical_scratch,
                )
            if state.num_kept == 0:
                return out_trellis
            return self._run_kept(layer, x, weights, topk_ids, decode)[:m] + out_trellis
        if state.num_kept == 0:
            assert state.prep_nf3 is not None
            assert state.launch_nf3 is not None
            assert state.emap_nf3 is not None
            assert runtime.out_nf3 is not None
            return self._run_tier(
                x,
                weights,
                topk_ids,
                state.prep_nf3,
                state.launch_nf3,
                state.emap_nf3,
                runtime.out_nf3[:m],
                decode,
            )
        out_kept = self._run_kept(layer, x, weights, topk_ids, decode)
        assert state.prep_nf3 is not None
        assert state.launch_nf3 is not None
        assert state.emap_nf3 is not None
        assert runtime.out_nf3 is not None
        out_nf3 = self._run_tier(
            x,
            weights,
            topk_ids,
            state.prep_nf3,
            state.launch_nf3,
            state.emap_nf3,
            runtime.out_nf3[:m],
            decode,
        )
        return out_kept + out_nf3

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the hybrid QSRT/X4T expert path.

        The optional repeat check deliberately wraps the complete hybrid
        dispatch rather than the ordinary modular-MoE adapter: QSRT invokes
        the B12X prepared launches directly, so a check in ``B12xExperts``
        cannot observe this path.  It is post-start and eager-only, and thus
        has no serving cost unless explicitly enabled for a runtime audit.
        """
        global _qsrt_repeat_check_reports

        output = self._apply_once(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )
        repeat_enabled = (
            os.getenv("B12X_MOE_REPEAT_CHECK", "0") == "1"
            or os.getenv("VLLM_B12X_MOE_REPEAT_CHECK", "0") == "1"
        )
        after_start = (
            os.getenv("B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START", "0") == "1"
            or os.getenv("VLLM_B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START", "0") == "1"
        )
        engine_started = os.getenv("B12X_VLLM_ENGINE_STARTED", "0") == "1"
        try:
            max_reports = int(os.getenv("B12X_MOE_REPEAT_CHECK_MAX_REPORTS", "8"))
        except ValueError:
            max_reports = 8
        is_capturing = bool(
            torch.accelerator.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        if (
            not repeat_enabled
            or (after_start and not engine_started)
            or is_capturing
            or _qsrt_repeat_check_reports >= max_reports
        ):
            return output

        # The hybrid implementation reuses shared output buffers.  Preserve
        # the first result before the second launch overwrites those buffers.
        original = output.clone()
        repeated = self._apply_once(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )
        original_f = original.float()
        repeated_f = repeated.float()
        diff = (original_f - repeated_f).abs()
        finite = bool(
            torch.isfinite(original_f).all().item()
            and torch.isfinite(repeated_f).all().item()
        )
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        denom = original_f.flatten().norm() * repeated_f.flatten().norm()
        cosine = (
            float((original_f.flatten().dot(repeated_f.flatten()) / denom).item())
            if float(denom.item()) != 0.0
            else 1.0
        )
        _qsrt_repeat_check_reports += 1
        logger.warning(
            "B12X MoE repeat check: finite=%s max_abs=%g mean_abs=%g "
            "cosine=%g shape=%s dtype=%s quant_mode=w4a16 "
            "implementation=w4a16",
            finite,
            max_abs,
            mean_abs,
            cosine,
            tuple(original.shape),
            original.dtype,
        )
        return repeated


NvFp4Nf3HybridConfig.FusedMoEMethodCls = NvFp4Nf3HybridMoEMethod
