# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KQuant mixed-expert serving for generic EXL3 and canonical QSRT.

The high-quality expert tier is stored as NVFP4 or MXFP4.  The secondary tier
is either generic ``exl3_3`` trellis tensors or TP-independent
``qsrt_sqg_e4m3`` atoms.  Generic EXL3 executes through B12X W4A16; the
uniform coupled-K2 atoms-v2 profile can optionally use B12X W4A8-MX for
prefill while retaining W4A16 for the small decode band.
"""

import dataclasses
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch

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

# Pinned CTA tiles (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n).
_B12X_TILES = (64, 256, 64, 256)
# Batches of at most this many tokens take the preplanned TC-decode launch.
_B12X_DECODE_M = 8
# QSRT fuses checkpoint-order [w1; w3] rows,
# which are [gate; up] for SiTU; B12X names that physical order ``w31``.
_QSRT_X4T_W13_LAYOUT = "w31"
_QSRT_X4T_W13_EXCEPTION_TASK_ROWS = 128
_QSRT_X4T_W2_EXCEPTION_TASK_ROWS = 896
_QSRT_X4T_W13_EXCEPTION_ROW_ROTATION = 0
_QSRT_ATOMS_V2_PROFILE_H308 = "k3x22_k4x2"
_QSRT_ATOMS_V2_PROFILE_COUPLED_K2 = "k2_coupled_h512_h128"
_QSRT_ATOMS_V2_PROFILE_COUPLED_H308 = "k3x22_k4x2_coupled_h512_h128"
_UNSERIALIZED_MULTIMODAL_DENSE_COMPONENTS = (
    "vision_tower",
    "mm_projector",
)


def _qsrt_atoms_v2_w4a8_prefill_enabled(*, pure_k2: bool) -> bool:
    """Enable W4A8 prefill only for the one profile it supports safely."""
    requested = os.getenv("VLLM_KQUANT_QSRT_W4A8_PREFILL", "0").strip().lower()
    enabled = requested in {"1", "true", "yes", "on"}
    if enabled and not pure_k2:
        raise ValueError(
            "VLLM_KQUANT_QSRT_W4A8_PREFILL requires the uniform coupled pure-K2 "
            f"profile {_QSRT_ATOMS_V2_PROFILE_COUPLED_K2!r}"
        )
    return enabled


def _qsrt_atoms_v2_w4a8_prequantized_input_enabled(
    *, prefill_enabled: bool
) -> bool:
    """Use an external MXFP8 expert input only with W4A8 prefill."""
    requested = os.getenv(
        "VLLM_KQUANT_QSRT_W4A8_PREQUANTIZED_INPUT", "0"
    ).strip().lower()
    enabled = requested in {"1", "true", "yes", "on"}
    if enabled and not prefill_enabled:
        raise ValueError(
            "VLLM_KQUANT_QSRT_W4A8_PREQUANTIZED_INPUT requires W4A8 prefill"
        )
    return enabled
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
            "kquant_hybrid kept kernel must return an unreduced rank-local partial"
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


def _b12x_tiles_for_geometry(
    hidden_size: int, intermediate_size: int
) -> tuple[int, int, int, int]:
    """Select one fixed B12X tile pair that exactly divides both GEMMs."""
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
        "kquant_hybrid has no fixed b12x tile configuration for "
        f"hidden={hidden_size}, intermediate={intermediate_size}"
    )


class _HybridSharedRuntime:
    """Process-wide B12X runtime shared by every hybrid MoE layer.

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
        # Capture-only route-major canonical pre-w2 scratch. EXL3 cache2 is
        # H128(h * down_suh); this stable buffer receives its inverse transform.
        self.kquant_logical_mid: torch.Tensor | None = None
        self.trellis_scratch: torch.Tensor | None = None
        self.trellis_prefill_scratch: torch.Tensor | None = None
        self.trellis_prefill_fallback_scratch: torch.Tensor | None = None
        self.trellis_output: torch.Tensor | None = None
        self.trellis_prefill_input: torch.Tensor | None = None
        self.trellis_prefill_input_values: torch.Tensor | None = None
        self.trellis_prefill_input_scales: torch.Tensor | None = None
        self.trellis_prefill_output: torch.Tensor | None = None
        self.trellis_prefill_topk_ids: torch.Tensor | None = None
        self.trellis_prefill_topk_weights: torch.Tensor | None = None
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
        # global expert id -> (tier, local index); tier 0 = kept, 1 = secondary.
        self.remap = remap
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.kept_mx = kept_mx
        self.num_kept = sum(1 for tier, _ in remap.values() if tier == 0)
        self.num_secondary = sum(1 for tier, _ in remap.values() if tier == 1)
        self.tiles = _b12x_tiles_for_geometry(hidden_size, intermediate_size)
        # B12X prepared kept weights.
        self.prep_kept: Any = None
        # Native MXFP4 W4A16 representation already owned by kept_kernel.
        # This is a view/metadata bundle, not a second resident weight copy.
        self.prep_kept_hybrid: Any = None
        # Global -> local id maps, -1 for experts outside the tier.
        self.emap_kept: torch.Tensor | None = None
        self.emap_secondary: torch.Tensor | None = None
        # (decode_launch, prefill_launch) for the kept tier, set at first apply.
        self.launch_kept: tuple[Any, Any] | None = None
        # MXFP4 kept tier: modular kernel + its weight-holder module and a
        # global -> local map; -1 is the inactive-route sentinel.
        self.kept_kernel: Any = None
        self.kept_module: torch.nn.Module | None = None
        self.kept_remap: torch.Tensor | None = None
        # Keeps kernel-format tensors alive: b12x prepared weights VIEW the
        # converted tensors, so dropping them would dangle the views.
        self.keepalive: Any = None
        # Canonical QSRT layers own compressed experts through a TP-independent
        # atom safetensors file rather than ordinary expert parameters.
        self.uses_qsrt_atoms = False
        self.trellis_weights: Any = None
        self.trellis_plan: Any = None
        self.trellis_prefill_weights: Any = None
        self.trellis_prefill_plan: Any = None
        self.trellis_prefill_fallback_plan: Any = None
        self.trellis_use_w4a8_prefill = False
        self.trellis_use_w4a8_prequantized_input = False
        self.runtime_ready = False


class KQuantHybridConfig(ModelOptNvFp4Config):
    """Config for mixed NVFP4/MXFP4 plus EXL3 or QSRT checkpoints.

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
        # How secondary experts are stored and executed. ``exl3_3`` is the
        # generic EXL3 tensor path; ``qsrt_sqg_e4m3`` is the canonical
        # TP-independent QSRT atom container and can carry fixed K2 or K3.
        self.demoted_format: str = "exl3_3"
        self.qsrt: dict[str, Any] | None = None
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
        return "kquant_hybrid"

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
            # attention/shared-expert linears); without an overlay spec this
            # falls through to plain BF16.
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
        if user_quant is not None and user_quant != "kquant_hybrid":
            # Respect an explicit --quantization choice.
            return None
        hybrid_bit_map, _ = _read_hybrid_keys(hf_quant_cfg)
        quantization = (
            hf_quant_cfg.get("quantization") if isinstance(hf_quant_cfg, dict) else None
        )
        demoted_format = (
            hf_quant_cfg.get("demoted_format")
            if isinstance(hf_quant_cfg, dict)
            else None
        )
        if demoted_format is None and isinstance(quantization, dict):
            demoted_format = quantization.get("demoted_format")
        if hybrid_bit_map and demoted_format == "qsrt_sqg_e4m3":
            return "kquant_hybrid"
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
    ) -> "KQuantHybridConfig":
        hybrid_bit_map, kept_format = _read_hybrid_keys(original_config)
        if not isinstance(hybrid_bit_map, dict) or not hybrid_bit_map:
            raise ValueError(
                "kquant_hybrid requires a non-empty 'hybrid_bit_map' dict "
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
        assert isinstance(config, KQuantHybridConfig)
        config.hybrid_bit_map = hybrid_bit_map
        config.kept_format = kept_format
        quantization = original_config.get("quantization")
        demoted_format = original_config.get("demoted_format")
        if demoted_format is None and isinstance(quantization, dict):
            demoted_format = quantization.get("demoted_format")
        if demoted_format is not None:
            if demoted_format not in ("exl3_3", "qsrt_sqg_e4m3"):
                raise ValueError(f"unsupported demoted_format {demoted_format!r}")
            config.demoted_format = demoted_format
        qsrt = original_config.get("qsrt")
        if qsrt is None and isinstance(quantization, dict):
            qsrt = quantization.get("qsrt")
        if demoted_format == "qsrt_sqg_e4m3":
            if not isinstance(qsrt, dict):
                raise ValueError(
                    "qsrt_sqg_e4m3 demotion requires a qsrt format descriptor"
                )
            expected_qsrt = {
                "encoding": "qsrt_sqg_e4m3",
                "codebook": "sqg_xor_cheb_t12",
                "artifact_manifest": "qsrt-manifest.json",
            }
            for name, expected in expected_qsrt.items():
                if qsrt.get(name) != expected:
                    raise ValueError(
                        f"QSRT {name} must be {expected!r}, got {qsrt.get(name)!r}"
                    )
            storage_format = qsrt.get("storage_format")
            storage_schemas = {
                "qsrt_atoms_v1": {"kquant_kimi_k3_qsrt_atoms_v1"},
                "qsrt_atoms_v2": {
                    "kquant_kimi_k3_qsrt_atoms_v2",
                    "qsrt_kimi_k3_qsrt_atoms_v2",
                },
            }
            if storage_format not in storage_schemas:
                raise ValueError(
                    "QSRT storage_format must be 'qsrt_atoms_v1' or "
                    f"'qsrt_atoms_v2', got {storage_format!r}"
                )
            expected_schemas = storage_schemas[storage_format]
            if qsrt.get("schema") not in expected_schemas:
                raise ValueError(
                    f"QSRT schema must be one of {sorted(expected_schemas)!r}, "
                    f"got {qsrt.get('schema')!r}"
                )
            profile = qsrt.get("profile")
            if storage_format == "qsrt_atoms_v2":
                profile = profile or _QSRT_ATOMS_V2_PROFILE_H308
                if profile not in {
                    _QSRT_ATOMS_V2_PROFILE_H308,
                    _QSRT_ATOMS_V2_PROFILE_COUPLED_K2,
                    _QSRT_ATOMS_V2_PROFILE_COUPLED_H308,
                }:
                    raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")
                qsrt = {**qsrt, "profile": profile}
            elif profile is not None:
                raise ValueError("QSRT profile is valid only for qsrt_atoms_v2 storage")
            config.kept_storage = "x4t"
            config.qsrt = dict(qsrt)
            config.trellis_codebook = "sqg_xor_cheb_t12"
        trellis = original_config.get("trellis")
        if trellis is None and isinstance(quantization, dict):
            trellis = quantization.get("trellis")
        if isinstance(trellis, dict):
            codebook = str(trellis.get("codebook", "mcg")).lower()
            if codebook not in {"mcg", "mul1-e4m3"}:
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
                dict.fromkeys(
                    [
                        *(original_config.get("ignored_layers") or []),
                        *_UNSERIALIZED_MULTIMODAL_DENSE_COMPONENTS,
                    ]
                )
            )
        return config


class KQuantHybridMoEMethod(FusedMoEMethodBase):
    """Fused-MoE method serving both hybrid tiers via the b12x W4A16 kernel.

    Generic EXL3 tensors are sharded at load time. Canonical QSRT atoms are
    already organized into directly shardable 32-channel extents. ``apply``
    returns only the routed-expert contribution.
    """

    def __init__(
        self,
        quant_config: KQuantHybridConfig,
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
                "kquant_hybrid only supports SiLU/SiTU-gated MoE layers, got "
                f"{layer.activation}."
            )
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
        qsrt_profile = (
            None
            if not isinstance(self.quant_config.qsrt, dict)
            else self.quant_config.qsrt.get("profile")
        )
        secondary_bits = 2 if qsrt_profile == _QSRT_ATOMS_V2_PROFILE_COUPLED_K2 else 3
        kept = [e for e, b in enumerate(bits) if b == 4]
        demoted = [e for e, b in enumerate(bits) if b == secondary_bits]
        if len(kept) + len(demoted) != num_experts:
            raise ValueError(
                f"hybrid_bit_map entry for {layer.layer_name} contains bit "
                f"widths other than 4 (kept) and {secondary_bits} (secondary)."
            )
        remap = {
            **{e: (0, i) for i, e in enumerate(kept)},
            **{e: (1, i) for i, e in enumerate(demoted)},
        }
        state = _HybridLayerState(remap, hidden, inter, num_experts, kept_mx)
        state.uses_qsrt_atoms = bool(
            mapped_layer and self.quant_config.demoted_format == "qsrt_sqg_e4m3"
        )
        layer.hybrid_state = state

        if state.uses_qsrt_atoms:
            if hidden != 3584 or num_experts != 896:
                raise ValueError(
                    "QSRT serving requires Kimi-K3's global "
                    "H=3584, I=3072, E=896 geometry"
                )
            if qsrt_profile == _QSRT_ATOMS_V2_PROFILE_COUPLED_K2:
                if inter % 128 or inter * tp_size < 3072:
                    raise ValueError(
                        "the coupled pure-K2 QSRT profile requires a padded "
                        "local intermediate extent divisible by 128 and at "
                        f"least 3,072 global channels, got I={inter} at TP={tp_size}"
                    )
                state.tiles = (128, 128, 128, 128)
            elif inter != 256:
                # The fixed H308 decoder has only been qualified for one
                # eight-atom (256-channel) local extent.
                raise ValueError(
                    "the fixed H308 QSRT kernel requires an eight-atom "
                    f"local extent (I=256), got I={inter} at TP={tp_size}"
                )
            if not kept_mx:
                raise ValueError("QSRT serving requires kept_format='mxfp4_e8m0k32'")
            placeholder = torch.nn.Parameter(
                torch.empty(
                    (0,),
                    dtype=torch.uint8,
                    device=torch.accelerator.current_device_index(),
                ),
                requires_grad=False,
            )
            layer.register_parameter("qsrt_atom_placeholder", placeholder)
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
            if tier != 0:
                raise ValueError(
                    f"secondary expert {expert_id} supplied non-EXL3 tensor {name!r}"
                )
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
            if "weight_scale" in name:
                target = getattr(layer, f"{family}_nv_scale")
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
        num_secondary = max(state.num_secondary, 1)
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
                (num_secondary, 2, hidden // 16, inter // 16, tb),
                torch.int16,
            )
            # shared-su artifacts: one broadcast row instead of per-expert
            # H-side vectors (saves ~1.4 GiB/rank at K3 scale).
            h_rows = (
                1
                if getattr(self.quant_config, "trellis_shared_su", False)
                else num_secondary
            )
            register("w13_exl3_suh", (h_rows, 2, hidden), torch.float16)
            register("w13_exl3_svh", (num_secondary, 2, inter), torch.float16)
            register(
                "w2_exl3_trellis",
                (num_secondary, inter // 16, hidden // 16, tb),
                torch.int16,
            )
            register("w2_exl3_suh", (num_secondary, inter), torch.float16)
            register("w2_exl3_svh", (h_rows, hidden), torch.float16)
        register("w13_weight", (num_kept, 2 * inter, hidden // 2))
        register("w13_weight_scale", (1,))
        register("w13_weight_scale_2", (num_kept, 2), torch.float32)
        register("w13_input_scale", (1,), torch.float32)
        register("w2_weight", (num_kept, hidden, inter // 2))
        register("w2_weight_scale", (1,))
        register("w2_weight_scale_2", (num_kept,), torch.float32)
        register("w2_input_scale", (1,), torch.float32)
        # Real block-scale storage, filled by the dispatcher (not routed by
        # the expert mapping). MXFP4 kept tier stores ue8m0 scales per 32
        # group (uint8) instead of e4m3 per group_size.
        nv_group = 32 if kept_mx else group_size
        nv_dtype = torch.uint8 if kept_mx else torch.float8_e4m3fn
        for name, shape, dtype in (
            ("w13_nv_scale", (num_kept, 2 * inter, hidden // nv_group), nv_dtype),
            ("w2_nv_scale", (num_kept, hidden, inter // nv_group), nv_dtype),
        ):
            scale_param = torch.nn.Parameter(
                torch.zeros(
                    shape,
                    dtype=dtype,
                    device=torch.accelerator.current_device_index(),
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, scale_param)

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
                "kquant_hybrid: released %.1f MiB/layer of pre-prepare kept "
                "scale storage (prepared grids are self-contained)",
                freed / 2**20,
            )

    def _load_qsrt_atoms(
        self,
        layer: "RoutedExperts",
        *,
        device: torch.device,
    ) -> None:
        """Load one deployment shard from canonical TP-independent QSRT files."""

        from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
        from b12x.moe import fused_moe
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_w4a16_x4t_weights,
        )

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
            open_qsrt_atom_extent,
            read_qsrt_atom_layer_metadata,
        )
        from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
            open_qsrt_atom_v2_extent,
            read_qsrt_atom_v2_layer_metadata,
        )
        from vllm.model_executor.layers.quantization.kquant_x4t import (
            X4TLayerReader,
        )

        state: _HybridLayerState = layer.hybrid_state
        match = re.search(r"layers\.(\d+)\b", layer.layer_name)
        if match is None:
            raise ValueError(f"cannot resolve a QSRT layer from {layer.layer_name!r}")
        layer_index = int(match.group(1))
        bits = self._layer_bits(layer)
        if bits is None:
            raise ValueError("QSRT layer is absent from hybrid_bit_map")
        model_root = Path(get_current_vllm_config().model_config.model)
        if not model_root.is_dir():
            raise ValueError(
                f"QSRT serving requires a local model directory, got {model_root}"
            )
        manifest_path = model_root / "qsrt-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"QSRT checkpoint is missing {manifest_path}"
            ) from None
        qsrt_config = self.quant_config.qsrt
        if not isinstance(qsrt_config, dict):
            raise ValueError("QSRT format descriptor is unavailable")
        storage_format = qsrt_config.get("storage_format")
        if (
            manifest.get("codec") != "QSRT"
            or manifest.get("storage_schema") != qsrt_config.get("schema")
            or manifest.get("storage_format") != storage_format
            or manifest.get("complete") is not True
        ):
            raise ValueError("QSRT manifest identity is invalid or incomplete")
        if storage_format == "qsrt_atoms_v2" and manifest.get(
            "profile"
        ) != qsrt_config.get("profile"):
            raise ValueError(
                "QSRT manifest profile disagrees with the model descriptor"
            )
        layer_entry = (manifest.get("layers") or {}).get(str(layer_index))
        if not isinstance(layer_entry, dict):
            raise ValueError(f"QSRT manifest omits layer {layer_index}")

        def manifest_file(field: str) -> Path:
            name = layer_entry.get(field)
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError(f"QSRT layer {layer_index} has invalid {field}")
            path = model_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        atom_path = manifest_file("qsrt_atoms")
        x4t_path = manifest_file("x4t") if storage_format == "qsrt_atoms_v1" else None
        if storage_format == "qsrt_atoms_v2":
            metadata_v2 = read_qsrt_atom_v2_layer_metadata(
                atom_path,
                layer=layer_index,
                expected_bits=bits,
            )
            if state.num_secondary != state.num_experts or state.num_kept:
                raise ValueError("QSRT atoms-v2 requires an all-QSRT expert layer")
            expected_profile = qsrt_config.get("profile")
            if metadata_v2.profile != expected_profile:
                raise ValueError(
                    "QSRT atoms-v2 profile disagrees with the model descriptor"
                )
            pure_k2 = metadata_v2.profile == _QSRT_ATOMS_V2_PROFILE_COUPLED_K2
            state.trellis_use_w4a8_prefill = (
                _qsrt_atoms_v2_w4a8_prefill_enabled(pure_k2=pure_k2)
            )
            state.trellis_use_w4a8_prequantized_input = (
                _qsrt_atoms_v2_w4a8_prequantized_input_enabled(
                    prefill_enabled=state.trellis_use_w4a8_prefill
                )
            )
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            weight_plan_kwargs: dict[str, Any] = dict(
                source_format="qsrt_sqg_e4m3",
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                num_experts=state.num_secondary,
                hidden_size=state.hidden_size,
                intermediate_size=state.intermediate_size,
                w13_layout="w13",
                trellis_bits=2 if pure_k2 else 3,
                trellis_tile_config=state.tiles,
                qsrt_storage_format="qsrt_atoms_v2",
                qsrt_profile=metadata_v2.profile,
            )
            plan = fused_moe.plan_weights(
                quant_modes="w4a16",
                **weight_plan_kwargs,
            )
            with open_qsrt_atom_v2_extent(
                metadata_v2,
                shard_count=tp_size,
                shard_index=tp_rank,
                # Keep the canonical ~1 GiB rank extent in host memory while
                # B12X streams bounded groups into its final CUDA layout.  A
                # full CUDA staging extent cannot coexist with the prepared
                # payload in the last layers of this checkpoint.
                device=None,
            ) as (first_atom_slot, atoms):
                state.trellis_weights = fused_moe.prepare_weights(
                    plan=plan,
                    params_dtype=self.moe.in_dtype,
                    qsrt_atom_payload=atoms,
                    qsrt_first_atom_slot=first_atom_slot,
                    qsrt_layer_index=layer_index,
                    gate_suh=metadata_v2.gate_suh.unsqueeze(0).to(device),
                    up_suh=metadata_v2.up_suh.unsqueeze(0).to(device),
                    down_svh=metadata_v2.down_svh.unsqueeze(0).to(device),
                    qsrt_rotation_draws=metadata_v2.rotation_draws,
                )
            if state.trellis_use_w4a8_prefill:
                # QSRT preparation currently accepts one quant mode per plan,
                # but both execution modes consume the exact same immutable
                # trellis-native tensors. Build a second validated package
                # that aliases those allocations instead of preparing a
                # second ~model-sized representation.
                w4a8_plan = fused_moe.plan_weights(
                    quant_modes="w4a8_mx",
                    **weight_plan_kwargs,
                )
                representation = state.trellis_weights.representation
                if representation is None:
                    raise RuntimeError("QSRT W4A16 representation was not prepared")
                state.trellis_prefill_weights = dataclasses.replace(
                    state.trellis_weights,
                    plan=w4a8_plan,
                    representation=dataclasses.replace(
                        representation,
                        quant_mode="w4a8_mx",
                    ),
                )
            layer.qsrt_atom_placeholder.data = (
                layer.qsrt_atom_placeholder.data.new_empty((0,))
            )
            logger.info(
                "Loaded QSRT layer %d atoms-v2 shard %d/%d: %d QSRT experts%s",
                layer_index,
                tp_rank,
                tp_size,
                state.num_secondary,
                " using W4A16 decode + W4A8 prefill"
                if state.trellis_use_w4a8_prefill
                else "",
            )
            return

        metadata = read_qsrt_atom_layer_metadata(
            atom_path,
            layer=layer_index,
            expected_bits=bits,
        )
        expected_secondary = tuple(
            expert
            for expert, (tier, _local) in sorted(
                state.remap.items(), key=lambda item: item[1][1]
            )
            if tier == 1
        )
        expected_kept = tuple(
            expert
            for expert, (tier, _local) in sorted(
                state.remap.items(), key=lambda item: item[1][1]
            )
            if tier == 0
        )
        if tuple(metadata.compressed_expert_ids.tolist()) != expected_secondary:
            raise ValueError("QSRT compressed expert order disagrees with remap")
        if tuple(metadata.x4t_expert_ids.tolist()) != expected_kept:
            raise ValueError("QSRT X4T expert order disagrees with remap")

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if state.num_secondary:
            plan = fused_moe.plan_weights(
                quant_modes="w4a16",
                source_format="qsrt_sqg_e4m3",
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                num_experts=state.num_secondary,
                hidden_size=state.hidden_size,
                intermediate_size=state.intermediate_size,
                w13_layout="w13",
                trellis_bits=3,
                trellis_tile_config=state.tiles,
                qsrt_storage_format="qsrt_atoms_v1",
            )
            with open_qsrt_atom_extent(
                metadata,
                shard_count=tp_size,
                shard_index=tp_rank,
                device=device,
            ) as (first_atom_slot, atoms):
                state.trellis_weights = fused_moe.prepare_weights(
                    plan=plan,
                    params_dtype=self.moe.in_dtype,
                    qsrt_atom_payload=atoms,
                    qsrt_first_atom_slot=first_atom_slot,
                    qsrt_layer_index=layer_index,
                    qsrt_expert_ids=metadata.compressed_expert_ids.to(device),
                    qsrt_format_codes=metadata.format_codes.index_select(
                        0, metadata.compressed_expert_ids.to(torch.int64)
                    ).to(device),
                    gate_suh=metadata.gate_suh.unsqueeze(0).to(device),
                    up_suh=metadata.up_suh.unsqueeze(0).to(device),
                    down_svh=metadata.down_svh.unsqueeze(0).to(device),
                )

        if state.num_kept:
            w13_packed: list[torch.Tensor] = []
            w2_packed: list[torch.Tensor] = []
            w13_fixed: list[torch.Tensor] = []
            w13_exceptions: list[torch.Tensor] = []
            w2_fixed: list[torch.Tensor] = []
            w2_exceptions: list[torch.Tensor] = []
            assert x4t_path is not None
            with X4TLayerReader(
                x4t_path,
                shard_count=tp_size,
                shard_index=tp_rank,
                device=device,
            ) as reader:
                if reader.layer != layer_index:
                    raise ValueError("QSRT X4T sidecar has the wrong layer")
                for expert in expected_kept:
                    w1, w3, w2 = reader.read_shard_triplet(expert)
                    w13_packed.append(torch.cat((w1.packed, w3.packed), dim=0))
                    w2_packed.append(w2.packed)
                    w13_scale = w1.scale.concatenate_rows(w3.scale)
                    w13_fixed.append(w13_scale.fixed)
                    w13_exceptions.append(w13_scale.exceptions)
                    w2_fixed.append(w2.scale.fixed)
                    w2_exceptions.append(w2.scale.exceptions)
            w13_x4t = make_x4t_scale_batch(
                w13_fixed,
                w13_exceptions,
                rows=2 * state.intermediate_size,
                columns=state.hidden_size // 32,
                device=device,
                exception_task_rows=_QSRT_X4T_W13_EXCEPTION_TASK_ROWS,
                exception_row_rotation=_QSRT_X4T_W13_EXCEPTION_ROW_ROTATION,
            )
            w2_x4t = make_x4t_scale_batch(
                w2_fixed,
                w2_exceptions,
                rows=state.hidden_size,
                columns=state.intermediate_size // 32,
                device=device,
                exception_task_rows=_QSRT_X4T_W2_EXCEPTION_TASK_ROWS,
            )
            runtime = self.quant_config.shared_runtime
            if runtime.x4t_w13_scale_scratch is None:
                runtime.x4t_w13_scale_scratch = torch.empty(
                    (
                        state.num_experts,
                        state.hidden_size // 32,
                        2 * state.intermediate_size,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                runtime.x4t_w2_scale_scratch = torch.empty(
                    (
                        state.num_experts,
                        state.intermediate_size // 32,
                        state.hidden_size,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
            assert runtime.x4t_w2_scale_scratch is not None
            global_scale = torch.ones(
                state.num_kept, dtype=torch.float32, device=device
            )
            state.prep_kept = prepare_w4a16_x4t_weights(
                torch.stack(w13_packed).to(device),
                w13_x4t,
                global_scale,
                torch.stack(w2_packed).to(device),
                w2_x4t,
                global_scale.clone(),
                runtime.x4t_w13_scale_scratch,
                runtime.x4t_w2_scale_scratch,
                activation=self.moe.activation.value,
                params_dtype=self.moe.in_dtype,
                w13_layout=_QSRT_X4T_W13_LAYOUT,
            )

        layer.qsrt_atom_placeholder.data = layer.qsrt_atom_placeholder.data.new_empty(
            (0,)
        )
        logger.info(
            "Loaded QSRT layer %d shard %d/%d: %d compressed, %d X4T experts",
            layer_index,
            tp_rank,
            tp_size,
            state.num_secondary,
            state.num_kept,
        )

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        """Prepare the kept tier and the selected W4A16 trellis tier."""
        from b12x.moe._shared.kernels.w4a16.prepare import (
            W4A16PackedWeights,
            _make_workspace,
            _permute_nvfp4_scales,
            _repack_weight,
        )

        state: _HybridLayerState = layer.hybrid_state
        hidden, inter = state.hidden_size, state.intermediate_size
        device = (
            torch.device("cuda", torch.accelerator.current_device_index())
            if state.uses_qsrt_atoms
            else layer.w13_weight.device
        )
        num_kept, num_secondary = state.num_kept, state.num_secondary
        emap_kept = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        emap_secondary = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            (emap_kept if tier == 0 else emap_secondary)[global_id] = local_id
        state.emap_kept, state.emap_secondary = emap_kept, emap_secondary

        if state.uses_qsrt_atoms:
            self._load_qsrt_atoms(layer, device=device)
        elif num_secondary > 0 and self.quant_config.demoted_format == "exl3_3":
            from b12x.moe import fused_moe

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
                num_experts=num_secondary,
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
        elif num_secondary > 0:
            raise ValueError(
                f"unsupported secondary format {self.quant_config.demoted_format!r}"
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
        from b12x.moe._shared.kernels.w4a16.host import (
            max_packed_route_slots,
        )
        from b12x.moe._shared.kernels.w4a16.kernel import (
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
                "kquant_hybrid: TC-decode launch compile failed (%s); "
                "decode steps fall back to the packed-route launch.",
                exc,
            )
        runtime.launches[key] = (decode, prefill)
        return runtime.launches[key]

    def _ensure_runtime(self, layer: "RoutedExperts", m: int, topk: int) -> None:
        """First-apply init: per-tier preplanned launches plus ONE shared
        scratch/buffer set. The first apply is vLLM's eager profile run at
        max_num_batched_tokens, so max_m sizes itself to the serving
        ceiling and nothing compiles during CUDA-graph capture."""
        from b12x.moe._shared.kernels.w4a16.host import (
            make_w4a16_packed_buffers,
            max_packed_route_slots,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if runtime.max_m is None:
            runtime.max_m = max(int(self.moe.max_num_tokens), int(m))
            runtime.topk = int(topk)
        if int(topk) != runtime.topk:
            raise RuntimeError(f"kquant_hybrid: topk changed {runtime.topk} -> {topk}")
        if state.prep_kept is not None:
            state.launch_kept = self._get_launch_pair(state.prep_kept, state)
        if getattr(state, "trellis_weights", None) is not None:
            from b12x.moe import fused_moe

            w4a16_max_m = (
                _B12X_DECODE_M
                if state.trellis_prefill_weights is not None
                else runtime.max_m
            )
            key = (
                "trellis",
                state.num_secondary,
                state.hidden_size,
                state.intermediate_size,
                self.moe.num_experts,
                self.moe.activation.value,
                state.tiles,
                runtime.topk,
                w4a16_max_m,
            )
            plan = runtime.launches.get(key)
            if plan is None:
                caps = fused_moe.Caps(
                    max_tokens=w4a16_max_m,
                    num_topk=runtime.topk,
                    device=torch.accelerator.current_device_index(),
                    weight_plan=state.trellis_weights.plan,
                    quant_mode="w4a16",
                    route_num_experts=self.moe.num_experts,
                    # Full-rotation trellis owns one immutable route geometry
                    # for prewarm, eager execution, and CUDA-graph replay.
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
                    device=spec.device,
                )
            output_shape = (w4a16_max_m, state.hidden_size)
            if (
                runtime.trellis_output is None
                or tuple(runtime.trellis_output.shape) != output_shape
                or runtime.trellis_output.device != spec.device
            ):
                runtime.trellis_output = torch.empty(
                    output_shape,
                    dtype=torch.float32,
                    device=spec.device,
                )
            if state.trellis_prefill_weights is not None:
                prefill_output_shape = (runtime.max_m, state.hidden_size)
                prefill_key = (
                    "trellis-w4a8-prefill",
                    state.num_secondary,
                    state.hidden_size,
                    state.intermediate_size,
                    self.moe.activation.value,
                    state.tiles,
                    runtime.topk,
                    runtime.max_m,
                    state.trellis_use_w4a8_prequantized_input,
                )
                prefill_plan = runtime.launches.get(prefill_key)
                if prefill_plan is None:
                    prefill_plan = fused_moe.plan(
                        fused_moe.Caps(
                            max_tokens=runtime.max_m,
                            num_topk=runtime.topk,
                            device=torch.accelerator.current_device_index(),
                            weight_plan=state.trellis_prefill_weights.plan,
                            quant_mode="w4a8_mx",
                            route_num_experts=0,
                            prequantized_input=(
                                state.trellis_use_w4a8_prequantized_input
                            ),
                        )
                    )
                    if prefill_plan.launch_plan.implementation != "dynamic":
                        raise RuntimeError(
                            "kquant_hybrid: W4A8 prefill plan is not dynamic"
                        )
                    runtime.launches[prefill_key] = prefill_plan
                state.trellis_prefill_plan = prefill_plan
                prefill_spec = prefill_plan.scratch_specs()[0]
                prefill_need = int(torch.Size(prefill_spec.shape).numel())
                prefill_scratch = runtime.trellis_prefill_scratch
                if prefill_scratch is None or (
                    prefill_scratch.numel() < prefill_need
                    or prefill_scratch.dtype != prefill_spec.dtype
                ):
                    runtime.trellis_prefill_scratch = torch.empty(
                        prefill_spec.shape,
                        dtype=prefill_spec.dtype,
                        device=prefill_spec.device,
                    )
                fallback_max_m = 0
                if state.trellis_use_w4a8_prequantized_input:
                    fallback_max_m = max(
                        (
                            candidate_m
                            for candidate_m in range(9, runtime.max_m + 1)
                            if not prefill_plan.supports_prequantized_input(
                                candidate_m
                            )
                        ),
                        default=0,
                    )
                fallback_need = 0
                if fallback_max_m:
                    fallback_key = (
                        "trellis-w4a8-prefill-fallback",
                        state.num_secondary,
                        state.hidden_size,
                        state.intermediate_size,
                        self.moe.activation.value,
                        state.tiles,
                        runtime.topk,
                        fallback_max_m,
                    )
                    fallback_plan = runtime.launches.get(fallback_key)
                    if fallback_plan is None:
                        fallback_plan = fused_moe.plan(
                            fused_moe.Caps(
                                max_tokens=fallback_max_m,
                                num_topk=runtime.topk,
                                device=torch.accelerator.current_device_index(),
                                weight_plan=state.trellis_prefill_weights.plan,
                                quant_mode="w4a8_mx",
                                route_num_experts=0,
                            )
                        )
                        runtime.launches[fallback_key] = fallback_plan
                    if fallback_plan.launch_plan.implementation != "dynamic":
                        raise RuntimeError(
                            "kquant_hybrid: W4A8 fallback plan is not dynamic"
                        )
                    state.trellis_prefill_fallback_plan = fallback_plan
                    fallback_spec = fallback_plan.scratch_specs()[0]
                    fallback_need = int(torch.Size(fallback_spec.shape).numel())
                    fallback_scratch = runtime.trellis_prefill_fallback_scratch
                    if fallback_scratch is None or (
                        fallback_scratch.numel() < fallback_need
                        or fallback_scratch.dtype != fallback_spec.dtype
                    ):
                        runtime.trellis_prefill_fallback_scratch = torch.empty(
                            fallback_spec.shape,
                            dtype=fallback_spec.dtype,
                            device=fallback_spec.device,
                        )
                else:
                    state.trellis_prefill_fallback_plan = None
                    runtime.trellis_prefill_fallback_scratch = None
                if state.trellis_use_w4a8_prequantized_input:
                    scale_shape = (runtime.max_m, state.hidden_size // 32)
                    if (
                        runtime.trellis_prefill_input_values is None
                        or tuple(runtime.trellis_prefill_input_values.shape)
                        != prefill_output_shape
                        or runtime.trellis_prefill_input_values.device
                        != prefill_spec.device
                    ):
                        runtime.trellis_prefill_input_values = torch.empty(
                            prefill_output_shape,
                            dtype=torch.float8_e4m3fn,
                            device=prefill_spec.device,
                        )
                    if (
                        runtime.trellis_prefill_input_scales is None
                        or tuple(runtime.trellis_prefill_input_scales.shape)
                        != scale_shape
                        or runtime.trellis_prefill_input_scales.device
                        != prefill_spec.device
                    ):
                        runtime.trellis_prefill_input_scales = torch.empty(
                            scale_shape,
                            dtype=torch.float8_e8m0fnu,
                            device=prefill_spec.device,
                        )
                    if fallback_max_m:
                        fallback_input_shape = (
                            fallback_max_m,
                            state.hidden_size,
                        )
                        if (
                            runtime.trellis_prefill_input is None
                            or tuple(runtime.trellis_prefill_input.shape)
                            != fallback_input_shape
                            or runtime.trellis_prefill_input.device
                            != prefill_spec.device
                            or runtime.trellis_prefill_input.dtype
                            != self.moe.in_dtype
                        ):
                            runtime.trellis_prefill_input = torch.empty(
                                fallback_input_shape,
                                dtype=self.moe.in_dtype,
                                device=prefill_spec.device,
                            )
                    else:
                        runtime.trellis_prefill_input = None
                else:
                    if (
                        runtime.trellis_prefill_input is None
                        or tuple(runtime.trellis_prefill_input.shape)
                        != prefill_output_shape
                        or runtime.trellis_prefill_input.device != prefill_spec.device
                        or runtime.trellis_prefill_input.dtype != self.moe.in_dtype
                    ):
                        runtime.trellis_prefill_input = torch.empty(
                            prefill_output_shape,
                            dtype=self.moe.in_dtype,
                            device=prefill_spec.device,
                        )
                    runtime.trellis_prefill_input_values = None
                    runtime.trellis_prefill_input_scales = None
                if state.trellis_use_w4a8_prequantized_input:
                    logger.info_once(
                        "kquant_hybrid: W4A8 prequantized MXFP8 input enabled "
                        "for M=%d..%d; BF16 fallback M=9..%d; scratch "
                        "primary=%.1f MiB fallback=%.1f MiB",
                        fallback_max_m + 1,
                        runtime.max_m,
                        fallback_max_m,
                        prefill_need / (1024 * 1024),
                        fallback_need / (1024 * 1024),
                    )
                if (
                    runtime.trellis_prefill_output is None
                    or tuple(runtime.trellis_prefill_output.shape)
                    != prefill_output_shape
                    or runtime.trellis_prefill_output.device != prefill_spec.device
                    or runtime.trellis_prefill_output.dtype != self.moe.in_dtype
                ):
                    runtime.trellis_prefill_output = torch.empty(
                        prefill_output_shape,
                        dtype=self.moe.in_dtype,
                        device=prefill_spec.device,
                    )
                routing_shape = (runtime.max_m, runtime.topk)
                if (
                    runtime.trellis_prefill_topk_ids is None
                    or tuple(runtime.trellis_prefill_topk_ids.shape)
                    != routing_shape
                    or runtime.trellis_prefill_topk_ids.device
                    != prefill_spec.device
                ):
                    runtime.trellis_prefill_topk_ids = torch.empty(
                        routing_shape,
                        dtype=torch.int32,
                        device=prefill_spec.device,
                    )
                if (
                    runtime.trellis_prefill_topk_weights is None
                    or tuple(runtime.trellis_prefill_topk_weights.shape)
                    != routing_shape
                    or runtime.trellis_prefill_topk_weights.device
                    != prefill_spec.device
                ):
                    runtime.trellis_prefill_topk_weights = torch.empty(
                        routing_shape,
                        dtype=torch.float32,
                        device=prefill_spec.device,
                    )
            if (
                os.getenv("VLLM_KQUANT_CAPTURE_DIR")
                and os.getenv("VLLM_KQUANT_CAPTURE_PROFILE", "sampled_hessian")
                == "sampled_hessian"
                and runtime.kquant_logical_mid is None
            ):
                runtime.kquant_logical_mid = torch.empty(
                    (runtime.max_m * runtime.topk, state.intermediate_size),
                    dtype=torch.float16,
                    device=torch.accelerator.current_device_index(),
                )
        if runtime.buffers is None and state.prep_kept is not None:
            prep_any = state.prep_kept
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
            # Fully overwritten by every kept-tier launch, so this output can
            # be shared across layers.
            runtime.out_kept = buffers.output
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
        from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe

        runtime = self.quant_config.shared_runtime
        use_decode = decode and launch_pair[0] is not launch_pair[1]
        launch = launch_pair[0] if use_decode else launch_pair[1]
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if use_decode:
            # Direct top-k path: the kernel reads flat LOCAL ids.  Unlike the
            # packed route builder, B12X's direct launcher cannot safely
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
            # creates thousands of surprise JIT compiles. Keep those tails on
            # the numerically safe packed route.
            packed_m = _B12X_DECODE_M + 1
            if runtime.max_m is None:
                raise RuntimeError("hybrid runtime was not initialized")
            if runtime.max_m < packed_m:
                raise RuntimeError(
                    "kquant_hybrid requires max_num_batched_tokens >= "
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
                f"kquant_hybrid: m={m} exceeds the planned launch "
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
        if state.num_secondary == 0:
            # Uniform kept layer (including all-MXFP4 decoder layers and an
            # unmapped NVFP4 MTP head): single-tier launch.
            return self._run_kept(layer, x, weights, topk_ids, decode)
        if getattr(state, "trellis_weights", None) is not None:
            from b12x.moe import fused_moe

            tids = (
                topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            )
            if not tids.is_contiguous():
                tids = tids.contiguous()
            if runtime.trellis_output is None:
                raise RuntimeError("QSRT trellis output was not allocated eagerly")
            use_w4a8_prefill = (
                state.trellis_prefill_weights is not None and not decode
            )
            use_prequantized_input = False
            if (
                use_w4a8_prefill
                and state.trellis_use_w4a8_prequantized_input
                and state.trellis_prefill_plan is not None
            ):
                use_prequantized_input = (
                    state.trellis_prefill_plan.supports_prequantized_input(m)
                )
            if use_w4a8_prefill:
                trellis_plan = (
                    state.trellis_prefill_plan
                    if use_prequantized_input
                    or not state.trellis_use_w4a8_prequantized_input
                    else state.trellis_prefill_fallback_plan
                )
            else:
                trellis_plan = state.trellis_plan
            trellis_weights = (
                state.trellis_prefill_weights
                if use_w4a8_prefill
                else state.trellis_weights
            )
            if use_w4a8_prefill:
                trellis_scratch = (
                    runtime.trellis_prefill_scratch
                    if use_prequantized_input
                    or not state.trellis_use_w4a8_prequantized_input
                    else runtime.trellis_prefill_fallback_scratch
                )
            else:
                trellis_scratch = runtime.trellis_scratch
            trellis_output = (
                runtime.trellis_prefill_output
                if use_w4a8_prefill
                else runtime.trellis_output
            )
            if trellis_scratch is None:
                raise RuntimeError("QSRT trellis scratch was not allocated eagerly")
            if (
                trellis_plan is None
                or trellis_weights is None
                or trellis_output is None
            ):
                raise RuntimeError("QSRT trellis execution resources are incomplete")
            launch_input = x if x.is_contiguous() else x.contiguous()
            use_outer_transform = use_w4a8_prefill
            prefill_input_values: torch.Tensor | None = None
            prefill_input_scales: torch.Tensor | None = None
            if use_outer_transform:
                from b12x.moe.fused_moe._impl import (
                    run_w4a8_coupled_outer_transform,
                    run_w4a8_coupled_outer_transform_mxfp8,
                    sanitize_w4a8_routing,
                )

                prefill_input = runtime.trellis_prefill_input
                prefill_input_values = runtime.trellis_prefill_input_values
                prefill_input_scales = runtime.trellis_prefill_input_scales
                safe_ids = runtime.trellis_prefill_topk_ids
                safe_weights = runtime.trellis_prefill_topk_weights
                prepared_value = trellis_weights.representation.value
                if safe_ids is None or safe_weights is None:
                    raise RuntimeError(
                        "QSRT W4A8 prefill buffers were not allocated eagerly"
                    )
                if use_prequantized_input:
                    if prefill_input_values is None or prefill_input_scales is None:
                        raise RuntimeError(
                            "QSRT prequantized input buffers were not allocated"
                        )
                    run_w4a8_coupled_outer_transform_mxfp8(
                        launch_input,
                        prefill_input_values[:m],
                        prefill_input_scales[:m],
                        prepared_value.gate_suh,
                    )
                else:
                    if prefill_input is None:
                        raise RuntimeError(
                            "QSRT BF16 outer-transform buffer was not allocated"
                        )
                    launch_input = run_w4a8_coupled_outer_transform(
                        launch_input,
                        prefill_input[:m],
                        prepared_value.gate_suh,
                        output_transform=False,
                    )
                tids, weights = sanitize_w4a8_routing(
                    tids,
                    weights,
                    safe_ids[:m],
                    safe_weights[:m],
                    num_experts=state.num_secondary,
                )
            bind_kwargs: dict[str, Any] = dict(
                scratch=trellis_scratch,
                a=launch_input,
                experts=trellis_weights,
                topk_weights=weights,
                topk_ids=tids,
                output=trellis_output[:m],
            )
            if use_prequantized_input:
                assert prefill_input_values is not None
                assert prefill_input_scales is not None
                bind_kwargs["a_prequant"] = prefill_input_values[:m]
                bind_kwargs["a_prequant_scale"] = prefill_input_scales[:m]
            if not use_outer_transform:
                bind_kwargs["route_expert_map"] = state.emap_secondary
            binding = fused_moe.bind(trellis_plan, **bind_kwargs)
            # W4A16 decode emits fp32 while W4A8 prefill emits model dtype;
            # normalize both contracts for downstream layers.
            out_trellis = fused_moe.run(binding=binding)[:m]
            if use_w4a8_prefill:
                out_trellis = run_w4a8_coupled_outer_transform(
                    out_trellis,
                    out_trellis,
                    prepared_value.down_svh,
                    output_transform=True,
                )
            out_trellis = out_trellis.to(x.dtype)
            if (
                os.getenv("VLLM_KQUANT_CAPTURE_DIR")
                and os.getenv("VLLM_KQUANT_CAPTURE_PROFILE", "sampled_hessian")
                == "sampled_hessian"
            ):
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
                    expert_map=state.emap_secondary,
                    intermediate_rotations=intermediate_rotations,
                    logical_scratch=logical_scratch,
                )
            if state.num_kept == 0:
                return out_trellis
            return self._run_kept(layer, x, weights, topk_ids, decode)[:m] + out_trellis
        raise RuntimeError("secondary trellis weights were not prepared")

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


KQuantHybridConfig.FusedMoEMethodCls = KQuantHybridMoEMethod
