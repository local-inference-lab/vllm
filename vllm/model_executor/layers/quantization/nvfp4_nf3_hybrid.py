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

Both tiers execute through the b12x W4A16 CuteDSL fused-MoE kernel as
preplanned launches sharing one scratch/route buffer set. All compiles
happen during vLLM's eager profile run (the first forward), so the path is
CUDA-graph safe. Decode steps (M <= 8) use the kernel's TC-decode launch
with direct top-k routing over -1-masked local expert ids and a fused
top-k sum (``zero_fc2_output=False``); larger batches use the packed-route
launch with an expert map.
"""

import dataclasses
import re
from typing import TYPE_CHECKING, Any

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
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts

logger = init_logger(__name__)

# Pinned CTA tiles (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n): the NF3
# flat-span weight layout is packed for a SPECIFIC tile_n, but the kernel's
# auto tile selection is m-dependent (fc1_tile_n flips 128<->256 across m).
# (64, 256, 64, 256) is what auto-selection picks for the max-m prefill, and
# its shared-memory/register footprint fits both moe_block_size 8 (decode)
# and 64 (prefill) at both scale formats.
_B12X_TILES = (64, 256, 64, 256)
# Batches of at most this many tokens take the preplanned TC-decode launch.
_B12X_DECODE_M = 8
# Global scale the NF3 prepare path expects (scales are stored pre-divided).
_NF3_GLOBAL_SCALE = 2.0**116
# Expert-chunk size for NF3 unpack/repack (bounds transient VRAM: the int32
# code planes are ~400 MB per 16 w13 experts at GLM-5.2 shapes).
_NF3_PACK_CHUNK = 16
# Exact one-grid decode specialization published for the TP4 hybrid checkpoint.
_GRID188_M = 4
_GRID188_TOPK = 8
_GRID188_HIDDEN = 6144
_GRID188_INTERMEDIATE = 512
_GRID188_NUM_KEPT = 64
_GRID188_NUM_NF3 = 192


def _combined_tier_local_descriptors(
    remap: dict[int, tuple[int, int]],
) -> list[int]:
    """Encode an exact E64/E192 partition for the mapped Grid188 kernel."""
    descriptors = [-1] * (_GRID188_NUM_KEPT + _GRID188_NUM_NF3)
    seen_local = (set(), set())
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
        local_limit = (
            _GRID188_NUM_KEPT if tier_i == 0 else _GRID188_NUM_NF3 if tier_i == 1 else 0
        )
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
        descriptors[global_id_i] = local_id_i if tier_i == 0 else 0x100 | local_id_i
    if any(descriptor < 0 for descriptor in descriptors):
        raise ValueError("heterogeneous remap does not cover all 256 global experts")
    if seen_local[0] != set(range(_GRID188_NUM_KEPT)) or seen_local[1] != set(
        range(_GRID188_NUM_NF3)
    ):
        raise ValueError("heterogeneous remap is not a complete E64/E192 partition")
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


def _unpack_nf3_codes(packed: torch.Tensor, size_k: int) -> torch.Tensor:
    """Unpack NF3 codes stored 8-per-3-bytes: uint8 [E, N, K//8*3] -> int32
    [E, N, K] codes in 0..7."""
    num_experts, rows, _ = packed.shape
    triplets = packed.reshape(num_experts, rows, size_k // 8, 3).to(torch.int32)
    word = triplets[..., 0] | (triplets[..., 1] << 8) | (triplets[..., 2] << 16)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 3
    codes = (word.unsqueeze(-1) >> shifts) & 7
    return codes.reshape(num_experts, rows, size_k)


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
        self.launches: dict[tuple, tuple[Any, Any]] = {}
        self.buffers: Any = None
        self.out_kept: torch.Tensor | None = None
        self.out_nf3: torch.Tensor | None = None
        self.grid188_launch: Any = None
        self.grid188_scratch: dict[str, torch.Tensor] | None = None
        self.grid188_sms: int | None = None
        self.grid188_max_shared_mem: int | None = None
        self.grid188_disabled_reason: str | None = None


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
        # b12x prepared weights (W4A16PackedWeights / PreparedNF3MoeWeights).
        self.prep_kept: Any = None
        self.prep_nf3: Any = None
        # Global -> local id maps, -1 for experts outside the tier.
        self.emap_kept: torch.Tensor | None = None
        self.emap_nf3: torch.Tensor | None = None
        # (decode_launch, prefill_launch) per tier, set at first apply.
        self.launch_kept: tuple[Any, Any] | None = None
        self.launch_nf3: tuple[Any, Any] | None = None
        # MXFP4 kept tier: modular kernel + its weight-holder module and a
        # global -> local map whose sentinel is num_kept (kernel drops it).
        self.kept_kernel: Any = None
        self.kept_module: torch.nn.Module | None = None
        self.kept_remap: torch.Tensor | None = None
        # Exact TP4 E64-NVFP4/E192-NF3 one-grid decode resources.
        self.grid188_weight_views: tuple[torch.Tensor, ...] | None = None
        self.grid188_tier_map: torch.Tensor | None = None
        self.grid188_output: torch.Tensor | None = None
        self.grid188_ready = False
        # Keeps kernel-format tensors alive: b12x prepared weights VIEW the
        # converted tensors, so dropping them would dangle the views.
        self.keepalive: Any = None
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
        self.shared_runtime = _HybridSharedRuntime()

    def get_name(self) -> QuantizationMethods:
        return "nvfp4_nf3_hybrid"

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
        if layer.activation is not MoEActivation.SILU:
            raise NotImplementedError(
                "nvfp4_nf3_hybrid only supports SiLU-gated MoE layers, got "
                f"{layer.activation}."
            )
        bits = self._layer_bits(layer)
        kept_mx = bits is not None and self.quant_config.kept_format == "mxfp4_e8m0k32"
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
        layer.hybrid_state = state

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
            tier, local_id = state.remap[int(expert_id)]
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
            elif "weight_packed" in name:  # NF3 packed codes
                target = getattr(layer, f"{family}_weight_packed")
            else:  # plain NVFP4/MXFP4 weight
                target = getattr(layer, f"{family}_weight")
            dst = target.data[local_id]
            if family == "w13" and shard_id in ("w1", "w3"):
                # gate -> top half, up -> bottom half of the fused rows.
                half = dst.shape[0] // 2
                dst = dst[:half] if shard_id == "w1" else dst[half:]
            dst.copy_(loaded_weight.reshape(dst.shape).to(dst.dtype))
            return True

        def register(name: str, shape: tuple[int, ...], dtype=torch.uint8) -> None:
            param = torch.nn.Parameter(
                torch.zeros(shape, dtype=dtype, device=torch.cuda.current_device()),
                requires_grad=False,
            )
            set_weight_attrs(param, {"weight_loader": hybrid_weight_loader})
            layer.register_parameter(name, param)

        num_kept = max(state.num_kept, 1)
        num_nf3 = max(state.num_nf3, 1)
        # Names the stock prefix-based expert mapping produces; the scalar
        # *_weight_scale / *_input_scale entries are dispatchers whose loads
        # are routed (or dropped) by hybrid_weight_loader above.
        register("w13_weight", (num_kept, 2 * inter, hidden // 2))
        register("w13_weight_packed", (num_nf3, 2 * inter, hidden // 8 * 3))
        register("w13_weight_scale", (1,))
        register("w13_weight_scale_2", (num_kept, 2), torch.float32)
        register("w13_input_scale", (1,), torch.float32)
        register("w2_weight", (num_kept, hidden, inter // 2))
        register("w2_weight_packed", (num_nf3, hidden, inter // 8 * 3))
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
            ("w13_nf3_scale", (num_nf3, 2 * inter, hidden // 32), torch.float8_e4m3fn),
            ("w2_nv_scale", (num_kept, hidden, inter // nv_group), nv_dtype),
            ("w2_nf3_scale", (num_nf3, hidden, inter // 32), torch.float8_e4m3fn),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.zeros(shape, dtype=dtype, device=torch.cuda.current_device()),
                    requires_grad=False,
                ),
            )

    def _build_kept_mxfp4(self, layer: "RoutedExperts") -> None:
        """Build the MXFP4 kept tier as a modular kernel over the kept
        experts via the stock mxfp4 oracle chain (W4A16 activations).

        The kernel is built over a no-parallel clone of the MoE config with
        the per-rank intermediate size: the weights are already TP-sharded
        by the weight loader, so the kernel must see tp=1 (the layer's
        post-apply all-reduce handles TP). ``apply`` remaps top-k ids so
        kept experts map to [0, num_kept) and everything else to the
        sentinel num_kept, which the kernel drops.
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
        kept_module = torch.nn.Module()
        kept_module.activation = layer.activation
        kept_module.moe_config = kept_moe
        kept_module.local_num_experts = num_kept
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
        kernel = make_mxfp4_moe_kernel(
            quant_config,
            kept_moe,
            experts_cls,
            mxfp4_backend=backend,
            routing_tables=None,
        )
        kernel.fused_experts.process_weights_after_loading(kept_module)
        # Owning a modular kernel makes supports_internal_mk True, so vLLM's
        # post-load maybe_init_modular_kernel() returns early instead of
        # rebuilding a kernel from the (freed) standard weight attrs.
        self.moe_kernel = kernel
        kept_remap = torch.full(
            (state.num_experts,), num_kept, dtype=torch.int32, device=device
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
        from b12x.moe.fused.w4a16.prepare import (
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
        device = layer.w13_weight.device
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
        fc1_tile_n, fc2_tile_n = _B12X_TILES[1], _B12X_TILES[3]

        if num_nf3 > 0:
            w13_planes, w2_planes = [], []
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                codes = _unpack_nf3_codes(
                    layer.w13_weight_packed[start : start + _NF3_PACK_CHUNK], hidden
                )
                w13_planes.append(
                    _nf3_pack_code_experts(
                        codes, size_k=hidden, size_n=2 * inter, tile_n=fc1_tile_n
                    )
                )
                del codes
            for start in range(0, num_nf3, _NF3_PACK_CHUNK):
                codes = _unpack_nf3_codes(
                    layer.w2_weight_packed[start : start + _NF3_PACK_CHUNK], inter
                )
                w2_planes.append(
                    _nf3_pack_code_experts(
                        codes, size_k=inter, size_n=hidden, tile_n=fc2_tile_n
                    )
                )
                del codes
            w13_nf3 = torch.cat(w13_planes, 0).contiguous()
            del w13_planes
            w2_nf3 = torch.cat(w2_planes, 0).contiguous()
            del w2_planes
            w13_nf3_scale = _nf3_pack_scale_experts(
                layer.w13_nf3_scale.float(), size_k=hidden, size_n=2 * inter
            )
            w2_nf3_scale = _nf3_pack_scale_experts(
                layer.w2_nf3_scale.float(), size_k=inter, size_n=hidden
            )
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

        if num_kept > 0 and state.kept_mx:
            self._build_kept_mxfp4(layer)
        elif num_kept > 0:
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
        # Free the NF3 originals (both tiers now live in kernel format).
        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_nf3_scale",
            "w2_nf3_scale",
        ):
            param = getattr(layer, name)
            param.data = param.data.new_empty((0,))

    def _get_launch_pair(self, prepared: Any) -> tuple[Any, Any]:
        """Compile (or fetch cached) preplanned launches for one tier.

        The prefill launch covers ALL m in [1, max_m]: packed block-64
        routes + expert_map + ``zero_fc2_output=True``. The decode launch
        (m <= 8) compiles at forced pin tiles with block-8 direct top-k
        routing and a fused top-k sum; if that compile is unavailable the
        packed launch also serves decode.
        """
        from b12x.moe.fused.w4a16.host import max_packed_route_slots
        from b12x.moe.fused.w4a16.kernel import compile_w4a16_fused_moe

        runtime = self.quant_config.shared_runtime
        hidden = self.moe.hidden_dim
        inter = self.moe.intermediate_size_per_partition
        key = (
            prepared.num_experts,
            prepared.weight_layout,
            prepared.scale_format,
            runtime.topk,
            runtime.max_m,
            hidden,
            inter,
        )
        cached = runtime.launches.get(key)
        if cached is not None:
            return cached
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        common = dict(
            hidden_size=hidden,
            intermediate_size=inter,
            num_experts=prepared.num_experts,
            top_k=runtime.topk,
            activation="silu",
            apply_router_weight_on_input=False,
            element_dtype="bf16",
            fast_math=True,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            ),
            weight_layout=prepared.weight_layout,
            scale_format=prepared.scale_format,
            force_tile_config=_B12X_TILES,
        )
        cap_slots = max_packed_route_slots(
            runtime.max_m * runtime.topk, 64, self.moe.num_experts
        )
        prefill = compile_w4a16_fused_moe(
            size_m=runtime.max_m,
            zero_fc2_output=True,
            moe_block_size=64,
            max_m_blocks=(cap_slots + 63) // 64,
            direct_topk_routes=False,
            tc_decode_fused_sum=False,
            **common,
        )
        assert (int(prefill.fc1_tile_n), int(prefill.fc2_tile_n)) == (
            _B12X_TILES[1],
            _B12X_TILES[3],
        ), "b12x tile pin failed"
        decode = prefill
        try:
            candidate = compile_w4a16_fused_moe(
                size_m=_B12X_DECODE_M,
                zero_fc2_output=False,
                moe_block_size=8,
                max_m_blocks=_B12X_DECODE_M * runtime.topk,
                direct_topk_routes=True,
                tc_decode_fused_sum=True,
                **common,
            )
            assert (int(candidate.fc1_tile_n), int(candidate.fc2_tile_n)) == (
                _B12X_TILES[1],
                _B12X_TILES[3],
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
    def _grid188_prepared_views(prepared: Any) -> tuple[torch.Tensor, ...]:
        return (
            prepared.w13.view(torch.int32).view(-1),
            prepared.w2.view(torch.int32).view(-1),
            prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
            prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
            prepared.w13_global_scale.view(-1),
            prepared.w2_global_scale.view(-1),
        )

    @staticmethod
    def _borrow_grid188_scratch(
        buffers: Any,
        *,
        device: torch.device,
        scratch_elements: int,
        workspace_words: int,
    ) -> dict[str, torch.Tensor]:
        """Borrow serial-path buffers; the two decode paths never overlap."""
        specs = (
            ("fc1", "intermediate_cache13", torch.bfloat16, (32, 1024)),
            ("activated", "intermediate_cache2", torch.bfloat16, (32, 512)),
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
                    f"Grid188 scratch source {source_name} failed admission"
                )
            storage_id = int(source.untyped_storage().data_ptr())
            if storage_id in storage_ids:
                raise RuntimeError("Grid188 scratch sources alias each other")
            storage_ids.add(storage_id)
            borrowed[target_name] = source.view(-1)[:elements].view(shape)
        borrowed["workspace"] = torch.zeros(
            (workspace_words,), dtype=torch.int32, device=device
        )
        return borrowed

    def _prepare_grid188(self, layer: "RoutedExperts", topk: int) -> None:
        """Arm the exact NF3 Grid188 path during the eager profile forward."""
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

            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            )
            if runtime.grid188_launch is None:
                from b12x.moe.fused.w4a16.kernel import (
                    compile_w4a16_hybrid_mapped_grid188,
                )

                launch = compile_w4a16_hybrid_mapped_grid188(
                    size_m=_GRID188_M,
                    hidden_size=_GRID188_HIDDEN,
                    intermediate_size=_GRID188_INTERMEDIATE,
                    nv_num_experts=_GRID188_NUM_KEPT,
                    nf_num_experts=_GRID188_NUM_NF3,
                    top_k=_GRID188_TOPK,
                    activation="silu",
                    element_dtype="bf16",
                    fast_math=True,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    force_tile_config=_B12X_TILES,
                )
                if (
                    int(launch.grid_x) != 188
                    or int(launch.blocks_per_sm) != 1
                    or int(launch.size_m) != _GRID188_M
                    or int(launch.shared_memory_bytes) != 45_184
                    or int(launch.route_slots) != _GRID188_M * _GRID188_TOPK
                    or int(launch.map_slots) != _GRID188_NUM_KEPT + _GRID188_NUM_NF3
                ):
                    raise RuntimeError("compiled Grid188 launch failed admission")
                if not hasattr(torch.ops.b12x, "w4a16_hybrid_mapped_grid188_launch"):
                    raise RuntimeError("Grid188 custom op is unavailable")
                runtime.grid188_scratch = self._borrow_grid188_scratch(
                    runtime.buffers,
                    device=prep_kept.w13.device,
                    scratch_elements=int(launch.scratch_elements),
                    workspace_words=int(launch.workspace_words),
                )
                runtime.grid188_sms = sms
                runtime.grid188_max_shared_mem = max_shared_mem
                runtime.grid188_launch = launch

            weight_views = (
                *self._grid188_prepared_views(prep_kept),
                *self._grid188_prepared_views(prep_nf3),
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
                "nvfp4_nf3_hybrid: armed exact TP4 Grid188 one-grid decode"
            )
        except Exception as exc:
            runtime.grid188_disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "nvfp4_nf3_hybrid: Grid188 unavailable; using serial decode: %s",
                runtime.grid188_disabled_reason,
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
        torch.ops.b12x.w4a16_hybrid_mapped_grid188_launch(
            x,
            *state.grid188_weight_views,
            topk_ids,
            state.grid188_tier_map,
            scratch["fc1"],
            scratch["activated"],
            state.grid188_output,
            topk_weights,
            scratch["fc1_c_tmp"],
            scratch["fc2_c_tmp"],
            scratch["workspace"],
            _GRID188_M,
            int(launch.size_m),
            int(launch.hidden_size),
            int(launch.intermediate_size),
            int(launch.nv_num_experts),
            int(launch.nf_num_experts),
            int(launch.top_k),
            launch.activation,
            launch.element_dtype,
            bool(launch.fast_math),
            runtime.grid188_sms,
            runtime.grid188_max_shared_mem,
            int(launch.fc1_tile_k),
            int(launch.fc1_tile_n),
            int(launch.fc2_tile_k),
            int(launch.fc2_tile_n),
            int(launch.grid_x),
            int(torch.cuda.current_stream(x.device).cuda_stream),
        )
        return state.grid188_output

    def _ensure_runtime(self, layer: "RoutedExperts", m: int, topk: int) -> None:
        """First-apply init: per-tier preplanned launches plus ONE shared
        scratch/buffer set. The first apply is vLLM's eager profile run at
        max_num_batched_tokens, so max_m sizes itself to the serving
        ceiling and nothing compiles during CUDA-graph capture."""
        from b12x.moe.fused.w4a16.host import (
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
            state.launch_kept = self._get_launch_pair(state.prep_kept)
        if state.prep_nf3 is not None:
            state.launch_nf3 = self._get_launch_pair(state.prep_nf3)
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
            # them, so sharing them across layers is safe.
            runtime.out_kept = buffers.output
            runtime.out_nf3 = torch.empty_like(buffers.output)
        self._prepare_grid188(layer, topk)
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
        from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

        runtime = self.quant_config.shared_runtime
        use_decode = decode and launch_pair[0] is not launch_pair[1]
        launch = launch_pair[0] if use_decode else launch_pair[1]
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if use_decode:
            # Direct top-k path: the kernel reads flat LOCAL ids and skips
            # negatives itself; expert_map doubles as the global -> local
            # lookup table (graph-safe gather) and must not be passed.
            ids = expert_map[ids.long()].to(torch.int32).contiguous()
            launch_expert_map = None
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
            activation="silu",
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
        the production modular kernel (sentinel-remapped top-k ids)."""
        state: _HybridLayerState = layer.hybrid_state
        runtime = self.quant_config.shared_runtime
        if state.prep_kept is not None:
            m = x.shape[0]
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
        kept_module = state.kept_module
        kept_ids = state.kept_remap[topk_ids.long()]
        return state.kept_kernel.apply(
            x,
            kept_module.w13_weight,
            kept_module.w2_weight,
            topk_weights,
            kept_ids,
            activation=kept_module.activation,
            global_num_experts=state.num_kept,
            expert_map=None,
            apply_router_weight_on_input=False,
            shared_experts=None,
            shared_experts_input=None,
        )

    def apply(
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
        if state.grid188_ready and m == _GRID188_M:
            grid_ids = (
                topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
            )
            if not grid_ids.is_contiguous():
                grid_ids = grid_ids.contiguous()
            if (
                x.dtype == torch.bfloat16
                and x.is_contiguous()
                and grid_ids.numel() == _GRID188_M * _GRID188_TOPK
                and grid_ids.is_cuda
                and grid_ids.device == x.device
                and grid_ids.data_ptr() % 16 == 0
                and weights.numel() == _GRID188_M * _GRID188_TOPK
            ):
                logger.info_once(
                    "nvfp4_nf3_hybrid: executing TP4 Grid188 one-grid decode"
                )
                return self._run_grid188(layer, x, weights, grid_ids)
        if state.num_nf3 == 0:
            # Uniform-NVFP4 layer (e.g. MTP head): single-tier launch.
            output = torch.empty((m, state.hidden_size), dtype=x.dtype, device=x.device)
            return self._run_tier(
                x,
                weights,
                topk_ids,
                state.prep_kept,
                state.launch_kept,
                state.emap_kept,
                output,
                decode,
            )
        out_kept = self._run_kept(layer, x, weights, topk_ids, decode)
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


NvFp4Nf3HybridConfig.FusedMoEMethodCls = NvFp4Nf3HybridMoEMethod
