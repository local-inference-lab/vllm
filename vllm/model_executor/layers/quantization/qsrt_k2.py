# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 coupled QSRT-K2 expert loading through canonical B12X plans."""

import re
from pathlib import Path
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedExperts
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.utils.qsrt_k2 import read_qsrt_k2_source
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

logger = init_logger(__name__)


class QsrtK2Config(ModelOptMxFp8Config):
    """Coupled K2 routed experts with serialized MXFP8 dense projections."""

    dense_format = "mxfp8"
    separate_mla_output_gate = True

    def __init__(self, ignored_layers: list[str]) -> None:
        self.dense_ignored_layers = tuple(ignored_layers)
        super().__init__(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[*ignored_layers, "lm_head", "in_proj_gfab"],
        )

    def get_name(self) -> QuantizationMethods:
        return "qsrt_k2"

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if hf_quant_cfg is not None and hf_quant_cfg.get("qsrt"):
            return "qsrt_k2"
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QsrtK2Config":
        expected = {
            "storage_format": "qsrt_atoms_v2",
            "profile": "k2_coupled_h512_h128",
            "schema": "kquant_kimi_k3_qsrt_atoms_v2",
            "encoding": "qsrt_sqg_e4m3",
            "codebook": "sqg_xor_cheb_t12",
            "artifact_manifest": "qsrt-manifest.json",
        }
        qsrt = config.get("qsrt", {})
        if (
            not isinstance(qsrt, dict)
            or any(qsrt.get(key) != value for key, value in expected.items())
            or config.get("dense_format") != "mxfp8"
        ):
            raise ValueError(
                "QSRT-K2 requires the coupled K2 atom schema and MXFP8 dense weights"
            )
        ignored = config.get("ignored_layers")
        if (
            not isinstance(ignored, list)
            or not all(isinstance(name, str) for name in ignored)
            or not {"g_proj", "f_a_proj", "f_b_proj", "b_proj", "kv_b_proj"}.issubset(
                ignored
            )
            or {"q_proj", "k_proj", "v_proj"}.intersection(ignored)
        ):
            raise ValueError(
                "QSRT-K2 requires BF16 KDA gates/factors/beta and MLA KV-B, "
                "with MXFP8 Q/K/V"
            )
        return cls(ignored)

    def is_layer_excluded(self, prefix: str) -> bool:
        if {"vision_tower", "vision_model", "mm_projector"}.intersection(
            prefix.split(".")
        ):
            return True
        # Abbreviated names match path components: b_proj must not exclude q_b_proj.
        return is_layer_skipped(
            prefix,
            self.exclude_modules,
            self.packed_modules_mapping,
            match_mode="suffix",
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return QsrtK2MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)


class QsrtK2MoEMethod(FusedMoEMethodBase):
    """Uniform coupled K2 experts; BF16 activations and exact source rotations."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError(
                "QSRT-K2 supports tensor parallelism without EP or DP"
            )
        if (
            moe.activation != MoEActivation.SITU
            or moe.activation_situ_beta != 4.0
            or moe.activation_situ_linear_beta != 25.0
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
        ):
            raise ValueError(
                "QSRT-K2 requires bias-free BF16 SiTU experts with beta 4/25"
            )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (num_experts, hidden_size, params_dtype) != (896, 3584, torch.bfloat16):
            raise ValueError(
                "QSRT-K2 requires 896 experts, latent width 3584 and BF16 activations"
            )
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None or not 1 <= int(match.group(1)) <= 92:
            raise ValueError("QSRT-K2 requires a routed Kimi-K3 layer numbered 1..92")
        self.layer_index = int(match.group(1))
        # Modular MoE passes these handles but resolves geometry and storage
        # through the prepared package. No padded expert payload is allocated.
        for name in ("w13_weight", "w2_weight"):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(0, dtype=torch.uint8), requires_grad=False
                ),
            )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="btx"),
            _w2=FusedMoEQuantDesc(dtype="btx"),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from b12x.moe import fused_moe

        config = get_current_vllm_config()
        root = Path(config.model_config.model)
        if not (root / "qsrt-manifest.json").is_file():
            raise ValueError(
                "QSRT-K2 requires a local checkpoint directory containing "
                "qsrt-manifest.json"
            )
        source = read_qsrt_k2_source(root, self.layer_index)
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        extent = source.read_extent(tp, rank)
        device = layer.w13_weight.device
        plan = fused_moe.plan_weights(
            source=fused_moe.BtxSource(manifest=extent.manifest),
            activation=fused_moe.ActivationSpec(
                mode="a16",
                nonlinearity="situ",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=896,
                hidden_size=3584,
                intermediate_size=extent.local_intermediate_size,
            ),
        )
        prepared = fused_moe.prepare_weights(
            plan=plan,
            weights=fused_moe.BtxWeights(layer=extent, device=device),
        )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )
        logger.info(
            "QSRT-K2 layer %d rank %d/%d: source slots [%d,%d), %d channels, "
            "B12X coupled K2 with checkpoint rotation draws; BF16 activations",
            self.layer_index,
            rank,
            tp,
            extent.first_slot,
            extent.first_slot + extent.slot_count,
            extent.local_intermediate_size,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
        workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            workspace=workspace,
        )
