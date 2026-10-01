# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly compressed DS4.1 routed experts with native dense tensors."""

import regex as re
import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
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
from vllm.utils.torch_utils import set_default_torch_num_threads

from .quant_config import DeepseekV41FP8Config

logger = init_logger(__name__)


class DeepseekV41Mxfp4CsfConfig(DeepseekV41FP8Config):
    """MXFP4-CSF main experts; unchanged native dense, shared and draft weights."""

    checkpoint_root: str
    scale_scratch: tuple[torch.Tensor, ...] | None

    @classmethod
    def get_name(cls):
        return "mxfp4_csf"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant in (None, cls.get_name())
            and hf_quant_cfg is not None
            and hf_quant_cfg.get("quant_method") == cls.get_name()
            and getattr(hf_config, "model_type", None)
            in ("deepseek_v41", "deepseek_v41_text")
        ):
            return cls.get_name()
        return None

    @classmethod
    def from_config(cls, config):
        if config.get("format_version") != 1 or not config.get("checkpoint_root"):
            raise ValueError("MXFP4-CSF requires format_version=1 and checkpoint_root")
        result = cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
        )
        result.checkpoint_root = config["checkpoint_root"]
        result.scale_scratch = None
        return result

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if match is None:
                raise ValueError("DS4.1 routed experts require a numbered layer")
            config = get_current_vllm_config()
            if int(match.group(1)) < config.model_config.hf_config.num_hidden_layers:
                if (
                    config.parallel_config.pipeline_parallel_size != 1
                    or config.parallel_config.use_ubatching
                ):
                    raise NotImplementedError(
                        "MXFP4-CSF shared scale scratch requires PP1 without ubatching"
                    )
                if config.load_config.load_format not in ("mxfp4_csf",):
                    raise ValueError("MXFP4-CSF requires --load-format mxfp4_csf ")
                return Mxfp4CsfMoEMethod(layer.moe_config, self)
        return super().get_quant_method(layer, prefix)


class Mxfp4CsfMoEMethod(FusedMoEMethodBase):
    """Expand scales for routed experts into serialized, model-owned scratch."""

    def __init__(self, moe, owner):
        super().__init__(moe)
        self.owner = owner
        parallel = moe.moe_parallel_config
        if (
            parallel.use_ep
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.use_all2all_kernels
            or parallel.enable_eplb
        ):
            raise NotImplementedError("MXFP4-CSF DS4.1 supports TP without EP/DP")
        if (
            moe.activation not in (MoEActivation.SILU, MoEActivation.SITU)
            or moe.in_dtype != torch.bfloat16
            or moe.has_bias
            or (
                moe.activation == MoEActivation.SITU
                and (
                    moe.activation_situ_beta != 4.0
                    or moe.activation_situ_linear_beta != 25.0
                )
            )
        ):
            raise ValueError(
                "MXFP4-CSF requires bias-free BF16 SwiGLU or SiTU(4,25) experts"
            )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None or params_dtype != torch.bfloat16:
            raise ValueError("MXFP4-CSF requires a numbered BF16 expert layer")
        self.layer_index = int(match.group(1))
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.local_intermediate = intermediate_size_per_partition
        for name in ("w13_weight", "w2_weight"):
            layer.register_buffer(
                name, torch.empty(0, dtype=torch.uint8), persistent=False
            )

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="mxfp4"),
            _w2=FusedMoEQuantDesc(dtype="mxfp4"),
        )

    def process_weights_after_loading(self, layer):
        from b12x.moe import fused_moe

        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            read_mxfp4_csf_layer,
        )

        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        device = layer.w13_weight.device
        e, h, n = self.num_experts, self.hidden_size, self.local_intermediate
        shapes = ((e, h // 32, 2 * n), (e, n // 32, h))
        if self.owner.scale_scratch is None:
            # MXFP4-CSF disallows ubatching: every layer consumes these scale grids
            # before its successor may overwrite them on the same stream.
            self.owner.scale_scratch = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )
        scratch = self.owner.scale_scratch
        if any(
            t.shape != shape or t.device != device for t, shape in zip(scratch, shapes)
        ):
            raise ValueError("MXFP4-CSF shared scale scratch geometry/device mismatch")
        # Per-expert CPU slices are too small to amortize intra-op barriers.
        # Restore the serving thread policy before kernel preparation.
        with set_default_torch_num_threads(1):
            weights = read_mxfp4_csf_layer(
                self.owner.checkpoint_root,
                self.layer_index,
                num_experts=e,
                hidden_size=h,
                intermediate_size=n * tp,
                tp_rank=rank,
                tp_size=tp,
                device=device,
                w13_scale_scratch=scratch[0],
                w2_scale_scratch=scratch[1],
            )
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=fused_moe.ActivationSpec(
                mode="a16",
                nonlinearity="situ"
                if self.moe.activation == MoEActivation.SITU
                else "silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=self.moe.swiglu_limit,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=e, hidden_size=h, intermediate_size=n
            ),
        )
        prepared = fused_moe.prepare_weights(plan=plan, weights=weights)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        backend = B12xExperts(self.moe, self.moe_quant_config)
        backend.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), backend
        )
        logger.info(
            "MXFP4-CSF lossless MXFP4 layer %d rank %d/%d: BF16 activations, "
            "compressed scales, shared scratch %d bytes",
            self.layer_index,
            rank,
            tp,
            sum(t.numel() for t in scratch),
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        workspace=None,
    ):
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
