# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt safetensors IQ2_XS dense and routed weights with BF16 activations."""

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.weight_transfer import copy_weight


class ModelOptIQ2XSMoEMethod(FusedMoEMethodBase):
    """Load 74-byte blocks and retain compact b12x prepared expert storage."""

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        if moe_config.moe_backend not in ("auto", "b12x"):
            raise ValueError("IQ2_XS routed experts require the b12x MoE backend")
        if not B12xExperts._supports_current_device():
            raise ValueError("IQ2_XS routed experts require b12x on SM12x")
        if not B12xExperts._supports_parallel_config(moe_config.moe_parallel_config):
            raise ValueError("IQ2_XS supports tensor parallelism without EP or EPLB")
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError("IQ2_XS routed experts require BF16 activations")
        if (
            moe_config.activation
            not in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)
            or moe_config.has_bias
        ):
            raise ValueError("IQ2_XS routed experts require bias-free SiLU or ReLU2")
        self.gated = moe_config.activation == MoEActivation.SILU
        if any(
            value is not None
            for value in (
                moe_config.swiglu_limit,
                moe_config.swiglu_alpha,
                moe_config.swiglu_beta,
            )
        ):
            raise ValueError("IQ2_XS routed experts require standard SiLU parameters")

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 256 or intermediate_size_per_partition % 256:
            raise ValueError(
                "IQ2_XS hidden and per-rank intermediate sizes must align to 256"
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self.maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            self.moe.moe_parallel_config,
        )
        for name, shape in (
            (
                "w13_weight",
                (
                    num_experts,
                    (2 if self.gated else 1) * intermediate_size_per_partition,
                    hidden_size // 256,
                    74,
                ),
            ),
            (
                "w2_weight",
                (num_experts, hidden_size, intermediate_size_per_partition // 256, 74),
            ),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=torch.empty(shape, dtype=torch.uint8),
                    input_dim=2,
                    output_dim=1,
                    weight_loader=self.weight_loader,
                ),
            )

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"invalid IQ2_XS expert projection: {shard_id}")
        if not self.gated and shard_id == "w3":
            raise ValueError("non-gated IQ2_XS experts have no w3 projection")
        local_i = self.moe.intermediate_size_per_partition
        global_i = local_i * self.moe.moe_parallel_config.tp_size
        hidden = self.moe.hidden_dim
        expected = (
            (hidden, global_i // 256, 74)
            if shard_id == "w2"
            else (global_i, hidden // 256, 74)
        )
        if loaded_weight.dtype != torch.uint8 or tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"invalid IQ2_XS tensor {weight_name}: expected uint8{expected}, "
                f"got {loaded_weight.dtype}{tuple(loaded_weight.shape)}"
            )
        if not 0 <= expert_id < param.shape[0]:
            raise ValueError(f"invalid IQ2_XS expert index: {expert_id}")
        destination = param.data[expert_id]
        if shard_id == "w2":
            source = loaded_weight.narrow(
                1, self.moe.tp_rank * (local_i // 256), local_i // 256
            )
        else:
            source = loaded_weight.narrow(0, self.moe.tp_rank * local_i, local_i)
            destination = destination.narrow(
                0, 0 if shard_id == "w1" else local_i, local_i
            )
        copy_weight(destination, source)
        return True if return_success else None

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="iq2_xs"),
            _w2=FusedMoEQuantDesc(dtype="iq2_xs"),
        )

    def process_weights_after_loading(self, layer):
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        assert prepare_finalize is not None
        experts = B12xExperts(self.moe, self.moe_quant_config)
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
        experts.process_weights_after_loading(layer)

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
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            workspace=workspace,
        )


ModelOptIQ2XSMoEMethod.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


@register_weight_loader_v2_supported_method
class ModelOptIQ2XSLinearMethod(LinearMethodBase):
    """Load native block payloads and execute the prepared b12x dense API."""

    def __init__(self):
        from vllm.model_executor.kernels.linear import _get_linear_backend
        from vllm.utils.b12x import get_b12x_blockscaled

        if _get_linear_backend(quantization="iq2_xs") not in ("auto", "b12x"):
            raise ValueError("IQ2_XS dense weights require the b12x linear backend")
        api = get_b12x_blockscaled()
        if (
            api is None
            or not hasattr(api, "IQ2XSLinearWeight")
            or not api.is_supported()
        ):
            raise ValueError("IQ2_XS dense weights require b12x IQ2_XS on SM12x")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        n = sum(output_partition_sizes)
        if params_dtype != torch.bfloat16 or input_size_per_partition % 256 or n % 8:
            raise ValueError("IQ2_XS dense weights require BF16, K256 and N8")
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    (n, input_size_per_partition // 256, 74), dtype=torch.uint8
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            ),
        )

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.kernels.linear.b12x_blockscaled import (
            B12xBlockscaledLinear,
        )
        from vllm.model_executor.utils import replace_parameter
        from vllm.utils.b12x import (
            b12x_layer_prefix,
            get_b12x_blockscaled,
            register_b12x_layer,
            set_b12x_preparation_provider,
        )
        from vllm.utils.torch_utils import _encode_layer_name

        api = get_b12x_blockscaled()
        assert api is not None
        packed = api.pack_weight(layer.weight.data, recipe="iq2_xs")
        name = b12x_layer_prefix(layer)
        layer.b12x_linear = B12xBlockscaledLinear(
            packed,
            recipe="iq2_xs",
            activation_mode="a16",
            layer_name=name,
        )
        layer.b12x_layer_name = _encode_layer_name(name)
        register_b12x_layer(name, layer)
        set_b12x_preparation_provider(layer, self)
        replace_parameter(
            layer,
            "weight",
            torch.empty(
                0,
                dtype=torch.uint8,
                device=packed.values.device,
            ),
        )

    def get_b12x_preparation_units(self, layer, workload):
        linear = layer.b12x_linear
        return (linear.unit(workload, name=f"linear.iq2_xs.{linear.layer_name}"),)

    def get_workspace_size(self, layer, rows):
        return layer.b12x_linear.get_workspace_size(rows)

    def apply(self, layer, x, bias=None):
        from vllm.utils.b12x import run_b12x_blockscaled_linear

        if x.dtype != torch.bfloat16:
            raise ValueError("IQ2_XS dense execution requires BF16 activations")
        n = layer.b12x_linear.out_features
        output = run_b12x_blockscaled_linear(
            x.reshape(-1, x.shape[-1]).contiguous(),
            bias,
            n,
            layer.b12x_layer_name,
        )
        return output.view(*x.shape[:-1], n)
