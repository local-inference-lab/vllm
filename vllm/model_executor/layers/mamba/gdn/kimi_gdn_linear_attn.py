# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from ...linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from ..mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

logger = init_logger(__name__)


@eager_break_during_capture
def kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1=g1,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_attention",
    op_func=kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kda_attention_fake,
)


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        # Q, K, and V have identical TP sharding and consume the same
        # activation.  Keep them in one physical projection so quantized
        # backends quantize the activation once and launch one GEMM instead of
        # repeating both operations three times.
        self.qkv_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [projection_size, projection_size, projection_size],
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def load_a_log(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            # K3 serializes A_log as a padded flat [128] tensor although the
            # reference implementation consumes only num_heads=96 entries.
            # Shard the logical prefix in the same contiguous head order as
            # q/k/v, then reshape it to the broadcast shape used by vLLM KDA.
            if loaded_weight.ndim != 1 or loaded_weight.numel() < self.num_heads:
                raise ValueError(
                    "K3 A_log must be a padded flat tensor with at least "
                    f"{self.num_heads} entries, got {tuple(loaded_weight.shape)}"
                )
            start = get_tensor_model_parallel_rank() * self.local_num_heads
            local = loaded_weight.narrow(0, start, self.local_num_heads)
            param.data.copy_(local.reshape(param.shape))

        set_weight_attrs(self.A_log, {"weight_loader": load_a_log})

        self.use_full_rank_gate = bool(kda_config.get("use_full_rank_gate", False))
        self.g_proj: ColumnParallelLinear | None
        self.g_a_proj: ReplicatedLinear | None
        self.g_b_proj: ColumnParallelLinear | None
        if self.use_full_rank_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_proj",
            )
            self.g_a_proj = None
            self.g_b_proj = None
        else:
            self.g_proj = None
            self.g_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.gate_lower_bound = kda_config.get("gate_lower_bound")
        if self.gate_lower_bound is not None:
            self.gate_lower_bound = float(self.gate_lower_bound)
            if not -20.0 <= self.gate_lower_bound < 0.0:
                raise ValueError(
                    "linear_attn_config.gate_lower_bound must be in [-20, 0), "
                    f"got {self.gate_lower_bound}"
                )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        qkv = self.qkv_proj(hidden_states)[0]
        q, k, v = qkv.chunk(3, dim=-1)

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        if self.g_proj is not None:
            g_proj_states = self.g_proj(hidden_states)[0]
        else:
            assert self.g_a_proj is not None
            assert self.g_b_proj is not None
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        if spec_sequence_masks is not None:
            if (
                attn_metadata_narrowed.num_prefills == 0
                and attn_metadata_narrowed.num_decodes == 0
            ):
                q_spec, k_spec, v_spec = q_proj_states, k_proj_states, v_proj_states
                g1_spec, beta_spec = g1, beta
                q_non_spec = k_non_spec = v_non_spec = None
                g1_non_spec = beta_non_spec = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                q_spec = q_proj_states.index_select(0, spec_token_indx)
                k_spec = k_proj_states.index_select(0, spec_token_indx)
                v_spec = v_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                q_non_spec = q_proj_states.index_select(0, non_spec_token_indx)
                k_non_spec = k_proj_states.index_select(0, non_spec_token_indx)
                v_non_spec = v_proj_states.index_select(0, non_spec_token_indx)
                g1_non_spec = g1.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            q_spec = k_spec = v_spec = None
            g1_spec = beta_spec = None
            q_non_spec, k_non_spec, v_non_spec = (
                q_proj_states,
                k_proj_states,
                v_proj_states,
            )
            g1_non_spec, beta_non_spec = g1, beta

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            spec_conv_indices = spec_state_indices_tensor[
                : attn_metadata_narrowed.num_spec_decodes, 0
            ]
            spec_conv_kwargs = {
                "conv_state_indices": spec_conv_indices,
                "num_accepted_tokens": num_accepted_tokens,
                "query_start_loc": spec_query_start_loc,
                "max_query_len": spec_state_indices_tensor.size(-1),
                "validate_data": False,
            }
            q_spec = causal_conv1d_update(
                q_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                **spec_conv_kwargs,
            )
            k_spec = causal_conv1d_update(
                k_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                **spec_conv_kwargs,
            )
            v_spec = causal_conv1d_update(
                v_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                **spec_conv_kwargs,
            )

        if attn_metadata_narrowed.num_prefills > 0:
            assert q_non_spec is not None
            assert k_non_spec is not None
            assert v_non_spec is not None
            q = causal_conv1d_fn(
                q_non_spec.transpose(0, 1),
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_non_spec.transpose(0, 1),
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_non_spec.transpose(0, 1),
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert q_non_spec is not None
            assert k_non_spec is not None
            assert v_non_spec is not None
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_actual_tokens
            ]
            q = causal_conv1d_update(
                q_non_spec,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_non_spec,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_non_spec,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
        else:
            q = k = v = None

        if q is not None:
            q, k, v = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q, k, v),
            )

        if q_spec is not None:
            q_spec, k_spec, v_spec = map(
                lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim),
                (q_spec, k_spec, v_spec),
            )
            assert g1_spec is not None
            assert beta_spec is not None
            g1_spec = fused_kda_gate(
                rearrange(g1_spec, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
            ).unsqueeze(0)
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                g=g1_spec,
                beta=beta_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=spec_query_start_loc,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
            )
        else:
            core_attn_out_spec = None

        if attn_metadata_narrowed.num_prefills > 0:
            assert q is not None and k is not None and v is not None
            assert g1_non_spec is not None and beta_non_spec is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=g1_non_spec,
                beta=beta_non_spec,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                lower_bound=self.gate_lower_bound,
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        elif attn_metadata_narrowed.num_decodes > 0:
            assert q is not None and k is not None and v is not None
            assert g1_non_spec is not None and beta_non_spec is not None
            assert non_spec_state_indices_tensor is not None
            assert non_spec_query_start_loc is not None
            g1 = fused_kda_gate(
                rearrange(g1_non_spec, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
            ).unsqueeze(0)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta_non_spec,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_non_spec = None

        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            core_attn_out[0, spec_token_indx] = core_attn_out_spec[0]
            core_attn_out[0, non_spec_token_indx] = core_attn_out_non_spec[0]
        elif core_attn_out_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_spec[0]
        else:
            assert core_attn_out_non_spec is not None
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[0]
