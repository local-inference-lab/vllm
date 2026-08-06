# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniMax M3 (text backbone) model.

The MiniMax-M3-preview config selects a single set of branches:
    * qk_norm_type == "per_head"
    * hidden_act == "swigluoai"
    * use_gemma_norm == True  -> Gemma-style RMSNorm everywhere
    * attention_output_gate == False
    * scoring_func == "sigmoid" with a routing-bias correction term
    * sparse_attention_config present -> a subset of layers run the extra
      "index" attention branch.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.virtual_tp import VIRTUAL_TP_PLAN_ATTR
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MinimaxM3IndexerQKParallelLinear,
    MinimaxM3QKVParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.models.minimax_m3.common.indexer import MiniMaxM3Indexer
from vllm.models.minimax_m3.common.mm_preprocess import (
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    select_main_impl_cls,
)
from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)


def _resolve_quant_algo(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> str | None:
    resolver = getattr(quant_config, "_resolve_quant_algo", None)
    if not callable(resolver):
        return None
    return resolver(prefix)


def _should_split_mxfp8_indexer_projection(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> bool:
    """Use a BF16 indexer projection for mixed checkpoints.

    MiniMax M3 mixed MXFP8 checkpoints quantize the main q/k/v projection but
    leave the sparse indexer q/k projections in BF16. A single fused projection
    cannot represent that mixed dtype faithfully.
    """
    if _resolve_quant_algo(quant_config, f"{prefix}.qkv_proj") != "MXFP8":
        return False
    index_q_algo = _resolve_quant_algo(quant_config, f"{prefix}.indexer.q_proj")
    index_k_algo = _resolve_quant_algo(quant_config, f"{prefix}.indexer.k_proj")
    return index_q_algo is None and index_k_algo is None


def _enable_minimax_m3_torch_compile(vllm_config: VllmConfig) -> bool:
    del vllm_config
    return (
        envs.VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE
        and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
    )


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    """Layer ids whose attention runs the extra sparse "index" branch."""
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether this layer's MLP is a sparse MoE block (vs a dense MLP)."""
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _get_minimax_m3_virtual_tp_plan(config: PretrainedConfig):
    plan = getattr(config, VIRTUAL_TP_PLAN_ATTR, None)
    if isinstance(plan, dict) and plan.get("model_type") == "minimax_m3":
        return plan
    return None


def _get_minimax_m3_virtual_tp_axis_local_size(
    config: PretrainedConfig,
    axis_name: str,
) -> int:
    plan = _get_minimax_m3_virtual_tp_plan(config)
    if plan is None:
        raise ValueError("MiniMax M3 virtual TP plan is not active.")
    axis = plan.get(axis_name)
    if not isinstance(axis, dict):
        raise ValueError(f"MiniMax M3 virtual TP plan missing {axis_name!r}.")
    return int(axis["local_size"])


def _get_minimax_m3_local_attention_heads(
    config: PretrainedConfig,
    tp_size: int,
) -> tuple[int, int]:
    if _get_minimax_m3_virtual_tp_plan(config) is not None:
        return (
            _get_minimax_m3_virtual_tp_axis_local_size(config, "attention_heads"),
            _get_minimax_m3_virtual_tp_axis_local_size(config, "kv_heads"),
        )

    total_num_heads = config.num_attention_heads
    assert total_num_heads % tp_size == 0
    num_heads = total_num_heads // tp_size
    total_num_kv_heads = config.num_key_value_heads
    if total_num_kv_heads >= tp_size:
        assert total_num_kv_heads % tp_size == 0
    else:
        assert tp_size % total_num_kv_heads == 0
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    return num_heads, num_kv_heads


def minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


def minimax_m3_qknorm_rope_kv_insert_fake(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_qknorm_rope_kv_insert",
    op_func=minimax_m3_qknorm_rope_kv_insert,
    mutates_args=["qkv", "kv_cache", "index_cache", "q_out", "index_q_out"],
    fake_impl=minimax_m3_qknorm_rope_kv_insert_fake,
)


def _run_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    if torch.compiler.is_compiling():
        torch.ops.vllm.minimax_m3_qknorm_rope_kv_insert(
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            num_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            index_q_norm_weight,
            index_k_norm_weight,
            num_index_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            kv_cache_dtype,
        )
    else:
        minimax_m3_qknorm_rope_kv_insert(
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            num_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            index_q_norm_weight,
            index_k_norm_weight,
            num_index_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            kv_cache_dtype,
        )


class MiniMAXGemmaRMSNorm(GemmaRMSNorm):
    """MiniMax M3 Gemma-style RMSNorm.

    Keep the local class name for checkpoint/module compatibility while using
    vLLM's GemmaRMSNorm custom-op path, which is safe to trace through
    torch.compile.
    """


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the leading dense layers)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        # gate * sigmoid(alpha * gate) * (up + beta), with both halves clamped.
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE block with a routing-bias correction and a shared
    expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        # Sigmoid routing uses a per-expert score-correction bias for selection.
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        # Router weights are stored in fp32; GateLinear upcasts the bf16
        # activations and computes the gate in fp32 (fp32 router logits).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: MiniMaxM3MLP | None = None
        if self.n_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 (gate_up_proj) is loaded packed via MergedColumnParallelLinear
            # ([all gates; all ups]), so use the uninterleaved SwiGLU-OAI variant
            # rather than the interleaved gpt-oss layout.
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=config.swiglu_beta,
            routed_scaling_factor=self.routed_scaling_factor,
            router_logits_dtype=self.gate.out_dtype,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.experts",
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); GateLinear casts to fp32.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    """Dense attention with per-head QK norm and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads, self.num_kv_heads = _get_minimax_m3_local_attention_heads(
            config, tp_size
        )
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        qkv_linear_cls = (
            MinimaxM3QKVParallelLinear
            if _get_minimax_m3_virtual_tp_plan(config) is not None
            else QKVParallelLinear
        )
        self.qkv_proj = qkv_linear_cls(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # Fused per-head Gemma QK-norm + partial NeoX RoPE on q/k, in place.

        _run_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer branch.

    This is a merged attention layer: it owns the projections (qkv + index
    q/k), per-head QK norms and RoPE, *and* the attention-backend wiring that a
    generic ``Attention`` layer would normally provide — it binds the
    ``MiniMaxM3SparseBackend`` + main impl, registers the main paged K/V cache,
    and owns the lightning indexer (``MiniMaxM3Indexer``), which holds the
    index-key side cache.

    The index branch (index_{q,k}_proj + index_{q,k}_norm) feeds the sparse
    top-k block selection. M3 always disables the index value/output
    projections (``sparse_disable_index_value`` set for every sparse layer), so
    ``index_{v,o}_proj`` are never created.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads, self.num_kv_heads = _get_minimax_m3_local_attention_heads(
            config, tp_size
        )
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Sparse "index" branch dims. index_q has the same head count as the KV
        # heads (sparse_num_index_heads == num_key_value_heads), so it shards
        # identically -- including replication when tp_size > num_key_value_heads.
        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = (
            _get_minimax_m3_virtual_tp_axis_local_size(config, "index_heads")
            if _get_minimax_m3_virtual_tp_plan(config) is not None
            else self.num_kv_heads
        )
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim

        self.split_indexer_projection = _should_split_mxfp8_indexer_projection(
            quant_config, prefix
        )
        if self.split_indexer_projection:
            qkv_linear_cls = (
                MinimaxM3QKVParallelLinear
                if _get_minimax_m3_virtual_tp_plan(config) is not None
                else QKVParallelLinear
            )
            self.qkv_proj = qkv_linear_cls(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_qk_proj = MinimaxM3IndexerQKParallelLinear(
                self.hidden_size,
                self.total_num_kv_heads,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.indexer_qk_proj",
            )
        else:
            # Single fused projection: q, k, v, index_q, index_k in one GEMM.
            self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_qk_proj = None
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_rotary_emb = self.rotary_emb

        # Attention-backend wiring.
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # MiniMax-M3 sparse attention owns its KV-cache insert/read path instead
        # of wrapping the generic Attention module. Keep the same runtime scale
        # attributes so FP8 KV reads can honor vLLM's per-layer descale contract.
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)
        # Indexer side-cache dtype, mirroring --kv-cache-dtype for the main
        # cache (--attention-config '{"indexer_kv_dtype": ...}').
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype

        # Shared top-k buffer: the indexer writes the selected blocks into it and
        # the attend impl reads them back (so nothing crosses the eager break as a
        # Python value, which would freeze at capture).
        self.topk_indices_buffer = topk_indices_buffer
        self.attn_backend = MiniMaxM3SparseBackend
        # Indexer (top-k selection) and main attention are separate impls, each
        # picking Triton vs MSA off its cache dtype. impl is AttentionImplBase
        # (broader than the AttentionImpl that AttentionLayerBase annotates).
        self.impl: MiniMaxM3SparseImpl = select_main_impl_cls(  # type: ignore[assignment]
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            kv_cache_dtype=self.kv_cache_dtype,
            num_kv_heads=self.num_kv_heads,
        )(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        # Self-contained nn.Module: owns its side cache, selects its impl in init.
        self.indexer = MiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            cache_config=cache_config,
            indexer_kv_dtype=self.indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

    def get_attn_backend(self) -> type[MiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Main GQA K/V cache. Block size may change after load, refresh it.
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _insert_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
        main_slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor,
    ) -> None:
        """Write main K/V and index-K into their paged caches."""
        self._insert_kv_into_caches(
            key,
            value,
            index_key,
            self.kv_cache,
            self.indexer.index_cache.kv_cache,
            main_slot_mapping,
            index_slot_mapping,
        )

    def _insert_kv_into_caches(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
        main_kv_cache: torch.Tensor,
        index_kv_cache: torch.Tensor,
        main_slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor,
    ) -> None:
        """Write main K/V and index-K into explicit paged cache tensors."""
        key_cache, value_cache = main_kv_cache.unbind(1)
        scale = torch.ones((), device=key.device)
        ops.reshape_and_cache_flash(
            key.view(-1, self.num_kv_heads, self.head_dim),
            value.view(-1, self.num_kv_heads, self.head_dim),
            key_cache,
            value_cache,
            main_slot_mapping,
            self.kv_cache_dtype,
            scale,
            scale,
        )

        # Index-key cache: single vector per token, scatter by slot.
        idx_cache = index_kv_cache.view(-1, self.idx_head_dim)
        idx_cache[index_slot_mapping] = index_key.to(idx_cache.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Projection result consumed by the fused norm/rope/cache-insert op as
        # [q | k | v | index_q | index_k].
        qkv, _ = self.qkv_proj(hidden_states)
        if self.indexer_qk_proj is not None:
            index_qk, _ = self.indexer_qk_proj(hidden_states)
            qkv = torch.cat((qkv, index_qk), dim=-1)

        # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on the
        # main (q/k) and index (index_q/index_k) branches, all read straight out
        # of the single fused ``qkv`` tensor (the "5 results").  Once the paged
        # caches are bound the kernel also inserts k/v and the index key into
        # them; the initial memory-profiling run (caches unbound, no slot_mapping)
        # short-circuits to zeros below. k/v and index_k are rewritten in place
        # inside qkv (and scatter-inserted into the caches); q and index_q are
        # de-interleaved
        # straight into the dedicated contiguous ``q``/``index_q`` buffers below.

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon
        num_tokens = qkv.shape[0]

        encoded_layer_name = _encode_layer_name(self.layer_name)
        main_slot_mapping = None
        index_slot_mapping = None
        # Slot mappings are live scheduler metadata. Keep the compiled path
        # from reading them in Python; the cache-update custom op resolves
        # them at runtime from the forward context.
        if not torch.compiler.is_compiling():
            fwd_slot_mapping = get_forward_context().slot_mapping
            if (
                not isinstance(fwd_slot_mapping, dict)
                or self.layer_name not in fwd_slot_mapping
            ):
                # Memory-profiling run: caches not yet bound, slot_mapping is
                # empty.
                return qkv.new_zeros((num_tokens, self.hidden_size))

            main_slot_mapping = fwd_slot_mapping[self.layer_name]
            index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]
        q = qkv.new_empty((num_tokens, self.q_size))
        # index_q matches the index-K cache dtype (e4m3 for the fp8 score path);
        # the fused kernel emits fp8 directly when this buffer is e4m3.
        index_q = qkv.new_empty(
            (num_tokens, self.index_q_size),
            dtype=self.indexer.index_cache.dtype,
        )
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            rotary_dim,
            eps,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            main_slot_mapping,
            index_slot_mapping,
            self.kv_cache if insert_via_fused else None,
            self.indexer.index_cache.kv_cache if insert_via_fused else None,
            self.kv_cache.size(2) if insert_via_fused else 0,
            q,
            index_q,
            self.kv_cache_dtype,
        )
        if torch.compiler.is_compiling() or not insert_via_fused:
            kv = self.num_kv_heads * self.head_dim
            k = qkv[:, self.q_size : self.q_size + kv]
            v = qkv[:, self.q_size + kv : self.q_size + 2 * kv]
            ik0 = self.q_size + 2 * kv + self.index_q_size
            index_k = qkv[:, ik0 : ik0 + self.num_idx_heads * self.idx_head_dim]
            if torch.compiler.is_compiling():
                kv_cache_dummy_dep = torch.ops.vllm.minimax_m3_sparse_kv_cache_update(
                    k,
                    v,
                    index_k,
                    encoded_layer_name,
                )
            else:
                assert main_slot_mapping is not None
                assert index_slot_mapping is not None
                self._insert_kv(k, v, index_k, main_slot_mapping, index_slot_mapping)

        output = torch.empty_like(q)
        if torch.compiler.is_compiling():
            torch.ops.vllm.minimax_m3_sparse_attention_with_output(
                q,
                index_q,
                self.kv_cache,
                self.indexer.index_cache.kv_cache,
                output,
                encoded_layer_name,
                kv_cache_dummy_dep,
            )
            attn_output = output
        else:
            attn_output = self._run_attention(q, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

    @eager_break_during_capture
    def _run_attention(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Single eager break around both: their split-K kernels read per-request
        # metadata and can't be captured into a cudagraph. The indexer writes its
        # top-k into the shared ``topk_indices_buffer``; the attend reads it back.
        self.indexer(index_query)
        return self.impl.forward(self, query, self.kv_cache, output)


@eager_break_during_capture
def minimax_m3_sparse_attention_with_output(
    query: torch.Tensor,
    index_query: torch.Tensor,
    main_kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    assert isinstance(layer, MiniMaxM3SparseAttention)
    topk_idx = layer.indexer.forward_with_cache(index_query, index_kv_cache)
    layer.impl.forward(layer, query, main_kv_cache, topk_idx, output)


def minimax_m3_sparse_attention_with_output_fake(
    query: torch.Tensor,
    index_query: torch.Tensor,
    main_kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_sparse_attention_with_output",
    op_func=minimax_m3_sparse_attention_with_output,
    mutates_args=["output"],
    fake_impl=minimax_m3_sparse_attention_with_output_fake,
)


def minimax_m3_sparse_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MiniMaxM3SparseAttention)
    slot_mapping = forward_context.slot_mapping
    if isinstance(slot_mapping, dict) and layer_name in slot_mapping:
        layer._insert_kv_into_caches(
            key,
            value,
            index_key,
            layer.kv_cache,
            layer.indexer.index_cache.kv_cache,
            slot_mapping[layer_name],
            slot_mapping[layer.indexer.index_cache.prefix],
        )
    return torch.empty(0, device=layer.kv_cache.device, dtype=layer.kv_cache.dtype)


def minimax_m3_sparse_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


direct_register_custom_op(
    op_name="minimax_m3_sparse_kv_cache_update",
    op_func=minimax_m3_sparse_kv_cache_update,
    mutates_args=[],
    fake_impl=minimax_m3_sparse_kv_cache_update_fake,
)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        force_sparse_attn: bool = False,
        force_moe: bool = False,
        is_mtp_block: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if is_mtp_block:
            assert vllm_config.speculative_config is not None
            config = vllm_config.speculative_config.draft_model_config.hf_config
        else:
            config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

        # When set, complete the preceding layer's deferred FFN all-reduce
        # fused into this layer's input_layernorm.
        # Configured by MiniMaxM3Model.__init__
        self.fuse_input_allreduce = False

        is_sparse_attention_layer = (
            force_sparse_attn or layer_id in _sparse_attention_layer_ids(config)
        )

        if is_sparse_attention_layer:
            self.self_attn = MiniMaxM3SparseAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            self.self_attn = MiniMaxM3Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        # Dense layers store the FFN under `mlp`; MoE layers under
        # `block_sparse_moe` -- matching the checkpoint's naming.
        # Leave the FFN output un-reduced so its all-reduce fuses into the
        # next RMSNorm. MTP blocks add the residual directly and PP sends
        # hidden states across stages, so both must reduce.
        reduce_results = (
            is_mtp_block or vllm_config.parallel_config.pipeline_parallel_size > 1
        )
        self.is_moe_layer = force_moe or _is_moe_layer(config, layer_id)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=reduce_results,
            )

        # config.use_gemma_norm is True for M3 -> Gemma-style RMSNorm.
        self.input_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_input_allreduce and residual is not None:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual

    @property
    def ffn_all_reduce_deferred(self) -> bool:
        """This layer's FFN output is left un-reduced; the caller fuses the
        all-reduce into the next RMSNorm."""
        if self.is_moe_layer:
            return self.block_sparse_moe.experts.moe_config.skip_final_all_reduce
        return not self.mlp.down_proj.reduce_results


@support_torch_compile(enable_if=_enable_minimax_m3_torch_compile)
class MiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Reserved top-k indices buffer shared by all sparse-attention indexer
        # layers (mirrors DeepseekV4); kept at a stable address so the indexer's
        # top-k output survives cudagraph capture/replay. Token-major
        # [total_q, num_index_heads, topk] so the indexer writes its native
        # [token, head, topk] top-k; the attend transposes to [H, tokens, topk].
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if sparse_cfg is not None:
            tp_size = get_tensor_model_parallel_world_size()
            num_index_heads = max(1, sparse_cfg["sparse_num_index_heads"] // tp_size)
            # Pad tokens to a multiple of 4 so the buffer head stride stays
            # int4-aligned for build_k2q_csr's vectorised int4 loads.
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            padded_num_tokens = (max_num_batched_tokens + 3) // 4 * 4
            self.topk_indices_buffer = torch.empty(
                padded_num_tokens,
                num_index_heads,
                sparse_cfg["sparse_topk_blocks"],
                dtype=torch.int32,
            )
        else:
            self.topk_indices_buffer = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Configure cross-layer all-reduce/RMSNorm fusion: a layer whose FFN output
        # is left un-reduced has that all-reduce fused into the next layer's
        # input_layernorm (or the final norm).
        prev_defers = False
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer.fuse_input_allreduce = idx > 0 and prev_defers
            prev_defers = layer.ffn_all_reduce_deferred
        self.fuse_final_norm_allreduce = prev_defers

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # EAGLE3 is not yet compatible with pipeline parallel
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.fuse_final_norm_allreduce:
            hidden_states, _ = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.norm
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Checkpoint experts use gate_proj=gate, down_proj=down, up_proj=up.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # q/k/v_proj -> fused qkv_proj; gate_proj/up_proj -> fused gate_up_proj
        # (dense MLP and shared expert). On sparse layers the indexer
        # self_attn.indexer.{q,k}_proj either fold into the same fused qkv_proj
        # or load into a separate BF16 indexer_qk_proj for mixed MXFP8 checkpoints.
        # These entries simply never match on dense layers, whose checkpoints
        # have no indexer projection weights.
        # Keep the indexer-specific keys before the generic q/k mappings.
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".indexer_qk_proj", ".indexer.q_proj", "index_q"),
            (".indexer_qk_proj", ".indexer.k_proj", "index_k"),
            (".indexer_qk_proj", ".index_q_proj", "index_q"),
            (".indexer_qk_proj", ".index_k_proj", "index_k"),
            (".qkv_proj", ".indexer.q_proj", "index_q"),
            (".qkv_proj", ".indexer.k_proj", "index_k"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # The MTP module is not modeled yet.
            if "mtp." in name:
                continue

            if ".self_attn.indexer.q_norm." in name:
                name = name.replace(
                    ".self_attn.indexer.q_norm.",
                    ".self_attn.index_q_norm.",
                )
            elif ".self_attn.indexer.k_norm." in name:
                name = name.replace(
                    ".self_attn.indexer.k_norm.",
                    ".self_attn.index_k_norm.",
                )

            # The checkpoint stores block scales as ``weight_scale_inv``; the
            # ModelOpt MXFP8 layers expose them as ``weight_scale``.
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                # Routed experts (w1/w2/w3) are handled below; don't let the
                # stacked mapping rewrite them.
                if (
                    "block_sparse_moe.experts." in mapped_name
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Modules not modeled yet (e.g. attention) are skipped until
                    # they are ported.
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMaxM3SparseForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    """MiniMax M3 (sparse/dense backbone) for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    """Top-level (VL) entry point for MiniMax M3.

    The vision tower is not modeled yet; this wrapper routes the text
    backbone by constructing ``MiniMaxM3SparseForCausalLM`` from the nested
    ``text_config`` and delegating generation to it.
    """

    # The vision tower runs replicated per rank under ``--mm-encoder-tp-mode
    # data``; ``run_dp_sharded_mrope_vision_model`` shards the work across
    # ranks (see ``_process_image_input`` / ``_process_video_input``).
    supports_encoder_tp_data = True
    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.embeddings.proj.": (
                "vision_tower.vision_model.embeddings.patch_embedding."
            ),
            "model.vision_tower.embeddings.": ("vision_tower.vision_model.embeddings."),
            "model.vision_tower.pre_layrnorm.": (
                "vision_tower.vision_model.pre_layrnorm."
            ),
            "model.vision_tower.layers.": "vision_tower.vision_model.encoder.layers.",
            "model.multi_modal_projector.merge_linear_1.": (
                "vision_tower.patch_merge_mlp.linear_1."
            ),
            "model.multi_modal_projector.merge_linear_2.": (
                "vision_tower.patch_merge_mlp.linear_2."
            ),
            "model.multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.gate.": ".block_sparse_moe.gate.",
            ".mlp.experts.": ".block_sparse_moe.experts.",
            ".mlp.shared_experts.": ".block_sparse_moe.shared_experts.",
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        assert self.multimodal_config is not None
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_config.hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            vision_config = config.vision_config
            self.vision_tower = MiniMaxVLVisionModel(
                config=PretrainedConfig.from_dict(vision_config),
                text_hidden_size=text_hidden_size,
                projector_hidden_size=projector_hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["MiniMaxM3SparseForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    # Expose language model / lm_head for EAGLE3 spec decode.
    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = image_input["pixel_values"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            # Already returns a per-item tuple of embeddings.
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        image_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )

        # Split the concatenated output into one tensor per image item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = video_input["pixel_values_videos"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            # Already returns a per-item tuple of embeddings.
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        video_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )

        # Split the concatenated output into one tensor per video item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings.extend(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings.extend(video_embeddings)

        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
