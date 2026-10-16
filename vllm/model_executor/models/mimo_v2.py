# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from copy import copy
from itertools import islice
from typing import ClassVar

import torch
from torch import nn

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
    str_dtype_to_torch_dtype,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import (
    declare_b12x_fused_allreduce_rms_norm_sites,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_quantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.model_executor.parameter import copy_tensor_parallel_shard
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import materialize_weight
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# TP3 pad scheme for the RL/MOPD checkpoints. Their QKV weights are fp8 with
# 128-row scale blocks and pre-sharded into checkpoint_tp_size=4 [Q|K|V]
# groups, so a rank's Q, K and V rows must all be whole 128-row blocks: an
# even number of KV heads per rank (192-row K heads), and Q heads per rank
# keeping the checkpoint's GQA ratio. Ranks 0 and 1 take whole checkpoint
# groups; the last rank's heads are zero weights and contribute nothing.
TP3 = 3
CHECKPOINT_TP3_GROUPS = 4


def _tp3_active(tp_size: int) -> bool:
    return tp_size == TP3


def _checkpoint_config_value(config, name):
    return getattr(config, f"original_{name}", getattr(config, name))


def _padded_head_counts(
    num_heads: int, num_kv_heads: int, tp_size: int
) -> tuple[int, int]:
    """Head totals padded for TP3; unchanged where heads already divide."""
    if not _tp3_active(tp_size) or (
        num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
    ):
        return num_heads, num_kv_heads
    q_per_kv = num_heads // num_kv_heads
    kv_per_rank = -(-num_kv_heads // tp_size)
    if kv_per_rank % 2:
        # An odd count leaves a rank's K rows off the 128-row block grid.
        kv_per_rank += 1
    q_per_rank = q_per_kv * kv_per_rank
    return q_per_rank * tp_size, kv_per_rank * tp_size


def _padded_fused_chunks(checkpoint_chunks: int, tp_size: int) -> int:
    """Fused-QKV chunk count whose per-rank share is whole chunks."""
    if not _tp3_active(tp_size) or checkpoint_chunks % tp_size == 0:
        return checkpoint_chunks
    return -(-checkpoint_chunks // tp_size) * tp_size


def _padded_expert_intermediate(intermediate_size: int, tp_size: int) -> int:
    """Expert width padded so every rank's shard keeps whole fp8 blocks."""
    if not _tp3_active(tp_size) or intermediate_size % tp_size == 0:
        return intermediate_size
    from math import lcm

    step = lcm(tp_size, 128)
    return -(-intermediate_size // step) * step


def _pad_tensor_to_param(loaded: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Zero-pad a checkpoint tensor out to its padded parameter shape."""
    pad = [p - s for p, s in zip(param.shape, loaded.shape) if p != s]
    if not pad:
        return loaded
    if any(p < s for p, s in zip(param.shape, loaded.shape)) or len(pad) > 1:
        raise ValueError(
            f"Cannot pad checkpoint tensor {tuple(loaded.shape)} to parameter "
            f"{tuple(param.shape)}"
        )
    padded = loaded.new_zeros(param.shape)
    slices = tuple(slice(0, s) for s in loaded.shape)
    padded[slices] = loaded
    return padded


def _pad_stacked_qkv(
    loaded: torch.Tensor, name: str, shard_id: str, attn: "MiMoV2Attention"
) -> torch.Tensor:
    """Zero-pad a separate q/k/v checkpoint tensor to the padded head count."""
    is_scale = name.endswith("weight_scale_inv")
    if shard_id == "q":
        gap_heads = attn.total_num_heads - attn.checkpoint_num_heads
        head_dim = attn.head_dim
    elif shard_id == "k":
        gap_heads = attn.total_num_kv_heads - attn.checkpoint_num_kv_heads
        head_dim = attn.head_dim
    else:
        gap_heads = attn.total_num_kv_heads - attn.checkpoint_num_kv_heads
        head_dim = attn.v_head_dim
    gap = gap_heads * head_dim
    if is_scale:
        gap //= 128
    if not gap:
        return loaded
    shape = list(loaded.shape)
    shape[0] = gap
    return torch.cat([loaded, loaded.new_zeros(shape)], dim=0)


def _pad_expert_tensor(
    loaded: torch.Tensor, name: str, shard_id: str, padded_intermediate: int
) -> torch.Tensor:
    """Zero-pad one expert tensor to the padded expert width.

    The MOPD/RL checkpoints store MXFP4 experts packed two values per byte
    along the input dim, one e8m0 scale per 32-column block: gate/up are
    [out, in/2] with scale [out, in/32] and shard on rows (the intermediate
    dim); down is [hidden, inter/2] with scale [hidden, inter/32] and shards
    on the packed input columns.
    """
    is_scale = name.endswith("weight_scale")
    if shard_id == "w2":
        dim = 1
        target = padded_intermediate // 32 if is_scale else padded_intermediate // 2
    else:
        dim = 0
        target = padded_intermediate
    gap = target - loaded.shape[dim]
    if gap <= 0:
        return loaded
    pad_shape = list(loaded.shape)
    pad_shape[dim] = gap
    return torch.cat([loaded, loaded.new_zeros(pad_shape)], dim=dim)


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        padded_intermediate = _padded_expert_intermediate(
            intermediate_size, get_tensor_model_parallel_world_size()
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [padded_intermediate] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            padded_intermediate,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if padded_intermediate != intermediate_size:
            for param in self.parameters():
                set_weight_attrs(param, {"allow_tp_padding": True})
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiMoV2MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        dtype = getattr(config, "moe_router_dtype", "float32")
        self.gate_dtype = str_dtype_to_torch_dtype(dtype)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            params_dtype=self.gate_dtype,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=self.gate.out_dtype)
        )

        padded_intermediate = _padded_expert_intermediate(
            config.moe_intermediate_size, self.tp_size
        )
        self.padded_intermediate = padded_intermediate
        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=padded_intermediate,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            router_logits_dtype=self.gate.out_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "MiMoV2MoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.gate_dtype is not None:
            gate_input = hidden_states.to(self.gate_dtype)
        else:
            gate_input = hidden_states
        router_logits, _ = self.gate(gate_input)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        v_scale: float | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        partial_rotary_factor: float = 1.0,
        fused_qkv_chunks: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.checkpoint_num_heads = num_heads
        self.checkpoint_num_kv_heads = num_kv_heads
        self.total_num_heads, self.total_num_kv_heads = _padded_head_counts(
            num_heads, num_kv_heads, tp_size
        )
        self.num_heads = self.total_num_heads // tp_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim

        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = v_scale
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # A fused-QKV checkpoint stores ``fused_qkv_chunks`` [Q | K | V]
        # chunks; see _fused_qkv_kv_chunk_rows for the rank layout.
        self.kv_chunk_rows = _fused_qkv_kv_chunk_rows(
            num_heads,
            num_kv_heads,
            head_dim,
            self.v_head_dim,
            tp_size,
            fused_qkv_chunks,
        )
        if self.kv_chunk_rows:
            self.qkv_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    _padded_fused_chunks(fused_qkv_chunks, tp_size)
                    * self.kv_chunk_rows,
                ],
                bias=attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
                v_head_size=self.v_head_dim,
            )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config if "mtp.layers" not in prefix else None,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        if self.total_num_heads != num_heads:
            for param in self.o_proj.parameters():
                set_weight_attrs(param, {"allow_tp_padding": True})

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > -1 else None
        if (
            sliding_window is None
            and cache_config is not None
            and cache_config.sliding_window is not None
        ):
            # MiMo declares each layer's window explicitly. Do not let the
            # model-wide SWA default turn a global layer into sliding attention.
            cache_config = copy(cache_config)
            cache_config.sliding_window = None

        # Honor B12X and DiffKV selections for unequal QK/V head dimensions.
        if self.v_head_dim != self.head_dim:
            requested = get_current_vllm_config().attention_config.backend
            if requested is not None and (
                requested == AttentionBackendEnum.B12X
                or requested.name.endswith("_DIFFKV")
            ):
                backend_enum = requested
            else:
                fa_backend = AttentionBackendEnum.FLASH_ATTN_DIFFKV.get_class()
                assert hasattr(fa_backend, "is_supported_on_current_device")
                if fa_backend.is_supported_on_current_device(
                    head_size=self.head_dim,
                    head_size_v=self.v_head_dim,
                    has_sinks=self.attention_sink_bias is not None,
                ):
                    backend_enum = AttentionBackendEnum.FLASH_ATTN_DIFFKV
                else:
                    backend_enum = AttentionBackendEnum.TRITON_ATTN_DIFFKV
            attn_backend = backend_enum.get_class()
            if backend_enum != AttentionBackendEnum.B12X:
                assert hasattr(attn_backend, "set_head_size_v")
                attn_backend.set_head_size_v(self.v_head_dim)
            logger.info_once("Using %s for attention.", attn_backend.get_name())
        else:
            attn_backend = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
            attn_backend=attn_backend,
            head_size_v=self.v_head_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = _split_qkv(
            qkv, self.q_size, self.k_size, self.v_size, self.kv_chunk_rows
        )
        q, k = self.rotary_emb(positions, q, k)

        # Apply v_scale before attention
        if self.v_scale is not None:
            v = v * self.v_scale

        attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        v_scale = getattr(config, "attention_value_scale", None)
        fused_qkv_chunks = (
            _checkpoint_config_value(config, "num_key_value_heads")
            if getattr(config, "attention_projection_layout", None) == "fused_qkv"
            and getattr(quant_config, "weight_block_size", None)
            else 0
        )

        if self.is_compressed_softmax_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=_checkpoint_config_value(config, "swa_num_attention_heads"),
                num_kv_heads=_checkpoint_config_value(
                    config, "swa_num_key_value_heads"
                ),
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                fused_qkv_chunks=fused_qkv_chunks,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=_checkpoint_config_value(config, "num_attention_heads"),
                num_kv_heads=_checkpoint_config_value(config, "num_key_value_heads"),
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                fused_qkv_chunks=fused_qkv_chunks,
                prefix=f"{prefix}.self_attn",
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        if self.is_layer_sparse:
            self.mlp = MiMoV2MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_compressed_softmax_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_id] == 1


def _fused_qkv_kv_chunk_rows(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_size: int,
    checkpoint_tp_size: int,
    block: int = 128,
) -> int:
    """Rows per checkpoint chunk of a rank's ``K | V`` fused-QKV section.

    Returns 0 when the de-interleaved ``[Q | K | V]`` rank layout is exact:
    a rank owns at most one checkpoint chunk, or every chunk's Q, K and V
    rows span whole FP8 scale blocks, so the layout is a block permutation.

    Otherwise, when a rank owns several chunks whose K and V rows share a
    scale block (MiMo global layers: 192 K + 128 V rows), the rank keeps each
    chunk's K and V rows together, zero-padded to whole blocks, so every row
    keeps its checkpoint scale:

        [Q_1 | ... | Q_g | K_1 V_1 pad | ... | K_g V_g pad]

    and this returns the padded ``K | V`` rows per chunk.
    """
    if checkpoint_tp_size <= tp_size:
        return 0
    if checkpoint_tp_size % tp_size and not (
        checkpoint_tp_size == CHECKPOINT_TP3_GROUPS and tp_size == TP3
    ):
        return 0
    q_rows = num_heads // checkpoint_tp_size * head_dim
    k_rows = num_kv_heads // checkpoint_tp_size * head_dim
    v_rows = num_kv_heads // checkpoint_tp_size * v_head_dim
    if q_rows % block or (k_rows % block == 0 and v_rows % block == 0):
        return 0
    return cdiv(k_rows + v_rows, block) * block


def _split_qkv(
    qkv: torch.Tensor, q_size: int, k_size: int, v_size: int, kv_chunk_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a rank's fused QKV output (see _fused_qkv_kv_chunk_rows)."""
    if not kv_chunk_rows:
        return qkv.split([q_size, k_size, v_size], dim=-1)
    q, kv = qkv.split([q_size, qkv.shape[-1] - q_size], dim=-1)
    kv = kv.unflatten(-1, (-1, kv_chunk_rows))
    k_rows, v_rows = k_size // kv.shape[-2], v_size // kv.shape[-2]
    k = kv[..., :k_rows].flatten(-2)
    v = kv[..., k_rows : k_rows + v_rows].flatten(-2)
    return q, k, v


def _shard_fp8_qkv_proj(
    w_full: torch.Tensor,
    s_full: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
    checkpoint_tp_size: int,
    block: int = 128,
    kv_chunk_rows: int = 0,
    padded_num_heads: int | None = None,
    padded_num_kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard the fp8 qkv_proj weights for ``tp_rank``.

    The checkpoint stores ``checkpoint_tp_size`` contiguous training shards,
    each ordered ``[Q | K | V]``. Sliding-window shards can contain multiple
    KV heads. The training TP size is the model's global-attention KV count:

        [Q_1 | K_1 | V_1 | Q_2 | K_2 | V_2 | ... | Q_n | K_n | V_n]

    Each TP rank owns ``g = checkpoint_tp_size / tp_size`` shards, and the
    forward expects them de-interleaved into a single Q, K, and V block:

        [Q_1 | Q_2 | ... | Q_g | K_1 | K_2 | ... | K_g | V_1 | V_2 | ... | V_g]

    When ``g == 1`` the rank's slice is already ``[Q | K | V]``, so a plain
    chunk suffices. When Q, K and V each span whole 128-row scale blocks, the
    layout above is a block permutation of the checkpoint and is exact. When
    a block straddles the K/V boundary (a 192-row K is 1.5 blocks), the rank
    uses the padded ``[Q_1 | ... | Q_g | K_1 V_1 pad | ... | K_g V_g pad]``
    layout of ``_fused_qkv_kv_chunk_rows`` instead (``kv_chunk_rows``), again
    exact. Only when neither applies do we dequantize this rank's groups to
    float, reorder the rows into the layout above, and re-quantize to fp8.
    """
    if checkpoint_tp_size == CHECKPOINT_TP3_GROUPS and tp_size == TP3:
        # Whole checkpoint groups on the first ranks; the last rank's heads
        # are zero weights (see _padded_head_counts).
        if tp_rank == tp_size - 1:
            padded_num_heads = padded_num_heads or num_heads
            padded_num_kv_heads = padded_num_kv_heads or num_kv_heads
            q_rows = padded_num_heads // tp_size * head_dim
            kv_heads_per_rank = padded_num_kv_heads // tp_size
            if kv_chunk_rows:
                rows = q_rows + kv_heads_per_rank * kv_chunk_rows
                scale_rows = cdiv(q_rows, block) + kv_heads_per_rank * cdiv(
                    kv_chunk_rows, block
                )
            else:
                k_rows = kv_heads_per_rank * head_dim
                v_rows = kv_heads_per_rank * v_head_dim
                rows = q_rows + k_rows + v_rows
                scale_rows = (
                    cdiv(q_rows, block) + cdiv(k_rows, block) + cdiv(v_rows, block)
                )
            return (
                torch.zeros(
                    rows, w_full.shape[1], dtype=w_full.dtype, device=w_full.device
                ),
                torch.zeros(
                    scale_rows,
                    s_full.shape[1],
                    dtype=s_full.dtype,
                    device=s_full.device,
                ),
            )
        return _shard_fp8_qkv_proj(
            w_full,
            s_full,
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            tp_rank,
            tp_size - 1,
            checkpoint_tp_size,
            block,
            kv_chunk_rows,
        )
    assert checkpoint_tp_size % tp_size == 0, (
        "TP size must evenly split the checkpoint TP size."
    )
    assert num_kv_heads % checkpoint_tp_size == 0

    groups_per_rank = checkpoint_tp_size // tp_size
    if groups_per_rank == 1:
        # One checkpoint shard per rank. Weights and scales can be sharded
        # without re-quantization.
        w = w_full.chunk(tp_size, dim=0)[tp_rank]
        s = s_full.chunk(tp_size, dim=0)[tp_rank]
        return w, s

    q_rows_per_group = (num_heads // checkpoint_tp_size) * head_dim
    k_rows_per_group = (num_kv_heads // checkpoint_tp_size) * head_dim
    v_rows_per_group = (num_kv_heads // checkpoint_tp_size) * v_head_dim
    rows_per_group = q_rows_per_group + k_rows_per_group + v_rows_per_group
    scale_rows_per_group = (rows_per_group + block - 1) // block
    assert w_full.shape[0] == rows_per_group * checkpoint_tp_size
    assert s_full.shape[0] == scale_rows_per_group * checkpoint_tp_size
    groups = range(tp_rank * groups_per_rank, (tp_rank + 1) * groups_per_rank)
    q_end = q_rows_per_group
    k_end = q_end + k_rows_per_group
    if kv_chunk_rows or all(
        n % block == 0 for n in (q_rows_per_group, k_rows_per_group, v_rows_per_group)
    ):
        # Exact: every row keeps its checkpoint FP8 value and block scale.
        def rows(g: int, start: int, stop: int) -> torch.Tensor:
            base = g * rows_per_group
            return materialize_weight(w_full[base + start : base + stop])

        def scales(g: int, start: int, stop: int) -> torch.Tensor:
            base = g * scale_rows_per_group
            return materialize_weight(
                s_full[base + start // block : base + cdiv(stop, block)]
            )

        if kv_chunk_rows:
            pad = kv_chunk_rows - (rows_per_group - q_end)
            weights = [rows(g, 0, q_end) for g in groups]
            for g in groups:
                kv = rows(g, q_end, rows_per_group).view(torch.uint8)
                zeros = kv.new_zeros((pad, kv.shape[1]))  # FP8 +0.0
                weights.append(torch.cat([kv, zeros]).view(w_full.dtype))
            scale_rows = [scales(g, 0, q_end) for g in groups]
            scale_rows += [scales(g, q_end, rows_per_group) for g in groups]
        else:
            segments = ((0, q_end), (q_end, k_end), (k_end, rows_per_group))
            weights = [rows(g, *seg) for seg in segments for g in groups]
            scale_rows = [scales(g, *seg) for seg in segments for g in groups]
        return (
            torch.cat([w.view(torch.uint8) for w in weights]).view(w_full.dtype),
            torch.cat(scale_rows),
        )
    qs, ks, vs = [], [], []
    for g_idx in groups:
        row_start = g_idx * rows_per_group
        scale_row_start = g_idx * scale_rows_per_group
        # Dequantize this group's weights.
        w_g = materialize_weight(w_full[row_start : row_start + rows_per_group]).to(
            torch.float32
        )
        s_g = materialize_weight(
            s_full[scale_row_start : scale_row_start + scale_rows_per_group]
        ).to(torch.float32)
        s_g_expanded = s_g.repeat_interleave(block, dim=0).repeat_interleave(
            block, dim=1
        )[:rows_per_group]
        w_g_dequant = w_g * s_g_expanded
        # Track the dequantized q, k, and v weights separately.
        qs.append(w_g_dequant[:q_rows_per_group])
        ks.append(w_g_dequant[q_rows_per_group : q_rows_per_group + k_rows_per_group])
        vs.append(w_g_dequant[q_rows_per_group + k_rows_per_group :])

    # Combine the q, k, and v weights into the following layout:
    # [Q_1, Q_2, .., Q_g, K_1, K_2, ..., K_g, V_1, V_2, ..., V_g]
    grouped = torch.cat([torch.cat(qs), torch.cat(ks), torch.cat(vs)], dim=0)
    # Quantize back to fp8.
    return scaled_quantize(
        grouped, GroupShape(block, block), w_full.dtype, compute_dtype=torch.float32
    )


_L2_PREFETCH_WINDOW_BYTES = 20 * 1000 * 1000


def _in_full_decode_capture() -> bool:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return torch.cuda.is_current_stream_capturing() and not (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def _install_l2_prefetch_windows(
    layer: "MiMoV2FlashDecoderLayer", nxt: "MiMoV2FlashDecoderLayer | None"
) -> None:
    """Prefetch upcoming decode weights into L2 while device memory idles.

    Window A (attention op): this layer's o_proj, router and post-attention
    norm. Window C (end of the MoE op, before its all-reduce): the next layer's
    input norm and qkv projection; the last layer rejoins the side stream.
    Prefetches are issued only inside FULL CUDA-graph decode captures and are
    cache hints: numerics are unchanged.
    """
    from vllm.models.glm5next.nvidia import l2_prefetch

    plans: dict[str, l2_prefetch.L2PrefetchPlan | None] = {}

    def build() -> None:
        device = layer.post_attention_layernorm.weight.device
        small = 16
        segments = l2_prefetch.segments_of(layer.self_attn.o_proj, "o_proj.")
        if layer.is_layer_sparse:
            segments += l2_prefetch.segments_of(
                layer.mlp.gate, "gate.", min_bytes=small
            )
        segments += l2_prefetch.segments_of(
            layer.post_attention_layernorm, "post_norm.", min_bytes=small
        )
        plans["a"] = l2_prefetch.make_plan(segments, _L2_PREFETCH_WINDOW_BYTES, device)[
            0
        ]
        plans["c"] = None
        if nxt is not None:
            segments = l2_prefetch.segments_of(
                nxt.input_layernorm, "next.input_norm.", min_bytes=small
            ) + l2_prefetch.segments_of(nxt.self_attn.qkv_proj, "next.qkv_proj.")
            plans["c"] = l2_prefetch.make_plan(
                segments, _L2_PREFETCH_WINDOW_BYTES, device
            )[0]
        l2_prefetch.L2Prefetcher.get(device)

    def window(name: str, final: bool):
        def hook(num_tokens: int) -> None:
            if not plans:
                if torch.cuda.is_current_stream_capturing():
                    return
                build()
            if not _in_full_decode_capture():
                return
            device = layer.post_attention_layernorm.weight.device
            prefetcher = l2_prefetch.L2Prefetcher.get(device)
            prefetcher.issue(plans[name], num_tokens)
            if final:
                prefetcher.join()

        return hook

    object.__setattr__(layer.self_attn.attn, "_l2_prefetch_hook", window("a", False))
    if layer.is_layer_sparse:
        object.__setattr__(
            layer.mlp.experts,
            "_l2_prefetch_post_experts_hook",
            window("c", nxt is None),
        )


@support_torch_compile
class MiMoV2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self._pending_fp8_qkv_proj: dict[str, dict[str, torch.Tensor]] = {}

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_size=128,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiMoV2FlashDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if envs.VLLM_MIMO_L2_PREFETCH:
            layers = list(islice(self.layers, self.start_layer, self.end_layer))
            for layer, nxt in zip(layers, [*layers[1:], None]):
                _install_l2_prefetch_windows(layer, nxt)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer()
        self._declare_fused_allreduce_rms_norm_sites()

    def _declare_fused_allreduce_rms_norm_sites(self) -> None:
        # Each post-attention norm follows the o_proj all-reduce, and each later
        # layer's input norm follows the previous layer's MLP/MoE all-reduce.
        norms = []
        layers = islice(self.layers, self.start_layer, self.end_layer)
        for i, layer in enumerate(layers):
            norms.append(
                (
                    f"layers.{layer.layer_id}.post_attention_layernorm",
                    layer.post_attention_layernorm,
                )
            )
            if i > 0:
                norms.append(
                    (f"layers.{layer.layer_id}.input_layernorm", layer.input_layernorm)
                )
        if isinstance(self.norm, RMSNorm):
            norms.append(("norm", self.norm))
        declare_b12x_fused_allreduce_rms_norm_sites(
            self, norms, self.config.hidden_size, "mimo_v2"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        padded_expert_intermediate = _padded_expert_intermediate(
            self.config.moe_intermediate_size, tp_size
        )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        # Pro-format fused qkv_proj arrives as two tensors (weight and
        # weight_scale_inv). Store them per-layer so that they can be
        # sharded together across AutoWeightsLoader's separate module groups.
        pending_fp8_qkv_proj = self._pending_fp8_qkv_proj
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "mtp" in name:
                continue

            expert_matched = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                name_rewritten = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if (
                    name_rewritten.endswith((".bias", "_bias"))
                    and name_rewritten not in params_dict
                ):
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = param.weight_loader

                if tp_size == TP3:
                    loaded_weight = _pad_expert_tensor(
                        loaded_weight,
                        name_rewritten,
                        shard_id,
                        padded_expert_intermediate,
                    )

                weight_loader(
                    param,
                    loaded_weight,
                    name_rewritten,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_rewritten)
                expert_matched = True
                break

            if expert_matched:
                continue
            # Support fused qkv_proj checkpoint (Pro format)
            if self._try_load_fp8_qkv_proj(
                name,
                loaded_weight,
                pending_fp8_qkv_proj,
                params_dict,
                loaded_params,
                tp_rank,
                tp_size,
            ):
                continue
            stacked_matched = False
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)

                if (
                    name_rewritten.endswith(".bias")
                    and name_rewritten not in params_dict
                ):
                    continue

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                if tp_size == TP3 and stacked_shard_id in ("q", "k", "v"):
                    module_prefix = name_rewritten.rsplit(".", 1)[0]
                    attn = self.get_submodule(module_prefix.rsplit(".", 1)[0])
                    loaded_weight = _pad_stacked_qkv(
                        loaded_weight, name_rewritten, stacked_shard_id, attn
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name_rewritten)

                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            orig_name = name
            mapped_name = maybe_remap_kv_scale_name(name, params_dict)
            name = mapped_name if mapped_name is not None else orig_name

            if name not in params_dict:
                continue

            param = params_dict[name]

            if "attention_sink_bias" in name:
                copy_tensor_parallel_shard(
                    param.data,
                    loaded_weight,
                    0,
                    tp_rank * param.shape[0],
                    param.shape[0],
                    allow_padding=tp_size == TP3,
                )
                loaded_params.add(name)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    def _try_load_fp8_qkv_proj(
        self,
        name: str,
        tensor: torch.Tensor,
        fp8_qkv_proj_dict: dict[str, dict[str, torch.Tensor]],
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: set[str],
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """
        The fused fp8 QKV projection weights and scale are stored separately.
        Special care must be taken while sharding these tensors across TP ranks.
        See _shard_fp8_qkv_proj for more details.

        Returns:
            True if ``tensor`` was an fp8 qkv_proj weight/scale and was consumed
            (caller should skip it); False otherwise, so the caller falls
            through to its normal loading path.
        """
        is_weight = (
            name.endswith("qkv_proj.weight") and tensor.dtype == torch.float8_e4m3fn
        )
        is_scale = name.endswith("qkv_proj.weight_scale_inv")
        if not is_weight and not is_scale:
            # Weight is not in FP8 format. Ignore.
            return False

        if is_pp_missing_parameter(name, self):
            # This qkv_proj is for a layer not on this PP rank.
            return True

        prefix, qkv_kind = name.rsplit(".", 1)
        entry = fp8_qkv_proj_dict.setdefault(prefix, {})
        entry[qkv_kind] = tensor
        if "weight" not in entry or "weight_scale_inv" not in entry:
            # Still waiting for the other param.
            return True
        del fp8_qkv_proj_dict[prefix]

        # Get self_attn module, which is a parent of qkv_proj.
        attn = self.get_submodule(prefix.rsplit(".", 1)[0])

        # Shard the qkv_proj per-rank.
        w_rank, s_rank = _shard_fp8_qkv_proj(
            entry["weight"],
            entry["weight_scale_inv"],
            num_heads=attn.checkpoint_num_heads,
            num_kv_heads=attn.checkpoint_num_kv_heads,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            checkpoint_tp_size=_checkpoint_config_value(
                self.config, "num_key_value_heads"
            ),
            kv_chunk_rows=getattr(attn, "kv_chunk_rows", 0),
            padded_num_heads=attn.total_num_heads,
            padded_num_kv_heads=attn.total_num_kv_heads,
        )
        sharded = {"weight": w_rank, "weight_scale_inv": s_rank}
        for kind, shard in sharded.items():
            param_name = f"{prefix}.{kind}"
            param = params_dict[param_name]
            if shard.shape[0] > param.shape[0]:
                shard = shard[: param.shape[0]]
            default_weight_loader(param, shard)
            loaded_params.add(param_name)
        return True


class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsEagle3):
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = MiMoV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                padding_size=128,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
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
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
