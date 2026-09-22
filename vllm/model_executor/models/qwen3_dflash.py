# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import io
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.compact import CompactRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import copy_tensor_parallel_shard
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_SLIDING_ATTENTION = "sliding_attention"
_STREAMED_AUX_MIN_TOKENS = 1024


def _enable_dflash_tp_padding(layer: nn.Module) -> None:
    """Zero checkpoint-absent tails before optional online MXFP8 conversion."""
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    if not isinstance(
        layer.quant_method, (UnquantizedLinearMethod, Mxfp8OnlineLinearMethod)
    ):
        raise ValueError(
            "DFlash/DSpark TP padding supports BF16/FP16 checkpoints with "
            "unquantized or online MXFP8 linear weights"
        )
    for parameter in layer.parameters(recurse=False):
        parameter.allow_tp_padding = True


def _dflash_padded_width(width: int, tp: int) -> int:
    # Whole 128-channel tiles preserve MXFP8 block boundaries and avoid extra
    # activation-padding copies in the W8A16 projection kernels.
    return width if width % tp == 0 else round_up(width, tp * 128)


def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """Resolve explicit causality before falling back to legacy layer defaults."""
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return bool(override)
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == _SLIDING_ATTENTION


def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
    """Whether the draft needs a non-causal-capable backend, resolved from config
    (config mirror of the model's ``get_draft_attn_causal``, usable pre-build)."""
    return not all(
        _dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)
    )


def _get_dflash_fc_input_size(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    config = spec_config.draft_model_config.hf_config
    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    num_features_to_use = len(aux_layers) if aux_layers else config.num_hidden_layers
    target_hidden_size = (
        getattr(config, "target_hidden_size", None) or config.hidden_size
    )
    return target_hidden_size * num_features_to_use


def _resolve_layer_attention(
    config: Qwen3Config, layer_idx: int
) -> tuple[int | None, bool]:
    """Resolve ``(sliding_window, causal)`` for one DFlash draft layer.

    +----------------------+-------------------------+--------------------------------+
    | Config               | ``layer_type``          | *``causal``                    |
    +======================+=========================+================================+
    | ``layer_types``      | SWA if ``use_swa``      | True if ``layer_types[i]=SWA`` |
    |                      | else ``layer_types[i]`` | else False                     |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | SWA                     | False                          |
    | + ``use_swa=True``   |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | Full                    | False                          |
    | + ``use_swa=False``  |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    * If ``dflash_config.causal`` is set, its value overrides ``causal`` for all layers.

    This is to support a varied ecosystem of checkpoints, including:
    - XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash (sets "use_swa", assumes non-causal)
    - z-lab/gemma-4-31B-it-DFlash (has mixed layer types, assumes causal only for SWA)
    - z-lab/Qwen3.5-9B-DFlash ("standard" DFlash, all full attn, assumes non-causal)
    """
    dflash_config = getattr(config, "dflash_config", None) or {}
    layer_types = getattr(config, "layer_types", None)
    use_swa = dflash_config.get("use_swa", False)

    any_sliding = False
    if layer_types is not None:
        num_sliding = sum(lt == _SLIDING_ATTENTION for lt in layer_types)
        any_sliding = num_sliding > 0
        # Mixed sliding/full attention needs multiple KV groups (V2 runner only).
        if (
            0 < num_sliding < len(layer_types)
            and not get_current_vllm_config().use_v2_model_runner
        ):
            raise NotImplementedError(
                "DFlash drafters with mixed sliding/full attention require "
                "the V2 model runner; relaunch with "
                "VLLM_USE_V2_MODEL_RUNNER=1."
            )

    # ``use_swa`` forces SWA on every layer, even an all-full ``layer_types``.
    if layer_types is None or (use_swa and not any_sliding):
        is_sliding = use_swa
    else:
        is_sliding = layer_types[layer_idx] == _SLIDING_ATTENTION

    sliding_window = None
    if is_sliding:
        sliding_window = dflash_config.get(
            "swa_window_size", getattr(config, "sliding_window", None)
        )
        if sliding_window is None:
            raise ValueError(
                "DFlash sliding attention requires a window size configured in "
                "dflash_config.swa_window_size or the top-level sliding_window."
            )

    return sliding_window, _dflash_layer_causal(config, layer_idx)


class DFlashAttention(Attention):
    """Attention whose small draft KV is replicated across DCP ranks."""

    dcp_replicated: ClassVar[bool] = True

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        dcp_replicated = vllm_config.parallel_config.decode_context_parallel_size > 1
        if self.sliding_window is not None:
            assert self.attn_type == AttentionType.DECODER
            return SlidingWindowSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
                page_size_padded=getattr(
                    vllm_config.cache_config, "skip_page_size_padded", None
                ),
                # Prefix lookup verifies one lookahead block and then drops it.
                # Keep one additional local window alive during chunked prefill
                # so the proof block is not recycled before it can be hashed.
                extra_retained_tokens=self.sliding_window,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
                dcp_replicated=dcp_replicated,
            )
        spec = super().get_kv_cache_spec(vllm_config)
        if dcp_replicated and isinstance(spec, FullAttentionSpec):
            spec = dataclasses.replace(spec, dcp_replicated=True)
        return spec


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        sliding_window: int | None = None,
        causal: bool = False,
        is_neox_style: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        pad_heads: bool = False,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if pad_heads:
            _enable_dflash_tp_padding(self.qkv_proj)
            _enable_dflash_tp_padding(self.o_proj)

        if envs.VLLM_DFLASH_COMPACT_ROPE:
            params = rope_parameters or {}
            if params.get("rope_type", "default") != "default" or set(params) - {
                "rope_type",
                "rope_theta",
            }:
                raise ValueError("DFlash compact RoPE requires unscaled full-head RoPE")
            self.rotary_emb = CompactRotaryEmbedding(
                self.head_dim,
                self.head_dim,
                max_position,
                params.get("rope_theta", 10000),
                is_neox_style,
                torch.get_default_dtype(),
                capacity=get_current_vllm_config().scheduler_config.max_num_batched_tokens,
            )
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                is_neox_style=is_neox_style,
                rope_parameters=rope_parameters,
            )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        self.sliding_window = sliding_window
        self.attn = DFlashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            sinks=self.attention_sink_bias,
        )
        self.causal = causal
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        # DFlash drafts store the sink-bias flag inside dflash_config; fall back
        # to the top-level attribute used by other (e.g. MiMo) configs.
        dflash_config = getattr(config, "dflash_config", None) or {}
        add_swa_attention_sink_bias = dflash_config.get(
            "attention_sink_bias",
            getattr(config, "add_swa_attention_sink_bias", False),
        )

        # Resolve this layer's attention mode (full vs sliding window, causal vs
        # non-causal) from the draft config.
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)

        # RoPE layout. The rotation applies to the draft's own Q/K, so this is
        # fixed by how the head was distilled, not by the target: a neox-trained
        # head on an interleaved target still needs neox. A mismatch is silent --
        # acceptance collapses and nothing errors -- so a checkpoint that was
        # distilled the other way has to say so here.
        is_neox_style = getattr(config, "is_neox_style", True)

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            add_swa_attention_sink_bias=add_swa_attention_sink_bias,
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=is_neox_style,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            pad_heads=config.num_attention_heads
            != getattr(
                config, "original_num_attention_heads", config.num_attention_heads
            ),
        )
        intermediate_size = _dflash_padded_width(
            config.intermediate_size, get_tensor_model_parallel_world_size()
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        if intermediate_size != config.intermediate_size:
            _enable_dflash_tp_padding(self.mlp.gate_up_proj)
            _enable_dflash_tp_padding(self.mlp.down_proj)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    decoder_layer_cls = DFlashQwen3DecoderLayer

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "midlayer.": "layers.0.",
            # Muse-Glimmer-30B-assistant names the aux-hidden-state encoder
            # `encoder.fc` / `encoder.output_norm_enc`; this head calls them
            # `fc` / `hidden_norm`. Same tensors and shapes, different names.
            "encoder.output_norm_enc.": "hidden_norm.",
            "encoder.fc.": "fc.",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        self.use_aux_hidden_state = drafter_config.get(
            "use_aux_hidden_state",
            getattr(self.config, "use_aux_hidden_state", True),
        )

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Masked query slots are fed to the draft as `mask_token_id`. Most DFlash
        # checkpoints will have the mask embedding in the vocabulary embedding table
        # at that slot id. Some checkpoints (XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash) ship
        # with a separate mask embedding tensor to use instead. When present, we load it
        # and substitute it for embed_tokens[mask_token_id] when computing embeddings.
        self.mask_token_id = drafter_config.get(
            "mask_token_id", getattr(self.config, "mask_token_id", None)
        )
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            if (
                envs.VLLM_DFLASH_AUX_MXFP8_STREAMING
                and envs.VLLM_DFLASH_AUX_BF16_STAGING
            ):
                raise ValueError("Select either MXFP8 or BF16 auxiliary input staging")
            fc_cls = ReplicatedLinear
            fc_kwargs = {}
            fc_output_size = self.config.hidden_size
            if envs.VLLM_DFLASH_SHARD_AUX_PROJECTION:
                fc_cls = ColumnParallelLinear
                fc_kwargs["gather_output"] = True
                fc_output_size = _dflash_padded_width(
                    fc_output_size, get_tensor_model_parallel_world_size()
                )
            self.fc = fc_cls(
                input_size=_get_dflash_fc_input_size(
                    vllm_config,
                ),
                output_size=fc_output_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
                **fc_kwargs,
            )
            if fc_output_size != self.config.hidden_size:
                _enable_dflash_tp_padding(self.fc)
            if envs.VLLM_DFLASH_AUX_MXFP8_STREAMING:
                from vllm.model_executor.kernels.linear.mxfp8.b12x import (
                    B12xMxfp8LinearKernel,
                    Mxfp8LinearLayerConfig,
                )
                from vllm.model_executor.layers.quantization.online.mxfp8 import (
                    Mxfp8OnlineLinearMethod,
                )

                method = self.fc.quant_method
                if not isinstance(method, Mxfp8OnlineLinearMethod):
                    raise ValueError(
                        "DFlash MXFP8 staging requires online MXFP8 FC weights"
                    )
                method.kernel = B12xMxfp8LinearKernel(Mxfp8LinearLayerConfig())
                method.use_a16 = False
                # The wrapper prepares small FC calls separately from staged
                # prefill, so their activation scratch is not reserved twice.
                self.fc.b12x_preparation_suppressed = True
            elif envs.VLLM_DFLASH_AUX_BF16_STAGING:
                from vllm.model_executor.kernels.linear.mxfp8.marlin import (
                    MarlinMxfp8LinearKernel,
                    Mxfp8LinearLayerConfig,
                )
                from vllm.model_executor.layers.quantization.online.mxfp8 import (
                    Mxfp8OnlineLinearMethod,
                )

                method = self.fc.quant_method
                if not isinstance(method, Mxfp8OnlineLinearMethod):
                    raise ValueError(
                        "DFlash BF16 staging requires online MXFP8 FC weights"
                    )
                # Preserve BF16 activations through the same W8A16 linear path
                # used without staging; weight sharding does not quantize inputs.
                method.kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig())
        self._aux_projection_tp_size = (
            get_tensor_model_parallel_world_size()
            if envs.VLLM_DFLASH_SHARD_AUX_PROJECTION
            else 1
        )
        self._streamed_aux_layer_ids = tuple(
            get_eagle3_aux_layers_from_config(vllm_config.speculative_config) or ()
        )
        self._target_hidden_size = int(
            getattr(self.config, "target_hidden_size", None) or self.config.hidden_size
        )
        self._streamed_aux_accumulator = None
        self._streamed_aux_scratch = None
        self._streamed_aux_tokens = self._streamed_aux_index = 0
        self._streamed_aux_generation = self._completed_stream_generation = 0
        self._consumed_stream_generation = 0
        self._completed_stream_result = None
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        # The context projection concatenates K/V weights from every draft
        # layer. It is not a LinearBase module because its output layout is a
        # DFlash-specific fusion. Serialized MXFP8 checkpoints retain an
        # independently packed copy in this parameter container.
        self._fused_kv_linear = nn.Module()
        self._fused_kv_quant_method = None
        self._fused_kv_weight: torch.Tensor | None = None
        self._fused_kv_weight_scale: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            # Replace masked slots with the dedicated mask embedding.
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    def bind_auxiliary_stream(
        self,
        accumulator: Any,
        scratch: torch.Tensor,
    ) -> None:
        """Bind retained storage for large target auxiliary-state projections."""
        if scratch.ndim != 2 or int(scratch.shape[1]) != int(self.config.hidden_size):
            raise ValueError(
                "DFlash auxiliary output scratch must have shape "
                f"[max_tokens,{self.config.hidden_size}], got {tuple(scratch.shape)}"
            )
        if int(scratch.shape[1]) < self._target_hidden_size:
            raise ValueError(
                "DFlash auxiliary output scratch also stages target residual "
                f"sums and must be at least {self._target_hidden_size} wide, "
                f"got {scratch.shape[1]}"
            )
        self._streamed_aux_accumulator = accumulator
        self._streamed_aux_scratch = scratch

    def can_stream_auxiliary_states(
        self,
        layer_ids: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> bool:
        """Return whether a target forward can release auxiliary states early."""
        accumulator = self._streamed_aux_accumulator
        scratch = self._streamed_aux_scratch
        if accumulator is None or scratch is None or not self.use_aux_hidden_state:
            return False
        if hidden_states.ndim != 2 or hidden_states.shape[0] < _STREAMED_AUX_MIN_TOKENS:
            return False
        if tuple(layer_ids) != self._streamed_aux_layer_ids:
            return False
        if hidden_states.shape[1] * len(layer_ids) != accumulator.input_width:
            return False
        if hidden_states.shape[0] > accumulator.max_tokens:
            return False
        if (
            hidden_states.dtype != scratch.dtype
            or hidden_states.device != scratch.device
        ):
            return False
        return not (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())

    def begin_auxiliary_stream(self, hidden_states: torch.Tensor) -> None:
        """Start one ordered sequence of target auxiliary-state slices."""
        if not self.can_stream_auxiliary_states(
            self._streamed_aux_layer_ids, hidden_states
        ):
            raise RuntimeError(
                "DFlash auxiliary streaming was started for an unsupported geometry"
            )
        self._streamed_aux_tokens = int(hidden_states.shape[0])
        self._streamed_aux_index = 0
        self._streamed_aux_generation += 1
        self._completed_stream_result = None
        self._streamed_aux_accumulator.begin(self._streamed_aux_tokens)
        logger.info_once(
            "DFlash staged auxiliary projection is active: "
            "tokens=%d target_width=%d slices=%d.",
            self._streamed_aux_tokens,
            self._target_hidden_size,
            len(self._streamed_aux_layer_ids),
        )

    def accumulate_auxiliary_state(
        self,
        primary: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        """Store one complete target state in the retained linear input."""
        if self._streamed_aux_index >= len(self._streamed_aux_layer_ids):
            raise RuntimeError("DFlash received too many auxiliary states")
        expected_shape = (self._streamed_aux_tokens, self._target_hidden_size)
        if tuple(primary.shape) != expected_shape:
            raise ValueError(
                "DFlash auxiliary state shape changed during streaming: "
                f"got={tuple(primary.shape)}, expected={expected_shape}"
            )
        source = primary
        if residual is not None:
            if tuple(residual.shape) != expected_shape:
                raise ValueError(
                    "DFlash auxiliary residual shape changed during streaming: "
                    f"got={tuple(residual.shape)}, expected={expected_shape}"
                )
            assert self._streamed_aux_scratch is not None
            source = self._streamed_aux_scratch[
                : self._streamed_aux_tokens, : self._target_hidden_size
            ]
            torch.add(primary, residual, out=source)
        self._streamed_aux_accumulator.append(source)
        self._streamed_aux_index += 1

    def finish_auxiliary_stream(self) -> torch.Tensor:
        """Project the assembled input with the draft's FC weights."""
        expected = len(self._streamed_aux_layer_ids)
        if self._streamed_aux_index != expected:
            raise RuntimeError(
                "DFlash auxiliary stream ended before every configured state "
                f"was received: received={self._streamed_aux_index}, "
                f"expected={expected}"
            )
        output = self._gather_auxiliary_projection(
            self._streamed_aux_accumulator.finish()
        )
        self._completed_stream_generation = self._streamed_aux_generation
        self._completed_stream_result = output
        return output

    def _gather_auxiliary_projection(self, output: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_aux_projection_tp_size", 1) > 1:
            output = tensor_model_parallel_all_gather(output, dim=-1)
        return output[..., : self.config.hidden_size]

    def is_streamed_context_states(self, states: list[torch.Tensor]) -> bool:
        """Claim one completed streamed projection exactly once."""
        if len(states) != 1 or self._completed_stream_result is None:
            return False
        candidate = states[0]
        expected = self._completed_stream_result
        matches = (
            self._completed_stream_generation > self._consumed_stream_generation
            and candidate is expected
            and candidate.shape == expected.shape
            and candidate.dtype == expected.dtype
            and candidate.device == expected.device
            and candidate.data_ptr() == expected.data_ptr()
        )
        if matches:
            self._consumed_stream_generation = self._completed_stream_generation
        return matches

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        quant_methods = [a.qkv_proj.quant_method for a in layers_attn]
        uses_mxfp8 = [
            isinstance(method, ModelOptLinearMethod)
            and method.spec.weight == kMxfp8Static
            for method in quant_methods
        ]
        if any(uses_mxfp8) and not all(uses_mxfp8):
            raise ValueError(
                "Every DFlash attention layer must use the same MXFP8 "
                "linear format for the fused context K/V projection."
            )

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)

        if all(uses_mxfp8):
            kv_scales = [a.qkv_proj.weight_scale[a.q_size :] for a in layers_attn]
            self._fused_kv_weight_scale = torch.cat(kv_scales, dim=0)
        else:
            self._fused_kv_weight_scale = None
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
        # tensor so the per-layer K-norm runs as a single grouped kernel.
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._build_context_kv_buffers(layers_attn, has_bias)

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def process_weights_after_loading(self) -> None:
        """Pack the serialized MXFP8 context projection for its GEMM backend."""
        quant_method = self.layers[0].self_attn.qkv_proj.quant_method
        if not (
            isinstance(quant_method, ModelOptLinearMethod)
            and quant_method.spec.weight == kMxfp8Static
        ):
            return
        if self._fused_kv_weight is None or self._fused_kv_weight_scale is None:
            raise RuntimeError(
                "The DFlash MXFP8 context projection requires serialized K/V "
                "weights and block scales to be fused during checkpoint loading."
            )

        output_size, input_size = self._fused_kv_weight.shape
        # Fused context K/V has an independent weight shape and kernel
        # lifecycle; do not repack through the query projection's method.
        fused_method = ModelOptLinearMethod(
            quant_method.spec, quant_method.ctx, quant_method.fmt
        )
        fused_method.input_dtype = quant_method.input_dtype
        fused_method.out_dtype = quant_method.out_dtype
        fused_method.marlin_input_dtype = quant_method.marlin_input_dtype
        self._fused_kv_linear.has_bias = self._fused_kv_bias is not None
        fused_method.create_weights(
            self._fused_kv_linear,
            input_size_per_partition=input_size,
            output_partition_sizes=[output_size],
            input_size=input_size,
            output_size=output_size,
            params_dtype=self.hidden_norm.weight.dtype,
        )
        self._fused_kv_linear.register_parameter(
            "weight", nn.Parameter(self._fused_kv_weight, requires_grad=False)
        )
        self._fused_kv_linear.register_parameter(
            "weight_scale",
            nn.Parameter(self._fused_kv_weight_scale, requires_grad=False),
        )
        fused_method.process_weights_after_loading(self._fused_kv_linear)
        self._fused_kv_quant_method = fused_method
        self._fused_kv_weight = None
        self._fused_kv_weight_scale = None
        logger.info_once(
            "Using %s for the fused DFlash context K/V projection.",
            type(fused_method.kernel).__name__,
        )

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        if self._fused_kv_quant_method is None:
            if self._fused_kv_weight is None:
                raise RuntimeError("DFlash context K/V projection is not initialized.")
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )
        else:
            all_kv_flat = self._fused_kv_quant_method.apply(
                self._fused_kv_linear,
                normed_context_states,
                self._fused_kv_bias,
            )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        # --- Grouped RMSNorm K across all layers ([L, num_ctx, nkv, hd]) ---
        # The weight is selected per layer by the outermost (layer) index.
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        all_k, all_v = self._project_context_kv(context_states, num_ctx, L, nkv, hd)
        all_k_normed = self._normalize_context_k(all_k)

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        rotary = self.layers[0].self_attn.rotary_emb
        if isinstance(rotary, CompactRotaryEmbedding):
            rope_positions, cos_sin_cache = rotary.materialize(context_positions)
        else:
            rope_positions, cos_sin_cache = context_positions, self._rope_cos_sin_cache
        positions_repeated = rope_positions.repeat(L)
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        per_layer = isinstance(context_slot_mapping, list | tuple)
        for i in range(L):
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue  # dummy run: skip cache ops
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _preprocess(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for name, loaded_weight in weights:
            if "attention_sink_bias" in name:
                # Sink bias is per-head; shard it across TP ranks like the
                # attention heads themselves.
                heads_per_rank = self.config.num_attention_heads // tp_size
                shard = loaded_weight.new_empty(heads_per_rank)
                copy_tensor_parallel_shard(
                    shard,
                    loaded_weight,
                    0,
                    tp_rank * heads_per_rank,
                    heads_per_rank,
                    allow_padding=True,
                )
                loaded_weight = shard
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._preprocess(weights), mapper=self.hf_to_vllm_mapper
        )


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    model_cls = DFlashQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        """Per-layer attention causality, aligned with
        get_draft_kv_cache_layer_names."""
        return [layer.self_attn.causal for layer in self.model.layers]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def process_weights_after_loading(self) -> None:
        """Finalize the DFlash fused context projection after linear packing."""
        self.model.process_weights_after_loading()

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlash drafter expects {expected} concatenated aux hidden "
                f"features but received {hidden_states.shape[-1]}. This usually "
                "means the draft model's target_layer_ids reference layers that "
                "do not exist in the target model (incompatible draft/target pair)."
            )
        accumulator = self.model._streamed_aux_accumulator
        if (
            accumulator is not None
            and hidden_states.shape[0] >= _STREAMED_AUX_MIN_TOKENS
        ):
            # Large non-streamed calls, including graph capture, use the same
            # staged buffers. Only smaller calls need a separate linear plan.
            accumulator.begin(hidden_states.shape[0])
            for source in hidden_states.split(accumulator.slice_width, dim=-1):
                accumulator.append(source)
            result = self.model._gather_auxiliary_projection(accumulator.finish())
        else:
            result = self.model.fc(hidden_states)
            result = result[..., : self.model.config.hidden_size]
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def bind_target_auxiliary_stream(self, target_model, scratch) -> None:
        """Stage large Kimi auxiliary states with the selected input precision."""
        bf16_staging = envs.VLLM_DFLASH_AUX_BF16_STAGING
        if not envs.VLLM_DFLASH_AUX_MXFP8_STREAMING and not bf16_staging:
            return
        from vllm.model_executor.kernels.linear.mxfp8.staged import (
            B12xMxfp8InputAccumulator,
        )
        from vllm.utils.b12x import set_b12x_preparation_provider

        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target = getattr(language_model, "model", language_model)
        setter = getattr(target, "set_aux_hidden_state_projector", None)
        if not callable(setter) or not self.model.use_aux_hidden_state:
            raise ValueError(
                "DFlash auxiliary staging requires a target auxiliary-state hook"
            )
        fc = self.model.fc
        if bf16_staging:
            from vllm.model_executor.kernels.linear.bf16_staging import (
                Bf16InputAccumulator,
            )

            accumulator = Bf16InputAccumulator(
                fc, scratch, self.model._target_hidden_size
            )
            self.model.bind_auxiliary_stream(accumulator, scratch)
            setter(self.model)
            logger.info_once("DFlash auxiliary projection retains BF16 input slices.")
            return
        local_width = fc.b12x_mxfp8_packed_weight.out_features
        output = scratch
        if local_width != scratch.shape[1]:
            output = torch.empty(
                (scratch.shape[0], local_width),
                dtype=scratch.dtype,
                device=scratch.device,
            )
        accumulator = B12xMxfp8InputAccumulator(
            fc, output, self.model._target_hidden_size
        )
        self.model.bind_auxiliary_stream(accumulator, scratch)
        setter(self.model)
        set_b12x_preparation_provider(self, self)
        logger.info_once("DFlash auxiliary projection uses prepared MXFP8 staging.")

    def get_b12x_preparation_units(self, layer, workload):
        accumulator = self.model._streamed_aux_accumulator
        if workload.stage != "weights" or accumulator is None:
            return ()
        if not callable(getattr(accumulator, "preparation_unit", None)):
            # BF16 input storage uses the ordinary Marlin FC without B12X plans.
            return ()
        capacity = min(workload.max_tokens, _STREAMED_AUX_MIN_TOKENS - 1)
        small_workload = dataclasses.replace(
            workload,
            max_tokens=capacity,
            token_counts=tuple(
                sorted({capacity, *(n for n in workload.token_counts if n < capacity)})
            ),
            fixed_token_counts=tuple(
                n for n in workload.fixed_token_counts if n < capacity
            ),
        )
        linear = self.model.fc.b12x_linear
        return (
            linear.unit(small_workload, name=f"linear.mxfp8.{linear.layer_name}"),
            accumulator.preparation_unit("dflash.auxiliary"),
        )

    def is_streamed_context_states(self, states: list[torch.Tensor]) -> bool:
        return self.model.is_streamed_context_states(states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash embeds masked slots via mask_token_id (optionally "
                "overridden by a mask_embedding.pt file); it should not ship a "
                "mask_hidden weight."
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # Route the separately-trained mask embedding (if shipped) through the
        # standard weight loader alongside the rest of the draft weights.
        mask_embedding = self._read_mask_embedding()
        if mask_embedding is not None:
            model_weights["model.mask_embedding"] = mask_embedding
            self.model.has_separate_mask_embedding = True

        orig_to_new_substr = {}
        if not includes_draft_id_mapping:
            orig_to_new_substr["draft_id_to_target_id"] = None
        if not includes_embed_tokens:
            orig_to_new_substr["embed_tokens"] = None
        if not self.model.use_aux_hidden_state:
            orig_to_new_substr["fc."] = None
        if not self.model.has_separate_mask_embedding:
            orig_to_new_substr["mask_embedding"] = None
        mapper = WeightsMapper(orig_to_new_substr=orig_to_new_substr)
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items(), mapper=mapper)
        self.model._build_fused_kv_buffers()

    def _read_mask_embedding(self) -> torch.Tensor | None:
        """Checks for an override mask embedding in `mask_embedding.pt` and returns it.

        Some checkpoints ship a separately-trained mask embedding for the mask token,
        which we use to overwrite the embedding for `mask_token_id`. This helper
        checks for the file, loads the pytorch tensor, and returns the embedding to use.

        Returns None if the override file is not present.
        """
        mask_token_id = self.model.mask_token_id
        if mask_token_id is None:
            return None

        MASK_EMBEDDING_FILENAME = "mask_embedding.pt"
        data = get_hf_file_bytes(
            MASK_EMBEDDING_FILENAME,
            self.draft_model_config.model,
            self.draft_model_config.revision,
        )
        if data is None:
            return None

        state = torch.load(io.BytesIO(data), weights_only=True)
        if isinstance(state, dict):
            if state.get("mask_token_id", mask_token_id) != mask_token_id:
                raise ValueError(
                    f"{MASK_EMBEDDING_FILENAME} mask_token_id does not match "
                    f"dflash_config.mask_token_id ({mask_token_id}). "
                    f"Got {state.get('mask_token_id')}."
                )
            state = state["embedding"]

        logger.info(
            "Loaded DFlash mask embedding for mask_token_id %s from %s",
            mask_token_id,
            MASK_EMBEDDING_FILENAME,
        )
        return state.reshape(-1)
