# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.8-Flash-Next model."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from itertools import islice

import torch
from torch import nn

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .config import Qwen3_8FlashNextConfig, Qwen3_8FlashNextTextConfig
from .hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
    HyperConnectionWorkspace,
)
from .ple_layer import Qwen3_8FlashNextPLELayer


def _remap_qsa_cache_scale_name(name: str, qsa_layer_ids: frozenset[int]) -> str:
    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        if suffix in scale_suffixes:
            return f"{name[: marker_start + len(marker)]}{scale_suffixes[suffix]}"
    return name


_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

_HC_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
    }
)


class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class Qwen3_8FlashNextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        workspace: HyperConnectionWorkspace,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.ple: Qwen3_8FlashNextPLELayer | None = None
        if self.layer_idx + 1 in config.ple_layer_ids:
            dense_ids = sorted(set(config.ple_layer_ids))
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=dense_ids.index(self.layer_idx + 1),
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                prefer_b12x_gdn_decode=True,
                overlap_input_projections=envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP,
            )
        elif layer_type == "full_attention":
            if getattr(config, "indexer_n_heads", None) is None:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                qsa_module = import_module("vllm.models.qwen3_8_flash_next.qsa")
                self.self_attn = qsa_module.Qwen3_8FlashNextQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"invalid layer_type {layer_type!r}")

        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe = self.layer_idx not in getattr(config, "mlp_only_layers", []) and (
            num_experts > 0 and absolute_layer_id % int(config.decoder_sparse_step) == 0
        )
        if is_moe:
            self.mlp = Qwen3_8FlashNextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            workspace,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            workspace,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        ple_prefetched: bool = False,
        output_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                prefetched=ple_prefetched,
            )
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        else:
            attn_out = self.self_attn(hidden_states=block_input, positions=positions)
        if output_indices is not None:
            hidden_states = hidden_states[output_indices]
            attn_out = attn_out[output_indices]
            assert injection is not None
            injection = injection[output_indices]
        hidden_states, block_input, injection = (
            self.mlp_hyper_connection.combine_and_mix(
                hidden_states, attn_out, injection
            )
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


class Qwen3_8FlashNextMixtureOfExperts(MixtureOfExperts):
    def set_moe_parameters(self, layers: Iterable[nn.Module]) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in layers:
            if isinstance(layer, Qwen3_8FlashNextDecoderLayer) and isinstance(
                layer.mlp, Qwen3_8FlashNextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_8FlashNextModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            index
            for index, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.hyper_connection_workspace = HyperConnectionWorkspace(
            hc_config, vllm_config.scheduler_config.max_num_batched_tokens
        )

        def get_layer(prefix: str) -> Qwen3_8FlashNextDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                config.layer_types[layer_idx],
                self.hyper_connection_workspace,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count
        )
        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                self.hyper_connection_workspace,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self.register_buffer(
                "_mtp_hidden_buffer",
                torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.hc_count * config.hidden_size,
                    dtype=vllm_config.model_config.dtype,
                    device=current_platform.current_device(),
                ),
                persistent=False,
            )
        else:
            self.register_buffer("_mtp_hidden_buffer", None, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        ple_prefetched = False
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            next_ple_prefetched = False
            if (
                envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP
                and current_platform.is_cuda()
                and layer_idx + 1 < self.end_layer
                and input_ids is not None
                and query_start_loc is not None
                and ngram_context is not None
            ):
                next_ple = self.layers[layer_idx + 1].ple
                if next_ple is not None:
                    next_ple.ple_embedding.prefetch(
                        input_ids, query_start_loc, ngram_context
                    )
                    next_ple_prefetched = True
            hidden_states, block_output, injection = layer(
                hidden_states,
                block_output,
                injection,
                positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                ple_prefetched=ple_prefetched,
            )
            ple_prefetched = next_ple_prefetched
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states,
                    block_output,
                    injection,
                )
                block_output = injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self._mtp_hidden_buffer is not None:
            self._mtp_hidden_buffer[: multi_hidden.shape[0]].copy_(multi_hidden)
        return sample_hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (_remap_qsa_cache_scale_name(name, self._qsa_layer_ids), weight)
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=(
                _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy()
            ),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_8FlashNextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen3_8FlashNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "mtp.": None,
        }
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next requires --mamba-cache-mode=align"
            )
        if config.ple_layer_ids and vllm_config.parallel_config.enable_dbo:
            raise NotImplementedError("Qwen3.8-Flash-Next PLE does not support DBO")
        self.model = Qwen3_8FlashNextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            lm_head=self.lm_head,
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        config = vllm_config.model_config.hf_text_config
        state_len = (config.ple_conv_kernel_size - 1) * config.ngram_size
        return MambaStateShapeCalculator.short_conv_state_shape(
            1,
            config.hidden_size * config.hc_count,
            state_len + 1,
            vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            vllm_config.num_speculative_tokens,
        )

    get_mamba_state_dtype_from_config = get_gdn_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = get_gdn_mamba_state_shape_from_config

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing = mamba_types - copy_funcs.keys()
        assert not missing, f"missing state copy funcs for {missing}"
        return {kind: copy_funcs[kind] for kind in mamba_types}

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=(
                _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy()
            ),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_8FlashNextProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen3_8FlashNextConfig:
        return self.ctx.get_hf_config(Qwen3_8FlashNextConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_8FlashNextProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_8FlashNextForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen3_8FlashNextMixtureOfExperts,
):
    requires_raw_input_tokens = True
    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ]
    }

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Qwen3.8-Flash-Next requires multimodal_config")
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_8FlashNextForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.language_model.model.layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if not multimodal_embeddings:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError("language_model_only does not accept multimodal inputs")
        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack, multimodal_embeddings = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack = None
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack is not None:
            self._set_deepstack_input_embeds(deepstack)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        deepstack = (
            self._get_deepstack_input_embeds(inputs_embeds.size(0))
            if inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self.use_deepstack
            else None
        )
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack,
        )
        if (
            inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self.use_deepstack
        ):
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=(
                ["visual."] if self.language_model_only else None
            ),
            ignore_unexpected_suffixes=(
                _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy()
            ),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    get_mamba_state_dtype_from_config = (
        Qwen3_8FlashNextForCausalLM.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_8FlashNextForCausalLM.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_func
    get_mamba_state_copy_funcs = Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_funcs


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextMixtureOfExperts",
    "Qwen3_8FlashNextModel",
    "Qwen3_8FlashNextSparseMoeBlock",
]
