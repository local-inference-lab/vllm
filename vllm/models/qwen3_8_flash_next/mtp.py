# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.8-Flash-Next multi-token predictor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import regex as re
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import LocalArgmaxMixin, SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.b12x import get_b12x_mtp_feedback, get_b12x_scratch_buffers
from vllm.utils.torch_utils import direct_register_custom_op

from .config import Qwen3_8FlashNextTextConfig
from .hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
    HyperConnectionWorkspace,
)
from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES,
    Qwen3_8FlashNextDecoderLayer,
    Qwen3_8FlashNextMixtureOfExperts,
    _remap_qsa_cache_scale_name,
)


def _mtp_api() -> Any:
    api = get_b12x_mtp_feedback()
    if api is None:
        raise ImportError(
            "Qwen3.8-Flash-Next MTP requires b12x.sequence.mtp_feedback; "
            "install the b12x serving extra"
        )
    return api


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    return [_remap_mtp_layer_name(name, mtp_start_layer_idx) for name in ignored_layers]


def _remap_mtp_layer_name(name: str, mtp_start_layer_idx: int) -> str:
    if not name.startswith("mtp."):
        return name
    return re.sub(
        r"(?<=\.layers\.)\d+",
        lambda match: str(mtp_start_layer_idx + int(match.group(0))),
        name,
    )


def _remap_mtp_quantized_layers(
    quantized_layers: dict[str, dict[str, Any]],
    mtp_start_layer_idx: int,
) -> dict[str, dict[str, Any]]:
    return {
        _remap_mtp_layer_name(name, mtp_start_layer_idx): layer_config
        for name, layer_config in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map target-checkpoint names into the standalone draft model."""
    for checkpoint_prefix in ("model.language_model.", "language_model."):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    for shared_head_prefix in (
        "mtp.shared_head.head.",
        "model.shared_head.head.",
        "shared_head.head.",
    ):
        if name.startswith(shared_head_prefix):
            return name.replace(shared_head_prefix, "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen3_8FlashNextMTP)
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            draft_quant_config.quantized_layers = (  # type: ignore[attr-defined]
                _remap_mtp_quantized_layers(
                    quantized_layers,
                    mtp_start_layer_idx,
                )
            )
        for attribute in ("ignored_layers", "exclude_modules"):
            names = getattr(draft_quant_config, attribute, None)
            if names:
                setattr(
                    draft_quant_config,
                    attribute,
                    _remap_ignored_layers(names, mtp_start_layer_idx),
                )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMultiTokenPredictor(nn.Module):
    """One-layer draft stack with fused token/multi-stream feedback."""

    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = int(getattr(config, "mtp_num_hidden_layers", 1))
        self.supports_mtp_prefill_compaction = (
            envs.VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT
            and vllm_config.scheduler_config.max_num_seqs == 1
        )
        self._prefill_output_indices: torch.Tensor | None = None
        if self.num_mtp_layers != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next MTP supports exactly one predictor layer"
            )

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
        )
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            self.fc_embedding = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=maybe_prefix(prefix, "fc_embedding"),
                return_bias=False,
            )
            self.fc_hidden = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=maybe_prefix(prefix, "fc_hidden"),
                return_bias=False,
            )
            hc_config = HyperConnectionConfig(
                hc_count=self.hc_count,
                hidden_size=self.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
            )
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.hyper_connection_workspace = HyperConnectionWorkspace(
                hc_config, max_tokens
            )
            self.layers = nn.ModuleList(
                [
                    Qwen3_8FlashNextDecoderLayer(
                        draft_vllm_config,
                        layer_type="full_attention",
                        workspace=self.hyper_connection_workspace,
                        prefix=(
                            f"{prefix}.layers.{self.mtp_start_layer_idx}"
                            if prefix
                            else f"layers.{self.mtp_start_layer_idx}"
                        ),
                    )
                ]
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                self.hyper_connection_workspace,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )

        self.pre_fc_norm_embedding = GroupedGemmaRMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
            group_size=None,
            dtype=torch.bfloat16,
        )
        self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(
            self.hidden_size * self.hc_count,
            eps=config.rms_norm_eps,
            group_size=None,
            dtype=torch.bfloat16,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

        device = torch.device(current_platform.current_device())
        self.register_buffer(
            "_decode_output_indices",
            torch.zeros(1, dtype=torch.int64, device=device),
            persistent=False,
        )
        if self.supports_mtp_prefill_compaction:
            self._prefill_output_indices = self._decode_output_indices
        caps = _mtp_api().Caps(
            device=device,
            max_tokens=max_tokens,
            hidden_size=self.hidden_size,
            streams=self.hc_count,
            dtype=torch.bfloat16,
        )
        self._feedback_plan = _mtp_api().plan(caps)
        (feedback_scratch,) = get_b12x_scratch_buffers(self._feedback_plan)
        self.register_buffer(
            "_feedback_scratch",
            feedback_scratch,
            persistent=False,
        )
        factory = dict(dtype=torch.bfloat16, device=device)
        self.register_buffer(
            "_feedback_token_embedding",
            torch.empty(max_tokens, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_feedback_multi_state",
            torch.empty(max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_feedback_output",
            torch.empty(max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.prefix = prefix

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def set_prefill_output_indices(self, output_indices: torch.Tensor | None) -> None:
        """Select request-tail outputs after populating the attention cache."""
        # AOT reuses the same compiled callable for first and subsequent drafts.
        # Keep the one-row selector's tensor contract stable in both phases.
        self._prefill_output_indices = (
            self._decode_output_indices if output_indices is None else output_indices
        )

    def snapshot_qsa_interval_starts(self) -> None:
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            snapshot = getattr(attention, "snapshot_speculative_interval_starts", None)
            if snapshot is not None:
                snapshot()

    def restore_qsa_interval_starts(self) -> None:
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            restore = getattr(attention, "restore_speculative_interval_starts", None)
            if restore is not None:
                restore()

    def _run_feedback(
        self,
        token_embedding: torch.Tensor,
        multi_state: torch.Tensor,
    ) -> None:
        num_tokens = token_embedding.shape[0]
        if num_tokens > self._feedback_plan.caps.max_tokens:
            raise ValueError(
                "Qwen3.8-Flash-Next MTP feedback capacity exceeded: "
                f"{num_tokens}/{self._feedback_plan.caps.max_tokens} tokens"
            )
        multi_state = multi_state.reshape(num_tokens, self.hc_count, self.hidden_size)
        self._feedback_token_embedding[:num_tokens].copy_(token_embedding)
        self._feedback_multi_state[:num_tokens].copy_(multi_state)
        binding = self._feedback_plan.bind(
            scratch=self._feedback_scratch,
            token_embedding=self._feedback_token_embedding,
            multi_state=self._feedback_multi_state,
            token_norm_weight=self.pre_fc_norm_embedding.weight,
            state_norm_weight=self.pre_fc_norm_hidden.weight,
            embedding_fc_weight=self.fc_embedding.weight.data,
            hidden_fc_weight=self.fc_hidden.weight.data,
            output=self._feedback_output,
            tokens=num_tokens,
        )
        _mtp_api().run(binding, eps=self.config.rms_norm_eps)

    def _prepare_feedback(
        self,
        token_embedding: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            torch.ops.vllm.qwen3_8_flash_next_mtp_feedback(
                token_embedding,
                hidden_states,
                self._feedback_output,
                self.prefix,
            )
        else:
            self._run_feedback(token_embedding, hidden_states)
        return self._feedback_output[: token_embedding.shape[0]].flatten(-2)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if hidden_states is None:
                raise ValueError("MTP requires target-model hidden states")
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                inputs_embeds = self.embed_input_ids(input_ids)
            hidden_states = self._prepare_feedback(inputs_embeds, hidden_states)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        layer = self.layers[spec_step_idx % self.num_mtp_layers]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
            output_indices=self._prefill_output_indices,
        )
        if not get_pp_group().is_last_rank:
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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


def _mtp_feedback_op(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._run_feedback(token_embedding, multi_state)


def _mtp_feedback_fake(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_8_flash_next_mtp_feedback",
    op_func=_mtp_feedback_op,
    mutates_args=["output"],
    fake_impl=_mtp_feedback_fake,
)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMTP(
    LocalArgmaxMixin,
    nn.Module,
    SupportsPP,
    Qwen3_8FlashNextMixtureOfExperts,
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next MTP requires --mamba-cache-mode=align"
            )
        self.quant_config = vllm_config.quant_config
        super().__init__()
        self.has_own_lm_head = envs.VLLM_MTP_NVFP4_LM_HEAD
        if (
            self.has_own_lm_head
            and envs.is_set("VLLM_MTP_NVFP4_LM_HEAD")
            and config.tie_word_embeddings
        ):
            raise ValueError("NVFP4 draft head requires untied word embeddings")
        self.config = config
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
                lm_head_quantization="nvfp4" if self.has_own_lm_head else None,
            )
            self.has_own_lm_head = self.lm_head.runtime_lm_head_quantization == "nvfp4"
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            lm_head=self.lm_head,
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qsa_layer_ids = frozenset(range(self.model.num_mtp_layers))

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield (
                        _remap_qsa_cache_scale_name(remapped_name, qsa_layer_ids),
                        weight,
                    )

        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=(
                _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy()
            ),
        )
        return loader.load_weights(remap_weight_names())


__all__ = ["Qwen3_8FlashNextMTP", "Qwen3_8FlashNextMultiTokenPredictor"]
