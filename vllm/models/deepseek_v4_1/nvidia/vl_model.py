# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 DeepSeek V4.1 with native vision and streaming checkpoint loading."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV4VLImagePixelInputs,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM as UpstreamDeepseekV41ForCausalLM,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper as _make_deepseek_v4_vl_weights_mapper,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .b12x_vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
    run_dp_sharded_vision_tower,
)
from .model import (
    DeepseekV41LLMForCausalLM,
    _linear_scale_param_name,
)


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(UpstreamDeepseekV41ForCausalLM):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    supports_encoder_tp_data = True

    # The MoE router applies bias_vl to raw image-span token IDs.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        with self._mark_tower_model(vllm_config, {"image"}):
            self.use_data_parallel = True
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV41LLMForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # The outer mapper (see load_weights) fully resolves HF names into
        # this wrapper's namespace before AutoWeightsLoader strips the
        # "language_model." prefix and delegates to the child's load_weights,
        # so the child's own mapper must be a no-op. Its suffix rules are not
        # idempotent (e.g. "lm_head.weight".endswith("head.weight") would
        # re-fire "head.weight" -> "lm_head.weight").
        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

    def _process_image_input(
        self,
        image_input: DeepseekV4VLImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        patches = image_input.patches.to(self.aligner.w1.weight.dtype)
        vit_grid = image_input.vit_grid.tolist()

        image_embeds_list: list[torch.Tensor]
        if self.use_data_parallel and get_tensor_model_parallel_world_size() > 1:
            # Data-parallel ViT: shard images across TP ranks and all-gather
            # the per-image embeddings (weights are replicated on every rank).
            image_embeds_list = run_dp_sharded_vision_tower(
                self.vision, self.aligner, patches, vit_grid
            )
        else:
            image_embeds_list = []
            vit_offset = 0
            for n_vit_h, n_vit_w in vit_grid:
                n_vit = n_vit_h * n_vit_w
                image_embeds_list.append(
                    self._encode_image(
                        patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w
                    )
                )
                vit_offset += n_vit

        embeds: list[torch.Tensor] = []
        span_offset = 0
        for image_embeds, (n_llm_h, n_llm_w) in zip(
            image_embeds_list, image_input.llm_grid.tolist(), strict=True
        ):
            span_len = n_llm_h * (n_llm_w + 1) + 2
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    image_input.types[span_offset : span_offset + span_len],
                )
            )
            span_offset += span_len
        return tuple(embeds)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    requires_accepted_token_lookback = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )

    def checkpoint_file_weight_filter(self, name: str) -> bool:
        return self.language_model.checkpoint_file_weight_filter(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Prefix groups may recur across checkpoint files. Consume each tensor
        # immediately; sorting payloads retains the entire checkpoint on device.
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

    def process_weights_after_loading(self) -> None:
        # Native vision and aligner declarations are collected from their loaded
        # owners by the worker preparation registry.
        self.language_model.process_weights_after_loading()
