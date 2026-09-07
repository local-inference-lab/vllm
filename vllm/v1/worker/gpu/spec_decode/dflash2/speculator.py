# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Colocated speculator for the DFlash2 Kimi-K3 draft.

The draft model is `DFlash2ForCausalLM` (the DSpark MLA backbone with
DFlash2's grouped convolutions and candidate selector, see
`vllm.models.kimi_k3.nvidia.dflash2_mla`). Loading, the fused latent context
KV, the target's auxiliary-state streaming and the query layout come from the
DSpark speculator; the proposal is DFlash's: one backbone forward over the
anchor plus mask tokens of every request, then one sampling pass over the
LM-head distributions of all block positions (probabilistic when the draft
distributions are handed to the rejection sampler, greedy otherwise). DSpark's
sequential Markov sampling and its confidence-driven draft capacity do not
apply: the DFlash2 checkpoint has neither head.

The candidate selector is available on the model (`model.candidate_selector`)
for the greedy lattice walk of the checkpoint's own runtime and for a
selector-conditioned sampling variant; this speculator's proposal does not
use it.

The embedding and LM head are the target's. The checkpoint's own embedding
table, mask-token row included, is identical to the target's, so no separate
mask embedding is loaded.
"""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.kimi_k3.nvidia.dflash2_mla import (
    dflash2_config_summary,
    is_dflash2_draft,
    normalize_dflash2_config,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)


class DFlash2Speculator(DSparkSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        if not is_dflash2_draft(hf_config):
            raise ValueError(
                "DFlash2Speculator requires a DFlash2DraftModel checkpoint with "
                "attention_mode 'mla'."
            )
        # Lift the nested fields the MLA backbone reads and pin the DFlash
        # query layout (anchor + mask tokens) before the DSpark base reads
        # `sample_from_anchor`.
        normalize_dflash2_config(hf_config)
        super().__init__(vllm_config, device)
        if self.sample_from_anchor or self.num_query_per_req != (
            1 + self.num_speculative_steps
        ):
            raise RuntimeError(
                "DFlash2 expects the anchor-plus-mask query layout, got "
                f"sample_from_anchor={self.sample_from_anchor}, "
                f"num_query_per_req={self.num_query_per_req}."
            )
        if self.use_draft_token_capacity:
            raise ValueError(
                "DFlash2 drafts have no confidence head; disable the DSpark "
                "draft-token capacity settings."
            )
        geometry = dflash2_config_summary(hf_config)
        logger.info_once(
            "DFlash2 draft: %d MLA layers (%s, window %s), block %d, conv taps %d "
            "x groups of %d, selector rank %d top-%d, target taps %s, "
            "%d speculative tokens per step, %s proposals.",
            geometry.layers,
            "/".join(t.replace("_attention", "") for t in geometry.layer_types),
            geometry.sliding_window,
            geometry.block_size,
            geometry.taps,
            geometry.group_size,
            geometry.selector_rank,
            geometry.selector_top_k,
            geometry.target_layer_ids,
            self.num_speculative_steps,
            "sampled" if self.draft_logits is not None else "greedy",
        )

    # DFlash's parallel proposal: backbone forward, then one sampling pass over
    # every block position. Replaces the DSpark Markov loop.
    _generate_draft = DFlashSpeculator._generate_draft
    _finish_captured_draft = DFlashSpeculator._finish_captured_draft
