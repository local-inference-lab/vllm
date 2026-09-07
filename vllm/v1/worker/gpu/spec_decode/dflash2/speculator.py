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

import copy
import inspect
import os
from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.models.kimi_k3.nvidia.dflash2_mla import (
    describe_dflash2_draft,
    dflash2_draft_query_rows,
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
        if self._draft_topk is not None:
            raise ValueError("DFlash2 drafts do not support dspark_draft_topk.")
        self.use_selector = bool(envs.VLLM_DFLASH2_SELECTOR)
        # Query rows per request and step: the checkpoint's trained block
        # (anchor + block_size - 1 masks) by default, of which the first
        # num_speculative_steps mask rows are proposed; the speculative
        # config derived the same number for the scheduler's KV lookahead.
        self.draft_query_rows = int(
            getattr(self.speculative_config, "draft_query_rows", None)
            or dflash2_draft_query_rows(
                hf_config,
                self.num_speculative_steps,
                bool(envs.VLLM_DFLASH2_FULL_BLOCK),
            )
        )
        if self.draft_query_rows < 1 + self.num_speculative_steps:
            raise ValueError(
                f"DFlash2 draft rows ({self.draft_query_rows}) cannot be fewer "
                f"than one anchor plus {self.num_speculative_steps} proposals."
            )
        self.num_query_per_req = self.draft_query_rows
        # Fidelity dump (VLLM_DFLASH2_DUMP_DIR): the first requests' target
        # auxiliary states, block token ids, draft hidden states and logits of
        # every step, for the offline comparison against the reference model.
        self._dump_dir = os.environ.get("VLLM_DFLASH2_DUMP_DIR") or None
        self._dump_step = 0
        # Per padded request count: the graph-owned (sample_hidden, base_logits)
        # of the last eager run or capture at that size; a replayed graph
        # rewrites the same storage, so the entry stays current.
        self._dump_stash: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.max_num_reqs * self.num_query_per_req > self.max_num_tokens:
            raise ValueError(
                "max_num_batched_tokens is too small for the DFlash2 draft block "
                f"({self.max_num_reqs * self.num_query_per_req} > "
                f"{self.max_num_tokens})."
            )
        logger.info_once(
            "%s",
            describe_dflash2_draft(
                hf_config,
                self.num_speculative_steps,
                sampled=self.draft_logits is not None,
            )
            + (
                " Positions sampled in order with the candidate selector."
                if self.use_selector
                else " Positions sampled in one parallel pass (selector off)."
            )
            + f" Draft block {self.draft_query_rows} rows per request.",
        )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """The draft attention view sized for the trained block.

        The attention builders derive their decode query-length threshold
        from the speculative config's proposal count; a block wider than the
        proposals needs the threshold to cover every row of it.
        """
        config = super().attn_vllm_config
        speculative_config = config.speculative_config
        if (
            speculative_config is not None
            and self.draft_query_rows > 1 + self.num_speculative_steps
        ):
            block_view = copy.copy(speculative_config)
            block_view.num_speculative_tokens = self.draft_query_rows - 1
            config.speculative_config = block_view
        return config

    def propose(self, input_batch, *args, **kwargs):  # type: ignore[override]
        draft_tokens = super().propose(input_batch, *args, **kwargs)
        if self._dump_dir is not None:
            bound = inspect.signature(DSparkSpeculator.propose).bind(
                self, input_batch, *args, **kwargs
            )
            self._dump_state(
                input_batch, bound.arguments.get("aux_hidden_states"), draft_tokens
            )
        return draft_tokens

    def _dump_state(self, input_batch, aux_hidden_states, draft_tokens) -> None:
        """Write one step of the first request (TP rank 0) for the offline
        comparison: the target auxiliary states of the tokens the target ran
        this step, the draft block's token ids and positions, the draft's
        sampled hidden states and unary logits, and the proposals."""
        if get_tensor_model_parallel_rank() != 0 or self._dump_step >= 64:
            return
        rows = self.num_query_per_req
        # Long prefills are not dumped (their auxiliary states are hundreds of
        # megabytes); the comparison uses short, uncached requests.
        max_tokens = int(os.environ.get("VLLM_DFLASH2_DUMP_MAX_TOKENS", "1024"))
        if input_batch.num_tokens > max_tokens:
            aux_hidden_states = None
        # The smallest captured (or eager) size covering this batch holds the
        # tensors of this step; a step without sampling has no entry.
        sizes = sorted(
            size for size in self._dump_stash if size >= input_batch.num_reqs
        )
        stash = self._dump_stash.get(sizes[0]) if sizes else None
        sample_hidden = stash[0] if stash is not None else None
        base_logits = stash[1] if stash is not None else None
        record = {
            "step": self._dump_step,
            "num_reqs": int(input_batch.num_reqs),
            "num_target_tokens": int(input_batch.num_tokens),
            "target_positions": input_batch.positions[: input_batch.num_tokens]
            .detach()
            .cpu()
            if hasattr(input_batch, "positions")
            else None,
            "aux_hidden_states": [
                aux[: input_batch.num_tokens].detach().cpu()
                for aux in aux_hidden_states
            ]
            if aux_hidden_states
            else None,
            "block_input_ids": self.input_buffers.input_ids[:rows].detach().cpu(),
            "block_positions": self.input_buffers.positions[:rows].detach().cpu()
            if hasattr(self.input_buffers, "positions")
            else None,
            "sample_hidden": sample_hidden[: self.num_speculative_steps].detach().cpu()
            if sample_hidden is not None
            else None,
            "base_logits": base_logits[: self.num_speculative_steps].detach().cpu()
            if base_logits is not None
            else None,
            "draft_tokens": draft_tokens[0].detach().cpu(),
        }
        os.makedirs(self._dump_dir, exist_ok=True)
        torch.save(
            record, os.path.join(self._dump_dir, f"step-{self._dump_step:04d}.pt")
        )
        self._dump_step += 1

    def _query_len_for_speculative_steps(self, num_speculative_steps: int) -> int:
        # Every step drafts the full block; fewer proposals do not shrink it.
        return max(1 + num_speculative_steps, self.draft_query_rows)

    def _speculative_steps_for_query_len(self, query_len: int) -> int:
        return min(query_len - 1, self.num_speculative_steps)

    # Nothing of the proposal runs outside the draft graph.
    _finish_captured_draft = DFlashSpeculator._finish_captured_draft

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        is_profile: bool = False,
        num_query_per_req: int | None = None,
        capture_only: bool = False,
    ) -> None:
        if not self.use_selector:
            # DFlash's parallel proposal: one sampling pass over every block
            # position of the backbone output.
            DFlashSpeculator._generate_draft(
                self,
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                is_profile,
                num_query_per_req,
                capture_only,
            )
            return
        if num_query_per_req is None:
            num_query_per_req = self.num_query_per_req
        n_spec = self._speculative_steps_for_query_len(num_query_per_req)
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if torch.cuda.is_current_stream_capturing():
            self._captured_backbone_outputs.append(head_hidden)
        self._sample_with_selector(num_reqs, head_hidden, n_spec, num_query_per_req)

    def _sample_with_selector(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        n_spec: int,
        num_query_per_req: int,
    ) -> None:
        """Sample the block in order; each position's logits carry the
        selector's transition scores from the token sampled before it."""
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        if self._dump_dir is not None:
            # Graph-owned intermediates keep their storage while referenced.
            self._dump_stash[num_reqs] = (sample_hidden, base_logits)
        selector = self.model.candidate_selector
        hidden, candidate_ids, successor_rows = selector.prepare_rows(
            sample_hidden, base_logits
        )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        hidden = hidden.view(num_reqs, n_spec, -1)
        candidate_ids = candidate_ids.view(num_reqs, n_spec, -1)
        successor_rows = successor_rows.view(num_reqs, n_spec, selector.top_k, -1)
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        # The anchor (bonus) token of each request: the first query row.
        prev = self.input_buffers.input_ids[
            : num_reqs * num_query_per_req : num_query_per_req
        ]
        for i in range(n_spec):
            logits_i = selector.condition(
                base_logits[:, i],
                prev,
                hidden[:, i],
                candidate_ids[:, i],
                successor_rows[:, i],
            )
            sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = sampled_i
            prev = sampled_i
