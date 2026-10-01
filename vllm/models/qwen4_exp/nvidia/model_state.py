# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE and QSA inputs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .b12x_ple import B12xNGramEmbedding
from .backend import uses_b12x


@dataclass
class Qwen4ExpAttnMetadata(MambaHybridAttnMetadata):
    """Package-local metadata consumed by QSA attention builders."""

    qsa_state_slot_ids: torch.Tensor | None = None
    qsa_state_is_fresh: torch.Tensor | None = None
    qsa_num_accepted_tokens: torch.Tensor | None = None
    qsa_is_prefilling: torch.Tensor | None = None
    # Set during CUDA graph capture: B12X QSA selects and stages its context
    # inside PIECEWISE graphs, so the captured context must cover max_model_len.
    qsa_max_seq_len: int | None = None

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_attn_kwargs(attn_metadata_builder, num_reqs)
        if not getattr(attn_metadata_builder, "requires_qsa_metadata", False):
            return kwargs
        assert self.qsa_state_slot_ids is not None
        assert self.qsa_state_is_fresh is not None
        assert self.qsa_num_accepted_tokens is not None
        assert self.qsa_is_prefilling is not None
        kwargs.update(
            qsa_state_slot_ids=self.qsa_state_slot_ids[:num_reqs],
            qsa_state_is_fresh=self.qsa_state_is_fresh[:num_reqs],
            qsa_num_accepted_tokens=self.qsa_num_accepted_tokens[:num_reqs],
            qsa_is_prefilling=self.qsa_is_prefilling[:num_reqs],
            qsa_max_seq_len=self.qsa_max_seq_len,
        )
        return kwargs


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe n-gram history and persistent QSA request identity."""

    specialize_full_decode_graphs = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_qsa = (
            uses_b12x(vllm_config)
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.qsa_state_is_fresh_gpu = torch.ones(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self.qsa_state_slot_ids = torch.arange(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._qsa_default_slot_ids = self.qsa_state_slot_ids.clone()
        self.qsa_state_is_fresh = torch.ones(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self.qsa_num_accepted_tokens = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.mamba_num_accepted_tokens = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.qsa_committed_num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.qsa_is_prefilling = CpuGpuBuffer(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self._qsa_draft_is_prefilling = torch.zeros(
            self.max_num_reqs,
            dtype=torch.bool,
            device="cpu",
        )
        self._qsa_draft_is_prefilling_gpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        self.disk_embeddings = tuple(
            module
            for module in model.modules()
            if isinstance(module, B12xNGramEmbedding)
            and module.requires_disk_preparation
        )
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "Qwen4Exp PLE requires pipeline_parallel_size=1 "
                "because later ranks do not receive raw input token IDs"
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("PLE n-gram context length must be positive")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # b12x hashing accepts signed int64 tokens and treats this tensor as
        # immutable committed history.  The runner rebuilds it from accepted
        # request state on every step, so rejected draft tokens never enter it.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self.uses_qsa:
            # A request-state slot can be recycled while QSA's raw logical-tag,
            # RoPE, and anchor pools still contain the prior owner's data.  The
            # flag remains set through the complete next model forward so every
            # QSA layer independently resets its slot.
            #
            # Only the boundary-checkpoint restore carries the producer's
            # committed selector pools into this slot: the complex
            # OffloadingConnector restores KV pages (main/compressed cache,
            # align-Mamba columns), never these per-slot module buffers.  A
            # resumed prefix without a checkpoint -- local hit or connector
            # external-resume -- still owns a recycled slot, so it keeps the
            # fresh reset.  The condition mirrors the rail's restore branch
            # (BoundaryCheckpointState.add_request), the only pool writer.
            checkpoint = new_req_data.boundary_checkpoint
            restored_selector_state = (
                checkpoint is not None
                and new_req_data.boundary_checkpoint_blocks is not None
            )
            self.qsa_state_is_fresh_gpu[req_index].fill_(not restored_selector_state)
            self.qsa_committed_num_accepted_tokens_gpu[req_index].fill_(1)
            if restored_selector_state:
                # The fresh-reset's anchor formula (first_position -
                # accepted, in the QSA reset kernel) is skipped with the
                # flag down, so seed the value it would have produced:
                # prefix_len - 1.  The commit kernel overwrites it during
                # the first forward (the anchor self-heals), so this keeps
                # the pre-forward selector state consistent, not just the
                # post-forward one.
                anchor = checkpoint.num_tokens - 1
                for module in self._qsa_state_modules():
                    module.set_recurrent_checkpoint_anchor(req_index, anchor)

    def _qsa_state_modules(self) -> tuple[Any, ...]:
        """QSA layer modules exposing the per-request selector pools.

        Mirrors ``BoundaryCheckpointState.target_modules``: the raw ring /
        logical-tag / RoPE / anchor pools are per-layer module buffers
        indexed by request slot, so seeding or reading one request's
        selector state goes through every module that publishes the
        recurrent-checkpoint accessors.
        """
        modules = self.__dict__.get("_qsa_state_module_cache")
        if modules is None:
            modules = tuple(
                module
                for module in self.model.modules()
                if hasattr(module, "set_recurrent_checkpoint_anchor")
            )
            self._qsa_state_module_cache = modules
        return modules

    def get_recurrent_checkpoint_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.qsa_state_is_fresh_gpu,
            self.qsa_committed_num_accepted_tokens_gpu,
        )

    def get_recurrent_checkpoint_acceptance(self) -> torch.Tensor:
        return self.qsa_committed_num_accepted_tokens_gpu

    def get_recurrent_checkpoint_fresh(self) -> torch.Tensor:
        return self.qsa_state_is_fresh_gpu

    def _prepare_qsa_state(
        self,
        input_batch: InputBatch,
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.qsa_state_slot_ids.copy_(self._qsa_default_slot_ids)
        self.qsa_state_is_fresh.fill_(True)
        self.qsa_num_accepted_tokens.fill_(1)
        num_actual_reqs = input_batch.num_reqs
        if num_actual_reqs:
            idx_mapping = input_batch.idx_mapping[:num_actual_reqs]
            self.qsa_state_slot_ids[:num_actual_reqs].copy_(idx_mapping)
            torch.index_select(
                self.qsa_state_is_fresh_gpu,
                0,
                idx_mapping,
                out=self.qsa_state_is_fresh[:num_actual_reqs],
            )
            torch.index_select(
                self.qsa_committed_num_accepted_tokens_gpu,
                0,
                idx_mapping,
                out=self.qsa_num_accepted_tokens[:num_actual_reqs],
            )
        return (
            self.qsa_state_slot_ids[:num_reqs],
            self.qsa_state_is_fresh[:num_reqs],
            self.qsa_num_accepted_tokens[:num_reqs],
        )

    def _prepare_mamba_acceptance(
        self,
        input_batch: InputBatch,
        num_reqs: int,
    ) -> torch.Tensor:
        accepted = self.mamba_num_accepted_tokens[:num_reqs]
        accepted.fill_(1)
        num_actual_reqs = input_batch.num_reqs
        if num_actual_reqs:
            torch.index_select(
                self.num_accepted_tokens_gpu,
                0,
                input_batch.idx_mapping[:num_actual_reqs],
                out=accepted[:num_actual_reqs],
            )
        return accepted

    def prepare_draft_attn_metadata(
        self,
        *,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        draft_index: int,
    ) -> Qwen4ExpAttnMetadata | None:
        if not self.uses_qsa:
            return None
        if draft_index < 1:
            raise RuntimeError(
                "Qwen4Exp supports QSA draft metadata only for "
                "autoregressive MTP lookahead (draft_index >= 1)"
            )
        if not 0 <= num_reqs <= num_reqs_padded <= self.max_num_reqs:
            raise ValueError(
                "draft request counts must satisfy "
                "0 <= num_reqs <= num_reqs_padded <= max_num_reqs"
            )
        if idx_mapping.numel() < num_reqs:
            raise ValueError("idx_mapping does not cover every active draft request")

        self.qsa_state_slot_ids[:num_reqs_padded].copy_(
            self._qsa_default_slot_ids[:num_reqs_padded]
        )
        self.qsa_state_is_fresh[:num_reqs_padded].fill_(True)
        self.qsa_num_accepted_tokens[:num_reqs_padded].fill_(1)
        if num_reqs:
            self.qsa_state_slot_ids[:num_reqs].copy_(idx_mapping[:num_reqs])
            # The draft prefill immediately preceding lookahead initialized the
            # MTP QSA selector state in these persistent request slots.
            self.qsa_state_is_fresh[:num_reqs].fill_(False)
            # Step one continues the reused draft-prefill interval by the
            # target's accepted prefix. Later steps continue a one-row draft
            # decode interval and therefore retain the neutral count of one.
            if draft_index == 1:
                torch.index_select(
                    self.qsa_committed_num_accepted_tokens_gpu,
                    0,
                    idx_mapping[:num_reqs],
                    out=self.qsa_num_accepted_tokens[:num_reqs],
                )

        return Qwen4ExpAttnMetadata(
            is_prefilling=self._qsa_draft_is_prefilling[:num_reqs_padded],
            num_accepted_tokens=self.qsa_num_accepted_tokens[:num_reqs_padded],
            qsa_state_slot_ids=self.qsa_state_slot_ids[:num_reqs_padded],
            qsa_state_is_fresh=self.qsa_state_is_fresh[:num_reqs_padded],
            qsa_num_accepted_tokens=self.qsa_num_accepted_tokens[:num_reqs_padded],
            qsa_is_prefilling=self._qsa_draft_is_prefilling_gpu[:num_reqs_padded],
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        if not self.uses_qsa:
            return super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=ubatch_idx,
            )
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        self.qsa_is_prefilling.np[:num_reqs] = False
        self.qsa_is_prefilling.np[: input_batch.num_reqs] = input_batch.is_prefilling_np
        is_prefilling = self.qsa_is_prefilling.cpu[:num_reqs]
        qsa_is_prefilling = self.qsa_is_prefilling.copy_to_gpu(num_reqs)
        (
            qsa_state_slot_ids,
            qsa_state_is_fresh,
            qsa_num_accepted_tokens,
        ) = self._prepare_qsa_state(input_batch, num_reqs)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            # Mamba page alignment resets its accepted-token offset to one
            # after migrating the accepted state into a new block. QSA keeps
            # the pre-migration accepted count to advance its independent
            # selector interval, so the two consumers must not share this
            # batch-aligned buffer.
            num_accepted_tokens = self._prepare_mamba_acceptance(
                input_batch,
                num_reqs,
            )
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)
        if self._align_mode:
            self._prepare_aligned_state_indices(
                input_batch.seq_lens,
                num_reqs,
                attn_groups,
                kv_cache_config,
                block_tables,
            )

        model_metadata = Qwen4ExpAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            qsa_state_slot_ids=qsa_state_slot_ids,
            qsa_state_is_fresh=qsa_state_is_fresh,
            qsa_num_accepted_tokens=qsa_num_accepted_tokens,
            qsa_is_prefilling=qsa_is_prefilling,
            qsa_max_seq_len=(
                self.max_model_len
                if for_capture or input_batch.cudagraph_capture
                else None
            ),
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=model_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            uniform_decode_graph=input_batch.uniform_decode_graph,
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
            )
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if self.uses_qsa and idx_mapping.numel():
            if isinstance(num_sampled, int):
                _fill_qsa_request_state_kernel[(idx_mapping.numel(),)](
                    idx_mapping,
                    self.qsa_committed_num_accepted_tokens_gpu,
                    self.qsa_state_is_fresh_gpu,
                    max(num_sampled, 1),
                )
            else:
                _commit_qsa_request_state_kernel[(idx_mapping.numel(),)](
                    idx_mapping,
                    num_sampled,
                    self.qsa_committed_num_accepted_tokens_gpu,
                    self.qsa_state_is_fresh_gpu,
                )
        super().postprocess_state(
            idx_mapping,
            num_sampled,
            num_computed_tokens,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        for embedding in self.disk_embeddings:
            embedding.prepare_disk(
                input_batch.input_ids, query_start_loc, ngram_context
            )
        model_inputs.update(
            query_start_loc=query_start_loc, ngram_context=ngram_context
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        for embedding in self.disk_embeddings:
            embedding.prepare_dummy_output(num_tokens)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


@triton.jit
def _commit_qsa_request_state_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    qsa_num_accepted_ptr,
    state_is_fresh_ptr,
):
    row = tl.program_id(0)
    state_slot = tl.load(idx_mapping_ptr + row)
    if state_slot >= 0:
        num_sampled = tl.load(num_sampled_ptr + row)
        tl.store(qsa_num_accepted_ptr + state_slot, tl.maximum(num_sampled, 1))
        tl.store(state_is_fresh_ptr + state_slot, 0)


@triton.jit
def _fill_qsa_request_state_kernel(
    idx_mapping_ptr,
    qsa_num_accepted_ptr,
    state_is_fresh_ptr,
    num_sampled,
):
    row = tl.program_id(0)
    state_slot = tl.load(idx_mapping_ptr + row)
    if state_slot >= 0:
        tl.store(qsa_num_accepted_ptr + state_slot, num_sampled)
        tl.store(state_is_fresh_ptr + state_slot, 0)


__all__ = [
    "Qwen4ExpAttnMetadata",
    "Qwen4ExpModelState",
]
