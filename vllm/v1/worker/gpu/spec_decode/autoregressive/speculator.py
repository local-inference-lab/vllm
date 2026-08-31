# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)


def _sparse_full_capture_request_sizes(max_num_reqs: int) -> frozenset[int]:
    """Return power-of-two request capacities, including the configured maximum."""
    request_sizes = []
    request_size = 1
    while request_size < max_num_reqs:
        request_sizes.append(request_size)
        request_size *= 2
    request_sizes.append(max_num_reqs)
    return frozenset(request_sizes)


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.inputs_embeds: torch.Tensor | None = None

        self.mrope_positions: torch.Tensor | None = None
        self.mrope_positions_scratch: torch.Tensor | None = None
        if self.draft_model_config.uses_mrope:
            # The extra column preserves the non-contiguous layout expected by
            # torch.compile, matching the target RopeState buffer.
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device=device,
            )
            self.mrope_positions_scratch = torch.empty(
                (3, self.max_num_reqs),
                dtype=torch.int64,
                device=device,
            )

        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False
        self.prefill_outputs_are_compact = False

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not self.supports_mm_inputs:
            return

        self.inputs_embeds = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    # Lifecycle hooks for model-specific optimizations. Subclasses override
    # the ones they need. These fire in both `capture` and `propose` so that
    # any state they toggle (e.g. attention flags baked into a CUDA graph) is
    # identical at capture time and replay time.
    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """
        Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._configure_fused_multi_step_decode()

    def _configure_fused_multi_step_decode(self) -> None:
        if self.num_speculative_steps == 1:
            self.use_fused_multi_step_decode = False
            return

        if self.speculative_config.uses_dynamic_speculative_decoding():
            self.use_fused_multi_step_decode = False
            return

        if not self.advance_draft_positions:
            self.use_fused_multi_step_decode = True
            return

        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        self.use_fused_multi_step_decode = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Fused multi-step draft decode is not supported by attention "
                "backend(s) %s; falling back to rebuilding attention metadata "
                "between draft steps.",
                ", ".join(unsupported_backends),
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        full_capture_request_sizes = _sparse_full_capture_request_sizes(
            self.max_num_reqs
        )
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
            full_capture_request_sizes=full_capture_request_sizes,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        # Draft prefill reuses the target model's attention metadata at
        # runtime, so capture builds its dummy metadata through the target
        # model runner's builders and buffers.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)

        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            self.target_input_buffers,
            self.block_tables,
            self.target_attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        self.on_multi_step_decode_begin(self.max_num_reqs)
        # Capture either the fused decode loop or one decode step per graph.
        assert self.decode_cudagraph_manager is not None
        decode_fn = (
            self._generate_fused_drafts
            if self.use_fused_multi_step_decode
            else self._generate_draft
        )
        self.decode_cudagraph_manager.capture(
            decode_fn,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_speculative_tokens: int | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_steps
        if not 1 <= num_speculative_tokens <= self.num_speculative_steps:
            raise ValueError(
                "num_speculative_tokens must be between 1 and "
                f"{self.num_speculative_steps}, got {num_speculative_tokens}."
            )

        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + num_speculative_tokens, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens_padded].copy_(hidden_states)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        # Get the input ids and last token indices for the speculator.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
            target_model_positions=self._target_model_positions(
                input_batch, is_profile
            ),
            draft_mrope_positions=self.mrope_positions,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            num_tokens,
            max_query_len,
            input_batch.has_prefill,
        )
        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens_padded,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        self._prepare_eplb_forward(num_tokens)

        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )
        self.on_prefill_end(num_reqs)

        if num_speculative_tokens == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        # Prepare the inputs for the decode steps.
        prepare_decode_inputs(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.max_model_len,
            self.max_num_reqs,
            advance_draft_positions=self.advance_draft_positions,
            mrope_positions=self.mrope_positions,
        )

        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining draft tokens.
        try:
            if self.use_fused_multi_step_decode:
                assert num_speculative_tokens == self.num_speculative_steps
                self._fused_multi_step_decode(
                    num_reqs,
                    dummy_run and skip_attn_for_dummy_run,
                    decode_batch_desc,
                    num_tokens_across_dp,
                    input_batch.seq_lens_cpu_upper_bound,
                )
            else:
                self._multi_step_decode(
                    num_reqs,
                    dummy_run and skip_attn_for_dummy_run,
                    decode_batch_desc,
                    num_tokens_across_dp,
                    input_batch.seq_lens_cpu_upper_bound,
                    num_speculative_tokens,
                )
        finally:
            self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs, :num_speculative_tokens]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                assert self.inputs_embeds is not None
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self._model_positions(num_tokens),
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Draft prefill with PIECEWISE cudagraph (compiled PW or breakable),
                # chosen inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                ret_hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _model_positions(self, num_tokens: int) -> torch.Tensor:
        if self.mrope_positions is not None:
            return self.mrope_positions[:, :num_tokens]
        return self.input_buffers.positions[:num_tokens]

    def _target_model_positions(
        self, input_batch: InputBatch, is_profile: bool
    ) -> torch.Tensor | None:
        if self.mrope_positions is None:
            return None
        if not hasattr(self, "model_state"):
            # KV-cache profiling runs before set_attn() binds the target state.
            # Its dummy text positions are identical on all MRoPE axes.
            if not is_profile:
                raise RuntimeError("target model state is not bound for MRoPE drafting")
            return input_batch.positions
        return self.model_state.get_model_positions(input_batch)

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        if self.prefill_outputs_are_compact:
            sample_hidden_states = last_hidden_states[:num_reqs]
            feedback_hidden_states = hidden_states[:num_reqs]
        else:
            sample_hidden_states = last_hidden_states[last_token_indices]
            feedback_hidden_states = hidden_states[last_token_indices]

        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        if last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = sample_hidden_states
        else:
            self.hidden_states[:num_reqs] = feedback_hidden_states
        if self.mrope_positions is not None:
            assert self.mrope_positions_scratch is not None
            compact_mrope_positions(
                self.mrope_positions,
                self.mrope_positions_scratch,
                last_token_indices,
                num_reqs,
            )
        self.input_buffers.positions[:num_reqs] = positions

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_tokens: int | None = None,
    ) -> None:
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_steps
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, num_speculative_tokens):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                batch_desc.num_tokens,
            )
            if batch_desc.cg_mode != CUDAGraphMode.FULL:
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
                num_tokens_padded=batch_desc.num_tokens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1,
            )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)

        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        last_hidden_states = last_hidden_states[:num_reqs]

        sample_positions = positions
        if not self.advance_draft_positions:
            # The forward pass holds positions fixed (Q-only, shared target KV),
            # but Gumbel sampling still needs the absolute draft position.
            sample_positions = positions + self.current_draft_step

        # Sample the draft tokens.
        draft_tokens = self.sample_draft(
            last_hidden_states,
            sample_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
            mrope_positions=self.mrope_positions,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    draft_mrope_positions_ptr,
    draft_mrope_positions_stride,
    target_model_positions_ptr,
    target_model_positions_stride,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    USES_MROPE: tl.constexpr,
    TARGET_USES_MROPE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)
        if USES_MROPE:
            for axis in tl.static_range(3):
                source_axis = axis if TARGET_USES_MROPE else 0
                model_pos = tl.load(
                    target_model_positions_ptr
                    + source_axis * target_model_positions_stride
                    + query_start
                    + block,
                    mask=mask,
                )
                tl.store(
                    draft_mrope_positions_ptr
                    + axis * draft_mrope_positions_stride
                    + query_start
                    + block,
                    model_pos,
                    mask=mask,
                )

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
    target_model_positions: torch.Tensor | None = None,
    draft_mrope_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    uses_mrope = draft_mrope_positions is not None
    if uses_mrope:
        assert draft_mrope_positions is not None
        if draft_mrope_positions.ndim != 2 or draft_mrope_positions.shape[0] != 3:
            raise ValueError("draft MRoPE positions must have shape [3, capacity]")
        if target_model_positions is None:
            target_model_positions = input_batch.positions
        if target_model_positions.ndim not in (1, 2):
            raise ValueError("target model positions must be one- or two-dimensional")
        if target_model_positions.ndim == 2 and target_model_positions.shape[0] != 3:
            raise ValueError("target MRoPE positions must have shape [3, rows]")
    else:
        draft_mrope_positions = input_buffers.positions
        target_model_positions = input_batch.positions
    target_uses_mrope = target_model_positions.ndim == 2
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        draft_mrope_positions,
        draft_mrope_positions.stride(0) if uses_mrope else 0,
        target_model_positions,
        target_model_positions.stride(0) if target_uses_mrope else 0,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
        USES_MROPE=uses_mrope,
        TARGET_USES_MROPE=target_uses_mrope,
    )
    return last_token_indices


@triton.jit
def _gather_mrope_positions_kernel(
    mrope_positions_ptr,
    mrope_positions_stride,
    scratch_ptr,
    scratch_stride,
    last_token_indices_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    axis = tl.program_id(0)
    req_idx = tl.arange(0, BLOCK_SIZE)
    mask = req_idx < num_reqs
    last_token_idx = tl.load(last_token_indices_ptr + req_idx, mask=mask)
    positions = tl.load(
        mrope_positions_ptr + axis * mrope_positions_stride + last_token_idx,
        mask=mask,
    )
    tl.store(
        scratch_ptr + axis * scratch_stride + req_idx,
        positions,
        mask=mask,
    )


@triton.jit
def _store_compacted_mrope_positions_kernel(
    mrope_positions_ptr,
    mrope_positions_stride,
    scratch_ptr,
    scratch_stride,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    axis = tl.program_id(0)
    req_idx = tl.arange(0, BLOCK_SIZE)
    mask = req_idx < num_reqs
    positions = tl.load(
        scratch_ptr + axis * scratch_stride + req_idx,
        mask=mask,
    )
    tl.store(
        mrope_positions_ptr + axis * mrope_positions_stride + req_idx,
        positions,
        mask=mask,
    )


def compact_mrope_positions(
    mrope_positions: torch.Tensor,
    scratch: torch.Tensor,
    last_token_indices: torch.Tensor,
    num_reqs: int,
) -> None:
    if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
        raise ValueError("draft MRoPE positions must have shape [3, capacity]")
    if scratch.ndim != 2 or scratch.shape[0] != 3:
        raise ValueError("MRoPE compaction scratch must have shape [3, capacity]")
    if num_reqs < 0 or num_reqs > scratch.shape[1]:
        raise ValueError("MRoPE compaction scratch is too small for the batch")
    if num_reqs > last_token_indices.numel():
        raise ValueError("last token indices are too small for the batch")
    if num_reqs == 0:
        return
    block_size = triton.next_power_of_2(num_reqs)
    _gather_mrope_positions_kernel[(3,)](
        mrope_positions,
        mrope_positions.stride(0),
        scratch,
        scratch.stride(0),
        last_token_indices,
        num_reqs,
        BLOCK_SIZE=block_size,
    )
    _store_compacted_mrope_positions_kernel[(3,)](
        mrope_positions,
        mrope_positions.stride(0),
        scratch,
        scratch.stride(0),
        num_reqs,
        BLOCK_SIZE=block_size,
    )


@triton.jit
def _prepare_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    mrope_positions_ptr,
    mrope_positions_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    USES_MROPE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        # Compute position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        if USES_MROPE:
            model_position = tl.load(mrope_positions_ptr + req_idx)
            model_position = tl.minimum(model_position + 1, max_model_len - 1)
            for axis in tl.static_range(3):
                tl.store(
                    mrope_positions_ptr + axis * mrope_positions_stride + req_idx,
                    model_position,
                )
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
    mrope_positions: torch.Tensor | None = None,
):
    num_reqs = draft_tokens.shape[0]
    uses_mrope = mrope_positions is not None
    if mrope_positions is None:
        mrope_positions = input_buffers.positions
        mrope_positions_stride = 0
    else:
        mrope_positions_stride = mrope_positions.stride(0)
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        mrope_positions,
        mrope_positions_stride,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        USES_MROPE=uses_mrope,
    )


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    mrope_positions_ptr,
    mrope_positions_stride,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
    USES_MROPE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        if USES_MROPE:
            model_position = tl.load(mrope_positions_ptr + req_idx)
            model_position = tl.minimum(model_position + 1, max_model_len - 1)
            for axis in tl.static_range(3):
                tl.store(
                    mrope_positions_ptr + axis * mrope_positions_stride + req_idx,
                    model_position,
                )

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
    mrope_positions: torch.Tensor | None = None,
):
    _, hidden_size = hidden_states.shape
    uses_mrope = mrope_positions is not None
    if mrope_positions is None:
        mrope_positions = input_buffers.positions
        mrope_positions_stride = 0
    else:
        mrope_positions_stride = mrope_positions.stride(0)
    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        mrope_positions,
        mrope_positions_stride,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        USES_MROPE=uses_mrope,
    )
