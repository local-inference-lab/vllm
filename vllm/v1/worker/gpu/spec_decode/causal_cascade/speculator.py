# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CausalCascade speculative decoder.

This path is correctness-oriented: it gathers graph-copied sparse-MLA cache rows
from the native CausalCascade live-state backend, runs the trained
CausalCascade module eagerly, and samples draft slots from the native
DFlash-style block. Slot 0 is the verifier bonus token at position t+1; slots
1..N are draft predictions for later positions. The source anchor hidden state
and sparse-MLA rows come from the verifier row at position t. The optimized path
can replace the eager attention replay with the production sparse-MLA backend
once the query adapter projection is lowered into serving kernels.
"""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    PrefillSpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    _profile_cg_mode,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.causal_cascade.live_state import (
    configure_causal_cascade_live_state,
    get_causal_cascade_live_state,
)

logger = init_logger(__name__)


class CausalCascadeSpeculator(AutoRegressiveSpeculator):
    supports_mm_inputs: bool = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        anchor_source = str(
            getattr(
                self.draft_model_config.hf_config,
                "anchor_hidden_state_source",
                "",
            )
        )
        if anchor_source != "native_glm_mtp_post_shared_head_rmsnorm":
            raise ValueError(
                "The unified CausalCascade runtime requires "
                "anchor_hidden_state_source="
                "'native_glm_mtp_post_shared_head_rmsnorm', got "
                f"{anchor_source!r}"
            )
        self.mtp_anchor_hidden_states = torch.empty(
            self.max_num_reqs,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        logger.info(
            "CausalCascade unified MTP anchor enabled: the embedded MTP layer "
            "will run once before the block draft."
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group()
        self._debug_propose_calls = 0
        self._debug_fallback_counts: dict[str, int] = {}
        self._debug_success_calls = 0
        self._debug_position_mismatch_count = 0
        self._use_capture_positions = os.environ.get(
            "CAUSAL_CASCADE_USE_CAPTURE_POSITIONS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._debug_dump_dir = os.environ.get("CAUSAL_CASCADE_DEBUG_DUMP_DIR")
        self._debug_dump_limit = int(
            os.environ.get("CAUSAL_CASCADE_DEBUG_DUMP_LIMIT", "0")
        )
        self._debug_dump_max_reqs = int(
            os.environ.get("CAUSAL_CASCADE_DEBUG_DUMP_MAX_REQS", "2")
        )
        self._debug_dump_count = 0
        self._ablate_cross_attention = os.environ.get(
            "CAUSAL_CASCADE_ABLATE_CROSS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._first_draft_slot = int(
            os.environ.get("CAUSAL_CASCADE_FIRST_DRAFT_SLOT", "1")
        )
        self._min_context_tokens = int(
            os.environ.get("CAUSAL_CASCADE_MIN_CONTEXT_TOKENS", "0")
        )
        if self._ablate_cross_attention:
            logger.warning(
                "CausalCascadeSpeculator live sparse-MLA cross attention is ablated"
            )
        logger.info(
            "CausalCascadeSpeculator first live draft slot=%d "
            "use_capture_positions=%s min_context_tokens=%d",
            self._first_draft_slot,
            self._use_capture_positions,
            self._min_context_tokens,
        )
        if self._debug_dump_dir and self._debug_dump_limit > 0:
            logger.warning(
                "CausalCascadeSpeculator live debug dumps enabled: dir=%s limit=%d",
                self._debug_dump_dir,
                self._debug_dump_limit,
            )

    def _debug_position_mismatch(
        self,
        capture_anchor_pos: torch.Tensor,
        input_anchor_pos: torch.Tensor,
    ) -> None:
        mismatch = capture_anchor_pos.ne(input_anchor_pos)
        if not bool(mismatch.any().item()):
            return
        self._debug_position_mismatch_count += 1
        count = self._debug_position_mismatch_count
        if not (count <= 8 or count & (count - 1) == 0):
            return
        delta = capture_anchor_pos.to(torch.long) - input_anchor_pos.to(torch.long)
        logger.warning(
            "CausalCascadeSpeculator capture/input anchor position mismatch "
            "count=%d use_capture_positions=%s mismatched_rows=%d "
            "delta_min=%d delta_max=%d capture_range=[%d,%d] input_range=[%d,%d]",
            count,
            self._use_capture_positions,
            int(mismatch.sum().item()),
            int(delta.min().item()),
            int(delta.max().item()),
            int(capture_anchor_pos.min().item()),
            int(capture_anchor_pos.max().item()),
            int(input_anchor_pos.min().item()),
            int(input_anchor_pos.max().item()),
        )

    def _debug_fallback(self, reason: str, num_reqs: int) -> None:
        count = self._debug_fallback_counts.get(reason, 0) + 1
        self._debug_fallback_counts[reason] = count
        if count <= 5 or count & (count - 1) == 0:
            logger.warning(
                "CausalCascadeSpeculator fallback reason=%s count=%d "
                "propose_calls=%d num_reqs=%d",
                reason,
                count,
                self._debug_propose_calls,
                num_reqs,
            )

    def _live_sparse_rows_ready(
        self,
        live_inputs: dict[str, torch.Tensor],
        *,
        num_reqs: int,
        configured_topk: int | None,
        input_anchor_pos: torch.Tensor,
    ) -> bool:
        if self._min_context_tokens > 0:
            context_len = input_anchor_pos[:num_reqs].to(torch.long) + 1
            if bool(context_len.lt(self._min_context_tokens).any().item()):
                self._debug_fallback("short_context_for_sparse_mla", num_reqs)
                return False

        topk = None if configured_topk is None else int(configured_topk)
        valid_mask = live_inputs.get("mla_cache_valid_mask")
        if valid_mask is not None:
            mask = valid_mask[:num_reqs]
            if topk is not None and mask.shape[-1] >= topk:
                mask = mask[..., :topk]
            if mask.ndim >= 3:
                ready = mask.reshape(num_reqs, -1, mask.shape[-1]).any(dim=-1)
                ready = ready.all(dim=-1)
            else:
                ready = mask.reshape(num_reqs, -1).any(dim=1)
            if not bool(ready.all().item()):
                self._debug_fallback("incomplete_sparse_mla_rows", num_reqs)
                return False

        topk_indices = live_inputs.get("mla_cache_topk_indices")
        if topk_indices is not None:
            indices = topk_indices[:num_reqs]
            if topk is not None and indices.shape[-1] >= topk:
                indices = indices[..., :topk]
            if valid_mask is not None:
                mask = valid_mask[:num_reqs]
                if topk is not None and mask.shape[-1] >= topk:
                    mask = mask[..., :topk]
                bad_valid_indices = mask.bool() & indices.lt(0)
                if bool(bad_valid_indices.reshape(num_reqs, -1).any().item()):
                    self._debug_fallback("negative_sparse_mla_indices", num_reqs)
                    return False
            else:
                nonnegative = indices.ge(0)
                if indices.ndim >= 3:
                    ready = nonnegative.reshape(
                        num_reqs, -1, nonnegative.shape[-1]
                    ).any(dim=-1)
                    ready = ready.all(dim=-1)
                else:
                    ready = nonnegative.reshape(num_reqs, -1).any(dim=1)
                if not bool(ready.all().item()):
                    self._debug_fallback("negative_sparse_mla_indices", num_reqs)
                    return False

        selected_nsa_lens = live_inputs.get("selected_nsa_lens")
        if topk is not None and selected_nsa_lens is not None:
            lens = selected_nsa_lens[:num_reqs].to(torch.long)
            if bool(lens.reshape(num_reqs, -1).le(0).any().item()):
                self._debug_fallback("short_sparse_mla_nsa_lens", num_reqs)
                return False

        return True

    def _debug_success(
        self,
        num_reqs: int,
        live_inputs: dict[str, torch.Tensor],
        anchor_hidden_state: torch.Tensor,
        first_draft_slot: int,
        hidden_source: str,
        hidden_max_abs_diff: float | None,
    ) -> None:
        self._debug_success_calls += 1
        count = self._debug_success_calls
        if not (count <= 5 or count & (count - 1) == 0):
            return
        sample_tokens = self.draft_tokens[: min(num_reqs, 2)].detach().cpu().tolist()
        rows = live_inputs["mla_cache_rows_packed"]
        position_ids = live_inputs["position_ids"]
        physical_slots = live_inputs["mla_cache_physical_slots"]
        topk_indices = live_inputs.get("mla_cache_topk_indices")
        topk_min = -1 if topk_indices is None else int(topk_indices.min().item())
        topk_max = -1 if topk_indices is None else int(topk_indices.max().item())
        logger.info(
            "CausalCascadeSpeculator proposal count=%d propose_calls=%d "
            "num_reqs=%d first_draft_slot=%d rows_shape=%s "
            "rows_dtype=%s rows_abs_mean=%.6f "
            "physical_slots=[%d,%d] topk_indices=[%d,%d] "
            "anchor_norm_mean=%.6f hidden_source=%s hidden_max_abs_diff=%s "
            "pos0=%s draft_tokens=%s",
            count,
            self._debug_propose_calls,
            num_reqs,
            first_draft_slot,
            tuple(rows.shape),
            rows.dtype,
            float(rows.float().abs().mean().item()),
            int(physical_slots.min().item()),
            int(physical_slots.max().item()),
            topk_min,
            topk_max,
            float(anchor_hidden_state.float().norm(dim=-1).mean().item()),
            hidden_source,
            ("n/a" if hidden_max_abs_diff is None else f"{hidden_max_abs_diff:.6f}"),
            position_ids[: min(num_reqs, 2)].detach().cpu().tolist(),
            sample_tokens,
        )

    def _finish_proposal(
        self,
        num_reqs: int,
        *,
        num_speculative_tokens: int | None = None,
        broadcast: bool = True,
    ) -> torch.Tensor:
        if num_speculative_tokens is None:
            num_speculative_tokens = getattr(
                self,
                "_active_num_speculative_tokens",
                self.num_speculative_steps,
            )
        draft_tokens = self.draft_tokens[:num_reqs, :num_speculative_tokens].clone()
        if broadcast and self.tp_group.world_size > 1:
            self.tp_group.broadcast(draft_tokens, src=0)
        return draft_tokens

    @staticmethod
    def _cpu_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().to(device="cpu").contiguous()

    def _maybe_dump_live_inputs(
        self,
        *,
        num_reqs: int,
        row_indices: torch.Tensor,
        capture_row_indices: torch.Tensor,
        valid_row_ends: torch.Tensor,
        live_inputs: dict[str, torch.Tensor],
        capture_anchor_pos: torch.Tensor,
        input_anchor_pos: torch.Tensor,
        capture_position_ids: torch.Tensor,
        input_position_ids: torch.Tensor,
        anchor_hidden_state: torch.Tensor,
        aux_anchor_hidden_state: torch.Tensor,
        captured_anchor_hidden_state: torch.Tensor | None,
        logits: torch.Tensor,
        step_logits: torch.Tensor,
        known_token_ids: torch.Tensor | None,
        verifier_bonus_token_ids: torch.Tensor | None,
        hidden_source: str,
        hidden_max_abs_diff: float | None,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        input_batch: InputBatch,
        first_draft_slot: int,
    ) -> None:
        if (
            not self._debug_dump_dir
            or self._debug_dump_limit <= 0
            or self._debug_dump_count >= self._debug_dump_limit
        ):
            return

        self._debug_dump_count += 1
        dump_dir = Path(self._debug_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_reqs = min(num_reqs, max(self._debug_dump_max_reqs, 1))

        def first_reqs(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.ndim > 0 and tensor.shape[0] >= num_reqs:
                tensor = tensor[:dump_reqs]
            return self._cpu_tensor(tensor)

        payload: dict[str, Any] = {
            "num_reqs": int(num_reqs),
            "dump_reqs": int(dump_reqs),
            "req_ids": list(input_batch.req_ids[:dump_reqs]),
            "first_draft_slot": int(first_draft_slot),
            "num_speculative_steps": int(self.num_speculative_steps),
            "block_size": int(self.model.block_size),
            "hidden_source": hidden_source,
            "hidden_max_abs_diff": hidden_max_abs_diff,
            "use_capture_positions": bool(self._use_capture_positions),
            "ablate_cross_attention": bool(self._ablate_cross_attention),
            "known_token_conditioning": getattr(
                self.model.config,
                "known_token_conditioning",
                None,
            ),
            "anchor_token_conditioning": getattr(
                self.model.config,
                "anchor_token_conditioning",
                None,
            ),
            "markov_head_rank": int(getattr(self.model.config, "markov_head_rank", 0)),
            "row_indices": self._cpu_tensor(row_indices[:dump_reqs]),
            "capture_row_indices": self._cpu_tensor(capture_row_indices[:dump_reqs]),
            "valid_row_ends": self._cpu_tensor(valid_row_ends[:dump_reqs]),
            "num_rejected": self._cpu_tensor(num_rejected[:dump_reqs]),
            "num_sampled": self._cpu_tensor(num_sampled[:dump_reqs]),
            "temperature": self._cpu_tensor(
                temperature.index_select(
                    0,
                    input_batch.idx_mapping[:dump_reqs].to(
                        device=temperature.device,
                        dtype=torch.long,
                    ),
                )
            ),
            "query_start_loc": self._cpu_tensor(
                input_batch.query_start_loc[: dump_reqs + 1]
            ),
            "cu_num_logits": self._cpu_tensor(
                input_batch.cu_num_logits[: dump_reqs + 1]
            ),
            "logits_indices": self._cpu_tensor(input_batch.logits_indices),
            "expanded_local_pos": self._cpu_tensor(input_batch.expanded_local_pos),
            "verified_input_ids": self._cpu_tensor(
                input_batch.input_ids.index_select(0, input_batch.logits_indices)
            ),
            "verified_positions": self._cpu_tensor(
                input_batch.positions.index_select(0, input_batch.logits_indices)
            ),
            "last_sampled": self._cpu_tensor(
                last_sampled.index_select(
                    0,
                    input_batch.idx_mapping[:dump_reqs].to(
                        device=last_sampled.device,
                        dtype=torch.long,
                    ),
                )
            ),
            "next_prefill_tokens": self._cpu_tensor(
                next_prefill_tokens.index_select(
                    0,
                    input_batch.idx_mapping[:dump_reqs].to(
                        device=next_prefill_tokens.device,
                        dtype=torch.long,
                    ),
                )
            ),
            "verifier_bonus_token_ids": first_reqs(verifier_bonus_token_ids),
            "known_token_ids": first_reqs(known_token_ids),
            "input_batch_positions_at_rows": self._cpu_tensor(
                input_anchor_pos[:dump_reqs]
            ),
            "capture_anchor_pos": self._cpu_tensor(capture_anchor_pos[:dump_reqs]),
            "used_anchor_pos": self._cpu_tensor(live_inputs["anchor_pos"][:dump_reqs]),
            "source_anchor_token_ids": first_reqs(
                live_inputs.get("source_anchor_token_ids")
            ),
            "source_anchor_token_ids_valid": first_reqs(
                live_inputs.get("source_anchor_token_ids_valid")
            ),
            "selected_cache_lens": first_reqs(live_inputs.get("selected_cache_lens")),
            "selected_nsa_lens": first_reqs(live_inputs.get("selected_nsa_lens")),
            "selected_req_ids": first_reqs(live_inputs.get("selected_req_ids")),
            "debug_cache_lens_head": self._cpu_tensor(
                live_inputs.get("debug_cache_lens_head")
            ),
            "debug_nsa_lens_head": self._cpu_tensor(
                live_inputs.get("debug_nsa_lens_head")
            ),
            "debug_req_ids_head": self._cpu_tensor(
                live_inputs.get("debug_req_ids_head")
            ),
            "capture_position_ids": self._cpu_tensor(capture_position_ids[:dump_reqs]),
            "input_position_ids": self._cpu_tensor(input_position_ids[:dump_reqs]),
            "used_position_ids": self._cpu_tensor(
                live_inputs["position_ids"][:dump_reqs]
            ),
            "anchor_token_ids": first_reqs(live_inputs["anchor_token_ids"]),
            "verifier_layer_ids": self._cpu_tensor(live_inputs["verifier_layer_ids"]),
            "mla_cache_rows_packed": first_reqs(live_inputs["mla_cache_rows_packed"]),
            "mla_cache_valid_mask": first_reqs(live_inputs.get("mla_cache_valid_mask")),
            "mla_cache_topk_indices": first_reqs(
                live_inputs.get("mla_cache_topk_indices")
            ),
            "mla_cache_physical_slots": first_reqs(
                live_inputs.get("mla_cache_physical_slots")
            ),
            "anchor_hidden_state": first_reqs(anchor_hidden_state),
            "aux_anchor_hidden_state": first_reqs(aux_anchor_hidden_state),
            "captured_anchor_hidden_state": first_reqs(captured_anchor_hidden_state),
            "logits": first_reqs(logits),
            "step_logits": first_reqs(step_logits),
            "draft_tokens": self._cpu_tensor(self.draft_tokens[:dump_reqs]),
        }
        path = dump_dir / f"live_proposal_{self._debug_dump_count:04d}.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        logger.warning("Wrote CausalCascade live debug dump: %s", path)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Only the one-pass embedded MTP prefill needs a graph. The seven
        # CausalCascade output positions are produced by one block forward, so
        # allocating a native MTP decode graph would be both wasteful and wrong.
        self.prefill_cudagraph_manager = PrefillSpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            2,
        )
        self.decode_cudagraph_manager = None

    def capture(
        self,
        attn_states: dict[Any, AttentionStatePair],
    ) -> None:
        del attn_states
        logger.info("Capturing embedded CausalCascade MTP prefill...")
        self.last_token_indices.zero_()
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        # The embedded MTP layer has its own attention layer and must always
        # build metadata from its one-step topology; verifier metadata describes
        # the wider verification block.
        self.prefill_cudagraph_manager.capture(
            self._run_embedded_mtp_graph,
            model_state=self.model_state,
            input_buffers=self.input_buffers,
            block_tables=self.block_tables,
            attn_groups=self.attn_groups,
            kv_cache_config=self.kv_cache_config,
            progress_bar_desc="Capturing embedded MTP CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_attn_layer_names
        draft_model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.draft_model_config,
        )
        draft_model.attach_target_shared_weights(target_model)
        populate = getattr(
            draft_model,
            "populate_target_compatible_mla_weights",
            None,
        )
        if populate is not None:
            populate(target_model)
        configure_causal_cascade_live_state(
            self.vllm_config,
            layer_ids=list(draft_model.target_layer_ids),
            topk=int(getattr(draft_model.config, "sparse_topk", 2048)),
            block_size=int(draft_model.block_size),
        )
        return draft_model

    def set_attn(self, model_state, kv_cache_config, block_tables) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        self.rebuild_prefill_attn_metadata = True

    def _run_embedded_mtp_graph(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        _, mtp_head_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self.mtp_anchor_hidden_states[:num_reqs].copy_(
            mtp_head_states[self.last_token_indices[:num_reqs]]
        )

    def _run_embedded_mtp_prefill(
        self,
        *,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None,
        is_profile: bool,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(max_seq_len + 1, self.max_model_len)
        self.hidden_states[:num_tokens].copy_(last_hidden_states)
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )
        self.active_num_reqs.fill_(num_reqs)
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
        )

        uniform_token_count = get_uniform_token_count(
            num_reqs,
            input_batch.num_tokens,
            max_query_len,
        )
        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            max_req_tokens=None,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        prefill_attn_metadata = attn_metadata
        prefill_slot_mappings = slot_mappings
        if getattr(self, "rebuild_prefill_attn_metadata", False):
            idx_mapping = self.idx_mapping[:num_reqs]
            rebuilt_slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                self.input_buffers.query_start_loc,
                self.input_buffers.positions,
                prefill_batch_desc.num_tokens,
            )
            prefill_slot_mappings = build_slot_mappings_by_layer(
                rebuilt_slot_mappings,
                self.kv_cache_config,
            )
            num_query_per_req = uniform_token_count or max_query_len
            prefill_attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=prefill_batch_desc.num_reqs or num_reqs,
                num_tokens_padded=prefill_batch_desc.num_tokens,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                max_seq_len_upper_bound=max_seq_len,
                query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            )
        self._prepare_eplb_forward(input_batch.num_tokens)

        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.prefill_cudagraph_manager is not None
            with record_function_or_nullcontext(
                "vllm:v2/speculator/causal_cascade/mtp/full_graph_replay"
            ):
                self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            with record_function_or_nullcontext(
                "vllm:v2/speculator/causal_cascade/mtp/"
                f"{_profile_cg_mode(prefill_batch_desc.cg_mode)}"
            ):
                self._run_embedded_mtp_graph(
                    num_reqs,
                    prefill_batch_desc.num_tokens,
                    prefill_attn_metadata,
                    prefill_slot_mappings,
                    num_tokens_across_dp,
                    prefill_batch_desc.cg_mode,
                )
        return self.mtp_anchor_hidden_states[:num_reqs]

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_speculative_tokens: int | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        del skip_attn_for_dummy_run, mm_inputs
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_steps
        if not 1 <= num_speculative_tokens <= self.num_speculative_steps:
            raise ValueError(
                "num_speculative_tokens must be between 1 and "
                f"{self.num_speculative_steps}, got {num_speculative_tokens}"
            )
        self._active_num_speculative_tokens = num_speculative_tokens
        native_mtp_anchor_hidden_state = self._run_embedded_mtp_prefill(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            is_profile=is_profile,
        )
        self.draft_tokens[:num_reqs].fill_(-1)

        if dummy_run:
            return self._finish_proposal(
                num_reqs,
                num_speculative_tokens=num_speculative_tokens,
                broadcast=False,
            )
        self._debug_propose_calls += 1
        if input_batch.is_prefilling_np[:num_reqs].any():
            self._debug_fallback("prefill_in_progress", num_reqs)
            return self._finish_proposal(num_reqs)
        if aux_hidden_states is None or not aux_hidden_states:
            self._debug_fallback("missing_aux_hidden_states", num_reqs)
            return self._finish_proposal(num_reqs)
        live_state = get_causal_cascade_live_state()
        if live_state is None:
            self._debug_fallback("missing_causal_cascade_live_state", num_reqs)
            return self._finish_proposal(num_reqs)

        get_inputs = getattr(live_state, "get_live_causal_cascade_inputs", None)
        if get_inputs is None:
            self._debug_fallback("live_state_missing_live_inputs", num_reqs)
            return self._finish_proposal(num_reqs)

        block_size = int(self.model.block_size)
        known_token_conditioning = getattr(
            self.model.config,
            "known_token_conditioning",
            "none",
        )
        markov_head_enabled = bool(getattr(self.model, "markov_head_enabled", False))
        slot1_native_anchor_enabled = bool(
            getattr(self.model, "slot1_native_anchor_enabled", False)
        )
        first_draft_slot = self._first_draft_slot
        if first_draft_slot < 0 or first_draft_slot >= block_size:
            raise RuntimeError(
                "CausalCascade first draft slot must be inside the trained "
                f"block: got {first_draft_slot}, block_size={block_size}"
            )
        max_draft_steps = block_size - first_draft_slot
        if num_speculative_tokens > max_draft_steps:
            raise RuntimeError(
                "CausalCascade num_speculative_tokens exceeds the trained "
                "block for its configured slot mapping: "
                f"got {num_speculative_tokens}, max={max_draft_steps}, "
                f"block_size={block_size}, first_draft_slot={first_draft_slot}, "
                f"known_token_conditioning={known_token_conditioning!r}"
            )
        if markov_head_enabled and first_draft_slot != 1:
            raise RuntimeError(
                "CausalCascade Markov decoding requires first_draft_slot=1 so "
                "the first transition is conditioned on the known verifier "
                f"bonus token; got {first_draft_slot}"
            )
        configured_topk = getattr(self.model.config, "sparse_topk", None)
        valid_row_ends = (
            input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        )
        aux_row_indices = valid_row_ends - 1
        # The live capture buffers are ordered like the flattened decode
        # batch, the same coordinate system used by aux_hidden_states. Using a
        # per-request relative row is only correct for a single active request
        # and silently mixes requests under concurrent serving.
        capture_row_indices = aux_row_indices
        if torch.any(aux_row_indices < 0):
            self._debug_fallback("negative_row_indices", num_reqs)
            return self._finish_proposal(num_reqs)
        if torch.any(capture_row_indices < 0):
            self._debug_fallback("negative_capture_row_indices", num_reqs)
            return self._finish_proposal(num_reqs)

        req_state_indices = input_batch.idx_mapping[:num_reqs].to(
            device=last_sampled.device,
            dtype=torch.long,
        )
        sampled_bonus_token_ids = (
            last_sampled.index_select(
                0,
                req_state_indices,
            )
            .to(device=input_batch.input_ids.device, dtype=torch.long)
            .reshape(-1)
        )
        prefill_bonus_token_ids = (
            next_prefill_tokens.index_select(
                0,
                req_state_indices.to(device=next_prefill_tokens.device),
            )
            .to(device=input_batch.input_ids.device, dtype=torch.long)
            .reshape(-1)
        )
        has_sampled_token = (
            num_sampled[:num_reqs]
            .to(
                device=input_batch.input_ids.device,
                dtype=torch.long,
            )
            .reshape(-1)
            > 0
        )
        verifier_bonus_token_ids = torch.where(
            has_sampled_token,
            sampled_bonus_token_ids[:num_reqs],
            prefill_bonus_token_ids[:num_reqs],
        )

        input_anchor_token_ids = input_batch.input_ids.index_select(
            0,
            aux_row_indices.to(device=input_batch.input_ids.device, dtype=torch.long),
        ).to(dtype=torch.long)

        live_inputs = get_inputs(
            list(self.model.target_layer_ids),
            block_size=block_size,
            topk=None if configured_topk is None else int(configured_topk),
            row_indices=capture_row_indices,
        )
        if live_inputs is None:
            self._debug_fallback("missing_live_inputs", num_reqs)
            return self._finish_proposal(num_reqs)
        anchor_token_ids_valid = live_inputs.get("anchor_token_ids_valid")
        live_inputs = dict(live_inputs)
        source_anchor_token_ids = input_anchor_token_ids
        source_anchor_token_ids_valid = torch.ones(
            num_reqs,
            device=input_anchor_token_ids.device,
            dtype=torch.bool,
        )
        if anchor_token_ids_valid is not None:
            valid_anchor_tokens = (
                anchor_token_ids_valid[:num_reqs]
                .to(
                    device=input_anchor_token_ids.device,
                    dtype=torch.bool,
                )
                .reshape(-1)
            )
            capture_anchor_token_ids = (
                live_inputs["anchor_token_ids"][:num_reqs]
                .to(device=input_anchor_token_ids.device, dtype=torch.long)
                .reshape(-1)
            )
            input_anchor_token_ids = input_anchor_token_ids.reshape(-1)
            source_anchor_token_ids = torch.where(
                valid_anchor_tokens,
                capture_anchor_token_ids,
                input_anchor_token_ids,
            )
            source_anchor_token_ids_valid = valid_anchor_tokens
            token_mismatch = valid_anchor_tokens & capture_anchor_token_ids.ne(
                input_anchor_token_ids,
            )
            if bool(token_mismatch.any().item()):
                self._debug_fallback("capture_anchor_token_mismatch", num_reqs)
        live_inputs["source_anchor_token_ids"] = source_anchor_token_ids
        live_inputs["source_anchor_token_ids_valid"] = source_anchor_token_ids_valid

        aux_anchor_hidden_state = aux_hidden_states[-1].index_select(
            0,
            aux_row_indices.to(device=aux_hidden_states[-1].device, dtype=torch.long),
        )
        verifier_head_hidden_state = last_hidden_states.index_select(
            0,
            aux_row_indices.to(device=last_hidden_states.device, dtype=torch.long),
        )
        get_anchor_hidden_states = getattr(
            live_state,
            "get_live_anchor_hidden_states",
            None,
        )
        captured_anchor_hidden_state = (
            get_anchor_hidden_states(capture_row_indices)
            if get_anchor_hidden_states
            else None
        )
        hidden_source = "aux"
        hidden_max_abs_diff: float | None = None
        anchor_hidden_state = aux_anchor_hidden_state
        if captured_anchor_hidden_state is not None:
            captured_anchor_hidden_state = captured_anchor_hidden_state.to(
                device=aux_anchor_hidden_state.device,
                dtype=aux_anchor_hidden_state.dtype,
            )
            hidden_max_abs_diff = float(
                (captured_anchor_hidden_state.float() - aux_anchor_hidden_state.float())
                .abs()
                .max()
                .item()
            )
            anchor_hidden_state = captured_anchor_hidden_state
            hidden_source = "capture"
        if native_mtp_anchor_hidden_state is not None:
            anchor_hidden_state = native_mtp_anchor_hidden_state.to(
                device=aux_anchor_hidden_state.device,
                dtype=aux_anchor_hidden_state.dtype,
            )
            hidden_source = "native_mtp"
        if live_inputs["mla_cache_rows_packed"].shape[0] < num_reqs:
            self._debug_fallback("short_live_inputs", num_reqs)
            return self._finish_proposal(num_reqs)
        input_anchor_pos = input_batch.positions.index_select(
            0,
            aux_row_indices.to(device=input_batch.positions.device, dtype=torch.long),
        ).to(dtype=torch.long)
        if not self._live_sparse_rows_ready(
            live_inputs,
            num_reqs=num_reqs,
            configured_topk=None if configured_topk is None else int(configured_topk),
            input_anchor_pos=input_anchor_pos,
        ):
            return self._finish_proposal(num_reqs)
        capture_anchor_pos = live_inputs["anchor_pos"][:num_reqs].to(
            device=input_anchor_pos.device,
            dtype=torch.long,
        )
        self._debug_position_mismatch(capture_anchor_pos, input_anchor_pos)
        anchor_pos = (
            capture_anchor_pos if self._use_capture_positions else input_anchor_pos
        )
        block_offsets = torch.arange(
            block_size,
            device=anchor_pos.device,
            dtype=torch.long,
        )
        capture_position_ids = live_inputs["position_ids"][:num_reqs]
        input_position_ids = (
            input_anchor_pos.unsqueeze(1) + 1 + block_offsets.unsqueeze(0)
        )
        live_inputs = dict(live_inputs)
        live_inputs["anchor_pos"] = anchor_pos
        live_inputs["known_token_pos"] = anchor_pos + 1
        live_inputs["position_ids"] = (
            anchor_pos.unsqueeze(1) + 1 + block_offsets.unsqueeze(0)
        )
        live_inputs["anchor_token_ids"] = verifier_bonus_token_ids.reshape(-1)
        live_inputs["anchor_token_ids_valid"] = torch.ones(
            num_reqs,
            device=verifier_bonus_token_ids.device,
            dtype=torch.bool,
        )
        known_token_ids = None
        if known_token_conditioning != "none" or markov_head_enabled:
            known_token_ids = verifier_bonus_token_ids.reshape(-1)[:num_reqs]

        mla_cache_valid_mask = live_inputs.get("mla_cache_valid_mask")
        logits = self.model.forward_logits(
            anchor_hidden_state=anchor_hidden_state,
            verifier_head_hidden_state=verifier_head_hidden_state,
            verifier_pre_norm_hidden_state=aux_anchor_hidden_state,
            anchor_token_ids=live_inputs["anchor_token_ids"][:num_reqs],
            mla_cache_rows_packed=live_inputs["mla_cache_rows_packed"][:num_reqs],
            mla_cache_valid_mask=mla_cache_valid_mask[:num_reqs]
            if mla_cache_valid_mask is not None
            else None,
            verifier_layer_ids=live_inputs["verifier_layer_ids"],
            position_ids=live_inputs["position_ids"][:num_reqs],
            known_token_ids=known_token_ids,
            ablate_sparse_mla_cross_attention=self._ablate_cross_attention,
        )
        step_logits = logits[
            :,
            first_draft_slot : first_draft_slot + num_speculative_tokens,
        ].contiguous()

        # The shared tensor-parallel verifier head has now completed all
        # collectives. Rank 0 alone performs the replicated Markov correction
        # and sampling; other ranks join the final token broadcast.
        if self.tp_rank != 0:
            return self._finish_proposal(num_reqs)

        if self.draft_logits is None:
            if markov_head_enabled:
                assert known_token_ids is not None
                previous_token_ids = known_token_ids
                corrected_step_logits: list[torch.Tensor] = []
                for draft_step in range(num_speculative_tokens):
                    # Training leaves slot 1's native-MTP residual prediction
                    # untouched and starts Markov conditioning at slot 2.
                    if draft_step == 0 and slot1_native_anchor_enabled:
                        current_logits = step_logits[:, draft_step]
                    else:
                        current_logits = self.model.apply_markov_head(
                            step_logits[:, draft_step],
                            previous_token_ids,
                        )
                    previous_token_ids = current_logits.argmax(dim=-1)
                    corrected_step_logits.append(current_logits)
                    self.draft_tokens[:num_reqs, draft_step] = previous_token_ids
                step_logits = torch.stack(corrected_step_logits, dim=1)
            else:
                self.draft_tokens[:num_reqs, :num_speculative_tokens] = (
                    step_logits.argmax(dim=-1)
                )
        else:
            self._copy_request_inputs(
                num_reqs,
                input_batch.idx_mapping,
                temperature,
                seeds,
            )
            if markov_head_enabled:
                assert known_token_ids is not None
                previous_token_ids = known_token_ids
                corrected_step_logits = []
                for draft_step in range(num_speculative_tokens):
                    if draft_step == 0 and slot1_native_anchor_enabled:
                        current_logits = step_logits[:, draft_step]
                    else:
                        current_logits = self.model.apply_markov_head(
                            step_logits[:, draft_step],
                            previous_token_ids,
                        )
                    current_positions = live_inputs["position_ids"][
                        :num_reqs,
                        first_draft_slot + draft_step,
                    ]
                    output_col = torch.full(
                        (num_reqs,),
                        draft_step,
                        device=current_logits.device,
                        dtype=torch.int32,
                    )
                    previous_token_ids = gumbel_sample(
                        current_logits,
                        self.idx_mapping[:num_reqs],
                        self.temperature,
                        self.seeds,
                        current_positions,
                        apply_temperature=True,
                        output_processed_logits=self.draft_logits,
                        output_processed_logits_col=output_col,
                        use_fp64=self.use_fp64_gumbel,
                    )
                    corrected_step_logits.append(current_logits)
                    self.draft_tokens[:num_reqs, draft_step] = previous_token_ids
                step_logits = torch.stack(corrected_step_logits, dim=1)
            else:
                flat_logits = step_logits.view(-1, step_logits.shape[-1])
                expanded_idx_mapping = self.idx_mapping[:num_reqs].repeat_interleave(
                    num_speculative_tokens
                )
                positions = live_inputs["position_ids"][
                    :num_reqs,
                    first_draft_slot : first_draft_slot + num_speculative_tokens,
                ].reshape(-1)
                draft_step = (
                    torch.arange(
                        num_speculative_tokens,
                        device=step_logits.device,
                        dtype=torch.int32,
                    )
                    .unsqueeze(0)
                    .expand(num_reqs, -1)
                    .reshape(-1)
                )
                sampled = gumbel_sample(
                    flat_logits,
                    expanded_idx_mapping,
                    self.temperature,
                    self.seeds,
                    positions,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=draft_step,
                    use_fp64=self.use_fp64_gumbel,
                )
                self.draft_tokens[:num_reqs, :num_speculative_tokens] = sampled.view(
                    num_reqs,
                    num_speculative_tokens,
                )
        self._maybe_dump_live_inputs(
            num_reqs=num_reqs,
            row_indices=aux_row_indices,
            capture_row_indices=capture_row_indices,
            valid_row_ends=valid_row_ends,
            live_inputs=live_inputs,
            capture_anchor_pos=capture_anchor_pos,
            input_anchor_pos=input_anchor_pos,
            capture_position_ids=capture_position_ids,
            input_position_ids=input_position_ids,
            anchor_hidden_state=anchor_hidden_state,
            aux_anchor_hidden_state=aux_anchor_hidden_state,
            captured_anchor_hidden_state=captured_anchor_hidden_state,
            logits=logits,
            step_logits=step_logits,
            known_token_ids=known_token_ids,
            verifier_bonus_token_ids=verifier_bonus_token_ids,
            hidden_source=hidden_source,
            hidden_max_abs_diff=hidden_max_abs_diff,
            num_rejected=num_rejected,
            num_sampled=num_sampled,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            input_batch=input_batch,
            first_draft_slot=first_draft_slot,
        )
        self._debug_success(
            num_reqs,
            live_inputs,
            anchor_hidden_state,
            first_draft_slot,
            hidden_source,
            hidden_max_abs_diff,
        )
        return self._finish_proposal(num_reqs)
