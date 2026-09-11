# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v4_1.ced import CED_WINDOW, CEDState, ced_decoder_start
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(
        num_computed_tokens_ptr + req_state_idx, mask=in_batch, other=0
    )

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - DEPTH + offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr
        + req_state_idx.to(tl.int64) * all_token_ids_stride
        + pos.to(tl.int64),
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.
    """

    _ced_streaming_prefix_start: tuple[str, int] | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.ced_state: CEDState | None = None
        self._ced_prompt_logprobs: set[str] = set()
        self._ced_prefix_start: dict[str, int] = {}
        self._ced_batch: InputBatch | None = None
        self.max_cudagraph_query_len: int | None = None
        if ced_decoder_start(self.model_config.hf_config) is not None:
            compact_max = min(self.max_num_tokens, self.max_num_reqs * CED_WINDOW)
            # Compaction changes the model's optional-input branch; FULL graphs
            # remain valid for every short/varlen decode split, never long chunks.
            self.max_cudagraph_query_len = CED_WINDOW
            capacities = tuple(
                sorted(
                    {
                        self.max_num_tokens,
                        compact_max,
                        *range(CED_WINDOW, compact_max + 1, CED_WINDOW),
                        *(
                            size
                            for size in (
                                vllm_config.compilation_config.cudagraph_capture_sizes
                                or ()
                            )
                            if 0 < size <= self.max_num_tokens
                        ),
                    }
                )
            )
            self.ced_state = CEDState(
                self.max_num_reqs, self.max_num_tokens, capacities, device
            )
            hf = self.model_config.hf_config
            self.ced_state.warmup(hf.hidden_size, hf.hc_mult)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )
        from .model import DeepseekV4Model

        self.disk_engram_models = tuple(
            module
            for module in model.modules()
            if isinstance(module, DeepseekV4Model) and module.disk_engram
        )

    def prepare_streaming_update(self, req_id: str) -> None:
        """Keep decoder history when the runner refreshes a live request."""
        self._ced_streaming_prefix_start = (
            req_id,
            self._ced_prefix_start.get(req_id, 0),
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        sampling = new_req_data.sampling_params
        if sampling is not None and sampling.prompt_logprobs is not None:
            self._ced_prompt_logprobs.add(new_req_data.req_id)
        else:
            self._ced_prompt_logprobs.discard(new_req_data.req_id)
        # Fresh decoder pages lack cached history; streaming updates retain
        # the live request's private pages and therefore its original boundary.
        pending = self._ced_streaming_prefix_start
        prefix_start = (
            pending[1]
            if pending is not None and pending[0] == new_req_data.req_id
            else new_req_data.num_computed_tokens
        )
        self._ced_streaming_prefix_start = None
        self._ced_prefix_start[new_req_data.req_id] = prefix_start

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._ced_prompt_logprobs.discard(req_id)
        self._ced_prefix_start.pop(req_id, None)

    def get_ced_indices(self) -> torch.Tensor | None:
        return None if self.ced_state is None else self.ced_state.get_indices()

    def _stage_ced(self, input_batch: InputBatch, *, force_full: bool = False) -> None:
        state = self.ced_state
        if state is None:
            return
        nr = input_batch.num_reqs
        state.stage(
            input_batch.query_start_loc,
            input_batch.seq_lens,
            input_batch.num_scheduled_tokens[:nr],
            [True] * nr
            if force_full
            else [
                req_id in self._ced_prompt_logprobs for req_id in input_batch.req_ids
            ],
            [0] * nr
            if force_full
            else [
                self._ced_prefix_start.get(req_id, 0) for req_id in input_batch.req_ids
            ],
        )
        self._ced_batch = input_batch

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        # Runtime prepares attention before model inputs. Capture does the reverse;
        # both paths populate the same persistent packing buffers.
        self._stage_ced(input_batch, force_full=for_capture)
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        state = self.ced_state
        if state is not None:
            seen: set[int] = set()
            for value in metadata.values():
                if isinstance(value, DeepseekV41B12xMetadata) and id(value) not in seen:
                    seen.add(id(value))
                    value.swa_replay_start = state.replay_start
                    if state.plan is not None:
                        value.decoder = state.decoder_metadata(value)
        return metadata

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.ced_state is not None:
            if self._ced_batch is not input_batch:
                self._stage_ced(input_batch)
            model_inputs["ced_indices"] = self.get_ced_indices()
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        for model in self.disk_engram_models:
            model.prepare_disk_engram(
                input_batch.input_ids, input_batch.query_start_loc, window
            )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.ced_state is not None:
            # FULL descriptors may pad a short request far beyond128 rows.
            # Capture their noncompact branch, regardless of dummy split;
            # long real queries are forced off FULL graphs by the runner.
            self.ced_state.plan = None
            self.ced_state.replay_start.zero_()
            self._ced_batch = None
            model_inputs["ced_indices"] = None
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        for model in self.disk_engram_models:
            model.prepare_dummy_engram(num_tokens)
        return model_inputs
