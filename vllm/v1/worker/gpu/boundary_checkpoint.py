# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU capture of request endpoints into budgeted KV-pool blocks."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.core.boundary_checkpoint import MAX_BOUNDARY_STOP_TOKENS
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, iter_layer_specs
from vllm.v1.worker.mamba_utils import _reinterpret_u64_as_i64

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.interface import ModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


@triton.jit
def prepare_boundary_capture(
    req_idx,
    batch_idx,
    num_computed,
    total_len,
    query_start,
    query_len,
    num_sampled,
    num_rejected,
    sampled_tokens_ptr,
    sampled_stride,
    metadata_ptr,
    stop_tokens_ptr,
    seen_ptr,
    capture_tokens_ptr,
    capture_bias_ptr,
    capture_rows_ptr,
    STOP_CAPACITY: tl.constexpr,
):
    enabled = tl.load(metadata_ptr + req_idx * 5)
    prompt_capture = 0
    response_capture = 0
    if enabled:
        prompt_len = tl.load(metadata_ptr + req_idx * 5 + 1)
        min_len = tl.load(metadata_ptr + req_idx * 5 + 2)
        max_len = tl.load(metadata_ptr + req_idx * 5 + 3)
        num_stops = tl.load(metadata_ptr + req_idx * 5 + 4)
        offsets = tl.arange(0, STOP_CAPACITY)
        stops = tl.load(
            stop_tokens_ptr + req_idx * STOP_CAPACITY + offsets,
            offsets < num_stops,
            other=-1,
        )
        stopped = False
        valid = num_sampled
        for i in range(num_sampled):
            token = tl.load(sampled_tokens_ptr + batch_idx * sampled_stride + i)
            length = total_len + i + 1
            is_stop = (
                tl.sum(((stops == token) & (offsets < num_stops)).to(tl.int32)) > 0
            )
            if not stopped and length >= min_len and (is_stop or length >= max_len):
                valid = i + 1
                stopped = True
        num_rejected += num_sampled - valid
        num_sampled = valid
        computed_after = num_computed + query_len - num_rejected
        if (
            num_computed < prompt_len
            and computed_after == prompt_len
            and tl.load(seen_ptr + req_idx * 2) == 0
        ):
            prompt_capture = prompt_len
            tl.store(seen_ptr + req_idx * 2, 1)
        if stopped and tl.load(seen_ptr + req_idx * 2 + 1) == 0:
            response_capture = computed_after
            tl.store(seen_ptr + req_idx * 2 + 1, 1)

    tl.store(capture_tokens_ptr + batch_idx * 2, prompt_capture)
    tl.store(capture_tokens_ptr + batch_idx * 2 + 1, response_capture)
    tl.store(capture_bias_ptr + batch_idx * 2, 0)
    tl.store(capture_bias_ptr + batch_idx * 2 + 1, tl.maximum(num_sampled - 1, 0))
    tl.store(
        capture_rows_ptr + batch_idx * 2,
        query_start + prompt_capture - num_computed - 1,
    )
    tl.store(
        capture_rows_ptr + batch_idx * 2 + 1,
        query_start + response_capture - num_computed - 1,
    )
    return num_sampled, num_rejected


@triton.jit
def _copy_auxiliary_state_kernel(
    idx_mapping_ptr,
    capture_tokens_ptr,
    capture_rows_ptr,
    destination_blocks_ptr,
    metadata_ptr,
    pool_ptr,
    pool_stride: tl.int64,
    hidden_ptr,
    hidden_stride: tl.int64,
    NUM_GROUPS: tl.constexpr,
    NUM_STATES: tl.constexpr,
    HIDDEN_OFFSET: tl.constexpr,
    HIDDEN_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) // 2
    kind = tl.program_id(0) % 2
    state = tl.program_id(1)
    if tl.load(capture_tokens_ptr + row * 2 + kind) <= 0:
        return
    slot = tl.load(idx_mapping_ptr + row)
    block = tl.load(
        destination_blocks_ptr + (slot * 2 + kind) * (NUM_GROUPS + 1) + NUM_GROUPS
    )
    if state < NUM_STATES:
        base = tl.load(metadata_ptr + state * 4)
        stride = tl.load(metadata_ptr + state * 4 + 1)
        size = tl.load(metadata_ptr + state * 4 + 2)
        offset = tl.load(metadata_ptr + state * 4 + 3)
        source = (base + slot.to(tl.int64) * stride).to(tl.pointer_type(tl.uint8))
    else:
        hidden_row = tl.load(capture_rows_ptr + row * 2 + kind)
        source = (
            hidden_ptr.to(tl.pointer_type(tl.uint8))
            + hidden_row.to(tl.int64) * hidden_stride
        )
        size = tl.full((), HIDDEN_BYTES, tl.int64)
        offset = tl.full((), HIDDEN_OFFSET, tl.int64)
    destination = pool_ptr + block.to(tl.int64) * pool_stride + offset
    x = tl.arange(0, BLOCK)
    for start in range(0, size, BLOCK):
        data = tl.load(source + start + x, start + x < size, other=0)
        tl.store(destination + start + x, data, start + x < size)


@triton.jit
def _restore_auxiliary_state_kernel(
    metadata_ptr,
    pool_ptr,
    pool_stride: tl.int64,
    block,
    slot,
    BLOCK: tl.constexpr,
):
    state = tl.program_id(0)
    base = tl.load(metadata_ptr + state * 4)
    stride = tl.load(metadata_ptr + state * 4 + 1)
    size = tl.load(metadata_ptr + state * 4 + 2)
    offset = tl.load(metadata_ptr + state * 4 + 3)
    source = pool_ptr + block.to(tl.int64) * pool_stride + offset
    destination = (base + slot.to(tl.int64) * stride).to(tl.pointer_type(tl.uint8))
    x = tl.arange(0, BLOCK)
    for start in range(0, size, BLOCK):
        data = tl.load(source + start + x, start + x < size, other=0)
        tl.store(destination + start + x, data, start + x < size)


@triton.jit
def _copy_attention_tails_kernel(
    idx_mapping_ptr,
    capture_tokens_ptr,
    destination_blocks_ptr,
    storage_metadata_ptr,
    block_table_ptrs,
    table_strides_ptr,
    kernel_block_sizes_ptr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0) // 2
    kind = tl.program_id(0) % 2
    storage = tl.program_id(1)
    tile = tl.program_id(2)
    count = tl.load(capture_tokens_ptr + row * 2 + kind)
    if count <= 0:
        return
    slot = tl.load(idx_mapping_ptr + row)
    base = tl.load(storage_metadata_ptr + storage * 4)
    stride = tl.load(storage_metadata_ptr + storage * 4 + 1)
    group = tl.load(storage_metadata_ptr + storage * 4 + 2)
    block_size = tl.load(storage_metadata_ptr + storage * 4 + 3)
    kernel_block_size = tl.load(kernel_block_sizes_ptr + group)
    table_stride = tl.load(table_strides_ptr + group)
    table = tl.load(block_table_ptrs + group).to(tl.pointer_type(tl.int32))
    source_block = tl.load(
        table + slot * table_stride + (count - 1) // kernel_block_size
    )
    source_block //= block_size // kernel_block_size
    destination_block = tl.load(
        destination_blocks_ptr + (slot * 2 + kind) * (NUM_GROUPS + 1) + group
    )
    source = (base + source_block.to(tl.int64) * stride).to(tl.pointer_type(tl.uint8))
    destination = (base + destination_block.to(tl.int64) * stride).to(
        tl.pointer_type(tl.uint8)
    )
    x = tile * BLOCK + tl.arange(0, BLOCK)
    for start in range(0, stride, BLOCK * TILES):
        data = tl.load(source + start + x, start + x < stride, other=0)
        tl.store(destination + start + x, data, start + x < stride)


class BoundaryCheckpointState:
    def __init__(
        self,
        model_state: "ModelState",
        kv_cache_config: KVCacheConfig,
        forward_context: dict,
        draft_model: torch.nn.Module | None = None,
    ) -> None:
        self.model_state = cast("MambaHybridModelState", model_state)
        self.device = model_state.device
        self.max_reqs = model_state.max_num_reqs
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        self.metadata = torch.zeros(
            (self.max_reqs, 5), dtype=torch.int32, device=self.device
        )
        self.stop_tokens = torch.full(
            (self.max_reqs, MAX_BOUNDARY_STOP_TOKENS),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.seen = torch.zeros(
            (self.max_reqs, 2), dtype=torch.int32, device=self.device
        )
        self.blocks = torch.zeros(
            (self.max_reqs, 2, self.num_groups + 1),
            dtype=torch.int32,
            device=self.device,
        )
        mamba_groups = []
        storage_metadata = []
        self._storages: list[torch.Tensor] = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            if all(
                isinstance(spec, MambaSpec)
                for spec in iter_layer_specs(group.kv_cache_spec)
            ):
                mamba_groups.append(group_id)
                continue
            seen_storages = set()
            for name in group.layer_names:
                cache = forward_context[name].kv_cache
                tensors: Iterable[torch.Tensor] = (
                    cache if isinstance(cache, (tuple, list)) else (cache,)
                )
                for tensor in tensors:
                    storage = tensor.untyped_storage()
                    if storage.data_ptr() in seen_storages:
                        continue
                    seen_storages.add(storage.data_ptr())
                    raw = torch.empty(0, dtype=torch.uint8, device=self.device).set_(
                        storage
                    )
                    raw = raw.view(kv_cache_config.num_blocks, -1)
                    blocks_per_page = tensor.shape[0] // kv_cache_config.num_blocks
                    assert tensor.stride(
                        0
                    ) * tensor.element_size() * blocks_per_page == raw.stride(0), (
                        "Boundary checkpoints require a block-outermost KV layout"
                    )
                    self._storages.append(raw)
                    storage_metadata.append(
                        [
                            _reinterpret_u64_as_i64(raw.data_ptr()),
                            raw.stride(0),
                            group_id,
                            group.kv_cache_spec.block_size,
                        ]
                    )
        if not self._storages:
            raise ValueError("Boundary checkpoints require an attention cache pool")
        self.pool = self._storages[0]
        self.mamba_group_ids = mamba_groups
        self.mamba_blocks = torch.zeros(
            (self.max_reqs, 2, len(mamba_groups)), dtype=torch.int32, device=self.device
        )
        self.storage_metadata = torch.tensor(
            storage_metadata, dtype=torch.int64, device=self.device
        )
        self._auxiliary_tensors = list(model_state.get_recurrent_checkpoint_tensors())
        self.target_modules = list(model_state.model.modules())
        for module in self.target_modules:
            if hasattr(module, "get_recurrent_checkpoint_tensors"):
                self._auxiliary_tensors.extend(
                    module.get_recurrent_checkpoint_tensors()
                )
        self.draft_modules = list(draft_model.modules()) if draft_model else []
        draft_tensors = []
        for module in self.draft_modules:
            if hasattr(module, "get_recurrent_checkpoint_tensors"):
                draft_tensors.extend(module.get_recurrent_checkpoint_tensors())
        target_tensor_count = len(self._auxiliary_tensors)
        self._auxiliary_tensors.extend(draft_tensors)
        auxiliary_metadata = []
        target_metadata: list[list[int]] = []
        draft_metadata: list[list[int]] = []
        offset = 0
        seen_ptrs = set()
        for i, tensor in enumerate(self._auxiliary_tensors):
            if tensor.data_ptr() in seen_ptrs:
                continue
            seen_ptrs.add(tensor.data_ptr())
            assert tensor.is_contiguous() and tensor.shape[0] == self.max_reqs
            size = tensor[0].numel() * tensor.element_size()
            auxiliary_metadata.append(
                [
                    _reinterpret_u64_as_i64(tensor.data_ptr()),
                    tensor.stride(0) * tensor.element_size(),
                    size,
                    offset,
                ]
            )
            (target_metadata if i < target_tensor_count else draft_metadata).append(
                auxiliary_metadata[-1]
            )
            offset += triton.cdiv(size, 8) * 8
        self.auxiliary_metadata = torch.tensor(
            auxiliary_metadata, dtype=torch.int64, device=self.device
        )
        self.target_metadata = torch.tensor(
            target_metadata, dtype=torch.int64, device=self.device
        ).reshape(-1, 4)
        self.draft_metadata = torch.tensor(
            draft_metadata, dtype=torch.int64, device=self.device
        ).reshape(-1, 4)
        self.hidden_offset = offset
        self.hidden_shape: tuple[int, ...] | None = None
        self.hidden_dtype: torch.dtype | None = None
        self.spec_hidden_offset = 0
        self.spec_hidden_shape: tuple[int, ...] | None = None
        self.spec_hidden_dtype: torch.dtype | None = None
        self._completion = torch.cuda.Event()

    def add_request(self, slot: int, request: "NewRequestData") -> None:
        self.seen[slot].zero_()
        self.metadata[slot].zero_()
        allocation = request.boundary_checkpoint_blocks
        if allocation is None:
            return
        params = request.sampling_params
        assert params is not None and params.max_tokens is not None
        stops = set(params.stop_token_ids or ())
        if params.eos_token_id is not None:
            stops.add(params.eos_token_id)
        if len(stops) > MAX_BOUNDARY_STOP_TOKENS:
            raise ValueError("Too many stop tokens for request-boundary caching")
        prompt_len = request.prompt_len
        self.metadata[slot].copy_(
            torch.tensor(
                [
                    1,
                    prompt_len,
                    prompt_len + params.min_tokens,
                    min(
                        prompt_len + params.max_tokens,
                        self.model_state.model_config.max_model_len,
                    ),
                    len(stops),
                ],
                dtype=torch.int32,
                device=self.device,
            )
        )
        if stops:
            self.stop_tokens[slot, : len(stops)].copy_(
                torch.tensor(sorted(stops), dtype=torch.int32, device=self.device)
            )
        self.blocks[slot].copy_(
            torch.tensor(allocation, dtype=torch.int32, device=self.device)
        )
        self.mamba_blocks[slot].copy_(
            torch.tensor(
                [
                    [kind[group_id] for group_id in self.mamba_group_ids]
                    for kind in allocation
                ],
                dtype=torch.int32,
                device=self.device,
            )
        )
        checkpoint = request.boundary_checkpoint
        if checkpoint is not None:
            if checkpoint.num_tokens >= prompt_len:
                self.seen[slot, 0] = 1
            _restore_auxiliary_state_kernel[(self.auxiliary_metadata.shape[0],)](
                self.auxiliary_metadata,
                self.pool,
                self.pool.stride(0),
                checkpoint.auxiliary_block_ids[0],
                slot,
                BLOCK=1024,
            )
            # Prefill consumes a canonical committed prefix, not the previous
            # verification interval's acceptance count.
            self.model_state.get_recurrent_checkpoint_acceptance()[slot] = 1
            for module in self.target_modules:
                if hasattr(module, "set_recurrent_checkpoint_anchor"):
                    module.set_recurrent_checkpoint_anchor(
                        slot, checkpoint.num_tokens - 1
                    )

    def capture_mamba(self, idx_mapping: torch.Tensor, capture: torch.Tensor) -> None:
        state = self.model_state
        if state._mamba_ctx is not None:
            state._mamba_ctx.checkpoint_request_boundaries(
                idx_mapping,
                state._mamba_state_idx_gpu,
                capture[0],
                capture[1],
                self.mamba_blocks,
            )

    def capture_auxiliary(
        self,
        idx_mapping: torch.Tensor,
        capture: torch.Tensor,
        hidden: torch.Tensor,
        spec_hidden: torch.Tensor | None = None,
    ) -> None:
        hidden_bytes = hidden[0].numel() * hidden.element_size()
        if self.hidden_offset + hidden_bytes > self.pool.shape[1]:
            raise RuntimeError("Boundary auxiliary state exceeds its reserved KV page")
        self.hidden_shape = tuple(hidden.shape[1:])
        self.hidden_dtype = hidden.dtype
        self.spec_hidden_offset = self.hidden_offset + hidden_bytes
        _copy_auxiliary_state_kernel[
            (idx_mapping.numel() * 2, self.target_metadata.shape[0] + 1)
        ](
            idx_mapping,
            capture[0],
            capture[2],
            self.blocks,
            self.target_metadata,
            self.pool,
            self.pool.stride(0),
            hidden,
            hidden.stride(0) * hidden.element_size(),
            NUM_GROUPS=self.num_groups,
            NUM_STATES=self.target_metadata.shape[0],
            HIDDEN_OFFSET=self.hidden_offset,
            HIDDEN_BYTES=hidden_bytes,
            BLOCK=1024,
        )
        if spec_hidden is not None:
            size = spec_hidden[0].numel() * spec_hidden.element_size()
            if self.spec_hidden_offset + size > self.pool.shape[1]:
                raise RuntimeError("Boundary MTP state exceeds its reserved KV page")
            self.spec_hidden_shape = tuple(spec_hidden.shape[1:])
            self.spec_hidden_dtype = spec_hidden.dtype
            _copy_auxiliary_state_kernel[(idx_mapping.numel() * 2, 1)](
                idx_mapping,
                capture[0],
                capture[2],
                self.blocks,
                self.target_metadata,
                self.pool,
                self.pool.stride(0),
                spec_hidden,
                spec_hidden.stride(0) * spec_hidden.element_size(),
                NUM_GROUPS=self.num_groups,
                NUM_STATES=0,
                HIDDEN_OFFSET=self.spec_hidden_offset,
                HIDDEN_BYTES=size,
                BLOCK=1024,
            )

    def capture_draft(self, idx_mapping: torch.Tensor, capture: torch.Tensor) -> None:
        _copy_auxiliary_state_kernel[
            (idx_mapping.numel() * 2, self.draft_metadata.shape[0])
        ](
            idx_mapping,
            capture[0],
            capture[2],
            self.blocks,
            self.draft_metadata,
            self.pool,
            self.pool.stride(0),
            self.pool,
            0,
            NUM_GROUPS=self.num_groups,
            NUM_STATES=self.draft_metadata.shape[0],
            HIDDEN_OFFSET=0,
            HIDDEN_BYTES=0,
            BLOCK=1024,
        )

    def set_draft_replay_anchor(
        self, slot: int, count: int, accepted: torch.Tensor | int
    ) -> None:
        for module in self.draft_modules:
            if hasattr(module, "set_recurrent_checkpoint_anchor"):
                module.set_recurrent_checkpoint_anchor(slot, count - accepted)

    def replay_draft(
        self,
        runner: "GPUModelRunner",
        req_id: str,
        count: int,
        block_id: int,
        *,
        token: int | None = None,
        num_speculative_tokens: int = 1,
    ) -> torch.Tensor:
        """Rebuild the lookahead-dependent draft row using its saved target hidden."""
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
        from vllm.v1.worker.workspace import use_workspace_lane

        speculator = runner.speculator
        assert speculator is not None and runner.sampler is not None
        slot = runner.req_states.req_id_to_index[req_id]
        batch = InputBatch.make_dummy(1, 1, InputBuffers(1, 1, self.device))
        batch.req_ids = [req_id]
        batch.idx_mapping.fill_(slot)
        batch.idx_mapping_np.fill(slot)
        batch.positions.fill_(count - 1)
        batch.seq_lens.fill_(count)
        batch.seq_lens_cpu_upper_bound.fill_(count)
        batch.num_computed_tokens_np.fill(count - 1)
        batch.is_padding.zero_()
        if self.model_state.rope_state is not None:
            self.model_state.rope_state.get_positions(1).fill_(count - 1)
        self.set_draft_replay_anchor(slot, count - 1, 1)
        sampled = runner.req_states.last_sampled_tokens
        if token is not None:
            sampled = sampled.clone()
            sampled[slot] = token
        ones = torch.ones(1, dtype=torch.int32, device=self.device)
        zeros = torch.zeros_like(ones)
        try:
            tables, slots = runner.prepare_attn(batch)
            metadata = runner.model_state.prepare_attn(
                batch,
                CUDAGraphMode.NONE,
                tables,
                slots,
                speculator.attn_groups,
                runner.kv_cache_config,
            )
            with use_workspace_lane(runner._draft_workspace_lane):
                return speculator.propose(
                    batch,
                    metadata,
                    build_slot_mappings_by_layer(slots, runner.kv_cache_config),
                    self.get_hidden_states(block_id, draft=True),
                    None,
                    ones,
                    zeros,
                    sampled,
                    runner.req_states.next_prefill_tokens,
                    runner.sampler.sampling_states.temperature.gpu,
                    runner.sampler.sampling_states.seeds.gpu,
                    num_speculative_tokens=num_speculative_tokens,
                    is_profile=True,
                )
        finally:
            self.set_draft_replay_anchor(slot, count, 1)

    def capture_attention(
        self,
        idx_mapping: torch.Tensor,
        capture: torch.Tensor,
        block_tables: "BlockTables",
    ) -> None:
        _copy_attention_tails_kernel[
            (idx_mapping.numel() * 2, self.storage_metadata.shape[0], 16)
        ](
            idx_mapping,
            capture[0],
            self.blocks,
            self.storage_metadata,
            block_tables.block_table_ptrs,
            block_tables.block_table_strides,
            block_tables.kernel_block_sizes_tensor,
            NUM_GROUPS=self.num_groups,
            BLOCK=4096,
            TILES=16,
        )
        self._completion.record()

    def wait_for_copies(self) -> None:
        self._completion.synchronize()

    def get_hidden_states(self, block_id: int, *, draft: bool = False) -> torch.Tensor:
        shape = self.spec_hidden_shape if draft else self.hidden_shape
        dtype = self.spec_hidden_dtype if draft else self.hidden_dtype
        offset = self.spec_hidden_offset if draft else self.hidden_offset
        assert shape is not None and dtype is not None
        size = 1
        for dim in shape:
            size *= dim
        row = self.pool[block_id, offset:]
        return row.view(dtype)[:size].view(1, *shape)
