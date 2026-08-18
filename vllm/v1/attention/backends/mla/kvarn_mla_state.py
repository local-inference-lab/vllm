# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle management for MLA KVarN precision-tail pool slots."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from weakref import WeakSet

import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata


class KVarNMLAImpl(Protocol):
    layer_name: str
    _is_kvarn_mla: bool
    _kvarn_group_key: tuple[str, ...] | None
    _kvarn_pool_size: int
    device: torch.device

    def _flush_kvarn_mla_blocks(
        self, block_ids: torch.Tensor, pool_slots: torch.Tensor
    ) -> None: ...


class KVarNMLARequestState(Protocol):
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


@dataclass
class KVarNMLALiveBlockTracker:
    """Tracks persistent and per-step exact blocks from scheduler CPU state."""

    group_configs: dict[int, tuple[int, int, int, int, int, int]]
    request_blocks: dict[str, dict[int, dict[int, int | None]]] = field(
        default_factory=dict
    )
    step_blocks: dict[int, dict[int, int | None]] = field(default_factory=dict)
    pending_blocks: dict[str, dict[int, dict[int, int]]] = field(default_factory=dict)
    resolved_blocks: dict[str, dict[int, dict[int, int]]] = field(default_factory=dict)

    @staticmethod
    def _physical_block_id(
        block_ids: list[int],
        logical_block: int,
        blocks_per_manager_block: int,
    ) -> int:
        manager_block, subblock = divmod(logical_block, blocks_per_manager_block)
        return block_ids[manager_block] * blocks_per_manager_block + subblock

    @staticmethod
    def _local_num_tokens(
        num_tokens: int,
        dcp_world_size: int,
        dcp_rank: int,
        dcp_interleave: int,
    ) -> int:
        cycle = dcp_world_size * dcp_interleave
        full_cycles, remainder = divmod(num_tokens, cycle)
        return full_cycles * dcp_interleave + min(
            max(remainder - dcp_rank * dcp_interleave, 0),
            dcp_interleave,
        )

    @staticmethod
    def _merge_fill(
        blocks: dict[int, int | None],
        block_id: int,
        fill: int | None,
    ) -> None:
        if block_id not in blocks:
            blocks[block_id] = fill
            return
        current = blocks[block_id]
        if fill is not None and (current is None or fill > current):
            blocks[block_id] = fill

    def _persistent_group_blocks(
        self,
        block_ids: list[int],
        group_size: int,
        sink_tokens: int,
        blocks_per_manager_block: int,
        local_min_end: int,
        local_max_end: int,
    ) -> dict[int, int | None]:
        num_logical_blocks = min(
            math.ceil(local_max_end / group_size),
            len(block_ids) * blocks_per_manager_block,
        )
        if num_logical_blocks == 0:
            return {}

        blocks: dict[int, int | None] = {}
        sink_blocks = min(
            math.ceil(sink_tokens / group_size),
            num_logical_blocks,
        )
        for logical_block in range(sink_blocks):
            fill = min(
                group_size,
                max(local_min_end - logical_block * group_size, 0),
            )
            blocks[
                self._physical_block_id(
                    block_ids,
                    logical_block,
                    blocks_per_manager_block,
                )
            ] = fill or None

        first_current = max(math.ceil(local_min_end / group_size) - 1, 0)
        for logical_block in range(first_current, num_logical_blocks):
            fill = min(
                group_size,
                max(local_min_end - logical_block * group_size, 0),
            )
            self._merge_fill(
                blocks,
                self._physical_block_id(
                    block_ids,
                    logical_block,
                    blocks_per_manager_block,
                ),
                fill or None,
            )
        return blocks

    def _consume_resolved(self, req_id: str) -> None:
        for group_id, blocks in self.resolved_blocks.pop(req_id, {}).items():
            step_blocks = self.step_blocks[group_id]
            for block_id, fill in blocks.items():
                self._merge_fill(step_blocks, block_id, fill)

    def resolve_async(
        self,
        req_id: str,
        request: KVarNMLARequestState,
        actual_end_tokens: int,
    ) -> None:
        """Resolve conservative ownership after async spec acceptance is known."""
        pending = self.pending_blocks.pop(req_id, None)
        if pending is None:
            return

        groups: dict[int, dict[int, int | None]] = {}
        resolved: dict[int, dict[int, int]] = {}
        for (
            group_id,
            (
                group_size,
                sink_tokens,
                blocks_per_manager_block,
                dcp_world_size,
                dcp_rank,
                dcp_interleave,
            ),
        ) in self.group_configs.items():
            if group_id >= len(request.block_ids) or actual_end_tokens <= 0:
                continue
            local_end = self._local_num_tokens(
                actual_end_tokens,
                dcp_world_size,
                dcp_rank,
                dcp_interleave,
            )
            block_ids = request.block_ids[group_id]
            groups[group_id] = self._persistent_group_blocks(
                block_ids,
                group_size,
                sink_tokens,
                blocks_per_manager_block,
                local_end,
                local_end,
            )

            resolved_group: dict[int, int] = {}
            for block_id, logical_block in pending.get(group_id, {}).items():
                fill = min(
                    group_size,
                    max(local_end - logical_block * group_size, 0),
                )
                if fill:
                    resolved_group[block_id] = fill
            if resolved_group:
                resolved[group_id] = resolved_group

        self.request_blocks[req_id] = groups
        if resolved:
            self.resolved_blocks[req_id] = resolved

    def update(
        self,
        requests: Mapping[str, KVarNMLARequestState],
        scheduled_tokens: Mapping[str, int],
        finished_req_ids: Iterable[str],
        preempted_req_ids: Iterable[str],
        rollback_tokens: Mapping[str, int] | None = None,
    ) -> None:
        self.step_blocks = {group_id: {} for group_id in self.group_configs}
        removed = set(finished_req_ids)
        removed.update(preempted_req_ids)
        for req_id in removed:
            self.request_blocks.pop(req_id, None)
            self.pending_blocks.pop(req_id, None)
            self.resolved_blocks.pop(req_id, None)
        for req_id in tuple(self.resolved_blocks):
            self._consume_resolved(req_id)

        rollback_tokens = rollback_tokens or {}
        for req_id, num_scheduled_tokens in scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            start_tokens = request.num_computed_tokens
            if req_id in self.pending_blocks:
                self.resolve_async(req_id, request, start_tokens)
                self._consume_resolved(req_id)

            rollback = rollback_tokens.get(req_id, 0)
            min_start_tokens = max(start_tokens - rollback, 0)
            min_end_tokens = min_start_tokens + num_scheduled_tokens
            max_end_tokens = start_tokens + num_scheduled_tokens
            groups: dict[int, dict[int, int | None]] = {}
            pending: dict[int, dict[int, int]] = {}
            for (
                group_id,
                (
                    group_size,
                    sink_tokens,
                    blocks_per_manager_block,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                ),
            ) in self.group_configs.items():
                if group_id >= len(request.block_ids) or max_end_tokens <= 0:
                    continue
                local_min_start = self._local_num_tokens(
                    min_start_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                local_min_end = self._local_num_tokens(
                    min_end_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                local_max_end = self._local_num_tokens(
                    max_end_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                block_ids = request.block_ids[group_id]
                num_logical_blocks = min(
                    math.ceil(local_max_end / group_size),
                    len(block_ids) * blocks_per_manager_block,
                )
                if num_logical_blocks == 0:
                    continue

                groups[group_id] = self._persistent_group_blocks(
                    block_ids,
                    group_size,
                    sink_tokens,
                    blocks_per_manager_block,
                    local_min_end,
                    local_max_end,
                )

                if local_min_start < local_max_end:
                    first_touched = local_min_start // group_size
                    last_touched = min(
                        (local_max_end - 1) // group_size,
                        num_logical_blocks - 1,
                    )
                    step_blocks = self.step_blocks[group_id]
                    pending_group: dict[int, int] = {}
                    for logical_block in range(first_touched, last_touched + 1):
                        block_id = self._physical_block_id(
                            block_ids,
                            logical_block,
                            blocks_per_manager_block,
                        )
                        fill = min(
                            group_size,
                            max(local_min_end - logical_block * group_size, 0),
                        )
                        self._merge_fill(step_blocks, block_id, fill or None)
                        pending_group[block_id] = logical_block
                    if rollback and pending_group:
                        pending[group_id] = pending_group
            self.request_blocks[req_id] = groups
            if pending:
                self.pending_blocks[req_id] = pending
            else:
                self.pending_blocks.pop(req_id, None)

    def block_fills(self, group_id: int) -> dict[int, int | None]:
        fills = dict(self.step_blocks.get(group_id, {}))
        for groups in self.request_blocks.values():
            for block_id, fill in groups.get(group_id, {}).items():
                self._merge_fill(fills, block_id, fill)
        return fills


@dataclass
class _GroupState:
    pool_size: int
    mapping: dict[int, int] = field(default_factory=dict)
    free_slots: list[int] = field(default_factory=list)
    block_fill: dict[int, int] = field(default_factory=dict)
    mirrors: dict[torch.device, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free_slots = list(range(self.pool_size - 1, -1, -1))


class KVarNMLAStateManager:
    """Shares physical-block-to-tail-slot ownership across MLA layers."""

    _impls: WeakSet[KVarNMLAImpl] = WeakSet()
    _groups: dict[tuple[str, ...], _GroupState] = {}

    @classmethod
    def register(cls, impl: KVarNMLAImpl) -> None:
        cls._impls.add(impl)

    @classmethod
    def reset_cache_bindings(cls) -> None:
        cls._groups.clear()
        for impl in cls._impls:
            impl._kvarn_group_key = None
            impl._kvarn_cache_ref = None  # type: ignore[attr-defined]
            impl._kvarn_block_to_slot = None  # type: ignore[attr-defined]

    @classmethod
    def ensure_mirror(
        cls,
        group_key: tuple[str, ...],
        device: torch.device,
        num_blocks: int,
    ) -> torch.Tensor:
        state = cls._groups[group_key]
        mirror = state.mirrors.get(device)
        if mirror is None or mirror.shape[0] < num_blocks:
            mirror = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            state.mirrors[device] = mirror
            cls._sync_mirror(state, mirror)
        return mirror

    @staticmethod
    def _sync_mirror(state: _GroupState, mirror: torch.Tensor) -> None:
        mirror.fill_(-1)
        if not state.mapping:
            return
        block_ids = torch.tensor(
            list(state.mapping), dtype=torch.long, device=mirror.device
        )
        slots = torch.tensor(
            list(state.mapping.values()), dtype=torch.int32, device=mirror.device
        )
        valid = block_ids < mirror.shape[0]
        mirror[block_ids[valid]] = slots[valid]

    @staticmethod
    def _update_mirror(mirror: torch.Tensor, updates: dict[int, int]) -> None:
        if not updates:
            return
        block_ids = torch.tensor(list(updates), dtype=torch.long, device=mirror.device)
        slots = torch.tensor(
            list(updates.values()), dtype=torch.int32, device=mirror.device
        )
        valid = block_ids < mirror.shape[0]
        mirror[block_ids[valid]] = slots[valid]

    @classmethod
    def prepare_step(
        cls,
        group_key: tuple[str, ...],
        layer_names: list[str],
        common_metadata: CommonAttentionMetadata,
        config: KVarNMLAConfig,
        dcp_world_size: int,
    ) -> None:
        impls = [
            impl
            for impl in cls._impls
            if impl._is_kvarn_mla and impl.layer_name in layer_names
        ]
        if not impls:
            return
        for impl in impls:
            impl._kvarn_group_key = group_key

        pool_size = min(impl._kvarn_pool_size for impl in impls)
        state = cls._groups.get(group_key)
        if state is None:
            state = _GroupState(pool_size=pool_size)
            cls._groups[group_key] = state
        elif state.pool_size != pool_size:
            raise RuntimeError(
                "MLA KVarN layers in one cache group must share pool size"
            )

        block_fills = common_metadata.kvarn_mla_block_fills
        if block_fills is None:
            raise RuntimeError("MLA KVarN requires exact-block ownership metadata")
        needed = set(block_fills)
        for block_id, fill in block_fills.items():
            if fill is None:
                state.block_fill.pop(block_id, None)
            else:
                state.block_fill[block_id] = max(
                    state.block_fill.get(block_id, 0), fill
                )

        retired = [block_id for block_id in state.mapping if block_id not in needed]
        flush_ids = [
            block_id
            for block_id in retired
            if state.block_fill.get(block_id, 0) >= config.group
        ]
        if flush_ids:
            device = impls[0].device
            block_ids = torch.tensor(flush_ids, dtype=torch.long, device=device)
            pool_slots = torch.tensor(
                [state.mapping[block_id] for block_id in flush_ids],
                dtype=torch.long,
                device=device,
            )
            for impl in impls:
                impl._flush_kvarn_mla_blocks(block_ids, pool_slots)

        for block_id in retired:
            state.free_slots.append(state.mapping.pop(block_id))
            state.block_fill.pop(block_id, None)

        missing = sorted(needed.difference(state.mapping))
        if len(missing) > len(state.free_slots):
            raise RuntimeError(
                "MLA KVarN exact-block pool exhausted: "
                f"need {len(missing)} slots, have {len(state.free_slots)}. "
                "Reduce max_num_seqs or max_num_batched_tokens."
            )
        mirror_updates = {block_id: -1 for block_id in retired}
        for block_id in missing:
            slot = state.free_slots.pop()
            state.mapping[block_id] = slot
            mirror_updates[block_id] = slot

        for mirror in state.mirrors.values():
            cls._update_mirror(mirror, mirror_updates)
