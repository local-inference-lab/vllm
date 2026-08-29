# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x sparse MLA attention backend."""

import os
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_sparse_mla
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata


_GLM_NEXT_MODEL_TYPES = frozenset(("glm5_next", "glm5_next_text"))
_GLM_NEXT_CACHE_RECORD_BYTES = 528
_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN = 132 // 4

logger = init_logger(__name__)


def _is_glm_next_config(hf_config: object | None) -> bool:
    return getattr(hf_config, "model_type", None) in _GLM_NEXT_MODEL_TYPES


def _current_hf_text_config() -> object | None:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return vllm_config.model_config.hf_text_config


def _is_glm_next_spec(spec: AttentionSpec) -> bool:
    if isinstance(spec, MLAAttentionSpec) and spec.model_version == "glm5_next":
        return True
    hf_config = _current_hf_text_config()
    return hf_config is not None and _is_glm_next_config(hf_config)


def _glm_next_recipe_error(hf_config: object) -> str | None:
    expected = {
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_kpool": 4,
    }
    mismatches = [
        f"{name}={getattr(hf_config, name, None)!r} (expected {value})"
        for name, value in expected.items()
        if getattr(hf_config, name, None) != value
    ]
    if mismatches:
        return "B12X GLM5Next sparse MLA requires " + ", ".join(mismatches)
    return None


def _glm_next_dcp_error(vllm_config: VllmConfig) -> str | None:
    parallel_config = vllm_config.parallel_config
    dcp_size = int(parallel_config.decode_context_parallel_size)
    if dcp_size <= 1:
        return None
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    if interleave % 4:
        return (
            "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
        )
    return None


def _selected_index_block_stride_rows(
    kv_cache: torch.Tensor,
    *,
    block_size: int,
    is_glm_next: bool,
) -> int:
    if is_glm_next:
        # GLM_NEXT selected indices are physical token slots. The b12x kernel
        # applies the cache's byte page stride itself.
        return block_size
    record_width = int(kv_cache.shape[-1])
    return int(kv_cache.stride(0)) // record_width


def _is_glm_next_ckv_source_layout(
    kv_cache: torch.Tensor,
    *,
    page_size: int,
) -> bool:
    return (
        kv_cache.dtype == torch.uint8
        and kv_cache.ndim == 3
        and tuple(kv_cache.shape[1:]) == (page_size, _GLM_NEXT_CACHE_RECORD_BYTES)
        and kv_cache.stride(1) == _GLM_NEXT_CACHE_RECORD_BYTES
        and kv_cache.stride(2) == 1
    )


def _use_b12x_sparse_decode_plan(
    *,
    max_query_len: int,
    num_tokens: int,
    num_reqs: int,
    is_spec_decode: bool,
    spec_extend_as_decode: bool,
    spec_extend_as_decode_force: bool,
    spec_decode_max_q: int,
    max_tokens: int,
) -> bool:
    if max_query_len <= 1:
        return True
    use_spec_decode = spec_extend_as_decode and (
        spec_extend_as_decode_force or is_spec_decode
    )
    return (
        use_spec_decode
        and max_query_len <= spec_decode_max_q
        and num_tokens <= num_reqs * spec_decode_max_q
        and num_tokens <= max_tokens
    )


def _use_b12x_full_ckv_gather(
    *,
    enabled: bool,
    is_glm_next: bool,
    dcp_world_size: int,
    max_query_len: int,
    num_tokens: int,
    is_spec_decode: bool,
    min_tokens: int,
    max_tokens: int,
) -> bool:
    return (
        enabled
        and is_glm_next
        and dcp_world_size > 1
        and max_query_len > 1
        and not is_spec_decode
        and num_tokens > min_tokens
        and num_tokens <= max_tokens
    )


def _ckv_prefetch_ring_slots(depth: int) -> int:
    return max(0, int(depth)) + 1


def _ckv_prefetch_workspace_nbytes(
    depth: int,
    dcp_world_size: int,
    local_capacity: int,
    record_bytes: int,
) -> int:
    """Return one lane's local staging plus gathered-cache ring size."""
    return (
        (1 + _ckv_prefetch_ring_slots(depth) * int(dcp_world_size))
        * int(local_capacity)
        * int(record_bytes)
    )


def _ckv_prefetch_execution_lanes(num_ubatches: int, speculative: bool) -> int:
    return max(1, int(num_ubatches)) * (2 if speculative else 1)


def _ckv_prefetch_depth_within_budget(
    requested_depth: int,
    workspace_budget_bytes: int,
    dcp_world_size: int,
    local_capacity: int,
    record_bytes: int,
) -> int:
    """Cap lookahead depth without removing the synchronous gather slot."""
    requested_depth = max(0, int(requested_depth))
    workspace_budget_bytes = int(workspace_budget_bytes)
    if workspace_budget_bytes <= 0:
        return requested_depth
    for depth in range(requested_depth, -1, -1):
        if (
            _ckv_prefetch_workspace_nbytes(
                depth,
                dcp_world_size,
                local_capacity,
                record_bytes,
            )
            <= workspace_budget_bytes
        ):
            return depth
    return 0


def _ckv_prefetch_target_indices(
    layer_idx: int,
    depth: int,
    layer_caches: list[torch.Tensor | None],
    pending_layers: dict[int, tuple[Any, int]],
) -> list[int]:
    targets: list[int] = []
    for distance in range(1, max(0, int(depth)) + 1):
        target_idx = layer_idx + distance
        if target_idx in pending_layers:
            continue
        if target_idx >= len(layer_caches) or layer_caches[target_idx] is None:
            break
        targets.append(target_idx)
    return targets


class _CKVPrefetchWorkspacePool:
    """Preallocated CKV rings shared by attention layers on one device."""

    def __init__(
        self,
        device: torch.device,
        slot_nbytes: int,
        max_slots: int,
    ) -> None:
        if slot_nbytes <= 0 or max_slots <= 0:
            raise ValueError(
                "CKV workspace pool requires positive slot size and count, got "
                f"slot_nbytes={slot_nbytes} max_slots={max_slots}"
            )
        self.device = device
        self.slot_nbytes = int(slot_nbytes)
        self.max_slots = int(max_slots)
        self.storage = torch.empty(
            (self.slot_nbytes * self.max_slots,),
            dtype=torch.uint8,
            device=device,
        )
        self._free_slots = list(reversed(range(self.max_slots)))
        self._leased_slots: set[int] = set()

    def acquire(self) -> tuple[int, torch.Tensor]:
        if not self._free_slots:
            raise RuntimeError(
                "CKV prefetch workspace pool exhausted. The runtime created more "
                f"than {self.max_slots} execution lanes."
            )
        slot = self._free_slots.pop()
        self._leased_slots.add(slot)
        start = slot * self.slot_nbytes
        return slot, self.storage.narrow(0, start, self.slot_nbytes)

    def release(self, slot: int) -> None:
        if slot not in self._leased_slots:
            raise RuntimeError(f"CKV workspace slot {slot} is not leased")
        self._leased_slots.remove(slot)
        self._free_slots.append(slot)


_CKV_PREFETCH_WORKSPACE_POOLS: dict[
    tuple[str, int | None, int, int], _CKVPrefetchWorkspacePool
] = {}


def _get_ckv_prefetch_workspace_pool(
    device: torch.device,
    slot_nbytes: int,
    max_slots: int,
) -> _CKVPrefetchWorkspacePool:
    key = (device.type, device.index, int(slot_nbytes), int(max_slots))
    pool = _CKV_PREFETCH_WORKSPACE_POOLS.get(key)
    if pool is None:
        pool = _CKVPrefetchWorkspacePool(device, slot_nbytes, max_slots)
        _CKV_PREFETCH_WORKSPACE_POOLS[key] = pool
    return pool


@dataclass(frozen=True)
class _CKVWorkspaceIdentity:
    device: torch.device
    storage_data_ptr: int
    storage_nbytes: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


def _ckv_workspace_identity(workspace: torch.Tensor) -> _CKVWorkspaceIdentity:
    storage = workspace.untyped_storage()
    return _CKVWorkspaceIdentity(
        device=workspace.device,
        storage_data_ptr=storage.data_ptr(),
        storage_nbytes=storage.nbytes(),
        data_ptr=workspace.data_ptr(),
        storage_offset=workspace.storage_offset(),
        shape=tuple(workspace.shape),
        stride=tuple(workspace.stride()),
        dtype=workspace.dtype,
    )


class _CKVPrefetchState:
    """Cross-layer state for one workspace allocation and execution lane."""

    def __init__(
        self,
        workspace_identity: _CKVWorkspaceIdentity,
        workspace: torch.Tensor,
        workspace_pool: _CKVPrefetchWorkspacePool,
    ) -> None:
        self.workspace_identity = workspace_identity
        self.workspace_storage_ref = weakref.ref(workspace.untyped_storage())
        self.workspace_pool = workspace_pool
        self.layer_caches: list[torch.Tensor | None] = []
        self.pending_layers: dict[int, tuple[Any, int]] = {}
        self.gather_stream: torch.cuda.Stream | None = None
        self.ckv_workspace: torch.Tensor | None = None
        self.ckv_workspace_slot: int | None = None
        self.last_layer_idx: int | None = None

    def begin_step(self) -> None:
        self.wait_for_pending_writes()
        self.pending_layers.clear()
        self.last_layer_idx = None

    def wait_for_pending_writes(self) -> None:
        for event, _ in self.pending_layers.values():
            event.wait()

    def enter_layer(self, layer_idx: int) -> None:
        if self.last_layer_idx is not None and layer_idx <= self.last_layer_idx:
            self.begin_step()
        self.last_layer_idx = layer_idx

    def register_cache(self, layer_idx: int, kv_cache: torch.Tensor) -> None:
        while len(self.layer_caches) <= layer_idx:
            self.layer_caches.append(None)
        self.layer_caches[layer_idx] = kv_cache

    def get_gather_stream(self) -> torch.cuda.Stream:
        if self.gather_stream is None:
            self.gather_stream = torch.cuda.Stream(
                device=self.workspace_identity.device
            )
        return self.gather_stream

    def get_ckv_workspace(self, nbytes: int) -> torch.Tensor:
        if nbytes != self.workspace_pool.slot_nbytes:
            raise ValueError(
                "CKV workspace size changed after persistent allocation: "
                f"pool={self.workspace_pool.slot_nbytes} requested={nbytes}"
            )
        if self.ckv_workspace is None:
            slot, workspace = self.workspace_pool.acquire()
            self.ckv_workspace_slot = slot
            self.ckv_workspace = workspace
        return self.ckv_workspace

    def close(self) -> None:
        for event, _ in self.pending_layers.values():
            event.synchronize()
        self.pending_layers.clear()
        self.last_layer_idx = None
        if self.ckv_workspace_slot is not None:
            self.workspace_pool.release(self.ckv_workspace_slot)
            self.ckv_workspace_slot = None
            self.ckv_workspace = None


_CKV_PREFETCH_STATE_REGISTRIES: weakref.WeakSet = weakref.WeakSet()


class _CKVPrefetchStateRegistry:
    """Builder-owned states partitioned by lane-scoped query workspace."""

    def __init__(self) -> None:
        self.states: dict[_CKVWorkspaceIdentity, _CKVPrefetchState] = {}
        self.workspace_pool: _CKVPrefetchWorkspacePool | None = None
        _CKV_PREFETCH_STATE_REGISTRIES.add(self)

    def _bind_workspace_pool(self, pool: _CKVPrefetchWorkspacePool) -> None:
        if self.workspace_pool is None:
            self.workspace_pool = pool
        elif self.workspace_pool is not pool:
            raise RuntimeError("CKV prefetch registry cannot switch workspace pools")

    def _retire(self, identities: list[_CKVWorkspaceIdentity]) -> None:
        for identity in identities:
            self.states.pop(identity).close()

    def _prune_released_workspaces(self) -> None:
        self._retire(
            [
                identity
                for identity, state in self.states.items()
                if state.workspace_storage_ref() is None
            ]
        )

    def begin_step(self) -> None:
        self._prune_released_workspaces()
        for state in self.states.values():
            state.begin_step()

    def clear(self) -> None:
        self._retire(list(self.states))

    def for_workspace(
        self,
        workspace: torch.Tensor,
        layer_idx: int | None,
        kv_cache: torch.Tensor | None,
        workspace_pool: _CKVPrefetchWorkspacePool,
    ) -> _CKVPrefetchState:
        self._prune_released_workspaces()
        self._bind_workspace_pool(workspace_pool)
        identity = _ckv_workspace_identity(workspace)
        state = self.states.get(identity)
        if state is None:
            stale_identities = [
                existing
                for existing, existing_state in self.states.items()
                if (
                    existing.device == identity.device
                    and existing.data_ptr == identity.data_ptr
                )
                or (
                    layer_idx is not None
                    and kv_cache is not None
                    and layer_idx < len(existing_state.layer_caches)
                    and existing_state.layer_caches[layer_idx] is kv_cache
                )
            ]
            self._retire(stale_identities)
            assert self.workspace_pool is not None
            state = _CKVPrefetchState(identity, workspace, self.workspace_pool)
            self.states[identity] = state
        return state


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if output_tensor.numel() != input_tensor.numel() * group.world_size:
        raise ValueError("CKV all-gather tensors have incompatible sizes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    output_tensor.copy_(group.all_gather(input_tensor, dim=0))


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Return each query token's causal length in the gathered global cache."""
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_tokens].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_tokens,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


@triton.jit
def _map_global_topk_to_gathered_ckv_kernel(
    req_id_ptr,
    token_indices_ptr,
    rank_req_starts_ptr,
    rank_req_lens_ptr,
    out_ptr,
    valid_count_ptr,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_TOPK_TOKENS
    req = tl.load(req_id_ptr + row)
    tok = tl.load(
        token_indices_ptr + row * ti_stride0 + cols * ti_stride1,
        mask=col_mask,
        other=-1,
    )
    owner = (tok // DCP_INTERLEAVE) % DCP_SIZE
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
    valid_tok = col_mask & (tok >= 0)
    req_start = tl.load(
        rank_req_starts_ptr + owner * starts_stride0 + req * starts_stride1,
        mask=valid_tok,
        other=0,
    )
    req_len = tl.load(
        rank_req_lens_ptr + owner * lens_stride0 + req * lens_stride1,
        mask=valid_tok,
        other=0,
    )
    valid = valid_tok & (local_idx >= 0) & (local_idx < req_len)
    gathered_slot = owner * padded_rank_tokens + req_start + local_idx
    valid_i32 = valid.to(tl.int32)
    local_offset = tl.cumsum(valid_i32) - valid_i32
    tile_valid_count = tl.sum(valid_i32)
    output_base = tl.atomic_add(valid_count_ptr + row, tile_valid_count)
    tl.store(
        out_ptr + row * out_stride0 + (output_base + local_offset) * out_stride1,
        gathered_slot,
        mask=valid,
    )


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")

    block_n = 128
    out.fill_(-1)
    valid_counts.zero_()
    _map_global_topk_to_gathered_ckv_kernel[
        (token_indices.shape[0], triton.cdiv(token_indices.shape[1], block_n))
    ](
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        rank_req_starts.stride(0),
        rank_req_starts.stride(1),
        rank_req_lens.stride(0),
        rank_req_lens.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
        padded_rank_tokens,
        DCP_SIZE=dcp_size,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS=token_indices.shape[1],
        BLOCK_N=block_n,
    )


class B12xMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not _is_glm_next_spec(spec):
            return spec
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        if spec.head_size != 512:
            raise ValueError(
                "B12X GLM5Next sparse MLA requires head_size=512, got "
                f"{spec.head_size}."
            )
        return replace(
            spec,
            state_content_bytes=_GLM_NEXT_CACHE_RECORD_BYTES,
            page_tail_bytes_per_token=_GLM_NEXT_INDEX_TAIL_BYTES_PER_TOKEN,
            model_version="glm5_next",
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Sparse index caches share manager blocks with their MLA layer. Keep
        # the layer dimension inside the manager's block so block copies and
        # swaps carry both cache regions together. DeepSeek-V4's index backend
        # already imposes the same constraint, so this preserves its layout.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        module = get_b12x_sparse_mla()
        if module is None:
            return "B12X sparse MLA requires the optional b12x package"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_text_config
            if getattr(hf_config, "index_topk", None) is None:
                return "B12X sparse MLA requires a model with index_topk"
            if _is_glm_next_config(hf_config):
                if recipe_error := _glm_next_recipe_error(hf_config):
                    return recipe_error
                if head_size != 512:
                    return "B12X GLM5Next sparse MLA requires head_size=512"
                if dcp_error := _glm_next_dcp_error(vllm_config):
                    return dcp_error
                return None
            if head_size != 576:
                return "B12X sparse MLA requires head_size=576"
            if int(getattr(hf_config, "kv_lora_rank", 0)) != 512:
                return "B12X sparse MLA requires kv_lora_rank=512"
            if int(getattr(hf_config, "qk_rope_head_dim", 0)) != 64:
                return "B12X sparse MLA requires qk_rope_head_dim=64"
        return None


class B12xGLM5NextMLASparseBackend(B12xMLASparseBackend):
    @staticmethod
    def get_builder_cls() -> type["B12xGLM5NextMLASparseMetadataBuilder"]:
        return B12xGLM5NextMLASparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if not isinstance(spec, MLAAttentionSpec):
            raise TypeError(
                "B12X GLM5Next sparse MLA requires an MLAAttentionSpec, got "
                f"{type(spec).__name__}."
            )
        return super().customize_spec(replace(spec, model_version="glm5_next"))

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Keep the hybrid manager page intact so its FP8 pooled-index tail is
        # copied and recycled with the corresponding MLA page.
        return [MultipleOf(64)]


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    is_spec_decode: bool = False
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1
    cache_seq_lens_per_token: torch.Tensor | None = None
    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None
    ckv_selected_indices: torch.Tensor | None = None
    ckv_active_counts: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_ckv_gather_eligible: bool = False
    ckv_prefetch_registry: _CKVPrefetchStateRegistry | None = None


class B12xMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[B12xMLASparseMetadata]
):
    metadata_cls = B12xMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    requires_glm_next_selector_metadata: bool

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        self.requires_glm_next_selector_metadata = _is_glm_next_config(hf_config)
        if self.requires_glm_next_selector_metadata and (
            dcp_error := _glm_next_dcp_error(vllm_config)
        ):
            raise ValueError(dcp_error)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = (
            self.requires_glm_next_selector_metadata
        )
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        scheduler_config = vllm_config.scheduler_config
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = int(
            getattr(speculative_config, "num_speculative_tokens", 0) or 0
        )
        max_tokens = scheduler_config.max_num_batched_tokens
        max_reqs = int(scheduler_config.max_num_seqs)
        self._ckv_max_reqs = max_reqs
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        if self.requires_glm_next_selector_metadata:
            self._capture_default_state_slot_ids = torch.arange(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_slot_ids = torch.empty(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_state_is_fresh = torch.ones(
                max_reqs, dtype=torch.bool, device=device
            )
            self._capture_num_accepted_tokens = torch.ones(
                max_reqs, dtype=torch.int32, device=device
            )
            self._capture_is_prefilling = torch.zeros(
                max_reqs, dtype=torch.bool, device=device
            )
        self._ckv_gather_requested = (
            self.requires_glm_next_selector_metadata
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        ckv_workspace_requested = (
            self.dcp_world_size > 1 and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        self.ckv_prefetch_registry = (
            _CKVPrefetchStateRegistry() if ckv_workspace_requested else None
        )
        if self._ckv_gather_requested:
            hf_config = vllm_config.model_config.hf_text_config
            ckv_topk_tokens = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
            self.ckv_selected_indices_buffer = torch.empty(
                (max_tokens, ckv_topk_tokens), dtype=torch.int32, device=device
            )
            self.ckv_active_counts_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_lens_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_rank_req_starts_buffer = torch.empty(
                (self.dcp_world_size, max_reqs), dtype=torch.int32, device=device
            )
            self.dcp_local_cu_seq_lens_buffer = torch.empty(
                (max_reqs + 1,), dtype=torch.int32, device=device
            )
        else:
            self.ckv_selected_indices_buffer = None
            self.ckv_active_counts_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None
        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )

    def _stage_glm_next_selector_metadata(
        self,
        *,
        num_reqs: int,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None,
        selector_state_is_fresh: torch.Tensor | None,
        selector_num_accepted_tokens: torch.Tensor | None,
        selector_is_prefilling: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        values = (
            selector_state_slot_ids,
            selector_state_is_fresh,
            selector_num_accepted_tokens,
            selector_is_prefilling,
        )
        if not self.requires_glm_next_selector_metadata:
            if any(value is not None for value in values):
                raise TypeError(
                    "GLM5Next selector metadata was provided to a non-GLM "
                    "B12X sparse MLA builder"
                )
            return (None, None, None, None)

        capacity = int(self._capture_state_slot_ids.numel())
        if not 0 <= num_reqs <= capacity:
            raise ValueError(
                "GLM5Next selector request count exceeds the metadata buffer "
                f"capacity: num_reqs={num_reqs}, capacity={capacity}"
            )
        if not for_cudagraph_capture and any(value is None for value in values):
            raise RuntimeError(
                "B12X GLM5Next sparse MLA requires selector state slots, fresh "
                "flags, accepted-token counts, and prefill flags"
            )

        if for_cudagraph_capture:
            self._capture_state_slot_ids[:num_reqs].copy_(
                self._capture_default_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
        else:
            typed_values = tuple(value for value in values if value is not None)
            if any(
                value.ndim != 1 or value.numel() < num_reqs for value in typed_values
            ):
                raise ValueError(
                    "GLM5Next selector metadata must be one-dimensional and "
                    "cover every padded request row"
                )
            self._capture_state_slot_ids[:num_reqs].fill_(-1)
            self._capture_state_is_fresh[:num_reqs].fill_(True)
            self._capture_num_accepted_tokens[:num_reqs].fill_(1)
            self._capture_is_prefilling[:num_reqs].fill_(False)
            assert selector_state_slot_ids is not None
            assert selector_state_is_fresh is not None
            assert selector_num_accepted_tokens is not None
            assert selector_is_prefilling is not None
            self._capture_state_slot_ids[:num_reqs].copy_(
                selector_state_slot_ids[:num_reqs]
            )
            self._capture_state_is_fresh[:num_reqs].copy_(
                selector_state_is_fresh[:num_reqs]
            )
            self._capture_num_accepted_tokens[:num_reqs].copy_(
                selector_num_accepted_tokens[:num_reqs]
            )
            self._capture_is_prefilling[:num_reqs].copy_(
                selector_is_prefilling[:num_reqs]
            )

        return (
            self._capture_state_slot_ids[:num_reqs],
            self._capture_state_is_fresh[:num_reqs],
            self._capture_num_accepted_tokens[:num_reqs],
            self._capture_is_prefilling[:num_reqs],
        )

    def _build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        *,
        for_cudagraph_capture: bool,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        if self.ckv_prefetch_registry is not None:
            self.ckv_prefetch_registry.begin_step()
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        common = common_attn_metadata
        num_tokens = common.num_actual_tokens
        use_dcp = self.dcp_world_size > 1
        seq_lens = (
            common.dcp_local_seq_lens
            if use_dcp and common.dcp_local_seq_lens is not None
            else common.seq_lens
        )
        metadata.seq_lens = seq_lens
        metadata.ckv_prefetch_registry = self.ckv_prefetch_registry

        if common.max_query_len <= 1 and num_tokens == common.num_reqs:
            per_token_lens = seq_lens[:num_tokens]
        elif not use_dcp and common.positions is not None:
            per_token_lens = common.positions[:num_tokens].to(torch.int32) + 1
        else:
            starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
            query_lens = np.diff(starts)
            seq_lens_cpu_source = (
                common.seq_lens_cpu_upper_bound
                if common.seq_lens_cpu_upper_bound is not None
                else common.seq_lens_cpu
            )
            seq_lens_cpu = seq_lens_cpu_source.numpy().astype(np.int32, copy=False)
            host_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, query_len in enumerate(query_lens):
                if query_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(query_len)
                request_lens = torch.arange(
                    context_len + 1,
                    context_len + int(query_len) + 1,
                    dtype=torch.int32,
                )
                if use_dcp:
                    request_lens = get_dcp_local_seq_lens(
                        request_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                host_lens[start:end] = request_lens.numpy()
            host_tensor = torch.from_numpy(host_lens).pin_memory()
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                host_tensor, non_blocking=True
            )
            per_token_lens = self.cache_seq_lens_per_token_buffer[:num_tokens]

        metadata.cache_seq_lens_per_token = per_token_lens
        metadata.is_spec_decode = False
        if (
            self.num_speculative_tokens > 0
            and 1 < common.max_query_len <= self.num_speculative_tokens + 1
            and common.is_prefilling is not None
        ):
            metadata.is_spec_decode = not bool(
                torch.any(common.is_prefilling[: common.num_reqs])
            )
        if metadata.num_prefills:
            prefill_start = metadata.num_decodes
            prefill_end = prefill_start + metadata.num_prefills + 1
            metadata.prefill_query_lens_cpu = torch.diff(
                common.query_start_loc_cpu[prefill_start:prefill_end]
            )
        if (
            _use_b12x_full_ckv_gather(
                enabled=self._ckv_gather_requested,
                is_glm_next=self.requires_glm_next_selector_metadata,
                dcp_world_size=self.dcp_world_size,
                max_query_len=common.max_query_len,
                num_tokens=num_tokens,
                is_spec_decode=metadata.is_spec_decode,
                min_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS,
                max_tokens=envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS,
            )
            and metadata.num_decode_tokens == 0
        ):
            assert self.ckv_selected_indices_buffer is not None
            assert self.ckv_active_counts_buffer is not None
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            global_seq_lens = common.seq_lens[: common.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_lens.copy_(all_rank_lens)
            rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : common.num_reqs
            ]
            rank_req_starts[:, 0].zero_()
            if common.num_reqs > 1:
                torch.cumsum(rank_req_lens[:, :-1], dim=1, out=rank_req_starts[:, 1:])
            local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: common.num_reqs + 1]
            local_cu_seq_lens[0].zero_()
            torch.cumsum(
                rank_req_lens[self.dcp_rank],
                dim=0,
                out=local_cu_seq_lens[1:],
            )
            rank_totals = rank_req_lens.sum(dim=1).tolist()
            local_total_tokens = int(rank_totals[self.dcp_rank])
            page_size = int(self.kv_cache_spec.block_size)
            padded_total_tokens = (
                (max(int(total) for total in rank_totals) + page_size - 1)
                // page_size
                * page_size
            )
            max_local_capacity = (
                (
                    (envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS + self.dcp_world_size - 1)
                    // self.dcp_world_size
                    + self._ckv_max_reqs * self.cp_kv_cache_interleave_size
                    + page_size
                    - 1
                )
                // page_size
                * page_size
            )
            if 0 < padded_total_tokens <= max_local_capacity:
                metadata.ckv_selected_indices = self.ckv_selected_indices_buffer[
                    :num_tokens
                ]
                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]
                metadata.dcp_rank_req_lens = rank_req_lens
                metadata.dcp_rank_req_starts = rank_req_starts
                metadata.dcp_local_cu_seq_lens = local_cu_seq_lens
                metadata.global_cache_seq_lens_per_req = global_seq_lens
                metadata.dcp_local_total_tokens = local_total_tokens
                metadata.dcp_padded_total_tokens = padded_total_tokens
                metadata.dcp_ckv_gather_eligible = True
        (
            metadata.selector_state_slot_ids,
            metadata.selector_state_is_fresh,
            metadata.selector_num_accepted_tokens,
            metadata.selector_is_prefilling,
        ) = self._stage_glm_next_selector_metadata(
            num_reqs=common.num_reqs,
            for_cudagraph_capture=for_cudagraph_capture,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
        selector_state_slot_ids: torch.Tensor | None = None,
        selector_state_is_fresh: torch.Tensor | None = None,
        selector_num_accepted_tokens: torch.Tensor | None = None,
        selector_is_prefilling: torch.Tensor | None = None,
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            for_cudagraph_capture=False,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> B12xMLASparseMetadata:
        return self._build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def update_draft_decode_metadata(
        self,
        metadata: B12xMLASparseMetadata,
    ) -> None:
        accepted = metadata.selector_num_accepted_tokens
        if not self.requires_glm_next_selector_metadata or accepted is None:
            raise RuntimeError(
                "GLM5Next draft decode metadata requires accepted-token counts"
            )
        accepted.fill_(1)


class B12xGLM5NextMLASparseMetadataBuilder(B12xMLASparseMetadataBuilder):
    # The pooled selector must commit every fresh or extended prompt row
    # through run_prefill; decode commits only accepted prior rows.
    treat_short_extends_as_decodes: ClassVar[bool] = False


class B12xMLASparseImpl(SparseMLACommonImpl[B12xMLASparseMetadata]):
    can_return_lse_for_decode = True
    lse_base_on_e = True
    supports_dense_mha_prefill = False
    supports_pcp = False

    @classmethod
    def reset_kv_cache_binding_state(cls) -> None:
        """Release class-wide CKV state before a cache allocation is replaced.

        This hook resets shared binding registries for every instance of this
        implementation class. The worker therefore invokes it once per concrete
        implementation type during cache unbinding.
        """
        for registry in tuple(_CKV_PREFETCH_STATE_REGISTRIES):
            registry.clear()

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "B12X sparse MLA does not support ALiBi, sliding window, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X sparse MLA supports decoder self-attention only."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_text_config
        self._is_glm_next = _is_glm_next_config(hf_config)
        self.supports_mtp_with_cp_non_trivial_interleave_size = self._is_glm_next
        if self._is_glm_next:
            if recipe_error := _glm_next_recipe_error(hf_config):
                raise ValueError(recipe_error)
            if dcp_error := _glm_next_dcp_error(vllm_config):
                raise ValueError(dcp_error)
            if head_size != 512:
                raise ValueError("B12X GLM5Next sparse MLA requires head_size=512.")
        else:
            if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
                raise ValueError(
                    "B12X sparse MLA requires kv_lora_rank=512 and qk_rope_head_dim=64."
                )
            if head_size != 576:
                raise ValueError("B12X sparse MLA requires head_size=576.")
        if self.topk_indices_buffer is None:
            raise ValueError("B12X sparse MLA requires a top-k index buffer.")
        if kv_cache_dtype != "fp8_ds_mla":
            raise ValueError(
                "B12X sparse MLA requires the packed fp8_ds_mla KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )

        module = get_b12x_sparse_mla()
        if module is None:
            raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
        if not module.is_supported():
            raise RuntimeError("B12X sparse MLA is not supported on this device.")
        for name in ("Caps", "plan", "run_decode", "run_extend"):
            getattr(module, name)
        self._run_decode = module.run_decode
        self._run_extend = module.run_extend
        self._model_type: int | None = None
        self._concat_and_cache_glm_next_mla = None
        if self._is_glm_next:
            self._model_type = int(module.ModelType.GLM_NEXT)
            self._concat_and_cache_glm_next_mla = module.concat_and_cache_glm_next_mla

        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(scheduler_config.max_num_batched_tokens)
        max_seqs = int(scheduler_config.max_num_seqs)
        self._input_num_heads = self.num_heads * self.dcp_world_size
        self._q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self._topk_tokens = int(self.topk_indices_buffer.shape[-1])
        if self._is_glm_next:
            expected_width = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
            if self._topk_tokens != expected_width:
                raise ValueError(
                    "B12X GLM5Next sparse MLA requires a selector output width "
                    f"of {expected_width}, got {self._topk_tokens}."
                )
        self._max_tokens = max_tokens
        self._max_seqs = max_seqs
        self._spec_decode_max_q = int(os.getenv("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", "8"))
        spec_decode_mode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "off").strip().lower()
        )
        disabled_modes = {"0", "false", "off", "no"}
        forced_modes = {"1", "true", "on", "yes"}
        if spec_decode_mode not in {"auto", *disabled_modes, *forced_modes}:
            raise ValueError(
                "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE must be auto, 0, or 1 "
                f"(got {spec_decode_mode!r})"
            )
        self._spec_extend_as_decode = spec_decode_mode not in disabled_modes
        self._spec_extend_as_decode_force = spec_decode_mode in forced_modes
        self._kv_dtype = torch.uint8
        kernel_page_size = (
            int(vllm_config.cache_config.block_size) if self._is_glm_next else 64
        )
        self._ckv_gather_enabled = (
            self._is_glm_next
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        max_ckv_tokens = envs.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        cp_kv_cache_interleave_size = int(
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self._ckv_capacity_tokens = (
            max_ckv_tokens + self.dcp_world_size - 1
        ) // self.dcp_world_size + max_seqs * cp_kv_cache_interleave_size
        self._ckv_local_capacity = 0

        self._module = module
        self._kernel_page_size = 0
        self._set_kernel_page_size(kernel_page_size)
        configured_prefetch_depth = max(0, int(envs.VLLM_B12X_MLA_CKV_PREFETCH_DEPTH))
        configured_workspace_mib = max(
            0, int(envs.VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB)
        )
        workspace_budget_bytes = configured_workspace_mib * 1024 * 1024
        requested_prefetch_depth = (
            configured_prefetch_depth if self._ckv_gather_enabled else 0
        )
        self._ckv_prefetch_depth = _ckv_prefetch_depth_within_budget(
            requested_prefetch_depth,
            workspace_budget_bytes,
            self.dcp_world_size,
            self._ckv_local_capacity,
            _GLM_NEXT_CACHE_RECORD_BYTES,
        )
        self._ckv_workspace_slots = _ckv_prefetch_ring_slots(self._ckv_prefetch_depth)
        self._ckv_workspace_nbytes = (
            _ckv_prefetch_workspace_nbytes(
                self._ckv_prefetch_depth,
                self.dcp_world_size,
                self._ckv_local_capacity,
                _GLM_NEXT_CACHE_RECORD_BYTES,
            )
            if self._ckv_gather_enabled
            else 0
        )
        execution_lanes = _ckv_prefetch_execution_lanes(
            vllm_config.parallel_config.num_ubatches,
            vllm_config.speculative_config is not None,
        )
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self._ckv_workspace_pool = (
            _get_ckv_prefetch_workspace_pool(
                device,
                self._ckv_workspace_nbytes,
                execution_lanes,
            )
            if self._ckv_workspace_nbytes > 0
            else None
        )
        self._ckv_current_chunk_kv_c: torch.Tensor | None = None
        if self._ckv_workspace_pool is not None:
            logger.info_once(
                "Using CKV layer prefetch depth=%d with %.1f MiB for %d "
                "execution lane(s)",
                self._ckv_prefetch_depth,
                self._ckv_workspace_pool.storage.numel() / (1024 * 1024),
                execution_lanes,
            )
        if self._ckv_prefetch_depth < requested_prefetch_depth:
            logger.info_once(
                "Capped CKV prefetch depth from %d to %d for the %d MiB "
                "per-lane workspace budget",
                requested_prefetch_depth,
                self._ckv_prefetch_depth,
                configured_workspace_mib,
            )
        self._pretouch_attention_workspace()
        self.supports_quant_query_input = False

    def _pretouch_attention_workspace(self) -> None:
        """Reserve the largest planned attention scratch before KV profiling."""
        candidates = [
            (
                (
                    (self._max_tokens, self._input_num_heads, self._q_head_dim),
                    torch.bfloat16,
                ),
                *self._decode_plan.shapes_and_dtypes(),
            ),
            (
                (
                    (self._max_tokens, self._input_num_heads, self._q_head_dim),
                    torch.bfloat16,
                ),
                *self._extend_plan.shapes_and_dtypes(),
            ),
        ]
        if self._ckv_extend_plan is not None:
            candidates.append(
                (
                    (
                        (self._max_tokens, self.num_heads, self._q_head_dim),
                        torch.bfloat16,
                    ),
                    *self._ckv_extend_plan.shapes_and_dtypes(),
                )
            )

        def workspace_bytes(specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]):
            total = 0
            for shape, dtype in specs:
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                nbytes = numel * torch.empty((), dtype=dtype).element_size()
                total += (nbytes + 255) // 256 * 256
            return total

        largest = max(candidates, key=workspace_bytes)
        reserved_bytes = workspace_bytes(largest)
        current_workspace_manager().get_simultaneous(*largest)
        logger.info_once(
            "Preallocated %.1f MiB of B12X sparse MLA scratch before KV profiling",
            reserved_bytes / (1024 * 1024),
        )

    def _set_kernel_page_size(self, kernel_page_size: int) -> None:
        if kernel_page_size <= 0 or kernel_page_size % 64:
            raise ValueError(
                "B12X sparse MLA kernel page size must be a positive multiple "
                f"of 64, got {kernel_page_size}."
            )
        if kernel_page_size == self._kernel_page_size:
            return

        def make_plan(mode: str, num_q_heads: int = self._input_num_heads):
            caps_kwargs = dict(
                device=torch.device("cuda", torch.accelerator.current_device_index()),
                num_q_heads=num_q_heads,
                max_q_rows=self._max_tokens,
                max_width=self._topk_tokens,
                dtype=torch.bfloat16,
                kv_dtype=self._kv_dtype,
                head_dim=self._q_head_dim,
                v_head_dim=self.kv_lora_rank,
                mode=mode,
                max_batch=self._max_tokens,
                max_chunks_per_row=max(1, (self._topk_tokens + 63) // 64),
                page_size=kernel_page_size,
            )
            if self._model_type is not None:
                caps_kwargs["model_type"] = self._model_type
            return self._module.plan(self._module.Caps(**caps_kwargs))

        decode_plan = make_plan("decode")
        extend_plan = make_plan("extend")
        self._decode_plan = decode_plan
        self._extend_plan = extend_plan
        self._ckv_extend_plan = (
            make_plan("extend", self.num_heads) if self._ckv_gather_enabled else None
        )
        self._ckv_local_capacity = (
            (self._ckv_capacity_tokens + kernel_page_size - 1)
            // kernel_page_size
            * kernel_page_size
        )
        self._kernel_page_size = kernel_page_size

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self._is_glm_next:
            if kv_cache.ndim != 3 or int(kv_cache.shape[-1]) != 528:
                raise ValueError(
                    "B12X GLM5Next cache must have shape "
                    "[pages, page_size, 528], got "
                    f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}, "
                    f"dtype={kv_cache.dtype}"
                )
            self._set_kernel_page_size(int(kv_cache.shape[1]))

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if not self._is_glm_next:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        del k_scale
        if kv_cache.numel() == 0:
            return
        if int(k_pe.shape[-1]) != 0:
            raise ValueError(
                "B12X GLM5Next cache updates require a zero-width RoPE tensor, "
                f"got shape={tuple(k_pe.shape)}."
            )
        assert self._concat_and_cache_glm_next_mla is not None
        self._concat_and_cache_glm_next_mla(
            kv_c_normed,
            kv_cache,
            slot_mapping.flatten(),
        )

    def uses_full_ckv_dcp(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if torch.cuda.is_current_stream_capturing():
            return False
        return (
            self._ckv_gather_enabled
            and attn_metadata.dcp_ckv_gather_eligible
            and attn_metadata.num_decode_tokens == 0
            and num_tokens == attn_metadata.num_actual_tokens
            and 0 < attn_metadata.dcp_padded_total_tokens <= self._ckv_local_capacity
            and attn_metadata.dcp_local_total_tokens
            <= attn_metadata.dcp_padded_total_tokens
            and all(
                value is not None
                for value in (
                    attn_metadata.ckv_selected_indices,
                    attn_metadata.ckv_active_counts,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    attn_metadata.dcp_local_cu_seq_lens,
                    attn_metadata.global_cache_seq_lens_per_req,
                )
            )
        )

    def _gather_full_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        local_buffer: torch.Tensor,
        gathered_buffer: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if not self.uses_full_ckv_dcp(attn_metadata, attn_metadata.num_actual_tokens):
            raise RuntimeError("full CKV gather called for an ineligible batch")
        if not _is_glm_next_ckv_source_layout(
            kv_cache, page_size=self._kernel_page_size
        ):
            raise ValueError(
                "GLM5Next CKV gather requires native 528-byte records; "
                f"got shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}"
            )
        expected_local_shape = (
            self._ckv_local_capacity,
            _GLM_NEXT_CACHE_RECORD_BYTES,
        )
        expected_gathered_shape = (
            self.dcp_world_size * self._ckv_local_capacity,
            _GLM_NEXT_CACHE_RECORD_BYTES,
        )
        if tuple(local_buffer.shape) != expected_local_shape:
            raise RuntimeError("CKV local workspace has an invalid shape")
        if tuple(gathered_buffer.shape) != expected_gathered_shape:
            raise RuntimeError("CKV gathered workspace has an invalid shape")

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        local_tokens = attn_metadata.dcp_local_total_tokens
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        if stream is not None:
            local_buffer.record_stream(stream)
            gathered_buffer.record_stream(stream)
            stream.wait_stream(torch.cuda.current_stream())
            stream_context = torch.cuda.stream(stream)
        else:
            stream_context = torch.cuda.stream(torch.cuda.current_stream())
        with stream_context:
            if local_tokens:
                ops.cp_gather_cache(
                    src_cache=kv_cache,
                    dst=local_buffer[:local_tokens],
                    block_table=attn_metadata.block_table,
                    cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                    batch_size=attn_metadata.num_reqs,
                )
            if local_tokens < padded_tokens:
                local_buffer[local_tokens:padded_tokens].zero_()
            if stream is None:
                dcp_group = get_dcp_group()
            else:
                from vllm.distributed.parallel_state import (
                    get_dcp_ckv_prefetch_group,
                )

                dcp_group = get_dcp_ckv_prefetch_group()
            _dcp_all_gather_current_stream(
                dcp_group,
                local_buffer[:padded_tokens].view(-1),
                gathered_buffer[: self.dcp_world_size * padded_tokens].view(-1),
            )
        return gathered_buffer.view(
            -1, self._kernel_page_size, _GLM_NEXT_CACHE_RECORD_BYTES
        )

    def _ckv_workspace_views(
        self,
        workspace: torch.Tensor,
        buf_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= buf_idx < self._ckv_workspace_slots:
            raise ValueError(f"CKV workspace slot out of range: {buf_idx}")
        local_nbytes = self._ckv_local_capacity * _GLM_NEXT_CACHE_RECORD_BYTES
        gathered_nbytes = self.dcp_world_size * local_nbytes
        local_buffer = workspace.narrow(0, 0, local_nbytes).view(
            self._ckv_local_capacity,
            _GLM_NEXT_CACHE_RECORD_BYTES,
        )
        gathered_offset = local_nbytes + buf_idx * gathered_nbytes
        gathered_buffer = workspace.narrow(0, gathered_offset, gathered_nbytes).view(
            self.dcp_world_size * self._ckv_local_capacity,
            _GLM_NEXT_CACHE_RECORD_BYTES,
        )
        return local_buffer, gathered_buffer

    def set_ckv_current_chunk_kv(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        del k_pe
        self._ckv_current_chunk_kv_c = kv_c_normed

    @staticmethod
    def _resolve_layer_index(layer: AttentionLayer) -> int | None:
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is not None:
            try:
                return int(layer_idx)
            except (TypeError, ValueError):
                return None
        layer_name = getattr(layer, "layer_name", None)
        if not layer_name:
            return None
        from vllm.model_executor.models.utils import extract_layer_index

        try:
            return extract_layer_index(layer_name)
        except (ValueError, AssertionError, IndexError):
            return None

    def _append_current_chunk_to_gathered(
        self,
        gathered_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> None:
        if self._ckv_current_chunk_kv_c is None or num_tokens == 0:
            return
        assert self._concat_and_cache_glm_next_mla is not None
        assert attn_metadata.global_cache_seq_lens_per_req is not None
        assert attn_metadata.dcp_rank_req_starts is not None
        req_ids = attn_metadata.req_id_per_token[:num_tokens].to(torch.int64)
        global_seq_lens = attn_metadata.global_cache_seq_lens_per_req[
            : attn_metadata.num_reqs
        ]
        seq_len_per_token = global_seq_lens[req_ids].to(torch.int32)
        query_start_loc = attn_metadata.query_start_loc[
            : attn_metadata.num_reqs + 1
        ].to(torch.int32)
        chunk_start = query_start_loc[:-1][req_ids]
        chunk_len = (query_start_loc[1:] - query_start_loc[:-1])[req_ids]
        token_idx = torch.arange(
            num_tokens,
            device=gathered_cache.device,
            dtype=torch.int32,
        )
        global_pos = seq_len_per_token - chunk_len + (token_idx - chunk_start)
        interleave = attn_metadata.cp_kv_cache_interleave_size
        owner = ((global_pos // interleave) % self.dcp_world_size).to(torch.int64)
        local_pos = (
            global_pos // (self.dcp_world_size * interleave) * interleave
            + global_pos % interleave
        ).to(torch.int64)
        rank_req_starts = attn_metadata.dcp_rank_req_starts
        flat_idx = owner * attn_metadata.num_reqs + req_ids
        rank_start = rank_req_starts.reshape(-1)[flat_idx].to(torch.int64)
        slots = owner * attn_metadata.dcp_padded_total_tokens + rank_start + local_pos
        self._concat_and_cache_glm_next_mla(
            self._ckv_current_chunk_kv_c[:num_tokens],
            gathered_cache,
            slots,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cache_page_size = int(kv_c_and_k_pe_cache.shape[1])
        metadata_page_size = int(attn_metadata.block_size)
        if self._is_glm_next and (
            cache_page_size != self._kernel_page_size
            or metadata_page_size != self._kernel_page_size
        ):
            raise RuntimeError(
                "B12X GLM5Next page geometry does not match the bound plan: "
                f"cache={cache_page_size}, metadata={metadata_page_size}, "
                f"plan={self._kernel_page_size}"
            )
        num_tokens = int(q[0].shape[0] if isinstance(q, tuple) else q.shape[0])
        use_decode = _use_b12x_sparse_decode_plan(
            max_query_len=attn_metadata.max_query_len,
            num_tokens=num_tokens,
            num_reqs=attn_metadata.num_reqs,
            is_spec_decode=attn_metadata.is_spec_decode,
            spec_extend_as_decode=self._spec_extend_as_decode,
            spec_extend_as_decode_force=self._spec_extend_as_decode_force,
            spec_decode_max_q=self._spec_decode_max_q,
            max_tokens=self._max_tokens,
        )
        use_ckv_gather = self.uses_full_ckv_dcp(attn_metadata, num_tokens)
        layer_idx = self._resolve_layer_index(layer) if use_ckv_gather else None
        prefetch_registry = attn_metadata.ckv_prefetch_registry
        if use_ckv_gather and self._ckv_workspace_pool is None:
            raise RuntimeError("CKV gather requires a persistent workspace pool")
        if use_ckv_gather and prefetch_registry is None:
            raise RuntimeError("CKV gather requires a prefetch state registry")
        use_persistent_ckv = use_ckv_gather
        if use_ckv_gather:
            assert self._ckv_extend_plan is not None
            plan = self._ckv_extend_plan
            logger.info_once("Using full-CKV gather for GLM5Next B12X DCP prefill")
        else:
            plan = self._decode_plan if use_decode else self._extend_plan
            if use_decode and attn_metadata.max_query_len > 1:
                logger.info_once("Using B12X decode plan for GLM5Next MTP verification")
        input_num_heads = self.num_heads if use_ckv_gather else self._input_num_heads
        q_spec = (
            (self._max_tokens, input_num_heads, self._q_head_dim),
            torch.bfloat16,
        )
        plan_specs = plan.shapes_and_dtypes()
        ckv_specs = (
            (
                (
                    (self._ckv_local_capacity, _GLM_NEXT_CACHE_RECORD_BYTES),
                    torch.uint8,
                ),
                (
                    (
                        self.dcp_world_size * self._ckv_local_capacity,
                        _GLM_NEXT_CACHE_RECORD_BYTES,
                    ),
                    torch.uint8,
                ),
            )
            if use_ckv_gather and not use_persistent_ckv
            else ()
        )
        workspaces = current_workspace_manager().get_simultaneous(
            q_spec, *plan_specs, *ckv_specs
        )
        q_buffer = workspaces[0]
        scratch_end = 1 + len(plan_specs)
        scratch = workspaces[1:scratch_end]

        if isinstance(q, tuple):
            q_nope, q_pe = q
            num_tokens = int(q_nope.shape[0])
            q_all = q_buffer[:num_tokens]
            if int(q_pe.shape[-1]) == 0:
                q_all.copy_(q_nope)
            else:
                ops.concat_mla_q(q_nope, q_pe, q_all)
        else:
            num_tokens = int(q.shape[0])
            q_all = q_buffer[:num_tokens]
            q_all.copy_(q)

        if int(q_all.shape[1]) != input_num_heads:
            raise ValueError(
                "B12X sparse MLA query heads do not match the planned head "
                f"count: {q_all.shape[1]} != {input_num_heads}."
            )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        kv_cache_for_run = kv_c_and_k_pe_cache
        prefetch_state: _CKVPrefetchState | None = None
        ckv_workspace: torch.Tensor | None = None
        if use_ckv_gather:
            if use_persistent_ckv:
                assert prefetch_registry is not None
                assert self._ckv_workspace_pool is not None
                prefetch_state = prefetch_registry.for_workspace(
                    q_buffer,
                    layer_idx,
                    kv_c_and_k_pe_cache,
                    self._ckv_workspace_pool,
                )
                if layer_idx is not None:
                    prefetch_state.enter_layer(layer_idx)
                    prefetch_state.register_cache(layer_idx, kv_c_and_k_pe_cache)
                ckv_workspace = prefetch_state.get_ckv_workspace(
                    self._ckv_workspace_nbytes
                )
                pending = (
                    prefetch_state.pending_layers.pop(layer_idx, None)
                    if layer_idx is not None
                    else None
                )
                if pending is not None:
                    gather_event, current_buf_idx = pending
                    gather_event.wait()
                    _, gathered_buffer = self._ckv_workspace_views(
                        ckv_workspace, current_buf_idx
                    )
                    kv_cache_for_run = gathered_buffer.view(
                        -1,
                        self._kernel_page_size,
                        _GLM_NEXT_CACHE_RECORD_BYTES,
                    )
                    self._append_current_chunk_to_gathered(
                        kv_cache_for_run,
                        attn_metadata,
                        num_tokens,
                    )
                else:
                    prefetch_state.wait_for_pending_writes()
                    current_buf_idx = (
                        layer_idx % self._ckv_workspace_slots
                        if layer_idx is not None
                        else 0
                    )
                    local_buffer, gathered_buffer = self._ckv_workspace_views(
                        ckv_workspace, current_buf_idx
                    )
                    kv_cache_for_run = self._gather_full_ckv(
                        kv_c_and_k_pe_cache,
                        attn_metadata,
                        local_buffer,
                        gathered_buffer,
                    )
            else:
                local_buffer, gathered_buffer = workspaces[scratch_end:]
                kv_cache_for_run = self._gather_full_ckv(
                    kv_c_and_k_pe_cache,
                    attn_metadata,
                    local_buffer,
                    gathered_buffer,
                )

            if (
                prefetch_state is not None
                and ckv_workspace is not None
                and layer_idx is not None
            ):
                targets = _ckv_prefetch_target_indices(
                    layer_idx,
                    self._ckv_prefetch_depth,
                    prefetch_state.layer_caches,
                    prefetch_state.pending_layers,
                )
                prefetch_stream = (
                    prefetch_state.get_gather_stream() if targets else None
                )
                for target_idx in targets:
                    assert prefetch_stream is not None
                    target_cache = prefetch_state.layer_caches[target_idx]
                    assert target_cache is not None
                    target_buf_idx = target_idx % self._ckv_workspace_slots
                    target_local, target_gathered = self._ckv_workspace_views(
                        ckv_workspace, target_buf_idx
                    )
                    self._gather_full_ckv(
                        target_cache,
                        attn_metadata,
                        target_local,
                        target_gathered,
                        stream=prefetch_stream,
                    )
                    target_event = torch.cuda.Event(blocking=False)
                    target_event.record(prefetch_stream)
                    prefetch_state.pending_layers[target_idx] = (
                        target_event,
                        target_buf_idx,
                    )
            assert attn_metadata.ckv_selected_indices is not None
            assert attn_metadata.ckv_active_counts is not None
            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            selected_indices = attn_metadata.ckv_selected_indices[
                :num_tokens, : topk_indices.shape[1]
            ]
            active_counts = attn_metadata.ckv_active_counts[:num_tokens]
            _map_global_topk_to_gathered_ckv(
                attn_metadata.req_id_per_token[:num_tokens],
                topk_indices,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                selected_indices,
                active_counts,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
            )
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            cache_seq_lens = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_tokens,
            ).contiguous()
            torch.minimum(active_counts, cache_seq_lens, out=active_counts)
            _mask_page_table_after_nsa_len(selected_indices, active_counts)
        elif self.dcp_world_size > 1:
            block_stride_rows = _selected_index_block_stride_rows(
                kv_c_and_k_pe_cache,
                block_size=attn_metadata.block_size,
                is_glm_next=self._is_glm_next,
            )
            selected_indices, active_counts = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            block_stride_rows = _selected_index_block_stride_rows(
                kv_c_and_k_pe_cache,
                block_size=attn_metadata.block_size,
                is_glm_next=self._is_glm_next,
            )
            selected_indices, active_counts = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_tokens],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        if not use_ckv_gather:
            cache_seq_lens = attn_metadata.cache_seq_lens_per_token
            assert cache_seq_lens is not None
            cache_seq_lens = cache_seq_lens[:num_tokens].contiguous()
        binding = plan.bind(
            scratch=scratch,
            q=q_all,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seq_lens,
            nsa_cache_seqlens_int32=active_counts,
        )
        run = self._run_decode if plan is self._decode_plan else self._run_extend
        run_kwargs = dict(
            binding=binding,
            kv_cache=kv_cache_for_run,
            sm_scale=self.scale,
            v_head_dim=self.kv_lora_rank,
            return_lse=self.need_to_return_lse_for_decode,
            lse_scale="natural",
        )
        if self._model_type is not None:
            run_kwargs["model_type"] = self._model_type
        result = run(**run_kwargs)
        if self.need_to_return_lse_for_decode:
            output, lse = result
            return output, lse
        assert isinstance(result, torch.Tensor)
        return result, None
