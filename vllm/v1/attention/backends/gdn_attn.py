# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.models.glm5next_cudagraph import (
    is_glm53_full_graph_path,
    require_glm53_full_graph_capacity,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Required when reusing a metadata build across equivalent Mamba cache
    # groups whose state block tables differ.
    num_reqs: int = 0
    seq_lens: torch.Tensor | None = None


class _PersistentGDNMetadataArena:
    """Fixed-address metadata storage bounded by scheduler maxima."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_spec: int,
        device: torch.device,
    ) -> None:
        if max_num_seqs <= 0 or max_num_batched_tokens <= 0:
            raise ValueError("GDN metadata arena capacities must be positive")
        self.request_rows = max_num_seqs
        self.token_rows = max_num_batched_tokens
        self.state_width = num_spec + 1
        chunk_rows = (max_num_batched_tokens + 63) // 64 + max_num_seqs
        conv_rows = 2 * max(1024, (max_num_batched_tokens + 7) // 8 + max_num_seqs)
        self.query_start_loc = torch.empty(
            max_num_seqs + 1, dtype=torch.int32, device=device
        )
        self.prefill_query_start_loc = torch.empty_like(self.query_start_loc)
        self.seq_lens = torch.empty(max_num_seqs, dtype=torch.int32, device=device)
        self.state_indices = torch.empty(
            (max_num_seqs, self.state_width), dtype=torch.int32, device=device
        )
        self.has_initial_state = torch.empty(
            max_num_seqs, dtype=torch.bool, device=device
        )
        self.chunk_indices = torch.empty(
            (chunk_rows, 2), dtype=torch.int32, device=device
        )
        self.chunk_offsets = torch.empty(
            max_num_seqs + 1, dtype=torch.int32, device=device
        )
        self.batch_ptr = torch.full(
            (conv_rows,), NULL_BLOCK_ID, dtype=torch.int32, device=device
        )
        self.token_chunk_offset_ptr = torch.full_like(self.batch_ptr, NULL_BLOCK_ID)
        self.spec_sequence_masks = torch.empty(
            max_num_seqs, dtype=torch.bool, device=device
        )
        self.num_accepted_tokens = torch.empty(
            max_num_seqs, dtype=torch.int32, device=device
        )
        self.spec_query_start_loc = torch.empty_like(self.query_start_loc)
        self.non_spec_query_start_loc = torch.empty_like(self.query_start_loc)
        self.spec_state_indices = torch.empty_like(self.state_indices)
        self.non_spec_state_indices = torch.empty(
            max_num_seqs, dtype=torch.int32, device=device
        )
        self.spec_token_indices = torch.empty(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.non_spec_token_indices = torch.empty_like(self.spec_token_indices)
        pin_memory = device.type == "cuda"
        self._conv_batch_cpu = torch.empty(
            conv_rows, dtype=torch.int32, pin_memory=pin_memory
        )
        self._conv_offset_cpu = torch.empty_like(
            self._conv_batch_cpu, pin_memory=pin_memory
        )
        self._conv_nums_cpu = torch.empty(
            max_num_seqs, dtype=torch.int32, pin_memory=pin_memory
        )
        self.nums_dict = {
            8: {
                "nums": self._conv_nums_cpu,
                "tot": 0,
                "mlist": self._conv_batch_cpu,
                "mlist_len": 0,
                "offsetlist": self._conv_offset_cpu,
                "batch_ptr": self.batch_ptr,
                "token_chunk_offset_ptr": self.token_chunk_offset_ptr,
            }
        }

    def require_fits(self, *, num_reqs: int, num_tokens: int) -> None:
        if num_reqs > self.request_rows:
            raise ValueError("GDN metadata request capacity exceeded")
        if num_tokens > self.token_rows:
            raise ValueError("GDN metadata token capacity exceeded")

    @staticmethod
    def stage(
        storage: torch.Tensor,
        value: torch.Tensor,
        *,
        fill: int | bool | None = None,
    ) -> torch.Tensor:
        rows = value.shape[0]
        if rows > storage.shape[0]:
            raise ValueError("GDN metadata arena staging capacity exceeded")
        storage[:rows].copy_(value, non_blocking=True)
        if fill is not None and rows < storage.shape[0]:
            storage[rows:].fill_(fill)
        return storage[:rows]

    def stage_causal_conv(
        self, query_start_loc_cpu: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        num_reqs = query_start_loc_cpu.numel() - 1
        if num_reqs > self.request_rows:
            raise ValueError("GDN causal-conv request capacity exceeded")
        total_programs = 0
        for request_idx in range(num_reqs):
            seq_len = int(
                query_start_loc_cpu[request_idx + 1] - query_start_loc_cpu[request_idx]
            )
            num_programs = (seq_len + 7) // 8
            self._conv_nums_cpu[request_idx] = num_programs
            end = total_programs + num_programs
            if end > self.batch_ptr.shape[0]:
                raise ValueError("GDN causal-conv program capacity exceeded")
            self._conv_batch_cpu[total_programs:end].fill_(request_idx)
            if num_programs:
                torch.arange(
                    num_programs,
                    dtype=torch.int32,
                    out=self._conv_offset_cpu[total_programs:end],
                )
            total_programs = end
        self.batch_ptr.fill_(NULL_BLOCK_ID)
        self.token_chunk_offset_ptr.fill_(NULL_BLOCK_ID)
        self.batch_ptr[:total_programs].copy_(
            self._conv_batch_cpu[:total_programs], non_blocking=True
        )
        self.token_chunk_offset_ptr[:total_programs].copy_(
            self._conv_offset_cpu[:total_programs], non_blocking=True
        )
        entry = self.nums_dict[8]
        entry["tot"] = total_programs
        entry["mlist_len"] = total_programs
        return self.nums_dict, self.batch_ptr, self.token_chunk_offset_ptr


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    _glm53_graph_safe_gdn_marker = "fixed-address-gdn-metadata-v1"
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        if (
            not isinstance(kv_cache_spec, MambaSpec)
            or not is_glm53_full_graph_path(vllm_config)
            or kv_cache_spec.block_size != 2304
        ):
            return AttentionCGSupport.UNIFORM_BATCH
        require_glm53_full_graph_capacity(vllm_config)
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
            B12xGLM5NextMLASparseMetadataBuilder,
        )

        marker = getattr(
            B12xGLM5NextMLASparseMetadataBuilder,
            "_glm53_graph_safe_selector_marker",
            None,
        )
        if marker != "fixed-address-vectorized-pooled-selector-v1":
            raise RuntimeError(
                "GLM-5.3 mixed FULL requires graph-safe pooled selection"
            )
        return AttentionCGSupport.ALWAYS

    mamba_aligned_state_indices: torch.Tensor | None = None

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.arena = _PersistentGDNMetadataArena(
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            num_spec=self.num_spec,
            device=device,
        )

    def _get_state_indices(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        if (
            self.vllm_config.cache_config.mamba_cache_mode == "align"
            and self.mamba_aligned_state_indices is not None
        ):
            return self.mamba_aligned_state_indices[:num_reqs]
        return mamba_get_block_table_tensor(
            block_table,
            seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        chunk_indices = prepare_chunk_indices(
            prefill_query_start_loc_cpu, FLA_CHUNK_SIZE
        )
        chunk_offsets = prepare_chunk_offsets(
            prefill_query_start_loc_cpu, FLA_CHUNK_SIZE
        )
        return (
            self.arena.stage(self.arena.chunk_indices, chunk_indices),
            self.arena.stage(self.arena.chunk_offsets, chunk_offsets),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        self.arena.require_fits(num_reqs=m.num_reqs, num_tokens=m.num_actual_tokens)

        query_start_loc = self.arena.stage(
            self.arena.query_start_loc, m.query_start_loc
        )
        seq_lens = self.arena.stage(self.arena.seq_lens, m.seq_lens)
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = self._get_state_indices(
            m.block_table_tensor,
            m.seq_lens,
            m.num_reqs,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.arena.stage(
                    self.arena.spec_sequence_masks,
                    spec_sequence_masks_cpu.to(
                        device=query_start_loc.device, non_blocking=True
                    ),
                    fill=False,
                )

        if spec_sequence_masks is None:
            assert m.is_prefilling is not None
            # Mamba cache pages are not allocator-zeroed. Fresh one-token
            # requests must take the prefill path so has_initial_state=False
            # masks both convolution and recurrent state.
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = self.arena.stage_causal_conv(
                non_spec_query_start_loc_cpu
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        if prefill_query_start_loc is not None:
            prefill_query_start_loc = self.arena.stage(
                self.arena.prefill_query_start_loc,
                prefill_query_start_loc,
            )
        if prefill_state_indices is not None:
            storage = (
                self.arena.state_indices
                if prefill_state_indices.ndim == 2
                else self.arena.non_spec_state_indices
            )
            prefill_state_indices = self.arena.stage(storage, prefill_state_indices)
        if prefill_has_initial_state is not None:
            prefill_has_initial_state = self.arena.stage(
                self.arena.has_initial_state,
                prefill_has_initial_state,
                fill=False,
            )

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            num_reqs=m.num_reqs,
            seq_lens=seq_lens,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        del slot_mapping
        assert metadata.num_reqs > 0
        assert metadata.seq_lens is not None

        state_indices = self._get_state_indices(
            blk_table,
            metadata.seq_lens,
            metadata.num_reqs,
        )
        spec_sequence_masks_cpu = metadata.spec_sequence_masks_cpu
        if spec_sequence_masks_cpu is None:
            spec_state_indices = None
            non_spec_state_indices = state_indices[:, 0]
        else:
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            spec_state_indices = state_indices[
                spec_sequence_masks_cpu, : self.num_spec + 1
            ]
            non_spec_state_indices = state_indices[non_spec_sequence_masks_cpu, 0]
        prefill_state_indices = metadata.prefill_state_indices
        if metadata.num_prefills > 0:
            if spec_sequence_masks_cpu is None and metadata.num_decodes > 0:
                prefill_state_indices = non_spec_state_indices[metadata.num_decodes :]
            else:
                prefill_state_indices = non_spec_state_indices

        spec_sequence_masks = metadata.spec_sequence_masks
        spec_token_indx = metadata.spec_token_indx
        non_spec_token_indx = metadata.non_spec_token_indx
        spec_query_start_loc = metadata.spec_query_start_loc
        num_accepted_tokens = metadata.num_accepted_tokens
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_state_indices is not None
            assert spec_sequence_masks is not None
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            assert spec_query_start_loc is not None
            assert num_accepted_tokens is not None

            self.spec_state_indices_tensor[: metadata.num_spec_decodes].copy_(
                spec_state_indices, non_blocking=True
            )
            spec_state_indices = self.spec_state_indices_tensor[: metadata.num_reqs]
            spec_state_indices[metadata.num_spec_decodes :].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[: metadata.num_reqs].copy_(
                spec_sequence_masks[: metadata.num_reqs], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[: metadata.num_reqs]

            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: metadata.num_reqs + 1].copy_(
                spec_query_start_loc[: metadata.num_reqs + 1], non_blocking=True
            )
            spec_query_start_loc = self.spec_query_start_loc[: metadata.num_reqs + 1]

            self.num_accepted_tokens[: metadata.num_reqs].copy_(
                num_accepted_tokens[: metadata.num_reqs], non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[: metadata.num_reqs]

        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[: metadata.num_decodes].copy_(
                non_spec_state_indices[: metadata.num_decodes], non_blocking=True
            )
            non_spec_state_indices = self.non_spec_state_indices_tensor[
                : metadata.num_reqs
            ]
            non_spec_state_indices[metadata.num_decodes :].fill_(NULL_BLOCK_ID)

            assert non_spec_query_start_loc is not None
            self.non_spec_query_start_loc[: metadata.num_reqs + 1].copy_(
                non_spec_query_start_loc[: metadata.num_reqs + 1],
                non_blocking=True,
            )
            non_spec_query_start_loc = self.non_spec_query_start_loc[
                : metadata.num_reqs + 1
            ]

        return replace(
            metadata,
            spec_state_indices_tensor=spec_state_indices,
            non_spec_state_indices_tensor=non_spec_state_indices,
            prefill_state_indices=prefill_state_indices,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            num_accepted_tokens=num_accepted_tokens,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """Build capture metadata under the selected path's capacity contract."""
        m = common_attn_metadata

        if is_glm53_full_graph_path(self.vllm_config):
            self.arena.require_fits(num_reqs=m.num_reqs, num_tokens=m.num_actual_tokens)
            if m.is_prefilling is not None and bool(m.is_prefilling.any()):
                return self.build(0, m, None, None)
            accepted = self.arena.num_accepted_tokens[: m.num_reqs]
            torch.diff(m.query_start_loc, out=accepted)
            return self.build(0, m, accepted, (accepted - 1).cpu())

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
