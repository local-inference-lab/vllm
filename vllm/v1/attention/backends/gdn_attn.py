# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from copy import copy
from dataclasses import dataclass, replace
from typing import Literal

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.b12x_gdn_metadata import B12xGdnMixedMetadata
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


@triton.jit(do_not_specialize=["num_reqs", "source_stride", "accepted_stride"])
def _fill_uniform_spec_metadata(
    source,
    accepted_source,
    state_indices,
    accepted,
    sequence_masks,
    token_indices,
    query_start_loc,
    num_reqs,
    source_stride,
    accepted_stride,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = token // WINDOW
    column = token % WINDOW
    live = row < num_reqs
    state = tl.load(source + row.to(tl.int64) * source_stride + column, live, other=0)
    tl.store(state_indices + token, state, live)
    tl.store(token_indices + token, token, live)
    first = live & (column == 0)
    count = tl.load(
        accepted_source + row.to(tl.int64) * accepted_stride, first, other=0
    )
    tl.store(accepted + row, count, first)
    tl.store(sequence_masks + row, True, first)
    tl.store(query_start_loc + row, token, first)
    if tl.program_id(0) == 0:
        tl.store(query_start_loc + num_reqs, num_reqs * WINDOW)


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

    checkpoint: MambaPrefillCheckpointMetadata | None = None
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
    uniform_spec_sequence_length: int | None = None  # None for ragged batches

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None
    aiter_prefill_metadata: object | None = None
    b12x_prefill_live_counts: torch.Tensor | None = None
    b12x_mixed: B12xGdnMixedMetadata | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Required when reusing a metadata build across equivalent Mamba cache
    # groups whose state block tables differ.
    num_reqs: int = 0
    seq_lens: torch.Tensor | None = None

    is_uniform_spec_decode: bool = False


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True
    supports_kda_state_recovery: bool = False

    # Runner-owned stable storage, with NULL_BLOCK_ID in padded request rows.
    mamba_aligned_state_indices: torch.Tensor | None = None

    reorder_batch_threshold: int = 1

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_decode_kernel,
            _resolve_gdn_prefill_backend,
        )

        model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", "")
        if model_type in {"qwen3_8_flash_next_text", "qwen4_exp_text"}:
            _, prefill = _resolve_gdn_prefill_backend(vllm_config)
            decode, _ = _resolve_gdn_decode_kernel(vllm_config)
            if prefill == decode == "b12x":
                return AttentionCGSupport.ALWAYS
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "aiter_flydsl", "b12x"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)
        self._check_chunk_metadata_override(type(self), self.gdn_prefill_backend)
        self._b12x_prefill_live_counts = (
            torch.zeros(2, dtype=torch.int32, device=device)
            if self.gdn_prefill_backend == "b12x"
            else None
        )

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        use_kda_state_recovery = (
            self.supports_kda_state_recovery
            and vllm_config.cache_config.use_kda_recoverssm
        )
        self.state_index_columns = 1 if use_kda_state_recovery else self.num_spec + 1
        self._b12x_mixed = (
            B12xGdnMixedMetadata(
                max_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                max_seqs=vllm_config.scheduler_config.max_num_seqs,
                state_columns=self.num_spec + 1,
                device=device,
            )
            if self.gdn_prefill_backend == "b12x" and not use_kda_state_recovery
            else None
        )
        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # Each group owns its recurrent state-index buffers; batch-level
        # query and acceptance tensors remain shared across groups.
        self.supports_update_block_table = type(self) is GDNAttentionMetadataBuilder
        self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.state_index_columns),
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
        self._decode_state_indices_source: torch.Tensor | None = None
        self._decode_state_indices_view: torch.Tensor | None = None
        self._reuse_spec_decode_inputs = (
            envs.VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH and not use_kda_state_recovery
        )
        # Constant sources for the uniform spec-decode fast path. They are
        # copied into the builder-owned graph buffers above, never handed to
        # the layers directly: a full cudagraph captured from a uniform batch
        # replays for a padded batch of the same size, which the generic path
        # builds into those same buffers, so both paths must share addresses.
        self._uniform_spec_masks_cpu = torch.ones(
            self.decode_cudagraph_max_bs, dtype=torch.bool
        )
        self._uniform_spec_tokens = torch.arange(
            self.decode_cudagraph_max_bs, dtype=torch.int32, device=device
        )
        self._uniform_spec_query_start = torch.arange(
            self.decode_cudagraph_max_bs + 1, dtype=torch.int32, device=device
        ) * (self.num_spec + 1)

    def _can_reuse_spec_inputs(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> bool:
        return (
            self._reuse_spec_decode_inputs
            and self.use_spec_decode
            and self.use_full_cuda_graph
            and self.vllm_config.cache_config.mamba_cache_mode == "align"
            and self.mamba_aligned_state_indices is not None
            and num_accepted_tokens is not None
            and num_decode_draft_tokens_cpu is not None
            and 0 < m.num_actual_tokens <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens == m.num_reqs * (self.num_spec + 1)
            and bool(torch.all(num_decode_draft_tokens_cpu == self.num_spec))
            and bool(torch.all(torch.diff(m.query_start_loc_cpu) == self.num_spec + 1))
            and (m.is_prefilling is None or not bool(torch.any(m.is_prefilling)))
        )

    def _build_uniform_spec_decode(
        self,
        m: CommonAttentionMetadata | GDNAttentionMetadata,
        num_accepted_tokens: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Populate uniform speculative metadata in builder-owned graph storage.

        Everything the layers read is written into the builder-owned graph
        buffers, exactly where the generic path writes it, so a full cudagraph
        captured through either path replays correctly through the other.
        """
        num_reqs = m.num_reqs
        num_tokens = m.num_actual_tokens
        source = self.mamba_aligned_state_indices
        assert source is not None
        spec_state_indices = self.spec_state_indices_tensor[:num_reqs]
        spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
        spec_token_indx = self.spec_token_indx[:num_tokens]
        spec_query_start_loc = self.spec_query_start_loc[: num_reqs + 1]
        accepted = self.num_accepted_tokens[:num_reqs]
        if source.is_cuda:
            _fill_uniform_spec_metadata[(triton.cdiv(num_tokens, 128),)](
                source,
                num_accepted_tokens,
                spec_state_indices,
                accepted,
                spec_sequence_masks,
                spec_token_indx,
                spec_query_start_loc,
                num_reqs,
                source.stride(0),
                num_accepted_tokens.stride(0),
                WINDOW=self.num_spec + 1,
                BLOCK=128,
            )
        else:
            spec_state_indices.copy_(source[:num_reqs, : self.num_spec + 1])
            spec_sequence_masks.fill_(True)
            spec_token_indx.copy_(self._uniform_spec_tokens[:num_tokens])
            spec_query_start_loc.copy_(self._uniform_spec_query_start[: num_reqs + 1])
            accepted.copy_(num_accepted_tokens[:num_reqs])
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_reqs,
            num_spec_decode_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            spec_query_start_loc=spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=self._uniform_spec_masks_cpu[:num_reqs],
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=self.non_spec_token_indx[:0],
            num_accepted_tokens=accepted,
            num_reqs=num_reqs,
            seq_lens=m.seq_lens,
            is_uniform_spec_decode=True,
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

    def _can_reuse_decode_inputs(self) -> bool:
        return (
            not self.use_spec_decode
            and self.vllm_config.cache_config.mamba_cache_mode == "align"
            and self.mamba_aligned_state_indices is not None
        )

    @staticmethod
    def _check_chunk_metadata_override(
        builder_cls: type["GDNAttentionMetadataBuilder"], backend: str
    ) -> None:
        """Reject a subclass whose chunk metadata the AITER path would skip.

        AITER brings its own varlen prefill metadata and never calls
        ``_build_chunk_metadata``, so a builder that overrides it would lose
        that override without a word. No in-tree subclass can get here --
        ``_resolve_gdn_prefill_backend`` only selects this backend for models
        with 128-dim GDN key and value heads, which the KDA builders are not --
        but a future one should be told rather than quietly ignored.
        """
        if backend != "aiter_flydsl":
            return
        if (
            builder_cls._build_chunk_metadata
            is GDNAttentionMetadataBuilder._build_chunk_metadata
        ):
            return
        raise RuntimeError(
            f"{builder_cls.__name__} builds its own FLA chunk metadata, which "
            "the 'aiter_flydsl' GDN prefill backend does not use. Select a "
            "different gdn_prefill_backend for this model."
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
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
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
        if m.uniform_decode_graph and self._can_reuse_spec_inputs(
            m, num_accepted_tokens, num_decode_draft_tokens_cpu
        ):
            assert num_accepted_tokens is not None
            # Uniform graphs read the packed decode metadata, not the mixed
            # prefill worklists. Mixed graphs keep their complete staging even
            # when the request rows happen to be uniform on this invocation.
            return self._build_uniform_spec_decode(m, num_accepted_tokens)
        mixed = getattr(self, "_b12x_mixed", None)
        if mixed is not None:
            mixed.stage(
                m,
                self._get_state_indices(m.block_table_tensor, m.seq_lens, m.num_reqs),
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                checkpoint_block_size=(
                    self.kv_cache_spec.block_size
                    if self.kv_cache_spec.num_prefill_checkpoint_blocks > 0
                    and self.vllm_config.cache_config.mamba_cache_mode == "align"
                    else None
                ),
            )
        if self._can_reuse_spec_inputs(
            m, num_accepted_tokens, num_decode_draft_tokens_cpu
        ):
            assert num_accepted_tokens is not None
            return replace(
                self._build_uniform_spec_decode(m, num_accepted_tokens),
                b12x_mixed=mixed,
            )

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = self._get_state_indices(
            m.block_table_tensor,
            m.seq_lens,
            m.num_reqs,
        )

        uniform_spec_sequence_length = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            # A speculative row contains exactly one target token followed by
            # its draft tokens. Profiling may provide a zero-draft marker for
            # a long prefill, which must not enter the bounded decode kernel.
            query_lens_cpu = query_start_loc_cpu.diff()
            spec_sequence_masks_cpu = (
                (num_decode_draft_tokens_cpu >= 0)
                & (num_decode_draft_tokens_cpu <= self.num_spec)
                & (query_lens_cpu == num_decode_draft_tokens_cpu + 1)
            )
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            # A batch whose rows all drafted nothing still has to run the
            # speculative path: that is the only path that applies each row's
            # accepted-token offset to the recurrent state.
            if num_spec_decodes == 0:
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            # V2 already excludes prefills from full decode graphs via
            # has_prefill. Classify first chunks as prefills to mask recycled
            # state; resumed one-token chunks can still use the decode kernels.
            assert m.seq_lens_cpu_upper_bound is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            no_prior_state = (query_lens_cpu > 0) & (
                m.seq_lens_cpu_upper_bound <= query_lens_cpu
            )
            # Capture batches also have seq_len == query_len, but are not
            # prefills.
            if m.is_prefilling is not None:
                no_prior_state &= m.is_prefilling
            else:
                no_prior_state = torch.zeros_like(no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m.replace(is_prefilling=no_prior_state),
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
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
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if spec_query_lens_cpu.numel() > 0:
                first_spec_sequence_length = int(spec_query_lens_cpu[0])
                if first_spec_sequence_length > 0 and bool(
                    torch.all(spec_query_lens_cpu == first_spec_sequence_length)
                ):
                    uniform_spec_sequence_length = first_spec_sequence_length

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
                # Padded sequences trail the spec decodes, so slice them off
                # rather than gather with the host mask (an H2D copy + a kernel).
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.state_index_columns
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
                    spec_sequence_masks_cpu, : self.state_index_columns
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
            if num_prefills == 0 and num_decodes == 0:
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        aiter_prefill_metadata: object | None = None
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

            if self.gdn_prefill_backend == "aiter_flydsl":
                # AITER carries its own reusable varlen metadata and has no use
                # for FLA's chunk indices, so it replaces them rather than
                # extending what _build_chunk_metadata returns.
                assert prefill_query_start_loc_cpu is not None
                aiter_prefill_metadata = (
                    rocm_aiter_ops.build_gdn_flydsl_prefill_metadata(
                        torch.diff(prefill_query_start_loc_cpu).tolist(),
                        cu_seqlens=prefill_query_start_loc,
                    )
                )
            elif self.gdn_prefill_backend != "b12x":
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
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        checkpoint: MambaPrefillCheckpointMetadata | None = None
        if num_prefills > 0:
            request_rows = list(range(m.num_reqs))
            if spec_sequence_masks_cpu is not None:
                request_rows = (~spec_sequence_masks_cpu).nonzero().flatten().tolist()
            checkpoint = self.checkpoint_builder.build(m, request_rows)

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
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

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
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

        prefill_live_counts = getattr(self, "_b12x_prefill_live_counts", None)
        if prefill_live_counts is not None:
            prefill_live_counts[0].fill_(num_prefills)
            prefill_live_counts[1].fill_(num_prefill_tokens)
        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            checkpoint=checkpoint,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            aiter_prefill_metadata=aiter_prefill_metadata,
            b12x_prefill_live_counts=prefill_live_counts,
            b12x_mixed=mixed,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            uniform_spec_sequence_length=uniform_spec_sequence_length,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            num_reqs=m.num_reqs,
            seq_lens=m.seq_lens,
        )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and not self._can_reuse_decode_inputs()
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        del slot_mapping
        assert metadata.num_reqs > 0
        assert metadata.seq_lens is not None
        if (
            metadata.is_uniform_spec_decode
            and metadata.b12x_mixed is None
            and self._reuse_spec_decode_inputs
            and self.mamba_aligned_state_indices is not None
        ):
            assert metadata.num_accepted_tokens is not None
            return self._build_uniform_spec_decode(
                metadata, metadata.num_accepted_tokens
            )
        mixed = None
        if metadata.b12x_mixed is not None:
            mixed = self._b12x_mixed
            assert mixed is not None
            mixed.copy_and_refresh_from(
                metadata.b12x_mixed,
                self._get_state_indices(
                    blk_table, metadata.seq_lens, metadata.num_reqs
                ),
                blk_table,
            )
        prefill_live_counts = getattr(self, "_b12x_prefill_live_counts", None)
        if prefill_live_counts is not None:
            prefill_live_counts[0].fill_(metadata.num_prefills)
            prefill_live_counts[1].fill_(metadata.num_prefill_tokens)

        if (
            metadata.is_uniform_spec_decode
            and self._reuse_spec_decode_inputs
            and self.mamba_aligned_state_indices is not None
        ):
            assert metadata.num_accepted_tokens is not None
            return replace(
                self._build_uniform_spec_decode(metadata, metadata.num_accepted_tokens),
                b12x_mixed=mixed,
                b12x_prefill_live_counts=prefill_live_counts,
            )

        if (
            metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and self._can_reuse_decode_inputs()
        ):
            source = self.mamba_aligned_state_indices
            assert source is not None
            if (
                self._decode_state_indices_source is not source
                or self._decode_state_indices_view is None
                or self._decode_state_indices_view.shape[0] != metadata.num_reqs
            ):
                self._decode_state_indices_source = source
                self._decode_state_indices_view = source[: metadata.num_reqs, 0]
            updated = copy(metadata)
            updated.non_spec_state_indices_tensor = self._decode_state_indices_view
            updated.b12x_mixed = mixed
            updated.b12x_prefill_live_counts = prefill_live_counts
            return updated

        m = metadata
        checkpoint = (
            m.checkpoint.regather_state_indices(blk_table)
            if m.checkpoint is not None
            else None
        )
        blk_table = self._get_state_indices(blk_table, m.seq_lens, m.num_reqs)
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        elif m.num_prefills == 0:
            # Same as build(): padded sequences trail the spec decodes.
            spec_indices = blk_table[: m.num_spec_decodes, : self.state_index_columns]
        else:
            spec_indices = blk_table[masks, : self.state_index_columns]
            non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            checkpoint=checkpoint,
            b12x_mixed=mixed,
            b12x_prefill_live_counts=prefill_live_counts,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        if self.gdn_prefill_backend == "b12x":
            lengths = torch.diff(m.query_start_loc_cpu)
            if self.use_spec_decode and m.max_query_len <= self.num_spec + 1:
                accepted = torch.ones(
                    m.num_reqs, dtype=torch.int32, device=m.query_start_loc.device
                )
                drafts = torch.where(lengths > 1, lengths - 1, -1)
                return self.build(0, m, accepted, drafts)
            return self.build(0, m)

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
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
