# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.kv_cache_utils import (
    request_endpoint_cache_debug,
    request_endpoint_cache_torch_copy,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.mamba_utils import (
    ENDPOINT_COPY_META_FIXED,
    EndpointCopyRecord,
    MambaSpecDecodeGPUContext,
    endpoint_layer_states,
    materialize_endpoints_torch,
    preprocess_mamba_align_fused_kernel,
    verify_endpoints,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _page_sig(tensor: torch.Tensor, block: int) -> str:
    """A cheap fingerprint of one block page for endpoint debugging."""
    page = tensor[block].float()
    flat = page.flatten()
    return (
        f"sum={page.sum().item():.6e} abs={page.abs().sum().item():.6e} "
        f"head={[round(v, 5) for v in flat[:3].tolist()]} "
        f"tail={[round(v, 5) for v in flat[-3:].tolist()]}"
    )


def _bytes_sig(raw: torch.Tensor, dtype: torch.dtype) -> str:
    """Fingerprint of a raw (uint8) shadow page viewed as ``dtype``."""
    elem = torch.empty((), dtype=dtype).element_size()
    usable = raw.numel() // elem * elem
    page = raw[:usable].view(dtype).float()
    return (
        f"sum={page.sum().item():.6e} abs={page.abs().sum().item():.6e} "
        f"head={[round(v, 5) for v in page[:3].tolist()]} "
        f"tail={[round(v, 5) for v in page[-3:].tolist()]}"
    )


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            # Every recurrent layer's state views in the context's state
            # order (torch reference path of the endpoint materialization).
            self._mamba_layer_states: list[tuple[torch.Tensor, ...]] = []
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            # Request-endpoint cache: keep every request's committed state in
            # a shadow slot so a finished request's state survives the step
            # that was already in flight when it finished.
            self._endpoint_shadow = bool(envs.VLLM_K3_REQUEST_ENDPOINT_CACHE)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            group_ids: list[int] = []
            specs: list[MambaSpec] = []
            for i, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    group_ids.append(i)
                    specs.append(spec)
            assert specs, "no mamba layers in the model"
            assert all(specs[0] == s for s in specs)
            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
            self._first_mamba_layer_name = kv_cache_config.kv_cache_groups[
                group_ids[0]
            ].layer_names[0]
        return self._mamba_group_ids, self._mamba_spec

    def _debug_mamba_states(self) -> list[torch.Tensor] | None:
        """The first recurrent layer's ``[conv_state, temporal_state]``."""
        name = getattr(self, "_first_mamba_layer_name", None)
        if name is None:
            return None
        forward_context = self.vllm_config.compilation_config.static_forward_context
        attention = forward_context.get(name)
        return getattr(attention, "kv_cache", None)

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        if self._mamba_ctx is None:
            copy_funcs = self.model.get_mamba_state_copy_func()
            # Both SD and DS conv layouts support a >0 spec-decode shift: the
            # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
            # ``token_bias = num_accepted - 1`` window shift per conv layout
            # (SD: contiguous slice; DS: per-dim-row strided slice), matching
            # the V1 ``get_conv_copy_spec`` semantics.
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                num_state_types=len(copy_funcs),
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(
                    n, dtype=dtype, device=self.device
                ),
            )
        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            # block_tables are batch-order slices of the persistent
            # input_block_tables (stable data_ptr), so the metadata is captured
            # once here and reused across steps.
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in mamba_group_ids],
            )
            self._mamba_layer_states = endpoint_layer_states(
                kv_cache_config, forward_context, mamba_group_ids
            )
        if self._endpoint_shadow and not ctx.has_endpoint_shadow:
            ctx.ensure_endpoint_shadow(self.max_num_reqs)
        return ctx

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
            snapshot_to_shadow=self._endpoint_shadow,
        )

    def materialize_request_endpoints(
        self,
        copies: list[Any],
        req_id_to_index: dict[str, int],
    ) -> None:
        """Write finished requests' committed states into pool blocks.

        Each ``MambaEndpointStateCopy`` names the destination block and either
        the request's shadow pages (``from_shadow``; ``token_bias`` selects the
        conv shift and the temporal slot) or its own state blocks per
        recurrent group. Runs on the current stream before the finished
        requests' slots are recycled, so it is ordered after the in-flight
        step that may have overwritten their state slots and before any
        consumer's copy-on-write of the destination block.

        The fused kernel performs the copies; with the torch-copy switch the
        per-state torch reference performs them instead. In debug mode the
        kernel result is verified against the reference for every state and
        the mismatches are logged on the first tensor-parallel rank.
        """
        if not copies or not self._align_mode:
            return
        ctx = self._mamba_ctx
        if ctx is None or not ctx.is_initialized:
            logger.warning_once(
                "Request-endpoint copies arrived before the recurrent-state "
                "context was initialized; %d entr(ies) stay unmaterialized.",
                len(copies),
            )
            return
        num_groups = ctx.num_groups
        width = ENDPOINT_COPY_META_FIXED + 3 * num_groups
        meta = np.full((len(copies), width), -1, dtype=np.int32)
        records: list[EndpointCopyRecord] = []
        rows = 0
        for copy in copies:
            req_idx = req_id_to_index.get(copy.req_id)
            if copy.from_shadow:
                if req_idx is None or not ctx.has_endpoint_shadow:
                    logger.warning(
                        "Request-endpoint copy for %s skipped: shadow source "
                        "unavailable (slot=%s, shadow=%s)",
                        copy.req_id,
                        req_idx,
                        ctx.has_endpoint_shadow,
                    )
                    continue
                meta[rows, 0] = 1
                meta[rows, 1] = req_idx
                meta[rows, 2] = copy.token_bias
            else:
                if (
                    len(copy.conv_src_block_ids) != num_groups
                    or len(copy.temporal_src_block_ids) != num_groups
                ):
                    logger.warning(
                        "Request-endpoint copy for %s skipped: %d/%d source "
                        "blocks for %d recurrent groups",
                        copy.req_id,
                        len(copy.conv_src_block_ids),
                        len(copy.temporal_src_block_ids),
                        num_groups,
                    )
                    continue
                meta[rows, 0] = 0
                meta[rows, 1] = 0
                meta[rows, 2] = copy.token_bias
                meta[rows, 4 + num_groups : 4 + 2 * num_groups] = (
                    copy.conv_src_block_ids
                )
                meta[rows, 4 + 2 * num_groups : width] = copy.temporal_src_block_ids
            if len(copy.dst_block_ids) != num_groups:
                logger.warning(
                    "Request-endpoint copy for %s skipped: %d destination "
                    "blocks for %d recurrent groups",
                    copy.req_id,
                    len(copy.dst_block_ids),
                    num_groups,
                )
                meta[rows] = -1
                continue
            meta[rows, 4 : 4 + num_groups] = copy.dst_block_ids
            records.append(
                EndpointCopyRecord(
                    from_shadow=bool(copy.from_shadow),
                    req_idx=req_idx if copy.from_shadow and req_idx is not None else 0,
                    token_bias=int(copy.token_bias),
                    dst_block_ids=tuple(int(b) for b in copy.dst_block_ids),
                    conv_src_block_ids=tuple(copy.conv_src_block_ids),
                    temporal_src_block_ids=tuple(copy.temporal_src_block_ids),
                )
            )
            rows += 1
        if rows == 0:
            return
        torch_copy = request_endpoint_cache_torch_copy()
        debug = request_endpoint_cache_debug() and get_tensor_model_parallel_rank() == 0
        conv_dim_first = is_conv_state_dim_first()
        layer_states = self._mamba_layer_states
        if debug:
            self._debug_log_endpoint_sources(copies, req_id_to_index, ctx)
        if not torch_copy or debug:
            copy_meta = torch.from_numpy(meta[:rows]).to(self.device)
            trace = None
            if debug:
                trace = torch.zeros(
                    (rows, ctx.num_layers * ctx.num_state_types, 4),
                    dtype=torch.int64,
                    device=self.device,
                )
            ctx.run_endpoint_materialize(copy_meta, trace)
            if debug:
                torch.accelerator.synchronize()
                self._debug_log_trace(records, trace)
                self._debug_log_verification(
                    "kernel", records, layer_states, conv_dim_first
                )
        if torch_copy:
            materialize_endpoints_torch(ctx, layer_states, records, conv_dim_first)
            if debug:
                torch.accelerator.synchronize()
                self._debug_log_verification(
                    "torch", records, layer_states, conv_dim_first
                )

    def _debug_log_trace(
        self, records: list[EndpointCopyRecord], trace: torch.Tensor | None
    ) -> None:
        """Compare the addresses the kernel resolved with the host
        expectation for every record and state."""
        ctx = self._mamba_ctx
        if ctx is None or trace is None:
            return
        expected = ctx.expected_endpoint_addresses(records)
        actual = trace.cpu().tolist()
        mismatches: list[dict[str, Any]] = []
        for record_idx, per_state in enumerate(expected):
            for state_idx, want in enumerate(per_state):
                got = tuple(actual[record_idx][state_idx])
                if got != want:
                    mismatches.append(
                        {
                            "record": record_idx,
                            "state_idx": state_idx,
                            "kernel": [hex(v) for v in got],
                            "expected": [hex(v) for v in want],
                        }
                    )
        logger.info(
            "[endpoint] trace records=%d states=%d address_mismatches=%d first=%s",
            len(records),
            len(expected[0]) if expected else 0,
            len(mismatches),
            mismatches[0] if mismatches else None,
        )

    def _debug_log_verification(
        self,
        path: str,
        records: list[EndpointCopyRecord],
        layer_states: list[tuple[torch.Tensor, ...]],
        conv_dim_first: bool,
    ) -> None:
        ctx = self._mamba_ctx
        assert ctx is not None
        mismatches = verify_endpoints(ctx, layer_states, records, conv_dim_first)
        total = len(records) * len(layer_states) * ctx.num_state_types
        by_record: dict[int, int] = {}
        for entry in mismatches:
            by_record[entry["record"]] = by_record.get(entry["record"], 0) + 1
        logger.info(
            "[endpoint] verify path=%s records=%d states=%d mismatched=%d "
            "per_record=%s first=%s",
            path,
            len(records),
            total,
            len(mismatches),
            {records[i].dst_block_ids[0]: n for i, n in sorted(by_record.items())},
            mismatches[0] if mismatches else None,
        )

    def _debug_log_endpoint_sources(self, copies, req_id_to_index, ctx) -> None:
        states = self._debug_mamba_states()
        if states is None or len(states) < 2 or not ctx.is_initialized:
            return
        torch.accelerator.synchronize()
        strides = ctx.shadow_page_strides_cpu[:2]
        base0 = int(ctx.state_base_addrs[0].item())
        base1 = int(ctx.state_base_addrs[1].item())
        logger.info(
            "[endpoint] views conv_ptr=%#x ctx_base0=%#x temporal_ptr=%#x "
            "ctx_base1=%#x "
            "shadow0_ptr=%#x ctx_shadow0=%#x shadow1_ptr=%#x ctx_shadow1=%#x layers=%d",
            states[0].data_ptr(),
            base0,
            states[1].data_ptr(),
            base1,
            ctx.shadow_buffers[0].data_ptr() if ctx.has_endpoint_shadow else 0,
            int(ctx.shadow_base_addrs[0].item()),
            ctx.shadow_buffers[1].data_ptr() if ctx.has_endpoint_shadow else 0,
            int(ctx.shadow_base_addrs[1].item()),
            len(self._mamba_layer_states),
        )
        for copy in copies:
            if copy.from_shadow:
                req_idx = req_id_to_index.get(copy.req_id)
                if req_idx is None or not ctx.has_endpoint_shadow:
                    continue
                bufs = ctx.shadow_buffers
                conv_page = bufs[0][req_idx * strides[0] : (req_idx + 1) * strides[0]]
                slot = req_idx * ctx.shadow_temporal_slots + copy.token_bias
                temporal_page = bufs[1][slot * strides[1] : (slot + 1) * strides[1]]
                conv_sig = _bytes_sig(conv_page, states[0].dtype)
                temporal_sig = _bytes_sig(temporal_page, states[1].dtype)
                src_desc = f"shadow slot={req_idx} temporal_slot={slot}"
            else:
                conv_sig = _page_sig(states[0], copy.conv_src_block_ids[0])
                temporal_sig = _page_sig(states[1], copy.temporal_src_block_ids[0])
                src_desc = (
                    f"blocks conv={copy.conv_src_block_ids[0]} "
                    f"temporal={copy.temporal_src_block_ids[0]}"
                )
            logger.info(
                "[endpoint] source req=%s %s bias=%d dst0=%d conv=%s temporal=%s "
                "state0=(base=%#x stride=%d) state1=(base=%#x stride=%d) "
                "shape0=%s stride0=%s shape1=%s stride1=%s",
                copy.req_id,
                src_desc,
                copy.token_bias,
                copy.dst_block_ids[0],
                conv_sig,
                temporal_sig,
                base0,
                int(strides[0]),
                base1,
                int(strides[1]),
                tuple(states[0].shape),
                tuple(states[0].stride()),
                tuple(states[1].shape),
                tuple(states[1].stride()),
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
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.max_query_len
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = input_batch.max_seq_len_upper_bound

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # A row is a spec-decode row only when its whole prompt is already
                # computed, i.e. exactly one non-draft (decode) token is scheduled.
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        mamba_attn_metadata = MambaHybridAttnMetadata(
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        return build_attn_metadata(
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
            max_seq_len_upper_bound=input_batch.max_seq_len_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            is_prefilling=is_prefilling,
        )

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        num_reqs = idx_mapping.shape[0]
        if not num_reqs:
            return

        if not isinstance(num_sampled, int):
            # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
            # kernel skips them rather than scattering with a host-side gather.
            _scatter_num_accepted_kernel[(num_reqs,)](
                idx_mapping, num_sampled, self.num_accepted_tokens_gpu
            )
        else:
            # Fill with single value.
            _fill_num_accepted_kernel[(num_reqs,)](
                idx_mapping, self.num_accepted_tokens_gpu, max(num_sampled, 1)
            )

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
