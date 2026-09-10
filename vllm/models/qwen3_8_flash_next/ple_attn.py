# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-address mixed-batch metadata for b12x PLE convolution."""

from dataclasses import dataclass, replace

import torch

from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    ShortConvAttentionBackend,
    ShortConvAttentionMetadata,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# vLLM reserves block zero; b12x PLE uses -1 for disabled requests.
_B12X_NULL_STATE_SLOT = -1


class PLEGraphInputs:
    """Builder-owned device inputs whose addresses survive batch changes."""

    def __init__(self, max_seqs: int, max_tokens: int, device: torch.device):
        self.max_seqs = max_seqs
        self.max_tokens = max_tokens
        self.query_start_loc = torch.zeros(
            max_seqs + 1, dtype=torch.int32, device=device
        )
        self.state_slot_ids = torch.full(
            (max_seqs,), _B12X_NULL_STATE_SLOT, dtype=torch.int64, device=device
        )
        self.state_is_fresh = torch.ones(max_seqs, dtype=torch.bool, device=device)
        self.num_accepted_tokens = torch.ones(
            max_seqs, dtype=torch.int32, device=device
        )
        self.request_is_prefill = torch.zeros(max_seqs, dtype=torch.bool, device=device)
        self.num_seqs = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    def stage(
        self, metadata: ShortConvAttentionMetadata, query_start_loc: torch.Tensor
    ):
        num_seqs = metadata.num_reqs
        if num_seqs > self.max_seqs or query_start_loc.numel() < num_seqs + 1:
            raise ValueError("PLE request metadata exceeds its planned capacity")
        self.query_start_loc[: num_seqs + 1].copy_(query_start_loc[: num_seqs + 1])
        self.query_start_loc[num_seqs + 1 :].zero_()
        self.state_slot_ids.fill_(_B12X_NULL_STATE_SLOT)
        self.state_is_fresh.fill_(True)
        self.num_accepted_tokens.fill_(1)
        self.request_is_prefill.zero_()
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        if num_decodes:
            state_d = metadata.state_indices_tensor_d
            if state_d is None:
                raise RuntimeError("decode PLE metadata is missing state indices")
            if state_d.ndim == 2:
                state_d = state_d[:, 0]
            self.state_slot_ids[:num_decodes].copy_(state_d[:num_decodes])
            self.state_is_fresh[:num_decodes] = False
            if metadata.num_accepted_tokens is not None:
                self.num_accepted_tokens[:num_decodes].copy_(
                    metadata.num_accepted_tokens[:num_decodes]
                )
        if num_prefills:
            state_p = metadata.state_indices_tensor_p
            has_initial = metadata.has_initial_states_p
            if state_p is None or has_initial is None:
                raise RuntimeError(
                    "prefill PLE metadata is missing state indices or flags"
                )
            end = num_decodes + num_prefills
            self.state_slot_ids[num_decodes:end].copy_(state_p[:num_prefills])
            self.state_is_fresh[num_decodes:end].copy_(~has_initial[:num_prefills])
            self.request_is_prefill[num_decodes:end] = True
        self.state_slot_ids.masked_fill_(
            self.state_slot_ids == NULL_BLOCK_ID, _B12X_NULL_STATE_SLOT
        )
        self.num_seqs.fill_(num_seqs)
        self.num_tokens.copy_(query_start_loc[num_seqs : num_seqs + 1])


@dataclass
class PLEAttentionMetadata(ShortConvAttentionMetadata):
    graph_inputs: PLEGraphInputs | None = None


class PLEAttentionBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "QWEN3_8_PLE"

    @staticmethod
    def get_builder_cls() -> type["PLEAttentionMetadataBuilder"]:
        return PLEAttentionMetadataBuilder


class PLEAttentionMetadataBuilder(ShortConvAttentionMetadataBuilder):
    metadata_cls = PLEAttentionMetadata
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        scheduler = vllm_config.scheduler_config
        self.graph_inputs = PLEGraphInputs(
            scheduler.max_num_seqs, scheduler.max_num_batched_tokens, device
        )

    def build(
        self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs
    ):
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build, **kwargs
        )
        assert isinstance(metadata, PLEAttentionMetadata)
        self.graph_inputs.stage(metadata, common_attn_metadata.query_start_loc)
        return replace(metadata, graph_inputs=self.graph_inputs)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        accepted = None
        if self.use_spec_decode:
            accepted = torch.ones_like(common_attn_metadata.seq_lens, dtype=torch.int32)
        return self.build(0, common_attn_metadata, num_accepted_tokens=accepted)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        assert metadata.graph_inputs is not None
        updated = super().update_block_table(metadata, blk_table, slot_mapping)
        assert isinstance(updated, PLEAttentionMetadata)
        self.graph_inputs.stage(updated, metadata.graph_inputs.query_start_loc)
        return replace(updated, graph_inputs=self.graph_inputs)
