# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..b12x_layers import B12xLinearMethod, B12xMHC, collapse, stream_mean
from ..b12x_layers import B12xRMSNorm as RMSNorm
from ..ced import ced_decoder_start, gather_rows, scatter_rows
from ..common.engram import Engram, EngramLayout, NgramHashState
from ..common.mm_preprocess import image_sentinel_mask
from .b12x_attention import DeepseekV41B12xAttention

if typing.TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def _select_dsv4_attn_cls(vllm_config):
    backend = vllm_config.attention_config.backend
    if backend not in (None, AttentionBackendEnum.B12X):
        raise ValueError("DeepSeek V4.1 requires B12X")
    return DeepseekV41B12xAttention


def _use_sequence_parallel(vllm_config):
    return False


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.use_sequence_parallel:
            self.attn.wo_b.reduce_results = False
        if config.scoring_func != "sqrtsoftplus" or not config.norm_topk_prob:
            raise ValueError("V4.1 requires normalized sqrtsoftplus routing")
        if getattr(config, "gate_temp", 1.0) != 1.0:
            raise ValueError("V4.1 requires gate_temp=1")
        is_draft = extract_layer_index(prefix) >= config.num_hidden_layers
        moe_config = vllm_config
        if is_draft:
            # Only the expert counts differ; use the same DSV4 TP implementation.
            moe_config = copy.copy(vllm_config)
            moe_config.model_config = copy.copy(vllm_config.model_config)
            moe_config.model_config.hf_config = copy.copy(config)
            moe_config.model_config.hf_config.n_routed_experts = (
                config.dspark_n_routed_experts
            )
            moe_config.model_config.hf_config.num_experts_per_tok = (
                config.dspark_num_experts_per_tok
            )
        gate = ReplicatedLinear(
            config.hidden_size,
            moe_config.model_config.hf_config.n_routed_experts,
            bias=False,
            prefix=f"{prefix}.ffn.gate",
        )
        gate.quant_method = B12xLinearMethod()
        gate.out_dtype = torch.float32
        self.ffn = DeepseekV4MoE(
            moe_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
            gate=gate,
            image_sentinel_lo=0 if is_draft else 129264,
            image_sentinel_count=1,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        self._b12x_mhc = B12xMHC(config)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_output = x if residual is not None else None
        apply_engram = self.engram is not None and engram_hashes is not None
        if residual is None:
            residual = x
        elif apply_engram:
            # Engram mutates the reconstructed residual; do not fuse across it.
            residual = self._b12x_mhc.post(x, residual, post_mix, res_mix)
            previous_output = None
        if apply_engram:
            if residual.ndim == 2:
                residual = (
                    residual[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
                )
            residual = self.engram(
                residual, engram_hashes[:, self.engram.layer_hash_index], engram_mask
            )
        fn = self.hc_attn_fn_broadcast if residual.ndim == 2 else self.hc_attn_fn
        residual, post_mix, res_mix, x, attn_pre = self._b12x_mhc.pre(
            residual,
            fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
            pre_mix,
            previous_output=previous_output,
            previous_post=post_mix if previous_output is not None else None,
            previous_comb=res_mix if previous_output is not None else None,
        )
        global_kv_ready = None
        if ced_indices is not None:
            # Decoder global KV needs every encoder row, but decoder queries,
            # residual updates and experts only need the selected replay rows.
            global_kv_ready = self.attn.prepare_global_kv(positions, x)
            x = gather_rows(x, ced_indices)
            residual = gather_rows(residual, ced_indices)
            post_mix = gather_rows(post_mix, ced_indices)
            res_mix = gather_rows(res_mix, ced_indices)
            attn_pre = gather_rows(attn_pre, ced_indices)
            positions = gather_rows(positions, ced_indices)
            if input_ids is not None:
                input_ids = gather_rows(input_ids, ced_indices)
        x = self.attn(positions, x, None, global_kv_ready=global_kv_ready)
        residual, post_mix, res_mix, x, ffn_pre = self._b12x_mhc.post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight,
            attn_pre,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.ced_decoder_start = ced_decoder_start(config)
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = False
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if vllm_config.parallel_config.use_ubatching:
            raise ValueError(
                "V4.1 fixed native layer buffers require ubatching/DBO disabled"
            )
        if vllm_config.lora_config is not None:
            raise ValueError("V4.1 native kernels do not support LoRA adapters")
        if quant_config is None or quant_config.get_name() != "deepseek_v41_fp8":
            raise ValueError("V4.1 requires its native block32 quantization config")
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = None

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # Hashing reads the runner's accepted-only lookback, never KV slot state.
        self.engram_hash: NgramHashState | None = None
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            local_engram = any(
                isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            )
            if local_engram:
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix
        self.disk_engram = (
            self.engram_layout is not None and self.engram_layout.table_memory == "disk"
        )
        self.file_backed_engram = (
            self.engram_layout is not None
            and self.engram_layout.table_memory in ("ram", "disk")
        )
        if self.disk_engram:
            plan = self.engram_layout.plans[0]
            self.register_buffer(
                "prepared_engram_hashes",
                torch.empty(
                    (plan.caps.max_tokens, len(self.engram_layout.layer_ids), 24),
                    dtype=torch.int64,
                    device=plan.caps.device,
                ),
                persistent=False,
            )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def prepare_disk_engram(self, input_ids, query_start_loc, lookback_token_ids):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        engrams = tuple(
            layer.engram
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if getattr(layer, "engram", None) is not None
        )
        try:
            for engram in engrams:
                engram.invalidate_disk_output()
            if self.engram_hash is None:
                raise RuntimeError("Disk Engram requires initialized hash state")
            hashes = self.prepared_engram_hashes[: input_ids.shape[0]]
            self.engram_hash.run_native(
                input_ids,
                ~image_sentinel_mask(input_ids),
                query_start_loc,
                lookback_token_ids,
                hashes,
            )
            for engram in engrams:
                engram.prepare_disk(
                    hashes[:, engram.layer_hash_index], self.engram_hash.num_tokens
                )
            # Independent tables can issue concurrent NVMe reads, but all rows
            # are ready before entering the existing target CUDA graph.
            if getattr(self.engram_layout, "disk_prefetch_max_tokens", 0):
                for engram in engrams:
                    engram.finish_disk()
        except BaseException:
            for engram in engrams:
                try:
                    engram.invalidate_disk_output(clear=True)
                except BaseException:
                    logger.exception(
                        "Failed to drain an Engram prefetch during cleanup"
                    )
            raise

    def prepare_dummy_engram(self, num_tokens):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        self.prepared_engram_hashes.fill_(-1)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            if getattr(layer, "engram", None) is not None:
                layer.engram.prepare_dummy_output(num_tokens)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if self.disk_engram and input_ids is not None:
            engram_hashes = self.prepared_engram_hashes[: input_ids.shape[0]]
            engram_mask = ~image_sentinel_mask(input_ids)
        elif self.engram_hash is not None and input_ids is not None:
            attn_metadata = (
                get_forward_context().attn_metadata
                if is_forward_context_available()
                else None
            )
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict):
                swa_metadata = attn_metadata[self.engram_swa_prefix]
                query_start_loc = swa_metadata.query_start_loc
                if lookback_token_ids is None:
                    raise ValueError(
                        "V4.1 Engram requires accepted-only lookback_token_ids"
                    )
            else:
                # Profiling must exercise the same native hash/lookup/projection
                # and gate kernels, not silently omit the Engram layers.
                query_start_loc = torch.zeros(
                    2, dtype=torch.int32, device=input_ids.device
                )
                query_start_loc[1] = input_ids.shape[0]
                lookback_token_ids = input_ids.new_full((1, 3), -1)
            image_mask = image_sentinel_mask(input_ids)
            engram_mask = ~image_mask
            engram_hashes = self.engram_hash(
                input_ids,
                positions,
                query_start_loc,
                image_mask,
                lookback_token_ids,
            )
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                engram = getattr(layer, "engram", None)
                if engram is not None:
                    engram.prepare_embeddings(engram_hashes[:, engram.layer_hash_index])

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        decoder_compacted = False
        if (
            ced_indices is not None
            and self.ced_decoder_start is not None
            and self.start_layer > self.ced_decoder_start
        ):
            # Pipeline transport keeps its original full-row ABI. A decoder
            # stage gathers only the rows materialized by the preceding stage.
            hidden_states = gather_rows(hidden_states, ced_indices)
            pre_mix = gather_rows(pre_mix, ced_indices)
            positions = gather_rows(positions, ced_indices)
            if input_ids is not None:
                input_ids = gather_rows(input_ids, ced_indices)
            decoder_compacted = True
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            boundary_indices = ced_indices if idx == self.ced_decoder_start else None
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
                ced_indices=boundary_indices,
            )
            if boundary_indices is not None:
                positions = gather_rows(positions, ced_indices)
                if input_ids is not None:
                    input_ids = gather_rows(input_ids, ced_indices)
                decoder_compacted = True
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = layer._b12x_mhc.post(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_state = stream_mean(aux_recon)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                if decoder_compacted:
                    aux_hidden_state = scatter_rows(
                        aux_hidden_state, ced_indices, full_num_tokens
                    )
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            else:
                hidden_states = layer._b12x_mhc.post(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            if decoder_compacted:
                hidden_states = scatter_rows(
                    hidden_states, ced_indices, full_num_tokens
                )
                pre_mix = scatter_rows(pre_mix, ced_indices, full_num_tokens)
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            buffer_states = (
                scatter_rows(hidden_states, ced_indices, full_num_tokens)
                if decoder_compacted
                else hidden_states
            )
            self._mtp_hidden_buffer[:full_num_tokens].copy_(buffer_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if decoder_compacted:
            hidden_states = scatter_rows(hidden_states, ced_indices, full_num_tokens)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if ".engram.embed_tokens." in name:
                if is_pp_missing_parameter(name, self):
                    continue
                module_name, _, leaf = name.rpartition(".")
                embedding = self.get_submodule(module_name)
                loaded_params.update(
                    f"{module_name}.{loaded}"
                    for loaded in embedding.load_weights(((leaf, loaded_weight),))
                )
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Native block32 linears retain the checkpoint-compatible scale name."""
    return "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV4MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV4MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def checkpoint_file_weight_filter(self, name: str) -> bool:
        return (
            self.model.file_backed_engram
            and re.fullmatch(r"layers\.\d+\.engram\.embed\.(?:weight|scale)", name)
            is not None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        # AutoWeightsLoader may revisit this child for another contiguous
        # prefix group. Finalize only in the root model's post-load hook.
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights()
        for module in self.modules():
            if isinstance(module, DeepseekV41B12xAttention):
                module.setup_wo_projection()
            if isinstance(module, Engram):
                module.process_weights_after_loading()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
