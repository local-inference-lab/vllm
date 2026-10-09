# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the
DSV4.1 DSpark, which reuses the target model's architecture similarly to MTP.
Its weights ship in the target checkpoint under the ``mtp.{0,1,2}.*`` prefix.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import copy
from bisect import bisect_left

import regex as re
import torch
import torch.nn as nn
from b12x.gemm import block_fp8_linear

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4Model
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV4DecoderLayer,
    _linear_scale_param_name,
    _use_sequence_parallel,
)
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_layer_prefix,
    register_b12x_layer,
    register_b12x_unit_provider,
    set_b12x_preparation_provider,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from . import l2_prefetch
from .b12x_layers import B12xRMSNorm as RMSNorm
from .b12x_layers import (
    _execution_capacities,
)
from .runtime import build_l2_prefetch_plans

logger = init_logger(__name__)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class _ContextKVProjection:
    """KV-only view of checkpoint block32 weights, packed once after loading."""

    def __init__(self, attn: nn.Module, capacity: int) -> None:
        fused = attn.fused_wqa_wkv
        start = attn.q_lora_rank
        if start % 32:
            raise ValueError("DSpark context KV must start on a checkpoint scale block")
        self.weight = block_fp8_linear.pack_weight(
            fused.weight[start:],
            fused.weight_scale_inv[start // 32 :],
            block_size=(32, 32),
        )
        self.capacities = tuple(
            sorted(
                {
                    capacity,
                    *(bound for bound in _execution_capacities() if bound <= capacity),
                }
            )
        )
        self.plans = tuple(
            block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=fused.weight.device,
                    max_tokens=bound,
                    in_features=fused.weight.shape[1],
                    out_features=fused.weight.shape[0] - start,
                    block_size=(32, 32),
                    output_mode="provided",
                )
            )
            for bound in self.capacities
        )
        register_b12x_unit_provider(self)

    def get_b12x_preparation_units(
        self, layer: object, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if self.weight.weight.values.is_meta:
            return ()

        def make_call(state, *, bound: int):
            from b12x.preparation import PreparedCall

            source = torch.zeros(
                (bound, self.weight.in_features),
                dtype=torch.bfloat16,
                device=state.device,
            )
            output = torch.empty(
                (bound, self.weight.out_features, 1),
                dtype=torch.bfloat16,
                device=source.device,
            )
            scratch = [
                torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
                for spec in state.scratch.scratch_specs()
            ]
            binding = state.bind(
                scratch=scratch,
                source=source,
                packed_weight=self.weight,
                output=output,
            )
            return PreparedCall(
                run=lambda: state.run_binding(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(self.weight,),
            )

        requests = tuple(
            plan.request(
                name=f"dspark.context_kv.{id(self):x}.m{bound}",
                prepare_call=lambda state, bound=bound: make_call(state, bound=bound),
                benchmark_call=lambda state, bound=bound: make_call(state, bound=bound),
            )
            for bound, plan in zip(self.capacities, self.plans)
        )
        return (
            B12xPreparationUnit(
                name="DSparkContextKV",
                key=(id(self), self.capacities),
                requests=requests,
                stage="weights",
            ),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        index = bisect_left(self.capacities, x.shape[0])
        if index == len(self.capacities):
            raise ValueError("DSpark context rows exceed prepared capacity")
        rows = x.shape[0]
        plan = self.plans[index]
        out = torch.empty(
            (rows, self.weight.out_features),
            dtype=torch.bfloat16,
            device=x.device,
        )
        scratch = current_workspace_manager().get_simultaneous(
            *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
        )
        binding = block_fp8_linear.bind(
            plan,
            scratch=scratch,
            source=x,
            packed_weight=self.weight,
            output=out.view(rows, self.weight.out_features, 1),
        )
        retain_cuda_graph_capture_resource(scratch)
        block_fp8_linear.run(binding=binding)
        return out


class DSparkContextCudaGraphs:
    """Serial, draft-workspace-lane owner for auxiliary projection and KV prep.

    Only bounded decode capacities are captured. Live source tensors are copied,
    never retained by a graph; rejected rows keep their caller-provided PAD slots.
    Larger prefills continue through the eager hooks with their actual row count.
    """

    def __init__(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor,
        layer_group_idx: list[int] | None,
        max_decode_tokens: int,
    ) -> None:
        self.model = model
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mappings = slot_mappings
        self.layer_group_idx = layer_group_idx
        self.width = model.config.hidden_size
        self.num_aux = len(model.config.dspark_target_layer_ids)
        limit = min(
            max_decode_tokens,
            hidden_states.shape[0],
            vllm_config.compilation_config.max_cudagraph_capture_size,
        )
        # Powers of two bound padding to less than 2x, independent of live rows.
        capacities = []
        capacity = 1

        while capacity < limit:
            capacities.append(capacity)
            capacity *= 2
        if limit > 0:
            capacities.append(limit)
        self.aux = torch.zeros(
            (limit, self.width * self.num_aux),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        config = copy.copy(vllm_config)
        config.compilation_config = copy.copy(vllm_config.compilation_config)
        config.compilation_config.cudagraph_capture_sizes = capacities
        config.compilation_config.max_cudagraph_capture_size = limit
        self.manager = CudaGraphManager(
            config, hidden_states.device, CUDAGraphMode.FULL, decode_query_len=1
        )

    def _forward(self, capacity: int) -> None:
        main_x = self.model.combine_hidden_states(self.aux[:capacity])
        self.hidden_states[:capacity].copy_(main_x)
        slots = (
            self.slot_mappings[0, :capacity]
            if self.layer_group_idx is None
            else [self.slot_mappings[i, :capacity] for i in self.layer_group_idx]
        )
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:capacity], self.positions[:capacity], slots
        )

    def capture(self) -> None:
        self.positions.zero_()
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # Reserve all native scratch and resolve every planned capacity BEFORE
        # the first graph. Never unlock a serving workspace to make capture fit.
        for capacity in self.manager.compilation_config.cudagraph_capture_sizes:
            self._forward(capacity)

        def create_forward_fn(desc, warmup):
            return lambda mode: self._forward(desc.num_tokens)

        self.manager.capture(
            create_forward_fn, progress_bar_desc="Capturing DSpark context CUDA graphs"
        )

    def can_run(self, num_tokens: int) -> bool:
        desc = self.manager.dispatch(1, num_tokens, None, 0)
        return desc in self.manager.graphs

    def run(
        self,
        aux_hidden_states: list[torch.Tensor],
        num_tokens: int,
        *,
        context_kv_is_restored: bool = False,
    ) -> None:
        if context_kv_is_restored:
            return
        desc = self.manager.dispatch(1, num_tokens, None, 0)
        if desc not in self.manager.graphs:
            raise ValueError("DSpark context rows exceed captured decode capacities")
        if len(aux_hidden_states) != self.num_aux or any(
            x.ndim != 2 or x.shape[0] < num_tokens or x.shape[1] != self.width
            for x in aux_hidden_states
        ):
            raise ValueError("DSpark context auxiliary states have invalid dimensions")
        for i, source in enumerate(aux_hidden_states):
            self.aux[:num_tokens, i * self.width : (i + 1) * self.width].copy_(
                source[:num_tokens]
            )
        # Context input preparation initializes rejected rows, but not rows
        # beyond the target batch. Scrub that tail on EVERY replay, including
        # transitions from a larger batch to a smaller one.
        capacity = desc.num_tokens
        self.aux[num_tokens:capacity].zero_()
        self.positions[num_tokens:capacity].zero_()
        self.slot_mappings[:, num_tokens:capacity].fill_(PAD_SLOT_ID)
        self.manager.run_fullgraph(desc)

    def close(self) -> None:
        """Destroy context graphs before their prepared executions are released."""
        self.manager.reset_graphs()
        self.manager.graphs.clear()
        self.manager.graph_capture_resources.clear()


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    attn.insert_context_kv(kv, positions, slot_mapping)


def initialize_model(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    nn.Module.__init__(self)
    self._use_b12x = True
    assert vllm_config.speculative_config is not None
    config = vllm_config.speculative_config.draft_model_config.hf_config
    self.config = config
    self.hidden_size = config.hidden_size
    self.hc_mult = config.hc_mult
    self.hc_eps = config.hc_eps
    self.rms_norm_eps = config.rms_norm_eps
    self.num_hidden_layers = config.num_hidden_layers
    self.target_layer_ids = tuple(config.dspark_target_layer_ids)
    self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
    self.context_capacity = vllm_config.scheduler_config.max_num_batched_tokens
    self._context_kv_projections = []

    self.num_dspark_layers = (
        getattr(config, "n_mtp_layers", None)
        or getattr(config, "num_nextn_predict_layers", None)
        or 3
    )

    # Bound from the loaded target, as in K3DSparkModel. Do not create an
    # unused vocabulary allocation while loading the draft checkpoint.
    self.embed_tokens = None

    self.main_proj = ReplicatedLinear(
        config.hidden_size * len(self.target_layer_ids),
        config.hidden_size,
        bias=False,
        return_bias=False,
        quant_config=vllm_config.quant_config,
        prefix=maybe_prefix(prefix, "main_proj"),
    )
    self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.topk_indices_buffer = torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        config.index_topk,
        dtype=torch.int32,
    )

    current_vllm_config = get_current_vllm_config()
    self._l2pf_ready = False
    self.layers = nn.ModuleList(
        [
            DeepseekV4DecoderLayer(
                current_vllm_config,
                prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                topk_indices_buffer=self.topk_indices_buffer,
            )
            for i in range(self.num_dspark_layers)
        ]
    )

    # Heads: final norm, and the Markov + confidence heads.
    # Loaded from the "final" MTP layer weights (mtp.*) in the target
    # checkpoint. v4.1 has no learned hc_head: the hc copies are
    # collapsed with a pre-mix derived from the last layer's hc_ffn
    # projection at the end of forward().
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
    self.markov_head = DSparkMarkovHead(
        config.vocab_size,
        draft_vocab_size,
        config.dspark_markov_rank,
        prefix=maybe_prefix(prefix, "markov_head"),
        # An NVFP4 transition head fits the GB10 L2, so the sequential
        # draft positions after the first reread it from L2.
        lm_head_quantization="nvfp4" if envs.VLLM_DS41_MARKOV_NVFP4 else None,
    )
    self.confidence_head = None
    if getattr(config, "enable_confidence_head", True):
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
        )


@torch.inference_mode()
def precompute_and_store_context_kv(
    self,
    main_x: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    """Insert the sliding-window context KV for every draft layer.

    Mirrors the reference DSparkAttention: each layer derives its context KV
    from the SAME projected target hidden ``main_x``, via that layer's own
    ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
    layer's context slots.

    ``context_slot_mappings`` is either common to all layers or a per-layer
    list (each entry is the mapping for that layer's kv-cache group).
    ``None`` (or a ``None`` entry) runs the projection to reserve workspace
    but writes nothing (profiling).
    """
    for i, layer in enumerate(self.layers):
        slot_mapping = (
            context_slot_mappings
            if context_slot_mappings is None
            or isinstance(context_slot_mappings, torch.Tensor)
            else context_slot_mappings[i]
        )
        attn = layer.attn
        kv = attn.kv_norm(self._context_kv_projections[i](main_x))
        if slot_mapping is None:
            continue
        _insert_context_kv(attn, kv, context_positions, slot_mapping)


def forward_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    full_num_tokens = positions.shape[0]
    if self.use_sequence_parallel:
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            forward_context = get_forward_context()
            forward_context.is_padding = sp_padding_mask(
                forward_context.is_padding, inputs_embeds
            )
        inputs_embeds = sp_shard(inputs_embeds)
        input_ids = sp_shard(input_ids)
    # The first layer's post-load broadcast weights consume [T, H] directly.
    hidden_states = inputs_embeds

    residual = post_mix = res_mix = pre_mix = None
    if l2_prefetch.ENABLED and not self._l2pf_ready:
        self._l2pf_ready = build_l2_prefetch_plans(
            list(self.layers), self.config.num_hidden_layers
        )
    for layer in self.layers:
        hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
        )
    l2_prefetch.join()
    hidden_states = layer._b12x_mhc.post(hidden_states, residual, post_mix, res_mix)
    if self.use_sequence_parallel:
        hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
    # Collapse the hc copies with the pre-mix from the last layer's FFN
    # mixes — the mix the reference forward_head applies (hc_pre with the
    # last block's ffn pre-mix). Return the PRE-norm head hidden;
    # compute_logits applies self.norm.
    assert pre_mix is not None
    hidden_states = layer._b12x_mhc.collapse(hidden_states, pre_mix)
    return hidden_states


def initialize_causal_lm(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    nn.Module.__init__(self)
    self._use_b12x = True
    assert vllm_config.speculative_config is not None
    self.draft_model_config = vllm_config.speculative_config.draft_model_config
    self.config = self.draft_model_config.hf_config
    self.quant_config = vllm_config.quant_config
    self.pad_shared_expert = True
    self.linear_scale_name = _linear_scale_param_name(
        vllm_config, getattr(self.config, "expert_dtype", "fp4")
    )
    self.model = DSparkDeepseekV4Model(
        vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
    )
    # Bound by load_dspark_model after loading. An unused online-quantized
    # head otherwise remains on meta and cannot be natively precompiled.
    self.lm_head = None
    self.logits_processor = LogitsProcessor(self.config.vocab_size)
    if envs.VLLM_DS41_DRAFT_NVFP4_HEAD:
        # A drafter-owned NVFP4 vocabulary head reads half the bytes of
        # the shared MXFP8 target head; only draft acceptance depends on it.
        head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
            lm_head_quantization="nvfp4",
        )
        if head.runtime_lm_head_quantization == "nvfp4":
            self.lm_head = head
            self.has_own_lm_head = True
            self.checkpoint_weight_name_prefixes = ("mtp.", "head.")


def prepare_weights(self) -> None:
    self.logits_processor.prepare_b12x_vocab_projection(
        self.model.markov_head.markov_w2
    )
    for layer in self.model.layers:
        layer.attn.setup_wo_projection()
    self.model._context_kv_projections = [
        _ContextKVProjection(layer.attn, self.model.context_capacity)
        for layer in self.model.layers
    ]
    first_layer = self.model.layers[0]
    broadcast = (
        first_layer.hc_attn_fn.detach()
        .view(-1, first_layer.hc_mult, first_layer.hidden_size)
        .sum(dim=1)
    )
    if first_layer.hc_attn_fn_broadcast is None:
        first_layer.hc_attn_fn_broadcast = broadcast
    else:
        first_layer.hc_attn_fn_broadcast.copy_(broadcast)
    for layer in self.model.layers:
        mhc = getattr(layer, "_b12x_mhc", None)
        if mhc is not None:
            set_b12x_preparation_provider(layer, mhc)
            name = b12x_layer_prefix(layer)
            register_b12x_layer(name, layer)
            mhc.bind_layer_name(name)
