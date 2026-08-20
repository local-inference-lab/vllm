# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 dense MLA draft model for DSpark speculative decoding."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce_in_place,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_COMPACT_ROPE_PROTECTED_IDS: set[int] = set()
_STREAMED_AUX_MIN_TOKENS = 1024


def _load_b12x_vocab_parallel_argmax() -> Any | None:
    """Return B12X's exact vocabulary reducer when its API is complete."""
    try:
        from b12x.comm.pcie import VocabParallelArgmax
    except ImportError:
        return None
    if not callable(getattr(VocabParallelArgmax, "from_exchange_group", None)):
        return None
    if not callable(getattr(VocabParallelArgmax, "fused_add_argmax", None)):
        return None
    return VocabParallelArgmax


@contextmanager
def protect_k3_compact_rope_sources(model: nn.Module) -> Iterator[None]:
    """Prevent a draft from releasing position tables owned by ``model``."""
    previous = _COMPACT_ROPE_PROTECTED_IDS.copy()
    _COMPACT_ROPE_PROTECTED_IDS.update(
        id(module) for module in model.modules() if hasattr(module, "cos_sin_cache")
    )
    try:
        yield
    finally:
        _COMPACT_ROPE_PROTECTED_IDS.clear()
        _COMPACT_ROPE_PROTECTED_IDS.update(previous)


def _duplicate_context_kv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load each layer's KV projection into the cross-layer linear."""
    for name, weight in weights:
        yield name, weight
        layer_prefix, marker, param_name = name.partition(
            ".self_attn.kv_a_proj_with_mqa."
        )
        if not marker:
            continue
        layer_idx_str = layer_prefix.rsplit(".", 1)[-1]
        if not layer_idx_str.isdecimal():
            continue
        layer_idx = int(layer_idx_str)
        if layer_idx >= num_layers:
            continue
        fused_weight = weight.detach()
        fused_weight.shard_id = layer_idx
        yield f"context_kv_proj.{param_name}", fused_weight


def _fill_compact_rope_cache(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    freqs_workspace: torch.Tensor,
    cache_workspace: torch.Tensor,
    *,
    mscale: float,
) -> torch.Tensor:
    """Materialize RoPE rows consumed by one Kimi-K3 draft forward."""
    num_positions = int(positions.shape[0])
    if num_positions > int(freqs_workspace.shape[0]):
        raise ValueError(
            "Kimi-K3 DSpark compact RoPE workspace is too small: "
            f"positions={num_positions}, capacity={freqs_workspace.shape[0]}."
        )
    freqs = freqs_workspace[:num_positions]
    cache = cache_workspace[:num_positions]
    half_dim = int(inv_freq.shape[0])
    torch.mul(positions[:, None], inv_freq[None, :], out=freqs)
    torch.cos(freqs, out=cache[:, :half_dim])
    torch.sin(freqs, out=cache[:, half_dim:])
    if mscale != 1.0:
        cache.mul_(mscale)
    return cache


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        quant_config = get_draft_quant_config(vllm_config)
        self.self_attn = MultiHeadLatentAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id + layer_idx}.self_attn"
            ),
            use_rope=True,
            non_causal_multi_token_decode=True,
        )
        # Both row-parallel outputs stay un-reduced; their all-reduces are fused
        # into the RMSNorm that follows via fused_allreduce_rms_norm.
        self.self_attn.o_proj.reduce_results = False
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos_sin_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rope_cos_sin_cache=rope_cos_sin_cache,
        )
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        # The MLP output is reduced by the next layer's input_layernorm (or by
        # the model's final_norm).
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
        self.context_proj_sharded = (
            tp_size > 1 and self.config.hidden_size % tp_size == 0
        )
        context_input_size = (
            self.config.target_hidden_size * self.config.num_target_layers
        )
        context_prefix = maybe_prefix(prefix, "context_proj")
        if self.context_proj_sharded:
            # Target auxiliary states are identical across TP ranks. Each rank
            # retains and evaluates only its output rows; the gathered result
            # preserves the draft hidden-state layout exactly. Keep this one
            # projection in BF16: large-prefill streaming reads contiguous
            # input-column slices before the complete target state exists.
            self.context_proj = ColumnParallelLinear(
                context_input_size,
                self.config.hidden_size,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=None,
                prefix=context_prefix,
            )
        else:
            self.context_proj = ReplicatedLinear(
                context_input_size,
                self.config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=self.quant_config,
                prefix=context_prefix,
            )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix,
                f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj",
            ),
            disable_tp=True,
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._streamed_aux_layer_ids = tuple(
            int(layer_id) + 1
            for layer_id in getattr(self.config, "target_layer_ids", ())
        )
        self._streamed_aux_scratch: torch.Tensor | None = None
        self._streamed_aux_tokens = 0
        self._streamed_aux_index = 0
        self._context_local_width = (
            int(self.config.hidden_size) // tp_size
            if self.context_proj_sharded
            else int(self.config.hidden_size)
        )
        self._context_local_start = (
            int(getattr(self.context_proj, "tp_rank", 0)) * self._context_local_width
        )
        model_dtype = getattr(
            getattr(vllm_config, "model_config", None),
            "dtype",
            torch.get_default_dtype(),
        )
        if self.context_proj_sharded:
            self.register_buffer(
                "_streamed_context_states",
                torch.empty(
                    self._max_num_context_tokens,
                    self._context_local_width,
                    dtype=model_dtype,
                ),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_streamed_context_states",
                torch.empty(0, dtype=model_dtype),
                persistent=False,
            )
        self._compact_rope_enabled = bool(envs.VLLM_DSPARK_COMPACT_ROPE)
        if self._compact_rope_enabled:
            self._init_compact_rope()

    def _init_compact_rope(self) -> None:
        rotary_modules = [layer.self_attn.rotary_emb for layer in self.layers]
        if not rotary_modules or any(rotary is None for rotary in rotary_modules):
            raise RuntimeError(
                "Kimi-K3 DSpark compact RoPE requires rotary draft layers."
            )
        rotary = rotary_modules[0]
        assert rotary is not None
        if any(candidate is not rotary for candidate in rotary_modules[1:]):
            raise RuntimeError(
                "Kimi-K3 DSpark compact RoPE requires every draft layer to "
                "share one immutable rotary embedding."
            )
        if not hasattr(rotary, "scaling_factor"):
            raise TypeError(
                "Kimi-K3 DSpark compact RoPE requires a YaRN rotary embedding, "
                f"got {type(rotary).__name__}."
            )

        full_cache = rotary.cos_sin_cache
        if full_cache.dtype != torch.float32 or full_cache.ndim != 2:
            raise TypeError(
                "Kimi-K3 DSpark compact RoPE requires a 2-D fp32 source cache, "
                f"got shape={tuple(full_cache.shape)}, dtype={full_cache.dtype}."
            )
        with torch.device(full_cache.device):
            inv_freq = rotary._compute_inv_freq(rotary.scaling_factor)  # noqa: SLF001
        inv_freq = inv_freq.to(dtype=torch.float32)
        half_dim = int(inv_freq.shape[0])
        if int(full_cache.shape[1]) != 2 * half_dim:
            raise ValueError(
                "Kimi-K3 DSpark compact RoPE frequency width does not match "
                f"its source table: inv_freq={half_dim}, "
                f"cache={full_cache.shape[1]}."
            )

        self.register_buffer("_compact_rope_inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "_compact_rope_freqs",
            torch.empty(
                (self._max_num_context_tokens, half_dim),
                dtype=torch.float32,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_compact_rope_cache",
            torch.empty(
                (self._max_num_context_tokens, 2 * half_dim),
                dtype=torch.float32,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_compact_rope_positions",
            torch.arange(
                self._max_num_context_tokens,
                dtype=torch.int64,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self._compact_rope_mscale = float(rotary.mscale)

        if id(rotary) in _COMPACT_ROPE_PROTECTED_IDS:
            logger.warning_once(
                "Kimi-K3 DSpark compact RoPE retains a position table shared "
                "with the target model."
            )
            return

        released_bytes = full_cache.numel() * full_cache.element_size()
        rotary.cos_sin_cache = torch.empty(
            (0, 2 * half_dim),
            dtype=torch.float32,
            device=full_cache.device,
        )
        logger.info_once(
            "Kimi-K3 DSpark compact RoPE released %.2f MiB/rank and "
            "materializes at most %d fp32 rows per forward.",
            released_bytes / (1024**2),
            self._max_num_context_tokens,
        )

    def _get_rope_inputs(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = self.layers[0].self_attn.rotary_emb
        assert rotary is not None
        if not self._compact_rope_enabled:
            return positions, rotary.cos_sin_cache
        cache = _fill_compact_rope_cache(
            positions,
            self._compact_rope_inv_freq,
            self._compact_rope_freqs,
            self._compact_rope_cache,
            mscale=self._compact_rope_mscale,
        )
        return self._compact_rope_positions[: positions.shape[0]], cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    def bind_auxiliary_stream_scratch(self, scratch: torch.Tensor) -> None:
        """Bind caller-owned storage used to form one target auxiliary state."""
        expected_width = int(self.config.target_hidden_size)
        if scratch.ndim != 2 or scratch.shape[1] != expected_width:
            raise ValueError(
                "Kimi-K3 DSpark auxiliary scratch must have shape "
                f"[tokens, {expected_width}], got {tuple(scratch.shape)}."
            )
        if scratch.shape[0] < self._max_num_context_tokens:
            raise ValueError(
                "Kimi-K3 DSpark auxiliary scratch has insufficient token capacity: "
                f"capacity={scratch.shape[0]}, required={self._max_num_context_tokens}."
            )
        self._streamed_aux_scratch = scratch

    def can_stream_auxiliary_states(
        self,
        layer_ids: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> bool:
        """Return whether a large target forward can use streamed projection."""
        if not self.context_proj_sharded or self._streamed_aux_scratch is None:
            return False
        if hidden_states.ndim != 2 or hidden_states.shape[0] < _STREAMED_AUX_MIN_TOKENS:
            return False
        if tuple(layer_ids) != self._streamed_aux_layer_ids:
            return False
        if hidden_states.shape[1] != self.config.target_hidden_size:
            return False
        if hidden_states.dtype != self.context_proj.weight.dtype:
            return False
        if hidden_states.device != self.context_proj.weight.device:
            return False
        if self._streamed_aux_scratch.dtype != hidden_states.dtype:
            return False
        if self._streamed_aux_scratch.device != hidden_states.device:
            return False
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        context_kv_weight = getattr(self.context_kv_proj, "weight", None)
        return context_kv_weight is not None and context_kv_weight.ndim == 2

    def begin_auxiliary_stream(self, hidden_states: torch.Tensor) -> None:
        """Start one ordered sequence of target auxiliary-state projections."""
        if not self.can_stream_auxiliary_states(
            self._streamed_aux_layer_ids, hidden_states
        ):
            raise RuntimeError(
                "Kimi-K3 DSpark auxiliary streaming was started for an "
                "unsupported tensor geometry."
            )
        self._streamed_aux_tokens = int(hidden_states.shape[0])
        self._streamed_aux_index = 0

    def accumulate_auxiliary_state(
        self,
        primary: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        """Project one target state into its TP-local draft-hidden rows."""
        index = self._streamed_aux_index
        if index >= len(self._streamed_aux_layer_ids):
            raise RuntimeError("Kimi-K3 DSpark received too many auxiliary states.")
        num_tokens = self._streamed_aux_tokens
        if tuple(primary.shape) != (num_tokens, self.config.target_hidden_size):
            raise ValueError(
                "Kimi-K3 DSpark auxiliary state shape changed during projection: "
                f"got={tuple(primary.shape)}, tokens={num_tokens}, "
                f"width={self.config.target_hidden_size}."
            )
        assert self._streamed_aux_scratch is not None
        scratch = self._streamed_aux_scratch[:num_tokens]
        if residual is None:
            scratch.copy_(primary)
        else:
            torch.add(primary, residual, out=scratch)

        input_width = int(self.config.target_hidden_size)
        weight = self.context_proj.weight[
            :, index * input_width : (index + 1) * input_width
        ]
        output = self._streamed_context_states[:num_tokens]
        if index == 0:
            torch.mm(scratch, weight.t(), out=output)
        else:
            torch.addmm(
                output,
                scratch,
                weight.t(),
                beta=1.0,
                alpha=1.0,
                out=output,
            )
        self._streamed_aux_index = index + 1

    def finish_auxiliary_stream(self) -> torch.Tensor:
        """Normalize and return the TP-local projected target context."""
        expected = len(self._streamed_aux_layer_ids)
        if self._streamed_aux_index != expected:
            raise RuntimeError(
                "Kimi-K3 DSpark auxiliary stream ended with an incomplete "
                f"projection: received={self._streamed_aux_index}, expected={expected}."
            )
        output = self._streamed_context_states[: self._streamed_aux_tokens]
        squared_norm = torch.linalg.vector_norm(
            output,
            ord=2,
            dim=-1,
            keepdim=True,
            dtype=torch.float32,
        ).square_()
        tensor_model_parallel_all_reduce_in_place(squared_norm)
        squared_norm.div_(self.config.hidden_size).add_(
            self.context_norm.variance_epsilon
        )
        squared_norm.rsqrt_()
        output.mul_(squared_norm.to(dtype=output.dtype))

        output.mul_(
            self.context_norm.weight[
                self._context_local_start : self._context_local_start
                + self._context_local_width
            ]
        )
        return output

    def is_streamed_context_states(self, states: list[torch.Tensor]) -> bool:
        """Return whether ``states`` is the completed TP-local stream output."""
        if not self.context_proj_sharded or len(states) != 1:
            return False
        candidate = states[0]
        expected = self._streamed_context_states[: self._streamed_aux_tokens]
        return (
            candidate.shape == expected.shape
            and candidate.dtype == expected.dtype
            and candidate.device == expected.device
            and candidate.data_ptr() == expected.data_ptr()
        )

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache."""
        if not hasattr(self, "_num_context_layers"):
            self._build_fused_context_kv_metadata()
        self._precompute_fused_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _build_fused_context_kv_metadata(self) -> None:
        """Build cross-layer metadata after checkpoint loading."""
        attentions = [layer.self_attn for layer in self.layers]
        assert attentions
        attn0 = attentions[0]
        assert attn0.q_lora_rank is not None
        kv_width = attn0.kv_lora_rank + attn0.qk_rope_head_dim
        for attn in attentions:
            assert attn.q_lora_rank is not None
            assert (
                attn.q_lora_rank == attn0.q_lora_rank
                and attn.kv_lora_rank == attn0.kv_lora_rank
                and attn.qk_rope_head_dim == attn0.qk_rope_head_dim
                and attn.kv_a_layernorm.variance_epsilon
                == attn0.kv_a_layernorm.variance_epsilon
            ), "All MLA DSpark layers must share their latent KV geometry."
        self._context_kv_norm_weights = torch.stack(
            [attn.kv_a_layernorm.weight.detach() for attn in attentions], dim=0
        ).contiguous()
        self._num_context_layers = len(attentions)
        self._context_kv_width = kv_width
        self._context_kv_lora_rank = attn0.kv_lora_rank
        self._context_rope_dim = attn0.qk_rope_head_dim
        self._context_rms_norm_eps = attn0.kv_a_layernorm.variance_epsilon

    def _precompute_fused_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        if (
            self.context_proj_sharded
            and context_states.shape[-1] == self._context_local_width
        ):
            self._precompute_streamed_context_kv(
                context_states, context_positions, context_slot_mapping
            )
            return

        num_ctx = context_states.shape[0]
        num_layers = self._num_context_layers

        # One KV-only GEMM replaces five full Q+KV GEMMs. For K3 this projects
        # 5*576 rows rather than 5*2112 rows (72.7% fewer A-projection FLOPs).
        all_kv = self.context_kv_proj(context_states)
        all_kv = all_kv.view(num_ctx, num_layers, self._context_kv_width)
        all_kv_c = all_kv[..., : self._context_kv_lora_rank]
        all_k_pe = all_kv[..., self._context_kv_lora_rank :]

        # Layer-major layout lets the 2-D RMSNorm weights select a distinct row
        # for each draft layer in one grouped kernel.
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )

        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        (repeated_positions,) = current_workspace_manager().get_simultaneous(
            ((num_layers * self._max_num_context_tokens,), torch.int64),
        )
        repeated_positions = repeated_positions[: num_layers * num_ctx]
        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(context_positions)
        repeated_positions.view(num_layers, num_ctx).copy_(rope_positions)
        # Keep the single-tensor context RoPE on vLLM's optimized CUDA op;
        # DeepSeek YaRN's FlashInfer wrapper assumes a non-null key tensor.
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rope_cos_sin_cache,
            rotary_emb.is_neox_style,
        )
        all_k_pe = all_k_pe_flat.view(num_layers, num_ctx, 1, self._context_rope_dim)

        if context_slot_mapping is None:
            return

        cache_layers = [layer.self_attn for layer in self.layers]
        if (
            not is_quantized_kv_cache(cache_layers[0].kv_cache_dtype)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            # Grouped context KV insert only supports unquantized (bf16) KV cache
            # and assumes that all layers share the same block layout.

            if isinstance(context_slot_mapping, (list, tuple)):
                per_layer_slot_mappings = [
                    s for s in context_slot_mapping if s is not None
                ]
                if len({s.data_ptr() for s in per_layer_slot_mappings}) == 1:
                    # All rows alias to the same slot mapping.
                    slot_mapping = (
                        per_layer_slot_mappings[0].unsqueeze(0).expand(num_layers, -1)
                    )
                else:
                    slot_mapping = torch.stack(per_layer_slot_mappings, dim=0)
            else:
                # Broadcast the single shared context_slot_mapping tensor.
                slot_mapping = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)

            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe.squeeze(2),
                self._get_context_kv_cache_ptrs(cache_layers),
                slot_mapping,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
            )
            return

        for layer_idx, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                all_kv_c_normed[layer_idx],
                all_k_pe[layer_idx],
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _precompute_streamed_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        """Project TP-local context one draft layer at a time."""
        num_ctx = int(context_states.shape[0])
        full_weight = self.context_kv_proj.weight
        if full_weight.ndim != 2:
            raise RuntimeError(
                "Streamed Kimi-K3 DSpark context KV requires an unquantized "
                "two-dimensional fused projection weight."
            )

        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(context_positions)
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None

        for layer_idx, layer in enumerate(self.layers):
            row_start = layer_idx * self._context_kv_width
            weight = full_weight[
                row_start : row_start + self._context_kv_width,
                self._context_local_start : self._context_local_start
                + self._context_local_width,
            ]
            layer_kv = F.linear(context_states, weight)
            tensor_model_parallel_all_reduce_in_place(layer_kv)

            kv_c = layer_kv[:, : self._context_kv_lora_rank].contiguous()
            ops.rms_norm(
                kv_c,
                kv_c,
                self._context_kv_norm_weights[layer_idx],
                self._context_rms_norm_eps,
            )
            k_pe = layer_kv[:, self._context_kv_lora_rank :].contiguous()
            k_pe = k_pe.view(num_ctx, 1, self._context_rope_dim)
            ops.rotary_embedding(
                rope_positions,
                k_pe,
                None,
                rotary_emb.head_size,
                rope_cos_sin_cache,
                rotary_emb.is_neox_style,
            )

            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                del layer_kv, kv_c, k_pe
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                kv_c,
                k_pe,
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )
            del layer_kv, kv_c, k_pe

    def _has_uniform_block_layout(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> bool:
        if not hasattr(self, "_layers_share_kv_block_layout"):
            ref_cache = cache_layers[0].kv_cache
            self._layers_share_kv_block_layout = all(
                cl.kv_cache.size(1) == ref_cache.size(1)
                and cl.kv_cache.stride(0) == ref_cache.stride(0)
                and cl.kv_cache.stride(1) == ref_cache.stride(1)
                for cl in cache_layers
            )
        return self._layers_share_kv_block_layout

    def _get_context_kv_cache_ptrs(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> torch.Tensor:
        # The per-layer KV cache base pointers are stable after allocation, so
        # build the pointer array once and return it on every call.
        if not hasattr(self, "_context_cache_ptrs"):
            ref_cache = cache_layers[0].kv_cache
            cache_ptrs = torch.tensor(
                [cl.kv_cache.data_ptr() for cl in cache_layers],
                dtype=torch.int64,
                device=ref_cache.device,
            )
            self._context_cache_ptrs = cache_ptrs
        return self._context_cache_ptrs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(positions)
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=rope_positions,
                hidden_states=hidden_states,
                residual=residual,
                rope_cos_sin_cache=rope_cos_sin_cache,
            )
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states, residual, self.final_norm
        )
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Assigned by load_dspark_model from the target. Keeping no placeholder
        # avoids a transient full-vocabulary allocation for this 163k-vocab model.
        self.lm_head: nn.Module | None = None
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        argmax_requested = bool(envs.VLLM_KIMI_K3_B12X_DSPARK_ARGMAX)
        self._b12x_dspark_argmax_cls = (
            _load_b12x_vocab_parallel_argmax() if argmax_requested else None
        )
        self._b12x_dspark_argmax_enabled = self._b12x_dspark_argmax_cls is not None
        if argmax_requested and not self._b12x_dspark_argmax_enabled:
            logger.warning_once(
                "B12X does not provide the complete VocabParallelArgmax API; "
                "Kimi-K3 DSpark uses the exact vocabulary all-gather fallback."
            )
        self._b12x_dspark_argmax_max_batch = min(
            vllm_config.scheduler_config.max_num_seqs, 8
        )
        self._b12x_dspark_argmax_runtime: Any = None
        self._b12x_dspark_argmax_output: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def bind_target_auxiliary_stream(
        self,
        target_model: nn.Module,
        scratch: torch.Tensor,
    ) -> None:
        """Connect the target's large-prefill states to the draft projector."""
        get_language_model = getattr(target_model, "get_language_model", None)
        target_language_model = (
            get_language_model() if callable(get_language_model) else target_model
        )
        target_inner = getattr(target_language_model, "model", None)
        setter = getattr(target_inner, "set_aux_hidden_state_projector", None)
        if not callable(setter):
            raise TypeError(
                "Kimi-K3 DSpark auxiliary streaming requires a target model "
                "that accepts a memory-bounded auxiliary-state projector."
            )
        self.model.bind_auxiliary_stream_scratch(scratch)
        setter(self.model)

    def is_streamed_context_states(self, states: list[torch.Tensor]) -> bool:
        """Return whether target auxiliary states are already TP-local."""
        return self.model.is_streamed_context_states(states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def supports_local_draft_argmax(self) -> bool:
        """Return whether rank-local target and Markov logits can be combined."""
        markov_head = self.model.markov_head
        if not markov_head.shard_across_tp:
            return False
        if not isinstance(self.lm_head, VocabParallelEmbedding):
            return False
        markov_w2 = markov_head.markov_w2
        if not isinstance(markov_w2, VocabParallelEmbedding):
            return False
        layout_fields = (
            "tp_size",
            "tp_rank",
            "num_embeddings",
            "org_vocab_size",
            "num_embeddings_padded",
            "num_embeddings_per_partition",
        )
        if any(
            getattr(self.lm_head, field) != getattr(markov_w2, field)
            for field in layout_fields
        ):
            return False
        if self.lm_head.shard_indices != markov_w2.shard_indices:
            return False
        # Rank-major base-vocabulary padding preserves B12X's global token
        # mapping. Added-vocabulary shards interleave padding and require a
        # different local-index mapping.
        if self.lm_head.num_embeddings != self.lm_head.org_vocab_size:
            return False
        if self.logits_processor.soft_cap is not None:
            return False
        return self.logits_processor.scale > 0.0

    def _mask_local_draft_padding(self, logits: torch.Tensor) -> None:
        """Exclude this rank's padded vocabulary tail from token selection."""
        assert isinstance(self.lm_head, VocabParallelEmbedding)
        valid_rows = self.lm_head.shard_indices.num_org_elements
        if valid_rows < logits.shape[-1]:
            logits[..., valid_rows:].fill_(float("-inf"))

    def compute_local_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project target hidden states into this rank's vocabulary shard."""
        assert isinstance(self.lm_head, VocabParallelEmbedding)
        return self.logits_processor._apply_head(self.lm_head, hidden_states, None)

    def compute_local_markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """Project a Markov embedding into the matching vocabulary shard."""
        return self.model.markov_head.local_bias(markov_embed, self.logits_processor)

    def gather_local_draft_logits(
        self,
        base_logits: torch.Tensor,
        markov_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Add matching shards and gather the exact complete distribution."""
        logits = tensor_model_parallel_all_gather(base_logits + markov_bias, dim=-1)
        return logits[..., : self.logits_processor.org_vocab_size]

    def _get_b12x_dspark_argmax(self, base_logits: torch.Tensor) -> Any | None:
        lm_head = self.lm_head
        assert isinstance(lm_head, VocabParallelEmbedding)
        runtime = self._b12x_dspark_argmax_runtime
        if runtime is not None:
            return runtime

        argmax_cls = self._b12x_dspark_argmax_cls
        if argmax_cls is None:
            self._b12x_dspark_argmax_enabled = False
            return None

        runtime = argmax_cls.from_exchange_group(
            exchange_group=get_tp_group().cpu_group,
            device=base_logits.device,
            local_vocab_size=base_logits.shape[-1],
            max_batch_size=self._b12x_dspark_argmax_max_batch,
        )
        self._b12x_dspark_argmax_output = torch.empty(
            self._b12x_dspark_argmax_max_batch,
            dtype=torch.int64,
            device=base_logits.device,
        )
        self._b12x_dspark_argmax_runtime = runtime
        logger.info_once(
            "Kimi-K3 DSpark uses B12X TP%d fused BF16 add and global argmax "
            "for up to %d requests.",
            lm_head.tp_size,
            self._b12x_dspark_argmax_max_batch,
        )
        return runtime

    def sample_local_draft_logits(
        self,
        base_logits: torch.Tensor,
        markov_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact greedy tokens through B12X or the full-logit fallback."""
        assert isinstance(self.lm_head, VocabParallelEmbedding)
        self._mask_local_draft_padding(base_logits)
        self._mask_local_draft_padding(markov_bias)
        batch_size = int(base_logits.shape[0])
        if (
            self._b12x_dspark_argmax_enabled
            and self.lm_head.tp_size in (8, 12, 16)
            and base_logits.dtype == torch.bfloat16
            and markov_bias.dtype == torch.bfloat16
            and 0 < batch_size <= self._b12x_dspark_argmax_max_batch
        ):
            runtime = self._get_b12x_dspark_argmax(base_logits)
            if runtime is not None:
                assert self._b12x_dspark_argmax_output is not None
                return runtime.fused_add_argmax(
                    base_logits,
                    markov_bias,
                    out=self._b12x_dspark_argmax_output[:batch_size],
                )
        return self.gather_local_draft_logits(base_logits, markov_bias).argmax(dim=-1)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # confidence_head is training-only. The frozen target embedding and LM
        # head are shared after this draft-specific checkpoint is loaded.
        loader = AutoWeightsLoader(
            self,
            skip_substrs=list(self.checkpoint_skip_substrs),
        )
        # read: 1. all weights. 2. context kv weights
        weights = _duplicate_context_kv_weights(weights, len(self.model.layers))
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model._build_fused_context_kv_metadata()
        return loaded_weights
