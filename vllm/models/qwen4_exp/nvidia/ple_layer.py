# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
)
from vllm import envs
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.weight_transfer import allocate_weights
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    get_b12x_scratch_buffers,
    set_b12x_preparation_provider,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from .b12x_ple import (
    B12xNGramEmbedding,
    _b12x_module,
    _register_ple_compilation_context,
    _resolve_ple_table_memory,
)
from .backend import uses_b12x
from .ngram_embedding import Qwen4ExpNGramEmbedding
from .ops.ple import ple_conv, ple_gate
from .ple_attn import PLEAttentionBackend, PLEAttentionMetadata


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(
            allocate_weights(torch.zeros, hidden_size, dtype=dtype)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._use_b12x = uses_b12x(vllm_config)
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_seqs = vllm_config.scheduler_config.max_num_seqs
        self.eps = float(config.rms_norm_eps)
        if self._use_b12x:
            if model_config.dtype != torch.bfloat16:
                raise TypeError("b12x PLE requires BF16 activations")
            if not is_conv_state_dim_first():
                raise RuntimeError("b12x PLE requires VLLM_SSM_CONV_STATE_LAYOUT=DS")
            if vllm_config.engram_config is not None and (
                vllm_config.engram_config.embedding_across_dp
                or vllm_config.engram_config.dp_shared_memory
            ):
                raise NotImplementedError(
                    "b12x PLE does not support cross-DP embedding storage"
                )
            self.ple_embedding = B12xNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                self.max_tokens,
                self.max_seqs,
                prefix,
                f"{prefix}.ple_embedding",
                model_config.dtype,
                _resolve_ple_table_memory(vllm_config.additional_config),
            )
        else:
            self.ple_embedding = Qwen4ExpNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                vllm_config.scheduler_config.max_num_batched_tokens,
                data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
                prefix=f"{prefix}.ple_embedding",
                quant_config=quant_config,
                params_dtype=model_config.dtype,
            )
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=None if self._use_b12x else quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = allocate_weights(
            nn.Conv1d,
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if self._use_b12x:
            self._init_b12x_buffers()
            _register_ple_compilation_context(compilation_config, prefix, self)
            self._preparation_prefix = prefix or f"qwen4_exp.ple.{self.layer_idx}"
            if not getattr(self, "b12x_preparation_suppressed", False):
                set_b12x_preparation_provider(self, self)
        else:
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

    def _dequantize_embeddings(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize PLE lookup output."""

        return self.ple_embedding.ngram_embedding.dequantize(
            embeddings,
            output_dtype,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(
        self,
    ) -> type[PLEAttentionBackend] | type[PleShortConvAttentionBackend]:
        return PLEAttentionBackend if self._use_b12x else PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]
        outer_residual = outer_residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                outer_residual=outer_residual,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                outer_residual_d, outer_residual_p = torch.split(
                    outer_residual,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                outer_residual_d = outer_residual_p = outer_residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    outer_residual=outer_residual_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.non_spec_query_start_loc
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            query_start_loc = query_start_loc[-num_prefills - 1 :] - num_decode_tokens
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                outer_residual=outer_residual_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                outer_residual=outer_residual,
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    # State routing consumes the current request metadata on every replay.
    @eager_break_during_capture
    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. Short convolution
        # is a no-op there, but preserve the outer residual addition.
        if attn_metadata is None:
            residual.add_(outer_residual)
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            residual.add_(outer_residual)
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            state_size = conv_state.size(-1)
            if state_size < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {state_size}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            outer_residual=outer_residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        token_count = hidden_states.shape[0]
        if input_ids.shape[0] != token_count:
            raise ValueError(
                "PLE input_ids and hidden states must have the same token count"
            )
        if self._use_b12x:
            embeddings = self.ple_embedding(
                input_ids,
                query_start_loc,
                ngram_context,
            )
        else:
            embeddings = self.ple_embedding(
                hidden_states, input_ids, query_start_loc, ngram_context
            )
            embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        if self._use_b12x:
            key = key.reshape(token_count, self.hc_count, self.hidden_size)
            residual = hidden_states.reshape(
                token_count, self.hc_count, self.hidden_size
            )
            if torch.compiler.is_compiling():
                torch.ops.vllm.qwen4_exp_b12x_ple(
                    residual,
                    key,
                    value,
                    query_start_loc,
                    self._out,
                    self.prefix,
                )
            else:
                self._run_ple(residual, key, value, query_start_loc)
            return hidden_states + self._out[:token_count].flatten(-2)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        self._short_conv(conv_input, gated_output, hidden_states)
        return gated_output

    def _init_b12x_buffers(self):
        dtype = self.model_config.dtype
        device = torch.device(current_platform.current_device())
        factory = dict(device=device, dtype=dtype)
        self.register_buffer(
            "_residual",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_key",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_value",
            torch.empty(self.max_tokens, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_out",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_state_slot_ids",
            torch.full(
                (self.max_seqs,), NULL_BLOCK_ID, dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "_state_is_fresh",
            torch.ones(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_accepted_tokens",
            torch.ones(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_request_is_prefill",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_seqs",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_tokens",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer("_scratch", None, persistent=False)
        self._state_caps = None
        self._state_plans: dict[int, object] = {}
        self.kv_cache = (torch.tensor([]),)

    def _make_caps(self, max_state_slots: int):
        api = _b12x_module("ple")
        return api.Caps(
            device=current_platform.current_device(),
            mode="mixed",
            max_tokens=self.max_tokens,
            max_seqs=self.max_seqs,
            max_state_slots=max_state_slots,
            max_speculative_tokens=self.num_spec_tokens,
            streams=self.hc_count,
            hidden_size=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            dilation=self.short_conv_dilation,
            dtype=self.model_config.dtype,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if not self._use_b12x:
            return
        conv_state = self.kv_cache[0]
        expected_tail = self.conv_state_len + self.num_spec_tokens
        if tuple(conv_state.shape[1:]) != (self.hc_hidden_size, expected_tail):
            raise RuntimeError(
                "unexpected PLE cache shape: expected "
                f"[slots,{self.hc_hidden_size},{expected_tail}], got "
                f"{tuple(conv_state.shape)}"
            )
        self._state_caps = self._make_caps(max_state_slots=conv_state.shape[0])
        self._state_plans = {}

    def _declare_state_plan(self, token_count: int):
        if self._state_caps is None:
            raise RuntimeError("PLE KV cache was not bound before preparation")
        return _b12x_module("ple").plan(
            replace(self._state_caps, max_tokens=token_count),
            invocation=self._state_invocation(token_count),
        )

    def _state_capacity_for(self, token_count: int) -> int:
        if not 0 <= token_count <= self.max_tokens:
            raise ValueError("PLE state token count exceeds capacity")
        return token_count if token_count in self._state_plans else self.max_tokens

    def _state_plan_for(self, token_count: int):
        """Resolve a live token count within the declared state capacity."""
        if self._state_caps is None:
            raise RuntimeError("PLE KV cache was not bound before inference")
        capacity = self._state_capacity_for(token_count)
        plan = self._state_plans.get(capacity)
        if plan is None:
            plan = self._declare_state_plan(capacity)
            self._state_plans[capacity] = plan
        return plan

    def _bind_ple(self, token_count: int):
        capacity = self._state_capacity_for(token_count)
        plan = self._state_plan_for(token_count)
        (scratch,) = get_b12x_scratch_buffers(plan)
        return _b12x_module("ple").bind(
            plan,
            scratch=scratch,
            residual=self._residual[:capacity],
            key=self._key[:capacity],
            value=self._value[:capacity],
            k_norm_weight=self.norm_key.weight,
            q_norm_weight=self.norm_query.weight,
            u_norm_weight=self.norm_conv.weight,
            conv_weight=self.conv1d.weight.squeeze(1),
            query_start_loc=self._query_start_loc,
            state_slot_ids=self._state_slot_ids,
            state_is_fresh=self._state_is_fresh,
            num_accepted_tokens=self._num_accepted_tokens,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            conv_state=self.kv_cache[0],
            out=self._out[:capacity],
            request_is_prefill=self._request_is_prefill,
        )

    def _state_request_name(self, token_count: int) -> str:
        return f"{self._preparation_prefix}.state.m{token_count}"

    def _state_invocation(self, token_count: int):
        api = _b12x_module("ple")
        return api.invocation_from_tensors(
            residual=self._residual[:token_count],
            key=self._key[:token_count],
            value=self._value[:token_count],
            k_norm_weight=self.norm_key.weight,
            q_norm_weight=self.norm_query.weight,
            u_norm_weight=self.norm_conv.weight,
            conv_weight=self.conv1d.weight.squeeze(1),
            query_start_loc=self._query_start_loc,
            state_slot_ids=self._state_slot_ids,
            state_is_fresh=self._state_is_fresh,
            num_accepted_tokens=self._num_accepted_tokens,
            request_is_prefill=self._request_is_prefill,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            conv_state=self.kv_cache[0],
            out=self._out[:token_count],
        )

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if layer is not self:
            raise ValueError("PLE state preparation owner mismatch")
        if self._state_caps is None:
            return ()
        requests = []
        plans: dict[int, object] = {}
        for token_count in sorted({self.max_tokens, *workload.fixed_token_counts}):
            if token_count > self.max_tokens:
                continue
            plan = self._declare_state_plan(token_count)
            plans[token_count] = plan
            requests.append(
                plan.request(
                    name=self._state_request_name(token_count),
                    prepare_call=self._state_prepare_call(token_count),
                    autotune=False,
                )
            )
        self._state_plans = plans
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="PLE state",
                key=self._preparation_prefix,
                requests=tuple(requests),
                stage="state",
                autotune=False,
            ),
        )

    def _state_prepare_call(self, token_count: int):
        def prepare(state):
            from b12x.preparation import PreparedCall

            # The prepare factory owns its scratch; the runtime binding in
            # _bind_ple draws from the workspace manager instead.
            scratch_spec = state.layout.scratch_specs()[0]
            scratch = torch.empty(
                scratch_spec.shape, dtype=scratch_spec.dtype, device=scratch_spec.device
            )
            # Exercise the actual mixed recurrent branch on one bounded slot.
            # The preparation lease owns a snapshot only of that touched region;
            # every reset/close restores it before production can observe it.
            state_slot = 0
            original_conv_state = self.kv_cache[0][state_slot].clone()
            # The call stages its own inputs in the layer's staging buffers, so
            # it snapshots and restores everything it overwrites: a plan can be
            # prepared on demand in the middle of a live forward pass.
            staging = (
                self._residual[:token_count],
                self._key[:token_count],
                self._value[:token_count],
                self._out[:token_count],
                self._query_start_loc,
                self._state_slot_ids,
                self._state_is_fresh,
                self._num_accepted_tokens,
                self._request_is_prefill,
                self._num_seqs,
                self._num_tokens,
            )
            snapshot = tuple(buffer.clone() for buffer in staging)
            binding = state.bind(
                scratch=scratch,
                residual=self._residual[:token_count],
                key=self._key[:token_count],
                value=self._value[:token_count],
                k_norm_weight=self.norm_key.weight,
                q_norm_weight=self.norm_query.weight,
                u_norm_weight=self.norm_conv.weight,
                conv_weight=self.conv1d.weight.squeeze(1),
                query_start_loc=self._query_start_loc,
                state_slot_ids=self._state_slot_ids,
                state_is_fresh=self._state_is_fresh,
                num_accepted_tokens=self._num_accepted_tokens,
                num_seqs=self._num_seqs,
                num_tokens=self._num_tokens,
                conv_state=self.kv_cache[0],
                out=self._out[:token_count],
                request_is_prefill=self._request_is_prefill,
            )

            def restore():
                self.kv_cache[0][state_slot].copy_(original_conv_state)
                for buffer, saved in zip(staging, snapshot):
                    buffer.copy_(saved)

            def reset():
                self.kv_cache[0][state_slot].copy_(original_conv_state)
                self._residual[:token_count].fill_(1)
                self._key[:token_count].fill_(1)
                self._value[:token_count].fill_(1)
                self._out[:token_count].zero_()
                self._query_start_loc.zero_()
                self._query_start_loc[1] = token_count
                self._state_slot_ids.fill_(NULL_BLOCK_ID)
                self._state_slot_ids[0] = state_slot
                self._state_is_fresh.fill_(True)
                self._num_accepted_tokens.fill_(1)
                self._request_is_prefill.zero_()
                self._num_seqs.fill_(1)
                self._num_tokens.fill_(token_count)

            checkpoint_offsets = torch.zeros_like(
                self._state_slot_ids, dtype=torch.int32
            )
            checkpoint_slots = torch.full_like(self._state_slot_ids, -1)

            def run():
                result = state.run(binding, eps=self.eps, token_count=token_count)
                if envs.VLLM_QWEN3_8_PREFILL_COALESCE:
                    # Capture can invoke export even when no request checkpoints.
                    # Prime its module with inactive slots before resolution freezes.
                    state.export_checkpoint(
                        binding, checkpoint_offsets, checkpoint_slots
                    )
                return result

            return PreparedCall(
                run=run,
                reset=reset,
                restore=restore,
                owners=(
                    scratch,
                    original_conv_state,
                    checkpoint_offsets,
                    checkpoint_slots,
                ),
            )

        return prepare

    def unbind_kv_cache(self) -> None:
        self._state_caps = None
        self._state_plans = {}
        super().unbind_kv_cache()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MambaSpec)
        if not envs.VLLM_QWEN3_8_PREFILL_COALESCE:
            return spec
        if (
            vllm_config.use_request_boundary_checkpoints
            or vllm_config.cache_config.mamba_cache_mode != "align"
        ):
            raise ValueError("Qwen PLE internal checkpoints require aligned caching")
        return replace(spec, num_prefill_checkpoint_blocks=1)

    def _prepare_metadata(
        self,
        metadata: PLEAttentionMetadata,
        query_start_loc: torch.Tensor,
        token_count: int,
    ) -> None:
        inputs = metadata.graph_inputs
        if inputs is None:
            raise RuntimeError("PLE metadata has no staged graph inputs")
        if inputs.max_seqs != self.max_seqs or token_count > self.max_tokens:
            raise ValueError(
                f"PLE capacity exceeded: tokens={token_count}/{self.max_tokens}, "
                f"requests={inputs.max_seqs}/{self.max_seqs}"
            )
        self._query_start_loc.copy_(inputs.query_start_loc)
        self._state_slot_ids.copy_(inputs.state_slot_ids)
        self._state_is_fresh.copy_(inputs.state_is_fresh)
        self._num_accepted_tokens.copy_(inputs.num_accepted_tokens)
        self._request_is_prefill.copy_(inputs.request_is_prefill)
        self._num_seqs.copy_(inputs.num_seqs)
        self._num_tokens.copy_(inputs.num_tokens)

    def _run_ple(
        self,
        residual: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        token_count = residual.shape[0]
        forward_context = get_forward_context()
        raw_metadata = forward_context.attn_metadata
        metadata = (
            raw_metadata.get(self.prefix) if isinstance(raw_metadata, dict) else None
        )
        if metadata is None:
            # Initial memory profiling deliberately runs without attention
            # metadata because the KV cache has not been admitted yet.
            self._out.zero_()
            self._out[:token_count].copy_(value[:, None, :].expand_as(residual))
            return
        if not isinstance(metadata, PLEAttentionMetadata):
            raise TypeError(
                f"expected PLEAttentionMetadata for {self.prefix}, got "
                f"{type(metadata).__name__}"
            )
        if self._state_caps is None:
            raise RuntimeError("PLE KV cache was not bound before inference")
        self._state_plan_for(token_count)
        self._residual[:token_count].copy_(residual)
        self._key[:token_count].copy_(key)
        self._value[:token_count].copy_(value)
        self._prepare_metadata(metadata, query_start_loc, token_count)
        binding = self._bind_ple(token_count)
        api = _b12x_module("ple")
        api.run_mixed(binding, eps=self.eps, token_count=token_count)
        if metadata.checkpoint_columns is not None:
            assert metadata.checkpoint_offsets is not None
            assert metadata.checkpoint_slots is not None
            api.export_checkpoint(
                binding,
                offsets=metadata.checkpoint_offsets,
                slots=metadata.checkpoint_slots,
            )


__all__ = [
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
