# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange
from torch import nn
from torch.nn.parameter import Parameter

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.triton_utils import tl, triton
from vllm.utils.b12x import get_b12x_gdn_decode, get_b12x_scratch_buffers
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.workspace import current_workspace_manager

from ...linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from ..mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.gather_initial_states import gather_initial_states

# Empirical lower bound for the KDA gate to avoid numerical underflow.
_KDA_GATE_LOGBOUND_MIN = -5.0


def is_flashkda_supported(
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    """Return whether FlashKDA supports the layer's prefill contract."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )


def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed bounded-gate KDA prefill into caller-owned buffers."""
    import vllm._flashkda_C  # noqa: F401

    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
        checkpoint_state,
        checkpoint_offsets.contiguous() if checkpoint_offsets is not None else None,
    )
    return out, final_state


@triton.jit
def _store_cache_checkpoints_kernel(
    x_ptr,
    conv_state_ptr,
    recurrent_checkpoint_ptr,
    recurrent_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    checkpoint_state_indices_ptr,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    checkpoint_stride_0: tl.constexpr,
    recurrent_state_stride_0: tl.constexpr,
    checkpoint_offset_stride: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    RECURRENT_ROW_SIZE: tl.constexpr,
    NULL_STATE_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Store FlashKDA recurrent and convolution state at an internal boundary."""
    seq_idx = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_idx_i64 = seq_idx.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx_i64)
    state_idx_i64 = state_idx.to(tl.int64)
    checkpoint_offset = tl.load(
        checkpoint_offsets_ptr + seq_idx_i64 * checkpoint_offset_stride
    )
    valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
    valid_conv = (
        (cols < WIDTH * STATE_LEN) & valid_checkpoint & (checkpoint_offset >= STATE_LEN)
    )
    width_idx = cols // STATE_LEN
    history_idx = cols % STATE_LEN
    checkpoint_end = tl.load(query_start_loc_ptr + seq_idx_i64) + checkpoint_offset
    token_idx = checkpoint_end.to(tl.int64) - STATE_LEN + history_idx.to(tl.int64)
    values = tl.load(
        x_ptr + token_idx * x_stride_0 + width_idx.to(tl.int64) * x_stride_1,
        mask=valid_conv,
    )
    tl.store(
        conv_state_ptr
        + state_idx_i64 * state_stride_0
        + width_idx.to(tl.int64) * state_stride_1
        + history_idx.to(tl.int64) * state_stride_2,
        values,
        mask=valid_conv,
    )

    valid_recurrent = (cols < RECURRENT_ROW_SIZE) & valid_checkpoint
    recurrent = tl.load(
        recurrent_checkpoint_ptr + seq_idx_i64 * checkpoint_stride_0 + cols_i64,
        mask=valid_recurrent,
    )
    tl.store(
        recurrent_state_ptr + state_idx_i64 * recurrent_state_stride_0 + cols_i64,
        recurrent,
        mask=valid_recurrent,
    )


def resolve_kda_prefill_backend(
    backend: str,
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> str:
    """Resolve the packed KDA prefill implementation for one server."""
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    supported = is_flashkda_supported(head_dim, dtype, lower_bound)
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, "
            "head_dim=128, and a bounded KDA gate."
        )
    if supported and backend != "triton":
        return "flashkda"
    return "triton"


def a_log_weight_loader(
    shard_axis: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load KDA A_log stored as either old 4D or current 1D weights."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if loaded_weight.dim() == 4:
            assert loaded_weight.shape[:2] == (1, 1), (
                f"Expected old A_log shape (1, 1, H, 1), got {loaded_weight.shape}"
            )
            assert loaded_weight.shape[-1] == 1, (
                f"Expected old A_log last dim to be 1, got {loaded_weight.shape}"
            )
            loaded_weight = loaded_weight.view(loaded_weight.shape[2])

        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)

    return loader


def _make_fused_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
) -> Callable[..., None]:
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)

    return weight_loader


class _KimiGDNMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with one output replicated across TP ranks.

    The replicated shard is represented as ``size * tp_size`` so the merged
    parameter reserves ``size`` local rows on every rank. Loading that shard
    from rank zero then gives every rank the complete checkpoint weight.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_id: int,
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_id = replicated_shard_id
        output_sizes = output_sizes.copy()
        output_sizes[replicated_shard_id] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    enable_b12x_kda_decode = False
    b12x_kda_null_state_index: int | None = None

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MambaSpec)
        return replace(
            spec,
            num_prefill_checkpoint_blocks=int(self.kda_prefill_backend == "flashkda"),
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)

        if self.use_full_rank_gate:
            # Keep f_a before the narrow beta shard, then pad each TP-local row
            # to select the aligned BF16 GEMM path. The padding also avoids an
            # Inductor correctness issue seen with the row-strided G view.
            qkvg_output_sizes = [self.projection_size] * 4
            in_proj_output_sizes = qkvg_output_sizes + [
                self.head_dim,
                self.num_heads,
            ]
            local_output_size = (
                4 * self.local_projection_size + self.head_dim + self.local_num_heads
            )
            self.in_proj_padding = -local_output_size % 16
            if self.in_proj_padding:
                in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        else:
            in_proj_output_sizes = [self.projection_size] * 3 + [
                self.num_heads,
                self.head_dim,
            ]
            self.in_proj_padding = 0
        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache let decode run a single conv update.
        # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_fused_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )
        self.use_safe_gate = self.gate_lower_bound is not None
        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        self.kda_prefill_backend = resolve_kda_prefill_backend(
            backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.gate_lower_bound,
        )
        self._flashkda_buffer_specs: (
            tuple[tuple[tuple[int, ...], torch.dtype], ...] | None
        ) = None
        if self.kda_prefill_backend == "flashkda":
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_sequences = vllm_config.scheduler_config.max_num_seqs
            heads, head_dim = self.local_num_heads, self.head_dim
            import vllm._flashkda_C  # noqa: F401

            workspace_size = torch.ops._flashkda_C.get_workspace_size(
                max_tokens,
                heads,
                max_sequences,
            )
            self._flashkda_buffer_specs = (
                ((1, max_tokens, heads, head_dim), self.model_config.dtype),
                (
                    (max_sequences, heads, head_dim, head_dim),
                    self.get_state_dtype()[1],
                ),
                (
                    (max_sequences, heads, head_dim, head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
            )
        if not self.use_full_rank_gate:
            self.g_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self._b12x_kda_api: Any | None = None
        self._b12x_kda_plan = None
        self._initialize_b12x_kda_decode(vllm_config)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _initialize_b12x_kda_decode(self, vllm_config: VllmConfig) -> None:
        if (
            not self.enable_b12x_kda_decode
            or self.gate_lower_bound is None
            or self.head_dim != 128
            or self.model_config.dtype != torch.bfloat16
            or self.get_state_dtype()[1] not in (torch.bfloat16, torch.float32)
            or not current_platform.is_cuda()
        ):
            return

        api = get_b12x_gdn_decode()
        device = torch.device(current_platform.current_device())
        if (
            api is None
            or not hasattr(api, "bind_kda")
            or not hasattr(api, "run_kda")
            or not api.is_supported(device)
        ):
            return

        max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        state_index_columns = max(1, self.num_spec + 1)
        if state_index_columns > 8:
            return
        max_tokens = max_seqs * state_index_columns

        self._b12x_kda_api = api
        self._b12x_kda_max_tokens = max_tokens
        self._b12x_kda_max_seqs = max_seqs
        self._b12x_kda_state_index_columns = state_index_columns

        self.register_buffer(
            "_b12x_kda_num_accepted_tokens",
            torch.ones(max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_b12x_kda_num_seqs",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_b12x_kda_num_tokens",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer("_b12x_kda_scratch", None, persistent=False)

    def _make_b12x_kda_plan(self, max_state_slots: int):
        api = self._b12x_kda_api
        if api is None:
            raise RuntimeError("b12x KDA decode was not initialized")
        return api.plan(
            api.Caps(
                device=current_platform.current_device(),
                max_tokens=self._b12x_kda_max_tokens,
                max_seqs=self._b12x_kda_max_seqs,
                max_state_slots=max_state_slots,
                key_heads=self.local_num_heads,
                value_heads=self.local_num_heads,
                key_head_dim=self.head_dim,
                value_head_dim=self.head_dim,
                state_index_columns=self._b12x_kda_state_index_columns,
                model_dtype=self.model_config.dtype,
                state_dtype=self.get_state_dtype()[1],
                gate_activation="sigmoid",
                qk_l2norm=True,
                null_state_index=self.b12x_kda_null_state_index,
                kda_metadata_validation="trusted",
            )
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        api = self._b12x_kda_api
        if api is None:
            return
        recurrent_state = self.kv_cache[1]
        plan = self._make_b12x_kda_plan(max_state_slots=recurrent_state.shape[0])
        (scratch,) = get_b12x_scratch_buffers(plan)
        self._b12x_kda_scratch = scratch
        self._b12x_kda_plan = plan

    def unbind_kv_cache(self) -> None:
        self._b12x_kda_plan = None
        self._b12x_kda_scratch = None
        super().unbind_kv_cache()

    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = mixed_qkv.shape[0]
        qkv = mixed_qkv.view(seq_len, 3, self.local_num_heads, self.head_dim)
        # Materialize all three row-strided inputs with one token-major to
        # QKV-major permutation. Each unbound tensor is then contiguous.
        qkv = qkv.permute(1, 0, 2, 3).contiguous().unsqueeze(1)
        return qkv.unbind(0)

    def _can_use_b12x_kda_decode(self, m: GDNAttentionMetadata) -> bool:
        if (
            self._b12x_kda_plan is None
            or self._b12x_kda_scratch is None
            or m.num_prefills != 0
            or (m.num_decodes == 0 and m.num_spec_decodes == 0)
        ):
            return False
        if m.spec_sequence_masks is None:
            return (
                m.num_spec_decodes == 0
                and m.non_spec_state_indices_tensor is not None
                and m.non_spec_query_start_loc is not None
            )
        return (
            m.num_decodes == 0
            and m.num_spec_decodes > 0
            and m.spec_state_indices_tensor is not None
            and m.spec_query_start_loc is not None
            and m.num_accepted_tokens is not None
        )

    def _run_b12x_kda_decode_post_conv(
        self,
        *,
        metadata: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None,
        num_requests: int,
    ) -> None:
        """Execute B12X KDA after the convolution projection.

        Args:
            metadata: Forward-context metadata used to share runtime-owned
                packed metadata tensors across compatible layers.
            mixed_qkv: Live packed query, key, and value projection.
            raw_g: Live unactivated forget gate.
            raw_beta: Live unactivated update gate.
            z: Live output gate.
            output: Caller-owned destination tensor.
            state_indices: Packed recurrent-state indices.
            query_start_loc: Packed request boundaries.
            num_accepted_tokens: Accepted speculative-token counts, or ``None``
                for one-token decode requests.
            num_requests: Number of packed requests.

        Raises:
            RuntimeError: If the KDA plan or cache is unavailable.
            ValueError: If the live batch exceeds the planned capacity.
        """
        api = self._b12x_kda_api
        plan = self._b12x_kda_plan
        scratch = self._b12x_kda_scratch
        if api is None or plan is None or scratch is None:
            raise RuntimeError("b12x KDA KV cache was not bound before inference")
        num_tokens = int(mixed_qkv.shape[0])
        state_columns = int(state_indices.shape[1])
        if (
            num_tokens > self._b12x_kda_max_tokens
            or num_requests > self._b12x_kda_max_seqs
            or state_columns > self._b12x_kda_state_index_columns
        ):
            raise ValueError(
                "b12x KDA capacity exceeded: "
                f"tokens={num_tokens}/{self._b12x_kda_max_tokens}, "
                f"requests={num_requests}/{self._b12x_kda_max_seqs}, "
                f"state_columns={state_columns}/"
                f"{self._b12x_kda_state_index_columns}"
            )

        forward_context = get_forward_context()
        cache = forward_context.additional_kwargs.setdefault(
            "b12x_kda_metadata_tensors", {}
        )
        cache_key = (
            id(metadata),
            num_tokens,
            num_requests,
            state_columns,
            plan.caps.max_state_slots,
        )
        bound_metadata = cache.get(cache_key)
        if bound_metadata is None:
            query_start_loc = query_start_loc[: num_requests + 1]
            state_indices = state_indices[:num_requests, :state_columns]
            if num_accepted_tokens is None:
                accepted_tokens = self._b12x_kda_num_accepted_tokens[:num_requests]
                accepted_tokens.fill_(1)
            else:
                accepted_tokens = num_accepted_tokens[:num_requests]
            self._b12x_kda_num_seqs.fill_(num_requests)
            self._b12x_kda_num_tokens.copy_(
                query_start_loc[num_requests : num_requests + 1]
            )
            bound_metadata = (
                query_start_loc,
                accepted_tokens,
                state_indices,
                self._b12x_kda_num_seqs,
                self._b12x_kda_num_tokens,
            )
            cache[cache_key] = bound_metadata
        (
            query_start_loc,
            accepted_tokens,
            state_indices,
            num_seqs,
            num_tokens_tensor,
        ) = bound_metadata

        binding = api.bind_kda(
            plan,
            scratch=scratch,
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            z=z,
            A_log=self.A_log,
            dt_bias=self.dt_bias.view(self.local_num_heads, self.head_dim),
            norm_weight=self.o_norm.weight,
            recurrent_state=self.kv_cache[1],
            query_start_loc=query_start_loc,
            num_accepted_tokens=accepted_tokens,
            state_indices=state_indices,
            num_seqs=num_seqs,
            num_tokens=num_tokens_tensor,
            output=output,
        )
        api.run_kda(
            binding,
            lower_bound=self.gate_lower_bound,
            eps=self.o_norm.eps,
            scale=self.head_dim**-0.5,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        # Optional model-installed callback (e.g. GLM-5.3 L2 weight prefetch of
        # o_proj while the small projections and the recurrence run).
        _hook = getattr(self, "_l2_prefetch_hook", None)
        if _hook is not None:
            _hook(hidden_states.shape[0])
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        else:
            mixed_qkv, beta, f_a = projected_qkvgfab.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        # Vendor-specific KDA kernels: AMD/ROCm and NVIDIA keep their own copies
        # under kimi_k3/{amd,nvidia}/ops so each can diverge independently.
        # These copies may have different signatures for Kimi-K3, but they agree
        # on the arguments used here.
        if TYPE_CHECKING:
            from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
                chunk_kda_with_fused_gate,
                fused_recurrent_kda,
                fused_recurrent_kda_packed_decode,
            )
        elif current_platform.is_rocm():
            from vllm.models.kimi_k3.amd.ops.third_party.kda import (  # type: ignore[assignment]
                chunk_kda_with_fused_gate,
                fused_recurrent_kda,
                fused_recurrent_kda_packed_decode,
            )
        else:
            from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
                chunk_kda_with_fused_gate,
                fused_recurrent_kda,
                fused_recurrent_kda_packed_decode,
            )

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs skip mamba-family metadata.
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        m = attn_metadata_narrowed
        prefill_checkpoint = m.prefill_checkpoint
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_sequence_masks = m.spec_sequence_masks
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        g2_actual = g2[:num_actual_tokens]
        use_b12x_kda = self._can_use_b12x_kda_decode(m)

        constant_caches = self.kv_cache

        conv_state, recurrent_state = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Split tokens into the multi-query spec-decode part and the remaining
        # (prefill / plain decode) part.
        if spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
                g2_spec, g2_ns = g2_actual, None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
                g2_spec = g2_ns = None
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta
            g2_spec, g2_ns = None, g2_actual

        # ---------- spec-decode multi-query path ----------
        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)

            # Sibling beta and, for full-rank gates, output-gate views remain
            # live, so write the convolution output separately.
            spec_conv_out = torch.empty(
                mixed_qkv_spec.shape,
                dtype=mixed_qkv_spec.dtype,
                device=mixed_qkv_spec.device,
            )
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            if use_b12x_kda:
                assert g2_spec is not None
                core_attn_out_spec = core_attn_out[:, : mixed_qkv_spec.shape[0]]
                self._run_b12x_kda_decode_post_conv(
                    metadata=m,
                    mixed_qkv=mixed_qkv_spec,
                    raw_g=g1_spec[0],
                    raw_beta=beta_spec[0],
                    z=g2_spec,
                    output=core_attn_out_spec[0],
                    state_indices=spec_state_indices_tensor,
                    query_start_loc=spec_cu_seqlens,
                    num_accepted_tokens=num_accepted_tokens,
                    num_requests=m.num_spec_decodes,
                )
            else:
                q_spec, k_spec, v_spec = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
                )
                # Spec-only batches write directly into core_attn_out.
                spec_out = (
                    core_attn_out[:, : q_spec.shape[1]]
                    if m.num_prefills == 0 and m.num_decodes == 0
                    else None
                )
                core_attn_out_spec, _ = fused_recurrent_kda(
                    q=q_spec,
                    k=k_spec,
                    v=v_spec,
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    cu_seqlens=spec_cu_seqlens,
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    out=spec_out,
                )

        # ---------- non-spec path (prefill or plain decode) ----------
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )
                prefill_mixed_qkv = mixed_qkv_ns

                # Packed prefill conv would require copying V solely to make
                # it dense for KDA. Separate calls accept the strided inputs
                # and produce dense Q/K/V without that extra traffic.
                # TODO: Use packed conv once every KDA prefill backend accepts
                # row-strided Q/K/V directly.
                def _prefill_conv(
                    x: torch.Tensor,
                    state: torch.Tensor,
                    weight: torch.Tensor,
                ) -> torch.Tensor:
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )

                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None

                # Mixed non-spec batches are decode-first. Peel the length-one
                # decodes off because the chunk kernel only consumes the
                # prefill-tail metadata produced by the GDN builder.
                core_attn_out_decode = None
                split_non_spec = spec_sequence_masks is None and m.num_decodes > 0
                if split_non_spec:
                    assert non_spec_query_start_loc is not None
                    nd_tok = m.num_decode_tokens
                    prefill_mixed_qkv = prefill_mixed_qkv[nd_tok:]
                    core_attn_out_decode, _ = fused_recurrent_kda(
                        q=q_ns[:, :nd_tok],
                        k=k_ns[:, :nd_tok],
                        v=v_ns[:, :nd_tok],
                        raw_g=g1_ns[:, :nd_tok],
                        raw_beta=beta_ns[:, :nd_tok],
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=recurrent_state,
                        cu_seqlens=non_spec_query_start_loc[: m.num_decodes + 1],
                        ssm_state_indices=non_spec_state_indices_tensor[
                            : m.num_decodes
                        ],
                    )
                    q_ns = q_ns[:, nd_tok:]
                    k_ns = k_ns[:, nd_tok:]
                    v_ns = v_ns[:, nd_tok:]
                    g1_ns = g1_ns[:, nd_tok:]
                    beta_ns = beta_ns[:, nd_tok:]
                    prefill_query_start_loc = m.prefill_query_start_loc
                    prefill_state_indices = m.prefill_state_indices
                    prefill_has_initial_state = m.prefill_has_initial_state
                    assert prefill_query_start_loc is not None
                    assert prefill_state_indices is not None
                    assert prefill_has_initial_state is not None
                else:
                    prefill_query_start_loc = non_spec_query_start_loc
                    prefill_state_indices = non_spec_state_indices_tensor
                    prefill_has_initial_state = has_initial_state

                initial_state = gather_initial_states(
                    recurrent_state,
                    prefill_state_indices,
                    prefill_has_initial_state,
                )
                if self.kda_prefill_backend == "flashkda":
                    assert self.gate_lower_bound is not None
                    assert self._flashkda_buffer_specs is not None
                    assert prefill_query_start_loc is not None
                    workspace_out, final_state, checkpoint_state, workspace = (
                        current_workspace_manager().get_simultaneous(
                            *self._flashkda_buffer_specs
                        )
                    )
                    flashkda_out = workspace_out[:, : q_ns.shape[1]]
                    if prefill_checkpoint is not None:
                        assert prefill_query_start_loc is not None
                        num_sequences = initial_state.shape[0]
                        assert prefill_checkpoint.checkpoint_offsets.shape == (
                            num_sequences,
                        )
                        final_state = final_state[:num_sequences]
                        checkpoint_state = checkpoint_state[:num_sequences]
                        _flashkda_prefill(
                            q=q_ns,
                            k=k_ns,
                            v=v_ns,
                            g=g1_ns,
                            beta=beta_ns,
                            A_log=self.A_log,
                            dt_bias=self.dt_bias,
                            lower_bound=self.gate_lower_bound,
                            initial_state=initial_state,
                            cu_seqlens=prefill_query_start_loc,
                            out=flashkda_out,
                            final_state=final_state,
                            workspace=workspace,
                            checkpoint_state=checkpoint_state,
                            checkpoint_offsets=(prefill_checkpoint.checkpoint_offsets),
                        )
                        core_attn_out_non_spec = flashkda_out
                        last_recurrent_state = final_state

                        state_len = conv_state.shape[-1]
                        width = prefill_mixed_qkv.shape[-1]
                        recurrent_row_size = checkpoint_state[0].numel()
                        store_block_size = 256
                        _store_cache_checkpoints_kernel[
                            (
                                prefill_checkpoint.checkpoint_offsets.numel(),
                                triton.cdiv(
                                    max(width * state_len, recurrent_row_size),
                                    store_block_size,
                                ),
                            )
                        ](
                            prefill_mixed_qkv,
                            conv_state,
                            checkpoint_state,
                            recurrent_state,
                            prefill_query_start_loc,
                            prefill_checkpoint.checkpoint_offsets,
                            prefill_checkpoint.state_indices,
                            prefill_mixed_qkv.stride(0),
                            prefill_mixed_qkv.stride(1),
                            conv_state.stride(0),
                            conv_state.stride(1),
                            conv_state.stride(2),
                            checkpoint_state.stride(0),
                            recurrent_state.stride(0),
                            prefill_checkpoint.checkpoint_offsets.stride(0),
                            state_len,
                            width,
                            recurrent_row_size,
                            NULL_BLOCK_ID,
                            store_block_size,
                        )
                    else:
                        (
                            core_attn_out_non_spec,
                            last_recurrent_state,
                        ) = _flashkda_prefill(
                            q=q_ns,
                            k=k_ns,
                            v=v_ns,
                            g=g1_ns,
                            beta=beta_ns,
                            A_log=self.A_log,
                            dt_bias=self.dt_bias,
                            lower_bound=self.gate_lower_bound,
                            initial_state=initial_state,
                            cu_seqlens=prefill_query_start_loc,
                            out=flashkda_out,
                            final_state=final_state[: initial_state.shape[0]],
                            workspace=workspace,
                        )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=prefill_query_start_loc,
                        chunk_indices=m.chunk_indices,
                        chunk_offsets=m.chunk_offsets,
                    )
                # Init cache
                recurrent_state[prefill_state_indices] = last_recurrent_state

                if split_non_spec:
                    core_attn_out_non_spec = torch.cat(
                        [core_attn_out_decode, core_attn_out_non_spec], dim=1
                    )

            else:
                # pure-decode non-spec batch
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                # Sibling beta and, for full-rank gates, output-gate views
                # remain live, so write the conv output separately.
                packed_conv_out = torch.empty(
                    mixed_qkv_ns.shape,
                    dtype=mixed_qkv_ns.dtype,
                    device=mixed_qkv_ns.device,
                )
                mixed_qkv_ns = causal_conv1d_update(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                    out=packed_conv_out,
                )
                if use_b12x_kda:
                    assert non_spec_query_start_loc is not None
                    assert g2_ns is not None
                    core_attn_out_non_spec = core_attn_out[:, : mixed_qkv_ns.shape[0]]
                    self._run_b12x_kda_decode_post_conv(
                        metadata=m,
                        mixed_qkv=mixed_qkv_ns,
                        raw_g=g1_ns[0],
                        raw_beta=beta_ns[0],
                        z=g2_ns,
                        output=core_attn_out_non_spec[0],
                        state_indices=non_spec_state_indices_tensor[
                            : m.num_decodes, None
                        ],
                        query_start_loc=non_spec_query_start_loc,
                        num_accepted_tokens=None,
                        num_requests=m.num_decodes,
                    )
                else:
                    core_attn_out_non_spec, _ = fused_recurrent_kda_packed_decode(
                        mixed_qkv=mixed_qkv_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=recurrent_state,
                        state_indices=decode_conv_indices,
                    )

        # ---------- merge spec and non-spec outputs ----------
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            # Mixed batches require indexed placement in the original order.
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0, :num_actual_tokens]
        elif core_attn_out_non_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_spec is not None
        if not use_b12x_kda:
            core_attn_out.copy_(self.o_norm(core_attn_out, g2))
