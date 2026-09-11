# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native ratio-one/two compression with request-owned partial-state rings."""

import torch
from b12x.attention import mla_compress
from b12x.gemm import bf16_gemv
from torch import nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4_1.b12x_layers import B12xLinearMethod, B12xRMSNorm
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import CircularBufferSpec


@triton.jit(do_not_specialize=["stride", "table_stride"])
def _load_partial_state(
    Pool,
    Table,
    Starts,
    Positions,
    Values,
    Gates,
    Tags,
    LocalIds,
    Counts,
    stride,
    table_stride,
    PAGE: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 512)
    nr = tl.load(Counts + 1)
    nt = tl.load(Counts)
    first = tl.load(Starts + r)
    end = tl.load(Starts + r + 1)
    pos = tl.load(Positions + r)
    live = (r < nr) & (first >= 0) & (first < end) & (end <= nt)
    live = live & (pos >= 0)
    predecessor = pos - 1
    needed = live & (pos > 0) & (pos % 2 == 1)
    page = tl.load(Table + r * table_stride, needed, other=0).to(tl.int64)
    valid = needed & (page > 0)
    base = Pool + page * stride + (predecessor % PAGE) * 1024
    value = tl.load(base + col, valid, other=0)
    gate = tl.load(base + 512 + col, valid, other=0)
    tl.store(Values + r * 512 + col, value)
    tl.store(Gates + r * 512 + col, gate)
    tl.store(Tags + r, tl.where(valid, predecessor, -1))
    tl.store(LocalIds + r, tl.where(live, r, -1))


@triton.jit(do_not_specialize=["stride"])
def _save_partial_states(
    Pool, Slots, Values, Gates, Counts, stride, PAGE: tl.constexpr
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 512)
    nt = tl.load(Counts)
    slot = tl.load(Slots + token).to(tl.int64)
    valid = (token < nt) & (slot >= PAGE)
    base = Pool + (slot // PAGE) * stride + (slot % PAGE) * 1024
    value = tl.load(Values + token * 512 + col, valid, other=0)
    gate = tl.load(Gates + token * 512 + col, valid, other=0)
    tl.store(base + col, value, valid)
    tl.store(base + 512 + col, gate, valid)


class CompressorBackend(DeepseekV41B12xBackend):
    @staticmethod
    def get_name():
        return "B12X_DSV41_COMPRESSOR"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]


class CompressorStateCache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        spec = config.speculative_config
        draft_tokens = spec.num_speculative_tokens if spec is not None else 0
        # Drafts, the bonus token, and the previous row must coexist until
        # rejection selects the accepted boundary, as in upstream V4.1.
        self.block_size = max(8, 1 << (draft_tokens + 1).bit_length())
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        return CircularBufferSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            head_size_v=0,
            dtype=torch.float32,
        )

    def get_attn_backend(self):
        return CompressorBackend

    def forward(self):
        raise RuntimeError("compression state is consumed by its owning compressor")


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config,
        compress_ratio,
        hidden_size,
        head_dim,
        rotate=False,
        prefix="",
        k_cache_prefix="",
    ):
        super().__init__()
        if compress_ratio not in (1, 2) or head_dim != 512:
            raise ValueError("V4.1 compression requires ratio1/2 and head512")
        self.prefix, self.compress_ratio = prefix, compress_ratio
        self.capacity = vllm_config.scheduler_config.max_num_batched_tokens
        self.requests = vllm_config.scheduler_config.max_num_seqs
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [512] * compress_ratio,
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
            params_dtype=torch.bfloat16,
        )
        self.fused_wkv_wgate.quant_method = B12xLinearMethod()
        self.fused_wkv_wgate.out_dtype = (
            torch.float32 if compress_ratio == 2 else torch.bfloat16
        )
        self.norm = B12xRMSNorm(512, vllm_config.model_config.hf_config.rms_norm_eps)
        self.norm.weight = nn.Parameter(
            torch.ones(512, dtype=torch.float32), requires_grad=False
        )
        self.state_cache = (
            CompressorStateCache(vllm_config, f"{prefix}.state_cache")
            if compress_ratio == 2
            else None
        )
        self._plan = None

    def prepare(self, device):
        if self._plan is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compressor must be warmed before graph capture")
        t, r = self.capacity, self.requests

        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self._values = alloc(
            (t, 512), torch.float32 if self.compress_ratio == 2 else torch.bfloat16
        )
        self._latent = alloc((t, 512), torch.bfloat16)
        self._emitted = alloc((t,), torch.bool)
        self._slots = alloc((t,), torch.int64)
        self._gates = self._pending_values = self._pending_gates = (
            self._pending_tags
        ) = None
        if self.compress_ratio == 2:
            self._gates = alloc((t, 512), torch.float32)
            self._pending_values = alloc((r, 512), torch.float32)
            self._pending_gates = alloc((r, 512), torch.float32)
            self._pending_tags = alloc((r,), torch.int64)
            self._local_ids = alloc((r,), torch.int64)
        self._row_ids = torch.arange(r, dtype=torch.int64, device=device)
        self._plan = mla_compress.plan(
            mla_compress.Caps(
                device=device,
                max_tokens=t,
                max_requests=r,
                max_states=r,
                ratio=self.compress_ratio,
            )
        )

    def forward(self, hidden_states, metadata, state_metadata=None):
        self.prepare(hidden_states.device)
        rows = hidden_states.shape[0]
        bf16_gemv.mm(
            hidden_states,
            self.fused_wkv_wgate.weight[:512],
            out=self._values[:rows],
            output_dtype=self._values.dtype,
        )
        if self.compress_ratio == 2:
            bf16_gemv.mm(
                hidden_states,
                self.fused_wkv_wgate.weight[512:],
                out=self._gates[:rows],
                output_dtype=torch.float32,
            )
            if state_metadata is None:
                raise RuntimeError("ratio2 compression requires request-state metadata")
            pool = self.state_cache.kv_cache
            _load_partial_state[(self.requests,)](
                pool,
                state_metadata.block_table,
                state_metadata.query_start_loc,
                state_metadata.request_positions,
                self._pending_values,
                self._pending_gates,
                self._pending_tags,
                self._local_ids,
                state_metadata.live_counts,
                pool.stride(0),
                state_metadata.block_table.stride(0),
                state_metadata.block_size,
            )
            _save_partial_states[(self.capacity,)](
                pool,
                state_metadata.slot_mapping,
                self._values,
                self._gates,
                state_metadata.live_counts,
                pool.stride(0),
                state_metadata.block_size,
            )
            ids = self._local_ids
        else:
            ids = self._row_ids
        binding = mla_compress.bind(
            self._plan,
            values=self._values,
            gates=self._gates,
            weight=self.norm.weight,
            query_start_loc=metadata.query_start_loc,
            positions=metadata.request_positions,
            state_ids=ids,
            destination_slots=metadata.slot_mapping,
            live_counts=metadata.live_counts,
            out=self._latent,
            emitted=self._emitted,
            emitted_slots=self._slots,
            pending_values=self._pending_values,
            pending_gates=self._pending_gates,
            pending_position=self._pending_tags,
        )
        mla_compress.run(binding)
        return self._latent[:rows], self._slots[:rows]
