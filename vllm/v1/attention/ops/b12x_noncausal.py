# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded paged-KV gather for prepared b12x noncausal sliding attention."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.b12x import B12xPreparationUnit, PreparationResourceUnavailableError


@triton.jit
def _gather_window(
    K,
    V,
    Pages,
    Lengths,
    QueryStarts,
    KScale,
    VScale,
    PackedK,
    PackedV,
    CuQ,
    CuK,
    IS_FP8: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_REQS: tl.constexpr,
    MAX_REQS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    requests = tl.arange(0, triton.next_power_of_2(MAX_REQS))
    starts = tl.load(QueryStarts + requests, requests < NUM_REQS, 0)
    ends = tl.load(QueryStarts + requests + 1, requests < NUM_REQS, 0)
    lengths = tl.load(Lengths + requests, requests < NUM_REQS, 0)
    kept = tl.where(ends > starts, tl.minimum(lengths, WINDOW_LEFT + ends - starts), 0)
    offset = tl.sum(tl.where(requests < req, kept, 0), 0)
    length = tl.sum(tl.where(requests == req, lengths, 0), 0)
    count = tl.sum(tl.where(requests == req, kept, 0), 0)
    if tile == 0:
        q_end = tl.load(QueryStarts + tl.minimum(req + 1, NUM_REQS))
        tl.store(CuQ + req + 1, q_end)
        tl.store(CuK + req + 1, offset + count)
        if req == 0:
            tl.store(CuQ, 0)
            tl.store(CuK, 0)
    rows = tile * ROWS + tl.arange(0, ROWS)
    logical = length - count + rows
    page = tl.load(
        Pages + req * TABLE_STRIDE + logical // PAGE_SIZE,
        (req < NUM_REQS) & (rows < count),
        0,
    ).to(tl.int64)
    channels = tl.arange(0, CHANNELS)
    head = channels // DIM
    dim = channels % DIM
    k_offset = (
        page[:, None] * K_STRIDES[0]
        + (logical[:, None] % PAGE_SIZE).to(tl.int64) * K_STRIDES[1]
        + head[None, :] * K_STRIDES[2]
        + dim[None, :] * K_STRIDES[3]
    )
    v_offset = (
        page[:, None] * V_STRIDES[0]
        + (logical[:, None] % PAGE_SIZE).to(tl.int64) * V_STRIDES[1]
        + head[None, :] * V_STRIDES[2]
        + dim[None, :] * V_STRIDES[3]
    )
    mask = (rows[:, None] < count) & (channels[None, :] < KV_HEADS * DIM)
    k = tl.load(K + k_offset, mask, 0.0)
    v = tl.load(V + v_offset, mask, 0.0)
    if IS_FP8:
        # Descale before the BF16 store, matching the packed attention inputs.
        k = k.to(tl.float32) * tl.load(KScale)
        v = v.to(tl.float32) * tl.load(VScale)
    destination = (offset + rows[:, None]).to(tl.int64) * KV_HEADS * DIM + channels[
        None, :
    ]
    tl.store(PackedK + destination, k, mask)
    tl.store(PackedV + destination, v, mask)


class B12xNoncausalAttention:
    """Gather the visible window, then run native b12x varlen attention.

    Storage and attention plans depend only on configured request/query limits.
    Device prefix sums pack variable-length windows without host synchronization.
    """

    def __init__(self, impl):
        if impl.window_left < 0 or impl._verify_q_per_req <= 0:
            raise NotImplementedError(
                "B12X noncausal attention requires sliding-window speculation."
            )
        if (
            impl.kv_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or impl.head_size != impl.output_head_size
        ):
            raise NotImplementedError(
                "B12X noncausal attention requires BF16 or FP8 E4M3 KV "
                "and equal QK/V dimensions."
            )
        self.impl = impl
        self.max_q = impl._verify_q_per_req
        self.max_k = impl.window_left + self.max_q
        self.batch = impl._max_num_seqs
        self.plan = None
        self.binding = None

    def preparation_units(self, layer, key_cache, value_cache, table_width):
        from b12x.attention import varlen
        from b12x.preparation import PreparedCall

        impl = self.impl
        if self.plan is None:

            def empty(*shape, dtype=torch.bfloat16):
                return torch.empty(shape, dtype=dtype, device=impl.device)

            self.q = empty(self.batch * self.max_q, impl.num_heads, impl.head_size)
            self.k = empty(self.batch * self.max_k, impl.num_kv_heads, impl.head_size)
            self.v = empty(*self.k.shape)
            self.cu_q = empty(self.batch + 1, dtype=torch.int32)
            self.cu_k = empty(self.batch + 1, dtype=torch.int32)
            self.plan = varlen.plan(
                self.q,
                self.k,
                self.v,
                self.cu_q,
                self.cu_k,
                max_seqlen_q=self.max_q,
                max_seqlen_k=self.max_k,
                causal=False,
                window_size=(impl.window_left, -1),
                attention_sink_bias=impl.sinks,
            )

        pages = torch.zeros(
            (self.batch, table_width), dtype=torch.int32, device=impl.device
        )
        lengths = torch.full(
            (self.batch,),
            min(self.max_k, table_width * key_cache.shape[1]),
            dtype=torch.int32,
            device=impl.device,
        )
        starts = (
            torch.arange(self.batch + 1, dtype=torch.int32, device=impl.device)
            * self.max_q
        )
        # Prime packing separately from the native attention program's audit.
        for batch in range(1, self.batch + 1):
            self.gather(
                key_cache,
                value_cache,
                pages[:batch],
                lengths[:batch],
                starts[: batch + 1],
                layer=layer,
            )

        plan = self.plan
        assert plan is not None

        def make_call(state, *, serving=False):
            q, k, v = self.q, self.k, self.v
            cu_q, cu_k = self.cu_q, self.cu_k
            (spec,) = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            binding = state.bind(
                plan=plan,
                scratch=scratch,
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=self.max_q,
                max_seqlen_k=self.max_k,
                causal=False,
                window_size=(impl.window_left, -1),
                attention_sink_bias=impl.sinks,
            )
            if serving:
                self.binding = binding
            kv_starts = (
                torch.arange(self.batch + 1, dtype=torch.int32, device=impl.device)
                * self.max_k
            )

            def produce():
                q.normal_(std=0.25)
                k.normal_(std=0.25)
                v.normal_(std=0.25)
                cu_q.copy_(starts)
                cu_k.copy_(kv_starts)

            return PreparedCall(
                run=lambda: state.run(binding),
                produce=produce,
                owners=(q, k, v, cu_q, cu_k, scratch),
            )

        return (
            B12xPreparationUnit(
                name="b12x noncausal sliding attention",
                key=(id(layer), tuple(key_cache.stride())),
                requests=(
                    self.plan.request(
                        name=f"attention.noncausal.{id(layer):x}",
                        prepare_call=lambda state: make_call(state, serving=True),
                        benchmark_call=make_call,
                    ),
                ),
                stage="state",
            ),
        )

    def _descales(self, layer, key_cache, value_cache):
        if (
            key_cache.dtype != self.impl.kv_torch_dtype
            or value_cache.dtype != self.impl.kv_torch_dtype
        ):
            raise TypeError("Noncausal KV must use the backend's typed cache views.")
        if self.impl.kv_torch_dtype == torch.bfloat16:
            return None, None
        # The cache writer uses tensor scales. Do not reinterpret a vector as
        # per-head scales: _prepare_fp8_descales treats vectors as per-request.
        for name in ("_k_scale", "_v_scale"):
            scale = getattr(layer, name, None)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"Noncausal FP8 {name} must be a tensor.")
            if scale.ndim > 1 or scale.numel() != 1:
                raise ValueError(f"Noncausal FP8 {name} must be scalar or shape (1,).")
        return self.impl._prepare_fp8_descales(layer, 1, key_cache.device)

    def gather(self, key_cache, value_cache, pages, lengths, starts, *, layer):
        k_scale, v_scale = self._descales(layer, key_cache, value_cache)
        _gather_window[(self.batch, triton.cdiv(self.max_k, 16))](
            key_cache,
            value_cache,
            pages,
            lengths,
            starts,
            k_scale,
            v_scale,
            self.k,
            self.v,
            self.cu_q,
            self.cu_k,
            self.impl.kv_torch_dtype == torch.float8_e4m3fn,
            key_cache.stride(),
            value_cache.stride(),
            pages.stride(0),
            key_cache.shape[1],
            lengths.shape[0],
            self.batch,
            self.impl.window_left,
            self.impl.num_kv_heads,
            self.impl.head_size,
            16,
            triton.next_power_of_2(self.impl.num_kv_heads * self.impl.head_size),
        )

    def forward(self, query, output, key_cache, value_cache, metadata, *, layer):
        from b12x.attention import varlen

        if self.binding is None:
            raise PreparationResourceUnavailableError(
                "B12X noncausal attention has not been prepared."
            )
        if (
            metadata.max_query_len > self.max_q
            or metadata.seq_lens.shape[0] > self.batch
            or query.shape[0] > self.q.shape[0]
        ):
            raise ValueError("B12X noncausal batch exceeds its prepared capacity.")
        self.q[: query.shape[0]].copy_(query)
        self.gather(
            key_cache,
            value_cache,
            metadata.block_table,
            metadata.seq_lens,
            metadata.query_start_loc,
            layer=layer,
        )
        attended, _ = varlen.run(binding=self.binding)
        output.copy_(attended[: output.shape[0]])
        return output
