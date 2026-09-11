# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Full/Reindex/Reuse topology with native b12x compute only."""

import re

import torch
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention import dsa_indexer
from b12x.attention.compressed_sparse_mla.preparation import rotate
from b12x.attention.compressed_sparse_mla.weight_scale import (
    scale_index_weights,
)
from b12x.gemm import bf16_gemv
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.models.deepseek_v4_1.b12x_layers import (
    B12xFP8LinearMethod,
    B12xLinearMethod,
    B12xRMSNorm,
)
from vllm.models.deepseek_v4_1.ced import ced_decoder_start
from vllm.models.deepseek_v4_1.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend, _chunk
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)


def _source(prefix, layer):
    return re.sub(r"(layers\.)\d+", rf"\g<1>{layer}", prefix)


def _native_linear(layer):
    method = layer.quant_method
    if isinstance(method, (B12xFP8LinearMethod, B12xLinearMethod)):
        return
    if type(method) is UnquantizedLinearMethod:
        layer.quant_method = B12xLinearMethod()
        return
    raise ValueError("V4.1 attention only supports native b12x linear methods")


def _scratch(plan):
    return current_workspace_manager().get_simultaneous(
        *((s.shape, s.dtype) for s in plan.scratch_specs())
    )


def _rotated(x, positions, cos_sin_cache, **kwargs):
    # This result can span nested GEMMs, which reuse the workspace arena.
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rotate(x, positions, cos_sin_cache, out=out, **kwargs)
    retain_cuda_graph_capture_resource((x, out))
    return out


@triton.jit(do_not_specialize=["offset", "stride", "width"])
def _pages(
    Reqs, Table, Out, offset, stride, width, WIDTH: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    req = tl.load(Reqs + offset + row)
    page = tl.load(
        Table + req.to(tl.int64) * stride + col, (req >= 0) & (col < width), other=-1
    )
    # vLLM reserves physical block zero; b12x page tables use -1 for holes.
    page = tl.where(page > 0, page, -1)
    tl.store(Out + row * WIDTH + col, page, col < WIDTH)


class _Cache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix, *, kind, ratio=1, window=0, draft=False):
        super().__init__()
        self.prefix, self.kind, self.ratio, self.window = prefix, kind, ratio, window
        self.draft = draft
        self.block_size = 32 if kind == "swa" else config.cache_config.block_size
        self.kv_cache = torch.tensor([])
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        # Keep allocator page stride: padding belongs to the allocator, not the ABI.
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        common = dict(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=512 if self.kind == "swa" else 68,
            state_content_bytes=528 if self.kind == "swa" else 68,
            dtype=torch.uint8,
            tokens_per_state=self.ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
        )
        if self.kind == "swa":
            boundary = ced_decoder_start(config.model_config.hf_config)
            layer_id = int(re.search(r"layers\.(\d+)", self.prefix).group(1))
            bounded_replay = self.draft or (
                boundary is not None and layer_id >= boundary
            )
            return SlidingWindowMLASpec(
                **common,
                sliding_window=self.window,
                extra_retained_tokens=int(self.draft),
                prefix_cache_enabled=not bounded_replay,
                prefill_replay_window=128 if bounded_replay else 0,
            )
        return MLAAttentionSpec(**common)

    def get_attn_backend(self):
        return DeepseekV41B12xBackend

    def forward(self):
        raise RuntimeError("V4.1 caches are consumed by their owning attention layer")


@triton.jit
def _unpack_wo_a(Weight, Scale, Out, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // K, i % K
    value = tl.load(Weight + i, i < N * K, other=0.0).to(tl.float32)
    exponent = tl.load(
        Scale + (row // 32) * tl.cdiv(K, 32) + col // 32, i < N * K, other=0
    ).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    tl.store(Out + i, value * scale, i < N * K)


class _GroupedLinearMethod(LinearMethodBase):
    """Reference WO-A: unpack weights once; never quantize inverse-RoPE output."""

    def __init__(self, original, groups):
        if type(original) is UnquantizedLinearMethod:
            original = B12xLinearMethod()
        if not isinstance(original, (B12xFP8LinearMethod, B12xLinearMethod)):
            raise ValueError("V4.1 grouped output requires native b12x weights")
        self.original, self.groups = original, groups

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype == torch.float8_e4m3fn:
            scales = layer.weight_scale_inv
            if scales.dtype != torch.float8_e8m0fnu:
                raise ValueError("V4.1 WO-A requires UE8M0 checkpoint scales")
            dense = torch.empty(
                layer.weight.shape, dtype=torch.bfloat16, device=layer.weight.device
            )
            _unpack_wo_a[(triton.cdiv(layer.weight.numel(), 256),)](
                layer.weight,
                scales.view(torch.uint8),
                dense,
                layer.weight.shape[0],
                layer.weight.shape[1],
                256,
            )
            layer.weight = nn.Parameter(dense, requires_grad=False)
            del layer.weight_scale_inv
        width = layer.weight.shape[0] // self.groups
        self.weights = []
        for group in range(self.groups):
            weight = layer.weight[group * width : (group + 1) * width]
            bf16_gemv.precompile(weight)
            self.weights.append(weight)

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 WO-A must be bias free")
        rows = x.shape[0]
        result = torch.empty(
            (rows, layer.weight.shape[0]), dtype=x.dtype, device=x.device
        )
        width = layer.weight.shape[0] // self.groups
        for group in range(self.groups):
            bf16_gemv.mm(
                x[:, group],
                self.weights[group],
                out=result[:, group * width : (group + 1) * width],
            )
        retain_cuda_graph_capture_resource((x, result))
        return result


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config, prefix, *, owns_k, k_cache, ratio):
        super().__init__()
        hf = config.model_config.hf_config
        self.prefix, self.owns_k, self.k_cache = prefix, owns_k, k_cache
        self.heads = hf.index_n_heads // get_tensor_model_parallel_world_size()
        self.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.index_n_heads * 128,
            bias=False,
            return_bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        _native_linear(self.wq_b)
        self.weights_proj = ColumnParallelLinear(
            hf.hidden_size,
            hf.index_n_heads,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.weights_proj.quant_method = B12xLinearMethod()
        if owns_k:
            self.wk = ReplicatedLinear(
                512,
                128,
                bias=False,
                return_bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.wk.quant_method = B12xLinearMethod()
            self.k_norm = B12xRMSNorm(128, hf.rms_norm_eps)
        self.ratio = ratio


@torch.library.custom_op("vllm::dsv41_b12x_attention", mutates_args=("out",))
def _attention(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    prefix: str,
    global_kv_ready: torch.Tensor | None,
) -> None:
    context = get_forward_context()
    layer = context.no_compile_layers[prefix]
    out.copy_(layer._forward(positions, hidden))


@_attention.register_fake
def _attention_fake(hidden, positions, out, prefix, global_kv_ready):
    return None


@torch.library.custom_op("vllm::dsv41_b12x_prepare_global_kv", mutates_args=())
def _prepare_global_kv(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
) -> torch.Tensor:
    context = get_forward_context()
    context.no_compile_layers[prefix]._prepare_global_kv(positions, hidden)
    # A small explicit data dependency orders the following opaque attention
    # call without asking functionalization to clone the aliased KV pool.
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


@_prepare_global_kv.register_fake
def _prepare_global_kv_fake(hidden, positions, prefix):
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


class DeepseekV4Attention(nn.Module, AttentionLayerBase):
    backend_cls = DeepseekV41B12xBackend
    INDEX_CHUNK = 64

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
        candidate_block_buffer=None,
    ):
        super().__init__()
        self.config = vllm_config
        hf = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.layer_id = int(re.search(r"layers\.(\d+)", prefix).group(1))
        self.hidden_size, self.head_dim = hf.hidden_size, hf.head_dim
        self.rope_head_dim = hf.qk_rope_head_dim
        tp = get_tensor_model_parallel_world_size()
        self.n_local_heads = hf.num_attention_heads // tp
        self.n_local_groups = hf.o_groups // tp
        self.n_groups, self.o_lora_rank = hf.o_groups, hf.o_lora_rank
        self.q_lora_rank, self.window_size = hf.q_lora_rank, hf.sliding_window
        spec = vllm_config.speculative_config
        self.is_draft = (
            self.layer_id >= hf.num_hidden_layers
            and spec is not None
            and spec.use_dspark()
        )
        boundary = ced_decoder_start(hf)
        self.is_ced_decoder = (
            not self.is_draft and boundary is not None and self.layer_id >= boundary
        )
        self.swa_width = self.window_size
        if self.is_draft:
            from vllm.v1.attention.backends.mla.compressor_utils import (
                get_dspark_swa_index_width,
            )

            self.swa_width = get_dspark_swa_index_width(
                self.window_size, spec.num_speculative_tokens
            )
        self.capacity = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.compress_ratio = (
            hf.compress_ratios[self.layer_id]
            if self.layer_id < len(hf.compress_ratios)
            else 0
        )
        self.is_kv_source = self.layer_id in hf.kv_source_layer_ids
        self.is_index_source = self.layer_id in hf.index_source_layer_ids
        self.kv_source_layer_id = (
            max(s for s in hf.kv_source_layer_ids if s <= self.layer_id)
            if self.compress_ratio
            else None
        )
        self.index_source_layer_id = (
            max(s for s in hf.index_source_layer_ids if s <= self.layer_id)
            if self.compress_ratio
            else None
        )
        self.candidate_source_layer = hf.candidate_source_layer_id
        if (
            self.head_dim != 512
            or self.rope_head_dim != 64
            or self.compress_ratio not in (0, 1, 2)
        ):
            raise ValueError("unsupported V4.1 native attention geometry")
        if (
            vllm_config.parallel_config.decode_context_parallel_size != 1
            or vllm_config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "V4.1 TP shards heads, not context; context parallel is unsupported"
            )
        self._context = vllm_config.compilation_config.static_forward_context
        if prefix in self._context:
            raise ValueError(f"Duplicate attention layer {prefix}")
        self._context[prefix] = self
        self.kv_cache = torch.tensor([])
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            hf.hidden_size,
            [hf.q_lora_rank, 512],
            bias=False,
            quant_config=vllm_config.quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_wqa_wkv",
        )
        self.q_norm = B12xRMSNorm(hf.q_lora_rank, hf.rms_norm_eps)
        self.kv_norm = B12xRMSNorm(512, hf.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.num_attention_heads * 512,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            hf.num_attention_heads * 512 // hf.o_groups,
            hf.o_groups * hf.o_lora_rank,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm, self.wo_a.bmm_batch_size = True, self.n_local_groups
        self.wo_a.quant_method = _GroupedLinearMethod(
            self.wo_a.quant_method, self.n_local_groups
        )
        self.wo_b = RowParallelLinear(
            hf.o_groups * hf.o_lora_rank,
            hf.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wo_b",
            reduce_results=False,
        )
        for linear in (self.fused_wqa_wkv, self.wq_b, self.wo_b):
            _native_linear(linear)
        self.rotary_emb = build_deepseek_v4_rope(
            hf,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=hf.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.swa_cache_layer = _Cache(
            vllm_config,
            f"{prefix}.swa_cache",
            kind="swa",
            window=self.window_size,
            draft=self.is_draft,
        )
        self.compressor = (
            DeepseekCompressor(
                vllm_config,
                self.compress_ratio,
                hf.hidden_size,
                512,
                prefix=f"{prefix}.compressor",
            )
            if self.is_kv_source
            else None
        )
        self.indexer = None
        if self.is_index_source:
            if self.is_kv_source:
                k_cache = _Cache(
                    vllm_config,
                    f"{prefix}.indexer.k_cache",
                    kind="index",
                    ratio=self.compress_ratio,
                )
            else:
                k_cache = self._context[
                    f"{_source(prefix, self.kv_source_layer_id)}.indexer.k_cache"
                ]
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=k_cache,
                ratio=self.compress_ratio,
            )
        self.topk_indices_buffer = topk_indices_buffer
        self._ready = False

    def get_attn_backend(self):
        return self.backend_cls

    def get_kv_cache_spec(self, config):
        if not self.is_kv_source:
            return None
        return MLAAttentionSpec(
            block_size=config.cache_config.block_size,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=288,
            dtype=torch.uint8,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
        )

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def _owner(self):
        return self._context[_source(self.prefix, self.kv_source_layer_id)]

    def _prepare(self, device):
        if self._ready:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 attention must be warmed before graph capture")

        def alloc(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)

        c = self.INDEX_CHUNK
        self._main_page = self.config.cache_config.block_size // max(
            self.compress_ratio, 1
        )
        self._main_width = (
            self.max_model_len + self.config.cache_config.block_size - 1
        ) // self.config.cache_config.block_size
        self._index_page = self._main_page
        self._index_width = self._main_width
        spec = self.config.speculative_config
        # Parallel drafting can admit two draft spans during verifier profiling.
        query_width = (
            1 + (2 if spec.parallel_drafting else 1) * spec.num_speculative_tokens
            if spec is not None
            else 1
        )
        decode_rows = self.config.scheduler_config.max_num_seqs * query_width
        # Graph buffers include padding beyond the live decode-token bound.
        decode_rows = max(
            decode_rows,
            self.config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._attention_workspace_specs = {}
        self._plans = {}
        for mode, capacity in (
            ("decode", min(self.capacity, decode_rows)),
            ("extend", self.capacity),
        ):
            plan = mla.plan(
                mla.Caps(
                    device=device,
                    num_q_heads=self.n_local_heads,
                    max_q_rows=capacity,
                    max_width=self.swa_width + (512 if self.compress_ratio else 0),
                    swa_width=self.swa_width,
                    indexed_width=512 if self.compress_ratio else 0,
                    swa_page_size=32,
                    indexed_page_size=self._main_page,
                    max_page_table_width=self._main_width,
                    mode=mode,
                    cache_format="deepseek_v41",
                    use_cuda_graph=True,
                )
            )
            self._plans[mode] = plan
            metadata_specs = (
                ((capacity, self.swa_width), torch.int32),
                ((capacity,), torch.int32),
                ((capacity,), torch.int32),
            )
            if self.compress_ratio:
                metadata_specs += (((capacity, self._main_width), torch.int32),)
            workspace_specs = metadata_specs + tuple(plan.shapes_and_dtypes())
            self._attention_workspace_specs[mode] = workspace_specs
            current_workspace_manager().get_simultaneous(*workspace_specs)
        if self.topk_indices_buffer is None and self.is_index_source:
            self.topk_indices_buffer = alloc((self.capacity, 512), torch.int32)
        if self.indexer is not None:
            self._index_pages = alloc((c, self._index_width), torch.int32)
            self._active = torch.full(
                (1,),
                self._index_width * self._index_page,
                dtype=torch.int32,
                device=device,
            )
            h = self.indexer.heads
            self._index_plans = {}
            for mode in ("decode", "prefill"):
                plan = dsa_indexer.plan(
                    dsa_indexer.Caps(
                        device=device,
                        num_q_heads=h,
                        max_q_rows=c,
                        max_page_table_width=self._index_width,
                        topk=512,
                        mode=mode,
                        cache_format="mxfp4",
                        page_size=self._index_page,
                        max_candidates=16384
                        if self.layer_id > self.candidate_source_layer
                        else 0,
                        candidate_topk_blocks=2048
                        if self.layer_id == self.candidate_source_layer
                        else 0,
                    )
                )
                self._index_plans[mode] = plan
                _scratch(plan)
            if self.layer_id == self.candidate_source_layer:
                self._candidates = alloc((self.capacity, 16384), torch.int32)
                self._candidate_lens = alloc((self.capacity,), torch.int32)
        if self.compressor is not None:
            self.compressor.prepare(device)
        self._ready = True

    def insert_context_kv(self, kv, positions, slot_mapping):
        self._prepare(kv.device)
        rotated = _rotated(kv, positions, self.rotary_emb.cos_sin_cache)
        mla.write_cache(
            rotated,
            self.swa_cache_layer.kv_cache,
            slot_mapping,
            page_size=32,
            cache_kind="swa",
            cache_format="deepseek_v41",
        )

    def _query_metadata(self, metadata):
        if self.is_ced_decoder and metadata.decoder is not None:
            return metadata.decoder
        return metadata

    def prepare_global_kv(self, positions, hidden_states):
        """Build full-row global KV before the CED boundary gathers decoder rows."""
        if self.compressor is None or hidden_states.shape[0] == 0:
            return
        return _prepare_global_kv(hidden_states, positions, self.prefix)

    def _prepare_global_kv(self, positions, hidden_states):
        self._prepare(hidden_states.device)
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or self.compressor is None:
            return
        rows = hidden_states.shape[0]
        # These are deliberately original metadata, never the decoder views.
        main = metadata[self._owner().prefix]
        state = (
            metadata[self.compressor.state_cache.prefix]
            if self.compressor.state_cache is not None
            else None
        )
        latent, slots = self.compressor(hidden_states, main, state)
        # Index K consumes the ordinary-normalized PRE-RoPE latent.
        key = self.indexer.k_norm(self.indexer.wk(latent))
        key = _rotated(
            key,
            positions,
            self.rotary_emb.cos_sin_cache,
            ratio=self.compress_ratio,
        )
        index_meta = metadata[self.indexer.k_cache.prefix]
        dsa_indexer.quantize_write_index_k_mxfp4(
            key,
            index_k_cache=self.indexer.k_cache.kv_cache,
            slot_mapping=index_meta.slot_mapping[:rows],
            page_size=self._index_page,
        )
        latent = _rotated(
            latent,
            positions,
            self.rotary_emb.cos_sin_cache,
            ratio=self.compress_ratio,
        )
        mla.write_cache(
            latent,
            self.kv_cache,
            slots,
            page_size=self._main_page,
            cache_kind="indexed",
            cache_format="deepseek_v41",
        )

    def forward(
        self, positions, hidden_states, llama_4_scaling=None, *, global_kv_ready=None
    ):
        out = torch.empty_like(hidden_states)
        if hidden_states.shape[0] == 0:
            return out
        _attention(hidden_states, positions, out, self.prefix, global_kv_ready)
        return out

    def _forward(self, positions, hidden_states):
        self._prepare(hidden_states.device)
        rows = hidden_states.shape[0]
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split((self.q_lora_rank, 512), dim=-1)
        qr, kv = self.q_norm(qr), self.kv_norm(kv)
        q = self.wq_b(qr).view(rows, self.n_local_heads, 512)
        q = _rotated(q, positions, self.rotary_emb.cos_sin_cache)
        output = torch.empty_like(q)
        retain_cuda_graph_capture_resource(output)
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            # vLLM memory profiling has no cache pages; no serving call uses this branch.
            output.zero_()
        else:
            original_swa = metadata[self.swa_cache_layer.prefix]
            swa = self._query_metadata(original_swa)
            self.insert_context_kv(kv, positions, swa.slot_mapping[:rows])
            if swa is original_swa:
                self._prepare_global_kv(positions, hidden_states)
            index_query = None
            if self.indexer is not None:
                h = self.indexer.heads
                iq = self.indexer.wq_b(qr).view(rows, h, 128)
                iq = _rotated(iq, positions, self.rotary_emb.cos_sin_cache)
                iq_data = torch.empty((rows, h, 64), dtype=torch.uint8, device=q.device)
                iq_scale = torch.empty((rows, h, 4), dtype=torch.uint8, device=q.device)
                dsa_indexer.quantize_q_mxfp4(iq, q_mxfp4=iq_data, q_scales=iq_scale)
                weights = self.indexer.weights_proj(hidden_states)
                iw = torch.empty_like(weights)
                scale_index_weights(weights, out=iw)
                index_query = (iq_data, iq_scale, iw)
                retain_cuda_graph_capture_resource((weights, index_query))
            self.forward_mqa(q, kv, positions, output, index_query=index_query)
        return self._o_proj(output, positions)

    def forward_mqa(self, q, kv, positions, output, *, index_query=None):
        metadata = get_forward_context().attn_metadata
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        main = (
            self._query_metadata(metadata[self._owner().prefix])
            if self.compress_ratio
            else None
        )
        mode = "decode" if swa.is_decode else "extend"
        rows = q.shape[0]
        plan = self._plans[mode]
        if rows > plan.caps.max_q_rows:
            raise ValueError(
                f"V4.1 {mode} rows {rows} exceed planned capacity {plan.caps.max_q_rows}"
            )
        owner = (
            self._context[_source(self.prefix, self.index_source_layer_id)]
            if self.compress_ratio
            else None
        )

        # Only the score matrix needs row chunking. All selected positions
        # survive in the source-owned buffer until their reuse interval ends.
        if self.indexer is not None:
            index_mode = "decode" if mode == "decode" else "prefill"
            iq_data, iq_scale, iw = index_query
            im = self._query_metadata(metadata[self.indexer.k_cache.prefix])
            index_plan = self._index_plans[index_mode]
            score_width = None
            if (
                mode == "extend"
                and self.layer_id <= self.candidate_source_layer
                and not torch.cuda.is_current_stream_capturing()
            ):
                score_width = min(
                    self._index_width * self._index_page,
                    max(1, triton.cdiv(im.max_seq_len, self.compress_ratio)),
                )
            for offset in range(0, rows, self.INDEX_CHUNK):
                end = min(offset + self.INDEX_CHUNK, rows)
                count = end - offset
                _pages[(count, triton.cdiv(self._index_width, 128))](
                    im.req_id_per_token,
                    im.block_table,
                    self._index_pages,
                    offset,
                    im.block_table.stride(0),
                    im.block_table.shape[1],
                    self._index_width,
                    128,
                )
                candidate_args = {}
                if self.layer_id == self.candidate_source_layer:
                    candidate_args = dict(
                        candidate_output=self._candidates[offset:end],
                        candidate_output_lengths=self._candidate_lens[offset:end],
                    )
                elif self.layer_id > self.candidate_source_layer:
                    source = self._context[
                        _source(self.prefix, self.candidate_source_layer)
                    ]
                    candidate_args = dict(
                        candidate_indices=source._candidates[offset:end],
                        candidate_lengths=source._candidate_lens[offset:end],
                    )
                # One borrow spans score -> TP reduction -> select.
                binding = dsa_indexer.bind(
                    index_plan,
                    scratch=_scratch(index_plan),
                    q_mxfp4=iq_data[offset:end],
                    q_scales=iq_scale[offset:end],
                    query_weights=iw[offset:end],
                    index_k_cache=self.indexer.k_cache.kv_cache,
                    page_table=self._index_pages[:count],
                    cache_lengths=im.cache_lengths[offset:end],
                    active_width=self._active,
                    score_width=score_width,
                    output_indices=self.topk_indices_buffer[offset:end],
                    **candidate_args,
                )
                retain_cuda_graph_capture_resource(binding)
                scores = dsa_indexer.score(binding)
                if get_tensor_model_parallel_world_size() > 1:
                    reduced = get_tp_group().all_reduce(scores)
                    retain_cuda_graph_capture_resource(reduced)
                    if reduced.data_ptr() != scores.data_ptr():
                        scores.copy_(reduced)
                dsa_indexer.select(binding)

        # Reuse the shared arena only after indexing completes. Keep full-batch
        # metadata beside, not overlapping, the native attention scratch.
        buffers = current_workspace_manager().get_simultaneous(
            *self._attention_workspace_specs[mode]
        )
        swa_indices, swa_lengths, top_lengths = buffers[:3]
        visible = main.cache_lengths if main is not None else swa.cache_lengths
        _chunk[(rows,)](
            swa.positions,
            swa.req_id_per_token,
            swa.block_table,
            swa_indices,
            swa_lengths,
            top_lengths,
            visible,
            swa.query_start_loc,
            swa.request_positions,
            0,
            swa.block_table.stride(0),
            32,
            self.window_size,
            self.swa_width,
            self.is_draft,
            triton.next_power_of_2(self.swa_width),
            swa_replay_start=swa.swa_replay_start if self.is_ced_decoder else None,
        )
        kwargs = {}
        metadata_count = 3
        if main is not None:
            main_pages = buffers[3]
            metadata_count += 1
            _pages[(rows, triton.cdiv(self._main_width, 128))](
                main.req_id_per_token,
                main.block_table,
                main_pages,
                0,
                main.block_table.stride(0),
                main.block_table.shape[1],
                self._main_width,
                128,
            )
            kwargs = dict(
                indexed_indices=owner.topk_indices_buffer[:rows],
                indexed_lengths=top_lengths[:rows],
                indexed_page_table=main_pages[:rows],
            )
        binding = mla.bind(
            plan,
            scratch=buffers[metadata_count:],
            q=q,
            swa_indices=swa_indices[:rows],
            swa_lengths=swa_lengths[:rows],
            **kwargs,
        )
        retain_cuda_graph_capture_resource(binding)
        mla.run(
            binding=binding,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            indexed_k_cache=self._owner().kv_cache if main is not None else None,
            swa_page_size=32,
            indexed_page_size=self._main_page,
            sm_scale=512**-0.5,
            attn_sink=self.attn_sink,
            out=output,
            cache_format="deepseek_v41",
        )

    def _o_proj(self, o, positions):
        rows = o.shape[0]
        inverse = _rotated(o, positions, self.rotary_emb.cos_sin_cache, inverse=True)
        grouped = inverse.view(rows, self.n_local_groups, -1)
        local = self.wo_b(self.wo_a.quant_method.apply(self.wo_a, grouped))
        if local.dtype != torch.bfloat16:
            raise TypeError("V4.1 WO-B must round the local projection to BF16")
        if get_tensor_model_parallel_world_size() > 1:
            local = get_tp_group().all_reduce(local)
        retain_cuda_graph_capture_resource(local)
        return local
