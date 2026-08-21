# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
# MLA Common Components

This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach. We generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is relatively large, often near
1) and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is small, often near 0).

NOTE what we deem small and large is currently determined by if it is labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is an example of both paths assuming batch size = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "forward_mha"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      sdpa_o shape [Sq, N, V]
sdpa_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return sdpa_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "forward_mqa"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q_nope, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      sdpa_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
sdpa_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`

To mitigate this, we chunk the computation of attention with respect to the
current context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a
fixed workspace size.

The chunked prefill approach is as follows:

W          Workspace rows, i.e. the context rows we may gather at once,
           used to bound the memory usage

The context is scheduled per request: requests are packed in order, splitting
the next request on an aligned boundary when needed to fill `W`.
So a chunk covers a contiguous run of prefills and only ever the tokens and
context rows of those prefills — attention, up-projection and merging are
charged only to the requests it covers, and no chunk contains an empty context
span. See `plan_mla_context_chunks`.

Only a chunk's first request can be a continuation, so accumulation performs at
most one request-slice merge and one bulk write per chunk.

q_c        = h_t @ W_DQ
q_nope     = (q_c @ W_UQ).view(Sq, N, P)
q_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c   = h_t @ W_DKV
new_k_pe   = RoPE(h_t @ W_KR)
new_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)
new_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)

// MHA between queries and new KV
//     with QK headdim = P + R
//           V headdim = V
//    curr_o   shape [Sq, N, V]
//    curr_lse shape [N, Sq], this is just order FA returns
curr_o, curr_lse = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    new_v,
    causal=True,
    return_softmax_lse=True
)

// Compute attention with the already existing context. Shown for a single
// request; with a batch, a chunk covers a run of requests and both `q` and the
// gathered context below are sliced to that run.
for chunk_idx in range(cdiv(C, W)):
    chunk_start  = chunk_idx * W
    chunk_end    = min(chunk_start + W, C)
    Sc           = chunk_end - chunk_start
    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]
    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]
    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)
    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)

    chunk_o, chunk_lse = scaled_dot_product_attention(
        torch.cat([q_nope, q_pe], dim=-1),
        torch.cat([cache_k_nope_chunk,
                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],
                   dim=-1),
        cache_v_chunk,
        causal=False,
        return_softmax_lse=True
    )

    curr_o, curr_lse = merge_attn_states(
        suffix_output=curr_o,
        suffix_lse=curr_lse,
        prefix_output=chunk_o,
        prefix_lse=chunk_lse,
    )

return curr_o @ W_O
"""

import functools
import itertools
import math
import os
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import lcm
from typing import ClassVar, Generic, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_tp_group,
    is_global_first_rank,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.b12x_mxfp8_bmm import (
    can_implement_b12x_mxfp8_bmm,
    can_implement_bf16_mla_query,
    can_implement_mxfp8_mla_query,
    run_b12x_mxfp8_bmm,
    run_bf16_mla_query,
    run_mxfp8_mla_query,
)
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    get_attention_context,
    set_default_quant_scales,
    should_load_quant_weights,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention.pcp import (
    finalize_mla_pcp_decode,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    get_and_maybe_dequant_weights,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_down, round_up
from vllm.utils.multi_stream_utils import is_vllm_cudagraph_capture_active
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
    np_to_pinned_tensor,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.prefill import (
    MLAPrefillBackend,
    get_mla_prefill_backend,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_num_attention_heads_from_layers,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import (
    cp_lse_ag_out_rs,
    cp_lse_ag_out_rs_into,
)
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_a2a_lse_reduce,
    dcp_b12x_all_gather_heads,
    sanitize_dcp_attn_empty_rows,
)
from vllm.v1.attention.ops.dcp_utils import MLADCPManager
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_cache_dcp_shard_count,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

_FP8_DTYPE = current_platform.fp8_dtype()
_B12X_ABSORB_BMM_MAX_M = 32


def _get_mla_kv_dcp_world_size(
    kv_cache_spec: KVCacheSpec, configured_dcp_world_size: int
) -> int:
    """Return the DCP group size that owns distinct KV position shards."""
    shard_count = get_kv_cache_dcp_shard_count(kv_cache_spec, configured_dcp_world_size)
    if shard_count not in (1, configured_dcp_world_size):
        raise NotImplementedError(
            "MLA metadata supports fully sharded or replicated DCP KV groups; "
            "partial DCP KV sharding requires a matching subgroup"
        )
    return shard_count


def _run_mla_query_bmm(
    query: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    use_safe_op: bool,
) -> None:
    if (
        use_safe_op
        and current_platform.is_cuda()
        and query.is_cuda
        and weight.is_cuda
        and output.is_cuda
        and query.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
    ):
        try:
            safe_bmm = torch.ops._C.safe_mla_query_bmm
        except AttributeError:
            safe_bmm = None
        if safe_bmm is not None:
            safe_bmm(query, weight, output)
            return

    # Fallback for CPU tests, non-BF16 paths, and builds without the CUDA op.
    # The copy keeps tight DCP/custom-allocation query views out of torch.bmm.
    torch.bmm(query.contiguous() if use_safe_op else query, weight, out=output)


@functools.cache
def _b12x_absorb_bmm_enabled() -> bool:
    return envs.VLLM_B12X_ABSORB_BMM


def _find_linear_weight_device(layer: torch.nn.Module) -> torch.device | None:
    """Find the device that owns a linear layer's loaded weights.

    Args:
        layer: Linear layer or a wrapper around one.

    Returns:
        The loaded weight device, or ``None`` when the layer owns no tensors.
    """
    while hasattr(layer, "base_layer") and hasattr(layer.base_layer, "quant_method"):
        layer = layer.base_layer

    for name in ("weight", "qweight", "weight_packed"):
        weight = getattr(layer, name, None)
        if isinstance(weight, torch.Tensor):
            return weight.device
    for parameter in layer.parameters(recurse=False):
        return parameter.device
    for buffer in layer.buffers(recurse=False):
        return buffer.device
    return None


def _preallocate_absorbed_mla_weights(
    layer: "MLAAttention", act_dtype: torch.dtype
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Allocate persistent MLA weights before temporary dequantization storage.

    Compatible weights from an earlier load are reused in place, preserving
    their addresses for CUDA graphs.

    Args:
        layer: MLA attention layer whose KV projection will be absorbed.
        act_dtype: Data type used by the absorbed projection weights.

    Returns:
        Optional new storage for ``W_UV`` and ``W_UK_T``, respectively.

    Raises:
        RuntimeError: If neither the source projection nor reusable absorbed
            weights identify the target device.
    """
    # Model-specific MLA wrappers may retain the global head count while their
    # absorbed projection is tensor-parallel and therefore uses local heads.
    num_heads = getattr(layer, "num_local_heads", layer.num_heads)
    w_uv_shape = (num_heads, layer.kv_lora_rank, layer.v_head_dim)
    w_uk_t_shape = (
        num_heads,
        layer.qk_nope_head_dim,
        layer.kv_lora_rank,
    )
    current_w_uv = getattr(layer, "W_UV", None)
    current_w_uk_t = getattr(layer, "W_UK_T", None)

    device = _find_linear_weight_device(layer.kv_b_proj)
    if device is None:
        current_devices = {
            weight.device
            for weight in (current_w_uv, current_w_uk_t)
            if isinstance(weight, torch.Tensor)
        }
        if len(current_devices) != 1:
            raise RuntimeError(
                "Cannot determine the device for absorbed MLA projection weights."
            )
        device = current_devices.pop()

    def needs_storage(weight: object, shape: tuple[int, ...]) -> bool:
        return not (
            isinstance(weight, torch.Tensor)
            and weight.shape == shape
            and weight.dtype == act_dtype
            and weight.device == device
        )

    pre_w_uv = (
        torch.empty(w_uv_shape, dtype=act_dtype, device=device)
        if needs_storage(current_w_uv, w_uv_shape)
        else None
    )
    pre_w_uk_t = (
        torch.empty(w_uk_t_shape, dtype=act_dtype, device=device)
        if needs_storage(current_w_uk_t, w_uk_t_shape)
        else None
    )
    return pre_w_uv, pre_w_uk_t


def _can_use_b12x_dcp_prefill_workspace(
    *,
    enabled: bool,
    project_before_merge: bool,
    dcp_use_b12x: bool,
    num_tokens: int,
    max_num_tokens: int,
    non_dbo_workspace: bool,
    is_sparse_impl: bool,
    backend_name: str,
    is_capturing: bool,
) -> bool:
    """Gate the B12X eager-prefill workspace contract."""
    return (
        enabled
        and project_before_merge
        and not dcp_use_b12x
        and 1025 <= num_tokens <= max_num_tokens
        and non_dbo_workspace
        and is_sparse_impl
        and backend_name == "B12X_MLA_SPARSE"
        and not is_capturing
    )


def _estimate_dcp_ag_rs_transient_bytes(
    *,
    num_tokens: int,
    local_heads: int,
    dcp_world_size: int,
    q_head_dim: int,
    output_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    project_before_merge: bool,
) -> int:
    """Upper-bound simultaneously live eager DCP AG/RS attention tensors."""
    if num_tokens <= 0 or local_heads <= 0 or dcp_world_size <= 1:
        return 0

    bf16_bytes = 2
    fp32_bytes = 4
    global_heads = local_heads * dcp_world_size

    gathered_query = num_tokens * global_heads * q_head_dim * bf16_bytes
    attention_output = num_tokens * global_heads * output_head_dim * bf16_bytes
    # CUDA communicator materializes a head-major contiguous RS input, then an
    # output and its token-major contiguous return while attention_output lives.
    reduce_scatter = attention_output + (
        2 * num_tokens * local_heads * output_head_dim * bf16_bytes
    )
    gathered_lse = (dcp_world_size + 1) * num_tokens * global_heads * fp32_bytes
    gathered_w_uv = (
        global_heads * kv_lora_rank * v_head_dim * bf16_bytes
        if project_before_merge
        else 0
    )
    return (
        gathered_query
        + attention_output
        + reduce_scatter
        + gathered_lse
        + gathered_w_uv
    )


def _should_allocate_sparse_profile_workspace(workspace_bytes: int) -> bool:
    return workspace_bytes > 0 and not is_vllm_cudagraph_capture_active()


def _extract_single_layer_index(layer_name: str) -> int | None:
    int_vals = [int(part) for part in layer_name.split(".") if part.isdecimal()]
    return int_vals[0] if len(int_vals) == 1 else None


def _match_merge_strides(
    prefix_output: torch.Tensor, suffix_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give both partial-attention outputs identical strides.

    The merge_attn_states CUDA kernel computes one source offset from the
    prefix strides and reads BOTH prefix and suffix with it. The FA prefill
    backend returns padded-V views (head stride > head_size) while merged
    chunk outputs are contiguous, so once the chunked-context loop runs more
    than one iteration (context > workspace, i.e. > 64k tokens) the two
    sides disagree and the kernel reads the suffix at wrong offsets.
    """
    if prefix_output.stride() != suffix_output.stride():
        if not prefix_output.is_contiguous():
            prefix_output = prefix_output.contiguous()
        if not suffix_output.is_contiguous():
            suffix_output = suffix_output.contiguous()
    return prefix_output, suffix_output


def _detect_output_quant_key(
    output: torch.Tensor,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
    output_dim: int,
) -> QuantKey | None:
    """Detect the output quantization key from fusion pass parameters.

    Returns the appropriate QuantKey, or None if no quantization is needed.
    Detection is based on output dtype and which scale tensors are present.
    """
    if output_scale is None and output_block_scale is None:
        return None
    if output_block_scale is not None:
        if output.dtype == _FP8_DTYPE:
            # Per-group FP8 uses block scales only, not a separate output_scale
            assert output_scale is None
            # Infer group size from scale shape
            num_groups = output_block_scale.shape[-1]
            group_size = output_dim // num_groups
            if group_size == 128:
                return kFp8Dynamic128Sym
            elif group_size == 64:
                return kFp8Dynamic64Sym
            else:
                raise ValueError(
                    f"Unsupported group FP8 group_size={group_size} "
                    f"(output_dim={output_dim}, num_groups={num_groups}). "
                    f"Only group_size 128 and 64 are supported."
                )
        # output_scale None implies MXFP4, not supported
        assert output_scale is not None
        return kNvfp4Dynamic
    return kFp8StaticTensorSym


def _canonicalize_sparse_mla_kv_cache_dtype(
    attn_backend: type[AttentionBackend],
    kv_cache_dtype: CacheDType,
) -> CacheDType:
    backend_name = attn_backend.get_name()
    if backend_name == "B12X_MLA_SPARSE" and kv_cache_dtype == "nvfp4_ds_mla":
        # B12X reads the packed 432B NVFP4 MLA record natively; do NOT coerce
        # it to fp8_ds_mla. [nvfp4_reader_port]
        return "nvfp4_ds_mla"
    if backend_name in (
        "FLASHMLA_SPARSE",
        "B12X_MLA_SPARSE",
    ) and is_quantized_kv_cache(kv_cache_dtype):
        # NOTE: nvfp4_ds_mla deliberately falls through to fp8_ds_mla for
        # FLASHMLA_SPARSE (no NVFP4 reader there).
        return "fp8_ds_mla"
    if backend_name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in (
        "auto",
        "fp8",
        "fp8_e4m3",
    ):
        return "fp8_ds_mla"
    return kv_cache_dtype


def _get_kv_b_proj_input_dtype(
    kv_b_proj: ColumnParallelLinear, use_fp8_prefill: bool
) -> torch.dtype | None:
    weight = getattr(kv_b_proj, "weight", None)
    weight_dtype = weight.dtype if weight is not None else kv_b_proj.params_dtype
    if weight_dtype == torch.int32:
        return kv_b_proj.params_dtype
    if weight_dtype == torch.uint8:
        return None
    if weight_dtype == current_platform.fp8_dtype() and not use_fp8_prefill:
        return None
    return weight_dtype


def _get_dcp_batch_metadata(
    attn_metadata: AttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned request lengths and query offsets for MLA DCP merge."""
    decode_metadata = getattr(attn_metadata, "decode", None)
    if decode_metadata is not None:
        num_reqs = attn_metadata.num_decodes  # type: ignore[attr-defined]
        seq_lens = cast(torch.Tensor, decode_metadata.seq_lens)
    else:
        # Sparse MLA routes prefill and decode through one MQA path. Its
        # metadata therefore describes every request at the top level rather
        # than exposing a nested decode-only object.
        num_reqs = attn_metadata.num_reqs  # type: ignore[attr-defined]
        seq_lens = cast(
            torch.Tensor, attn_metadata.seq_lens
        )[  # type: ignore[attr-defined]
            :num_reqs
        ]
    query_start_loc = attn_metadata.query_start_loc[  # type: ignore[attr-defined]
        : num_reqs + 1
    ]
    return seq_lens, query_start_loc


class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    supports_dense_mha_prefill: ClassVar[bool] = True

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        dcp_q_replicate: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        non_causal_multi_token_decode: bool = False,
        sliding_window: int | None = None,
        prefill_backend_cls: type[MLAPrefillBackend] | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.dcp_q_replicate = dcp_q_replicate
        self.W_UK_T_dcp_qrep: torch.Tensor | None = None
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer
        self.non_causal_multi_token_decode = non_causal_multi_token_decode
        self.sliding_window = sliding_window
        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if cache_config is not None:
            kv_cache_dtype: CacheDType = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"
        self.quant_config = quant_config

        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                kv_cache_dtype = "auto"
            logger.debug(
                "Layer %s: kv_cache_dtype=%s",
                prefix,
                kv_cache_dtype,
            )

        dtype = torch.get_default_dtype()
        if attn_backend is not None:
            assert attn_backend.is_mla(), (
                f"MLAAttention: attn_backend must be an MLA backend, "
                f"got {attn_backend.get_name()} instead"
            )
            self.attn_backend = attn_backend
        else:
            self.attn_backend = get_attn_backend(
                self.head_size,
                dtype,
                kv_cache_dtype,
                use_mla=True,
                use_sparse=use_sparse,
                num_heads=self.num_heads,
            )

        normalized_kv_cache_dtype = _canonicalize_sparse_mla_kv_cache_dtype(
            self.attn_backend, kv_cache_dtype
        )
        if normalized_kv_cache_dtype != kv_cache_dtype:
            if cache_config is not None:
                cache_config.cache_dtype = normalized_kv_cache_dtype
            kv_cache_dtype = normalized_kv_cache_dtype
            logger.info_once(
                "Using %s KV cache format for %s backend.",
                kv_cache_dtype,
                self.attn_backend.get_name(),
            )

        if (
            self.attn_backend.get_name() == "FLASHINFER_MLA_SPARSE"
            and kv_cache_dtype != "fp8_ds_mla"
            and is_quantized_kv_cache(kv_cache_dtype)
        ):
            logger.info_once(
                "Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla "
                "KV cache format, please set `--attention-backend FLASHMLA_SPARSE`"
            )

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        _init_kv_cache_quant(self, quant_config, prefix)

        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and envs.VLLM_BATCH_INVARIANT
            and (
                self.attn_backend.get_name() == "TRITON_MLA"
                or self.attn_backend.get_name() == "FLASHINFER"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for TRITON_MLA / FLASHINFER "
                "with batch invariance, as it is not yet supported.",
            )
            cache_config.enable_prefix_caching = False

        # Sparse MLA reads top-k indices from a shared buffer. Pass it
        # explicitly so backbone "skip" layers (indexer=None) still find it.
        if use_sparse:
            extra_impl_args["topk_indices_buffer"] = topk_indices_buffer

        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an MLAAttentionImpl subclass
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=sliding_window,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )
        self.q_pad_num_heads = getattr(self.impl, "q_pad_num_heads", None)
        self.use_safe_mla_query_bmm = getattr(
            self.impl, "use_safe_mla_query_bmm", False
        )
        self.use_direct_call = not current_platform.opaque_attention_op()

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.prefill_backend: MLAPrefillBackend | None
        if self.impl.is_sparse and not (
            self.impl.supports_dense_mha_prefill and self.supports_dense_mha_prefill
        ):
            logger.warning_once(
                "Sparse MLA layer has no dense-MHA prefill path; using the top-k "
                "MQA path only."
            )
            self.prefill_backend = None
        else:
            try:
                prefill_backend_cls = prefill_backend_cls or get_mla_prefill_backend(
                    vllm_config
                )
            except ValueError:
                if (
                    not self.impl.is_sparse
                    or vllm_config.attention_config.mla_prefill_backend is not None
                ):
                    raise
                logger.warning_once(
                    "No MLA prefill backend supports this model; sparse MLA will "
                    "use the top-k MQA path only (no dense-MHA prefill)."
                )
                self.prefill_backend = None
            else:
                self.prefill_backend = prefill_backend_cls(
                    num_heads=self.num_heads,
                    scale=self.scale,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    vllm_config=vllm_config,
                )

        self.kv_cache = torch.tensor([])

        self.use_sparse = use_sparse

        self.dcp_manager: MLADCPManager | None = None
        if self.impl.dcp_world_size > 1:
            query_dtype = (
                current_platform.fp8_dtype()
                if is_quantized_kv_cache(self.kv_cache_dtype)
                and self.kv_cache_dtype != "fp8_ds_mla"
                and self.impl.supports_quant_query_input
                else dtype
            )
            self.dcp_manager = MLADCPManager(
                vllm_config=vllm_config,
                device=next(kv_b_proj.parameters()).device,
                num_heads=self.num_heads,
                query_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                output_head_dim=self.kv_lora_rank,
                query_dtype=query_dtype,
                output_dtype=dtype,
                padded_num_heads=self.q_pad_num_heads,
                is_lse_base_on_e=self.impl.lse_base_on_e,
                use_pcp=self.use_pcp,
            )
        self.dcp_a2a = self.dcp_manager is not None and self.dcp_manager.use_a2a
        self.dcp_b12x = (
            self.dcp_a2a
            and not self.use_pcp
            and envs.VLLM_USE_B12X_DCP_A2A
            and self.attn_backend.get_name() == "B12X_MLA_SPARSE"
            # The B12X PCIe DCP channel only exists for world sizes 2/4/8;
            # other DCP sizes (e.g. TP6 with DCP3/DCP6) use NCCL collectives.
            and parallel_config.decode_context_parallel_size in (2, 4, 8)
        )
        self.dcp_project_before_merge = (
            self.dcp_manager is not None
            and not self.use_pcp
            and envs.VLLM_DCP_PROJECT_BEFORE_MERGE
            and getattr(self.impl, "supports_dcp_project_before_merge", False)
        )
        self.dcp_project_before_merge_min_prefill_tokens = (
            envs.VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS
        )
        if self.dcp_project_before_merge_min_prefill_tokens < 0:
            raise ValueError(
                "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS must be "
                "non-negative, got "
                f"{self.dcp_project_before_merge_min_prefill_tokens}."
            )
        max_capture_size = max(compilation_config.cudagraph_capture_sizes, default=0)
        if (
            self.dcp_project_before_merge
            and self.dcp_project_before_merge_min_prefill_tokens < max_capture_size
        ):
            raise ValueError(
                "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS "
                f"({self.dcp_project_before_merge_min_prefill_tokens}) must "
                "be at least the maximum cudagraph capture size "
                f"({max_capture_size})."
            )
        self.dcp_max_batch_size = int(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        # Hybrid DCP dispatch: the one-shot A2A/B12X exchange is
        # latency-optimal for small decode batches but loses to pipelined
        # NCCL collectives on large prefill/extend batches. Batches with more
        # tokens than the cap take VLLM_DCP_A2A_LARGE_BACKEND instead
        # (0 = uncapped, pure A2A).
        self.dcp_a2a_max_tokens = envs.VLLM_DCP_A2A_MAX_TOKENS if self.dcp_a2a else 0
        self.dcp_a2a_large_backend = envs.VLLM_DCP_A2A_LARGE_BACKEND
        if self.dcp_a2a and self.dcp_a2a_large_backend not in ("ag_rs", "a2a"):
            raise ValueError(
                "VLLM_DCP_A2A_LARGE_BACKEND must be 'ag_rs' or 'a2a', got "
                f"{self.dcp_a2a_large_backend!r}."
            )

        self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()

        # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
        self.is_aiter_triton_fp4_bmm_enabled = (
            rocm_aiter_ops.is_fp4bmm_enabled()
            and hasattr(self.kv_b_proj, "weight")
            and self.kv_b_proj.weight.dtype == torch.bfloat16
        )

        # Attributes for forward_impl method
        self._vllm_config = get_current_vllm_config()
        self._chunked_prefill_workspace_size: int | None = None
        self._decode_concat_quant_fp8_op = _DecodeConcatQuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            compile_native=True,
        )
        self._quant_fp8_op = QuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            compile_native=True,
        )

    def _get_sparse_memory_profile_bytes(self) -> int:
        if (
            not envs.VLLM_MEMORY_PROFILE_INCLUDE_ATTN
            or self.attn_backend.get_name() != "B12X_MLA_SPARSE"
            or self.impl.dcp_world_size <= 1
        ):
            return 0

        max_tokens = int(getattr(self.impl, "_max_batched", 0))
        if max_tokens <= 0:
            return 0

        # Pure A2A does not enter the allocating NCCL AG/RS path. Hybrid A2A
        # enters it immediately above the configured small-batch cap.
        if self.dcp_a2a:
            if self.dcp_a2a_max_tokens <= 0 or self.dcp_a2a_large_backend == "a2a":
                return 0
            first_ag_rs_row = self.dcp_a2a_max_tokens + 1
        else:
            first_ag_rs_row = 1

        project_threshold = self.dcp_project_before_merge_min_prefill_tokens
        workspace_start = max(1025, project_threshold + 1)
        workspace_eligible = (
            workspace_start <= max_tokens
            and _can_use_b12x_dcp_prefill_workspace(
                enabled=envs.VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE,
                project_before_merge=self.dcp_project_before_merge,
                dcp_use_b12x=False,
                num_tokens=workspace_start,
                max_num_tokens=max_tokens,
                non_dbo_workspace=getattr(self.impl, "dcp_workspace_non_dbo", False),
                is_sparse_impl=self.impl.is_sparse,
                backend_name=self.attn_backend.get_name(),
                is_capturing=False,
            )
        )
        last_ag_rs_row = (
            min(max_tokens, workspace_start - 1) if workspace_eligible else max_tokens
        )
        if first_ag_rs_row > last_ag_rs_row:
            return 0

        candidates: list[tuple[int, bool]] = []
        if not self.dcp_project_before_merge:
            candidates.append((last_ag_rs_row, False))
        else:
            unprojected_rows = min(last_ag_rs_row, project_threshold)
            if unprojected_rows >= first_ag_rs_row:
                candidates.append((unprojected_rows, False))
            if last_ag_rs_row > project_threshold:
                candidates.append((last_ag_rs_row, True))

        return max(
            (
                _estimate_dcp_ag_rs_transient_bytes(
                    num_tokens=num_tokens,
                    local_heads=self.num_heads,
                    dcp_world_size=self.impl.dcp_world_size,
                    q_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                    output_head_dim=(
                        self.v_head_dim if projected else self.kv_lora_rank
                    ),
                    kv_lora_rank=self.kv_lora_rank,
                    v_head_dim=self.v_head_dim,
                    project_before_merge=projected,
                )
                for num_tokens, projected in candidates
            ),
            default=0,
        )

    @property
    def chunked_prefill_workspace_size(self) -> int:
        if self._chunked_prefill_workspace_size is None:
            self._chunked_prefill_workspace_size = (
                MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                    self._vllm_config
                )
            )
        return self._chunked_prefill_workspace_size

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata_raw = forward_context.attn_metadata
            attn_metadata: MLACommonMetadata
            if isinstance(attn_metadata_raw, dict):
                attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
            elif isinstance(attn_metadata_raw, list):
                # list[dict[str, AttentionMetadata]]: used in speculative decoding
                # where [0] is the base-model (non-speculative) metadata dict.
                attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
            else:
                attn_metadata = attn_metadata_raw
            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            assert isinstance(slot_mapping, dict), (
                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            )
            layer_slot_mapping = slot_mapping.get(self.layer_name)
            kv_for_cache, kpe_for_cache, layer_slot_mapping = (
                maybe_gather_mla_latent_cache_inputs(
                    kv_c_normed,
                    k_pe,
                    layer_slot_mapping,
                    attn_metadata.num_decode_tokens
                    if attn_metadata is not None
                    else None,
                    self.use_pcp,
                )
            )
            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
                kv_for_cache,
                kpe_for_cache,
                self_kv_cache,
                layer_slot_mapping,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.forward_impl(
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                output=output,
                q_dcp_replicated=q_dcp_replicated,
            )
            return output
        else:
            encoded = _encode_layer_name(self.layer_name)
            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                encoded,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            torch.ops.vllm.unified_mla_attention_with_output(
                q,
                kv_c_normed,
                k_pe,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                q_dcp_replicated=q_dcp_replicated,
            )
            return output

    def _try_fused_mla_query(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fuse a qualified BF16/MXFP8 query BMM with query assembly."""
        if self.is_aiter_triton_fp4_bmm_enabled or self.is_aiter_triton_fp8_bmm_enabled:
            return None

        num_heads, num_tokens, nope_dim = q_nope.shape
        output_dtype = self._fused_mla_query_output_dtype
        weight: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        runner: Callable[..., torch.Tensor]
        if getattr(self, "_use_b12x_absorb_bmm", False):
            weight = self._b12x_absorb_uk_rhs
            if not can_implement_mxfp8_mla_query(
                num_heads=num_heads,
                max_m=num_tokens,
                nope_dim=nope_dim,
                latent_dim=self.kv_lora_rank,
                output_dtype=output_dtype,
                device=q_nope.device,
            ):
                return None
            runner = run_mxfp8_mla_query
        else:
            weight = getattr(self, "W_UK_T", None)
            if not isinstance(weight, torch.Tensor) or not can_implement_bf16_mla_query(
                num_heads=num_heads,
                max_m=num_tokens,
                nope_dim=nope_dim,
                latent_dim=self.kv_lora_rank,
                output_dtype=output_dtype,
                device=q_nope.device,
            ):
                return None
            runner = run_bf16_mla_query

        workspace_getter = getattr(self.impl, "get_fused_mla_query_output", None)
        if callable(workspace_getter):
            output = workspace_getter(num_tokens, num_heads, output_dtype)
            # DCP1 can return the final query workspace. DCP and padded
            # virtual-TP layouts return None and use a graph-owned local query
            # tensor that their established gather/copy path consumes.
            if output is None:
                output = torch.empty(
                    (
                        num_tokens,
                        num_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=output_dtype,
                    device=q_nope.device,
                )
        else:
            output = torch.empty(
                (
                    num_tokens,
                    num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=output_dtype,
                device=q_nope.device,
            )
        runner(
            q_nope,
            weight,
            q_pe,
            self._q_scale,
            output,
        )
        return output

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
        quant_scale_ue8m0: bool | None = None,
        quant_col_major: bool | None = None,
        quant_tma_aligned: bool | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        quant_key = _detect_output_quant_key(
            output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
        )
        if quant_key is not None:
            # The fusion pass has allocated output with quantized dtype
            # (FP8 or uint8 for FP4). We can't write into it directly,
            # so we swap in a temp buffer for computation, then quantize
            # into the real output at the end.
            # NOTE(carlyou): this is temporary until kernels support fp8 output
            quant_output = output
            output = torch.empty(
                output.shape[0],
                self.num_heads * self.v_head_dim,
                dtype=q.dtype,
                device=output.device,
            )

        if attn_metadata is None:
            if not self.impl.is_sparse:
                # During the profile run try to simulate to worse case output
                # size for `self.kv_b_proj(kv_c_normed)` in
                # `_compute_prefill_context` since this can be large. Sparse
                # MLA never takes the dense chunked-prefill context path, so
                # skip the allocation to keep the profile peak accurate.
                _ = torch.empty(
                    (
                        self.chunked_prefill_workspace_size,
                        self.num_heads,
                        self.qk_nope_head_dim + self.v_head_dim,
                    ),
                    device=k_c_normed.device,
                    dtype=k_c_normed.dtype,
                )
            else:
                profile_workspace_bytes = self._get_sparse_memory_profile_bytes()
                # This synthetic allocation models eager runtime workspace for
                # KV sizing. attn_metadata is also None during CUDA graph
                # capture, where allocating it again only consumes capture-time
                # memory and can OOM after KV cache allocation.
                if _should_allocate_sparse_profile_workspace(profile_workspace_bytes):
                    _ = torch.empty(
                        (profile_workspace_bytes,),
                        device=k_c_normed.device,
                        dtype=torch.uint8,
                    )
                    logger.info_once(
                        "Including %.2f MiB of B12X sparse DCP transient "
                        "memory in the profile peak",
                        profile_workspace_bytes / (1 << 20),
                    )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            if quant_key is not None:
                return quant_output.fill_(0)
            return output.fill_(0)

        fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)

        num_actual_toks = attn_metadata.num_actual_tokens
        if self.use_pcp and self.impl.dcp_world_size > 1 and quant_key is not None:
            raise NotImplementedError(
                "MRV2 MLA PCP+DCP does not support fused output quantization yet."
            )

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        if q_dcp_replicated is not None:
            q_dcp_replicated = q_dcp_replicated[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        if fp8_attention and self.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla"):
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens
        is_sparse_impl = self.impl.is_sparse

        if self.impl.is_sparse and num_mha_tokens > 0:
            prefill = getattr(attn_metadata, "prefill", None)
            use_dense_mha = getattr(prefill, "use_dense_mha", False)
            prefill_max_seq_len = attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
            use_masked_mha = (
                self.prefill_backend is not None
                and self.impl.masked_mha_available  # type: ignore[attr-defined]
                and self.impl.dcp_world_size <= 1
                and prefill is not None
                and _use_masked_mha(
                    backend_name=self.attn_backend.get_name(),
                    tensor_parallel_size=self._vllm_config.parallel_config.tensor_parallel_size,
                    query_len=prefill.max_query_len,
                    seq_len=prefill_max_seq_len,
                )
                and self.impl.masked_mha_workspace_fits(prefill)  # type: ignore[attr-defined]
            )
            use_mha = (use_dense_mha or use_masked_mha) and not (
                self._vllm_config.attention_config.sparse_mla_force_mqa
            )
            if not use_mha:
                num_mqa_tokens = q.size(0)
                num_mha_tokens = 0

        ondemand_w_uv_capable = (
            getattr(self, "dcp_project_before_merge", False)
            and self.impl.dcp_world_size > 1
            and self.impl.is_sparse
            and getattr(self.impl, "supports_dcp_project_before_merge", False)
            and (
                (hasattr(self, "W_UV") and self.W_UV.dtype == torch.bfloat16)
                or getattr(self, "_use_b12x_absorb_bmm", False)
            )
        )
        project_before_merge_min_tokens = getattr(
            self,
            "dcp_project_before_merge_min_prefill_tokens",
            1024,
        )
        use_ondemand_w_uv = (
            ondemand_w_uv_capable
            and attn_metadata.max_query_len > 1
            and num_mqa_tokens > project_before_merge_min_tokens
        )
        mha_use_quant_output = (
            quant_key is not None
            and self.prefill_backend is not None
            and self.prefill_backend.supports_quant_output(quant_key)
            and (
                not self.impl.is_sparse
                or attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
                <= attn_metadata.topk_tokens  # type: ignore[attr-defined]
            )
            and attn_metadata is not None
            and attn_metadata.prefill is not None
            and attn_metadata.prefill.chunked_context is None
            and self.impl.dcp_world_size <= 1
        )

        if num_mha_tokens > 0:
            if mha_use_quant_output:
                mha_output = quant_output
                mha_output_scale = output_scale
            else:
                mha_output = output
                mha_output_scale = None

            self.impl.forward_mha(  # type: ignore[attr-defined]
                q[num_mqa_tokens:],
                k_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                kv_cache,
                attn_metadata,
                self._k_scale,
                output=mha_output[num_mqa_tokens:num_actual_toks],
                output_scale=mha_output_scale,
            )

        if num_mqa_tokens > 0:
            if q_dcp_replicated is not None:
                mqa_q = q_dcp_replicated[:num_mqa_tokens]
                qrep_decode = True
            else:
                mqa_q = q[:num_mqa_tokens]
                qrep_decode = False
            mqa_output_slice = output[:num_mqa_tokens]

            mqa_q_nope, mqa_q_pe = mqa_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            mqa_q_nope = mqa_q_nope.transpose(0, 1)

            if self.q_pad_num_heads is not None:
                B, N, L = mqa_q_pe.shape
                mqa_pe_padded = mqa_q_pe.new_empty((B, self.q_pad_num_heads, L))
                mqa_pe_padded.resize_((B, N, L))
                mqa_pe_padded.copy_(mqa_q_pe)
                mqa_q_pe = mqa_pe_padded

            fused_mqa_q = self._try_fused_mla_query(mqa_q_nope, mqa_q_pe)

            if fused_mqa_q is not None:
                mqa_q = fused_mqa_q
            elif self.is_aiter_triton_fp4_bmm_enabled:
                from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

                mqa_ql_nope = batched_gemm_a16wfp4(
                    mqa_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    transpose_bm=True,
                    prequant=True,
                    y_scale=self._q_scale if fp8_attention else None,
                )
            elif self.is_aiter_triton_fp8_bmm_enabled:
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                mqa_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                    mqa_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    transpose_bm=True,
                )
            else:
                # Pads the head_dim if necessary (for the underlying kernel)
                N, B, P = mqa_q_nope.shape
                use_b12x_absorb_bmm = getattr(self, "_use_b12x_absorb_bmm", False)
                W_UK_T = None
                if use_b12x_absorb_bmm:
                    assert not qrep_decode, (
                        "B12X absorbed BMM does not support DCP-replicated queries."
                    )
                    L = self.kv_lora_rank
                else:
                    W_UK_T = self.W_UK_T_dcp_qrep if qrep_decode else self.W_UK_T
                    assert W_UK_T is not None
                    _, _, L = W_UK_T.shape

                if self.q_pad_num_heads is not None:
                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                    mqa_ql_nope.resize_((N, B, L))
                else:
                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                if use_b12x_absorb_bmm:
                    if B <= _B12X_ABSORB_BMM_MAX_M:
                        run_b12x_mxfp8_bmm(
                            mqa_q_nope,
                            self._b12x_absorb_uk_rhs,
                            mqa_ql_nope,
                            b_major="n",
                        )
                    else:
                        _run_mla_query_bmm(
                            mqa_q_nope,
                            self._dequant_b12x_absorbed_pair()[0],
                            mqa_ql_nope,
                            use_safe_op=self.use_safe_mla_query_bmm,
                        )
                else:
                    assert W_UK_T is not None
                    _run_mla_query_bmm(
                        mqa_q_nope,
                        W_UK_T,
                        mqa_ql_nope,
                        use_safe_op=self.use_safe_mla_query_bmm,
                    )

                # Convert from (N, B, L) to (B, N, L)
                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

            if fused_mqa_q is None:
                if fp8_attention and self.impl.supports_quant_query_input:
                    assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
                    assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
                    mqa_q = self._decode_concat_quant_fp8_op(
                        mqa_ql_nope, mqa_q_pe, self._q_scale
                    )
                else:
                    mqa_q = (mqa_ql_nope, mqa_q_pe)
            dcp_use_b12x = False
            use_ag_rs_fallback = False
            project_before_merge = False
            workspace_gather_used = False
            ckv_gather_used = False
            if self.impl.dcp_world_size > 1:
                assert self.dcp_manager is not None
                ckv_gather_selector = getattr(
                    self.impl, "dcp_prefill_ckv_gather_eligible", None
                )
                ckv_gather_used = bool(
                    not self.use_pcp
                    and not qrep_decode
                    and callable(ckv_gather_selector)
                    and ckv_gather_selector(attn_metadata, num_mqa_tokens)
                )
                if not ckv_gather_used:
                    if not self.impl.can_return_lse_for_decode:
                        raise NotImplementedError(
                            f"{type(self.impl).__name__} cannot use DCP because it "
                            "does not return decode softmax LSE."
                        )
                    self.impl.need_to_return_lse_for_decode = True
                # A fused BF16 query is also a single tensor. Only an actual
                # FP8 query requires the backend's DCP quant-input contract.
                if (
                    fp8_attention
                    and isinstance(mqa_q, torch.Tensor)
                    and mqa_q.dtype == _FP8_DTYPE
                    and not getattr(self.impl, "supports_dcp_quant_query_input", False)
                ):
                    raise NotImplementedError(
                        f"{type(self.impl).__name__} does not declare support for "
                        "DCP with FP8 KV cache and pre-quantized query input."
                    )
                # Hybrid dispatch on the per-step token count. This is
                # CUDA-graph safe: under capture the branch sees the padded
                # capture size, so every graph bakes in one path, and eager
                # prefill re-evaluates per step. All DCP ranks run the same
                # batch, so the choice is uniform across the group.
                dcp_small_batch = (
                    self.dcp_a2a_max_tokens <= 0
                    or num_mqa_tokens <= self.dcp_a2a_max_tokens
                )
                use_ag_rs_fallback = (
                    self.dcp_a2a
                    and not self.use_pcp
                    and not dcp_small_batch
                    and self.dcp_a2a_large_backend == "ag_rs"
                )
                project_before_merge = (
                    use_ondemand_w_uv
                    and not self.use_pcp
                    and (not self.dcp_a2a or use_ag_rs_fallback)
                    and not ckv_gather_used
                )
                dcp_use_b12x = self.dcp_b12x and dcp_small_batch
                workspace_gather_eligible = (
                    not qrep_decode
                    and not ckv_gather_used
                    and _can_use_b12x_dcp_prefill_workspace(
                        enabled=envs.VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE,
                        project_before_merge=project_before_merge,
                        dcp_use_b12x=dcp_use_b12x,
                        num_tokens=num_mqa_tokens,
                        max_num_tokens=getattr(self.impl, "_max_batched", 0),
                        non_dbo_workspace=getattr(
                            self.impl, "dcp_workspace_non_dbo", False
                        ),
                        is_sparse_impl=is_sparse_impl,
                        backend_name=self.attn_backend.get_name(),
                        is_capturing=torch.cuda.is_current_stream_capturing(),
                    )
                )

                if self.use_pcp:
                    if self.impl.dcp_world_size > self.impl.pcp_world_size:
                        if isinstance(mqa_q, tuple):
                            mqa_q = torch.cat(mqa_q, dim=-1)
                        mqa_q = get_tp_group().all_gather(mqa_q, dim=1)
                else:
                    if (
                        isinstance(mqa_q, tuple)
                        and not workspace_gather_eligible
                        and not ckv_gather_used
                    ):
                        mqa_q = torch.cat(mqa_q, dim=-1)
                    if not qrep_decode:
                        if ckv_gather_used:
                            logger.info_once(
                                "Keeping local query heads for transient full-CKV "
                                "B12X sparse MLA prefill"
                            )
                        elif dcp_use_b12x:
                            mqa_q = dcp_b12x_all_gather_heads(
                                mqa_q,
                                get_dcp_group(),
                                max_batch_size=self.dcp_max_batch_size,
                                output_head_dim=self.kv_lora_rank,
                            )
                        elif workspace_gather_eligible:
                            workspace_gather = getattr(
                                self.impl,
                                "dcp_all_gather_query_in_workspace",
                                None,
                            )
                            if not getattr(
                                self.impl,
                                "supports_dcp_gather_query_in_workspace",
                                False,
                            ) or not callable(workspace_gather):
                                raise RuntimeError(
                                    f"{type(self.impl).__name__} does not support "
                                    "the enabled workspace DCP query gather"
                                )
                            mqa_q = workspace_gather(mqa_q)
                            workspace_gather_used = True
                            logger.info_once(
                                "Using borrowed B12X workspaces for sparse MLA "
                                "DCP prefill"
                            )
                        else:
                            assert self.dcp_manager.query_gather is not None
                            mqa_q = self.dcp_manager.query_gather(mqa_q)

            # call decode attn
            if not self.impl.is_sparse:
                assert attn_metadata.decode is not None
            if ckv_gather_used:
                ckv_setter = getattr(self.impl, "set_ckv_current_chunk_kv", None)
                if callable(ckv_setter):
                    ckv_setter(k_c_normed, k_pe)
            attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]

            # correct dcp attn_out with lse.
            if self.impl.dcp_world_size > 1 and not ckv_gather_used:
                if lse is None:
                    raise RuntimeError(
                        f"{type(self.impl).__name__} did not return decode "
                        "softmax LSE required by DCP."
                    )
                assert self.dcp_manager is not None
                seq_lens, query_start_loc = _get_dcp_batch_metadata(attn_metadata)
                valid_counts = None
                if project_before_merge:
                    valid_counts_tensor = getattr(
                        attn_metadata, "nsa_cache_seqlens", None
                    )
                    if (
                        not isinstance(valid_counts_tensor, torch.Tensor)
                        or valid_counts_tensor.ndim != 1
                        or valid_counts_tensor.numel() < num_mqa_tokens
                        or valid_counts_tensor.dtype != torch.int32
                        or valid_counts_tensor.device != attn_out.device
                    ):
                        raise RuntimeError(
                            "Projected DCP merge requires a one-dimensional "
                            f"int32 nsa_cache_seqlens tensor with at least "
                            f"{num_mqa_tokens} rows on {attn_out.device}."
                        )
                    valid_counts = valid_counts_tensor[:num_mqa_tokens]
                    if not valid_counts.is_contiguous():
                        raise RuntimeError(
                            "Projected DCP valid counts must be contiguous."
                        )
                    local_w_uv = (
                        self._dequant_b12x_absorbed_pair()[1].contiguous()
                        if getattr(self, "_use_b12x_absorb_bmm", False)
                        else self.W_UV.contiguous()
                    )
                    w_uv_dcp = get_dcp_group().all_gather(local_w_uv, dim=0)
                    expected_shape = (
                        self.num_heads * get_dcp_group().world_size,
                        self.kv_lora_rank,
                        self.v_head_dim,
                    )
                    if (
                        w_uv_dcp.shape != expected_shape
                        or w_uv_dcp.dtype != local_w_uv.dtype
                        or w_uv_dcp.device != local_w_uv.device
                        or not w_uv_dcp.is_contiguous()
                    ):
                        raise RuntimeError(
                            "Invalid rank-major DCP W_UV gather: expected "
                            f"contiguous {expected_shape} on "
                            f"{local_w_uv.device}/{local_w_uv.dtype}, got "
                            f"{tuple(w_uv_dcp.shape)} on "
                            f"{w_uv_dcp.device}/{w_uv_dcp.dtype}."
                        )
                    if workspace_gather_used:
                        workspace_project = getattr(
                            self.impl,
                            "dcp_project_before_merge_in_workspace",
                            None,
                        )
                        if not getattr(
                            self.impl,
                            "supports_dcp_project_before_merge_in_workspace",
                            False,
                        ) or not callable(workspace_project):
                            raise RuntimeError(
                                f"{type(self.impl).__name__} does not support "
                                "workspace DCP projection"
                            )
                        attn_out = workspace_project(attn_out, lse, w_uv_dcp)
                    else:
                        projected = attn_out.new_empty(
                            attn_out.shape[0], attn_out.shape[1], self.v_head_dim
                        )
                        self._v_up_proj_bmm_chunked(attn_out, projected, w_uv_dcp)
                        attn_out = projected
                if dcp_use_b12x:
                    attn_out = dcp_a2a_lse_reduce(
                        attn_out,
                        lse,
                        get_dcp_group(),
                        is_lse_base_on_e=self.impl.lse_base_on_e,
                        use_b12x=dcp_use_b12x,
                        b12x_max_batch_size=self.dcp_max_batch_size,
                        b12x_query_head_dim=(self.kv_lora_rank + self.qk_rope_head_dim),
                        seq_lens=seq_lens,
                        query_start_loc=query_start_loc,
                    )
                elif project_before_merge or use_ag_rs_fallback:
                    if project_before_merge:
                        sanitize_dcp_attn_empty_rows(attn_out, lse, valid_counts)
                    if workspace_gather_used:
                        workspace_output = getattr(
                            self.impl,
                            "dcp_reduce_scatter_output_in_workspace",
                            None,
                        )
                        if not getattr(
                            self.impl,
                            "supports_dcp_reduce_scatter_output_in_workspace",
                            False,
                        ) or not callable(workspace_output):
                            raise RuntimeError(
                                f"{type(self.impl).__name__} does not support "
                                "workspace DCP reduce-scatter output"
                            )
                        attn_out = cp_lse_ag_out_rs_into(
                            attn_out,
                            lse,
                            get_dcp_group(),
                            output_provider=workspace_output,
                            is_lse_base_on_e=self.impl.lse_base_on_e,
                            seq_lens=seq_lens,
                            query_start_loc=query_start_loc,
                        )
                    else:
                        attn_out = cp_lse_ag_out_rs(
                            attn_out,
                            lse,
                            get_dcp_group(),
                            is_lse_base_on_e=self.impl.lse_base_on_e,
                            seq_lens=seq_lens,
                            query_start_loc=query_start_loc,
                            head_major_output=True,
                        )
                else:
                    attn_out = self.dcp_manager.combine(
                        attn_out,
                        lse,
                        seq_lens=seq_lens,
                        query_start_loc=query_start_loc,
                    )
                if self.use_pcp:
                    attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)

            if project_before_merge:
                if workspace_gather_used:
                    expected_shape = (
                        num_mqa_tokens,
                        self.num_heads,
                        self.v_head_dim,
                    )
                    expected_stride = (
                        self.v_head_dim,
                        num_mqa_tokens * self.v_head_dim,
                        1,
                    )
                    if (
                        tuple(attn_out.shape) != expected_shape
                        or tuple(attn_out.stride()) != expected_stride
                        or not attn_out.movedim(0, 1).is_contiguous()
                    ):
                        raise RuntimeError(
                            "Workspace DCP reduce-scatter returned an invalid "
                            f"layout: shape={tuple(attn_out.shape)}, "
                            f"stride={tuple(attn_out.stride())}"
                        )
                    mqa_output_slice.view(expected_shape).copy_(attn_out)
                else:
                    mqa_output_slice.copy_(attn_out.reshape(mqa_output_slice.shape))
            else:
                self._v_up_proj(attn_out, out=mqa_output_slice)

        if quant_key is not None:
            quant_idx = num_mqa_tokens if mha_use_quant_output else num_actual_toks
            if quant_idx == 0:
                return quant_output
            actual = output[:quant_idx]
            if quant_key == kNvfp4Dynamic:
                # NVFP4: two FP4 values packed into one uint8
                assert output_block_scale is not None
                fp4_data, fp4_scales = ops.scaled_fp4_quant(actual, output_scale)
                quant_output[:quant_idx].copy_(fp4_data)
                output_block_scale[: fp4_scales.shape[0]].copy_(fp4_scales)
            elif quant_key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym):
                # Per-group FP8
                assert output_block_scale is not None
                assert quant_group_size is not None, (
                    "Group FP8 output quant requested but "
                    "quant_group_size not passed through custom op"
                )
                finfo = torch.finfo(_FP8_DTYPE)
                torch.ops._C.per_token_group_fp8_quant(
                    actual,
                    quant_output[:quant_idx],
                    output_block_scale[:quant_idx],
                    quant_group_size,
                    1e-10,  # eps
                    finfo.min,
                    finfo.max,
                    quant_scale_ue8m0,
                    quant_col_major,
                    quant_tma_aligned,
                )
            elif quant_key == kFp8StaticTensorSym:
                # Static FP8 quantization
                fp8_data, _ = self._quant_fp8_op(actual, output_scale)
                quant_output[:quant_idx].copy_(fp8_data)
            else:
                raise ValueError(f"Unsupported quant_key: {quant_key}")
            return quant_output

        if self.use_pcp and output_padded.shape[0] > num_actual_toks:
            output_padded[num_actual_toks:].zero_()
        return output_padded

    def _prepare_b12x_absorb_bmm(self, act_dtype: torch.dtype) -> bool:
        for name in ("_b12x_absorb_uk_rhs", "_b12x_absorb_uv_rhs"):
            if hasattr(self, name):
                delattr(self, name)
        if not _b12x_absorb_bmm_enabled():
            return False
        if (
            act_dtype != torch.bfloat16
            or self.is_aiter_triton_fp4_bmm_enabled
            or self.is_aiter_triton_fp8_bmm_enabled
        ):
            logger.warning_once(
                "VLLM_B12X_ABSORB_BMM=1 requires BF16 MLA projections; "
                "falling back to the materialized absorbed weights."
            )
            return False
        if getattr(self, "dcp_q_replicate", False):
            logger.warning_once(
                "VLLM_B12X_ABSORB_BMM=1 does not support DCP query "
                "replication; falling back to the gathered BF16 weights."
            )
            return False

        weight = getattr(self.kv_b_proj, "weight", None)
        weight_scale = getattr(self.kv_b_proj, "weight_scale", None)
        head_stride = self.qk_nope_head_dim + self.v_head_dim
        expected_weight_shape = (
            self.num_heads * head_stride,
            self.kv_lora_rank,
        )
        expected_scale_shape = (
            self.num_heads * head_stride,
            self.kv_lora_rank // 32,
        )
        if (
            not isinstance(weight, torch.Tensor)
            or not isinstance(weight_scale, torch.Tensor)
            or weight.dtype != torch.float8_e4m3fn
            or weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or tuple(weight.shape) != expected_weight_shape
            or tuple(weight_scale.shape) != expected_scale_shape
            or not weight.is_contiguous()
            or not weight_scale.is_contiguous()
        ):
            logger.warning_once(
                "VLLM_B12X_ABSORB_BMM=1 requires a contiguous ModelOpt MXFP8 "
                "kv_b_proj pack; falling back to the materialized absorbed weights."
            )
            return False

        values = weight.view(self.num_heads, head_stride, self.kv_lora_rank)
        scales = weight_scale.view(
            self.num_heads,
            head_stride,
            self.kv_lora_rank // 32,
        )
        uk_rhs = (
            values[:, : self.qk_nope_head_dim, :],
            scales[:, : self.qk_nope_head_dim, :],
        )
        uv_rhs = (
            values[:, self.qk_nope_head_dim :, :],
            scales[:, self.qk_nope_head_dim :, :],
        )
        uk_supported = can_implement_b12x_mxfp8_bmm(
            batch=self.num_heads,
            max_m=_B12X_ABSORB_BMM_MAX_M,
            n=self.kv_lora_rank,
            k=self.qk_nope_head_dim,
            b_major="n",
            device=weight.device,
        )
        uv_supported = can_implement_b12x_mxfp8_bmm(
            batch=self.num_heads,
            max_m=_B12X_ABSORB_BMM_MAX_M,
            n=self.v_head_dim,
            k=self.kv_lora_rank,
            b_major="k",
            device=weight.device,
        )
        if not (uk_supported and uv_supported):
            logger.warning_once(
                "VLLM_B12X_ABSORB_BMM=1 but the MLA geometry is outside the "
                "b12x.gemm.bmm envelope; falling back to the materialized "
                "absorbed weights."
            )
            return False

        self._b12x_absorb_uk_rhs = uk_rhs
        self._b12x_absorb_uv_rhs = uv_rhs
        for name in ("W_UK_T", "W_UV"):
            if hasattr(self, name):
                delattr(self, name)
        logger.info_once(
            "Serving MLA absorbed projections directly from the B12X MXFP8 pack."
        )
        return True

    def _dequant_b12x_absorbed_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.kv_b_proj.weight
        weight_scale = self.kv_b_proj.weight_scale
        if weight_scale.dtype != torch.float8_e8m0fnu:
            weight_scale = weight_scale.view(torch.float8_e8m0fnu)
        dequant = weight.to(torch.bfloat16) * weight_scale.to(
            torch.bfloat16
        ).repeat_interleave(32, dim=-1)
        kv_b = dequant.T.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = kv_b.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)

    def _process_materialized_absorbed_weights(self, act_dtype: torch.dtype):
        pre_w_uv = pre_w_uk_t = None
        if not (
            self.is_aiter_triton_fp4_bmm_enabled or self.is_aiter_triton_fp8_bmm_enabled
        ):
            # Seat persistent weights before the transient dequantization scratch.
            pre_w_uv, pre_w_uk_t = _preallocate_absorbed_mla_weights(self, act_dtype)

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T

        if self.dcp_q_replicate:
            # qrep wired here: validate unsupported decode backends once.
            assert self.q_pad_num_heads in (None, self.num_heads), (
                "DCP query replication is unsupported on head-padding MLA "
                "backends (q_pad_num_heads)."
            )
            if (
                self.is_aiter_triton_fp4_bmm_enabled
                or self.is_aiter_triton_fp8_bmm_enabled
            ):
                raise NotImplementedError(
                    "DCP query replication is not implemented for the aiter "
                    "FP4/FP8 MLA BMM paths."
                )

        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
        if self.is_aiter_triton_fp4_bmm_enabled:
            from vllm.model_executor.layers.quantization.quark.utils import (
                quark_quantize_weight_to_mxfp4,
            )

            self.W_K, self.W_K_scale = quark_quantize_weight_to_mxfp4(W_UK)
            # Convert from (L, N, P) to (N, L, P)
            self.W_K = self.W_K.transpose(0, 1)
            self.W_K_scale = self.W_K_scale.transpose(0, 1)

            self.W_V, self.W_V_scale = quark_quantize_weight_to_mxfp4(
                W_UV.permute(1, 2, 0)
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=current_platform.fp8_dtype()
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=current_platform.fp8_dtype()
            )

            # The kernel operates on non-padded inputs. Hence, pre-compiling
            # triton kernel to avoid runtime compilation for unseen batch sizes
            # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
            # On DS-R1, this step adds roughly 50s to the model loading time.
            max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            pre_compilation_list = list(range(1, max_batch_size + 1))
            if is_global_first_rank():
                pre_compilation_list = tqdm(
                    pre_compilation_list,
                    desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                    total=max_batch_size,
                )

            for m in pre_compilation_list:
                x = torch.empty(
                    (self.W_K.shape[0], m, self.W_K.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_K.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            w_uv = W_UV.transpose(0, 1)
            if pre_w_uv is not None:
                pre_w_uv.copy_(w_uv)
                w_uv = pre_w_uv
            replace_parameter(self, "W_UV", w_uv, prefer_copy=True)
            # Convert from (L, N, P) to (N, P, L)
            w_uk_t = W_UK.permute(1, 2, 0)
            if pre_w_uk_t is not None:
                pre_w_uk_t.copy_(w_uk_t)
                w_uk_t = pre_w_uk_t
            replace_parameter(self, "W_UK_T", w_uk_t, prefer_copy=True)
            if self.dcp_q_replicate:
                self.W_UK_T_dcp_qrep = get_dcp_group().all_gather(
                    self.W_UK_T.contiguous(), dim=0
                )

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self._use_b12x_absorb_bmm = self._prepare_b12x_absorb_bmm(act_dtype)
        self._fused_mla_query_output_dtype = (
            current_platform.fp8_dtype()
            if is_quantized_kv_cache(self.kv_cache_dtype)
            and self.impl.supports_quant_query_input
            else torch.bfloat16
        )
        if not self._use_b12x_absorb_bmm:
            self._process_materialized_absorbed_weights(act_dtype)
            return

        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        common_kwargs = dict(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=self.kv_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
        if self.sliding_window is not None:
            return SlidingWindowMLASpec(
                **common_kwargs,
                sliding_window=self.sliding_window,
                non_causal_multi_token_decode=self.non_causal_multi_token_decode,
            )

        layer_id = _extract_single_layer_index(self.layer_name)
        num_hidden_layers = getattr(
            vllm_config.model_config.hf_config, "num_hidden_layers", None
        )
        shard_draft = os.environ.get("VLLM_DCP_SHARD_DRAFT", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        dcp_replicated = (
            not shard_draft
            and layer_id is not None
            and num_hidden_layers is not None
            and layer_id >= int(num_hidden_layers)
        )
        model_type = getattr(vllm_config.model_config.hf_config, "model_type", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        target_model_config = getattr(speculative_config, "target_model_config", None)
        target_model_type = (
            getattr(target_model_config.hf_config, "model_type", None)
            if target_model_config is not None
            else None
        )
        glm_model_or_mtp = bool(
            model_type == "glm_moe_dsa"
            or (model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa")
        )
        glm_fp8_rope = bool(
            os.environ.get("KV_FP8_ROPE", "0") == "1"
            and self.kv_cache_dtype == "nvfp4_ds_mla"
            and glm_model_or_mtp
        )
        return MLAAttentionSpec(
            **common_kwargs,
            non_causal_multi_token_decode=self.non_causal_multi_token_decode,
            model_version="glm_fp8_rope" if glm_fp8_rope else None,
            dcp_replicated=dcp_replicated,
        )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_heads, self.v_head_dim)
        if self.is_aiter_triton_fp4_bmm_enabled:
            out = rocm_aiter_ops.batched_gemm_a16wfp4(
                x,
                self.W_V,
                self.W_V_scale,
                out,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
            )
            x = out.view(-1, self.num_heads * self.v_head_dim)
        elif self.is_aiter_triton_fp8_bmm_enabled:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            x = rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
            )
        elif getattr(self, "_use_b12x_absorb_bmm", False):
            if x.shape[1] <= _B12X_ABSORB_BMM_MAX_M:
                run_b12x_mxfp8_bmm(
                    x,
                    self._b12x_absorb_uv_rhs,
                    out.transpose(0, 1),
                    b_major="k",
                )
            else:
                torch.bmm(
                    x,
                    self._dequant_b12x_absorbed_pair()[1],
                    out=out.transpose(0, 1),
                )
        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))

    def _v_up_proj_bmm(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> None:
        """Project BF16 DCP partials with rank-major gathered W_UV."""
        if x.ndim != 3 or out.ndim != 3 or w_uv.ndim != 3:
            raise ValueError("DCP projection expects rank-three tensors.")
        num_tokens, num_heads, latent_dim = x.shape
        expected_out_shape = (num_tokens, num_heads, self.v_head_dim)
        expected_weight_shape = (
            num_heads,
            self.kv_lora_rank,
            self.v_head_dim,
        )
        if (
            latent_dim != self.kv_lora_rank
            or out.shape != expected_out_shape
            or w_uv.shape != expected_weight_shape
        ):
            raise ValueError(
                "DCP projection geometry mismatch: "
                f"x={tuple(x.shape)}, out={tuple(out.shape)}, "
                f"w_uv={tuple(w_uv.shape)}."
            )
        if (
            x.dtype != torch.bfloat16
            or out.dtype != x.dtype
            or w_uv.dtype != x.dtype
            or out.device != x.device
            or w_uv.device != x.device
            or not w_uv.is_contiguous()
        ):
            raise ValueError(
                "DCP projection requires contiguous BF16 weights and matching "
                "BF16 inputs/outputs on one device."
            )
        x_head_major = x.transpose(0, 1).contiguous()
        projected_head_major = torch.empty(
            (num_heads, num_tokens, self.v_head_dim),
            dtype=out.dtype,
            device=out.device,
        )
        torch.bmm(x_head_major, w_uv, out=projected_head_major)
        out.copy_(projected_head_major.transpose(0, 1))

    def _v_up_proj_bmm_chunked(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> None:
        """Bound temporary BF16 DCP projection storage to 144 MiB."""
        if x.ndim != 3 or out.ndim != 3 or w_uv.ndim != 3:
            raise ValueError(
                "DCP projection expects rank-three tensors: "
                f"x={tuple(x.shape)}, out={tuple(out.shape)}, "
                f"w_uv={tuple(w_uv.shape)}."
            )
        num_tokens, num_heads, latent_dim = x.shape
        if latent_dim != self.kv_lora_rank or w_uv.shape[0] != num_heads:
            raise ValueError(
                "DCP projection geometry mismatch: "
                f"x={tuple(x.shape)}, w_uv={tuple(w_uv.shape)}."
            )

        temp_budget_bytes = 144 * 1024 * 1024
        temp_bytes_per_token = (
            num_heads * (self.kv_lora_rank + self.v_head_dim) * x.element_size()
        )
        max_chunk_tokens = max(1, temp_budget_bytes // temp_bytes_per_token)
        for start in range(0, num_tokens, max_chunk_tokens):
            end = min(start + max_chunk_tokens, num_tokens)
            self._v_up_proj_bmm(x[start:end], out[start:end], w_uv)


def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )
    if layer_slot_mapping is not None:
        kv_c_normed, k_pe, layer_slot_mapping = maybe_gather_mla_latent_cache_inputs(
            kv_c_normed,
            k_pe,
            layer_slot_mapping,
            attn_metadata.num_decode_tokens if attn_metadata is not None else None,
            attn_layer.use_pcp,
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            kv_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


direct_register_custom_op(
    op_name="unified_mla_kv_cache_update",
    op_func=unified_mla_kv_cache_update,
    fake_impl=unified_mla_kv_cache_update_fake,
)


@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    layer.forward_impl(
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
        quant_group_size=quant_group_size,
        quant_scale_ue8m0=quant_scale_ue8m0,
        quant_col_major=quant_col_major,
        quant_tma_aligned=quant_tma_aligned,
        q_dcp_replicated=q_dcp_replicated,
    )


def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_mla_attention_with_output",
    op_func=unified_mla_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_mla_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.flexible_layout,),
)


class QueryLenSupport(Enum):
    """Defines the level of query length support for an attention backend's
    decode pipeline.

    - SINGLE_ONLY: Decode pipeline only supports single-token queries
                   (query_len=1)
    - UNIFORM: Decode pipeline supports uniform multi-token queries
               (all requests must have same query_len > 1)
    - VARLEN: Decode pipeline supports variable-length queries
              (mixed query lengths in same batch)
    """

    SINGLE_ONLY = "single_only"
    UNIFORM = "uniform"
    VARLEN = "varlen"


def dynamic_per_batched_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


@CustomOp.register(
    "mla_decode_concat_quant_fp8",
    dynamic_arg_dims={"decode_ql_nope": 0, "decode_q_pe": 0},
)
class _DecodeConcatQuantFP8(QuantFP8):
    """
    QuantFP8 variant that concatenates decode_ql_nope and decode_q_pe before
    quantization. When disabled, forward_native is compiled via torch.compile,
    fusing cat/reshape/quant/view together.
    """

    def _make_forward(quant_fn):  # noqa: N805
        """Factory to create forward methods that concat before quantization."""

        def forward(
            self,
            decode_ql_nope: torch.Tensor,
            decode_q_pe: torch.Tensor,
            scale: torch.Tensor,
            scale_ub: torch.Tensor | None = None,
        ) -> torch.Tensor:
            decode_q0 = torch.cat((decode_ql_nope, decode_q_pe), dim=-1)
            decode_q_flat = decode_q0.reshape(decode_q0.shape[0], -1)
            decode_q, _ = quant_fn(self, decode_q_flat, scale, scale_ub)
            return decode_q.view(decode_q0.shape)

        return forward

    forward_native = _make_forward(QuantFP8.forward_native)  # type: ignore[arg-type]
    forward_cuda = _make_forward(QuantFP8.forward_cuda)  # type: ignore[arg-type]
    forward_hip = _make_forward(QuantFP8.forward_hip)  # type: ignore[arg-type]


class MLACommonBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # Default to identity permutation to signal cross-layer allocation
            # is unsupported. Each MLA backend must opt in to support cross-layer
            # allocation by overriding this method.
            return (0, 1, 2, 3)
        return (0, 1, 2)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [320, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True


@dataclass
class MLACommonPrefillMetadata:
    """Prefill Specific Metadata"""

    @dataclass
    class ContextChunk:
        """One workspace-sized slice of paged context for a run of prefills."""

        index: int
        request_slice: slice
        token_slice: slice
        continuation_token_end: int
        is_continuation: bool
        num_context_tokens: int
        query_start_loc: torch.Tensor
        max_query_len: int
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        max_seq_len: int
        seq_lens: torch.Tensor
        token_to_seq: torch.Tensor

        # for mla DCP
        padded_local_seq_lens: list[int] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        padded_local_cu_seq_lens: torch.Tensor | None = None
        padded_local_token_to_seq: torch.Tensor | None = None
        num_local_context_tokens: int = 0
        local_starts: list[int] | None = None

        @property
        def num_requests(self) -> int:
            return self.request_slice.stop - self.request_slice.start

    @dataclass
    class ChunkedContextMetadata:
        context_lens: torch.Tensor
        workspace: torch.Tensor
        chunks: "list[MLACommonPrefillMetadata.ContextChunk]"
        context_lens_list: list[int]
        empty_token_slices: list[slice]
        dcp_manager: MLADCPManager | None = None
        direct_dcp_kv_gather: bool = False

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: ChunkedContextMetadata | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    prefill_backend: MLAPrefillBackend | None = None
    query_lens_cpu: torch.Tensor | None = None
    use_dense_mha: bool = False
    topk_mask_workspace: torch.Tensor | None = None


@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None


D = TypeVar("D", bound=MLACommonDecodeMetadata)


@dataclass
class MLACommonMetadata(AttentionMetadata, Generic[D]):
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    causal: bool = True

    # The dimension of the attention heads
    head_dim: int | None = None

    prefill: MLACommonPrefillMetadata | None = None
    decode: D | None = None

    def __post_init__(self):
        if self.head_dim is not None and not MLACommonBackend.supports_head_size(
            self.head_dim
        ):
            raise ValueError(f"Head dimension {self.head_dim} is not supported by MLA.")


M = TypeVar("M", bound=MLACommonMetadata)
A = TypeVar("A", bound=AttentionMetadata)


@dataclass
class MLADims:
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


def get_mla_dims(model_config: ModelConfig) -> MLADims:
    hf_text_config = model_config.hf_text_config

    # Check if this is a DeepseekV4 config (uses unified head_dim + rope_head_dim)
    if hasattr(hf_text_config, "compress_ratios"):
        # DeepseekV4 style config: unified head_dim with rope_head_dim
        head_dim = hf_text_config.head_dim
        rope_head_dim = hf_text_config.qk_rope_head_dim
        return MLADims(
            q_lora_rank=hf_text_config.q_lora_rank,
            kv_lora_rank=head_dim,
            qk_nope_head_dim=head_dim - rope_head_dim,
            qk_rope_head_dim=rope_head_dim,
            v_head_dim=head_dim,
        )

    # DeepseekV2/V3 style config
    return MLADims(
        q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
        kv_lora_rank=hf_text_config.kv_lora_rank,
        qk_nope_head_dim=hf_text_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_text_config.qk_rope_head_dim,
        v_head_dim=hf_text_config.v_head_dim,
    )


_DSV32_MASKED_MHA_THRESHOLDS: dict[str, dict[int, tuple[int | None, ...]]] = {
    "FLASHMLA_SPARSE": {
        1: (1536, 4096, None, None, None),
        2: (512, 1024, 4096, None, None),
        4: (512, 1024, 1536, 8192, None),
        8: (512, 512, 1024, 2048, 16384),
    },
    "FLASHINFER_MLA_SPARSE": {
        8: (512, 1024, 1024, 2048, 32768),
    },
}
_DSV32_SEQ_LEN_BUCKETS = (2048, 4096, 8192, 16384, 32768)


def _use_masked_mha(
    *,
    backend_name: str,
    tensor_parallel_size: int,
    query_len: int,
    seq_len: int,
) -> bool:
    thresholds = _DSV32_MASKED_MHA_THRESHOLDS.get(backend_name, {}).get(
        tensor_parallel_size
    )
    if thresholds is None:
        return False
    for bucket_idx, bucket_seq_len in enumerate(_DSV32_SEQ_LEN_BUCKETS):
        if seq_len <= bucket_seq_len:
            min_query_len = thresholds[bucket_idx]
            return min_query_len is not None and query_len >= min_query_len
    return False


@functools.cache
def backend_supports_prefill_query_quantization() -> bool:
    """Check if the selected MLA prefill backend supports query quantization.

    Currently supported backends:
    - FlashInfer
    - TRT-LLM Ragged

    Not supported:
    - FlashAttention (FA3/FA4)
    - Non-GB200 devices (FP8 prefill requires device capability 100)
    """
    # FP8 prefill query quantization requires GB200 (device capability 100)
    # for the necessary FP8 kernels at the moment.
    if not current_platform.is_device_capability_family(100):
        return False

    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    vllm_config = get_current_vllm_config()
    backend_cls = get_mla_prefill_backend(vllm_config)
    return backend_cls.get_name() in (
        "FLASHINFER",
        "TRTLLM_RAGGED",
        "TOKENSPEED_MLA",
    )


@dataclass
class _ContextChunkPlan:
    """Request-space layout of one context chunk, before tensors are built."""

    request_start: int
    request_end: int
    # Per-request context offset and row count, in request order.
    starts: list[int]
    seq_lens: list[int]
    is_continuation: bool


def plan_mla_context_chunks(
    context_lens: list[int],
    row_budget: int,
    max_context_chunk: int,
    split_alignment: int,
    padded_rows: Callable[[int], int],
) -> list[_ContextChunkPlan]:
    """Pack per-request contexts into workspace-sized chunks."""
    assert max_context_chunk > 0
    assert split_alignment > 0

    def aligned_split_len(start: int, remaining: int, available: int) -> int:
        max_split = min(
            max_context_chunk,
            round_down(remaining - 1, split_alignment),
        )
        low, high = 0, max_split // split_alignment
        while low < high:
            mid = (low + high + 1) // 2
            length = mid * split_alignment
            rows = padded_rows(start + length) - padded_rows(start)
            if rows <= available:
                low = mid
            else:
                high = mid - 1
        return low * split_alignment

    plans: list[_ContextChunkPlan] = []
    num_requests = len(context_lens)
    request = 0
    start = 0
    while request < num_requests:
        context_len = context_lens[request]
        if context_len == 0:
            assert start == 0
            request += 1
            continue

        request_start = request
        starts: list[int] = []
        seq_lens: list[int] = []
        rows = 0
        while request < num_requests and context_lens[request] > 0:
            remaining = context_lens[request] - start
            request_rows = padded_rows(start + remaining) - padded_rows(start)
            if rows + request_rows > row_budget:
                split_len = aligned_split_len(start, remaining, row_budget - rows)
                if split_len > 0:
                    starts.append(start)
                    seq_lens.append(split_len)
                    rows += padded_rows(start + split_len) - padded_rows(start)
                    start += split_len
                break
            rows += request_rows
            starts.append(start)
            seq_lens.append(remaining)
            request += 1
            start = 0
        assert seq_lens
        plans.append(
            _ContextChunkPlan(
                request_start=request_start,
                request_end=request_start + len(seq_lens),
                starts=starts,
                seq_lens=seq_lens,
                is_continuation=starts[0] > 0,
            )
        )
    return plans


def _flat_int32(values: list[int] | np.ndarray) -> torch.Tensor:
    """Pinned int32 CPU tensor backing one concatenated per-chunk field."""
    return np_to_pinned_tensor(np.asarray(values, dtype=np.int32))


def align_mla_chunked_context_workspace_size(
    vllm_config: VllmConfig,
    workspace_size: int,
) -> int:
    parallel_config = vllm_config.parallel_config
    alignment = vllm_config.cache_config.block_size
    if parallel_config.decode_context_parallel_size > 1:
        alignment = lcm(
            alignment,
            parallel_config.decode_context_parallel_size
            * parallel_config.cp_kv_cache_interleave_size,
        )
    workspace_size = max(
        workspace_size,
        vllm_config.scheduler_config.max_num_seqs * alignment,
    )
    return round_up(workspace_size, alignment)


def build_mla_chunked_context_metadata(
    *,
    context_lens_cpu: torch.Tensor,
    prefill_query_start_loc_cpu: torch.Tensor,
    chunked_prefill_workspace: torch.Tensor,
    chunked_prefill_workspace_size: int,
    block_size: int,
    align_chunk_to_block: bool,
    device: torch.device,
    dcp_world_size: int,
    dcp_local_block_size: int,
    dcp_virtual_block_size: int,
    dcp_manager: MLADCPManager | None = None,
    direct_dcp_kv_gather: bool = False,
) -> "MLACommonPrefillMetadata.ChunkedContextMetadata | None":
    """Build chunked-context metadata for an MLA prefill.

    Shared by dense and sparse builders. Packs the prefill contexts into a flat
    list of workspace-sized per-request chunks and, under DCP, plans the
    per-rank interleaved local chunks the all-gather reduction consumes.

    Args:
        context_lens_cpu: Per-prefill context length (seq_len - query_len).
        prefill_query_start_loc_cpu: Prefill query cumulative offsets (0-based).
        chunked_prefill_workspace: Scratch buffer the context gather writes to.
        chunked_prefill_workspace_size: Row capacity of the workspace.
        block_size: KV cache page size for chunk-start alignment.
        align_chunk_to_block: Round the chunk size down to ``block_size``.
        device: Target device for the returned tensors.
        dcp_world_size: Decode-context-parallel world size (1 if disabled).
        dcp_local_block_size: Per-rank interleave block size for DCP.
        dcp_virtual_block_size: ``dcp_local_block_size * dcp_world_size``.
        dcp_manager: Shared MLA DCP collective manager.
        direct_dcp_kv_gather: Gather context KV through the process DCP group
            when the decode backend owns its query and output collectives.

    Returns:
        The chunked-context metadata, or None when no prefill has any context.
    """
    # NOTE: it is recommended you read the `Chunked Prefill` section in the
    # comment at the top of the file before trying to understand this code.
    context_lens = context_lens_cpu.tolist()
    if max(context_lens, default=0) <= 0:
        return None

    # The `gather_and_maybe_dequant_cache` kernel cannot handle chunk starts
    # that are not aligned to block_size, so a split request advances in
    # block-aligned steps.
    chunk_alignment = block_size if align_chunk_to_block else 1
    if dcp_world_size > 1:
        # Each rank gathers only its own shard, so a chunk's workspace cost is
        # 1/dcp_world_size of its context rows, rounded up to the interleave
        # block size. Sub-chunks are additionally aligned to the virtual block
        # size so that every rank's local start stays interleave-aligned.
        row_budget = chunked_prefill_workspace_size // dcp_world_size
        chunk_alignment = math.lcm(chunk_alignment, dcp_virtual_block_size)

        def padded_rows(rows: int) -> int:
            return cdiv(rows, dcp_virtual_block_size) * dcp_local_block_size
    else:
        row_budget = chunked_prefill_workspace_size

        def padded_rows(rows: int) -> int:
            return rows

    max_context_chunk = round_down(chunked_prefill_workspace_size, chunk_alignment)
    assert max_context_chunk > 0, (
        f"chunked prefill workspace ({chunked_prefill_workspace_size} rows) is "
        f"smaller than the context chunk alignment ({chunk_alignment} rows)"
    )

    plans = plan_mla_context_chunks(
        context_lens,
        row_budget,
        max_context_chunk,
        chunk_alignment,
        padded_rows,
    )

    query_start_loc = prefill_query_start_loc_cpu.tolist()
    empty_token_slices = [
        slice(query_start_loc[i], query_start_loc[i + 1])
        for i, context_len in enumerate(context_lens)
        if context_len == 0
    ]

    use_dcp = dcp_world_size > 1
    local_context_lens_allranks = (
        get_dcp_local_seq_lens(
            context_lens_cpu, dcp_world_size, None, dcp_local_block_size
        ).tolist()
        if use_dcp
        else None
    )

    # Concatenate the per-chunk fields so each one costs a single
    # host-to-device transfer, and remember where every chunk's slice lands.
    starts_flat: list[int] = []
    seq_lens_flat: list[int] = []
    cu_seq_lens_flat: list[int] = []
    cu_seqlens_q_flat: list[int] = []
    token_to_seq_parts: list[np.ndarray] = []
    padded_local_cu_seq_lens_flat: list[int] = []
    padded_local_token_to_seq_parts: list[np.ndarray] = []
    local_starts_per_chunk: list[list[int]] = []
    local_seq_lens_per_chunk: list[list[int]] = []
    layouts: list[tuple[slice, slice, slice, slice]] = []

    request_offset = boundary_offset = token_offset = local_token_offset = 0
    for plan in plans:
        num_requests = len(plan.seq_lens)
        num_tokens = sum(plan.seq_lens)
        num_local_tokens = num_tokens

        seq_lens_flat.extend(plan.seq_lens)
        cu_seq_lens_flat.extend(itertools.accumulate(plan.seq_lens, initial=0))
        query_base = query_start_loc[plan.request_start]
        cu_seqlens_q_flat.extend(
            query_start_loc[request] - query_base
            for request in range(plan.request_start, plan.request_end + 1)
        )
        token_to_seq_parts.append(
            np.repeat(np.arange(num_requests, dtype=np.int32), plan.seq_lens)
        )

        if use_dcp:
            # A request's local rows are its context rows sharded across ranks
            # and rounded up to the interleave block size, so the local start of
            # a continuation is the padded row count of the context before it.
            local_starts = [padded_rows(start) for start in plan.starts]
            local_seq_lens = [
                padded_rows(start + length) - local_start
                for start, length, local_start in zip(
                    plan.starts, plan.seq_lens, local_starts
                )
            ]
            local_starts_per_chunk.append(local_starts)
            local_seq_lens_per_chunk.append(local_seq_lens)
            padded_local_cu_seq_lens_flat.extend(
                itertools.accumulate(local_seq_lens, initial=0)
            )
            padded_local_token_to_seq_parts.append(
                np.repeat(np.arange(num_requests, dtype=np.int32), local_seq_lens)
            )
            num_local_tokens = sum(local_seq_lens)
            # The gather takes per-rank local offsets under DCP.
            starts_flat.extend(local_starts)
        else:
            starts_flat.extend(plan.starts)
        assert num_local_tokens <= row_budget

        layouts.append(
            (
                slice(request_offset, request_offset + num_requests),
                slice(boundary_offset, boundary_offset + num_requests + 1),
                slice(token_offset, token_offset + num_tokens),
                slice(local_token_offset, local_token_offset + num_local_tokens),
            )
        )
        request_offset += num_requests
        boundary_offset += num_requests + 1
        token_offset += num_tokens
        local_token_offset += num_local_tokens

    seq_lens_cpu = _flat_int32(seq_lens_flat)
    starts = _flat_int32(starts_flat).to(device, non_blocking=True)
    cu_seq_lens = _flat_int32(cu_seq_lens_flat).to(device, non_blocking=True)
    cu_seqlens_q = _flat_int32(cu_seqlens_q_flat).to(device, non_blocking=True)
    token_to_seq = _flat_int32(np.concatenate(token_to_seq_parts)).to(
        device, non_blocking=True
    )
    if use_dcp:
        padded_local_cu_seq_lens = _flat_int32(padded_local_cu_seq_lens_flat).to(
            device, non_blocking=True
        )
        padded_local_token_to_seq = _flat_int32(
            np.concatenate(padded_local_token_to_seq_parts)
        ).to(device, non_blocking=True)

    chunks: list[MLACommonPrefillMetadata.ContextChunk] = []
    for index, (plan, layout) in enumerate(zip(plans, layouts)):
        request_slice, boundary_slice, token_slice, local_token_slice = layout
        query_lens = [
            query_start_loc[request + 1] - query_start_loc[request]
            for request in range(plan.request_start, plan.request_end)
        ]
        chunk = MLACommonPrefillMetadata.ContextChunk(
            index=index,
            request_slice=slice(plan.request_start, plan.request_end),
            token_slice=slice(
                query_start_loc[plan.request_start], query_start_loc[plan.request_end]
            ),
            continuation_token_end=query_start_loc[plan.request_start + 1],
            is_continuation=plan.is_continuation,
            num_context_tokens=token_slice.stop - token_slice.start,
            query_start_loc=cu_seqlens_q[boundary_slice],
            max_query_len=max(query_lens),
            cu_seq_lens=cu_seq_lens[boundary_slice],
            starts=starts[request_slice],
            max_seq_len=max(plan.seq_lens),
            seq_lens=seq_lens_cpu[request_slice],
            token_to_seq=token_to_seq[token_slice],
            num_local_context_tokens=local_token_slice.stop - local_token_slice.start,
        )
        if use_dcp:
            assert local_context_lens_allranks is not None
            chunk.padded_local_seq_lens = local_seq_lens_per_chunk[index]
            chunk.local_context_lens_allranks = local_context_lens_allranks[
                plan.request_start : plan.request_end
            ]
            chunk.padded_local_cu_seq_lens = padded_local_cu_seq_lens[boundary_slice]
            chunk.padded_local_token_to_seq = padded_local_token_to_seq[
                local_token_slice
            ]
            chunk.local_starts = local_starts_per_chunk[index]
        chunks.append(chunk)

    return MLACommonPrefillMetadata.ChunkedContextMetadata(
        context_lens=context_lens_cpu.to(device, non_blocking=True),
        workspace=chunked_prefill_workspace,
        chunks=chunks,
        context_lens_list=context_lens,
        empty_token_slices=empty_token_slices,
        dcp_manager=dcp_manager,
        direct_dcp_kv_gather=direct_dcp_kv_gather,
    )


class MLACommonMetadataBuilder(AttentionMetadataBuilder[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    kv_cache_spec: AttentionSpec

    # Defines the level of query length support for this backend.
    # - SINGLE_ONLY: Only single-token queries (no spec decode support)
    # - UNIFORM: Supports uniform multi-token queries (spec decode with uniform lengths)
    # - VARLEN: Supports variable-length queries (spec decode with mixed lengths)
    # If set to UNIFORM or VARLEN, this will increase `reorder_batch_threshold` when
    # speculative decoding is enabled.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.SINGLE_ONLY

    # Whether this builder can flatten a non-causal query block into decode rows.
    supports_non_causal_multi_token_decode: ClassVar[bool] = False

    # A decode backend that owns DCP query and output collectives can omit an
    # MLADCPManager. Chunked-context prefill still gathers KV through the
    # process DCP group before it runs the configured prefill backend.
    supports_direct_dcp_kv_gather: ClassVar[bool] = False

    # The threshold for reordering the batch into decode and prefill requests.
    # If > 1, the batch will be reordered such that requests with
    # query length <= threshold are classified as decode requests.
    # Use `query_len_support` (above) to set this automatically
    # when speculative decoding is enabled.
    reorder_batch_threshold: int = 1

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        configured_workspace_size = envs.VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE
        if configured_workspace_size < 0:
            raise ValueError(
                "VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE must be non-negative, "
                f"got {configured_workspace_size}."
            )
        if configured_workspace_size:
            chunked_prefill_workspace_size = configured_workspace_size
        else:
            chunked_prefill_workspace_size = min(
                # Try for 8 full-length requests or at least 4 pages per request.
                max(
                    8 * model_config.max_model_len,
                    4 * scheduler_config.max_num_seqs * cache_config.block_size,
                ),
                # Bound persistent compressed-KV storage and transient dense K/V
                # projections for long-context models.
                64 * 1024,
            )

        return align_mla_chunked_context_workspace_size(
            vllm_config,
            chunked_prefill_workspace_size,
        )

    @staticmethod
    def determine_prefill_query_data_type(
        vllm_config: VllmConfig,
        model_dtype: torch.dtype,
    ) -> torch.dtype:
        """
        Determine the query data type for prefill queries.
        Return FP8 dtype if cache is FP8 and prefill query quantization
        is enabled, else model dtype.
        """
        use_fp8 = (
            is_quantized_kv_cache(vllm_config.cache_config.cache_dtype)
            and vllm_config.attention_config.use_prefill_query_quantization
            and backend_supports_prefill_query_quantization()
        )

        if use_fp8:
            fp8_dtype = current_platform.fp8_dtype()
            logger.info_once("FP8 prefill attention enabled: query data type is FP8")
            return fp8_dtype
        elif vllm_config.attention_config.use_prefill_query_quantization:
            logger.info_once(
                "Unable to perform FP8 prefill attention when"
                " use_prefill_query_quantization is enabled. Please"
                " ensure that --kv-cache-dtype is set to fp8 and your prefill"
                " backend is compatible with FP8 attention.",
            )
            return model_dtype
        elif (
            is_quantized_kv_cache(vllm_config.cache_config.cache_dtype)
            and backend_supports_prefill_query_quantization()
        ):
            logger.warning_once(
                "FP8 KV cache is enabled but prefill queries are not "
                "quantized to FP8. For long-context workloads (ISL >= 4K), "
                "enabling FP8 prefill attention can significantly optimize "
                "prefill latency. To enable, add: "
                '--attention-config \'{"use_prefill_query_quantization"'
                ": true}'",
            )

        return model_dtype

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[M] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (
            metadata_cls if metadata_cls is not None else MLACommonMetadata
        )
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.device = device
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.non_causal_multi_token_decode = getattr(
            kv_cache_spec, "non_causal_multi_token_decode", False
        )

        # A draft cache group can have a different head count from the target.
        self.num_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(parallel_config)
        # Hybrid MLA models may use different latent dimensions per KV group.
        layer = self.compilation_config.static_forward_context[layer_names[0]]
        self.mla_dims = MLADims(
            q_lora_rank=layer.q_lora_rank,
            kv_lora_rank=layer.kv_lora_rank,
            qk_nope_head_dim=layer.qk_nope_head_dim,
            qk_rope_head_dim=layer.qk_rope_head_dim,
            v_head_dim=layer.v_head_dim,
        )
        self.aot_schedule = current_platform.is_cuda()

        self.kv_cache_spec = kv_cache_spec
        self.q_data_type = self.determine_prefill_query_data_type(
            vllm_config, self.model_config.dtype
        )
        attention_layer = self.compilation_config.static_forward_context[layer_names[0]]

        try:
            configured_dcp_world_size = int(get_dcp_group().world_size)
        except AssertionError:
            # DCP might not be initialized in testing
            configured_dcp_world_size = 1
        self.dcp_world_size = _get_mla_kv_dcp_world_size(
            kv_cache_spec, configured_dcp_world_size
        )
        self.dcp_local_block_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        self.page_size = self.kv_cache_spec.block_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )

        use_packed_fp8_cache = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        self.dcp_manager: MLADCPManager | None = None
        if self.dcp_world_size > 1:
            # Note(hc): The local kvcache is incomplete when DCP is triggered,
            # an additional kvcache allgather across the DCP group is therefore
            # required, so the workspace has to be enlarged by 1/DCP relative
            # to the original TP allocation.
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size
                    + self.chunked_prefill_workspace_size // self.dcp_world_size,
                    self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim,
                ),
                dtype=torch.bfloat16
                if use_packed_fp8_cache
                else self.model_config.dtype,
                device=device,
            )
            self.dcp_manager = getattr(attention_layer, "dcp_manager", None)
            if self.dcp_manager is not None:
                if not isinstance(self.dcp_manager, MLADCPManager):
                    raise TypeError(
                        "MLA attention layer dcp_manager must be an "
                        f"MLADCPManager, got {type(self.dcp_manager).__name__}."
                    )
                self.dcp_manager.init_kv_gather(
                    self.chunked_prefill_workspace,
                    self.chunked_prefill_workspace_size,
                )
            elif not self.supports_direct_dcp_kv_gather:
                raise RuntimeError(
                    f"{type(self).__name__} requires MLADCPManager when DCP is enabled."
                )
        else:
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim,
                ),
                dtype=torch.bfloat16 if use_packed_fp8_cache else self.q_data_type,
                device=device,
            )

        # Metadata builders are created per ubatch when DBO is enabled. MLA
        # prefill backends keep the prepared metadata on the backend object, so
        # each builder needs its own backend instance to avoid cross-ubatch races.
        self._prefill_backend = attention_layer.prefill_backend.clone()

        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode, supports_dcp_with_varlen
        )

        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        return MLACommonDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MLA.
        """
        m = common_attn_metadata
        assert m.num_reqs <= (m.num_actual_tokens * self.reorder_batch_threshold), (
            "MLA only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len <= self.reorder_batch_threshold  # decode only

        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        non_causal_decode = common_attn_metadata.causal is False
        if non_causal_decode:
            if not (
                self.supports_non_causal_multi_token_decode
                and self.non_causal_multi_token_decode
            ):
                raise ValueError(
                    "Non-causal multi-token MLA requires an explicitly supported "
                    "attention group."
                )
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            num_active_reqs = int(torch.count_nonzero(query_lens > 0))
            uniform_active_queries = num_active_reqs > 0 and bool(
                torch.all(query_lens[:num_active_reqs] == query_lens[0])
            )
            trailing_graph_padding = bool(torch.all(query_lens[num_active_reqs:] == 0))
            if not (uniform_active_queries and trailing_graph_padding):
                raise ValueError(
                    "Non-causal MLA requires a uniform query block; got query "
                    f"lengths {query_lens.tolist()}."
                )
            # Use exact GPU sequence lengths instead of the prefill path's CPU
            # context-length upper bounds.
            num_decodes = num_reqs
            num_prefills = 0
            num_decode_tokens = num_tokens
            num_prefill_tokens = 0
        else:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
                    treat_short_extends_as_decodes=not self.use_pcp,
                )
            )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start

            # Upper bound is exact for prefill rows (no D2H sync).
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_query_lens_cpu = (
                query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
                - query_start_loc_cpu[reqs_start:num_reqs]
            )
            context_lens_cpu = (
                seq_lens_cpu[reqs_start:num_reqs] - prefill_query_lens_cpu
            )
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
            )

            chunked_context_metadata = build_mla_chunked_context_metadata(
                context_lens_cpu=context_lens_cpu,
                prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
                chunked_prefill_workspace=self.chunked_prefill_workspace,
                chunked_prefill_workspace_size=self.chunked_prefill_workspace_size,
                block_size=self.page_size,
                align_chunk_to_block=True,
                device=device,
                dcp_world_size=self.dcp_world_size,
                dcp_local_block_size=self.dcp_local_block_size,
                dcp_virtual_block_size=self.dcp_virtual_block_size,
                dcp_manager=self.dcp_manager,
                direct_dcp_kv_gather=(
                    self.dcp_world_size > 1
                    and self.dcp_manager is None
                    and self.supports_direct_dcp_kv_gather
                ),
            )

            prefill_metadata = MLACommonPrefillMetadata(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
                output_dtype=self.model_config.dtype,
                q_data_type=self.q_data_type,
                prefill_backend=self._prefill_backend,
            )

            self._prefill_backend.prepare_metadata(prefill_metadata)

        decode_metadata = None
        if num_decodes > 0:
            dcp_tot_seq_lens_device = None
            if self.dcp_world_size > 1:
                dcp_tot_seq_lens_device = seq_lens[:num_decodes]
                seq_lens = dcp_local_seq_lens

                # After DCP distribution, the maximum number of tokens for any rank is
                # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
                # and I is cp_kv_cache_interleave_size.
                # This eliminates GPU->CPU sync while minimizing workspace
                # over-allocation.
                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_seq_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
            )

        attn_metadata = self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            causal=not non_causal_decode,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata  # type: ignore[return-value]


def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list[int],
    local_context_lens_allranks: list[list[int]],
    local_starts: list[int],
    sum_seq_len: int,
    max_seq_len: int,
    toks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
    e.g.
    allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                              T0_4, T0_5, pad, pad, T1_2, pad, ...]
    -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                  T1_0, T1_1, T1_2, ...]
    Args:
        padded_local_chunk_seq_lens_lst: local chunk context lengths
            under current CP rank.
        local_context_lens_allranks: local context lengths on each CP rank.
        local_starts: per-request local offset into the context this chunk
            starts at.
        sum_seq_len: the sum of cp_chunk_seq_lens_lst.
        max_seq_len: the max value of cp_chunk_seq_lens_lst.
        toks: the number of tokens for local gather cache.
    """
    kv_c_segments = []
    k_pe_segments = []
    src_token_idx = 0
    max_seq_len_check = 0
    for padded_local_chunk_seq_len, local_context_lens, local_start in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks, local_starts
    ):
        cur_seq_len = 0
        for rank, local_context_len in enumerate(local_context_lens):
            # Note(qcs): We split the context into multiple chunks,
            # depending on the size of the workspace.
            # local_context in dcp0:   |-----------------|
            # local_context in dcp1:   |--------------|
            # n*padded_local_chunk:    |-----|-----|-----|
            # local_chunk_len in dcp1: |-----|-----|--|
            # so we need update the last chunk length in dcp1.
            local_chunk_len = min(
                max(0, local_context_len - local_start),
                padded_local_chunk_seq_len,
            )
            if local_chunk_len != 0:
                kv_c_segment = allgatered_kv_c_normed[
                    rank * toks + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                k_pe_segment = allgatered_k_pe[
                    rank * toks + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                kv_c_segments.append(kv_c_segment)
                k_pe_segments.append(k_pe_segment)
                cur_seq_len += local_chunk_len
        max_seq_len_check = max(max_seq_len_check, cur_seq_len)
        src_token_idx += padded_local_chunk_seq_len
    reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
    reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
    assert reorganized_kv_c_normed.shape[0] == sum_seq_len
    assert reorganized_k_pe.shape[0] == sum_seq_len
    assert max_seq_len_check == max_seq_len
    return reorganized_kv_c_normed, reorganized_k_pe


def _gather_dcp_context_kv(
    gathered_kv: torch.Tensor,
    local_kv: torch.Tensor,
    *,
    dcp_manager: MLADCPManager | None,
    direct_dcp_kv_gather: bool,
) -> None:
    """Gather one chunk of DCP-sharded context KV into caller storage."""
    if dcp_manager is not None:
        dcp_manager.kv_gather(gathered_kv, local_kv)
        return
    if direct_dcp_kv_gather:
        gathered_kv.copy_(get_dcp_group().all_gather(local_kv, dim=0))
        return
    raise RuntimeError("MLA DCP chunked prefill has no configured KV gather path.")


def init_mla_context_partial(
    chunked_context: "MLACommonPrefillMetadata.ChunkedContextMetadata",
    attn_output: torch.Tensor,
    attn_softmax_lse: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the running context partial over all prefill tokens.

    Laid out like the chunk partials so the final whole-batch merge against the
    suffix partial sees matching head strides.
    """
    output = torch.empty(
        (num_tokens, *attn_output.shape[1:]),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )
    output_lse = torch.empty(
        (attn_softmax_lse.shape[0], num_tokens),
        dtype=attn_softmax_lse.dtype,
        device=attn_softmax_lse.device,
    )
    # No chunk covers a prefill without context, so neutralize its partial.
    for token_slice in chunked_context.empty_token_slices:
        output[token_slice].zero_()
        output_lse[:, token_slice].fill_(float("-inf"))
    return output, output_lse


def accumulate_mla_context_chunk(
    chunk: "MLACommonPrefillMetadata.ContextChunk",
    attn_output: torch.Tensor,
    attn_softmax_lse: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
) -> None:
    """Fold one chunk's partial into the running context partial.

    Only the first request may be a continuation; its tokens are merged and the
    remaining token range is initialized.
    """
    token_start = chunk.token_slice.start
    token_end = chunk.token_slice.stop
    init_start = token_start
    if chunk.is_continuation:
        init_start = chunk.continuation_token_end
        num_merged = init_start - token_start
        prefix_output, suffix_output = _match_merge_strides(
            output[token_start:init_start], attn_output[:num_merged]
        )
        merge_attn_states(
            output=output[token_start:init_start],
            output_lse=output_lse[:, token_start:init_start],
            prefix_output=prefix_output,
            prefix_lse=output_lse[:, token_start:init_start],
            suffix_output=suffix_output,
            suffix_lse=attn_softmax_lse[:, :num_merged],
        )
    if init_start < token_end:
        written = init_start - token_start
        output[init_start:token_end].copy_(attn_output[written:])
        output_lse[:, init_start:token_end].copy_(attn_softmax_lse[:, written:])


class MLACommonBaseImpl(MLAAttentionImpl[A], Generic[A]):
    """
    Shared MLA base providing dense-MHA prefill (via the selected
    MLAPrefillBackend) for both dense and sparse impls; subclasses add decode
    (``forward_mqa``).
    """

    _use_flashinfer_concat_mla_k: bool

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.

        Args:
            k_nope: Tensor of shape [..., nope_dim]
            k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
                or [..., pe_dim]

        Returns:
            Tensor of shape [..., nope_dim + pe_dim]
        """
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )

        if self._use_flashinfer_concat_mla_k:
            torch.ops.vllm.flashinfer_concat_mla_k(k, k_nope, k_pe)
        else:
            # Fallback: Direct copies with efficient broadcasting
            k[..., : k_nope.shape[-1]] = k_nope
            k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()
        kv_b_proj_input_dtype = _get_kv_b_proj_input_dtype(
            self.kv_b_proj, use_fp8_prefill
        )
        workspace = chunked_context.workspace

        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        output = None
        output_lse = None
        for chunk in chunked_context.chunks:
            toks = chunk.num_context_tokens
            block_table = prefill_metadata.block_table[chunk.request_slice]
            if self.kv_cache_dtype == "fp8_ds_mla":
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace[:toks],
                    block_table=block_table,
                    workspace_starts=chunk.cu_seq_lens,
                    batch_size=chunk.num_requests,
                    seq_starts=chunk.starts,
                )
            elif not use_fp8_prefill:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                    num_tokens=toks,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=chunk.starts,
                )
            else:
                # FP8 path: gather cache without dequantization
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=chunk.cu_seq_lens,
                    batch_size=chunk.num_requests,
                    seq_starts=chunk.starts,
                )

            # Extract kv_c_normed from workspace
            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            if kv_b_proj_input_dtype is not None:
                kv_c_normed = kv_c_normed.to(kv_b_proj_input_dtype)

            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk=chunk,
                    q=q[chunk.token_slice],
                    k=k,
                    v=v,
                )
            )
            if output is None:
                if (
                    len(chunked_context.chunks) == 1
                    and not chunked_context.empty_token_slices
                ):
                    return attn_output, attn_softmax_lse
                output, output_lse = init_mla_context_partial(
                    chunked_context,
                    attn_output,
                    attn_softmax_lse,
                    num_tokens=q.shape[0],
                )
            accumulate_mla_context_chunk(
                chunk, attn_output, attn_softmax_lse, output, output_lse
            )

        return output, output_lse

    def _context_parallel_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        dcp_world_size: int,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()
        kv_b_proj_input_dtype = _get_kv_b_proj_input_dtype(
            self.kv_b_proj, use_fp8_prefill
        )
        output = None
        output_lse = None
        workspace = chunked_context.workspace

        for chunk in chunked_context.chunks:
            assert chunk.padded_local_seq_lens is not None
            assert chunk.local_context_lens_allranks is not None
            assert chunk.padded_local_cu_seq_lens is not None
            assert chunk.padded_local_token_to_seq is not None
            assert chunk.local_starts is not None

            toks = chunk.num_local_context_tokens
            padded_local_cu_seq_lens = chunk.padded_local_cu_seq_lens
            block_table = prefill_metadata.block_table[chunk.request_slice]
            if self.kv_cache_dtype == "fp8_ds_mla":
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace[:toks],
                    block_table=block_table,
                    workspace_starts=padded_local_cu_seq_lens,
                    batch_size=chunk.num_requests,
                    seq_starts=chunk.starts,
                )
            elif is_quantized_kv_cache(self.kv_cache_dtype):
                assert k_scale is not None
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=padded_local_cu_seq_lens,
                    token_to_seq=chunk.padded_local_token_to_seq,
                    num_tokens=toks,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=chunk.starts,
                )
            else:
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=block_table,
                    cu_seq_lens=padded_local_cu_seq_lens,
                    batch_size=chunk.num_requests,
                    seq_starts=chunk.starts,
                )
            # workspace
            # |------- N tokens --------|--------- N*dcp_size tokens ----------|
            # |<- use for local_gather ->|<--------- use for allgather -------->|
            allgather_offset = workspace.shape[0] // (dcp_world_size + 1)
            assert allgather_offset * (dcp_world_size + 1) == workspace.shape[0]
            assert toks <= allgather_offset
            local_gathered_kvcache = workspace[:toks]
            cur_allgather_workspace = workspace[
                allgather_offset : allgather_offset * (1 + dcp_world_size)
            ]
            assert toks * dcp_world_size <= cur_allgather_workspace.shape[0]
            cur_allgather_kvcache = cur_allgather_workspace[: toks * dcp_world_size]
            _gather_dcp_context_kv(
                cur_allgather_kvcache,
                local_gathered_kvcache,
                dcp_manager=chunked_context.dcp_manager,
                direct_dcp_kv_gather=chunked_context.direct_dcp_kv_gather,
            )
            assert (
                cur_allgather_kvcache.shape[-1]
                == self.kv_lora_rank + self.qk_rope_head_dim
            )
            allgatered_kv_c_normed, allgatered_k_pe = cur_allgather_kvcache.unsqueeze(
                1
            ).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            kv_c_normed, k_pe = reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=chunk.padded_local_seq_lens,
                local_context_lens_allranks=chunk.local_context_lens_allranks,
                local_starts=chunk.local_starts,
                sum_seq_len=chunk.num_context_tokens,
                max_seq_len=chunk.max_seq_len,
                toks=toks,
            )
            if kv_b_proj_input_dtype is not None:
                kv_c_normed = kv_c_normed.to(kv_b_proj_input_dtype)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk=chunk,
                    q=q[chunk.token_slice],
                    k=k,
                    v=v,
                )
            )

            if output is None:
                if (
                    len(chunked_context.chunks) == 1
                    and not chunked_context.empty_token_slices
                ):
                    return attn_output, attn_softmax_lse
                output, output_lse = init_mla_context_partial(
                    chunked_context,
                    attn_output,
                    attn_softmax_lse,
                    num_tokens=q.shape[0],
                )
            accumulate_mla_context_chunk(
                chunk, attn_output, attn_softmax_lse, output, output_lse
            )

        return output, output_lse

    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        # Convert q to FP8 if FP8 prefill attention is enabled
        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        has_context = prefill_metadata.chunked_context is not None
        assert output_scale is None or not has_context, (
            "Fused FP8 output is only wired for the non-chunked-context path"
        )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        if use_fp8_prefill:
            k = k.to(prefill_metadata.q_data_type)
            v = v.to(prefill_metadata.q_data_type)

        output_prefill = prefill_metadata.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
            out=(
                output.view(-1, self.num_heads, self.v_head_dim)
                if output_scale is not None
                else None
            ),
            output_scale=output_scale,
        )

        if has_context:
            assert prefill_metadata.chunked_context is not None
            suffix_output, suffix_lse = output_prefill
            if self.dcp_world_size > 1:
                context_output, context_lse = (
                    self._context_parallel_compute_prefill_context(
                        q,
                        kv_c_and_k_pe_cache,
                        attn_metadata,
                        k_scale=k_scale,
                        dcp_world_size=self.dcp_world_size,
                    )
                )
            else:
                context_output, context_lse = self._compute_prefill_context(
                    q, kv_c_and_k_pe_cache, attn_metadata, k_scale
                )

            context_output = context_output[..., : self.v_head_dim]
            suffix_output = suffix_output[..., : self.v_head_dim]

            output = output.view(-1, self.num_heads, self.v_head_dim)
            context_output, suffix_output = _match_merge_strides(
                context_output, suffix_output
            )
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
        elif output_scale is None:
            # With output_scale set, backend already wrote into `output` in place.
            assert isinstance(output_prefill, torch.Tensor)
            output_prefill = output_prefill[..., : self.v_head_dim]
            output_prefill = output_prefill.flatten(start_dim=-2)
            output.copy_(output_prefill)


class MLACommonImpl(MLACommonBaseImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def fused_output_quant_supported(self, quant_key):
        return quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
        )

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        # DSV3.2 MLA Specific Arguments
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA")

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )
        self.q_lora_rank = q_lora_rank
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads
        self.supports_quant_query_input = True
        self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()

        # Use flashinfer's optimized concat_mla_k kernel when available.
        # The kernel is optimized for DeepSeek V3 dimensions:
        # num_heads=128, nope_dim=128, rope_dim=64
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )

        parallel_config = get_current_vllm_config().parallel_config
        # Avoid requiring an initialized DCP group in tests.
        self.dcp_world_size: int = parallel_config.decode_context_parallel_size
        self.cp_kv_cache_interleave_size: int = (
            parallel_config.cp_kv_cache_interleave_size
        )

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError
