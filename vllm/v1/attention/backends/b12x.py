# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x paged causal attention backend for SM12x."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Mapping

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_paged_attention,
    get_b12x_scratch_buffers,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    get_dtype_size,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout, KVCacheSpec

logger = init_logger(__name__)

_B12X_SUPPORTED_PAGE_SIZES = (64, 128)
_B12X_PREFERRED_PAGE_SIZE = 128
_MIN_PAGED_TILE_Q = 16
_B12X_FP8_KV_CACHE_DTYPES: tuple[CacheDType, ...] = ("fp8", "fp8_e4m3")
_B12X_SUPPORTED_KV_CACHE_DTYPES: tuple[CacheDType, ...] = (
    "auto",
    "bfloat16",
    *_B12X_FP8_KV_CACHE_DTYPES,
)


def _max_page_table_width(
    max_model_len: int,
    block_size: int,
    max_num_batched_tokens: int,
    is_hybrid: bool,
) -> int:
    width = max(cdiv(max(max_model_len, 1), block_size), 1)
    if is_hybrid:
        # Hybrid cache setup can enlarge the storage block after attention
        # layers are initialized. Its expansion into kernel-sized blocks adds
        # at most one storage block of trailing page-table capacity.
        width += cdiv(max_num_batched_tokens, block_size)
    return width


def _kv_page_size(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return the static kernel page geometry negotiated by vLLM.

    The KV manager can split the configured storage block into a smaller
    kernel page when another backend shares its cache group. Cache shapes are
    fixed before graph capture, so this is not a live-length policy decision.
    """
    if key_cache.ndim < 2 or value_cache.ndim < 2:
        raise ValueError(
            "b12x expects paged K/V caches with a page dimension, got "
            f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}."
        )
    key_page_size = int(key_cache.shape[1])
    value_page_size = int(value_cache.shape[1])
    if key_page_size != value_page_size:
        raise ValueError(
            "b12x requires matching K/V page sizes, got "
            f"{key_page_size} and {value_page_size}."
        )
    if key_page_size not in _B12X_SUPPORTED_PAGE_SIZES:
        raise ValueError(
            "b12x requires runtime page size in "
            f"{_B12X_SUPPORTED_PAGE_SIZES}, got {key_page_size}."
        )
    return key_page_size


def _capture_alloc_forbidden() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _ensure_i32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"b12x would convert {name} to int32 during CUDA graph "
                "capture. Prepare int32 metadata before capture."
            )
        tensor = tensor.to(torch.int32)
    if not tensor.is_contiguous():
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"b12x would make {name} contiguous during CUDA graph "
                "capture. Prepare contiguous metadata before capture."
            )
        tensor = tensor.contiguous()
    return tensor


def _dtype_from_cache_config(
    kv_cache_dtype: str,
    vllm_config: VllmConfig,
) -> torch.dtype:
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES:
        return current_platform.fp8_dtype()
    if kv_cache_dtype != "auto":
        raise NotImplementedError(
            "b12x currently supports only auto, bfloat16, "
            "fp8, and fp8_e4m3 "
            f"KV cache dtypes; got {kv_cache_dtype!r}."
        )
    return vllm_config.model_config.dtype


def _is_b12x_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES


class B12xPagedAttentionBackend(AttentionBackend):
    """b12x paged attention backend for regular/GQA decoder layers."""

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v, (
            "Separate K/V planes require symmetric K/V head sizes."
        )
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        _B12X_SUPPORTED_KV_CACHE_DTYPES
    )

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "B12X"

    @classmethod
    def get_impl_cls(cls) -> type[AttentionImplBase]:
        return B12xPagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[B12xPagedMetadataBuilder]:
        return B12xPagedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return list(_B12X_SUPPORTED_PAGE_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) in _B12X_SUPPORTED_PAGE_SIZES

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if int(default_block_size) in _B12X_SUPPORTED_PAGE_SIZES:
            return int(default_block_size)
        return _B12X_PREFERRED_PAGE_SIZE

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 192, 256]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The b12x paged kernels also gate
        # internally, but keep vLLM selection fail-fast and explicit.
        return (capability.major, capability.minor) in ((12, 0), (12, 1))

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if dtype != torch.bfloat16:
            return "b12x currently requires bfloat16 queries"
        if kv_cache_dtype == "float16":
            return "b12x does not support float16 KV cache"
        if (
            kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
            and not _is_b12x_fp8_kv_cache(kv_cache_dtype)
        ):
            return "b12x currently supports only fp8/fp8_e4m3 quantized KV cache dtypes"
        paged_attention = get_b12x_paged_attention()
        if paged_attention is None:
            return "Install the b12x backend with `pip install vllm[b12x]`"
        if not paged_attention.is_supported():
            return "b12x paged attention is not supported on the current device"
        return None

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


@dataclass
class B12xPagedMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class B12xPagedMetadataBuilder(AttentionMetadataBuilder[B12xPagedMetadata]):
    """Metadata builder for b12x.

    Decode and uniform speculative-verifier batches use preplanned graph
    buckets. Extend/prefill remains eager and does not affect uniform decode
    graph eligibility.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        del vllm_config, kv_cache_spec
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xPagedMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        return B12xPagedMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            causal=cm.causal,
        )

    def update_block_table(
        self,
        metadata: B12xPagedMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> B12xPagedMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


class B12xPagedAttentionImpl(AttentionImpl[B12xPagedMetadata]):
    """b12x paged GQA attention implementation."""

    can_return_lse_for_decode: bool = False
    supports_dcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("b12x does not support ALiBi.")
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("b12x does not support logits soft cap.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "b12x currently supports decoder self-attention only."
            )
        if is_quantized_kv_cache(kv_cache_dtype) and not _is_b12x_fp8_kv_cache(
            kv_cache_dtype
        ):
            raise NotImplementedError(
                "b12x currently supports only fp8/fp8_e4m3 quantized KV cache dtypes."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError("b12x requires q heads divisible by kv heads.")

        expected_scale = head_size**-0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise NotImplementedError(
                "b12x currently requires canonical softmax scale "
                f"head_dim**-0.5={expected_scale}, got {scale}."
            )
        if self.total_cp_world_size > 1:
            raise NotImplementedError(
                "b12x does not yet support decode/prefill context parallelism."
            )

        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.output_head_size = self.head_size
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.window_left = -1 if sliding_window is None else int(sliding_window) - 1

        self._sinks_source = sinks
        if sinks is not None and (
            sinks.ndim != 1 or int(sinks.shape[0]) != self.num_heads
        ):
            raise ValueError(
                "b12x sinks must have shape "
                f"[{self.num_heads}], got {tuple(sinks.shape)}."
            )
        self.sinks = sinks if sinks is None or sinks.dtype == torch.float32 else None

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        spec_config = vllm_config.speculative_config
        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "b12x requires --block-size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got "
                f"{cache_config.block_size}."
            )

        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.dtype = model_config.dtype
        self.kv_torch_dtype = _dtype_from_cache_config(kv_cache_dtype, vllm_config)
        if self.dtype != torch.bfloat16:
            raise NotImplementedError("b12x currently requires bfloat16 queries.")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        max_model_len = int(model_config.max_model_len)
        self._max_num_seqs = max_num_seqs
        max_page_table_widths = {
            page_size: _max_page_table_width(
                max_model_len,
                page_size,
                max_batched,
                model_config.is_hybrid,
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        }

        # Extend dispatch may depend on the static Q tensor capacity, but never
        # on live per-request lengths. Keep a small set of capacity buckets so
        # short/tail prefills do not replay the maximum 8K CTA grid.
        self._extend_q_capacities = tuple(
            sorted(
                {
                    min(max_batched, q_capacity)
                    for q_capacity in (128, 512, 1024, 2048, 4096, max_batched)
                    if q_capacity > 0
                }
            )
        )

        paged_attention = get_b12x_paged_attention()
        if paged_attention is None:
            raise RuntimeError("B12X paged attention requires the b12x package.")
        if not paged_attention.is_supported(self.device):
            raise RuntimeError("B12X paged attention is not supported on this device.")
        self._paged_attention = paged_attention
        self._max_batched = max_batched
        self._max_num_seqs = max_num_seqs
        self._max_model_len = max_model_len
        self._max_page_table_widths = max_page_table_widths
        self._verify_q_per_req = (
            1 + int(getattr(spec_config, "num_speculative_tokens", None) or 0)
            if spec_config is not None
            else 0
        )
        if self._verify_q_per_req <= 1:
            self._verify_q_per_req = 0
        self._extend_q_capacities = tuple(sorted({
            min(max_batched, capacity)
            for capacity in (128, 512, 1024, 2048, 4096, max_batched)
            if capacity > 0
        }))
        self._plans: dict[tuple[str, int, int, int], object] = {}
        self.supports_quant_query_input = False

        logger.info_once(
            "Using b12x with q_heads=%d kv_heads=%d head_dim_qk=%d "
            "head_dim_vo=%d window_left=%d verify_q_per_req=%d "
            "extend_q_capacities=%s.",
            self.num_heads, self.num_kv_heads, self.head_size,
            self.output_head_size, self.window_left, self._verify_q_per_req,
            self._extend_q_capacities,
        )
    def _request_name(self, owner: object, key: tuple[str, int, int, int]) -> str:
        mode, page_size, batch, total_q = key
        return f"attention.paged.{id(owner):x}.{mode}.p{page_size}.b{batch}.q{total_q}"

    def _caps(
        self, *, page_size: int, mode: str, batch: int, total_q: int,
        max_work_items: int, max_partial_rows: int, copy_runtime_metadata: bool,
        num_cache_pages: int,
    ):
        return self._paged_attention.Caps(
            device=self.device, mode=mode, dtype=self.dtype,
            kv_dtype=self.kv_torch_dtype, num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads, head_dim_qk=self.head_size,
            head_dim_vo=self.output_head_size, page_size=page_size,
            max_total_q=total_q, max_batch=batch,
            max_page_table_width=self._max_page_table_widths[page_size],
            max_work_items=max_work_items, max_partial_rows=max_partial_rows,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=True, copy_runtime_metadata=copy_runtime_metadata,
        )

    @staticmethod
    def _descriptor(shape, strides, dtype: torch.dtype) -> Mapping[str, object]:
        return {
            "shape": tuple(int(value) for value in shape),
            "strides": tuple(int(value) for value in strides),
            "dtype": str(dtype).removeprefix("torch."),
            "alignment": 16,
        }

    def _declaration(
        self, *, page_size: int, mode: str, batch: int, total_q: int,
        key_cache: torch.Tensor, value_cache: torch.Tensor, owner: object,
    ):
        width = self._max_page_table_widths[page_size]
        if mode == "decode":
            capacity = self._paged_attention.decode_graph_capacity(
                device=self.device, q_dtype=self.dtype, kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.output_head_size,
                page_size=page_size, batch=batch, max_cache_page_count=width,
                window_left=self.window_left,
            )
            max_work_items, max_partial_rows, copy_metadata = (
                capacity.max_work_items, capacity.max_partial_rows, True)
        elif mode == "verify":
            capacity = self._paged_attention.verify_graph_capacity(
                device=self.device, q_dtype=self.dtype, kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.output_head_size,
                page_size=page_size, batch=batch, query_len=self._verify_q_per_req,
                max_cache_page_count=width, window_left=self.window_left,
            )
            max_work_items, max_partial_rows, copy_metadata = (
                capacity.max_work_items, capacity.max_partial_rows, True)
        else:
            capacity = self._paged_attention.extend_graph_capacity(
                device=self.device, q_dtype=self.dtype, kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size, head_dim_vo=self.output_head_size,
                page_size=page_size, batch=batch, total_q_capacity=total_q,
                max_cache_page_count=width, window_left=self.window_left,
            )
            max_work_items, max_partial_rows, copy_metadata = (
                capacity.max_work_items, 0, False)
        caps = self._caps(
            page_size=page_size, mode=mode, batch=batch, total_q=total_q,
            max_work_items=max_work_items, max_partial_rows=max_partial_rows,
            copy_runtime_metadata=copy_metadata,
            num_cache_pages=int(key_cache.shape[0]),
        )
        k_descale, v_descale = self._prepare_fp8_descales(owner, batch, self.device)
        q_shape = (total_q, self.num_heads, self.head_size)
        out_shape = (total_q, self.num_heads, self.output_head_size)
        operands = {
            "q": self._descriptor(q_shape, (self.num_heads * self.head_size, self.head_size, 1), self.dtype),
            "k_cache": {
                **self._descriptor(key_cache.shape, key_cache.stride(), key_cache.dtype),
                "alignment": min(16, int(key_cache.data_ptr()) & -int(key_cache.data_ptr())) if key_cache.data_ptr() else 16,
            },
            "v_cache": {
                **self._descriptor(value_cache.shape, value_cache.stride(), value_cache.dtype),
                "alignment": min(16, int(value_cache.data_ptr()) & -int(value_cache.data_ptr())) if value_cache.data_ptr() else 16,
            },
            "output": self._descriptor(out_shape, (self.num_heads * self.output_head_size, self.output_head_size, 1), self.dtype),
            "page_table": self._descriptor((batch, width), (width, 1), torch.int32),
            "cache_seqlens": self._descriptor((batch,), (1,), torch.int32),
            "cu_seqlens_q": self._descriptor((batch + 1,), (1,), torch.int32),
            "q2k_indices": None,
            "k_descale": None if k_descale is None else self._descriptor(
                k_descale.shape, k_descale.stride(), k_descale.dtype),
            "v_descale": None if v_descale is None else self._descriptor(
                v_descale.shape, v_descale.stride(), v_descale.dtype),
            "attention_sink_bias": None if self.sinks is None else self._descriptor((self.num_heads,), (1,), torch.float32),
            "relative_attention_bias": None,
        }
        invocation = dict(
            self._paged_attention.invocation_from_descriptors(caps, operands=operands)
        )
        # The native route is specialized on the window, so a declaration
        # carries the same window its bindings request.
        invocation["window_left"] = self.window_left
        return self._paged_attention.plan(caps, invocation=invocation)

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload,
    ) -> tuple[B12xPreparationUnit, ...]:
        kv_cache = getattr(layer, "kv_cache", None)
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            return ()
        key_cache, value_cache = self._kv_cache_views(kv_cache)
        page_size = _kv_page_size(key_cache, value_cache)
        if workload.output_dtype != self.dtype:
            raise ValueError("b12x paged output dtype differs from its loaded contract")
        plans: dict[tuple[str, int, int, int], object] = {}
        requests = []
        batches = tuple(sorted(set(
            count for count in workload.token_counts if count <= self._max_num_seqs
        ) | {self._max_num_seqs}))
        for batch in batches:
            key = ("decode", page_size, batch, batch)
            declaration = self._declaration(
                page_size=page_size, mode="decode", batch=batch, total_q=batch,
                key_cache=key_cache, value_cache=value_cache, owner=layer)
            plans[key] = declaration
            requests.append(declaration.request(
                name=self._request_name(layer, key),
                prepare_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=False),
                benchmark_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=True),
            ))
            if self._verify_q_per_req > 1:
                total_q = batch * self._verify_q_per_req
                key = ("verify", page_size, batch, total_q)
                declaration = self._declaration(
                    page_size=page_size, mode="verify", batch=batch, total_q=total_q,
                    key_cache=key_cache, value_cache=value_cache, owner=layer)
                plans[key] = declaration
                requests.append(declaration.request(
                    name=self._request_name(layer, key),
                    prepare_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=False),
                    benchmark_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=True),
                ))
        for batch in range(1, self._max_num_seqs + 1):
            for total_q in self._extend_q_capacities:
                if batch >= total_q:
                    continue
                key = ("extend", page_size, batch, total_q)
                declaration = self._declaration(
                    page_size=page_size, mode="extend", batch=batch, total_q=total_q,
                    key_cache=key_cache, value_cache=value_cache, owner=layer)
                plans[key] = declaration
                requests.append(declaration.request(
                    name=self._request_name(layer, key),
                    prepare_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=False),
                    benchmark_call=lambda state, key=key, layer=layer: self._prepared_call(layer, state, key, benchmark=True),
                ))
        self._plans = plans
        if not requests:
            return ()
        return (B12xPreparationUnit(
            name="b12x paged attention",
            key=(id(layer), page_size),
            requests=tuple(requests),
            stage="state",
            autotune=not workload.eager_only,
        ),)

    def _prepared_call(
        self, owner: object, state: object, key, *, benchmark: bool, caches=None,
    ):
        from b12x.preparation import PreparedCall

        mode, page_size, batch, total_q = key
        if caches is None:
            key_cache, value_cache = self._kv_cache_views(owner.kv_cache)
        else:
            key_cache, value_cache = caches
        if _kv_page_size(key_cache, value_cache) != page_size:
            raise PreparationResourceUnavailableError("b12x paged cache generation changed")
        specs = state.scratch_plan.scratch_specs()
        # Trial and prepare factories own their scratch; the runtime binding
        # in forward() draws from the workspace manager instead.
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
            for spec in specs
        )
        q = torch.empty((total_q, self.num_heads, self.head_size), dtype=self.dtype, device=self.device)
        output = torch.empty((total_q, self.num_heads, self.output_head_size), dtype=self.dtype, device=self.device)
        page_table = torch.zeros((batch, self._max_page_table_widths[page_size]), dtype=torch.int32, device=self.device)
        cache_seqlens = torch.full((batch,), page_size, dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[-1] = total_q
        k_descale, v_descale = self._prepare_fp8_descales(owner, batch, self.device)
        binding = state.bind(
            scratch=scratch, q=q, k_cache=key_cache, v_cache=value_cache, output=output,
            page_table=page_table, cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, window_left=self.window_left,
            active_total_q=None if mode == "extend" else total_q,
            attention_sink_bias=self.sinks, k_descale=k_descale, v_descale=v_descale,
        )
        return PreparedCall(
            run=lambda: state.run(binding),
            produce=lambda: q.zero_(),
            owners=(scratch, q, output, page_table, cache_seqlens, cu_seqlens_q),
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        source_sinks = self._sinks_source
        if source_sinks is None:
            return
        if source_sinks.dtype == torch.float32:
            self.sinks = source_sinks
        elif self.sinks is None or self.sinks.dtype != torch.float32:
            self.sinks = source_sinks.to(torch.float32)
        else:
            self.sinks.copy_(source_sinks)

    def _prepare_fp8_descales(
        self,
        layer: AttentionLayer,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None
        if num_reqs <= 0:
            raise ValueError("b12x fp8 KV descale request count must be positive.")

        def _prepare(scale: torch.Tensor, name: str) -> torch.Tensor:
            if scale.device != device:
                raise RuntimeError(f"b12x {name} must be on the query device.")
            if scale.dtype != torch.float32:
                raise RuntimeError(f"b12x {name} must be float32.")
            if scale.ndim == 0:
                return scale.expand(num_reqs)
            if scale.ndim == 1:
                if int(scale.shape[0]) == 1:
                    return scale.expand(num_reqs)
                if int(scale.shape[0]) >= num_reqs:
                    return scale[:num_reqs]
            raise ValueError(
                f"b12x {name} must be scalar or rank-1 with at least "
                f"{num_reqs} values; got shape {tuple(scale.shape)}."
            )

        return _prepare(layer._k_scale, "k_scale"), _prepare(layer._v_scale, "v_scale")

    def _kv_cache_views(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = key_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        value_cache = value_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            fp8_dtype = current_platform.fp8_dtype()
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(fp8_dtype)
            if value_cache.dtype == torch.uint8:
                value_cache = value_cache.view(fp8_dtype)
        if (
            key_cache.dtype != self.kv_torch_dtype
            or value_cache.dtype != self.kv_torch_dtype
        ):
            raise TypeError(
                f"b12x plan expects KV dtype {self.kv_torch_dtype}, got "
                f"{key_cache.dtype}/{value_cache.dtype}."
            )
        return key_cache, value_cache

    def _select_plan(
        self,
        attn_metadata: B12xPagedMetadata,
        total_q: int,
        q_capacity: int,
        num_reqs: int,
        page_size: int,
        *,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        layer: object | None = None,
    ) -> tuple[object, tuple[str, int, int, int]]:
        if attn_metadata.max_query_len <= 1 and total_q == num_reqs:
            key = ("decode", page_size, total_q, total_q)
        elif (
            self._verify_q_per_req > 1
            and attn_metadata.max_query_len == self._verify_q_per_req
            and total_q == num_reqs * self._verify_q_per_req
        ):
            key = ("verify", page_size, num_reqs, total_q)
        else:
            capacity = next(
                (candidate for candidate in self._extend_q_capacities
                 if q_capacity <= candidate),
                None,
            )
            if capacity is None:
                raise ValueError(
                    f"b12x extend Q capacity {q_capacity} exceeds prepared "
                    f"maximum {self._extend_q_capacities[-1]}."
                )
            key = ("extend", page_size, num_reqs, capacity)
        plan = self._plans.get(key)
        if plan is not None:
            return plan, key
        # A variant the preparation pass did not declare is declared here with
        # its default configuration and materialized on first use.
        if key_cache is None or value_cache is None:
            raise PreparationResourceUnavailableError(
                "b12x paged plan is not prepared for cache generation "
                f"and variant {key!r}"
            )
        mode, _, batch, total = key
        plan = self._declaration(
            page_size=page_size, mode=mode, batch=batch, total_q=total,
            key_cache=key_cache, value_cache=value_cache, owner=layer,
        )
        self._plans[key] = plan
        return plan, key

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: B12xPagedMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "b12x does not support fused output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if output.shape[-1] != self.output_head_size:
            raise ValueError(
                f"b12x expected output head dim {self.output_head_size}, got "
                f"{output.shape[-1]}."
            )
        if kv_cache.numel() == 0:
            return output.fill_(0)

        # In FULL cudagraph mode vLLM may pad attention metadata to the graph
        # bucket while still passing per-layer Q/output tensors with only the
        # real rows. Use tensor capacity as the launch contract and avoid
        # selecting decode graph replay for padded virtual requests.
        q_capacity = min(
            int(query.shape[0]),
            int(output.shape[0]),
        )
        num_actual_tokens = min(
            int(attn_metadata.num_actual_tokens),
            q_capacity,
        )
        if num_actual_tokens <= 0:
            return output
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        if q.dtype != self.dtype or out.dtype != self.dtype:
            raise TypeError(
                f"b12x plan expects dtype {self.dtype}, got "
                f"q={q.dtype}, output={out.dtype}."
            )

        key_cache, value_cache = self._kv_cache_views(kv_cache)
        page_size = _kv_page_size(key_cache, value_cache)
        if not attn_metadata.causal:
            raise NotImplementedError("b12x supports causal attention only.")

        page_table = _ensure_i32_contiguous(attn_metadata.block_table, "block_table")
        cache_seqlens = _ensure_i32_contiguous(attn_metadata.seq_lens, "seq_lens")
        cu_seqlens_q = _ensure_i32_contiguous(
            attn_metadata.query_start_loc,
            "query_start_loc",
        )
        num_reqs = int(cache_seqlens.shape[0])
        if attn_metadata.max_query_len <= 1 and num_actual_tokens < num_reqs:
            num_reqs = num_actual_tokens
            page_table = page_table[:num_reqs]
            cache_seqlens = cache_seqlens[:num_reqs]
            cu_seqlens_q = cu_seqlens_q[: num_reqs + 1]
        k_descale, v_descale = self._prepare_fp8_descales(
            layer,
            num_reqs,
            q.device,
        )
        plan, plan_key = self._select_plan(
            attn_metadata, num_actual_tokens, q_capacity, num_reqs, page_size,
            key_cache=key_cache, value_cache=value_cache, layer=layer,
        )
        scratch = tuple(get_b12x_scratch_buffers(plan))
        binding = self._paged_attention.bind(
            plan,
            scratch=scratch,
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=self.window_left,
            active_total_q=(
                None if plan_key[0] == "extend" else num_actual_tokens
            ),
            attention_sink_bias=self.sinks,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        self._paged_attention.run(binding=binding, plan=plan)
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        key_cache, value_cache = self._kv_cache_views(kv_cache)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
