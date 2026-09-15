# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x compressed sparse MLA for DeepSeek V4."""

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.b12x_indexer import (
    DeepseekV4B12xIndexerBackend,
    DeepseekV4B12xSparseIndexer,
    b12x_indexer_is_supported,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import bf16_o_proj
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    get_b12x_compressed_sparse_mla,
    get_b12x_mhc,
    get_b12x_wo_projection,
    set_b12x_preparation_provider,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

_DSV4_HEAD_DIM = 512
_DSV4_CACHE_BYTES_PER_TOKEN = 584
_C128A_TOPK_ALIGNMENT = 128


def _require_b12x_compressed_sparse_mla() -> Any:
    module = get_b12x_compressed_sparse_mla()
    if module is None:
        raise RuntimeError(
            "DeepSeek V4 B12x attention requires `pip install vllm[b12x]`."
        )
    if not module.is_supported():
        raise RuntimeError(
            "B12x compressed sparse MLA is not supported on this device."
        )
    for name in ("Caps", "plan", "run", "split_chunks_for_contract"):
        getattr(module, name)
    return module


def _require_b12x_wo_projection() -> Any:
    module = get_b12x_wo_projection()
    if module is None:
        raise RuntimeError(
            "DeepSeek V4 B12x output projection requires `pip install vllm[b12x]`."
        )
    if not module.is_supported():
        raise RuntimeError("B12x output projection is not supported on this device.")
    for name in ("pack_weights", "run_inv_rope"):
        getattr(module, name)
    return module


def _require_b12x_mhc() -> Any:
    module = get_b12x_mhc()
    if module is None:
        raise RuntimeError("DeepSeek V4 B12x mHC requires `pip install vllm[b12x]`.")
    if not module.is_supported():
        raise RuntimeError("B12x mHC is not supported on this device.")
    for name in (
        "Caps",
        "plan",
        "run_pre",
        "run_post_pre",
        "run_post",
        "MULT",
        "DEFAULT_BLOCK_K",
        "DEFAULT_BLOCK_H",
    ):
        getattr(module, name)
    return module


def b12x_dsv4_is_supported() -> bool:
    attention_module = get_b12x_compressed_sparse_mla()
    wo_module = get_b12x_wo_projection()
    mhc_module = get_b12x_mhc()
    return bool(
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and attention_module is not None
        and attention_module.is_supported()
        and wo_module is not None
        and wo_module.is_supported()
        and mhc_module is not None
        and mhc_module.is_supported()
        and b12x_indexer_is_supported()
    )


class B12xMHCResidual:
    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> None:
        module = _require_b12x_mhc()
        self._caps = module.Caps
        self._plan_factory = module.plan
        self._run_pre = module.run_pre
        self._run_post = module.run_post
        self._run_post_pre = module.run_post_pre
        self._plans: dict[tuple[str, int], object] = {}
        self._plan_key: tuple[int, ...] | None = None

        expected_hc_mult = int(module.MULT)
        if hc_mult != expected_hc_mult:
            raise NotImplementedError(
                f"B12x mHC requires hc_mult={expected_hc_mult}, got {hc_mult}."
            )

        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.block_k = int(module.DEFAULT_BLOCK_K)
        self.block_h = int(module.DEFAULT_BLOCK_H)
        total_k = self.hc_mult * self.hidden_size
        if total_k % self.block_k != 0:
            raise ValueError(
                "B12x mHC requires hc_mult * hidden_size to be divisible by "
                f"block_k={self.block_k}, got {total_k}."
            )
        self.split_k = total_k // self.block_k

    def _plan_for(self, operation: str, tokens: int, device=None) -> object:
        tokens = int(tokens)
        exact = self._plans.get((operation, tokens))
        if exact is not None:
            return exact
        capacity = max((rows for op, rows in self._plans if op == operation), default=0)
        if tokens <= capacity and capacity:
            return self._plans[(operation, capacity)]
        # Profiling and other pre-preparation callers execute against a live
        # batch before the weights-stage provider declares any plan. Build an
        # on-demand plan for this bucket; require_prepared lazily materializes
        # it with the default configuration (a warning, not an error, outside
        # a frozen session), and the served path keeps using the prepared plan.
        from b12x.preparation import FrozenMapping

        plan_operation = (
            "post_pre" if operation in ("post_pre", "post_pre_bf16") else operation
        )
        invocation = FrozenMapping(
            {
                "operation": plan_operation,
                "has_norm_weight": plan_operation != "post",
                "norm_weight_dtype": "bfloat16",
                "has_fn_bf16": operation == "post_pre_bf16",
                "lagged_mix": False,
                "bf16x2_eligible": True,
                "output_mode": "functional",
                "rms_eps": self.rms_eps,
                "hc_eps": self.hc_eps,
                "sinkhorn_iters": self.sinkhorn_iters,
                "norm_eps": self.rms_eps,
                "block_k": self.block_k,
                "block_h": self.block_h,
            }
        )
        plan = self._plan_factory(
            self._caps(
                device=device,
                dtype=torch.bfloat16,
                max_tokens=max(tokens, 1),
                hidden_size=self.hidden_size,
                split_k=self.split_k,
            ),
            invocation=invocation,
        )
        self._plans[(operation, max(tokens, 1))] = plan
        return plan

    def run_pre(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._run_pre(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=float(norm_eps),
            plan=self._plan_for("pre", int(residual.shape[0]), residual.device),
        )

    def run_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
        hc_fn_bf16: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = int(residual.shape[0])
        return self._run_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=float(norm_eps),
            fn_bf16=hc_fn_bf16,
            plan=self._plan_for(
                "post_pre_bf16" if hc_fn_bf16 is not None else "post_pre",
                tokens,
                residual.device,
            ),
        )

    def run_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_post(
            x,
            residual,
            post,
            comb,
            plan=self._plan_for("post", int(residual.shape[0]), residual.device),
        )

    def _request_name(self, layer: object, operation: str, tokens: int) -> str:
        return f"deepseek_v4.mhc.{id(layer):x}.{operation}.m{tokens}"

    def get_b12x_preparation_units(
        self, layer: object, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        """Declare every real mHC operand after decoder weights are published."""
        from b12x.preparation import FrozenMapping

        parameters = (
            layer.hc_attn_fn_broadcast,
            layer.hc_attn_fn,
            layer.hc_ffn_fn,
            layer.hc_ffn_fn_bf16,
            layer.hc_attn_scale,
            layer.hc_ffn_scale,
            layer.hc_attn_base,
            layer.hc_ffn_base,
            layer.attn_norm.weight,
            layer.ffn_norm.weight,
        )
        if any(parameter is None or parameter.is_meta for parameter in parameters):
            return ()

        key = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        if not self._plans or self._plan_key != key:
            plans: dict[tuple[str, int], object] = {}
            for tokens in key:
                for operation, plan_operation, has_fn_bf16 in (
                    ("pre", "pre", False),
                    ("post_pre", "post_pre", False),
                    ("post_pre_bf16", "post_pre", True),
                    ("post", "post", False),
                ):
                    norm = (
                        layer.attn_norm
                        if operation in ("pre", "post_pre")
                        else layer.ffn_norm
                    )
                    invocation = FrozenMapping(
                        {
                            "operation": plan_operation,
                            "has_norm_weight": plan_operation != "post",
                            "norm_weight_dtype": "bfloat16",
                            "has_fn_bf16": has_fn_bf16,
                            "lagged_mix": False,
                            "bf16x2_eligible": True,
                            "output_mode": "functional",
                            "rms_eps": self.rms_eps,
                            "hc_eps": self.hc_eps,
                            "sinkhorn_iters": self.sinkhorn_iters,
                            "norm_eps": float(norm.variance_epsilon),
                            "block_k": self.block_k,
                            "block_h": self.block_h,
                        }
                    )
                    plans[(operation, tokens)] = self._plan_factory(
                        self._caps(
                            device=layer.hc_attn_fn.device,
                            dtype=torch.bfloat16,
                            max_tokens=tokens,
                            hidden_size=self.hidden_size,
                            split_k=self.split_k,
                        ),
                        invocation=invocation,
                    )
            self._plans = plans
            self._plan_key = key

        requests = [
            self._plans[(operation, tokens)].request(
                name=self._request_name(layer, operation, tokens),
                prepare_call=self._prepare_call(layer, operation, tokens),
                benchmark_call=self._prepare_call(layer, operation, tokens),
            )
            for tokens in key
            for operation in ("pre", "post_pre", "post_pre_bf16", "post")
        ]
        return (
            B12xPreparationUnit(
                name="DeepseekV4MHC",
                key=(id(layer), self.hidden_size, key),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def _prepare_call(self, layer: object, operation: str, tokens: int):
        """Prime/benchmark only borrowed checkpoint tensors and fresh activations."""

        def prepare(state):
            from b12x.norm.mhc import _impl
            from b12x.preparation import PreparedCall

            device = layer.hc_attn_fn.device
            residual = torch.empty(
                (tokens, self.hidden_size)
                if operation == "pre"
                else (tokens, self.hc_mult, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            x = torch.empty(
                (tokens, self.hidden_size), dtype=torch.bfloat16, device=device
            )
            post = torch.empty(
                (tokens, self.hc_mult), dtype=torch.float32, device=device
            )
            comb = torch.empty(
                (tokens, self.hc_mult, self.hc_mult),
                dtype=torch.float32,
                device=device,
            )

            def produce():
                residual.normal_()
                x.normal_()
                post.normal_()
                comb.normal_()

            if operation == "pre":
                run = lambda: _impl._b12x_mhc_pre_impl(
                    residual,
                    layer.hc_attn_fn_broadcast,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.sinkhorn_iters,
                    norm_weight=layer.attn_norm.weight,
                    norm_eps=float(layer.attn_norm.variance_epsilon),
                    _state=state,
                )
            elif operation in ("post_pre", "post_pre_bf16"):
                if operation == "post_pre":
                    fn, scale, base, norm, fn_bf16 = (
                        layer.hc_attn_fn,
                        layer.hc_attn_scale,
                        layer.hc_attn_base,
                        layer.attn_norm,
                        None,
                    )
                else:
                    fn, scale, base, norm, fn_bf16 = (
                        layer.hc_ffn_fn,
                        layer.hc_ffn_scale,
                        layer.hc_ffn_base,
                        layer.ffn_norm,
                        layer.hc_ffn_fn_bf16,
                    )
                run = lambda: _impl._b12x_mhc_post_pre_impl(
                    x,
                    residual,
                    post,
                    comb,
                    fn,
                    scale,
                    base,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.sinkhorn_iters,
                    fn_bf16=fn_bf16,
                    norm_weight=norm.weight,
                    norm_eps=float(norm.variance_epsilon),
                    _state=state,
                )
            else:
                run = lambda: _impl._b12x_mhc_post_impl(
                    x, residual, post, comb, _state=state
                )
            return PreparedCall(
                run=run,
                produce=produce,
                owners=(
                    layer.hc_attn_fn_broadcast,
                    layer.hc_attn_fn,
                    layer.hc_ffn_fn,
                    layer.hc_ffn_fn_bf16,
                    layer.hc_attn_scale,
                    layer.hc_ffn_scale,
                    layer.hc_attn_base,
                    layer.hc_ffn_base,
                    layer.attn_norm.weight,
                    layer.ffn_norm.weight,
                ),
            )

        return prepare


def _get_dspark_decode_row_capacity(vllm_config: VllmConfig) -> int | None:
    """Return the largest target-verifier row count the scheduler can emit."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_dspark():
        return None
    num_speculative_tokens = int(speculative_config.num_speculative_tokens or 0)
    if num_speculative_tokens <= 0:
        return None
    scheduler_config = vllm_config.scheduler_config
    return min(
        int(scheduler_config.max_num_batched_tokens),
        int(scheduler_config.max_num_seqs) * (1 + num_speculative_tokens),
    )


def _c128a_topk_width(max_model_len: int, compress_ratio: int) -> int:
    compressed_width = cdiv(max_model_len, compress_ratio)
    return cdiv(compressed_width, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT


@cache
def _max_q_chunks(
    max_rows: int,
    width: int,
    split_chunks_for_contract: Callable[..., int],
    decode_row_capacity: int | None,
) -> int:
    return max(
        rows
        * split_chunks_for_contract(
            rows=rows,
            width=width,
            decode_row_capacity=decode_row_capacity,
        )
        for rows in range(1, max(int(max_rows), 1) + 1)
    )


def _cache_page_view(
    cache: torch.Tensor,
    page_size: int,
    name: str,
) -> torch.Tensor:
    page_nbytes = int(page_size) * _DSV4_CACHE_BYTES_PER_TOKEN
    if page_nbytes <= 0:
        raise ValueError(f"{name} page_size must be positive, got {page_size}")

    byte_cache = cache if cache.dtype == torch.uint8 else cache.view(torch.uint8)
    if byte_cache.ndim < 2:
        raise RuntimeError(
            f"{name} expected a paged cache tensor, got shape {tuple(cache.shape)}"
        )

    page_stride = int(byte_cache.stride(0))
    if page_stride < page_nbytes:
        raise RuntimeError(
            f"{name} page stride {page_stride} is smaller than its "
            f"{page_nbytes}-byte payload"
        )

    expected_stride = 1
    for dim in range(byte_cache.ndim - 1, 0, -1):
        if int(byte_cache.stride(dim)) != expected_stride:
            raise RuntimeError(
                f"{name} page payload must be contiguous, got stride "
                f"{tuple(byte_cache.stride())}"
            )
        expected_stride *= int(byte_cache.shape[dim])
    if expected_stride < page_nbytes:
        raise RuntimeError(
            f"{name} page width {expected_stride} is smaller than its "
            f"{page_nbytes}-byte payload"
        )

    return torch.as_strided(
        byte_cache,
        size=(int(byte_cache.shape[0]), page_nbytes),
        stride=(page_stride, 1),
    )


def _descriptor_from_tensor(tensor: torch.Tensor) -> dict[str, object]:
    """Mirror b12x `_tensor_metadata` without importing private helpers."""
    pointer = tensor.data_ptr()
    return {
        "shape": tuple(int(value) for value in tensor.shape),
        "stride": tuple(int(value) for value in tensor.stride()),
        "alignment": min(16, pointer & -pointer) if pointer else 16,
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _cache_page_view_key(
    cache: torch.Tensor,
    page_size: int,
) -> tuple[int, int, torch.dtype, int, tuple[int, ...], tuple[int, ...]]:
    return (
        int(cache.untyped_storage().data_ptr()),
        int(cache.storage_offset()),
        cache.dtype,
        int(page_size),
        tuple(int(dim) for dim in cache.shape),
        tuple(int(stride) for stride in cache.stride()),
    )


def _run_compressed_sparse_mla(
    *,
    q: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lens: torch.Tensor | None,
    indexed_page_size: int | None,
    mode: Literal["decode", "extend"],
    decode_row_capacity: int | None = None,
) -> None:
    module = _require_b12x_compressed_sparse_mla()
    rows, heads = int(q.shape[0]), int(q.shape[1])
    if (
        mode == "decode"
        and decode_row_capacity is not None
        and rows > int(decode_row_capacity)
    ):
        raise ValueError(
            f"B12x decode rows {rows} exceed the declared capacity "
            f"{decode_row_capacity}"
        )

    q = q.contiguous()
    swa_indices = swa_indices.contiguous()
    swa_lens = swa_lens.contiguous()
    if indexed_indices is not None:
        indexed_indices = indexed_indices.contiguous()
    if indexed_lens is not None:
        indexed_lens = indexed_lens.contiguous()

    width = int(swa_indices.shape[-1])
    if indexed_indices is not None:
        width += int(indexed_indices.shape[-1])
    max_chunks_per_row = module.split_chunks_for_contract(
        rows=max(rows, 1),
        width=max(width, 1),
        decode_row_capacity=decode_row_capacity,
    )
    # plan() keys the compile identity on the caller's storage ABI
    # (b12x >=77351c13), so describe the exact operands bound below.
    invocation = module.invocation_from_tensors(
        q=q,
        swa_k_cache=swa_k_cache,
        indexed_k_cache=indexed_k_cache,
    )
    indexed_width_serving = (
        int(indexed_indices.shape[-1]) if indexed_indices is not None else 0
    )
    plan = module.plan(
        module.Caps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=max(rows, 1),
            max_width=max(width, 1),
            swa_width=max(int(width) - indexed_width_serving, 0),
            indexed_width=indexed_width_serving,
            head_dim=_DSV4_HEAD_DIM,
            v_head_dim=_DSV4_HEAD_DIM,
            page_size=int(swa_page_size),
            indexed_page_size=(
                int(indexed_page_size) if indexed_page_size is not None else None
            ),
            max_chunks_per_row=max_chunks_per_row,
            decode_row_capacity=decode_row_capacity,
        ),
        invocation=invocation,
    )
    scratch = current_workspace_manager().get_simultaneous(
        *((s.shape, s.dtype) for s in plan.scratch_specs())
    )
    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lens,
    )
    binding.scratch.mode = mode
    module.run(
        binding=binding,
        swa_k_cache=swa_k_cache,
        swa_page_size=int(swa_page_size),
        indexed_k_cache=indexed_k_cache,
        indexed_page_size=indexed_page_size,
        attn_sink=attn_sink[:heads].contiguous(),
        sm_scale=scale,
        expected_num_q_heads=heads,
        out=output,
    )


class DeepseekV4B12xSparseMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAB12xMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekSparseSWAB12xBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAB12xMetadataBuilder]:
        return DeepseekSparseSWAB12xMetadataBuilder


class DeepseekV4B12xSparseMLABackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "B12X"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4B12xSparseMLAMetadataBuilder]:
        return DeepseekV4B12xSparseMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) in ((12, 0), (12, 1))


class DeepseekV4B12xAttention(DeepseekV4Attention):
    backend_cls = DeepseekV4B12xSparseMLABackend
    swa_backend_cls = DeepseekSparseSWAB12xBackend
    indexer_backend_cls = DeepseekV4B12xIndexerBackend
    indexer_op_cls = DeepseekV4B12xSparseIndexer

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError(
                "B12X compressed sparse MLA does not support decode context "
                "parallelism."
            )
        if parallel_config.prefill_context_parallel_size != 1:
            raise NotImplementedError(
                "B12X compressed sparse MLA does not support prefill context "
                "parallelism."
            )
        _require_b12x_compressed_sparse_mla()
        _require_b12x_wo_projection()
        self.vllm_config = vllm_config
        self._b12x_cache_page_views: dict[object, torch.Tensor] = {}
        self._b12x_wo_projection_weights: Any | None = None
        self._b12x_wo_plans: dict[int, object] = {}
        super().__init__(vllm_config, *args, **kwargs)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        for supported in (16, 32, 64, 128):
            if num_heads <= supported:
                return supported
        raise ValueError(
            f"DeepSeek V4 B12x sparse MLA does not support {num_heads} heads."
        )

    def _validate_wo_projection_tensors(self) -> tuple[int, int, int, int]:
        if not hasattr(self.wo_a, "weight_scale_inv"):
            raise RuntimeError("B12x WO-A requires wo_a.weight_scale_inv.")
        if not hasattr(self.wo_b, "weight_scale_inv"):
            raise RuntimeError("B12x WO-B requires wo_b.weight_scale_inv.")

        groups = self.n_local_groups
        heads_per_group = self.n_local_heads // groups
        group_width = heads_per_group * self.head_dim
        rank = self.o_lora_rank
        hidden = self.hidden_size
        wo_a_shape = (groups * rank, group_width)
        wo_b_shape = (hidden, groups * rank)
        wo_a_scale_shape = (
            groups * cdiv(rank, 128),
            cdiv(group_width, 128),
        )
        wo_b_scale_shape = (cdiv(hidden, 128), cdiv(groups * rank, 128))

        tensors = (
            ("WO-A weight", self.wo_a.weight, wo_a_shape),
            ("WO-B weight", self.wo_b.weight, wo_b_shape),
            ("WO-A scale", self.wo_a.weight_scale_inv, wo_a_scale_shape),
            ("WO-B scale", self.wo_b.weight_scale_inv, wo_b_scale_shape),
        )
        for name, tensor, expected_shape in tensors:
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"B12x {name} shape mismatch: expected {expected_shape}, "
                    f"got {tuple(tensor.shape)}."
                )
        if self.wo_a.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "B12x WO-A weight must be torch.float8_e4m3fn, "
                f"got {self.wo_a.weight.dtype}."
            )
        if self.wo_b.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "B12x WO-B weight must be torch.float8_e4m3fn, "
                f"got {self.wo_b.weight.dtype}."
            )
        return groups, group_width, rank, hidden

    def setup_b12x_wo_projection(self) -> None:
        if self.wo_a.weight.dtype == self.wo_b.weight.dtype == torch.bfloat16:
            return
        if self._b12x_wo_projection_weights is not None:
            return

        groups, group_width, rank, hidden = self._validate_wo_projection_tensors()
        # The packed fused owner supersedes the two generic linear owners.
        # Suppress and unpublish them so only the packed WO-projection plan is
        # collected, and keep late generic registration from republishing them.
        for child in (self.wo_a, self.wo_b):
            child.b12x_preparation_suppressed = True
            set_b12x_preparation_provider(child, None)
        module = _require_b12x_wo_projection()
        self._b12x_wo_projection_weights = module.pack_weights(
            self.wo_a.weight.detach(),
            self.wo_a.weight_scale_inv.detach(),
            self.wo_b.weight.detach(),
            self.wo_b.weight_scale_inv.detach(),
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
        )

    def _b12x_wo_plan(self, tokens: int):
        if self._b12x_wo_projection_weights is None:
            raise RuntimeError("B12x WO-A/WO-B weights were not packed after loading.")
        module = _require_b12x_wo_projection()
        weights = self._b12x_wo_projection_weights
        table = self.rotary_emb.cos_sin_cache
        plan = self._b12x_wo_plans.get(tokens)
        if plan is None:
            plan = module.plan(
                module.Caps(
                    device=table.device,
                    max_tokens=tokens,
                    groups=weights.groups,
                    group_width=weights.group_width,
                    rank=weights.rank,
                    hidden=weights.hidden,
                ),
                invocation=dict(
                    operation="inv_rope",
                    heads_per_group=self.n_local_heads // self.n_local_groups,
                    nope_dim=self.nope_head_dim,
                    rope_dim=self.rope_head_dim,
                    positions_dtype="int64",
                    cos_sin_dtype=str(table.dtype).removeprefix("torch."),
                ),
            )
            self._b12x_wo_plans[tokens] = plan
        return plan

    def get_b12x_preparation_units(
        self, layer: object, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        """Declare inv-RoPE WO plans so preparation primes them."""
        if self._b12x_wo_projection_weights is None:
            return ()
        counts = tuple(sorted(set(workload.token_counts)))

        def prepare(state):
            from b12x.preparation import PreparedCall

            device = self.rotary_emb.cos_sin_cache.device
            source = torch.empty(
                (state.query.max_tokens, self.n_local_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            positions = torch.arange(
                state.query.max_tokens, dtype=torch.int64, device=device
            )
            positions.remainder_(self.rotary_emb.cos_sin_cache.shape[0])
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in state._scratch_state.scratch_specs()
            )
            binding = state.bind_inv_rope(
                scratch=scratch,
                o=source,
                positions=positions,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                weights=self._b12x_wo_projection_weights,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
            )
            return PreparedCall(
                run=lambda: state.run_inv_rope(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(
                    self._b12x_wo_projection_weights,
                    self.rotary_emb.cos_sin_cache,
                ),
            )

        requests = [
            self._b12x_wo_plan(rows).request(
                name=f"deepseek_v4.wo.m{rows}",
                prepare_call=prepare,
                benchmark_call=prepare,
            )
            for rows in counts
        ]
        return (
            B12xPreparationUnit(
                name="DeepseekV4WOProjection",
                key=(id(self), counts),
                requests=tuple(requests),
                stage="weights",
                autotune=not workload.eager_only,
            ),
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype == self.wo_b.weight.dtype == torch.bfloat16:
            return bf16_o_proj(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.wo_a,
                self.wo_b,
                n_groups=self.n_local_groups,
                nope_dim=self.nope_head_dim,
                o_lora_rank=self.o_lora_rank,
            )
        if self._b12x_wo_projection_weights is None:
            raise RuntimeError("B12x WO-A/WO-B weights were not packed after loading.")
        from dataclasses import replace as _replace_binding

        from b12x.preparation import require_prepared

        tokens = o.shape[0]
        module = _require_b12x_wo_projection()
        plan = self._b12x_wo_plan(tokens)
        require_prepared(plan, "gemm.wo_projection", o.device)
        binding = module.bind_inv_rope(
            plan,
            scratch=current_workspace_manager().get_simultaneous(
                *((s.shape, s.dtype) for s in plan.scratch_specs())
            ),
            o=o,
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            weights=self._b12x_wo_projection_weights,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
        )
        output = torch.empty_strided(
            (tokens, self.hidden_size, 1),
            (self.hidden_size, 1, tokens * self.hidden_size),
            dtype=torch.bfloat16,
            device=o.device,
        )
        binding = _replace_binding(binding, output=output)
        retain_cuda_graph_capture_resource(binding)
        out = module.run_inv_rope(
            binding=binding, plan=plan, stream=current_stream().cuda_stream
        )
        if self.wo_b.reduce_results and self.wo_b.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _get_cache_page_view(
        self,
        cache: torch.Tensor,
        page_size: int,
        name: str,
    ) -> torch.Tensor:
        key = _cache_page_view_key(cache, page_size)
        view = self._b12x_cache_page_views.get(key)
        if view is None:
            view = _cache_page_view(cache, page_size, name)
            self._b12x_cache_page_views[key] = view

    def _reserve_profile_workspace(self, q: torch.Tensor) -> None:
        module = _require_b12x_compressed_sparse_mla()
        indexed_width = 0
        if self.compress_ratio == 4:
            if self.topk_indices_buffer is not None:
                indexed_width = int(self.topk_indices_buffer.shape[-1])
            elif self.indexer is not None:
                indexed_width = int(self.indexer.topk_tokens)
        elif self.compress_ratio > 1:
            indexed_width = _c128a_topk_width(
                self.max_model_len,
                self.compress_ratio,
            )

        swa_width = int(self.window_size) + self.max_image_tokens
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is not None and speculative_config.use_dspark():
            swa_width = max(
                swa_width,
                get_dspark_swa_index_width(
                    int(self.window_size),
                    speculative_config.num_speculative_tokens or 0,
                ),
            )
        width = max(swa_width + indexed_width, 1)
        rows = max(int(self.max_num_batched_tokens), 1)
        decode_row_capacity = _get_dspark_decode_row_capacity(self.vllm_config)
        max_chunks_per_row = module.split_chunks_for_contract(
            rows=rows,
            width=width,
            decode_row_capacity=decode_row_capacity,
        )
        max_q_chunks = _max_q_chunks(
            rows,
            width,
            module.split_chunks_for_contract,
            decode_row_capacity,
        )
        page_size = int(self.swa_cache_layer.block_size)
        # plan() keys the compile identity on the caller's storage ABI
        # (b12x >=77351c13). Memory profiling pass 1 runs before any KV cache is
        # allocated (the minimal profiling cache binds only inside
        # _profile_deepseek_v4_attention), so when the SWA layer still holds its
        # empty placeholder, synthesize the descriptors from the layer's static
        # geometry instead of reading a live tensor. Reserved workspace sizing
        # depends only on Caps, so the synthesized shape never changes what the
        # reservation holds.
        swa_kv_cache = self.swa_cache_layer.kv_cache
        page_nbytes = page_size * _DSV4_CACHE_BYTES_PER_TOKEN
        swa_meta = (
            {
                "shape": (1, page_nbytes),
                "stride": (page_nbytes, 1),
                "alignment": 16,
                "dtype": str(self.swa_cache_layer.dtype).removeprefix("torch."),
            }
            if swa_kv_cache.ndim < 2
            else None
        )
        if indexed_width:
            indexed_page_nbytes = (
                page_size // self.compress_ratio
            ) * _DSV4_CACHE_BYTES_PER_TOKEN
            indexed_meta = {
                "shape": (1, indexed_page_nbytes),
                "stride": (indexed_page_nbytes, 1),
                "alignment": 16,
                "dtype": str(self.swa_cache_layer.dtype).removeprefix("torch."),
            }
        else:
            indexed_meta = None
        invocation = module.invocation_from_descriptors(
            q=_descriptor_from_tensor(q),
            swa_cache=(
                swa_meta
                if swa_meta is not None
                else _descriptor_from_tensor(
                    self._get_cache_page_view(
                        swa_kv_cache,
                        page_size,
                        "swa_k_cache",
                    )
                )
            ),
            indexed_cache=(
                indexed_meta
                if indexed_meta is not None
                else (
                    _descriptor_from_tensor(
                        self._get_cache_page_view(
                            compressed_cache,
                            page_size // self.compress_ratio,
                            "indexed_k_cache",
                        )
                    )
                    if (compressed_cache := self.kv_cache) is not None
                    and self.compress_ratio > 1
                    else None
                )
            ),
            attn_sink_present=False,
            return_lse=False,
            lse_scale="base2",
            output_mode="internal",
        )
        plan = module.plan(
            module.Caps(
                device=q.device,
                num_q_heads=int(q.shape[1]),
                max_q_rows=rows,
                max_width=width,
                # Split-width declaration keeps Caps.indexed_width equal to the
                # local indexed_width that gates the descriptor below. The
                # legacy both-None normalization would instead set
                # Caps.indexed_width = max_width, forcing the cache descriptor
                # present even when this layer declares none.
                swa_width=max(width - indexed_width, 0),
                indexed_width=indexed_width,
                head_dim=_DSV4_HEAD_DIM,
                v_head_dim=_DSV4_HEAD_DIM,
                page_size=page_size,
                max_chunks_per_row=max_chunks_per_row,
                max_q_chunks=max_q_chunks,
                decode_row_capacity=decode_row_capacity,
            ),
            invocation=invocation,
        )
        current_workspace_manager().get_simultaneous(
            *((s.shape, s.dtype) for s in plan.scratch_specs())
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv, positions
        if output.shape != q.shape or output.dtype != q.dtype:
            raise RuntimeError(
                f"B12x output {output.shape}/{output.dtype} must match "
                f"q {q.shape}/{q.dtype}."
            )

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            output.zero_()
            self._reserve_profile_workspace(q)
            return

        assert isinstance(attn_metadata, dict)
        sparse_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        compressed_cache = self.kv_cache if self.compress_ratio > 1 else None
        if swa_metadata.num_prefills > 0:
            prefill_end = num_decode_tokens + num_prefill_tokens
            self._forward_prefill(
                q=q[num_decode_tokens:prefill_end],
                output=output[num_decode_tokens:prefill_end],
                compressed_cache=compressed_cache,
                sparse_metadata=sparse_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                output=output[:num_decode_tokens],
                compressed_cache=compressed_cache,
                sparse_metadata=sparse_metadata,
                swa_metadata=swa_metadata,
            )

    def _indexed_region(
        self,
        *,
        compressed_cache: torch.Tensor | None,
        sparse_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        token_slice: slice,
        decode: bool,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int | None,
    ]:
        if self.compress_ratio <= 1:
            return None, None, None, None
        assert compressed_cache is not None
        assert sparse_metadata is not None
        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.token_to_req_indices is not None

        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            local_indices = self.topk_indices_buffer[token_slice]
        elif decode:
            local_indices = sparse_metadata.c128a_global_decode_topk_indices
            assert local_indices is not None
            topk_lens = sparse_metadata.c128a_decode_topk_lens
            assert topk_lens is not None
            page_size = sparse_metadata.block_size // self.compress_ratio
            cache_view = self._get_cache_page_view(
                compressed_cache, page_size, "indexed_k_cache"
            )
            return cache_view, local_indices, topk_lens, page_size
        else:
            local_indices = sparse_metadata.c128a_prefill_topk_indices
            assert local_indices is not None

        page_size = sparse_metadata.block_size // self.compress_ratio
        global_indices, topk_lens = compute_global_topk_indices_and_lens(
            local_indices,
            swa_metadata.token_to_req_indices[token_slice],
            sparse_metadata.block_table,
            page_size,
            swa_metadata.is_valid_token[token_slice],
        )
        cache_view = self._get_cache_page_view(
            compressed_cache, page_size, "indexed_k_cache"
        )
        return cache_view, global_indices, topk_lens, page_size

    def _forward_decode(
        self,
        *,
        q: torch.Tensor,
        output: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        sparse_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        num_tokens = swa_metadata.num_decode_tokens
        indexed_cache, indexed_indices, indexed_lens, indexed_page_size = (
            self._indexed_region(
                compressed_cache=compressed_cache,
                sparse_metadata=sparse_metadata,
                swa_metadata=swa_metadata,
                token_slice=slice(0, num_tokens),
                decode=True,
            )
        )
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        swa_cache = self._get_cache_page_view(
            self.swa_cache_layer.kv_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )
        _run_compressed_sparse_mla(
            q=q,
            output=output,
            attn_sink=self.attn_sink,
            scale=self.scale,
            swa_k_cache=swa_cache,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_page_size=swa_metadata.block_size,
            indexed_k_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_lens=indexed_lens,
            indexed_page_size=indexed_page_size,
            mode="decode",
            decode_row_capacity=_get_dspark_decode_row_capacity(self.vllm_config),
        )

    def _forward_prefill(
        self,
        *,
        q: torch.Tensor,
        output: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        sparse_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        token_slice = slice(
            num_decode_tokens,
            num_decode_tokens + num_prefill_tokens,
        )
        indexed_cache, indexed_indices, indexed_lens, indexed_page_size = (
            self._indexed_region(
                compressed_cache=compressed_cache,
                sparse_metadata=sparse_metadata,
                swa_metadata=swa_metadata,
                token_slice=token_slice,
                decode=False,
            )
        )
        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        assert swa_metadata.query_start_loc_cpu is not None
        swa_cache = self._get_cache_page_view(
            self.swa_cache_layer.kv_cache,
            swa_metadata.block_size,
            "swa_k_cache",
        )

        num_decodes = swa_metadata.num_decodes
        prefill_base = swa_metadata.query_start_loc_cpu[num_decodes]
        for request_start in range(
            0,
            swa_metadata.num_prefills,
            self.PREFILL_CHUNK_SIZE,
        ):
            request_end = min(
                request_start + self.PREFILL_CHUNK_SIZE,
                swa_metadata.num_prefills,
            )
            query_start = (
                swa_metadata.query_start_loc_cpu[num_decodes + request_start]
                - prefill_base
            )
            query_end = (
                swa_metadata.query_start_loc_cpu[num_decodes + request_end]
                - prefill_base
            )
            _run_compressed_sparse_mla(
                q=q[query_start:query_end],
                output=output[query_start:query_end],
                attn_sink=self.attn_sink,
                scale=self.scale,
                swa_k_cache=swa_cache,
                swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                swa_page_size=swa_metadata.block_size,
                indexed_k_cache=indexed_cache,
                indexed_indices=(
                    indexed_indices[query_start:query_end]
                    if indexed_indices is not None
                    else None
                ),
                indexed_lens=(
                    indexed_lens[query_start:query_end]
                    if indexed_lens is not None
                    else None
                ),
                indexed_page_size=indexed_page_size,
                mode="extend",
            )
