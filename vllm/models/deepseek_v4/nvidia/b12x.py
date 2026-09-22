# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x compressed sparse MLA for DeepSeek V4."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

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
    b12x_layer,
    b12x_layer_prefix,
    get_b12x_compressed_sparse_mla,
    get_b12x_mhc,
    get_b12x_scratch_buffers,
    get_b12x_wo_projection,
    register_b12x_layer,
    set_b12x_preparation_provider,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.worker.workspace import (
    retain_cuda_graph_capture_resource,
)

if TYPE_CHECKING:
    from b12x.preparation import Plan

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
    for name in (
        "Caps",
        "plan",
        "bind",
        "run",
        "invocation_from_descriptors",
        "split_chunks_for_contract",
    ):
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
    for name in ("Caps", "plan", "pack_weights", "bind_inv_rope", "run_inv_rope"):
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


@dataclass(frozen=True)
class MHCOperands:
    """Decoder-layer attribute names that B12xMHCResidual prepares against.

    The defaults are DeepSeek V4's. A model whose layers name their norms
    differently, or that keeps no BF16 copy of the FFN mixing projection,
    declares that here so preparation reads the operands the layer really
    runs with and declares only the operations it executes.
    """

    attn_norm: str = "attn_norm"
    ffn_norm: str = "ffn_norm"
    ffn_fn_bf16: str | None = "hc_ffn_fn_bf16"


class B12xMHCResidual:
    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        operands: MHCOperands | None = None,
    ) -> None:
        module = _require_b12x_mhc()
        self.operands = operands if operands is not None else MHCOperands()
        self._caps = module.Caps
        self._plan_factory = module.plan
        self._run_pre = module.run_pre
        self._run_post = module.run_post
        self._run_post_pre = module.run_post_pre
        self._plans: dict[tuple[str, int], Plan] = {}

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

    def _plan_for(self, operation: str, tokens: int) -> Plan:
        tokens = int(tokens)
        exact = self._plans.get((operation, tokens))
        if exact is not None:
            return exact
        capacity = max((rows for op, rows in self._plans if op == operation), default=0)
        if 0 <= tokens <= capacity and capacity:
            return self._plans[(operation, capacity)]
        raise RuntimeError(
            f"B12x mHC {operation} live M={tokens} exceeds prepared capacity {capacity}"
        )

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
            plan=self._plan_for("pre", int(residual.shape[0])),
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
            plan=self._plan_for("post", int(residual.shape[0])),
        )

    def _request_name(self, layer: torch.nn.Module, operation: str, tokens: int) -> str:
        return f"deepseek_v4.mhc.{id(layer):x}.{operation}.m{tokens}"

    def _attn_norm(self, layer: torch.nn.Module) -> Any:
        return getattr(layer, self.operands.attn_norm)

    def _ffn_norm(self, layer: torch.nn.Module) -> Any:
        return getattr(layer, self.operands.ffn_norm)

    def _ffn_fn_bf16(self, layer: torch.nn.Module) -> torch.Tensor | None:
        name = self.operands.ffn_fn_bf16
        return None if name is None else getattr(layer, name)

    def _operations(self, layer: torch.nn.Module) -> tuple[str, ...]:
        """The mHC operations ``layer`` executes, in declaration order.

        Only the first decoder layer holds the published broadcast attention
        projection and runs ``pre``; every layer runs the fused ``post_pre``
        and may end the stream with ``post``. ``post_pre_bf16`` exists only
        when the model keeps a BF16 copy of the FFN mixing projection.
        """
        operations: tuple[str, ...] = (
            ("pre",) if layer.hc_attn_fn_broadcast is not None else ()
        )
        operations += ("post_pre",)
        if self.operands.ffn_fn_bf16 is not None:
            operations += ("post_pre_bf16",)
        return operations + ("post",)

    def _operands(
        self, layer: torch.nn.Module, operations: tuple[str, ...]
    ) -> tuple[torch.Tensor | None, ...]:
        """Every checkpoint tensor the declared ``operations`` read."""
        operands: tuple[torch.Tensor | None, ...] = (
            layer.hc_attn_fn,
            layer.hc_ffn_fn,
            layer.hc_attn_scale,
            layer.hc_ffn_scale,
            layer.hc_attn_base,
            layer.hc_ffn_base,
            self._attn_norm(layer).weight,
            self._ffn_norm(layer).weight,
        )
        if "pre" in operations:
            operands += (layer.hc_attn_fn_broadcast,)
        if "post_pre_bf16" in operations:
            operands += (self._ffn_fn_bf16(layer),)
        return operands

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        """Declare every real mHC operand after decoder weights are published."""
        from b12x.preparation import FrozenMapping

        operations = self._operations(layer)
        if any(
            operand is None or operand.is_meta
            for operand in self._operands(layer, operations)
        ):
            return ()

        key = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        for tokens in key:
            for operation in operations:
                if (operation, tokens) in self._plans:
                    continue
                has_fn_bf16 = operation == "post_pre_bf16"
                plan_operation = "post_pre" if has_fn_bf16 else operation
                norm = (
                    self._attn_norm(layer)
                    if operation in ("pre", "post_pre")
                    else self._ffn_norm(layer)
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
                self._plans[(operation, tokens)] = self._plan_factory(
                    self._caps(
                        device=layer.hc_attn_fn.device,
                        dtype=torch.bfloat16,
                        max_tokens=tokens,
                        hidden_size=self.hidden_size,
                        split_k=self.split_k,
                    ),
                    invocation=invocation,
                )

        requests = [
            self._plans[(operation, tokens)].request(
                name=self._request_name(layer, operation, tokens),
                prepare_call=self._prepare_call(layer, operation, tokens),
                benchmark_call=self._prepare_call(layer, operation, tokens),
            )
            for tokens in key
            for operation in operations
        ]
        return (
            B12xPreparationUnit(
                name="DeepseekV4MHC",
                key=(id(layer), self.hidden_size, key),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def _prepare_call(self, layer: torch.nn.Module, operation: str, tokens: int):
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
            # Each candidate owns one output set. Functional allocations inside
            # repeated capture samples would reserve additional graph-pool copies
            # that are absent from the tuner's primed-residency measurement.
            residual_out = torch.empty(
                (tokens, self.hc_mult, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            outputs = (
                {}
                if operation == "post"
                else {
                    "residual_out": residual_out,
                    "y_out": torch.empty_like(x),
                    "post_out": torch.empty_like(post),
                    "comb_out": torch.empty_like(comb),
                }
            )

            def produce():
                residual.normal_()
                x.normal_()
                post.normal_()
                comb.normal_()

            if operation == "pre":
                attn_norm = self._attn_norm(layer)
                run = lambda: _impl._b12x_mhc_pre_impl(
                    residual,
                    layer.hc_attn_fn_broadcast,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.sinkhorn_iters,
                    norm_weight=attn_norm.weight,
                    norm_eps=float(attn_norm.variance_epsilon),
                    _state=state,
                    **outputs,
                )
            elif operation in ("post_pre", "post_pre_bf16"):
                if operation == "post_pre":
                    fn, scale, base, norm, fn_bf16 = (
                        layer.hc_attn_fn,
                        layer.hc_attn_scale,
                        layer.hc_attn_base,
                        self._attn_norm(layer),
                        None,
                    )
                else:
                    fn, scale, base, norm, fn_bf16 = (
                        layer.hc_ffn_fn,
                        layer.hc_ffn_scale,
                        layer.hc_ffn_base,
                        self._ffn_norm(layer),
                        self._ffn_fn_bf16(layer),
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
                    **outputs,
                )
            else:
                run = lambda: _impl._b12x_mhc_post_impl(
                    x, residual, post, comb, out=residual_out, _state=state
                )

            def execute():
                # Trial outputs must not keep a full prefill result resident
                # for every candidate before the timing graphs are captured.
                run()

            return PreparedCall(
                run=execute,
                produce=produce,
                owners=self._operands(layer, self._operations(layer)),
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


def _c128a_profile_widths(max_width: int) -> tuple[int, ...]:
    """Widths emitted by C128 metadata's power-of-two, capacity-capped views."""
    widths = {max_width}
    width = _C128A_TOPK_ALIGNMENT
    while width < max_width:
        widths.add(width)
        width *= 2
    return tuple(sorted(widths))


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
    plan: Plan,
) -> None:
    from b12x.preparation import require_prepared

    module = _require_b12x_compressed_sparse_mla()
    heads = int(q.shape[1])
    require_prepared(plan, "attention.compressed_sparse_mla", q.device)

    q = q.contiguous()
    swa_indices = swa_indices.contiguous()
    swa_lens = swa_lens.contiguous()
    if indexed_indices is not None:
        indexed_indices = indexed_indices.contiguous()
    if indexed_lens is not None:
        indexed_lens = indexed_lens.contiguous()

    binding = module.bind(
        plan,
        scratch=get_b12x_scratch_buffers(plan),
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lens,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lens,
    )
    retain_cuda_graph_capture_resource(binding)
    module.run(
        plan=plan,
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


def _b12x_dsv4_wo_projection(
    o: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer = b12x_layer(_resolve_layer_name(layer_name))
    layer._run_b12x_wo_projection(o, positions, output)


def _b12x_dsv4_wo_projection_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    pass


direct_register_custom_op(
    op_name="b12x_dsv4_wo_projection",
    op_func=_b12x_dsv4_wo_projection,
    mutates_args=["output"],
    fake_impl=_b12x_dsv4_wo_projection_fake,
)


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
        self._b12x_wo_plans: dict[int, Any] = {}
        self._b12x_mla_plans: dict[tuple[str, int, int, int], Plan] = {}
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
        set_b12x_preparation_provider(self, self)
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
        prefix = b12x_layer_prefix(self)
        self._b12x_wo_layer_name = _encode_layer_name(prefix)
        register_b12x_layer(prefix, self)

    def _b12x_wo_plan(self, rows: int):
        plan = self._b12x_wo_plans.get(rows)
        if plan is None:
            weights = self._b12x_wo_projection_weights
            if weights is None:
                raise RuntimeError(
                    "B12x WO-A/WO-B weights were not packed after loading."
                )
            module = _require_b12x_wo_projection()
            table = self.rotary_emb.cos_sin_cache
            plan = module.plan(
                module.Caps(
                    device=table.device,
                    max_tokens=rows,
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
                    sfb_k_replicated=weights.sfb_k_replicated,
                    wo_a_tiled=weights.wo_a.values_tiled is not None,
                    wo_b_tiled=weights.wo_b.values_tiled is not None,
                ),
            )
            self._b12x_wo_plans[rows] = plan
        return plan

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        from b12x.preparation import PreparedCall

        if layer is not self:
            raise ValueError("B12x attention preparation owner mismatch")
        if workload.stage == "state":
            if self.indexer is not None and self.indexer.k_cache.kv_cache.numel():
                self.indexer.indexer_op.set_b12x_index_cache(
                    self.indexer.k_cache.kv_cache,
                    num_q_heads=self.indexer.n_head,
                )
            return self._b12x_mla_preparation_units(workload)
        if workload.stage != "weights" or self._b12x_wo_projection_weights is None:
            return ()
        weights = self._b12x_wo_projection_weights
        table = self.rotary_emb.cos_sin_cache

        def prepare(state):
            rows = state.query.max_tokens
            source = torch.empty(
                (rows, self.n_local_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=table.device,
            )
            positions = torch.arange(rows, dtype=torch.int64, device=table.device)
            positions.remainder_(table.shape[0])
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in state._scratch_state.scratch_specs()
            )
            binding = state.bind_inv_rope(
                scratch=scratch,
                o=source,
                positions=positions,
                cos_sin_cache=table,
                weights=weights,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
            )
            return PreparedCall(
                run=lambda: state.run_inv_rope(binding),
                produce=lambda: source.normal_(std=0.25),
                owners=(weights, table),
            )

        requests = tuple(
            self._b12x_wo_plan(rows).request(
                name=f"{self.prefix}.wo.m{rows}",
                prepare_call=prepare,
                benchmark_call=prepare,
            )
            for rows in workload.token_counts
        )
        return (
            B12xPreparationUnit(
                name="DeepseekV4WOProjection",
                key=self.prefix,
                requests=requests,
                stage="weights",
            ),
        )

    def _b12x_mla_preparation_units(
        self, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        swa_cache = self.swa_cache_layer.kv_cache
        indexed_cache = self.kv_cache if self.compress_ratio > 1 else None
        if swa_cache.numel() == 0 or (
            indexed_cache is not None and indexed_cache.numel() == 0
        ):
            return ()
        swa_page_size = int(self.swa_cache_layer.block_size)
        indexed_page_size = (
            int(self.vllm_config.cache_config.block_size) // self.compress_ratio
            if indexed_cache is not None
            else swa_page_size
        )
        caches = (swa_cache,) if indexed_cache is None else (swa_cache, indexed_cache)
        cache_key = tuple(
            _cache_page_view_key(cache, page_size)
            for cache, page_size in zip(caches, (swa_page_size, indexed_page_size))
        )
        key = (
            cache_key,
            workload.max_tokens,
            workload.max_seqs,
            workload.fixed_token_counts,
            workload.max_model_len,
        )
        if key != getattr(self, "_b12x_mla_plan_key", None):
            self._b12x_cache_page_views.clear()
            self._b12x_mla_plans = {}
            self._b12x_mla_plan_key = key
        swa_cache = self._get_cache_page_view(swa_cache, swa_page_size, "swa_k_cache")
        if indexed_cache is not None:
            indexed_cache = self._get_cache_page_view(
                indexed_cache, indexed_page_size, "indexed_k_cache"
            )
        indexed_widths: tuple[int, ...]
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            indexed_widths = (int(self.topk_indices_buffer.shape[-1]),)
        elif self.compress_ratio > 1:
            indexed_widths = _c128a_profile_widths(
                _c128a_topk_width(workload.max_model_len, self.compress_ratio)
            )
        else:
            indexed_widths = (0,)
        decode_widths = {int(self.window_size)}
        spec = self.vllm_config.speculative_config
        if spec is not None and spec.use_dspark():
            decode_widths.add(
                get_dspark_swa_index_width(
                    self.window_size, spec.num_speculative_tokens or 0
                )
            )
        decode_rows = min(
            workload.max_tokens,
            max(
                workload.max_seqs,
                *workload.fixed_token_counts,
                _get_dspark_decode_row_capacity(self.vllm_config) or 0,
            ),
        )
        self._b12x_mla_prefill_rows = workload.max_tokens
        regimes = [
            ("decode", rows, width)
            for rows in range(1, decode_rows + 1)
            for width in sorted(decode_widths)
        ]
        regimes.append(
            (
                "extend",
                workload.max_tokens,
                int(self.window_size + self.max_image_tokens),
            )
        )
        module = _require_b12x_compressed_sparse_mla()

        def descriptor(tensor):
            return dict(
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                alignment=min(16, tensor.data_ptr() & -tensor.data_ptr()),
                dtype=str(tensor.dtype).removeprefix("torch."),
            )

        requests = []
        for mode, rows, swa_width in regimes:
            for indexed_width in indexed_widths:
                plan_key = (mode, rows, swa_width, indexed_width)
                plan = self._b12x_mla_plans.get(plan_key)
                if plan is None:
                    plan = module.plan(
                        module.Caps(
                            device=swa_cache.device,
                            num_q_heads=self.padded_heads,
                            max_q_rows=rows,
                            max_width=swa_width + indexed_width,
                            mode=mode,
                            swa_width=swa_width,
                            indexed_width=indexed_width,
                            swa_page_size=swa_page_size,
                            indexed_page_size=indexed_page_size,
                            decode_row_capacity=(
                                _get_dspark_decode_row_capacity(self.vllm_config)
                                if mode == "decode"
                                else None
                            ),
                            use_cuda_graph=mode == "decode",
                        ),
                        invocation=module.invocation_from_descriptors(
                            q=dict(
                                shape=(rows, self.padded_heads, _DSV4_HEAD_DIM),
                                stride=(
                                    self.padded_heads * _DSV4_HEAD_DIM,
                                    _DSV4_HEAD_DIM,
                                    1,
                                ),
                                alignment=16,
                                dtype="bfloat16",
                            ),
                            swa_cache=descriptor(swa_cache),
                            indexed_cache=(
                                None
                                if indexed_cache is None
                                else descriptor(indexed_cache)
                            ),
                            attn_sink_present=True,
                            output_mode="provided",
                        ),
                    )
                    self._b12x_mla_plans[plan_key] = plan
                prepare = self._b12x_mla_call(swa_cache, indexed_cache)
                requests.append(
                    plan.request(
                        name=f"{self.prefix}.mla.{mode}.m{rows}.s{swa_width}.i{indexed_width}",
                        prepare_call=prepare,
                        benchmark_call=prepare,
                    )
                )
        return (
            B12xPreparationUnit(
                name="DeepseekV4CompressedMLA",
                key=(self.prefix, key),
                requests=tuple(requests),
                stage="state",
            ),
        )

    def _b12x_mla_call(self, swa_cache, indexed_cache):
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        borrowers = 0

        def prepare(state):
            nonlocal borrowers
            from b12x.preparation import PreparedCall

            query = state.query
            device = swa_cache.device
            q = torch.empty(
                (query.query_rows, query.num_q_heads, _DSV4_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            output = torch.empty_like(q)
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in state.scratch_plan.scratch_specs()
            )
            cache_prefixes = []
            producers = []
            identity_rope = torch.zeros((1, 64), dtype=torch.float32, device=device)
            identity_rope[:, :32] = 1

            def cache_inputs(cache, page_size, width):
                pages = min(cache.shape[0], cdiv(width, page_size))
                live = cache[:pages]
                cache_prefixes.append(live)
                count = min(width, pages * page_size)
                source = torch.empty(
                    (count, _DSV4_HEAD_DIM), dtype=torch.bfloat16, device=device
                )
                slots = torch.arange(count, dtype=torch.int64, device=device)
                positions = torch.zeros(count, dtype=torch.int64, device=device)
                q_output = torch.empty(
                    (count, 8, _DSV4_HEAD_DIM), dtype=torch.bfloat16, device=device
                )
                producers.append((source, cache, slots, page_size, positions, q_output))
                indices = torch.full(
                    (query.query_rows, width), -1, dtype=torch.int32, device=device
                )
                indices[:, :count] = slots.to(torch.int32)
                lengths = torch.full(
                    (query.query_rows,), count, dtype=torch.int32, device=device
                )
                return indices, lengths

            swa_indices, swa_lengths = cache_inputs(
                swa_cache, query.swa_page_size, query.swa_width
            )
            indexed_indices = indexed_lengths = None
            if indexed_cache is not None:
                indexed_indices, indexed_lengths = cache_inputs(
                    indexed_cache, query.indexed_page_size, query.indexed_width
                )
            binding = state.bind_for_preparation(
                scratch=scratch,
                q=q,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                indexed_indices=indexed_indices,
                indexed_lengths=indexed_lengths,
            )
            if borrowers == 0:
                snapshots.extend((live, live.clone()) for live in cache_prefixes)
            borrowers += 1

            def produce():
                q.normal_(std=0.25)
                for (
                    source,
                    page_cache,
                    slots,
                    page_size,
                    positions,
                    q_output,
                ) in producers:
                    source.normal_(std=0.25)
                    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.out(
                        source.unsqueeze(1),
                        source,
                        q_output,
                        page_cache,
                        slots,
                        positions,
                        identity_rope,
                        1e-6,
                        page_size,
                    )

            def restore():
                nonlocal borrowers
                for live, saved in snapshots:
                    live.copy_(saved)
                borrowers -= 1
                if borrowers == 0:
                    snapshots.clear()

            return PreparedCall(
                run=lambda: state.run(
                    binding,
                    swa_k_cache=swa_cache,
                    indexed_k_cache=indexed_cache,
                    swa_page_size=query.swa_page_size,
                    indexed_page_size=query.indexed_page_size,
                    attn_sink=self.attn_sink,
                    sm_scale=self.scale,
                    out=output,
                ),
                produce=produce,
                reset=lambda: output.fill_(float("nan")),
                restore=restore,
                output=output,
                owners=(swa_cache, indexed_cache, self.attn_sink),
            )

        return prepare

    def _b12x_mla_plan(self, mode, rows, swa_indices, indexed_indices):
        planned_rows = rows if mode == "decode" else self._b12x_mla_prefill_rows
        key = (
            mode,
            planned_rows,
            int(swa_indices.shape[-1]),
            0 if indexed_indices is None else int(indexed_indices.shape[-1]),
        )
        plan = self._b12x_mla_plans.get(key)
        if plan is None:
            raise RuntimeError(
                f"B12x compressed MLA has no prepared declaration for {key}"
            )
        if rows > plan.query.query_rows:
            raise ValueError(
                f"B12x compressed MLA {mode} rows {rows} exceed declared capacity "
                f"{plan.query.query_rows}"
            )
        return plan

    def _run_b12x_wo_projection(
        self, o: torch.Tensor, positions: torch.Tensor, output: torch.Tensor
    ) -> None:
        from b12x.preparation import require_prepared

        rows = int(o.shape[0])
        plan = self._b12x_wo_plan(rows)
        require_prepared(plan, "gemm.wo_projection", o.device)
        module = _require_b12x_wo_projection()
        binding = module.bind_inv_rope(
            plan,
            scratch=get_b12x_scratch_buffers(plan),
            o=o,
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            weights=self._b12x_wo_projection_weights,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
        )
        binding = replace(
            binding,
            output=output.as_strided(
                (rows, self.hidden_size, 1),
                (self.hidden_size, 1, rows * self.hidden_size),
            ),
        )
        retain_cuda_graph_capture_resource(binding)
        module.run_inv_rope(
            binding=binding, plan=plan, stream=current_stream().cuda_stream
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
        out = torch.empty(
            (o.shape[0], self.hidden_size), dtype=torch.bfloat16, device=o.device
        )
        torch.ops.vllm.b12x_dsv4_wo_projection(
            o, positions, out, self._b12x_wo_layer_name
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
        return view

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
            # State preparation reserves each declared operation's scratch
            # before KV admission. This metadata-free pass skips attention.
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
        swa_metadata: DeepseekSparseSWAMetadata,
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
        swa_metadata: DeepseekSparseSWAMetadata,
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
            plan=self._b12x_mla_plan(
                "decode",
                int(q.shape[0]),
                swa_metadata.decode_swa_indices,
                indexed_indices,
            ),
        )

    def _forward_prefill(
        self,
        *,
        q: torch.Tensor,
        output: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        sparse_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
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
                plan=self._b12x_mla_plan(
                    "extend",
                    query_end - query_start,
                    swa_metadata.prefill_swa_indices,
                    indexed_indices,
                ),
            )
