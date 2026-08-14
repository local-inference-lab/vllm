# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.b12x_mxfp8_bmm import (
    warmup_b12x_mla_mxfp8_bmm,
    warmup_fused_mla_query,
)
from vllm.model_executor.kernels.linear.mxfp8.b12x import warmup_b12x_mxfp8_linear
from vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor import (
    warmup_b12x_tensor_fp8_linear,
)
from vllm.model_executor.layers.fused_moe.b12x_moe import warmup_b12x_moe_dynamic
from vllm.model_executor.layers.quantization.exl3 import (
    warmup_exl3_mixed_trellis_route_pack,
)
from vllm.model_executor.warmup.b12x_sparse_indexer_warmup import (
    warmup_b12x_sparse_indexer,
)
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
    minimax_m3_msa_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
    sparse_mla_triton_warmup_if_needed,
)
from vllm.model_executor.warmup.v1_block_table_warmup import (
    warm_v1_block_table_kernels,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LL_BF16_WARMUP_MODEL_SHAPES: tuple[tuple[int, int], ...] = (
    (6144, 264),  # Inkling
    (7168, 256),  # DSV3
    (7168, 384),  # DSV4-Pro
    (14400, 256),  # DSV4-Flash
)
_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _warmup_ll_bf16_router_gemm() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    logger.info("Warming up ll_bf16 router GEMM kernels.")
    ll_bf16_gemm_kernel.warmup(
        shapes=_LL_BF16_WARMUP_MODEL_SHAPES,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def _is_flashinfer_backend(backend) -> bool:
    try:
        return backend.get_name() == "FLASHINFER"
    except NotImplementedError:
        return False


def _is_flashinfer_object(obj: object) -> bool:
    cls = obj.__class__
    name = cls.__name__.lower()
    module = cls.__module__.lower()
    return "flashinfer" in name or "flashinfer" in module


def _contains_flashinfer_object(
    obj: object,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> bool:
    if obj is None or isinstance(obj, (str, bytes, int, float, bool, torch.Tensor)):
        return False
    if _is_flashinfer_object(obj):
        return True
    if depth >= 3:
        return False
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    if isinstance(obj, nn.Module):
        return False
    values: Iterable[object]
    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)):
        values = obj
    elif hasattr(obj, "__dict__"):
        values = vars(obj).values()
    else:
        return False

    return any(
        _contains_flashinfer_object(value, depth=depth + 1, seen=seen)
        for value in values
    )


def _uses_flashinfer_attention(runner: "GPUModelRunner") -> bool:
    attn_groups = getattr(runner, "attn_groups", None)
    return bool(
        attn_groups
        and any(
            _is_flashinfer_backend(group.backend)
            for groups in attn_groups
            for group in groups
        )
    )


def _uses_flashinfer_model_kernels(model: nn.Module) -> bool:
    for module in model.modules():
        if _is_flashinfer_object(module):
            return True
        if any(
            _contains_flashinfer_object(value)
            for value in vars(module).values()
            if not isinstance(value, nn.Module)
        ):
            return True
    return False


def _uses_flashinfer_compute_kernels(worker: "Worker") -> bool:
    return _uses_flashinfer_attention(
        worker.model_runner
    ) or _uses_flashinfer_model_kernels(worker.get_model())


def _warmup_b12x_dcp_a2a(worker: "Worker") -> int:
    if not envs.VLLM_USE_B12X_DCP_A2A:
        return 0
    parallel_config = getattr(worker.vllm_config, "parallel_config", None)
    if parallel_config is None:
        return 0
    dcp_world_size = parallel_config.decode_context_parallel_size
    if dcp_world_size <= 1 or parallel_config.dcp_comm_backend != "a2a":
        return 0

    # Pooling models do not use the generation slot-mapping path.
    if not worker.use_v2_model_runner and not worker.model_runner.is_pooling_model:
        warm_v1_block_table_kernels(
            getattr(worker.model_runner, "device", torch.device("cuda")),
            worker.scheduler_config.max_num_batched_tokens,
        )
    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    from vllm.distributed.parallel_state import get_dcp_group
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.models.deepseek_v4.nvidia.b12x import (
        DeepseekV4B12xMLAAttention,
    )
    from vllm.v1.attention.ops.dcp_alltoall import warmup_b12x_dcp_a2a

    model = worker.get_model()
    candidates = list(model.modules())
    candidates.extend(
        worker.vllm_config.compilation_config.static_forward_context.values()
    )
    seen_modules: set[int] = set()
    warmed_signatures: set[tuple[torch.device, torch.dtype, int, int, int]] = set()
    for module in candidates:
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))

        dtype = worker.model_config.dtype
        if isinstance(module, DeepseekV4B12xMLAAttention):
            device = module.attn_sink.device
            total_heads = int(module.n_local_heads) * dcp_world_size
            query_head_dim = int(module.head_dim)
            output_head_dim = int(module.head_dim)
        elif isinstance(module, MLAAttention) and module.dcp_b12x:
            device = next(module.parameters()).device
            total_heads = int(module.num_heads) * dcp_world_size
            query_head_dim = int(module.kv_lora_rank + module.qk_rope_head_dim)
            output_head_dim = int(module.kv_lora_rank)
        else:
            continue

        signature = (
            device,
            dtype,
            total_heads,
            query_head_dim,
            output_head_dim,
        )
        if signature in warmed_signatures:
            continue

        warmup_b12x_dcp_a2a(
            get_dcp_group(),
            device=device,
            dtype=dtype,
            max_batch_size=worker.scheduler_config.max_num_batched_tokens,
            total_heads=total_heads,
            head_dim=output_head_dim,
            query_head_dim=query_head_dim,
        )
        warmed_signatures.add(signature)

    return len(warmed_signatures)


def kernel_warmup(worker: "Worker"):
    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])
    compile_sizes = [
        size
        for size in (getattr(compilation_config, "compile_sizes", None) or [])
        if isinstance(size, int)
    ]
    mhc_warmup_token_sizes = list(cudagraph_capture_sizes)
    max_num_scheduled_tokens = getattr(
        worker.scheduler_config, "max_num_scheduled_tokens", None
    )
    if max_num_scheduled_tokens is not None:
        mhc_warmup_token_sizes.append(max_num_scheduled_tokens)

    # DSv4 mHC kernels run every decoder layer per token; warm them across
    # token sizes first so the first real request doesn't pay JIT cost. No-op
    # for non-DSv4 models (gated inside); still warms the boundary TileLang
    # kernels used by the b12x mHC forward path.
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=mhc_warmup_token_sizes,
    )

    warmed_dcp_a2a = _warmup_b12x_dcp_a2a(worker)
    if warmed_dcp_a2a:
        logger.info(
            "Warmed up %d B12X DCP collective signature(s).",
            warmed_dcp_a2a,
        )

    # Run next so input-prep kernels JIT against pristine runner state.
    sparse_mla_triton_warmup_if_needed(worker)
    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    warmed_mxfp8 = warmup_b12x_mxfp8_linear(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        output_dtype=getattr(
            getattr(worker, "model_config", None),
            "dtype",
            torch.bfloat16,
        ),
    )
    if warmed_mxfp8:
        logger.info("Warmed up %d B12X MXFP8 linear GEMM signatures.", warmed_mxfp8)

    warmed_tensor_fp8 = warmup_b12x_tensor_fp8_linear(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        output_dtype=getattr(
            getattr(worker, "model_config", None),
            "dtype",
            torch.bfloat16,
        ),
    )
    if warmed_tensor_fp8:
        logger.info(
            "Warmed up %d B12X tensor FP8 linear GEMM signatures.",
            warmed_tensor_fp8,
        )

    warmed_mla_bmm = warmup_b12x_mla_mxfp8_bmm(worker.get_model())
    if warmed_mla_bmm:
        logger.info(
            "Warmed up %d B12X MLA MXFP8 BMM variants.",
            warmed_mla_bmm,
        )

    # Graph replay cannot JIT a missing specialization. Prewarm only the
    # graph-visible token counts covered by the small-M kernel instead of
    # retaining all 32 possible variants in every worker. M=1 also covers
    # eager-only configurations and the ordinary single-request decode path.
    mla_query_warmup_sizes = sorted(
        {
            1,
            *(size for size in cudagraph_capture_sizes if 1 <= size <= 32),
        }
    )
    warmed_mla_query = warmup_fused_mla_query(
        worker.get_model(),
        m_values=mla_query_warmup_sizes,
    )
    if warmed_mla_query:
        logger.info(
            "Warmed up %d fused MLA BF16/MXFP8 query variants.",
            warmed_mla_query,
        )

    warmed_indexer = warmup_b12x_sparse_indexer(worker)
    if warmed_indexer:
        logger.info("Warmed up %d B12X sparse-indexer decode variants.", warmed_indexer)

    moe_token_counts = [
        worker.scheduler_config.max_num_batched_tokens,
        *cudagraph_capture_sizes,
        *compile_sizes,
    ]
    if max_num_scheduled_tokens is not None:
        moe_token_counts.append(max_num_scheduled_tokens)
    warmup_b12x_moe_dynamic(
        worker.get_model(),
        max_tokens=max(moe_token_counts),
        token_counts=moe_token_counts,
    )
    free_before_exl3_warmup = torch.accelerator.get_memory_info()[0]
    warmed_exl3_route_pack = warmup_exl3_mixed_trellis_route_pack(worker.get_model())
    if warmed_exl3_route_pack:
        free_after_exl3_warmup = torch.accelerator.get_memory_info()[0]
        logger.info(
            "Warmed up %d EXL3 mixed-Trellis route-pack variant(s); "
            "free-memory delta %.1f MiB.",
            warmed_exl3_route_pack,
            (free_before_exl3_warmup - free_after_exl3_warmup) / (1 << 20),
        )

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif not has_flashinfer():
        logger.info("Skipping FlashInfer autotune because FlashInfer is unavailable.")
    elif not current_platform.has_device_capability(90):
        logger.info(
            "Skipping FlashInfer autotune because the device capability is below 90."
        )
    elif not _uses_flashinfer_compute_kernels(worker):
        logger.info(
            "Skipping FlashInfer autotune because no FlashInfer compute kernels "
            "are active."
        )
    else:
        flashinfer_autotune(worker.model_runner)

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm()

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    attn_groups = getattr(worker.model_runner, "attn_groups", None)
    if (
        not worker.model_runner.is_pooling_model
        and attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        cutedsl_warmup()


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info(
            "Skipping FlashInfer autotuning for ops %s",
            sorted(skip_ops),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    use_persistent_cache = True

    # When distributed, tune on every rank so the collectives stay synchronized.
    if get_world_group().world_size > 1:
        use_persistent_cache = False

    if not use_persistent_cache:
        with torch.inference_mode(), fi_utils.autotune(**autotune_kwargs):
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(
                tune_mode=True, cache=str(cache_path), **autotune_kwargs
            ):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        if not is_leader and world.local_rank == 0:
            write_flashinfer_autotune_cache(cache_path, tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
