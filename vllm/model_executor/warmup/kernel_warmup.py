# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _is_flashinfer_backend(backend) -> bool:
    try:
        return backend.get_name() == "FLASHINFER"
    except NotImplementedError:
        return False


def _uses_only_flashinfer_attention(runner: "GPUModelRunner") -> bool:
    attn_groups = getattr(runner, "attn_groups", None)
    return bool(attn_groups) and all(
        _is_flashinfer_backend(group.backend)
        for groups in attn_groups
        for group in groups
    )


_DEFAULT_FLASHINFER_AUTOTUNE_TOKEN_SIZES = (
    24,
    56,
    75,
    80,
    120,
    248,
    496,
    640,
)

_FLASHINFER_FP8_LINEAR_PROBE_TOKEN_SIZES = (
    4,
    16,
    75,
    128,
    512,
    640,
    8192,
    16384,
)


def _flashinfer_autotune_token_sizes(
    max_num_tokens: int,
    capture_sizes: Iterable[int] | None = None,
    max_bucket_tokens: int | None = None,
) -> tuple[int, ...]:
    max_bucket_tokens = max(max_num_tokens, max_bucket_tokens or max_num_tokens)
    raw_token_sizes = envs.VLLM_FLASHINFER_AUTOTUNE_TOKEN_SIZES
    if not raw_token_sizes:
        token_sizes = set(_DEFAULT_FLASHINFER_AUTOTUNE_TOKEN_SIZES)
    else:
        try:
            parsed_token_sizes = {
                int(token_size.strip())
                for token_size in raw_token_sizes.split(",")
                if token_size.strip()
            }
        except ValueError:
            logger.warning(
                "Invalid VLLM_FLASHINFER_AUTOTUNE_TOKEN_SIZES=%r; "
                "using default token sizes.",
                raw_token_sizes,
            )
            parsed_token_sizes = set(_DEFAULT_FLASHINFER_AUTOTUNE_TOKEN_SIZES)

        if not parsed_token_sizes or any(size <= 0 for size in parsed_token_sizes):
            logger.warning(
                "Invalid VLLM_FLASHINFER_AUTOTUNE_TOKEN_SIZES=%r; "
                "using default token sizes.",
                raw_token_sizes,
            )
            parsed_token_sizes = set(_DEFAULT_FLASHINFER_AUTOTUNE_TOKEN_SIZES)
        token_sizes = parsed_token_sizes

    if capture_sizes is not None:
        token_sizes.update(int(size) for size in capture_sizes if int(size) > 0)

    token_sizes = {size for size in token_sizes if size <= max_bucket_tokens}
    token_sizes.add(max_num_tokens)
    token_sizes.add(max_bucket_tokens)

    return tuple(sorted(token_sizes))


def _flashinfer_autotune_capture_sizes(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    compilation_config = runner.vllm_config.compilation_config
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if not capture_sizes:
        return ()
    return tuple(int(size) for size in capture_sizes if int(size) > 0)


def _flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    token_sizes = _flashinfer_autotune_token_sizes_for_runner(runner)
    factors = aot_compile_hash_factors(runner.vllm_config)
    factors.extend(
        [
            f"flashinfer_autotune_token_sizes={token_sizes}",
        ]
    )
    return hashlib.sha256(str(factors).encode()).hexdigest()


def _flashinfer_autotune_max_bucket_tokens(runner: "GPUModelRunner") -> int:
    max_num_tokens = runner.scheduler_config.max_num_batched_tokens
    max_bucket_tokens = max_num_tokens

    parallel_config = runner.vllm_config.parallel_config
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    if dcp_size > 1:
        # MLA DCP long-context prefill can dispatch local GEMM token dimensions
        # above max_num_batched_tokens, e.g. Kimi 128k DCP8 emits M=16384 with
        # max_num_batched_tokens=8192. Cover that common case by default.
        max_bucket_tokens = max(max_bucket_tokens, max_num_tokens * 2)

    return max_bucket_tokens


def _flashinfer_autotune_token_sizes_for_runner(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    return _flashinfer_autotune_token_sizes(
        runner.scheduler_config.max_num_batched_tokens,
        _flashinfer_autotune_capture_sizes(runner),
        _flashinfer_autotune_max_bucket_tokens(runner),
    )


def _flashinfer_autotune_dummy_token_sizes(
    runner: "GPUModelRunner",
    token_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    max_num_tokens = runner.scheduler_config.max_num_batched_tokens
    return tuple(size for size in token_sizes if size <= max_num_tokens)


def _flashinfer_autotune_fp8_probe_token_sizes(
    token_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    available_token_sizes = set(token_sizes)
    probe_token_sizes = [
        token_size
        for token_size in _FLASHINFER_FP8_LINEAR_PROBE_TOKEN_SIZES
        if token_size in available_token_sizes
    ]
    if not probe_token_sizes:
        probe_token_sizes = [min(token_sizes)]
    return tuple(probe_token_sizes)


def _activate_flashinfer_autotune_runtime_buckets(
    runner: "GPUModelRunner",
    autotune_kwargs: dict,
) -> None:
    """Keep FlashInfer runtime cache keys aligned with autotune cache keys.

    FlashInfer includes the effective tuning-bucket mapper in several runner
    hashes. If autotune runs with custom buckets but CUDA graph capture/runtime
    falls back to FlashInfer's default buckets, tuned entries are invisible and
    those shapes fall back to heuristic tactics.
    """
    import vllm.utils.flashinfer as fi_utils
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    if not getattr(AutoTuner, "_vllm_global_override_installed", False):
        original_buckets_property = AutoTuner._override_tuning_buckets
        original_round_up_property = AutoTuner._override_round_up

        def _override_tuning_buckets(self):
            buckets = original_buckets_property.fget(self)
            if buckets is not None:
                return buckets
            return getattr(self, "_vllm_global_override_tuning_buckets", None)

        def _override_round_up(self):
            round_up = original_round_up_property.fget(self)
            if round_up:
                return round_up
            return getattr(self, "_vllm_global_override_round_up", False)

        AutoTuner._override_tuning_buckets = property(_override_tuning_buckets)
        AutoTuner._override_round_up = property(_override_round_up)
        AutoTuner._vllm_global_override_installed = True

    tuner._vllm_global_override_tuning_buckets = tuple(
        autotune_kwargs["tuning_buckets"]
    )
    tuner._vllm_global_override_round_up = bool(autotune_kwargs.get("round_up"))

    previous_context = getattr(
        runner, "_flashinfer_autotune_runtime_context", None
    )
    if previous_context is not None:
        previous_context.__exit__(None, None, None)

    context = fi_utils.autotune(tune_mode=False, **autotune_kwargs)
    context.__enter__()
    runner._flashinfer_autotune_runtime_context = context
    runner._flashinfer_autotune_runtime_kwargs = autotune_kwargs
    logger.info(
        "Activated FlashInfer runtime autotune buckets: %s.",
        autotune_kwargs,
    )


def _resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / _flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def _flashinfer_autotune_fp8_linear_shapes(
    runner: "GPUModelRunner",
    token_sizes: tuple[int, ...],
) -> None:
    """Autotune dense FP8 GEMM shapes without mutating proposer state.

    Kimi's MTP/drafter path can hit FP8 linear shapes that are not reached by
    the main model dummy run with the same execution metadata as real
    speculative decoding. Warming the raw GEMM shapes covers the FlashInfer
    cache key directly while avoiding incomplete calls into proposer logits.
    """
    if not token_sizes:
        return

    try:
        from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm_out
    except Exception:
        return

    models = [
        ("target", getattr(runner, "model", None)),
    ]
    for attr in ("speculator", "drafter"):
        speculator = getattr(runner, attr, None)
        model = getattr(speculator, "model", None)
        if model is not None:
            models.append((speculator.__class__.__name__, model))

    shapes: dict[tuple[int, int], str] = {}
    for owner_name, model in models:
        if model is None or not hasattr(model, "named_modules"):
            continue
        for module_name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            fp8_linear = getattr(quant_method, "fp8_linear", None)
            if fp8_linear is None:
                continue
            kernel_name = fp8_linear.__class__.__name__
            if kernel_name != "FlashInferFP8ScaledMMLinearKernel":
                continue

            weight = getattr(module, "weight", None)
            input_scale = getattr(module, "input_scale", None)
            weight_scale = getattr(module, "weight_scale", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 2
                or weight.dtype != torch.float8_e4m3fn
                or not isinstance(input_scale, torch.Tensor)
                or not isinstance(weight_scale, torch.Tensor)
                or input_scale.numel() != 1
                or weight_scale.numel() != 1
            ):
                continue

            k_dim, n_dim = (int(weight.shape[0]), int(weight.shape[1]))
            shapes.setdefault((k_dim, n_dim), f"{owner_name}.{module_name}")

    if not shapes:
        return

    probe_token_sizes = _flashinfer_autotune_fp8_probe_token_sizes(token_sizes)
    device = runner.device
    out_dtype = runner.model_config.dtype
    input_scale = torch.ones((), dtype=torch.float32, device=device)
    weight_scale = torch.ones((), dtype=torch.float32, device=device)

    logger.info(
        "Running FlashInfer autotune for %d dense FP8 linear shape(s) "
        "and validating token sizes %s.",
        len(shapes),
        probe_token_sizes,
    )
    for (k_dim, n_dim), source in sorted(shapes.items()):
        b = torch.zeros((k_dim, n_dim), dtype=torch.float8_e4m3fn, device=device)
        for num_tokens in probe_token_sizes:
            logger.debug(
                "Autotuning FlashInfer dense FP8 linear shape M=%d K=%d "
                "N=%d from %s.",
                num_tokens,
                k_dim,
                n_dim,
                source,
            )
            a = None
            out = None
            try:
                a = torch.zeros(
                    (num_tokens, k_dim),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                out = torch.empty(
                    (num_tokens, n_dim), dtype=out_dtype, device=device
                )
                flashinfer_scaled_fp8_mm_out(
                    a,
                    b,
                    input_scale,
                    weight_scale,
                    out,
                    out_dtype=out_dtype,
                )
                torch.cuda.synchronize()
            except Exception:
                logger.exception(
                    "FlashInfer dense FP8 linear autotune validation failed "
                    "for M=%d K=%d N=%d from %s.",
                    num_tokens,
                    k_dim,
                    n_dim,
                    source,
                )
                raise
            finally:
                if a is not None:
                    del a
                if out is not None:
                    del out
        del b
    torch.cuda.empty_cache()


def _dummy_run_for_flashinfer_autotune(
    runner: "GPUModelRunner",
    *,
    skip_attn: bool,
    **kwargs,
) -> None:
    """Call GPUModelRunner._dummy_run across both v1 runner variants.

    The newer GPU runner accepts skip_attn directly.  The Kimi/MLA runner
    instead defaults to no attention metadata for profile dummy runs and uses
    force_attention=True when attention must be covered.
    """
    dummy_run_params = inspect.signature(runner._dummy_run).parameters
    if "skip_attn" in dummy_run_params:
        runner._dummy_run(skip_attn=skip_attn, **kwargs)
        return

    if not skip_attn and "force_attention" in dummy_run_params:
        kwargs["force_attention"] = True
    runner._dummy_run(**kwargs)


def _flashinfer_autotune_dummy_run(
    runner: "GPUModelRunner",
    num_tokens: int,
) -> None:
    _dummy_run_for_flashinfer_autotune(
        runner,
        num_tokens=num_tokens,
        skip_attn=True,
        skip_eplb=True,
        is_profile=True,
    )


def _flashinfer_autotune_uniform_decode(
    runner: "GPUModelRunner",
    token_sizes: tuple[int, ...],
) -> None:
    decode_query_len = getattr(runner, "decode_query_len", None)
    max_num_reqs = getattr(runner, "max_num_reqs", None)
    if not decode_query_len or not max_num_reqs:
        return

    max_decode_tokens = max_num_reqs * decode_query_len
    uniform_token_sizes = [
        num_tokens
        for num_tokens in token_sizes
        if num_tokens <= max_decode_tokens and num_tokens % decode_query_len == 0
    ]
    if not uniform_token_sizes:
        return

    logger.info(
        "Running FlashInfer autotune for uniform decode token sizes: %s.",
        uniform_token_sizes,
    )
    for num_tokens in uniform_token_sizes:
        _dummy_run_for_flashinfer_autotune(
            runner,
            num_tokens=num_tokens,
            skip_attn=True,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=True,
        )
    torch.cuda.empty_cache()


def _flashinfer_autotune_attention_capture_runs(
    runner: "GPUModelRunner",
    token_sizes: tuple[int, ...],
) -> None:
    if not _uses_only_flashinfer_attention(runner):
        logger.info(
            "Skipping FlashInfer attention autotune because the active "
            "attention backend is not FlashInfer."
        )
        return

    decode_query_len = getattr(runner, "decode_query_len", None)
    max_num_reqs = getattr(runner, "max_num_reqs", None)
    max_decode_tokens = (
        max_num_reqs * decode_query_len
        if decode_query_len and max_num_reqs
        else None
    )

    max_num_tokens = runner.scheduler_config.max_num_batched_tokens
    attention_token_sizes = [
        num_tokens
        for num_tokens in token_sizes
        if num_tokens >= 4
        and num_tokens < max_num_tokens
        and (max_decode_tokens is None or num_tokens > max_decode_tokens)
    ]
    if not attention_token_sizes:
        return

    logger.info(
        "Running FlashInfer autotune with attention for capture token sizes: %s.",
        attention_token_sizes,
    )
    for num_tokens in attention_token_sizes:
        try:
            _dummy_run_for_flashinfer_autotune(
                runner,
                num_tokens=num_tokens,
                skip_attn=False,
                skip_eplb=True,
                is_profile=True,
            )
        except Exception:
            logger.exception(
                "FlashInfer attention autotune failed for num_tokens=%d.",
                num_tokens,
            )
            raise
    torch.cuda.empty_cache()


def kernel_warmup(worker: "Worker"):
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

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    if (
        not worker.model_runner.is_pooling_model
        and _uses_only_flashinfer_attention(worker.model_runner)
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


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False


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

    token_sizes = _flashinfer_autotune_token_sizes_for_runner(runner)
    dummy_token_sizes = _flashinfer_autotune_dummy_token_sizes(runner, token_sizes)
    logger.info(
        "Running FlashInfer autotune with token buckets %s; dummy runs use %s.",
        token_sizes,
        dummy_token_sizes,
    )

    autotune_kwargs = dict(tuning_buckets=token_sizes, round_up=True)

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune(**autotune_kwargs):
            for num_tokens in dummy_token_sizes:
                _flashinfer_autotune_dummy_run(runner, num_tokens)
            _flashinfer_autotune_uniform_decode(runner, dummy_token_sizes)
            _flashinfer_autotune_attention_capture_runs(runner, dummy_token_sizes)
            _flashinfer_autotune_fp8_linear_shapes(runner, token_sizes)
        _activate_flashinfer_autotune_runtime_buckets(runner, autotune_kwargs)
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = _resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # FlashInfer's cuBLAS FP8 runner includes exact A/B shapes in its cache
    # extras, so non-bucket runtime token counts need explicit dummy runs.
    dummy_run_kwargs = dict(skip_eplb=True, is_profile=True)

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(
                tune_mode=True,
                cache=str(cache_path),
                **autotune_kwargs,
            ):
                for num_tokens in dummy_token_sizes:
                    _dummy_run_for_flashinfer_autotune(
                        runner,
                        num_tokens=num_tokens,
                        skip_attn=True,
                        **dummy_run_kwargs,
                    )
                _flashinfer_autotune_uniform_decode(runner, dummy_token_sizes)
                _flashinfer_autotune_attention_capture_runs(
                    runner, dummy_token_sizes
                )
                _flashinfer_autotune_fp8_linear_shapes(runner, token_sizes)
        else:
            for num_tokens in dummy_token_sizes:
                _dummy_run_for_flashinfer_autotune(
                    runner,
                    num_tokens=num_tokens,
                    skip_attn=True,
                    **dummy_run_kwargs,
                )
            _flashinfer_autotune_uniform_decode(runner, dummy_token_sizes)
            _flashinfer_autotune_attention_capture_runs(runner, dummy_token_sizes)
            _flashinfer_autotune_fp8_linear_shapes(runner, token_sizes)

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
            with open(cache_path, "wb") as f:
                f.write(tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )

    _activate_flashinfer_autotune_runtime_buckets(runner, autotune_kwargs)
