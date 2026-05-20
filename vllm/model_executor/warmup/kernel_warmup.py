# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
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


def _flashinfer_autotune_token_sizes(
    max_num_tokens: int,
    capture_sizes: Iterable[int] | None = None,
) -> tuple[int, ...]:
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

    token_sizes = {size for size in token_sizes if size <= max_num_tokens}
    token_sizes.add(max_num_tokens)

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
    token_sizes = _flashinfer_autotune_token_sizes(
        runner.scheduler_config.max_num_batched_tokens,
        _flashinfer_autotune_capture_sizes(runner),
    )
    factors = aot_compile_hash_factors(runner.vllm_config)
    factors.extend(
        [
            f"flashinfer_autotune_token_sizes={token_sizes}",
        ]
    )
    return hashlib.sha256(str(factors).encode()).hexdigest()


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


def _flashinfer_autotune_speculator_logits(
    runner: "GPUModelRunner",
    token_sizes: tuple[int, ...],
) -> None:
    speculator = getattr(runner, "speculator", None)
    if speculator is None:
        return

    model = getattr(speculator, "model", None)
    hidden_states = getattr(speculator, "hidden_states", None)
    if (
        model is None
        or hidden_states is None
        or not hasattr(model, "compute_logits")
    ):
        return

    max_tokens = min(
        hidden_states.shape[0],
        getattr(runner, "max_num_reqs", hidden_states.shape[0]),
    )
    logger.info(
        "Running FlashInfer autotune for Eagle speculator logits up to %d "
        "sample tokens.",
        max_tokens,
    )
    for num_tokens in token_sizes:
        if num_tokens > max_tokens:
            continue
        logits = model.compute_logits(hidden_states[:num_tokens])
        del logits
    torch.cuda.empty_cache()


def _flashinfer_autotune_dummy_run(
    runner: "GPUModelRunner",
    num_tokens: int,
) -> None:
    runner._dummy_run(
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
        runner._dummy_run(
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
        if num_tokens < max_num_tokens
        and (max_decode_tokens is None or num_tokens > max_decode_tokens)
    ]
    if not attention_token_sizes:
        return

    logger.info(
        "Running FlashInfer autotune with attention for capture token sizes: %s.",
        attention_token_sizes,
    )
    for num_tokens in attention_token_sizes:
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_attn=False,
            skip_eplb=True,
            is_profile=True,
        )
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
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
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

    token_sizes = _flashinfer_autotune_token_sizes(
        runner.scheduler_config.max_num_batched_tokens,
        _flashinfer_autotune_capture_sizes(runner),
    )
    logger.info(
        "Running FlashInfer autotune for exact token sizes: %s.",
        token_sizes,
    )

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            for num_tokens in token_sizes:
                _flashinfer_autotune_dummy_run(runner, num_tokens)
            _flashinfer_autotune_uniform_decode(runner, token_sizes)
            _flashinfer_autotune_attention_capture_runs(runner, token_sizes)
            _flashinfer_autotune_speculator_logits(runner, token_sizes)
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
    dummy_run_kwargs = dict(skip_attn=True, skip_eplb=True, is_profile=True)

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(
                tune_mode=True,
                cache=str(cache_path),
            ):
                for num_tokens in token_sizes:
                    runner._dummy_run(num_tokens=num_tokens, **dummy_run_kwargs)
                _flashinfer_autotune_uniform_decode(runner, token_sizes)
                _flashinfer_autotune_attention_capture_runs(runner, token_sizes)
                _flashinfer_autotune_speculator_logits(runner, token_sizes)
        else:
            for num_tokens in token_sizes:
                runner._dummy_run(num_tokens=num_tokens, **dummy_run_kwargs)
            _flashinfer_autotune_uniform_decode(runner, token_sizes)
            _flashinfer_autotune_attention_capture_runs(runner, token_sizes)
            _flashinfer_autotune_speculator_logits(runner, token_sizes)

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
