# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from flashinfer.autotuner import AutoTuner

    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def read_flashinfer_autotune_cache(cache_path: Path) -> bytes | None:
    """Treat interrupted JSON writes as cache misses, not model-load failures."""
    try:
        contents = cache_path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        json.loads(contents)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(
            "Ignoring invalid FlashInfer autotune JSON at %s; "
            "autotuning will regenerate it.",
            cache_path,
        )
        return None
    return contents


def save_flashinfer_autotune_cache(cache_path: Path, tuner: "AutoTuner") -> None:
    """Publish only complete serialized tuner state to the shared cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_path.parent) as directory:
        temporary_path = Path(directory) / cache_path.name
        tuner.save_configs(str(temporary_path))
        contents = temporary_path.read_bytes()
        json.loads(contents)
        write_flashinfer_autotune_cache(cache_path, contents)


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    config_hash = runner.vllm_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
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

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.platforms import current_platform
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
        directory_fd = os.open(cache_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
