# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


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
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


# Multi-rank caches hold the union of every rank's entries. FlashInfer MoE
# cache keys include the expert-parallel rank, so a leader-only file misses
# the other ranks' keys: on a warm start the leader hits its cache and skips
# profiling while its peers miss and enter synchronized (all-reduced)
# profiling, which deadlocks. Older leader-only files use the base name and
# are never read for a multi-rank world.
RANK_UNION_CACHE_NAME = "autotune_configs.allranks-v1.json"
RANK_UNION_MARKER_KEY = "_vllm_rank_union"
_METADATA_KEY = "_metadata"
_GENERATION_KEY = "_generation"


def rank_union_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(RANK_UNION_CACHE_NAME)


def gather_objects(group: "GroupCoordinator", obj: Any) -> list[Any]:
    """Every rank's ``obj``, in rank order, on every rank."""
    return [
        group.broadcast_object(obj if group.rank_in_group == rank else None, src=rank)
        for rank in range(group.world_size)
    ]


def merge_rank_autotune_configs(
    configs: list[dict[str, Any]], world_size: int
) -> tuple[bytes, list[str]]:
    """Union per-rank FlashInfer autotune configs into one cache file.

    Returns the file contents and the keys on which ranks disagreed. Tuning is
    synchronized across ranks, so shared keys should agree; on a disagreement
    the lowest rank's entry is kept. Reserved ``_``-prefixed dict sections
    (FlashInfer's namespaced records) are merged per entry the same way; the
    lowest rank's ``_metadata`` is kept and ``_generation`` is recomputed.
    """
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for config in configs:
        for key, value in config.items():
            if key in (_GENERATION_KEY, RANK_UNION_MARKER_KEY):
                continue
            if key == _METADATA_KEY:
                merged.setdefault(key, value)
                continue
            if key.startswith("_") and isinstance(value, dict):
                section = merged.setdefault(key, {})
                for name, entry in value.items():
                    if name in section and section[name] != entry:
                        conflicts.append(f"{key}.{name}")
                        continue
                    section[name] = entry
                continue
            if key in merged and merged[key] != value:
                conflicts.append(key)
                continue
            merged[key] = value
    ordered: dict[str, Any] = {}
    if _METADATA_KEY in merged:
        ordered[_METADATA_KEY] = merged.pop(_METADATA_KEY)
    for key in sorted(merged):
        ordered[key] = merged[key]
    ordered[_GENERATION_KEY] = hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ordered[RANK_UNION_MARKER_KEY] = {"version": 1, "world_size": world_size}
    return json.dumps(ordered, indent=2).encode(), sorted(set(conflicts))


def read_rank_union_cache(contents: bytes, world_size: int) -> bytes | None:
    """Validate a rank-union cache and strip the vLLM marker for FlashInfer.

    Returns None when the file is not a union of ``world_size`` ranks (for
    example a leader-only file), so it must not be loaded by any rank.
    """
    try:
        config = json.loads(contents)
    except ValueError:
        return None
    if not isinstance(config, dict):
        return None
    marker = config.pop(RANK_UNION_MARKER_KEY, None)
    if not isinstance(marker, dict) or marker.get("world_size") != world_size:
        return None
    return json.dumps(config, indent=2).encode()


def save_rank_union_autotune_cache(
    cache_path: Path, tuner: Any, group: "GroupCoordinator"
) -> None:
    """Persist the union of every rank's autotune entries (collective)."""
    with tempfile.TemporaryDirectory() as directory:
        local_path = Path(directory) / "local.json"
        tuner.save_configs(str(local_path))
        local = json.loads(local_path.read_text()) if local_path.exists() else {}
    configs = gather_objects(group, local)
    if group.rank_in_group == 0:
        contents, conflicts = merge_rank_autotune_configs(configs, group.world_size)
        if conflicts:
            logger.warning(
                "FlashInfer autotune ranks disagree on %d cache entries; keeping "
                "the lowest rank's (first: %s).",
                len(conflicts),
                conflicts[0],
            )
        write_flashinfer_autotune_cache(cache_path, contents)
    group.barrier()


def load_autotune_cache_on_all_ranks(
    cache_path: Path, tuner: Any, group: "GroupCoordinator"
) -> bool:
    """Load the leader's cache on every rank so all make the same decisions.

    Collective. Within the synchronized tuning pass, a cache hit skips a
    profile's all-reduce, so the ranks must hold identical entries: either
    every rank loads the same validated cache or every rank starts empty.
    A multi-rank world only accepts a rank-union file for its world size.
    """
    world_size = group.world_size
    contents: bytes | None = None
    if group.rank_in_group == 0 and cache_path.exists():
        contents = cache_path.read_bytes()
        if world_size > 1:
            validated = read_rank_union_cache(contents, world_size)
            if validated is None:
                logger.warning(
                    "Ignoring FlashInfer autotune cache %s: not a %d-rank union.",
                    cache_path,
                    world_size,
                )
            contents = validated
    contents = group.broadcast_object(contents, src=0)
    if contents is None:
        return False

    try:
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "autotune_configs.json"
            local_path.write_bytes(contents)
            loaded = bool(tuner.load_configs(str(local_path)))
    except Exception as exc:
        logger.warning("Failed to load FlashInfer autotune cache: %s", exc)
        loaded = False
    if world_size > 1 and not all(gather_objects(group, loaded)):
        # Some rank rejected the cache (e.g. an environment mismatch): discard
        # it everywhere so no rank skips a profile its peers run.
        tuner.clear_cache()
        if group.rank_in_group == 0:
            logger.warning(
                "FlashInfer autotune cache %s was not accepted by every rank; "
                "retuning on all ranks.",
                cache_path,
            )
        return False
    return loaded
