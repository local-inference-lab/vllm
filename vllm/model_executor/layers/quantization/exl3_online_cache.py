# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Persistent cache for deterministic online EXL3 weight encoding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import filelock
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_CACHE_SCHEMA = 1
_CACHE_TENSORS = frozenset({"trellis", "suh", "svh"})
# A competing loader may need several minutes to encode a large projection.
# After this bound, local encoding is preferable to blocking model startup.
_CACHE_LOCK_TIMEOUT_SECONDS = 600.0
CacheMode = Literal["off", "readonly", "readwrite"]
QuantizeFn = Callable[[], tuple[Mapping[str, torch.Tensor], float | None]]


class Exl3OnlineNonFiniteError(ValueError):
    """An online EXL3 payload contains a non-finite scale or error metric."""


@dataclass(frozen=True)
class Exl3OnlineCacheKey:
    """All inputs that can change one rank-local online EXL3 payload."""

    model_identity: str
    encoder_identity: str
    prefix: str
    bits: int
    seed: int
    tp_world_size: int
    tp_rank: int
    input_size: int
    output_size: int
    codebook: str = "mcg"
    apply_out_scales: bool = True
    schema: int = _CACHE_SCHEMA

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class Exl3OnlineCacheResult:
    tensors: dict[str, torch.Tensor]
    proxy_error: float | None
    hit: bool
    path: Path | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_identity(
    model_name: str,
    *,
    revision: str | None = None,
    hf_config: Any | None = None,
) -> str:
    """Fingerprint a Hub revision or a local checkpoint without reading weights.

    .. note::

        For local checkpoints this fingerprint is **not** a content hash of the
        weights. A single-file checkpoint larger than 16 MiB is identified only
        by its resolved path, size, and ``mtime_ns``; a directory checkpoint is
        identified by hashed config/marker files plus per-shard ``(name, path,
        size, mtime_ns)`` tuples — the shard bytes themselves are never hashed.
        Consequently an in-place weight replacement that preserves file size and
        ``mtime_ns`` (e.g. ``cp`` followed by ``touch -r``) produces the same
        identity and may cause stale encoded EXL3 tensors to be reused from
        ``VLLM_EXL3_ONLINE_CACHE_DIR``. After replacing local weights in place,
        operators should either delete the affected cache entry (or the whole
        ``VLLM_EXL3_ONLINE_CACHE_DIR``) or set
        ``VLLM_EXL3_ONLINE_CACHE_MODE=off`` to force re-encoding.
    """

    resolved_revision = getattr(hf_config, "_commit_hash", None) or revision
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        if not resolved_revision:
            raise ValueError(
                "online EXL3 caching requires a resolved revision for Hub "
                f"checkpoint {model_name!r}; pass --revision or set "
                "VLLM_EXL3_ONLINE_CACHE_MODE=off"
            )
        payload = {
            "kind": "hub",
            "model": model_name,
            "revision": resolved_revision,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    model_path = model_path.resolve()
    if model_path.is_file():
        stat = model_path.stat()
        payload = {
            "kind": "file",
            "path": str(model_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 16 * 1024 * 1024:
            payload["sha256"] = _sha256_file(model_path)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    markers: list[tuple[str, str]] = []
    for name in (
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tier_bitmap.json",
    ):
        path = model_path / name
        if path.is_file():
            markers.append((name, _sha256_file(path)))

    shards: list[tuple[str, str, int, int]] = []
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        for path in sorted(model_path.glob(pattern)):
            resolved = path.resolve()
            stat = resolved.stat()
            shards.append(
                (
                    path.name,
                    str(resolved),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
    payload = {
        "kind": "directory",
        "path": str(model_path),
        "revision": resolved_revision,
        "markers": markers,
        "shards": shards,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_encoder_identity(
    package_root: str | Path,
    *,
    revision: str | None = None,
) -> str:
    """Identify the encoder implementation used to produce cached tensors."""

    root = Path(package_root).expanduser().resolve()
    if revision and revision.strip():
        payload = {"root": str(root), "revision": revision.strip()}
    else:
        files = [
            (str(path.relative_to(root)), _sha256_file(path))
            for path in sorted(root.rglob("*.py"))
            if path.is_file()
        ]
        if not files:
            raise ValueError(f"EXL3 encoder source contains no Python files: {root}")
        payload = {"root": str(root), "files": files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_mode() -> CacheMode:
    raw = os.getenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite").strip().lower()
    aliases = {"read-only": "readonly", "rw": "readwrite", "none": "off"}
    raw = aliases.get(raw, raw)
    if raw not in {"off", "readonly", "readwrite"}:
        raise ValueError(
            "VLLM_EXL3_ONLINE_CACHE_MODE must be off, readonly, or readwrite; "
            f"got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def cache_root() -> Path:
    """Return the trusted online-weight cache directory.

    Cached payloads become model weights after schema validation. The directory
    must therefore be writable only by the serving user and trusted to the same
    degree as the source checkpoint.
    """

    configured = os.getenv("VLLM_EXL3_ONLINE_CACHE_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path(envs.VLLM_CACHE_ROOT) / "exl3_online"


def cache_path(key: Exl3OnlineCacheKey) -> Path:
    digest = key.digest()
    model_dir = key.model_identity[:20]
    rank_dir = f"tp{key.tp_world_size}-rank{key.tp_rank}"
    return (
        cache_root()
        / f"v{_CACHE_SCHEMA}"
        / model_dir
        / rank_dir
        / f"{digest}.safetensors"
    )


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor], key: Exl3OnlineCacheKey
) -> None:
    if set(tensors) != _CACHE_TENSORS:
        raise ValueError(
            f"online EXL3 cache tensor set is {sorted(tensors)}, "
            f"expected {sorted(_CACHE_TENSORS)}"
        )
    expected = {
        "trellis": (
            torch.int16,
            (key.input_size // 16, key.output_size // 16, key.bits * 16),
        ),
        "suh": (torch.float16, (key.input_size,)),
        "svh": (torch.float16, (key.output_size,)),
    }
    for name, (dtype, shape) in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(
                f"online EXL3 cache {name} has {tensor.dtype}/{tuple(tensor.shape)}, "
                f"expected {dtype}/{shape}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"online EXL3 cache {name} must be contiguous")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise Exl3OnlineNonFiniteError(
                f"online EXL3 cache {name} contains non-finite values for "
                f"{key.prefix} (TP rank {key.tp_rank})"
            )


def _validate_proxy_error(proxy_error: float | None, key: Exl3OnlineCacheKey) -> None:
    if proxy_error is not None and not math.isfinite(proxy_error):
        raise Exl3OnlineNonFiniteError(
            "online EXL3 encoder returned a non-finite proxy error for "
            f"{key.prefix} (TP rank {key.tp_rank})"
        )


def _load(path: Path, key: Exl3OnlineCacheKey) -> Exl3OnlineCacheResult:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("cache_key") != key.canonical_json():
            raise ValueError("online EXL3 cache key metadata does not match")
        # ``safe_open`` exposes ``keys()`` but is not itself iterable.
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    _validate_tensors(tensors, key)
    raw_error = metadata.get("proxy_error")
    proxy_error = None if raw_error in (None, "") else float(raw_error)
    _validate_proxy_error(proxy_error, key)
    return Exl3OnlineCacheResult(dict(tensors), proxy_error, True, path)


def _to_device(
    result: Exl3OnlineCacheResult, device: torch.device
) -> Exl3OnlineCacheResult:
    tensors = {
        name: tensor.to(device=device, non_blocking=False).contiguous()
        for name, tensor in result.tensors.items()
    }
    return Exl3OnlineCacheResult(tensors, result.proxy_error, result.hit, result.path)


def _quantize(
    key: Exl3OnlineCacheKey,
    quantize: QuantizeFn,
    *,
    path: Path | None,
) -> Exl3OnlineCacheResult:
    tensors, proxy_error = quantize()
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    _validate_tensors(tensors, key)
    _validate_proxy_error(proxy_error, key)
    return Exl3OnlineCacheResult(tensors, proxy_error, False, path)


def _save(path: Path, key: Exl3OnlineCacheKey, result: Exl3OnlineCacheResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensors = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in result.tensors.items()
    }
    metadata = {
        "cache_key": key.canonical_json(),
        "proxy_error": "" if result.proxy_error is None else repr(result.proxy_error),
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(cpu_tensors, temporary, metadata=metadata)
        with temporary.open("rb") as output:
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_quantize(
    key: Exl3OnlineCacheKey,
    *,
    device: torch.device,
    quantize: QuantizeFn,
) -> Exl3OnlineCacheResult:
    """Load one rank-local payload or encode and atomically publish it."""

    mode = cache_mode()
    if mode == "off":
        return _quantize(key, quantize, path=None)

    path = cache_path(key)
    if path.is_file():
        try:
            return _to_device(_load(path, key), device)
        except Exception as exc:  # noqa: BLE001 - invalid caches are regenerated
            logger.warning("Ignoring invalid online EXL3 cache %s: %s", path, exc)

    if mode == "readonly":
        return _quantize(key, quantize, path=None)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(f"{path}.lock", timeout=_CACHE_LOCK_TIMEOUT_SECONDS)
        with lock:
            if path.is_file():
                try:
                    return _to_device(_load(path, key), device)
                except Exception as exc:  # noqa: BLE001 - invalid caches are replaced
                    logger.warning(
                        "Replacing invalid online EXL3 cache %s: %s", path, exc
                    )
                    path.unlink(missing_ok=True)
            result = _quantize(key, quantize, path=path)
            try:
                _save(path, key, result)
            except Exception as exc:  # noqa: BLE001 - cache publication is optional
                logger.warning(
                    "Unable to publish online EXL3 cache %s; using the encoded "
                    "weights for this process: %s",
                    path,
                    exc,
                )
                return Exl3OnlineCacheResult(
                    result.tensors, result.proxy_error, False, None
                )
            return result
    except filelock.Timeout as exc:
        logger.warning(
            "Online EXL3 cache lock remained held for %.0f seconds at %s; "
            "encoding without cache: %s",
            _CACHE_LOCK_TIMEOUT_SECONDS,
            path,
            exc,
        )
        return _quantize(key, quantize, path=None)
    except OSError as exc:
        logger.warning(
            "Online EXL3 cache is unavailable at %s; encoding without cache: %s",
            path,
            exc,
        )
        return _quantize(key, quantize, path=None)
