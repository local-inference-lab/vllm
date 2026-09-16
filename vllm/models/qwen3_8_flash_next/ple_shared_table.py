# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-shared PLE table files for independent Qwen3.8-Flash-Next processes.

Several TP1 replicas on one host can serve one CPU-resident copy of the n-gram
table.  The table lives in a tmpfs directory as contiguous files in the b12x
storage layout.  The first process to take the directory lock writes them from
the checkpoint through the ordinary weight-loading path; later processes
``mmap(MAP_SHARED)`` the same files and ``cudaHostRegister`` the mapping, which
pins each page once no matter how many processes map it.  b12x only requires
mapped-host tensors whose device alias equals the host pointer, which
registered memory satisfies under unified addressing, so b12x is unchanged.

Layout under ``<dir>``::

    <key>/manifest.json   geometry and provenance; written before READY
    <key>/weight.bin      plan.weight_shape rows, plan.weight_dtype, exact size
    <key>/weight_scale.bin  only for mapped-host scales (nvfp4_group16)
    <key>/READY           empty marker created last
    <key>.lock            flock target, separate so <key>/ can be replaced

``key`` derives from the checkpoint identity and the planned table geometry,
so TP ranks and checkpoint revisions never share a directory.  A manifest that
disagrees with the live plan in any field fails startup; nothing here falls
back to a private allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import mmap
import os
import shutil
import socket
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

LAYOUT_VERSION = 1
MANIFEST_FORMAT = "vllm-ple-shared-table/1"
MANIFEST_FILE = "manifest.json"
READY_FILE = "READY"
WEIGHT_FILE = "weight.bin"
WEIGHT_SCALE_FILE = "weight_scale.bin"
DEFAULT_DIRECTORY = "/dev/shm/vllm-ple"
ROLES = ("auto", "populate", "attach")
_KEY_HEX_DIGITS = 16
_LOCK_POLL_S = 0.5
_MS_SYNC = 4  # <sys/mman.h>; the mmap module does not export it.
_MAPPED_HOST_QUANT_MODES = frozenset({"nvfp4_group16"})


class SharedTableError(RuntimeError):
    """Base class for shared-table failures."""


class SharedTableMissing(SharedTableError):
    """No complete table exists for the key and the role forbids populating."""


class SharedTableMismatch(SharedTableError):
    """A manifest field disagrees with the live plan."""

    def __init__(self, field: str, expected: Any, found: Any, path: Path) -> None:
        self.field = field
        self.expected = expected
        self.found = found
        super().__init__(
            f"shared PLE table at {path} does not match the live plan: "
            f"{field} expected {expected!r}, manifest has {found!r}"
        )


class SharedTableLockTimeout(SharedTableError, TimeoutError):
    """The directory lock was not acquired within the timeout."""


@dataclass(frozen=True, kw_only=True)
class SharedTableConfig:
    """Operator-facing settings resolved by the PLE layer."""

    directory: str = DEFAULT_DIRECTORY
    role: str = "auto"
    lock_timeout_s: float = 3600.0
    model_path: str
    revision: str | None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"ple_shared_table_role must be one of {ROLES}, got {self.role!r}"
            )
        if self.lock_timeout_s < 0:
            raise ValueError("ple_shared_table_lock_timeout_s must be non-negative")


@dataclass(frozen=True)
class TableFileSpec:
    name: str
    attribute: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown tensor dtype {name!r}")
    return dtype


@dataclass(frozen=True, kw_only=True)
class SharedTableGeometry:
    """Every input that decides the bytes of one TP rank's table files."""

    model_path: str
    revision: str | None
    quant_mode: str
    tp_size: int
    tp_rank: int
    dense_layer_ordinal: int
    padded_vocab_size: int
    shard_start: int
    shard_end: int
    embedding_dim: int
    weight_shape: tuple[int, ...]
    weight_dtype: str
    weight_scale_shape: tuple[int, ...] | None
    weight_scale_dtype: str | None
    layout_version: int = LAYOUT_VERSION

    @classmethod
    def from_layout(
        cls,
        layout: Any,
        *,
        model_path: str,
        revision: str | None,
        embedding_dim: int,
        dense_layer_ordinal: int,
    ) -> SharedTableGeometry:
        """Read the geometry from a b12x ``Plan`` or storage layout.

        ``dense_layer_ordinal`` identifies the PLE layer: each one owns its
        own checkpoint shards and therefore its own table.
        """
        caps = layout.caps
        return cls(
            model_path=str(model_path),
            revision=None if revision is None else str(revision),
            quant_mode=str(caps.quant_mode),
            tp_size=int(caps.tp_size),
            tp_rank=int(caps.tp_rank),
            dense_layer_ordinal=int(dense_layer_ordinal),
            padded_vocab_size=int(layout.padded_vocab_size),
            shard_start=int(layout.shard_start),
            shard_end=int(layout.shard_end),
            embedding_dim=int(embedding_dim),
            weight_shape=tuple(int(extent) for extent in layout.weight_shape),
            weight_dtype=_dtype_name(layout.weight_dtype),
            weight_scale_shape=(
                None
                if layout.weight_scale_shape is None
                else tuple(int(extent) for extent in layout.weight_scale_shape)
            ),
            weight_scale_dtype=_dtype_name(layout.weight_scale_dtype),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedTableGeometry:
        fields = dict(data)
        for name in ("weight_shape", "weight_scale_shape"):
            if fields.get(name) is not None:
                fields[name] = tuple(int(extent) for extent in fields[name])
        return cls(**fields)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "model_path": self.model_path,
            "revision": self.revision,
            "quant_mode": self.quant_mode,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "dense_layer_ordinal": self.dense_layer_ordinal,
            "padded_vocab_size": self.padded_vocab_size,
            "shard_start": self.shard_start,
            "shard_end": self.shard_end,
            "embedding_dim": self.embedding_dim,
            "weight_shape": list(self.weight_shape),
            "weight_dtype": self.weight_dtype,
            "weight_scale_shape": (
                None
                if self.weight_scale_shape is None
                else list(self.weight_scale_shape)
            ),
            "weight_scale_dtype": self.weight_scale_dtype,
            "layout_version": self.layout_version,
        }
        return data

    @property
    def key(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:_KEY_HEX_DIGITS]

    @property
    def files(self) -> tuple[TableFileSpec, ...]:
        """Files backing the mapped-host tensors, in b12x storage order."""
        specs = [
            TableFileSpec(
                WEIGHT_FILE,
                "weight",
                self.weight_shape,
                _dtype_from_name(self.weight_dtype),
            )
        ]
        if (
            self.weight_scale_shape is not None
            and self.quant_mode in _MAPPED_HOST_QUANT_MODES
        ):
            assert self.weight_scale_dtype is not None
            specs.append(
                TableFileSpec(
                    WEIGHT_SCALE_FILE,
                    "weight_scale",
                    self.weight_scale_shape,
                    _dtype_from_name(self.weight_scale_dtype),
                )
            )
        return tuple(specs)

    @property
    def nbytes(self) -> int:
        return sum(spec.nbytes for spec in self.files)


def table_directory(directory: str | os.PathLike[str], key: str) -> Path:
    return Path(directory) / key


def lock_path(directory: str | os.PathLike[str], key: str) -> Path:
    return Path(directory) / f"{key}.lock"


class SharedTableLock:
    """Exclusive ``flock`` on ``<dir>/<key>.lock`` with a polling timeout."""

    def __init__(self, path: Path, timeout_s: float) -> None:
        self.path = path
        self.timeout_s = float(timeout_s)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        deadline = time.monotonic() + self.timeout_s
        logged = False
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SharedTableLockTimeout(
                        f"timed out after {self.timeout_s:.0f}s waiting for the "
                        f"shared PLE table lock {self.path}"
                    )
                if not logged:
                    logger.info(
                        "Waiting for the shared PLE table lock %s (another process "
                        "is populating; timeout %.0fs)",
                        self.path,
                        self.timeout_s,
                    )
                    logged = True
                time.sleep(min(_LOCK_POLL_S, remaining))
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> SharedTableLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _try_lock(path: Path) -> tuple[int | None, bool]:
    """Return ``(fd, busy)``: a locked descriptor, or ``busy`` when another
    process holds the lock.  A missing lock file yields ``(None, False)``."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except FileNotFoundError:
        return None, False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, True
    return fd, False


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.msync.restype = ctypes.c_int
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return libc


def _cudart() -> Any:
    from cuda.bindings import runtime as cudart

    return cudart


def _check_cuda(error: Any, operation: str) -> None:
    cudart = _cudart()
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    # Same construction as b12x MappedHostAllocation; copied so this module
    # does not import a private b12x helper.
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host storage requires torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage, 0, shape, _contiguous_strides(shape)
    )


def _host_register_read_only_supported(device: torch.device) -> bool:
    cudart = _cudart()
    error, value = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported, device.index
    )
    return error == cudart.cudaError_t.cudaSuccess and bool(value)


class SharedTableMapping:
    """Own one ``MAP_SHARED`` file mapping, its CUDA registration and aliases.

    ``host_view`` is a CPU tensor over the mapping; ``device_view`` is the CUDA
    tensor b12x binds.  With a CPU ``device`` (tests) nothing is registered
    and ``device_view`` is ``host_view``.
    """

    def __init__(
        self,
        path: Path,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        writable: bool,
    ) -> None:
        self.path = Path(path)
        self.shape = tuple(int(extent) for extent in shape)
        self.dtype = dtype
        self.device = device
        self.nbytes = math.prod(self.shape) * dtype.itemsize
        self.writable = bool(writable)
        self.read_only_registration = False
        self._fd: int | None = None
        self._pointer = 0
        self._registered = False
        self._libc = _libc()
        if self.nbytes <= 0:
            raise ValueError(f"shared table file {path} must be non-empty")
        if device.type == "cuda" and device.index is None:
            raise ValueError(
                f"shared table mapping requires an indexed CUDA device, got {device}"
            )
        # Attachers still open read-write when the file allows it, so the
        # registration fallback below can map with PROT_WRITE.
        open_flags = os.O_RDWR if writable or os.access(path, os.W_OK) else os.O_RDONLY
        self._fd = os.open(path, open_flags | os.O_CLOEXEC)
        try:
            size = os.fstat(self._fd).st_size
            if size != self.nbytes:
                raise SharedTableError(
                    f"shared PLE table file {path} has {size} bytes, expected "
                    f"{self.nbytes}"
                )
            if self.writable:
                self._map(mmap.PROT_READ | mmap.PROT_WRITE)
                self._register(read_only=False)
            else:
                self._map(mmap.PROT_READ)
                try:
                    self._register(read_only=True)
                except RuntimeError as exc:
                    if open_flags != os.O_RDWR or device.type != "cuda":
                        raise
                    logger.warning(
                        "Read-only registration of the shared PLE table %s was "
                        "rejected (%s); mapping it writable instead",
                        path,
                        exc,
                    )
                    self._unmap()
                    self._map(mmap.PROT_READ | mmap.PROT_WRITE)
                    self._register(read_only=False)
            self.host_view = _tensor_from_pointer(
                self._pointer,
                shape=self.shape,
                dtype=dtype,
                device=torch.device("cpu"),
                nbytes=self.nbytes,
            )
            if device.type == "cuda":
                cudart = _cudart()
                with torch.cuda.device(device):
                    error, device_pointer = cudart.cudaHostGetDevicePointer(
                        self._pointer, 0
                    )
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=self.shape,
                    dtype=dtype,
                    device=device,
                    nbytes=self.nbytes,
                )
            else:
                self.device_view = self.host_view
        except BaseException:
            self.close()
            raise

    def _map(self, prot: int) -> None:
        assert self._fd is not None
        pointer = self._libc.mmap(None, self.nbytes, prot, mmap.MAP_SHARED, self._fd, 0)
        if pointer is None or pointer == ctypes.c_void_p(-1).value:
            code = ctypes.get_errno()
            raise OSError(code, f"mmap of {self.path} failed: {os.strerror(code)}")
        self._pointer = int(pointer)

    def _unmap(self) -> None:
        if self._pointer:
            self._libc.munmap(self._pointer, self.nbytes)
            self._pointer = 0

    def _register(self, *, read_only: bool) -> None:
        if self.device.type != "cuda":
            return
        cudart = _cudart()
        flags = cudart.cudaHostRegisterMapped
        if read_only:
            if not _host_register_read_only_supported(self.device):
                raise RuntimeError(
                    "cudaHostRegisterReadOnly is not supported on this device"
                )
            flags |= cudart.cudaHostRegisterReadOnly
        with torch.cuda.device(self.device):
            (error,) = cudart.cudaHostRegister(self._pointer, self.nbytes, flags)
        _check_cuda(error, "cudaHostRegister")
        self._registered = True
        self.read_only_registration = read_only

    def sync(self) -> None:
        """Flush the mapping and file so another process sees every byte."""
        if not self.writable:
            return
        if self._pointer and self._libc.msync(self._pointer, self.nbytes, _MS_SYNC):
            code = ctypes.get_errno()
            raise OSError(code, f"msync of {self.path} failed: {os.strerror(code)}")
        if self._fd is not None:
            os.fsync(self._fd)

    def close(self) -> None:
        if self._registered:
            cudart = _cudart()
            torch.cuda.synchronize(self.device)
            cudart.cudaHostUnregister(self._pointer)
            self._registered = False
        self._unmap()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        if sys is None or sys.is_finalizing():
            return
        with suppress(Exception):
            self.close()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_manifest(table_dir: Path) -> dict[str, Any] | None:
    """Return the manifest of a complete table, or None when READY is absent."""
    if not (table_dir / READY_FILE).exists():
        return None
    manifest_path = table_dir / MANIFEST_FILE
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SharedTableError(
            f"shared PLE table {table_dir} is marked READY but its manifest "
            f"cannot be read: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SharedTableError(
            f"shared PLE table manifest {manifest_path} is not an object"
        )
    return manifest


def check_manifest(
    manifest: dict[str, Any], geometry: SharedTableGeometry, table_dir: Path
) -> None:
    """Fail with the offending field when a manifest disagrees with the plan."""
    if manifest.get("format") != MANIFEST_FORMAT:
        raise SharedTableMismatch(
            "format", MANIFEST_FORMAT, manifest.get("format"), table_dir
        )
    recorded = manifest.get("geometry")
    if not isinstance(recorded, dict):
        raise SharedTableMismatch("geometry", geometry.as_dict(), recorded, table_dir)
    expected = geometry.as_dict()
    for field, value in expected.items():
        found = recorded.get(field)
        if found != value:
            raise SharedTableMismatch(field, value, found, table_dir)
    extra = sorted(set(recorded) - set(expected))
    if extra:
        raise SharedTableMismatch(extra[0], None, recorded[extra[0]], table_dir)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SharedTableMismatch("files", {}, files, table_dir)
    expected_files = {spec.name: spec.nbytes for spec in geometry.files}
    if files != expected_files:
        raise SharedTableMismatch("files", expected_files, files, table_dir)
    for name, nbytes in expected_files.items():
        path = table_dir / name
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            raise SharedTableMismatch(
                f"files.{name}", nbytes, "missing", table_dir
            ) from None
        if size != nbytes:
            raise SharedTableMismatch(f"files.{name}", nbytes, size, table_dir)


def _provenance() -> dict[str, Any]:
    import importlib.metadata

    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    ple_layer = Path(__file__).with_name("ple_layer.py")
    try:
        ple_layer_sha256 = hashlib.sha256(ple_layer.read_bytes()).hexdigest()
    except OSError:
        ple_layer_sha256 = None
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "creator": {"hostname": socket.gethostname(), "pid": os.getpid()},
        "vllm_version": version("vllm"),
        "b12x_version": version("b12x"),
        "ple_layer_sha256": ple_layer_sha256,
    }


def _remove_table_dir(table_dir: Path) -> None:
    if table_dir.exists():
        shutil.rmtree(table_dir)


class SharedTable:
    """One process's view of a shared table: mappings plus, when populating,
    the held directory lock until :meth:`publish`.

    Exposes ``nbytes`` and ``close`` so it can stand in the
    ``_mapped_allocations`` tuple of a b12x ``TableStorage``.
    """

    def __init__(
        self,
        *,
        geometry: SharedTableGeometry,
        directory: Path,
        mappings: dict[str, SharedTableMapping],
        lock: SharedTableLock | None,
    ) -> None:
        self.geometry = geometry
        self.key = geometry.key
        self.directory = directory
        self.table_dir = table_directory(directory, self.key)
        self._mappings = mappings
        self._lock = lock
        self._published = lock is None
        self._closed = False
        self._started = time.monotonic()
        self.nbytes = sum(mapping.nbytes for mapping in mappings.values())

    @property
    def populating(self) -> bool:
        return self._lock is not None and not self._published

    @property
    def attached(self) -> bool:
        """True when the bytes came from another process and are complete."""
        return self._lock is None

    @property
    def read_only(self) -> bool:
        return all(
            mapping.read_only_registration for mapping in self._mappings.values()
        )

    def device_view(self, attribute: str) -> torch.Tensor | None:
        mapping = self._mappings.get(attribute)
        return None if mapping is None else mapping.device_view

    def host_view(self, attribute: str) -> torch.Tensor | None:
        mapping = self._mappings.get(attribute)
        return None if mapping is None else mapping.host_view

    def publish(self) -> None:
        """Durably mark the table complete and release the lock.

        Call exactly once, after the checkpoint loader has filled every row of
        every host view and their coverage has been validated.
        """
        if self._published:
            return
        assert self._lock is not None
        try:
            for mapping in self._mappings.values():
                mapping.sync()
            manifest = {
                "format": MANIFEST_FORMAT,
                "key": self.key,
                "geometry": self.geometry.as_dict(),
                "files": {spec.name: spec.nbytes for spec in self.geometry.files},
                **_provenance(),
            }
            _write_json(self.table_dir / MANIFEST_FILE, manifest)
            ready = self.table_dir / READY_FILE
            fd = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_directory(self.table_dir)
        except BaseException:
            _remove_table_dir(self.table_dir)
            self._lock.release()
            self._published = True
            raise
        self._published = True
        self._lock.release()
        logger.info(
            "Populated shared PLE table %s (%.2f GiB) in %s after %.1fs",
            self.key,
            self.nbytes / (1 << 30),
            self.directory,
            time.monotonic() - self._started,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for mapping in reversed(list(self._mappings.values())):
                mapping.close()
        finally:
            if self._lock is not None and not self._published:
                # Never leave a half-written table behind for followers.
                _remove_table_dir(self.table_dir)
                self._lock.release()


def _open_mappings(
    geometry: SharedTableGeometry,
    table_dir: Path,
    *,
    device: torch.device,
    writable: bool,
) -> dict[str, SharedTableMapping]:
    mappings: dict[str, SharedTableMapping] = {}
    try:
        for spec in geometry.files:
            mappings[spec.attribute] = SharedTableMapping(
                table_dir / spec.name,
                shape=spec.shape,
                dtype=spec.dtype,
                device=device,
                writable=writable,
            )
    except BaseException:
        for mapping in mappings.values():
            mapping.close()
        raise
    return mappings


def _create_table_files(geometry: SharedTableGeometry, table_dir: Path) -> None:
    _remove_table_dir(table_dir)
    table_dir.mkdir(parents=True)
    for spec in geometry.files:
        fd = os.open(table_dir / spec.name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            # Reserve every page now so a full tmpfs fails here with ENOSPC
            # instead of raising SIGBUS in the checkpoint copy later.
            os.posix_fallocate(fd, 0, spec.nbytes)
        finally:
            os.close(fd)


def open_shared_table(
    geometry: SharedTableGeometry,
    config: SharedTableConfig,
    *,
    device: torch.device,
) -> SharedTable:
    """Attach to a complete table or become its populator, per ``config.role``.

    A populator returns holding the directory lock; the caller must fill the
    host views, then :meth:`SharedTable.publish`.  Any failure before that
    removes the directory, so a waiting follower repopulates instead of
    attaching to partial bytes.
    """
    directory = Path(config.directory)
    key = geometry.key
    table_dir = table_directory(directory, key)
    lock = SharedTableLock(lock_path(directory, key), config.lock_timeout_s)
    lock.acquire()
    try:
        manifest = None if config.role == "populate" else read_manifest(table_dir)
        if manifest is not None:
            check_manifest(manifest, geometry, table_dir)
            mappings = _open_mappings(
                geometry, table_dir, device=device, writable=False
            )
            lock.release()
            table = SharedTable(
                geometry=geometry, directory=directory, mappings=mappings, lock=None
            )
            logger.info(
                "Attached shared PLE table %s (%.2f GiB, %s) from %s",
                key,
                table.nbytes / (1 << 30),
                "read-only" if table.read_only else "writable mapping",
                directory,
            )
            return table
        if config.role == "attach":
            raise SharedTableMissing(
                f"no complete shared PLE table {key} under {directory} and "
                "ple_shared_table_role=attach forbids populating one; start a "
                "populate/auto replica first or check the directory"
            )
        logger.info(
            "Populating shared PLE table %s (%.2f GiB) in %s",
            key,
            geometry.nbytes / (1 << 30),
            directory,
        )
        _create_table_files(geometry, table_dir)
        try:
            mappings = _open_mappings(geometry, table_dir, device=device, writable=True)
        except BaseException:
            _remove_table_dir(table_dir)
            raise
        return SharedTable(
            geometry=geometry, directory=directory, mappings=mappings, lock=lock
        )
    except BaseException:
        lock.release()
        raise


# --- maintenance CLI -------------------------------------------------------


@dataclass(frozen=True)
class TableStatus:
    key: str
    state: str  # "ready", "incomplete" or "populating"
    nbytes: int
    manifest: dict[str, Any] | None


def _directory_nbytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.iterdir() if entry.is_file())


def list_tables(directory: str | os.PathLike[str]) -> list[TableStatus]:
    root = Path(directory)
    if not root.is_dir():
        return []
    statuses = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        fd, busy = _try_lock(lock_path(root, entry.name))
        state = "populating" if busy else "incomplete"
        if fd is not None:
            os.close(fd)
        manifest = None
        if state != "populating":
            try:
                manifest = read_manifest(entry)
            except SharedTableError:
                manifest = None
            if manifest is not None:
                state = "ready"
        statuses.append(
            TableStatus(entry.name, state, _directory_nbytes(entry), manifest)
        )
    return statuses


def prune_tables(
    directory: str | os.PathLike[str],
    *,
    keep: Sequence[str],
    dry_run: bool = False,
) -> list[str]:
    """Remove every table directory whose key is not in ``keep``.

    Tables whose lock is held (being populated) are left alone.  Removing a
    table that running processes have mapped is safe: tmpfs keeps the pages
    until the last mapping closes, and no new process can attach to it.
    """
    root = Path(directory)
    kept = set(keep)
    removed = []
    for status in list_tables(root):
        if status.key in kept or status.state == "populating":
            continue
        fd, busy = _try_lock(lock_path(root, status.key))
        if busy:
            continue
        try:
            if not dry_run:
                _remove_table_dir(root / status.key)
                lock_path(root, status.key).unlink(missing_ok=True)
            removed.append(status.key)
        finally:
            if fd is not None:
                os.close(fd)
    return removed


def _format_status(status: TableStatus) -> str:
    manifest = status.manifest or {}
    geometry = manifest.get("geometry", {})
    creator = manifest.get("creator", {})
    return " ".join(
        [
            status.key,
            f"{status.state:<11}",
            f"{status.nbytes / (1 << 30):7.2f} GiB",
            f"model={geometry.get('model_path', '?')}",
            f"revision={geometry.get('revision', '?')}",
            f"tp={geometry.get('tp_rank', '?')}/{geometry.get('tp_size', '?')}",
            f"layer={geometry.get('dense_layer_ordinal', '?')}",
            f"created={manifest.get('created_at', '?')}",
            f"by={creator.get('hostname', '?')}:{creator.get('pid', '?')}",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.models.qwen3_8_flash_next.ple_shared_table",
        description="Inspect or prune shared PLE table directories.",
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get("VLLM_PLE_SHARED_TABLE_DIR", DEFAULT_DIRECTORY),
        help="table root directory (default: $VLLM_PLE_SHARED_TABLE_DIR or "
        f"{DEFAULT_DIRECTORY})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="list tables and their state")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")
    prune_parser = commands.add_parser(
        "prune", help="remove every table except the keys given with --keep"
    )
    prune_parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="KEY",
        help="key to keep; repeat for several",
    )
    prune_parser.add_argument(
        "--dry-run", action="store_true", help="print what would be removed"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        statuses = list_tables(args.dir)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "key": status.key,
                            "state": status.state,
                            "nbytes": status.nbytes,
                            "manifest": status.manifest,
                        }
                        for status in statuses
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            for status in statuses:
                print(_format_status(status))
        return 0
    removed = prune_tables(args.dir, keep=args.keep, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    for key in removed:
        print(f"{verb} {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
