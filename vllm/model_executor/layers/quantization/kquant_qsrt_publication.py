# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-stdlib authentication for sealed QSRT model packages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHA256_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_MARKER = "QSRT_COMPLETE.json"
_CHECKSUM_MANIFEST = "MANIFEST.sha256"
_PACKAGE_MANIFEST = "qsrt-manifest.json"
_HF_TRANSPORT_FILES = frozenset({".gitattributes"})
_HF_LOCAL_CACHE_PREFIX = (".cache", "huggingface")
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIRECTORY_FLAGS = _READ_FLAGS | os.O_DIRECTORY


@dataclass
class QSRTPublicationSeal:
    """Authenticated package identity plus stable atom-file descriptors."""

    root: Path
    manifest: dict[str, Any]
    descriptor: dict[str, Any]
    checksums: dict[str, str]
    _atom_descriptors: dict[str, int] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def authenticated_atom_path(self, filename: str) -> Path:
        """Return a procfs path that reopens the already-authenticated inode."""

        if self._closed:
            raise RuntimeError("QSRT publication seal is closed")
        try:
            descriptor = self._atom_descriptors[filename]
        except KeyError as exc:
            raise ValueError(
                f"QSRT publication has no authenticated atom {filename!r}"
            ) from exc
        return Path(f"/proc/self/fd/{descriptor}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in self._atom_descriptors.values():
            os.close(descriptor)
        self._atom_descriptors.clear()

    def __enter__(self) -> QSRTPublicationSeal:
        if self._closed:
            raise RuntimeError("QSRT publication seal is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _sha256_field(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_DIGITS for character in value)
    ):
        raise ValueError(f"QSRT publication {name} is not a lowercase SHA-256 digest")
    return value


def expected_complete_sha256_from_env() -> str:
    """Read Fruit's independent completion-marker trust root."""

    return _sha256_field(
        os.environ.get("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256"),
        name="FRUIT_QSRT_EXPECTED_COMPLETE_SHA256",
    )


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 8 << 20):
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 8 << 20):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_beneath(root_descriptor: int, relative: Path) -> int:
    directory = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=directory)
    except OSError as exc:
        raise FileNotFoundError(relative.as_posix()) from exc
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise FileNotFoundError(relative.as_posix())
    return descriptor


def _json_bytes(data: bytes, *, kind: str, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return value


def _published_files(root_descriptor: int) -> set[str]:
    published: set[str] = set()

    def visit(directory: int, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = (*prefix, entry.name)
                display = "/".join(relative)
                if entry.is_symlink():
                    raise ValueError(
                        f"QSRT package must not contain symbolic links: {display}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=directory)
                    try:
                        visit(child, relative)
                    finally:
                        os.close(child)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if (
                    display in _HF_TRANSPORT_FILES
                    or relative[:2] == _HF_LOCAL_CACHE_PREFIX
                ):
                    continue
                published.add(display)

    visit(root_descriptor, ())
    return published


def verify_qsrt_publication(
    root: str | Path,
    *,
    expected_complete_sha256: str | None = None,
) -> QSRTPublicationSeal:
    """Authenticate a sealed local QSRT package before any runtime import."""

    expected_complete_sha256 = _sha256_field(
        expected_complete_sha256, name="expected completion marker"
    )
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("QSRT model root must not be a symbolic link")
    try:
        resolved_root = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QSRT model root does not exist: {supplied}") from exc
    try:
        root_descriptor = os.open(supplied, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(f"cannot securely open QSRT model root: {supplied}") from exc
    if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
        os.close(root_descriptor)
        raise ValueError("QSRT model root must be a directory")

    opened: dict[str, int] = {}
    try:
        marker_descriptor = _open_beneath(root_descriptor, Path(_PUBLICATION_MARKER))
        try:
            marker_bytes = _read_descriptor(marker_descriptor)
        finally:
            os.close(marker_descriptor)
        if hashlib.sha256(marker_bytes).hexdigest() != expected_complete_sha256:
            raise ValueError("QSRT completion marker does not match the trusted digest")
        marker = _json_bytes(
            marker_bytes,
            kind="QSRT completion marker",
            path=resolved_root / _PUBLICATION_MARKER,
        )
        expected_marker_fields = {
            "schema",
            "package_manifest_sha256",
            "checksum_manifest_sha256",
            "model_index_sha256",
            "source",
            "base_manifest_sha256",
            "producer_fingerprint",
            "encoder_fingerprint",
        }
        if (
            set(marker) != expected_marker_fields
            or marker.get("schema") != "kquant_qsrt_complete_v2"
        ):
            raise ValueError("QSRT completion marker identity is invalid")
        for name in (
            "package_manifest_sha256",
            "checksum_manifest_sha256",
            "model_index_sha256",
            "base_manifest_sha256",
            "producer_fingerprint",
            "encoder_fingerprint",
        ):
            _sha256_field(marker.get(name), name=name)
        marker_source = marker.get("source")
        if (
            not isinstance(marker_source, dict)
            or set(marker_source) != {"kind", "sha256"}
            or not isinstance(marker_source.get("kind"), str)
            or not marker_source["kind"]
        ):
            raise ValueError("QSRT completion marker source identity is invalid")
        _sha256_field(marker_source.get("sha256"), name="source.sha256")

        checksum_descriptor = _open_beneath(root_descriptor, Path(_CHECKSUM_MANIFEST))
        try:
            checksum_bytes = _read_descriptor(checksum_descriptor)
        finally:
            os.close(checksum_descriptor)
        if (
            hashlib.sha256(checksum_bytes).hexdigest()
            != marker["checksum_manifest_sha256"]
        ):
            raise ValueError(
                "QSRT checksum manifest does not match the completion marker"
            )
        try:
            checksum_text = checksum_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("QSRT checksum manifest is not UTF-8") from exc
        checksums: dict[str, str] = {}
        for line in checksum_text.splitlines():
            digest, separator, filename = line.partition("  ")
            relative = Path(filename)
            if (
                separator != "  "
                or len(digest) != 64
                or any(character not in _SHA256_DIGITS for character in digest)
                or not filename
                or relative.is_absolute()
                or relative.as_posix() != filename
                or ".." in relative.parts
                or filename in checksums
            ):
                raise ValueError(f"invalid QSRT checksum entry: {line!r}")
            descriptor = _open_beneath(root_descriptor, relative)
            opened[filename] = descriptor
            if _sha256_descriptor(descriptor) != digest:
                raise ValueError(f"QSRT package hash mismatch for {filename}")
            checksums[filename] = digest

        published_files = _published_files(root_descriptor)
        expected_files = set(checksums) | {_CHECKSUM_MANIFEST, _PUBLICATION_MARKER}
        if published_files != expected_files:
            raise ValueError("QSRT checksum inventory does not match the package files")
        required_files = {
            "config.json",
            "model.safetensors.index.json",
            _PACKAGE_MANIFEST,
        }
        if not required_files.issubset(checksums):
            raise ValueError("QSRT checksum manifest omits a required identity file")
        if checksums[_PACKAGE_MANIFEST] != marker["package_manifest_sha256"]:
            raise ValueError(
                "QSRT package manifest digest disagrees with completion marker"
            )
        if checksums["model.safetensors.index.json"] != marker["model_index_sha256"]:
            raise ValueError("QSRT model index digest disagrees with completion marker")

        manifest = _json_bytes(
            _read_descriptor(opened[_PACKAGE_MANIFEST]),
            kind="QSRT package manifest",
            path=resolved_root / _PACKAGE_MANIFEST,
        )
        producer = manifest.get("producer")
        source = manifest.get("source")
        base_model = manifest.get("base_model")
        if (
            manifest.get("schema") != "kquant_qsrt_model_manifest_v1"
            or manifest.get("version") != 1
            or manifest.get("complete") is not True
            or not isinstance(producer, dict)
            or not isinstance(producer.get("encoder"), dict)
            or not isinstance(source, dict)
            or not isinstance(base_model, dict)
        ):
            raise ValueError("QSRT package manifest identity is invalid or incomplete")
        expected_source = {
            "kind": source.get("source_kind"),
            "sha256": source.get("source_sha256"),
        }
        if marker_source != expected_source:
            raise ValueError(
                "QSRT package source identity disagrees with completion marker"
            )
        if (
            producer.get("fingerprint") != marker["producer_fingerprint"]
            or producer["encoder"].get("fingerprint") != marker["encoder_fingerprint"]
            or base_model.get("manifest_sha256") != marker["base_manifest_sha256"]
        ):
            raise ValueError("QSRT producer identity disagrees with completion marker")

        config = _json_bytes(
            _read_descriptor(opened["config.json"]),
            kind="QSRT model config",
            path=resolved_root / "config.json",
        )
        quantization = config.get("quantization_config")
        descriptor = (
            quantization.get("qsrt") if isinstance(quantization, dict) else None
        )
        if not isinstance(descriptor, dict):
            raise ValueError("QSRT model config omits its format descriptor")
        expected_descriptor = {
            "schema": manifest.get("storage_schema"),
            "storage_format": "qsrt_atoms_v1",
            "encoding": manifest.get("encoding"),
            "codebook": manifest.get("codebook"),
            "profile_id": manifest.get("profile_id"),
            "artifact_manifest": _PACKAGE_MANIFEST,
            "producer_fingerprint": marker["producer_fingerprint"],
            "encoder_fingerprint": marker["encoder_fingerprint"],
            "source_kind": marker_source["kind"],
            "source_sha256": marker_source["sha256"],
        }
        if any(value is None for value in expected_descriptor.values()):
            raise ValueError("QSRT package manifest omits a required descriptor field")
        if any(
            descriptor.get(name) != value for name, value in expected_descriptor.items()
        ):
            raise ValueError("QSRT model descriptor disagrees with the sealed manifest")

        atom_names: set[str] = set()
        layers = manifest.get("layers")
        if isinstance(layers, dict):
            for layer in layers.values():
                if isinstance(layer, dict) and isinstance(layer.get("qsrt_atoms"), str):
                    atom_names.add(layer["qsrt_atoms"])
        missing_atoms = atom_names - checksums.keys()
        if missing_atoms:
            raise ValueError(
                f"QSRT checksum manifest omits atom files: {sorted(missing_atoms)}"
            )
        atom_descriptors = {name: opened.pop(name) for name in atom_names}
        for descriptor_fd in opened.values():
            os.close(descriptor_fd)
        opened.clear()
        return QSRTPublicationSeal(
            root=resolved_root,
            manifest=manifest,
            descriptor=descriptor,
            checksums=checksums,
            _atom_descriptors=atom_descriptors,
        )
    except BaseException:
        for descriptor_fd in opened.values():
            os.close(descriptor_fd)
        raise
    finally:
        os.close(root_descriptor)
