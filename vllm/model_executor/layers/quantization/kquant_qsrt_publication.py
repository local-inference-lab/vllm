# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-stdlib authentication for sealed QSRT model packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_MARKER = "QSRT_COMPLETE.json"
_CHECKSUM_MANIFEST = "MANIFEST.sha256"
_PACKAGE_MANIFEST = "qsrt-manifest.json"
_HF_TRANSPORT_FILES = frozenset({".gitattributes"})
_HF_LOCAL_CACHE_PREFIX = (".cache", "huggingface")


@dataclass(frozen=True)
class QSRTPublicationSeal:
    root: Path
    manifest: dict[str, Any]
    descriptor: dict[str, Any]
    checksums: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return value


def _sha256_field(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_DIGITS for character in value)
    ):
        raise ValueError(f"QSRT publication {name} is not a SHA-256 digest")
    return value


def verify_qsrt_publication(root: str | Path) -> QSRTPublicationSeal:
    """Authenticate a sealed local QSRT package before any runtime import."""

    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("QSRT model root must not be a symbolic link")
    try:
        root = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QSRT model root does not exist: {supplied}") from exc
    if not root.is_dir():
        raise ValueError("QSRT model root must be a directory")

    marker = _json_object(root / _PUBLICATION_MARKER, kind="QSRT completion marker")
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

    checksum_path = root / _CHECKSUM_MANIFEST
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checksum_bytes = checksum_path.read_bytes()
    if hashlib.sha256(checksum_bytes).hexdigest() != marker["checksum_manifest_sha256"]:
        raise ValueError("QSRT checksum manifest does not match the completion marker")
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
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != digest:
            raise ValueError(f"QSRT package hash mismatch for {filename}")
        checksums[filename] = digest
    published_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"QSRT package must not contain symbolic links: {path}")
        if not path.is_file():
            continue
        if (
            relative.as_posix() in _HF_TRANSPORT_FILES
            or relative.parts[:2] == _HF_LOCAL_CACHE_PREFIX
        ):
            continue
        published_files.add(relative.as_posix())
    expected_files = set(checksums) | {_CHECKSUM_MANIFEST, _PUBLICATION_MARKER}
    if published_files != expected_files:
        raise ValueError("QSRT checksum inventory does not match the package files")
    required_files = {"config.json", "model.safetensors.index.json", _PACKAGE_MANIFEST}
    if not required_files.issubset(checksums):
        raise ValueError("QSRT checksum manifest omits a required identity file")
    if checksums[_PACKAGE_MANIFEST] != marker["package_manifest_sha256"]:
        raise ValueError(
            "QSRT package manifest digest disagrees with completion marker"
        )
    if checksums["model.safetensors.index.json"] != marker["model_index_sha256"]:
        raise ValueError("QSRT model index digest disagrees with completion marker")

    manifest = _json_object(root / _PACKAGE_MANIFEST, kind="QSRT package manifest")
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

    config = _json_object(root / "config.json", kind="QSRT model config")
    quantization = config.get("quantization_config")
    descriptor = quantization.get("qsrt") if isinstance(quantization, dict) else None
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
    return QSRTPublicationSeal(
        root=root,
        manifest=manifest,
        descriptor=descriptor,
        checksums=checksums,
    )
