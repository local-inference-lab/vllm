#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify immutable vLLM wheel release assets before publication or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_BUNDLE_ASSETS = {
    "install.sh",
    "manifest.json",
    "requirements-github.txt",
    "runtime.lock",
}


def sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject an asset that violates the publication contract."""
    if not condition:
        raise ValueError(message)


def checksum_entries(path: Path) -> dict[str, str]:
    """Read a complete SHA-256 inventory without accepting duplicate names."""
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed checksum entry")
        digest, name = fields
        name = name.removeprefix("*")
        require(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "invalid SHA-256",
        )
        require(name not in entries, f"duplicate checksum entry: {name}")
        entries[name] = digest
    return entries


def verify_release(
    directory: Path,
    source_commit: str,
    beta_tag: str,
    *,
    promotion: bool = False,
    reference_directory: Path | None = None,
) -> None:
    """Verify flat release assets and, when supplied, independent reference bytes."""
    require(
        len(source_commit) == 40
        and all(c in "0123456789abcdef" for c in source_commit),
        "invalid source commit",
    )
    require(beta_tag == f"vllm-jovian-cu134-beta-{source_commit}", "invalid beta tag")
    manifest_path = directory / "manifest.json"
    require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "manifest must be a regular file",
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "local-inference-vllm-wheel-release/v2":
        raise ValueError("release schema mismatch")
    if manifest["source"]["commit"] != source_commit:
        raise ValueError("source commit mismatch")
    if manifest["release_tag"] != beta_tag:
        raise ValueError("beta tag mismatch")
    packages = manifest.get("packages")
    require(
        isinstance(packages, list) and len(packages) == 1,
        "release must contain exactly one vllm package",
    )
    package = packages[0]
    require(package.get("name") == "vllm", "release package must be vllm")
    wheel_name = package.get("file", "")
    require(
        isinstance(wheel_name, str)
        and wheel_name.startswith("vllm-")
        and wheel_name.endswith(".whl")
        and all(c.isascii() and (c.isalnum() or c in "_.+-") for c in wheel_name),
        "invalid wheel filename",
    )
    archive = f"vllm-jovian-cu134-{source_commit}.tar.zst"
    beta_assets = REQUIRED_BUNDLE_ASSETS | {
        wheel_name,
        "SHA256SUMS",
        archive,
        f"{archive}.sha256",
    }
    expected = beta_assets | ({"stable-promotion.json"} if promotion else set())
    require(
        {p.name for p in directory.iterdir()} == expected, "release asset set differs"
    )
    require(
        all(
            (directory / name).is_file() and not (directory / name).is_symlink()
            for name in expected
        ),
        "release assets must be regular files",
    )
    require(
        sha256(directory / wheel_name) == package["sha256"], "wheel digest mismatch"
    )
    entries = checksum_entries(directory / "SHA256SUMS")
    require(
        set(entries) == REQUIRED_BUNDLE_ASSETS | {f"wheels/{wheel_name}"},
        "SHA256SUMS does not describe the complete bundle",
    )
    for name, digest in entries.items():
        require(
            sha256(directory / Path(name).name) == digest,
            f"SHA256SUMS mismatch: {name}",
        )
    require(
        checksum_entries(directory / f"{archive}.sha256")
        == {archive: sha256(directory / archive)},
        "archive checksum mismatch",
    )

    if reference_directory is not None:
        require(
            {p.name for p in reference_directory.iterdir()} == beta_assets,
            "reference asset set differs",
        )
        for name in beta_assets:
            reference = reference_directory / name
            require(
                reference.is_file() and not reference.is_symlink(),
                "reference assets must be regular files",
            )
            require(
                sha256(directory / name) == sha256(reference),
                f"independent reference mismatch: {name}",
            )
    if promotion:
        require(
            reference_directory is not None,
            "stable promotion requires the source beta reference",
        )
        record = json.loads((directory / "stable-promotion.json").read_text())
        required_fields = {
            "schema": "local-inference-vllm-promotion/v1",
            "status": "byte-identical-promotion",
            "source_release": beta_tag,
            "source_commit": source_commit,
            "source_manifest_sha256": sha256(manifest_path),
        }
        for name, value in required_fields.items():
            require(record.get(name) == value, f"promotion {name} mismatch")


def main() -> None:
    """Parse command-line arguments and enforce the release contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    parser.add_argument(
        "--reference-directory",
        type=Path,
        help="Flat assets from a fresh build or source beta",
    )
    args = parser.parse_args()
    verify_release(
        args.directory,
        args.source_commit,
        args.beta_tag,
        promotion=args.promotion,
        reference_directory=args.reference_directory,
    )


if __name__ == "__main__":
    main()
