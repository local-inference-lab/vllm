#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare and validate a compiled vLLM/B12X Fruit runtime image."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path

from packaging.requirements import Requirement


def _validate_b12x_dependencies() -> None:
    for raw_requirement in requires("b12x") or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"missing B12X dependency: {requirement.name}") from exc
        if requirement.specifier and not requirement.specifier.contains(
            installed, prereleases=True
        ):
            raise RuntimeError(
                f"B12X dependency mismatch: {requirement.name} {installed} "
                f"does not satisfy {requirement.specifier}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_runtime_manifest(
    manifest: Path,
    *,
    vllm_root: Path,
    b12x_root: Path,
) -> int:
    roots = (("vllm", vllm_root / "vllm"), ("b12x", b12x_root / "b12x"))
    resolved_manifest = manifest.resolve()
    if any(resolved_manifest.is_relative_to(root) for _, root in roots):
        raise ValueError("runtime manifest must be outside the package roots")
    entries: list[str] = []
    for namespace, root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"runtime package path is a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(f"runtime package path is not regular: {path}")
            relative = path.relative_to(root).as_posix()
            entries.append(f"{_sha256(path)}  {namespace}/{relative}\n")
    if not entries:
        raise RuntimeError("runtime package manifest would be empty")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(entries), encoding="utf-8")
    os.replace(temporary, manifest)
    return len(entries)


def _copy_compiled_extensions(vllm_root: Path) -> int:
    import vllm

    installed_root = Path(vllm.__file__).resolve().parent
    target_root = vllm_root / "vllm"
    extensions = tuple(installed_root.rglob("*.so"))
    if not extensions:
        raise RuntimeError("installed vLLM package has no compiled extensions")
    for source in extensions:
        destination = target_root / source.relative_to(installed_root)
        if destination.is_symlink():
            raise RuntimeError(
                f"vLLM extension destination is a symlink: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return len(extensions)


def _smoke_local_sources(vllm_root: Path, b12x_root: Path) -> None:
    code = """
from pathlib import Path
import b12x
import vllm
import vllm._C_stable_libtorch
assert Path(vllm.__file__).resolve().is_relative_to(Path(VLLM_ROOT) / "vllm")
assert Path(b12x.__file__).resolve().is_relative_to(Path(B12X_ROOT) / "b12x")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(vllm_root), str(b12x_root)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        (
            sys.executable,
            "-P",
            "-c",
            f"VLLM_ROOT={str(vllm_root)!r}; B12X_ROOT={str(b12x_root)!r};{code}",
        ),
        check=True,
        env=environment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--b12x-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vllm_root = args.vllm_root.resolve(strict=True)
    b12x_root = args.b12x_root.resolve(strict=True)
    _validate_b12x_dependencies()
    count = _copy_compiled_extensions(vllm_root)
    _smoke_local_sources(vllm_root, b12x_root)
    files = _write_runtime_manifest(
        args.manifest,
        vllm_root=vllm_root,
        b12x_root=b12x_root,
    )
    print(
        f"prepared Fruit QSRT runtime with {count} compiled vLLM extensions "
        f"and {files} sealed runtime files"
    )


if __name__ == "__main__":
    main()
