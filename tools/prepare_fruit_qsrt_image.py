#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare and validate a compiled vLLM/B12X Fruit runtime image."""

from __future__ import annotations

import argparse
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
        if requirement.specifier and installed not in requirement.specifier:
            raise RuntimeError(
                f"B12X dependency mismatch: {requirement.name} {installed} "
                f"does not satisfy {requirement.specifier}"
            )


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vllm_root = args.vllm_root.resolve(strict=True)
    b12x_root = args.b12x_root.resolve(strict=True)
    _validate_b12x_dependencies()
    count = _copy_compiled_extensions(vllm_root)
    _smoke_local_sources(vllm_root, b12x_root)
    print(f"prepared Fruit QSRT runtime with {count} compiled vLLM extensions")


if __name__ == "__main__":
    main()
