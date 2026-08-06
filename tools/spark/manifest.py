#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

SYSTEM_PACKAGES = (
    "ccache",
    "cmake",
    "libcudnn9-cuda-13",
    "libcudnn9-dev-cuda-13",
    "libibverbs-dev",
    "libopenmpi-dev",
    "libprotobuf-dev",
    "ninja-build",
    "protobuf-compiler",
    "rdma-core",
)


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    digest = hashlib.sha256()
    for encoded_name in sorted(name for name in files if name):
        source = path / os.fsdecode(encoded_name)
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        if source.is_file():
            digest.update(bytes.fromhex(file_sha256(source)))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def git_record(path: Path) -> dict[str, str]:
    patch = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "path": str(path),
        "head": run("git", "rev-parse", "HEAD", cwd=path),
        "status": run("git", "status", "--short", cwd=path),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "source_sha256": source_sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--uv", required=True)
    args = parser.parse_args()

    artifacts = {}
    if args.artifact_root.exists():
        for path in sorted(args.artifact_root.rglob("*")):
            if path.is_file() and path != args.output:
                name = str(path.relative_to(args.artifact_root))
                artifacts[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }

    record = {
        "schema": 1,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "kernel": run("uname", "-a"),
            "gpu": run(
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ),
            "cuda": run(os.environ.get("NVCC", "nvcc"), "--version"),
            "toolchain": {
                "cargo": run("cargo", "--version"),
                "ccache": run("ccache", "--version"),
                "cmake": run("cmake", "--version"),
                "ninja": run("ninja", "--version"),
                "protoc": run("protoc", "--version"),
                "rustc": run("rustc", "--version"),
            },
            "system_packages": run(
                "dpkg-query",
                "-W",
                "-f=${Package}=${Version}\\n",
                *SYSTEM_PACKAGES,
            ).splitlines(),
        },
        "repositories": {
            "vllm": git_record(args.vllm_root),
            "b12x": git_record(args.b12x_root),
            "flashinfer": git_record(args.flashinfer_root),
            "spark_vllm_docker": git_record(args.recipe_root),
        },
        "packages": run(
            args.uv,
            "pip",
            "freeze",
            "--python",
            str(args.vllm_root / ".venv" / "bin" / "python"),
        ).splitlines(),
        "artifacts": artifacts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
