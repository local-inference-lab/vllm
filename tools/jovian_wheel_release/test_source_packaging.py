# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Vendored build output must be copyable into a portable source checkout."""

import os
import shutil
import subprocess
from pathlib import Path


def test_deepgemm_build_output_can_populate_source_checkout(tmp_path):
    source = Path(__file__).resolve().parents[2] / "vllm/third_party/deep_gemm"
    destination = tmp_path / "deep_gemm"
    if source.is_symlink() and Path(os.readlink(source)).is_absolute():
        # A clean builder has no contributor-local checkout at the link target.
        destination.symlink_to(tmp_path / "absent-author-checkout/deep_gemm")
    built = tmp_path / "build/deep_gemm"
    built.mkdir(parents=True)
    (built / "__init__.py").write_text("# Vendored build output\n")
    shutil.copytree(built, destination, dirs_exist_ok=True)
    assert (destination / "__init__.py").read_bytes() == (
        built / "__init__.py"
    ).read_bytes()


def test_dependency_recipe_keys_mutable_fetch_and_native_caches(tmp_path):
    """Python-only edits reuse cache; dependency patches select separate storage."""
    root = Path(__file__).resolve().parents[2]
    builder = (root / "tools/jovian_wheel_release/build_bundle.sh").read_text()
    dockerfile = (root / "tools/jovian_wheel_release/Dockerfile").read_text()
    assert "rev-parse HEAD:cmake/external_projects" in builder
    assert '--build-arg "DEPENDENCY_RECIPE=${dependency_recipe}"' in builder
    for cache in (
        "native-cu134-torch214-sm120",
        "fetchcontent-cu134",
        "generated-cu134-torch214-sm120",
    ):
        assert f"id=lil-vllm-{cache}-${{DEPENDENCY_RECIPE}}," in dockerfile

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "Cache contract test")
    git("config", "user.email", "cache@example.invalid")
    git("config", "commit.gpgsign", "false")
    recipes = tmp_path / "cmake/external_projects"
    recipes.mkdir(parents=True)
    (recipes / "flashkda.cmake").write_text("dependency recipe A\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Dependency fixture")
    identity = git("rev-parse", "HEAD:cmake/external_projects")
    (tmp_path / "model.py").write_text("# Independent Python serving source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Python fixture")
    assert git("rev-parse", "HEAD:cmake/external_projects") == identity
    (recipes / "flashkda.patch").write_text("tracked dependency patch\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Patched dependency fixture")
    assert git("rev-parse", "HEAD:cmake/external_projects") != identity


def test_wheel_build_drops_python_staged_by_other_branches(tmp_path):
    """The shared native cache keeps compiled objects, not staged modules."""
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/jovian_wheel_release/build_vllm_wheel.sh"
    ).read_text()
    start = script.index('find "${source_root}/build" -mindepth 1 -maxdepth 1')
    cleanup = script[start : script.index("\n\n", start)]
    assert script.index(cleanup) < script.index("uv build")
    build = tmp_path / "build"
    for staged in (
        "lib.linux-x86_64-cpython-312/vllm/other_branch.py",
        "bdist.linux-x86_64/wheel/vllm/other_branch.py",
        "temp.linux-x86_64-cpython-312/_C.o",
    ):
        (build / staged).parent.mkdir(parents=True, exist_ok=True)
        (build / staged).write_text("staged by an earlier build\n")
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", cleanup],
        env={**os.environ, "source_root": str(tmp_path)},
        check=True,
    )
    assert [path.name for path in build.iterdir()] == ["temp.linux-x86_64-cpython-312"]
    assert (build / "temp.linux-x86_64-cpython-312/_C.o").is_file()
