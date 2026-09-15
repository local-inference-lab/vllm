# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source-addressed wheel tags must not change vLLM package version discovery."""

from __future__ import annotations

import ast
import os
import runpy
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from setuptools_scm import get_version
from setuptools_scm.git import DEFAULT_DESCRIBE

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "setup.py"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.fixture
def source_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Callable[[], str], Callable[[], str]]:
    """Execute setup's version function without loading Torch or native builders."""
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "Version test")
    git(tmp_path, "config", "user.email", "version@example.invalid")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "tag.gpgsign", "false")
    git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "Source version fixture")
    (tmp_path / "vllm").mkdir()
    monkeypatch.chdir(tmp_path)
    for name in os.environ:
        if name.startswith("SETUPTOOLS_SCM_") or name in (
            "VLLM_VERSION_OVERRIDE",
            "VLLM_RS_BUILD_VERSION",
        ):
            monkeypatch.delenv(name)
    node = next(
        node
        for node in ast.parse(SETUP.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_vllm_version"
    )
    namespace = {
        "os": os,
        "get_version": get_version,
        "DEFAULT_DESCRIBE": DEFAULT_DESCRIBE,
        "_no_device": lambda: True,
        "envs": SimpleNamespace(VLLM_TARGET_DEVICE="cpu"),
        "ROOT_DIR": tmp_path,
        "VLLM_RS_BUILD_VERSION": "VLLM_RS_BUILD_VERSION",
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(SETUP), "exec"), namespace
    )
    rust_node = next(
        node
        for node in ast.parse((ROOT / "tools/build_rust.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_build_environment"
    )
    exec(
        compile(
            ast.Module(body=[rust_node], type_ignores=[]), "tools/build_rust.py", "exec"
        ),
        namespace,
    )
    return namespace["get_vllm_version"], namespace["prepare_build_environment"]


@pytest.mark.parametrize("source_tag", [None, "v0.27.2", "v0.27.2rc0"])
@pytest.mark.parametrize("descendant", [False, True])
def test_wheel_release_tags_preserve_source_version(
    tmp_path: Path,
    source_version: tuple[Callable[[], str], Callable[[], str]],
    source_tag: str | None,
    descendant: bool,
) -> None:
    if source_tag:
        git(tmp_path, "tag", source_tag)
    if descendant:
        git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "Source descendant")
    expected = get_version(root=str(tmp_path))
    commit = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "tag", f"vllm-jovian-cu134-beta-{commit}")
    git(
        tmp_path,
        "tag",
        "-a",
        f"vllm-jovian-cu134-stable-{commit}",
        "-m",
        "Wheel assets",
    )

    python_version, rust_version = source_version
    assert python_version() == expected
    assert rust_version() == expected
    assert os.environ["VLLM_RS_BUILD_VERSION"] == expected
    assert runpy.run_path(str(tmp_path / "vllm/_version.py"))["__version__"] == expected


def test_explicit_version_override_is_preserved(
    tmp_path: Path,
    source_version: tuple[Callable[[], str], Callable[[], str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git(tmp_path, "tag", f"vllm-jovian-cu134-beta-{git(tmp_path, 'rev-parse', 'HEAD')}")
    monkeypatch.setenv("VLLM_VERSION_OVERRIDE", "0.27.2+custom")
    python_version, rust_version = source_version
    assert python_version() == "0.27.2+custom"
    assert rust_version() == "0.27.2+custom"


def test_explicit_rust_version_override_is_preserved(
    source_version: tuple[Callable[[], str], Callable[[], str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_RS_BUILD_VERSION", "0.27.2+rust")
    assert source_version[1]() == "0.27.2+rust"
