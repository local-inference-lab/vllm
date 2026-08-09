# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from tools import prepare_fruit_qsrt_image as prepare


@pytest.mark.parametrize("ambient_bytecode_setting", [None, ""])
def test_smoke_import_does_not_change_sealed_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_bytecode_setting: str | None,
) -> None:
    if ambient_bytecode_setting is None:
        monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    else:
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", ambient_bytecode_setting)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    vllm_root = tmp_path / "vllm-root"
    b12x_root = tmp_path / "b12x-root"
    (vllm_root / "vllm").mkdir(parents=True)
    (b12x_root / "b12x").mkdir(parents=True)
    (vllm_root / "vllm/__init__.py").write_text("SOURCE = 'vllm'\n")
    (vllm_root / "vllm/_C_stable_libtorch.py").write_text("LOADED = True\n")
    (b12x_root / "b12x/__init__.py").write_text("SOURCE = 'b12x'\n")
    manifest = tmp_path / "MANIFEST.sha256"

    prepare._smoke_local_sources(vllm_root, b12x_root)
    prepare._write_runtime_manifest(
        manifest,
        vllm_root=vllm_root,
        b12x_root=b12x_root,
    )
    first_inventory = manifest.read_text(encoding="utf-8")
    prepare._smoke_local_sources(vllm_root, b12x_root)
    prepare._write_runtime_manifest(
        manifest,
        vllm_root=vllm_root,
        b12x_root=b12x_root,
    )

    assert not tuple(vllm_root.rglob("*.pyc"))
    assert not tuple(b12x_root.rglob("*.pyc"))
    assert manifest.read_text(encoding="utf-8") == first_inventory
    assert {line.split("  ", 1)[1] for line in first_inventory.splitlines()} == {
        "b12x/__init__.py",
        "vllm/_C_stable_libtorch.py",
        "vllm/__init__.py",
    }


@pytest.mark.parametrize("installed", ["2.11.0.dev20260809", "2.11.0rc1"])
def test_dependency_validation_accepts_satisfying_installed_prereleases(
    monkeypatch: pytest.MonkeyPatch,
    installed: str,
) -> None:
    monkeypatch.setattr(prepare, "requires", lambda _: ["torch>=2.10.0"])
    monkeypatch.setattr(prepare, "version", lambda _: installed)

    prepare._validate_b12x_dependencies()


def test_dependency_validation_rejects_prerelease_below_required_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare, "requires", lambda _: ["torch>=2.11.0"])
    monkeypatch.setattr(prepare, "version", lambda _: "2.11.0rc1")

    with pytest.raises(RuntimeError, match="B12X dependency mismatch"):
        prepare._validate_b12x_dependencies()
