# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCHER = _REPO_ROOT / "serve-ds4-flash.sh"


def _dry_run(tmp_path: Path, **overrides: str) -> str:
    """Run the DS4 launcher without starting a server.

    Args:
        tmp_path: Isolated home and cache root for the launcher.
        **overrides: Environment values applied to the launcher defaults.

    Returns:
        The launcher's diagnostic stderr output.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        "MODE": "mtp0",
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def test_ds4_launcher_auto_selects_b12x_for_tp2(tmp_path: Path) -> None:
    """Verify that automatic selection uses B12X policy at TP2."""
    output = _dry_run(tmp_path, TP="2")

    assert "allreduce=b12x" in output


def test_ds4_launcher_auto_keeps_b12x_for_tp4(tmp_path: Path) -> None:
    """Verify that automatic selection retains B12X at TP4."""
    output = _dry_run(tmp_path, TP="4")

    assert "allreduce=b12x" in output


def test_ds4_launcher_explicit_allreduce_overrides_auto(tmp_path: Path) -> None:
    """Verify that an explicit all-reduce mode overrides automatic selection."""
    output = _dry_run(tmp_path, TP="2", ALLREDUCE_MODE="b12x")

    assert "allreduce=b12x" in output
