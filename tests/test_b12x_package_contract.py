# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def test_vllm_uses_b12x_package_names():
    """Reject runtime references to the superseded SparkInfer package name."""
    root = Path(__file__).parents[1]
    markers = ("sparkinfer.", "SPARKINFER_")
    offenders: list[str] = []

    for source_root in (root / "vllm", root / "tests"):
        for path in source_root.rglob("*.py"):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in markers):
                offenders.append(str(path.relative_to(root)))

    assert not offenders, f"legacy SparkInfer package references: {offenders}"
