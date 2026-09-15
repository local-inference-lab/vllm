# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Vendored build output must be copyable into a portable source checkout."""

import os
import shutil
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
