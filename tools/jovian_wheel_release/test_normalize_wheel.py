# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the CUDA foundation wheel metadata contract."""

from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path

from tools.jovian_wheel_release.normalize_wheel import (
    portable_rpath,
    rewrite_requirements,
)


def test_rewrites_foundation_dependencies() -> None:
    """Foundation-owned requirements match the exact published wheel set."""
    metadata = b"""Metadata-Version: 2.4
Name: vllm
Version: 1.0
Requires-Dist: torch==2.13.0
Requires-Dist: torchvision==0.28.0
Requires-Dist: torchaudio==2.11.0
Requires-Dist: flashinfer-python==0.6.17
Requires-Dist: apache-tvm-ffi==0.1.11
Requires-Dist: torchcodec>=0.14
Requires-Dist: PyNvVideoCodec==2.0.4
Requires-Dist: tilelang==0.1.12
Requires-Dist: fastsafetensors>=0.3.3
Requires-Dist: fastsafetensors>=0.3.3; extra == "fastsafetensors"
Requires-Dist: quack-kernels==0.6.4
Requires-Dist: tokenspeed-mla==0.1.8; platform_system == "Linux"
Requires-Dist: humming-kernels[cu13]==0.1.12
Requires-Dist: click>=8

"""
    output = rewrite_requirements(
        metadata,
        torch_version="2.14.0a0+nv",
        torchvision_version="0.29.0a0+nv",
        flashinfer_version="0.6.18",
    )
    message = BytesParser(policy=compat32).parsebytes(output)
    assert message.get_all("Requires-Dist") == [
        "torch==2.14.0a0+nv",
        "torchvision==0.29.0a0+nv",
        "flashinfer-python==0.6.18",
        'fastsafetensors>=0.3.3; extra == "fastsafetensors"',
        "click>=8",
    ]
    assert message["X-Local-Inference-Runtime-Profile"] == "qwen38-sm120"
    assert message["X-Local-Inference-Unsupported-Extra"] == "audio,video"


def test_native_library_paths_are_relative_to_site_packages() -> None:
    """An extension resolves the foundation without absolute container paths."""
    rpath = portable_rpath(Path("vllm/_C.abi3.so"))
    assert "$ORIGIN/../torch/lib" in rpath
    assert "$ORIGIN/../nvidia/cu13/lib" in rpath
    assert "/usr/local" not in rpath
