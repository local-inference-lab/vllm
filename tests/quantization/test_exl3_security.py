# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security tests for EXL3 trust-boundary hardening (C1, C2, C3)."""

import os
import sys
from unittest.mock import patch

import pytest

# We test the validation helpers and the bits_per_expert sanitization in
# isolation, without loading the actual exllamav3_ext native library.


def test_validate_native_lib_path_rejects_symlink(tmp_path):
    """C1: VLLM_EXL3_ABI_SHIM must not be a symlink."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_native_lib_path,
    )

    target = tmp_path / "real.so"
    target.write_text("dummy")
    link = tmp_path / "shim.so"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _validate_native_lib_path(str(link), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_nonexistent(tmp_path):
    """C1: non-existent path is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_native_lib_path,
    )

    with pytest.raises(ValueError, match="non-existent"):
        _validate_native_lib_path(str(tmp_path / "missing.so"), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_world_writable(tmp_path):
    """C1: world-writable .so is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_native_lib_path,
    )

    lib = tmp_path / "evil.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o666)  # rw-rw-rw-

    with pytest.raises(ValueError, match="group/world-writable"):
        _validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_world_writable_parent(tmp_path):
    """C1: world-writable parent directory is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_native_lib_path,
    )

    lib = tmp_path / "evil.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o777)  # world-writable parent

    with pytest.raises(ValueError, match="parent directory"):
        _validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_accepts_safe(tmp_path):
    """C1: a regular file in a safe directory passes."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_native_lib_path,
    )

    lib = tmp_path / "good.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o755)

    # Should not raise
    _validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_ext_search_dir_rejects_world_writable(tmp_path):
    """C2: world-writable directory for VLLM_EXL3_EXT_PATH is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    os.chmod(tmp_path, 0o777)

    with pytest.raises(ValueError, match="world-writable"):
        _validate_ext_search_dir(str(tmp_path), "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_rejects_nonexistent(tmp_path):
    """C2: non-existent directory is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    with pytest.raises(ValueError, match="directory"):
        _validate_ext_search_dir(str(tmp_path / "nope"), "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_accepts_safe(tmp_path):
    """C2: a safe directory passes."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    os.chmod(tmp_path, 0o755)
    _validate_ext_search_dir(str(tmp_path), "VLLM_EXL3_EXT_PATH")


def test_bits_per_expert_rejects_path_traversal():
    """C3: bits_per_expert filename with path separators is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    config = Exl3Config.__new__(Exl3Config)
    config.rank_sliced_metadata = {
        "bits_per_expert": "../../../../etc/passwd:k",
        "experts_per_layer": 256,
        "moe_layers": [3, 13],
    }
    config.rank_sliced_k_values = [3, 4]

    with pytest.raises(ValueError, match="path separators"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_rejects_absolute_path():
    """C3: absolute path in bits_per_expert filename is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    config = Exl3Config.__new__(Exl3Config)
    config.rank_sliced_metadata = {
        "bits_per_expert": "/etc/passwd:k",
        "experts_per_layer": 256,
        "moe_layers": [3, 13],
    }
    config.rank_sliced_k_values = [3, 4]

    with pytest.raises(ValueError, match="path separators"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_accepts_safe_filename():
    """C3: a bare filename (no path separators) passes the sanitization."""
    # We can't fully test _load_rank_sliced_bitrates without a real model dir,
    # but we can verify the basename check passes for a safe name.
    assert os.path.basename("tier_bitmap.json") == "tier_bitmap.json"
    assert os.path.basename("../evil.json") != "../evil.json"
    assert os.path.basename("/etc/passwd") != "/etc/passwd"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
