# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security test for NCCL .so path validation (H1)."""

import os
from unittest.mock import patch

import pytest

from vllm.utils.path_validation import validate_native_lib_path


def test_validate_nccl_path_rejects_symlink(tmp_path):
    """H1: VLLM_NCCL_SO_PATH must not be a symlink."""
    target = tmp_path / "real_libnccl.so.2"
    target.write_text("dummy")
    link = tmp_path / "libnccl.so.2"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        validate_native_lib_path(str(link), "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_nonexistent(tmp_path):
    """H1: non-existent path is rejected."""
    with pytest.raises(ValueError, match="non-existent"):
        validate_native_lib_path(
            str(tmp_path / "missing.so"), "VLLM_NCCL_SO_PATH"
        )


def test_validate_nccl_path_rejects_world_writable(tmp_path):
    """H1: world-writable .so is rejected."""
    lib = tmp_path / "libnccl.so.2"
    lib.write_text("dummy")
    os.chmod(lib, 0o666)

    with pytest.raises(ValueError, match="group/world-writable"):
        validate_native_lib_path(str(lib), "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_world_writable_parent(tmp_path):
    """H1: world-writable parent directory is rejected."""
    lib = tmp_path / "libnccl.so.2"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o777)

    with pytest.raises(ValueError, match="parent directory"):
        validate_native_lib_path(str(lib), "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_accepts_safe(tmp_path):
    """H1: a regular file in a safe directory passes."""
    lib = tmp_path / "libnccl-local-inference.so.2.30.4"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o755)

    validate_native_lib_path(str(lib), "VLLM_NCCL_SO_PATH")


def test_find_nccl_library_validates_explicit_path(tmp_path):
    """H1: find_nccl_library validates VLLM_NCCL_SO_PATH when set."""
    from vllm.utils.nccl import find_nccl_library

    # A safe path should work (returns the path, doesn't raise)
    lib = tmp_path / "libnccl.so.2"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", str(lib)):
        result = find_nccl_library()
        assert result == str(lib)


def test_find_nccl_library_rejects_world_writable(tmp_path):
    """H1: find_nccl_library rejects a world-writable VLLM_NCCL_SO_PATH."""
    from vllm.utils.nccl import find_nccl_library

    lib = tmp_path / "libnccl.so.2"
    lib.write_text("dummy")
    os.chmod(lib, 0o666)  # world-writable
    os.chmod(tmp_path, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", str(lib)):
        with pytest.raises(ValueError, match="group/world-writable"):
            find_nccl_library()


def test_find_nccl_library_rejects_symlink(tmp_path):
    """H1: find_nccl_library rejects a symlink VLLM_NCCL_SO_PATH."""
    from vllm.utils.nccl import find_nccl_library

    target = tmp_path / "real.so"
    target.write_text("dummy")
    link = tmp_path / "libnccl.so.2"
    link.symlink_to(target)
    os.chmod(target, 0o644)
    os.chmod(tmp_path, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", str(link)):
        with pytest.raises(ValueError, match="symlink"):
            find_nccl_library()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
