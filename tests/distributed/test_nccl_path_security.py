# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

from vllm.utils.path_validation import validate_native_lib_path


def test_native_lib_path_rejects_relative_path():
    with pytest.raises(ValueError, match="absolute, canonical"):
        validate_native_lib_path("libnccl.so.2", "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_final_symlink(tmp_path):
    target = tmp_path / "real.so"
    target.write_text("dummy")
    link = tmp_path / "libnccl.so.2"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        validate_native_lib_path(str(link), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_symlinked_parent(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    library = real_dir / "libnccl.so.2"
    library.write_text("dummy")
    redirect = tmp_path / "redirect"
    redirect.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_native_lib_path(str(redirect / library.name), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_nonexistent_or_nonregular(tmp_path):
    with pytest.raises(ValueError, match="non-existent or non-regular"):
        validate_native_lib_path(str(tmp_path / "missing.so"), "VLLM_NCCL_SO_PATH")
    with pytest.raises(ValueError, match="non-existent or non-regular"):
        validate_native_lib_path(str(tmp_path), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_group_writable_file(tmp_path):
    library = tmp_path / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(library, 0o664)  # noqa: S103

    with pytest.raises(ValueError, match="file must not be group/world-writable"):
        validate_native_lib_path(str(library), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_group_writable_parent(tmp_path):
    library = tmp_path / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(library, 0o644)  # noqa: S103
    os.chmod(tmp_path, 0o775)  # noqa: S103

    with pytest.raises(ValueError, match="parent directory"):
        validate_native_lib_path(str(library), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_rejects_writable_grandparent(tmp_path):
    grandparent = tmp_path / "writable"
    grandparent.mkdir()
    parent = grandparent / "secure"
    parent.mkdir()
    library = parent / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(grandparent, 0o775)  # noqa: S103
    os.chmod(parent, 0o755)  # noqa: S103
    os.chmod(library, 0o644)  # noqa: S103

    with pytest.raises(ValueError, match="parent directory chain"):
        validate_native_lib_path(str(library), "VLLM_NCCL_SO_PATH")


def test_native_lib_path_accepts_secure_regular_file(tmp_path):
    library = tmp_path / "libnccl-local-inference.so.2.30.4"
    library.write_text("dummy")
    os.chmod(library, 0o644)  # noqa: S103
    os.chmod(tmp_path, 0o755)  # noqa: S103

    validate_native_lib_path(str(library), "VLLM_NCCL_SO_PATH")


def test_find_nccl_library_validates_configured_override(tmp_path):
    from vllm.utils.nccl import find_nccl_library

    library = tmp_path / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(library, 0o664)  # noqa: S103
    os.chmod(tmp_path, 0o755)  # noqa: S103

    with (
        patch("vllm.envs.VLLM_NCCL_SO_PATH", str(library)),
        pytest.raises(ValueError, match="group/world-writable"),
    ):
        find_nccl_library()


def test_find_nccl_library_preserves_secure_override(tmp_path):
    from vllm.utils.nccl import find_nccl_library

    library = tmp_path / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(library, 0o644)  # noqa: S103
    os.chmod(tmp_path, 0o755)  # noqa: S103

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", str(library)):
        assert find_nccl_library() == str(library)


def test_nccl_library_validates_direct_path_before_loading(tmp_path):
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    library = tmp_path / "libnccl.so.2"
    library.write_text("dummy")
    os.chmod(library, 0o664)  # noqa: S103
    os.chmod(tmp_path, 0o755)  # noqa: S103

    with (
        patch(
            "vllm.distributed.device_communicators.pynccl_wrapper.ctypes.CDLL"
        ) as load,
        pytest.raises(ValueError, match="group/world-writable"),
    ):
        NCCLLibrary(str(library))
    load.assert_not_called()
