# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security test for NCCL .so path validation (H1)."""

import os
from unittest.mock import patch

import pytest

from vllm.utils.path_validation import validate_native_lib_path


def _base(tmp_path):
    """Return a realpath-resolved base dir so the /var -> /private/var
    symlink on macOS does not itself trip the symlink-component check."""
    return os.path.realpath(str(tmp_path))


def test_validate_nccl_path_rejects_symlink(tmp_path):
    """H1: VLLM_NCCL_SO_PATH must not be a symlink."""
    base = _base(tmp_path)
    target = os.path.join(base, "real_libnccl.so.2")
    with open(target, "w") as f:
        f.write("dummy")
    link = os.path.join(base, "libnccl.so.2")
    os.symlink(target, link)

    with pytest.raises(ValueError, match="symlink"):
        validate_native_lib_path(link, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_nonexistent(tmp_path):
    """H1: non-existent path is rejected (lstat raises FileNotFoundError)."""
    base = _base(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_native_lib_path(
            os.path.join(base, "missing.so"), "VLLM_NCCL_SO_PATH"
        )


def test_validate_nccl_path_rejects_world_writable(tmp_path):
    """H1: world-writable .so is rejected."""
    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o666)  # noqa: S103 -- intentional world-writable fixture

    with pytest.raises(ValueError, match="group/world-writable"):
        validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_group_writable_file(tmp_path):
    """H1: group-writable (but not world-writable) .so is rejected.

    Uses 0o664 so the rejection is distinguishable from the world-writable
    0o666 case (N4).
    """
    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o664)  # noqa: S103 -- intentional group-writable fixture

    with pytest.raises(ValueError, match="group/world-writable"):
        validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_world_writable_parent(tmp_path):
    """H1: world-writable parent directory is rejected."""
    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(base, 0o777)  # noqa: S103 -- intentional world-writable parent

    with pytest.raises(ValueError, match="ancestor directory"):
        validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_group_writable_parent(tmp_path):
    """H1: group-writable parent directory (0o775, no sticky) is rejected."""
    base = _base(tmp_path)
    sub = os.path.join(base, "subdir")
    os.makedirs(sub, mode=0o755)
    lib = os.path.join(sub, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(sub, 0o775)  # noqa: S103 -- intentional group-writable parent

    with pytest.raises(ValueError, match="ancestor directory"):
        validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_symlinked_parent_component(tmp_path):
    """H1: a symlink in a parent component is rejected even when the final
    path element is a real file.

    Builds evil_parent/real/lib.so (the file the kernel would load) and a
    symlink link -> evil_parent/real, then passes link/../real/lib.so.
    os.path.normpath collapses the '..' lexically giving base/real/lib.so,
    but os.path.realpath resolves through the symlink giving
    base/evil_parent/real/lib.so — the two diverge, proving the bypass
    (B4) and that the validator catches it.
    """
    base = _base(tmp_path)
    evil_parent = os.path.join(base, "evil_parent")
    real_dir = os.path.join(evil_parent, "real")
    os.makedirs(real_dir, mode=0o755)
    lib = os.path.join(real_dir, "lib.so")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(evil_parent, 0o755)

    # Symlink at base/link pointing to evil_parent/real (different parent
    # than link itself, so normpath(..) and realpath(..) diverge).
    link = os.path.join(base, "link")
    os.symlink(real_dir, link)

    # normpath -> base/real/lib.so  (lexical collapse of link/..)
    # realpath -> base/evil_parent/real/lib.so  (kernel resolves link first)
    bypass_path = os.path.join(link, "..", "real", "lib.so")

    with pytest.raises(ValueError, match="symlink component"):
        validate_native_lib_path(bypass_path, "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_rejects_relative_path(tmp_path):
    """H1: a relative path is rejected (dlopen would search its own path)."""
    with pytest.raises(ValueError, match="absolute path"):
        validate_native_lib_path("libnccl.so.2", "VLLM_NCCL_SO_PATH")


def test_validate_nccl_path_returns_canonical_resolved_path(tmp_path):
    """H1: the returned value is the canonical realpath-resolved path."""
    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(base, 0o755)

    result = validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")
    assert result == os.path.realpath(lib)
    assert os.path.isabs(result)
    assert result == lib  # already canonical


def test_validate_nccl_path_accepts_safe(tmp_path):
    """H1: a regular file in a safe directory passes."""
    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl-local-inference.so.2.30.4")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(base, 0o755)

    validate_native_lib_path(lib, "VLLM_NCCL_SO_PATH")


def test_find_nccl_library_validates_explicit_path(tmp_path):
    """H1: find_nccl_library validates VLLM_NCCL_SO_PATH when set and
    returns the canonical path."""
    from vllm.utils.nccl import find_nccl_library

    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o644)
    os.chmod(base, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", lib):
        result = find_nccl_library()
        assert result == lib


def test_find_nccl_library_rejects_world_writable(tmp_path):
    """H1: find_nccl_library rejects a world-writable VLLM_NCCL_SO_PATH."""
    from vllm.utils.nccl import find_nccl_library

    base = _base(tmp_path)
    lib = os.path.join(base, "libnccl.so.2")
    with open(lib, "w") as f:
        f.write("dummy")
    os.chmod(lib, 0o666)  # noqa: S103 -- intentional world-writable fixture
    os.chmod(base, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", lib):
        with pytest.raises(ValueError, match="group/world-writable"):
            find_nccl_library()


def test_find_nccl_library_rejects_symlink(tmp_path):
    """H1: find_nccl_library rejects a symlink VLLM_NCCL_SO_PATH."""
    from vllm.utils.nccl import find_nccl_library

    base = _base(tmp_path)
    target = os.path.join(base, "real.so")
    with open(target, "w") as f:
        f.write("dummy")
    link = os.path.join(base, "libnccl.so.2")
    os.symlink(target, link)
    os.chmod(target, 0o644)
    os.chmod(base, 0o755)

    with patch("vllm.envs.VLLM_NCCL_SO_PATH", link):
        with pytest.raises(ValueError, match="symlink"):
            find_nccl_library()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
