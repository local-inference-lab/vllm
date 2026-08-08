# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Path validation helpers for operator-supplied native library paths.

Used by callers that load shared libraries (``.so``) via ``ctypes.CDLL``
from paths taken from environment variables (e.g. ``VLLM_NCCL_SO_PATH``,
``VLLM_EXL3_ABI_SHIM``). Without validation, any process that can set
the env var achieves arbitrary native code execution (supply-chain /
local privilege boundary).
"""

import os

_WORLD_WRITABLE = 0o002
_GROUP_WRITABLE = 0o020


def validate_native_lib_path(path: str, env_var: str) -> None:
    """Validate a path to a native .so before loading it via ctypes.CDLL.

    Checks:
      - The path resolves to a regular file (not a directory or device node).
      - The path is not a symlink (prevents symlink-swap attacks).
      - The file is not group/world-writable (prevents in-place replacement
        by an unprivileged user).
      - The parent directory is not world-writable (prevents an attacker
        from creating a new .so in the same directory).

    Args:
        path: The filesystem path to validate.
        env_var: The name of the env var the path came from, used in error
            messages for operator diagnostics.

    Raises:
        ValueError: If any check fails.
    """
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise ValueError(
            f"{env_var} points to a non-existent or non-regular file: {path!r}"
        )
    if os.path.islink(path):
        raise ValueError(f"{env_var} must not be a symlink: {path!r}")
    st = os.stat(p)
    if st.st_mode & (_WORLD_WRITABLE | _GROUP_WRITABLE):
        raise ValueError(
            f"{env_var} file must not be group/world-writable "
            f"(mode {oct(st.st_mode & 0o777)}): {path!r}"
        )
    parent = os.path.dirname(p)
    if parent:
        pst = os.stat(parent)
        if pst.st_mode & _WORLD_WRITABLE:
            raise ValueError(
                f"{env_var} parent directory must not be world-writable: "
                f"{parent!r}"
            )
