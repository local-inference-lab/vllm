# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation helpers for operator-supplied native library paths."""

import os
import stat

_GROUP_WRITABLE = stat.S_IWGRP
_WORLD_WRITABLE = stat.S_IWOTH


def validate_native_lib_path(path: str, env_var: str) -> str:
    """Validate a native library path and return the path to load.

    The returned value MUST be the one handed to ``ctypes.CDLL``. Validating
    one path and loading another re-opens the bypass this function closes.

    Args:
        path: Operator-supplied path from ``env_var``.
        env_var: Environment variable name, used in error messages.

    Returns:
        The canonical absolute path that passed validation.

    Raises:
        ValueError: The path is relative, contains a symlink component, is not
            a regular file, or it/an ancestor is group- or world-writable.
    """
    if not os.path.isabs(path):
        # A bare soname makes dlopen search its own path, so the inode that
        # was validated is not necessarily the inode that gets loaded.
        raise ValueError(f"{env_var} must be an absolute path: {path!r}")

    resolved = os.path.realpath(path)
    if resolved != os.path.normpath(path):
        # realpath differs => at least one component is a symlink. islink()
        # only inspects the final component, so it cannot catch this.
        raise ValueError(
            f"{env_var} must not contain a symlink component: "
            f"{path!r} resolves to {resolved!r}"
        )

    st = os.lstat(resolved)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{env_var} must not be a symlink: {resolved!r}")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{env_var} must be a regular file: {resolved!r}")
    if st.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE):
        raise ValueError(
            f"{env_var} must not be group/world-writable: {resolved!r}"
        )

    # Walk every ancestor: a writable directory anywhere on the chain lets an
    # attacker replace a component and redirect the load.
    parent = os.path.dirname(resolved)
    while True:
        pst = os.lstat(parent)
        writable = pst.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE)
        # The sticky bit (/tmp) stops non-owners from replacing entries.
        if writable and not (pst.st_mode & stat.S_ISVTX):
            raise ValueError(
                f"{env_var} ancestor directory must not be "
                f"group/world-writable: {parent!r}"
            )
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt

    return resolved
