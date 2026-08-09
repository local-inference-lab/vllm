# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation for operator-supplied native library paths."""

import os
import stat

_GROUP_WRITABLE = stat.S_IWGRP
_WORLD_WRITABLE = stat.S_IWOTH


def validate_native_lib_path(path: str, env_var: str) -> None:
    """Validate an operator-supplied native library path before loading."""
    resolved_path = os.path.abspath(path)
    if path != resolved_path:
        raise ValueError(f"{env_var} must be an absolute, canonical path: {path!r}")
    try:
        file_stat = os.lstat(resolved_path)
    except (FileNotFoundError, NotADirectoryError):
        raise ValueError(
            f"{env_var} points to a non-existent or non-regular file: {path!r}"
        ) from None
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"{env_var} must not be a symlink: {path!r}")
    if os.path.realpath(resolved_path) != resolved_path:
        raise ValueError(f"{env_var} path must not contain symlinks: {path!r}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(
            f"{env_var} points to a non-existent or non-regular file: {path!r}"
        )
    if file_stat.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE):
        raise ValueError(
            f"{env_var} file must not be group/world-writable "
            f"(mode {oct(file_stat.st_mode & 0o777)}): {path!r}"
        )

    parent = os.path.dirname(resolved_path)
    if parent:
        parent_stat = os.stat(parent)
        if parent_stat.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE):
            raise ValueError(
                f"{env_var} parent directory must not be group/world-writable: "
                f"{parent!r}"
            )
