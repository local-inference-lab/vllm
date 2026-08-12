# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation for operator-supplied native library paths."""

import os
import stat

_GROUP_WRITABLE = stat.S_IWGRP
_WORLD_WRITABLE = stat.S_IWOTH


def _trusted_owner(metadata: os.stat_result) -> bool:
    return metadata.st_uid in {0, os.geteuid()}


def _validate_parent_chain(path: str, env_var: str) -> None:
    parent = os.path.dirname(path)
    while parent:
        metadata = os.stat(parent, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{env_var} parent path is not a directory: {parent!r}")
        if not _trusted_owner(metadata):
            raise ValueError(
                f"{env_var} parent directory has an untrusted owner: {parent!r}"
            )
        writable = metadata.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE)
        if writable and not metadata.st_mode & stat.S_ISVTX:
            raise ValueError(
                f"{env_var} parent directory chain must not be "
                f"group/world-writable: {parent!r}"
            )
        ancestor = os.path.dirname(parent)
        if ancestor == parent:
            break
        parent = ancestor


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
    if not _trusted_owner(file_stat):
        raise ValueError(f"{env_var} file has an untrusted owner: {path!r}")
    if file_stat.st_mode & (_GROUP_WRITABLE | _WORLD_WRITABLE):
        raise ValueError(
            f"{env_var} file must not be group/world-writable "
            f"(mode {oct(file_stat.st_mode & 0o777)}): {path!r}"
        )

    _validate_parent_chain(resolved_path, env_var)
