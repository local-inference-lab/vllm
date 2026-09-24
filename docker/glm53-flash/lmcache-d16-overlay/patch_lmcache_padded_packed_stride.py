#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Enable physical dim-0 stride for LMCache packed HND format 12."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

RELATIVE = Path("lmcache/v1/gpu_connector/utils.py")
EXPECTED_BEFORE = "659fd0f7c980e405ec6086004c8e912b4371577b4089bbc25610fe23fcfcba5b"
EXPECTED_AFTER = "71f6e2a740a082955a59df3284fa13dc67d123d61268a3a71c986a4488477df4"

OLD_ALLOWLIST = """\
_BLOCK_AXIS_FORMATS: frozenset = frozenset(
    {
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
    }
)
"""

NEW_ALLOWLIST = """\
_BLOCK_AXIS_FORMATS: frozenset = frozenset(
    {
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS,
    }
)
"""

OLD_ERROR = """\
                    "a supported dim-0-padded format (only "
                    "NL_X_NB_BS_HS is); downstream transfer kernels "
"""

NEW_ERROR = """\
                    "a supported dim-0-padded format (only NL_X_NB_BS_HS and "
                    "NL_X_NB_NH_BS_CS are); downstream transfer kernels "
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch(root: Path, *, discover: bool = False) -> str:
    path = root / RELATIVE
    before_bytes = path.read_bytes()
    before = digest(before_bytes)
    if before == EXPECTED_AFTER:
        return before
    if before != EXPECTED_BEFORE:
        raise RuntimeError(f"unexpected source hash for {path}: {before}")

    text = before_bytes.decode()
    for old, new in (
        (OLD_ALLOWLIST, NEW_ALLOWLIST),
        (OLD_ERROR, NEW_ERROR),
    ):
        if text.count(old) != 1:
            raise RuntimeError(f"expected exactly one stride contract site in {path}")
        text = text.replace(old, new)

    after_bytes = text.encode()
    after = digest(after_bytes)
    if not discover and after != EXPECTED_AFTER:
        raise RuntimeError(
            f"patched source hash mismatch for {path}: {after}, "
            f"expected {EXPECTED_AFTER}"
        )
    path.write_bytes(after_bytes)
    return after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    print(patch(args.root, discover=args.discover))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
