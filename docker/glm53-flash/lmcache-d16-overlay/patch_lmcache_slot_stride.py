#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forward physical block strides through LMCache slot-transfer connectors."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import regex as re

CONNECTORS = Path("lmcache/v1/gpu_connector/gpu_connectors.py")
TORCH_OPS = Path("lmcache/v1/platform/torch_ops.py")
EXPECTED_BEFORE = {
    CONNECTORS: "5ae3307a1eef4d66f02c848a3826a97a3895cb125f95da425c2d0eb7614bfe2d",
    TORCH_OPS: "c7267f48f2ea5a8c1f32e03c528e6b6f2f6b99f1df6ef1e4d622be59b899eb1f",
}
EXPECTED_AFTER = {
    CONNECTORS: "94b230d6ac7cb2fb1676a54a52dfdd322553180729e0eb812d9371d4a494a221",
    TORCH_OPS: "5df72787a4ed23e126d260f0ad3fa753599866ef6ae41acf3e9777bcc14f859f",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch site in {path}")
    return text.replace(old, new)


def patch_connectors(text: str, path: Path) -> str:
    text = replace_once(
        text,
        """\
    normalize_and_discover_per_layer_formats,
    normalize_kv_and_discover_format,
)
""",
        """\
    normalize_and_discover_per_layer_formats,
    normalize_kv_and_discover_format,
    resolve_block_stride_and_log_layout,
)
""",
        path,
    )
    text = replace_once(
        text,
        """\
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.engine_kv_format)

        return self.kv_cache_pointers_on_gpu[idx]
""",
        """\
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.engine_kv_format)
        self.block_stride_elems = (
            resolve_block_stride_and_log_layout(
                kv_caches, self.engine_kv_format, layer_idx=0, group_idx=0
            )
            or 0
        )

        return self.kv_cache_pointers_on_gpu[idx]
""",
        path,
    )
    text = replace_once(
        text,
        """\
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(self.kvcaches, self.engine_kv_format)

        if self.metadata.kv_layer_groups_manager is None:
""",
        """\
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(self.kvcaches, self.engine_kv_format)
        self.block_stride_elems = (
            resolve_block_stride_and_log_layout(
                self.kvcaches, self.engine_kv_format, layer_idx=0, group_idx=0
            )
            or 0
        )

        if self.metadata.kv_layer_groups_manager is None:
""",
        path,
    )
    text, count = re.subn(
        r"(?m)^(\s*)head_size=self\.head_size,$",
        r"\1head_size=self.head_size,\n\1block_stride_elems=self.block_stride_elems,",
        text,
    )
    if count != 6:
        raise RuntimeError(f"expected exactly six slot-transfer call sites in {path}")
    return text


def patch_torch_ops(text: str, path: Path) -> str:
    return replace_once(
        text,
        """\
    block_size: int = 0,
    head_size: int = 0,
    skip_prefix_n_tokens: int = 0,
):
""",
        """\
    block_size: int = 0,
    head_size: int = 0,
    skip_prefix_n_tokens: int = 0,
    block_stride_elems: int = 0,
):
""",
        path,
    )


def patch(root: Path, *, discover: bool = False) -> dict[Path, str]:
    results: dict[Path, str] = {}
    for relative, transform in (
        (CONNECTORS, patch_connectors),
        (TORCH_OPS, patch_torch_ops),
    ):
        path = root / relative
        before_bytes = path.read_bytes()
        before = digest(before_bytes)
        if before == EXPECTED_AFTER[relative]:
            results[relative] = before
            continue
        if before != EXPECTED_BEFORE[relative]:
            raise RuntimeError(f"unexpected source hash for {path}: {before}")
        path.write_text(transform(before_bytes.decode(), path))
        after = digest(path.read_bytes())
        if not discover and after != EXPECTED_AFTER[relative]:
            raise RuntimeError(
                f"patched source hash mismatch for {path}: {after}, "
                f"expected {EXPECTED_AFTER[relative]}"
            )
        results[relative] = after
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    for relative, after in patch(args.root, discover=args.discover).items():
        print(f"{relative}={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
