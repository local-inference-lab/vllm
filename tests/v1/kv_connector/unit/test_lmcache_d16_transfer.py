# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[4]
TRANSFER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
)
CUMEM = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/v1/platform/cuda/cumem_ipc.py"
)
IPC_WRAPPER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/v1/platform/cuda/ipc_wrapper.py"
)


def test_compressed_groups_select_production_safe_legacy_d2h() -> None:
    tree = ast.parse(TRANSFER.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_object_group_requires_legacy_d2h"
    )
    namespace = {
        "BaseCacheContext": object,
        "logger": logging.getLogger(__name__),
        "_production_lmcache_legacy_d2h_logged": False,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
            str(TRANSFER),
            "exec",
        ),
        namespace,
    )
    requires_legacy = cast(
        Callable[[Any, int], bool],
        namespace["_object_group_requires_legacy_d2h"],
    )

    def context(tokens_per_block: int, slots_per_block: int):
        return SimpleNamespace(
            kv_layer_groups_manager=SimpleNamespace(
                object_groups=[SimpleNamespace(kernel_group_indices=[0])],
                kernel_groups=[
                    SimpleNamespace(
                        tokens_per_block=tokens_per_block,
                        slots_per_block=slots_per_block,
                    )
                ],
            )
        )

    assert requires_legacy(context(8, 4), 0)
    assert not requires_legacy(context(4, 4), 0)
    source = ast.unparse(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "transfer_kv_per_object_group"
        )
    )
    assert "direction == lmcache_native.TransferDirection.D2H" in source
    assert "_object_group_requires_legacy_d2h" in source


def test_cumem_lifecycle_preserves_reusable_cuda_contexts() -> None:
    source = CUMEM.read_text() + IPC_WRAPPER.read_text()
    assert "cudaDeviceReset" not in source
    assert "ImportedCuMemRegistry" in source
    assert "release" in source
    assert "close" in source
