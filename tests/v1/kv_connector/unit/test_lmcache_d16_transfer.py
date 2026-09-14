# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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


def _load_transfer_function(name: str) -> Callable[..., Any]:
    tree = ast.parse(TRANSFER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "BaseCacheContext": object,
        "ObjectGroupInfo": object,
        "Sequence": list,
        "torch": SimpleNamespace(Tensor=object),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(TRANSFER),
            "exec",
        ),
        namespace,
    )
    return cast(Callable[..., Any], namespace[name])


def _sparse_transfer_context() -> SimpleNamespace:
    manager = SimpleNamespace(
        num_kernel_groups=2,
        get_subchunk_sw_size_tokens=lambda group: 2304 if group == 0 else 9216,
    )
    return SimpleNamespace(
        lmcache_tokens_per_chunk=9216,
        kv_layer_groups_manager=manager,
        calculate_num_blocks=lambda tokens, group: tokens
        // (2304 if group == 0 else 9216),
        stage_block_ids=lambda ids: ids,
    )


def test_sparse_transfer_uses_effective_sliding_window_block_counts() -> None:
    effective_blocks = _load_transfer_function("effective_blocks_per_chunk")

    assert effective_blocks(_sparse_transfer_context()) == [1, 1]


def test_sparse_transfer_accepts_raw_and_exact_state_block_tables() -> None:
    downsample = _load_transfer_function("downsample_and_stage_block_ids")
    context = _sparse_transfer_context()
    exact_ids = [[104, 108], [9, 10]]
    raw_ids = [[1, 2, 3, 104, 5, 6, 7, 108], [9, 10]]

    assert downsample(context, exact_ids, num_chunks=2) == exact_ids
    assert downsample(context, raw_ids, num_chunks=2) == exact_ids

    with pytest.raises(AssertionError, match="raw count 8 or reduced count 2"):
        downsample(context, [[1, 2, 3], [9, 10]], num_chunks=2)


def test_sparse_transfer_masks_mamba_after_block_normalization() -> None:
    downsample = _load_transfer_function("downsample_and_stage_block_ids")
    all_null_masks = _load_transfer_function("all_null_chunk_masks")
    context = _sparse_transfer_context()
    block_ids = [[0, 0, 0, 0, 0, 0, 0, 104], [9, 10]]

    downsample(context, block_ids, num_chunks=2)
    masks = all_null_masks(
        block_ids,
        [
            SimpleNamespace(kernel_group_indices=[0]),
            SimpleNamespace(kernel_group_indices=[1]),
        ],
        [1, 1],
        num_chunks=2,
    )

    assert masks == [[True, False], [False, False]]


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
