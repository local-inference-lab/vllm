# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of target verification and parallel drafter graph widths."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def make_manager(monkeypatch):
    monkeypatch.setattr(
        cgu,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cgu.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cgu, "is_breakable_cudagraph_enabled", lambda: False)

    def make(cls, width, adaptation, schedule=None):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32],
                max_cudagraph_capture_size=32,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1, tensor_parallel_size=1
            ),
            speculative_config=SimpleNamespace(
                uses_acceptance_length_adaptation=lambda: adaptation,
                uses_batch_size_dynamic_speculative_decoding=lambda: (
                    schedule is not None
                ),
                num_speculative_tokens_per_batch_size=schedule,
            ),
            num_speculative_tokens=7,
        )
        return cls(config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, width)

    return make


@pytest.mark.parametrize("width", [7, 8], ids=["dspark-anchor", "dflash"])
@pytest.mark.parametrize(
    "adaptation,schedule",
    [
        (False, None),
        (True, None),
        (True, [(1, 2, 3), (3, 4, 0)]),
        (False, [(1, 2, 3), (3, 4, 0)]),
    ],
    ids=["fixed", "history", "history-with-caps", "batch-size"],
)
def test_parallel_drafter_capture_matches_fixed_runtime_width(
    make_manager, width, adaptation, schedule
):
    manager = make_manager(DFlashCudaGraphManager, width, adaptation, schedule)
    descs = manager._capture_descs[CUDAGraphMode.FULL]
    assert descs
    # The forward and its attention metadata must describe the same query rows.
    assert all(d.num_tokens == d.num_reqs * width for d in descs)
    assert {d.uniform_token_count for d in descs} == {width}
    manager._graphs_captured = True
    for num_reqs in range(1, 5):
        desc = manager.dispatch(num_reqs, num_reqs * width, width, 0)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc in descs
        assert desc.num_tokens == desc.num_reqs * width
    # A shorter target verification batch must never replay a drafter graph.
    assert manager.dispatch(1, width - 1, width - 1, 0).cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "adaptation,schedule,expected",
    [
        (False, None, {8}),
        (True, None, set(range(2, 9))),
        (True, [(1, 2, 3), (3, 4, 0)], {1, 2, 3, 4}),
        (False, [(1, 2, 3), (3, 4, 0)], {1, 4}),
    ],
    ids=["fixed", "history", "history-with-caps", "batch-size"],
)
def test_target_preserves_verification_widths(
    make_manager, adaptation, schedule, expected
):
    manager = make_manager(cgu.ModelCudaGraphManager, 8, adaptation, schedule)
    descs = manager._capture_descs[CUDAGraphMode.FULL]
    assert {d.uniform_token_count for d in descs} == expected
    manager._graphs_captured = True
    for width in expected:
        desc = manager.dispatch(1, width, width, 0)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.uniform_token_count == width


def test_autoregressive_drafter_preserves_single_query(make_manager):
    manager = make_manager(SpeculatorCudaGraphManager, 1, True)
    descs = manager._capture_descs[CUDAGraphMode.FULL]
    assert {d.uniform_token_count for d in descs} == {1}
    assert all(d.num_tokens == d.num_reqs for d in descs)
