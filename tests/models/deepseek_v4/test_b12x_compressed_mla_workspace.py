# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import b12x as b12x_mod
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_c128a_topk_width,
    get_compressed_mla_max_q_chunks,
    get_compressed_mla_split_cap,
    get_dspark_swa_index_width,
)

_MAX_ROWS = 2048
_LOCAL_HEADS = 32
_PAGE_SIZE = 64
_WINDOW_SIZE = 128
_INDEX_TOPK = 2048


class _RecordingWorkspaceManager:
    def __init__(self) -> None:
        self.specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None

    def get_simultaneous(
        self, *specs: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.specs = specs
        return []


def _install_recording_sparkinfer(monkeypatch, caps_calls: list[SimpleNamespace]):
    sparkinfer = types.ModuleType("sparkinfer")
    sparkinfer.__path__ = []
    attention = types.ModuleType("sparkinfer.attention")
    attention.__path__ = []
    compressed_mla = types.ModuleType("sparkinfer.attention.compressed_mla")

    def caps(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def plan(caps_value):
        caps_calls.append(caps_value)
        return SimpleNamespace(
            shapes_and_dtypes=lambda: (((1,), torch.uint8),),
        )

    def split_chunks_for_contract(*, rows: int, width: int, max_chunks: int) -> int:
        del width
        return min(max_chunks, 8 if rows <= 256 else 1)

    compressed_mla.Caps = caps  # type: ignore[attr-defined]
    compressed_mla.plan = plan  # type: ignore[attr-defined]
    compressed_mla.split_chunks_for_contract = (  # type: ignore[attr-defined]
        split_chunks_for_contract
    )
    monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer)
    monkeypatch.setitem(sys.modules, "sparkinfer.attention", attention)
    monkeypatch.setitem(
        sys.modules,
        "sparkinfer.attention.compressed_mla",
        compressed_mla,
    )
    return split_chunks_for_contract


def _make_layer(
    *,
    compress_ratio: int,
    dspark: bool,
    dcp_world_size: int = 1,
):
    layer = object.__new__(b12x_mod.DeepseekV4B12xMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = compress_ratio
    layer.topk_indices_buffer = (
        torch.empty((1, _INDEX_TOPK), dtype=torch.int32)
        if compress_ratio == 4
        else None
    )
    layer.indexer = None
    layer.max_model_len = 524288
    layer.window_size = _WINDOW_SIZE
    layer.max_num_batched_tokens = _MAX_ROWS
    layer.swa_cache_layer = SimpleNamespace(block_size=_PAGE_SIZE)
    speculative_config = (
        SimpleNamespace(
            use_dspark=lambda: True,
            num_speculative_tokens=5,
        )
        if dspark
        else None
    )
    layer.vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_world_size,
        ),
    )
    return layer


@pytest.mark.parametrize(
    ("compress_ratio", "dspark", "expected_width"),
    [
        pytest.param(1, False, 128, id="swa-causal"),
        pytest.param(1, True, 512, id="swa-dspark"),
        pytest.param(4, False, 2176, id="c4-causal"),
        pytest.param(4, True, 2560, id="c4-dspark"),
        pytest.param(128, False, 4224, id="c128-causal"),
        pytest.param(128, True, 4608, id="c128-dspark"),
    ],
)
def test_reserve_uses_full_runtime_width(
    monkeypatch,
    compress_ratio: int,
    dspark: bool,
    expected_width: int,
) -> None:
    caps_calls: list[SimpleNamespace] = []
    split_chunks = _install_recording_sparkinfer(monkeypatch, caps_calls)
    workspace = _RecordingWorkspaceManager()
    monkeypatch.setattr(b12x_mod, "current_workspace_manager", lambda: workspace)
    layer = _make_layer(compress_ratio=compress_ratio, dspark=dspark)

    layer._reserve_dummy_compressed_mla_scratch(
        torch.empty((1, _LOCAL_HEADS, 512), dtype=torch.bfloat16)
    )

    caps = caps_calls[-1]
    split_cap = get_compressed_mla_split_cap(expected_width)
    assert caps.max_width == expected_width
    assert caps.max_q_rows == _MAX_ROWS
    assert caps.max_q_chunks == get_compressed_mla_max_q_chunks(
        _MAX_ROWS,
        expected_width,
        split_cap,
        split_chunks,
    )
    assert workspace.specs is not None


def test_reserve_uses_dcp_gathered_heads(monkeypatch) -> None:
    caps_calls: list[SimpleNamespace] = []
    _install_recording_sparkinfer(monkeypatch, caps_calls)
    monkeypatch.setattr(
        b12x_mod,
        "current_workspace_manager",
        lambda: _RecordingWorkspaceManager(),
    )
    layer = _make_layer(compress_ratio=128, dspark=False, dcp_world_size=2)

    layer._reserve_dummy_compressed_mla_scratch(
        torch.empty((1, _LOCAL_HEADS, 512), dtype=torch.bfloat16)
    )

    assert caps_calls[-1].num_q_heads == _LOCAL_HEADS * 2


def test_workspace_width_helpers_match_reporter_geometry() -> None:
    assert get_c128a_topk_width(524288, 128) == 4096
    assert get_dspark_swa_index_width(128, 5) == 512
    assert get_dspark_swa_index_width(512, 5) == 1024


def test_q_chunk_envelope_is_cached() -> None:
    call_count = 0

    def split_chunks_for_contract(**kwargs) -> int:
        nonlocal call_count
        call_count += 1
        return min(kwargs["max_chunks"], 8 if kwargs["rows"] <= 256 else 1)

    get_compressed_mla_max_q_chunks.cache_clear()
    split_cap = get_compressed_mla_split_cap(4608)
    for _ in range(2):
        assert (
            get_compressed_mla_max_q_chunks(
                _MAX_ROWS,
                4608,
                split_cap,
                split_chunks_for_contract,
            )
            == _MAX_ROWS
        )
    assert call_count == _MAX_ROWS


def _aligned_nbytes(
    specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> int:
    return sum(
        round_up(math.prod(shape) * dtype.itemsize, 256) for shape, dtype in specs
    )


def _runtime_plan_nbytes(compressed_mla, *, rows: int, width: int, heads: int) -> int:
    split_cap = get_compressed_mla_split_cap(width)
    splits = compressed_mla.split_chunks_for_contract(
        rows=rows,
        width=width,
        max_chunks=split_cap,
    )
    runtime_plan = compressed_mla.plan(
        compressed_mla.Caps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            head_dim=512,
            v_head_dim=512,
            page_size=_PAGE_SIZE,
            max_chunks_per_row=splits,
        )
    )
    return _aligned_nbytes(runtime_plan.shapes_and_dtypes())


@pytest.mark.parametrize(
    (
        "compress_ratio",
        "dspark",
        "dcp_world_size",
        "runtime_widths",
        "expected_reserve_mib",
    ),
    [
        pytest.param(1, False, 1, (128,), 64.502929688, id="swa-causal"),
        pytest.param(1, True, 1, (128, 512), 64.502929688, id="swa-dspark"),
        pytest.param(4, False, 1, (2176,), 273.315429688, id="c4-causal"),
        pytest.param(
            4,
            True,
            1,
            (2176, 2560),
            321.502929688,
            id="c4-dspark",
        ),
        pytest.param(128, False, 1, (4224,), 530.315429688, id="c128-causal"),
        pytest.param(
            128,
            True,
            1,
            (4224, 4608),
            578.502929688,
            id="c128-dspark",
        ),
        pytest.param(128, False, 2, (4224,), 1060.627929688, id="c128-dcp2"),
    ],
)
def test_real_sparkinfer_reserve_dominates_runtime_envelope(
    monkeypatch,
    compress_ratio: int,
    dspark: bool,
    dcp_world_size: int,
    runtime_widths: tuple[int, ...],
    expected_reserve_mib: float,
) -> None:
    compressed_mla = pytest.importorskip("sparkinfer.attention.compressed_mla")
    workspace = _RecordingWorkspaceManager()
    monkeypatch.setattr(b12x_mod, "current_workspace_manager", lambda: workspace)
    layer = _make_layer(
        compress_ratio=compress_ratio,
        dspark=dspark,
        dcp_world_size=dcp_world_size,
    )

    layer._reserve_dummy_compressed_mla_scratch(
        torch.empty((1, _LOCAL_HEADS, 512), dtype=torch.bfloat16)
    )

    assert workspace.specs is not None
    reserve_bytes = _aligned_nbytes(workspace.specs)
    assert reserve_bytes / (1 << 20) == pytest.approx(expected_reserve_mib)
    runtime_heads = _LOCAL_HEADS * dcp_world_size
    runtime_bytes = max(
        _runtime_plan_nbytes(
            compressed_mla,
            rows=rows,
            width=runtime_width,
            heads=runtime_heads,
        )
        for runtime_width in runtime_widths
        for rows in range(1, _MAX_ROWS + 1)
    )
    assert reserve_bytes >= runtime_bytes
