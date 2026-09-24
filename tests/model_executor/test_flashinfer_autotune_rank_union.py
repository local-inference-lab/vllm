# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-rank FlashInfer autotune cache: rank union and consistent loading.

FlashInfer MoE cache keys include the EP rank. A leader-only cache let rank 0
hit its keys and skip synchronized profiling while its peers missed and
entered the all-reduce, hanging the warm start. These tests run the
collective helpers on threads that stand in for ranks.
"""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    RANK_UNION_CACHE_NAME,
    RANK_UNION_MARKER_KEY,
    load_autotune_cache_on_all_ranks,
    merge_rank_autotune_configs,
    rank_union_cache_path,
    read_rank_union_cache,
    save_rank_union_autotune_cache,
)

pytestmark = pytest.mark.cpu_test

WORLD = 3
META = {"gpu": "RTX PRO 6000", "flashinfer": "0.6.18"}


class _Exchange:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=10)
        self.slot: Any = None


class _FakeGroup:
    """GroupCoordinator stand-in: broadcast_object and barrier over threads."""

    def __init__(self, exchange: _Exchange, rank: int) -> None:
        self._exchange = exchange
        self.rank_in_group = rank
        self.world_size = exchange.world_size

    def broadcast_object(self, obj: Any = None, src: int = 0) -> Any:
        if self.rank_in_group == src:
            self._exchange.slot = obj
        self._exchange.barrier.wait()
        value = self._exchange.slot
        self._exchange.barrier.wait()
        return value

    def barrier(self) -> None:
        self._exchange.barrier.wait()


class _FakeTuner:
    """AutoTuner stand-in with FlashInfer's load/save/clear contract."""

    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata or META
        self.file_configs: dict[str, Any] = {}
        self.profiled: dict[str, Any] = {}

    def load_configs(self, path: str) -> bool:
        configs = json.loads(Path(path).read_text())
        assert RANK_UNION_MARKER_KEY not in configs
        saved_meta = configs.pop("_metadata", None)
        configs.pop("_generation", None)
        if saved_meta is not None and saved_meta != self.metadata:
            return False
        self.file_configs.update(configs)
        return True

    def save_configs(self, path: str) -> None:
        configs = {"_metadata": self.metadata, **self.file_configs, **self.profiled}
        Path(path).write_text(json.dumps(configs))

    def clear_cache(self) -> None:
        self.file_configs.clear()
        self.profiled.clear()

    def tune(self, key: str, value: Any) -> bool:
        """Return whether ``key`` needs synchronized profiling (a cache miss)."""
        if key in self.file_configs or key in self.profiled:
            return False
        self.profiled[key] = value
        return True


def _run_ranks(fn, world_size: int = WORLD) -> list[Any]:
    exchange = _Exchange(world_size)
    results: list[Any] = [None] * world_size
    errors: list[BaseException] = []

    def target(rank: int) -> None:
        try:
            results[rank] = fn(_FakeGroup(exchange, rank), rank)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            exchange.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def _rank_keys(rank: int) -> list[str]:
    # One shared GEMM key and one EP-rank-specific MoE key per rank.
    return ["gemm_m128", f"moe_ep{rank}_m128"]


def _autotune_pass(cache_path: Path, tuners: list[_FakeTuner]) -> list[list[str]]:
    """One start: load, tune every rank's keys, save. Returns misses per rank."""

    def run(group: _FakeGroup, rank: int) -> list[str]:
        tuner = tuners[rank]
        load_autotune_cache_on_all_ranks(cache_path, tuner, group)
        misses = [key for key in _rank_keys(rank) if tuner.tune(key, ["Runner", 1])]
        save_rank_union_autotune_cache(cache_path, tuner, group)
        return misses

    return _run_ranks(run)


def test_merge_unions_rank_specific_keys() -> None:
    configs = [
        {"_metadata": META, "_generation": "a", "gemm": ["R", 1], "moe_ep0": ["M", 2]},
        {"_metadata": {"other": 1}, "gemm": ["R", 1], "moe_ep1": ["M", 3]},
        {"gemm": ["R", 1], "moe_ep2": ["M", 4], "_records": {"ns": {"x": 1}}},
    ]
    contents, conflicts = merge_rank_autotune_configs(configs, WORLD)
    merged = json.loads(contents)
    assert conflicts == []
    assert merged["_metadata"] == META
    assert {k for k in merged if not k.startswith("_")} == {
        "gemm",
        "moe_ep0",
        "moe_ep1",
        "moe_ep2",
    }
    assert merged["_records"] == {"ns": {"x": 1}}
    assert merged[RANK_UNION_MARKER_KEY] == {"version": 1, "world_size": WORLD}
    assert merged["_generation"] != "a"
    # Deterministic output for identical inputs.
    assert merge_rank_autotune_configs(configs, WORLD)[0] == contents


def test_merge_keeps_lowest_rank_on_disagreement() -> None:
    contents, conflicts = merge_rank_autotune_configs(
        [
            {"gemm": ["R", 1], "_records": {"ns": {"x": 1}}},
            {"gemm": ["R", 2], "_records": {"ns": {"x": 2}}},
        ],
        2,
    )
    merged = json.loads(contents)
    assert merged["gemm"] == ["R", 1]
    assert merged["_records"] == {"ns": {"x": 1}}
    assert conflicts == ["_records.ns", "gemm"]


def test_union_cache_validation() -> None:
    contents, _ = merge_rank_autotune_configs([{"gemm": ["R", 1]}], WORLD)
    stripped = json.loads(read_rank_union_cache(contents, WORLD))
    assert RANK_UNION_MARKER_KEY not in stripped
    assert stripped["gemm"] == ["R", 1]
    assert read_rank_union_cache(contents, 2) is None
    # Leader-only files (no marker) and garbage are rejected.
    assert read_rank_union_cache(json.dumps({"gemm": ["R", 1]}).encode(), 3) is None
    assert read_rank_union_cache(b"not json", 3) is None
    assert rank_union_cache_path(Path("/c/autotune_configs.json")) == Path(
        "/c", RANK_UNION_CACHE_NAME
    )


def test_warm_start_hits_on_every_rank(tmp_path) -> None:
    cache_path = tmp_path / RANK_UNION_CACHE_NAME
    cold = _autotune_pass(cache_path, [_FakeTuner() for _ in range(WORLD)])
    assert cold == [_rank_keys(rank) for rank in range(WORLD)]
    saved = json.loads(cache_path.read_text())
    for rank in range(WORLD):
        assert f"moe_ep{rank}_m128" in saved

    # Fresh processes: every rank finds its own EP key, so none profiles.
    warm = _autotune_pass(cache_path, [_FakeTuner() for _ in range(WORLD)])
    assert warm == [[] for _ in range(WORLD)]


def test_leader_only_cache_is_ignored_by_all_ranks(tmp_path) -> None:
    cache_path = tmp_path / RANK_UNION_CACHE_NAME
    # The pre-fix format: rank 0's keys only, no union marker.
    cache_path.write_text(
        json.dumps({"_metadata": META, "gemm_m128": ["R", 1], "moe_ep0_m128": ["M", 1]})
    )
    misses = _autotune_pass(cache_path, [_FakeTuner() for _ in range(WORLD)])
    # Every rank retunes everything, so the synchronized profiles line up.
    assert misses == [_rank_keys(rank) for rank in range(WORLD)]


def test_cache_rejected_on_one_rank_is_dropped_everywhere(tmp_path) -> None:
    cache_path = tmp_path / RANK_UNION_CACHE_NAME
    _autotune_pass(cache_path, [_FakeTuner() for _ in range(WORLD)])

    tuners = [_FakeTuner() for _ in range(WORLD)]
    tuners[2].metadata = {"gpu": "different"}

    def run(group: _FakeGroup, rank: int) -> bool:
        return load_autotune_cache_on_all_ranks(cache_path, tuners[rank], group)

    assert _run_ranks(run) == [False] * WORLD
    assert all(not tuner.file_configs for tuner in tuners)


def test_single_rank_loads_plain_cache(tmp_path) -> None:
    cache_path = tmp_path / "autotune_configs.json"
    cache_path.write_text(json.dumps({"_metadata": META, "gemm": ["R", 1]}))
    tuner = _FakeTuner()

    def run(group: _FakeGroup, rank: int) -> bool:
        return load_autotune_cache_on_all_ranks(cache_path, tuner, group)

    assert _run_ranks(run, world_size=1) == [True]
    assert tuner.file_configs == {"gemm": ["R", 1]}
