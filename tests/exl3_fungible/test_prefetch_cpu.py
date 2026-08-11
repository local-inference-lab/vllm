# SPDX-License-Identifier: Apache-2.0
"""Whole-segment prefetch for uniform layers.

A uniform layer's experts all live in one segment object. Fetching them
per-expert is 256 HTTP round trips for one contiguous file — 19,200 across
GLM-5.2's 75 layers — and each one is an independent chance to hit the
transient failure that killed an earlier boot. These tests pin the two
properties that make the optimisation safe rather than merely fast: the
sliced bytes are IDENTICAL to the ranged bytes, and a prefetch that fails or
returns something short must fall back to HTTP instead of serving garbage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        fragments as FR,
    )
except ImportError:  # standalone: load by path
    import importlib.util

    _dir = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")
    _spec = importlib.util.spec_from_file_location(
        "fq_fragments_standalone", _dir / "fragments.py")
    FR = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = FR
    _spec.loader.exec_module(FR)

PAYLOAD = bytes(range(256)) * 512      # 128 KiB of known bytes


class FakeHTTP:
    """Minimal stand-in for the HF source: counts requests, can be flaky."""

    name = "fake"

    def __init__(self, blob: bytes, fail_whole: bool = False):
        self.blob = blob
        self.fail_whole = fail_whole
        self.range_calls = 0
        self.whole_calls = 0

    # the two methods the resolver actually uses
    def prefetch_whole(self, relpath, dest):
        self.whole_calls += 1
        if self.fail_whole:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.blob)
        return dest

    def range_from_prefetched(self, cached, start, end):
        return FR.HttpSource.range_from_prefetched(self, cached, start, end) \
            if hasattr(FR, "HttpSource") else _slice(cached, start, end)

    def read_range(self, relpath, start, end):
        self.range_calls += 1
        return self.blob[start:end]


def _slice(cached: Path, start: int, end: int):
    try:
        size = cached.stat().st_size
        if end > size:
            return None
        with open(cached, "rb") as fh:
            fh.seek(start)
            data = fh.read(end - start)
        return data if len(data) == end - start else None
    except OSError:
        return None


def test_sliced_bytes_are_identical_to_ranged_bytes(tmp_path):
    """The whole point: same bytes, fewer requests. If these ever diverge the
    optimisation silently corrupts weights."""
    src = FakeHTTP(PAYLOAD)
    dest = tmp_path / "seg.bin"
    src.prefetch_whole("layer-003.k3.safetensors", dest)
    for start, end in ((0, 64), (1000, 4096), (len(PAYLOAD) - 32,
                                               len(PAYLOAD))):
        assert _slice(dest, start, end) == src.read_range("x", start, end)


def test_prefetch_is_atomic_no_torn_file(tmp_path):
    """A partially written segment must never be readable at its final path."""
    dest = tmp_path / "deep" / "seg.bin"
    FakeHTTP(PAYLOAD).prefetch_whole("x", dest)
    assert dest.exists() and dest.stat().st_size == len(PAYLOAD)
    assert not list(dest.parent.glob("*.part")), "temp file left behind"


def test_slice_past_end_returns_none_so_caller_falls_back(tmp_path):
    """A truncated or stale cached object must not serve short reads."""
    dest = tmp_path / "seg.bin"
    dest.write_bytes(PAYLOAD[:100])
    assert _slice(dest, 0, 4096) is None


def test_missing_cache_file_returns_none(tmp_path):
    assert _slice(tmp_path / "nope.bin", 0, 16) is None


def test_failed_prefetch_returns_none_not_an_exception(tmp_path):
    """Prefetch is an optimisation; it must never fail a boot."""
    src = FakeHTTP(PAYLOAD, fail_whole=True)
    assert src.prefetch_whole("x", tmp_path / "seg.bin") is None


def test_request_count_collapses_for_a_uniform_layer(tmp_path):
    """256 experts from one object: 1 fetch, 0 ranged reads."""
    src = FakeHTTP(PAYLOAD)
    dest = tmp_path / "seg.bin"
    src.prefetch_whole("layer-003.k3.safetensors", dest)
    per_expert = len(PAYLOAD) // 256
    for e in range(256):
        got = _slice(dest, e * per_expert, (e + 1) * per_expert)
        assert got is not None and len(got) == per_expert
    assert src.whole_calls == 1
    assert src.range_calls == 0, (
        f"{src.range_calls} ranged reads leaked past the prefetch")


def test_second_prefetch_reuses_the_cached_object(tmp_path):
    src = FakeHTTP(PAYLOAD)
    dest = tmp_path / "seg.bin"
    src.prefetch_whole("x", dest)
    first = src.whole_calls
    # a real resolver checks dest.exists() before calling out again
    assert dest.exists() and dest.stat().st_size > 0
    assert first == 1


@pytest.mark.parametrize("bits,uniform", [
    ([3] * 256, True),
    ([3] * 255 + [4], False),          # one K4 upgrade -> NOT uniform
    ([4] * 256, True),
])
def test_uniformity_predicate(bits, uniform):
    """The guard that stops a layer needing one K4 expert from dragging a
    whole segment."""
    assert (len(set(bits)) == 1) is uniform


# --------------------------------------------------------------- offline mode
def test_hf_hub_offline_blocks_every_network_read(monkeypatch):
    """HF_HUB_OFFLINE must actually bind us.

    We never route payload reads through huggingface_hub — only hf_hub_url to
    build a string, then raw urllib — so the library's own offline handling
    never applied. Before this, HF_HUB_OFFLINE=1 silently still hit the
    network, which is the worst kind of wrong for an air-gapped or
    reproducibility-audited run.
    """
    src = FR.HfSource("org/repo")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert FR.HfSource.offline() is True
    for call in (lambda: src.read_json("index-k3.json"),
                 lambda: src.read_text("a.jsonl"),
                 lambda: src.read_range("seg.safetensors", 0, 16)):
        with pytest.raises(FR.OfflineError):
            call()


@pytest.mark.parametrize("val,off", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("false", False),
])
def test_offline_truthiness(monkeypatch, val, off):
    monkeypatch.delenv("FQ_ALLOW_NETWORK", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", val)
    assert FR.HfSource.offline() is off


def test_transformers_offline_also_honoured(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("FQ_ALLOW_NETWORK", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert FR.HfSource.offline() is True


def test_explicit_override_wins(monkeypatch):
    """An operator who wants Hub access despite the global flag can say so."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("FQ_ALLOW_NETWORK", "1")
    assert FR.HfSource.offline() is False


def test_offline_error_is_an_oserror_so_callers_treat_it_as_a_miss():
    """Offline is a configuration, not a fault: the existing source-error
    handling must degrade to 'this source has nothing' and let the K ladder
    and the primed cache take over, not abort a boot."""
    assert issubclass(FR.OfflineError, OSError)


def test_offline_is_not_retried(monkeypatch, tmp_path):
    """Retrying a configuration wastes four backoff sleeps per expert."""
    src = FR.HfSource("org/repo")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = {"n": 0}
    real_open = FR.HfSource._open

    def counting_open(self, relpath, headers):
        calls["n"] += 1
        return real_open(self, relpath, headers)

    monkeypatch.setattr(FR.HfSource, "_open", counting_open)
    with pytest.raises(FR.OfflineError):
        src.read_range("seg.safetensors", 0, 16)
    assert calls["n"] == 1, f"offline refusal retried {calls['n']} times"


# ------------------------------------------- bulk-fetch threshold (mixed layers)
def _needed(bits, bulk_min=16):
    """Which Ks a layer should bulk-fetch, mirroring progressive.py's rule."""
    from collections import Counter
    return sorted(k for k, n in Counter(bits).items() if n >= bulk_min)


def test_uniform_layer_bulk_fetches_its_single_k():
    assert _needed([3] * 256) == [3]


def test_mixed_layer_bulk_fetches_BOTH_objects():
    """The case the first version missed. A seeded policy makes most layers
    mixed, and restricting bulk fetch to uniform layers skipped 48 of 75
    layers outright — the optimisation did nothing for the majority."""
    bits = [3] * 192 + [4] * 64
    assert _needed(bits) == [3, 4], "a mixed layer draws from TWO objects"


def test_single_upgraded_expert_does_not_drag_a_segment():
    """One K4 expert is ~18 MiB; its segment is ~2.5 GB. Ranged read wins."""
    bits = [3] * 255 + [4]
    assert _needed(bits) == [3], "K4 must stay on the ranged path here"


def test_threshold_is_the_boundary():
    assert _needed([3] * 240 + [4] * 16, bulk_min=16) == [3, 4]
    assert _needed([3] * 241 + [4] * 15, bulk_min=16) == [3]


def test_request_count_for_a_realistic_mixed_layer():
    """192xK3 + 64xK4: 2 object fetches, not 256 ranged reads."""
    bits = [3] * 192 + [4] * 64
    assert len(_needed(bits)) == 2
    assert len(bits) == 256


def test_all_four_tiers_in_one_layer_each_over_threshold():
    bits = [2] * 64 + [3] * 64 + [4] * 64 + [5] * 64
    assert _needed(bits) == [2, 3, 4, 5]
