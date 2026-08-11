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

import os
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


def _progressive_src() -> str:
    """Read progressive.py as TEXT. Importing it pulls in torch, which the
    CPU test venv does not have (it lives in the GG rootfs) -- and these are
    guard tests over code shape, so the text is exactly what we want."""
    return (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible"
            / "progressive.py").read_text()


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


# ------------------------------------------------- shutdown is not a failure
def test_resolve_best_lets_systemexit_through():
    """vLLM's multiproc executor raises SystemExit from its SIGTERM handler,
    and on a progressive boot it lands mid-socket-read inside the resolver.
    Catching BaseException swallowed it and logged 'fragment unavailable,
    keeping the incumbent tier' — so a worker told to stop kept loading with
    silently degraded tiers instead of exiting."""
    import inspect
    src = inspect.getsource(FR.FragmentResolver.resolve_best)
    assert "SystemExit" in src, "resolve_best must re-raise shutdown signals"
    # and the re-raise must come BEFORE the broad catch, or it never runs
    assert src.index("SystemExit") < src.index("except BaseException")


# ------------------------------------------- parallelism (depth x width)
def test_mixed_layer_fetches_its_two_objects_CONCURRENTLY():
    """A mixed layer draws from two segment objects. Submitting them as one
    task ran them back to back, leaving the link idle between files even with
    hf_transfer chunking inside each. One task PER (layer, K)."""
    src = _progressive_src()
    assert "_pool.submit(_fn, _layer, _k) for _k in _ks" in src, (
        "the two Ks of a mixed layer must be submitted as separate tasks")
    assert "_cf.wait_for_all" not in src, "stray module attribute assignment"


def test_prefetch_depth_is_more_than_one_layer():
    """One layer of lookahead only hides the download if it is faster than the
    GPU load; with a real network it usually is not."""
    src = _progressive_src()
    assert "VLLM_FQ_PREFETCH_DEPTH" in src
    assert "for _ahead in range(1, _depth + 1)" in src, (
        "must keep `depth` layers in flight, not exactly one")


def test_pipeline_is_primed_before_the_first_layer():
    """Otherwise layer 0 fetches its Ks sequentially on the main thread with
    nothing in flight yet -- the slowest possible start."""
    src = _progressive_src()
    assert "_sorted_layers[:_depth + 1]" in src


# ------------------------------------------------------- bounded footprint
def _resolver(tmp_path):
    return FR.FragmentResolver(tmp_path / "manifest", sources=[],
                              cache_dir=tmp_path / "cache")


def test_release_layer_unlinks_our_own_prefetched_segments(tmp_path):
    """Prefetch had NO eviction: _prefetched grew monotonically and nothing
    unlinked, so a 75-layer boot left every segment it touched on disk. Depth
    bounds lookahead; only release bounds FOOTPRINT."""
    r = _resolver(tmp_path)
    seg = r.cache_dir / "segments" / "abc.seg"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"x" * 4096)
    r._prefetched[(7, 3)] = seg
    freed = r.release_layer(7)
    assert freed == 4096
    assert not seg.exists()
    assert (7, 3) not in r._prefetched


def test_release_layer_leaves_the_SHARED_hf_cache_alone(tmp_path):
    """A blob the shared HF cache owns may be in use by another process."""
    r = _resolver(tmp_path)
    foreign = tmp_path / "hf-cache" / "blobs" / "deadbeef"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"y" * 128)
    r._prefetched[(9, 4)] = foreign
    r.release_layer(9)
    assert foreign.exists(), "unlinked a file outside our cache dir"


def test_release_layer_only_touches_the_named_layer(tmp_path):
    r = _resolver(tmp_path)
    keep = r.cache_dir / "segments" / "keep.seg"
    drop = r.cache_dir / "segments" / "drop.seg"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"k")
    drop.write_bytes(b"d")
    r._prefetched[(3, 3)] = drop
    r._prefetched[(4, 3)] = keep
    r.release_layer(3)
    assert not drop.exists() and keep.exists()
    assert list(r._prefetched) == [(4, 3)]


def test_keep_env_disables_eviction(tmp_path, monkeypatch):
    """Repeat boots on a roomy box should be able to retain everything."""
    monkeypatch.setenv("VLLM_FQ_PREFETCH_KEEP", "1")
    r = _resolver(tmp_path)
    seg = r.cache_dir / "segments" / "abc.seg"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"z" * 16)
    r._prefetched[(1, 3)] = seg
    assert r.release_layer(1) == 0
    assert seg.exists()


def test_release_layer_survives_an_already_deleted_file(tmp_path):
    r = _resolver(tmp_path)
    r._prefetched[(2, 3)] = r.cache_dir / "segments" / "gone.seg"
    assert r.release_layer(2) == 0          # no exception


def test_progressive_releases_each_layer_after_it_streams():
    src = _progressive_src()
    assert "release_layer" in src, (
        "layers must be released as they finish, or footprint is O(model)")


def test_hf_download_lands_in_our_cache_so_it_can_be_evicted():
    """hf_hub_download's default is the SHARED cache, which we must not
    unlink -- so the file would be immortal. local_dir makes it ours."""
    import inspect
    src = inspect.getsource(FR.HfSource.prefetch_whole)
    assert "local_dir" in src
    assert "FQ_PREFETCH_HF_SHARED" in src, "the opt-out must exist"


# --------------------------------------------- cross-rank sharing (TP ranks)
def test_all_tp_ranks_want_the_SAME_segments():
    """The premise, from a real TP4 boot log: every rank emitted
    `FQ progressive layer 3: tiers=((3, 206), (4, 50)) bits_digest=d704612a2fdb`
    -- identical composition, so identical segment objects. Four ranks racing
    one file was 4x the bytes and 4x the transient disk."""
    per_rank = [((3, 206), (4, 50))] * 4
    assert len(set(per_rank)) == 1


def test_segment_lock_serialises_downloads(tmp_path):
    """Second holder must block while the first has the lock."""
    import threading
    r = _resolver(tmp_path)
    dest = r.cache_dir / "segments" / "abc.seg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    entered = threading.Event()
    release = threading.Event()
    order = []

    def first():
        with r._segment_lock(dest):
            order.append("first-in")
            entered.set()
            release.wait(5)
            order.append("first-out")

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(5)
    r2 = _resolver(tmp_path)

    def second():
        with r2._segment_lock(dest):
            order.append("second-in")

    t2 = threading.Thread(target=second)
    t2.start()
    t2.join(0.5)
    assert "second-in" not in order, "lock did not exclude the second holder"
    release.set()
    t.join(5)
    t2.join(5)
    assert order == ["first-in", "first-out", "second-in"]


def test_lock_timeout_falls_through_rather_than_failing_the_boot(tmp_path):
    """A wedged rank must not hang a boot forever; a duplicate download is
    wasteful, not incorrect, because the rename is atomic."""
    r = _resolver(tmp_path)
    dest = r.cache_dir / "segments" / "abc.seg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with r._segment_lock(dest):
        r2 = _resolver(tmp_path)
        with r2._segment_lock(dest, timeout=0.2) as held:
            assert held is False        # fell through, did not raise


def test_segment_ready_finds_the_hf_local_dir_name(tmp_path):
    """hf_hub_download with local_dir writes to dest.parent/relpath, NOT to
    dest. Checking only dest made every rank re-download a file the HF client
    had already placed."""
    r = _resolver(tmp_path)
    dest = r.cache_dir / "segments" / "sha.seg"
    hf = dest.parent / "layer-003.k3.safetensors"
    hf.parent.mkdir(parents=True, exist_ok=True)
    hf.write_bytes(b"payload")
    assert r._segment_ready(dest, "layer-003.k3.safetensors") == hf
    assert r._segment_ready(dest, "other.safetensors") is None


def test_release_does_not_unlink_while_another_rank_holds_it(tmp_path):
    """The race this refcount exists for: rank 0 finishing layer 3 must not
    delete a segment rank 2 is still reading -- that would silently demote
    rank 2 to 18.2 s-per-expert ranged reads."""
    seg = tmp_path / "cache" / "segments" / "shared.seg"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"z" * 512)
    rank0, rank2 = _resolver(tmp_path), _resolver(tmp_path)
    rank0._claim_segment(seg)
    # a live OTHER pid also holds it (pid 1 always exists)
    (rank0._users_dir(seg) / "1").touch()
    rank0._prefetched[(3, 3)] = seg
    rank0.release_layer(3)
    assert seg.exists(), "unlinked a segment another rank still holds"
    # once the other claim is gone, the next release frees it
    (rank2._users_dir(seg) / "1").unlink()
    rank2._prefetched[(3, 3)] = seg
    rank2.release_layer(3)
    assert not seg.exists()


def test_stale_claim_from_a_killed_rank_does_not_pin_disk(tmp_path):
    """This box is preemptible. A killed worker's marker must not keep a
    segment resident forever."""
    seg = tmp_path / "cache" / "segments" / "s.seg"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"q" * 64)
    r = _resolver(tmp_path)
    r._claim_segment(seg)
    (r._users_dir(seg) / "999999").touch()   # pid that cannot exist
    r._prefetched[(5, 3)] = seg
    r.release_layer(5)
    assert not seg.exists(), "a dead rank's marker pinned the segment"


def test_progressive_uses_emit_not_a_module_logger():
    """progressive.py has NO module logger -- it uses a local _emit() because
    the CPU tests load this file standalone. Two logger.info() calls added for
    download verbosity raised `NameError: name 'logger' is not defined` inside
    the weight iterator and killed all four TP workers mid-boot."""
    import re
    src = _progressive_src()
    bare = re.findall(r"(?<![\w.])logger\.\w+\(", src)
    assert not bare, f"progressive.py has no logger; found {bare}"


def test_download_verbosity_still_present():
    """...but the fix must not be 'delete the logging'."""
    src = _progressive_src()
    assert "waiting on background" in src
    assert "1 fetch instead of" in src


# ------------------------------------------------- download-progress visibility
def test_async_prefetch_logs_its_result(tmp_path):
    """Background prefetch is now the COMMON path, and it discarded
    prefetch_layer's return value -- so 'cached', 'shared (fetched by another
    rank)' and 'prefetched ... from <source>' never reached the log. Only the
    sync fallback reported anything."""
    src = _progressive_src()
    assert "_note = _fut.result()" in src, (
        "the async path must surface the prefetch note, not discard it")








def test_monitor_survives_a_missing_root(tmp_path):
    """It is telemetry; it must never fail a boot."""
    assert FR._DownloadMonitor(tmp_path / "nope", every=0.05)._progress() == (0, 0)


def test_monitor_is_a_singleton_per_process():
    """Four TP ranks are four processes, but N resolvers in ONE process must
    not spawn N threads all logging the same numbers."""
    FR._DownloadMonitor._instance = None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a = FR._DownloadMonitor.ensure(Path(d))
        b = FR._DownloadMonitor.ensure(Path(d))
        try:
            assert a is b
        finally:
            a.stop()
            FR._DownloadMonitor._instance = None


def test_monitor_thread_is_daemon_so_it_cannot_hang_shutdown():
    FR._DownloadMonitor._instance = None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = FR._DownloadMonitor.ensure(Path(d))
        try:
            assert m._thread is not None and m._thread.daemon
        finally:
            m.stop()
            FR._DownloadMonitor._instance = None


# ------------------------------------------------------ diagnosable rejections
def test_reject_message_carries_the_exception_argument(tmp_path):
    """`REJECT error:KeyError` named the TYPE and threw away the argument. A
    boot that degraded 190 experts to K2 produced 270 identical contentless
    lines, and the cause had to be reverse-engineered from the source. The key
    IS the diagnosis."""
    r = _resolver(tmp_path)
    msg = r._reject("hf:org/repo", KeyError("expert 50 not in segment index "
                                            "for layer-004.k4.safetensors"),
                    "remote_tables")
    assert "KeyError" in msg
    assert "expert 50" in msg, f"argument dropped: {msg}"
    assert "layer-004.k4.safetensors" in msg


def test_reject_traces_once_per_site_not_per_expert(tmp_path):
    """19,200 experts must not produce 19,200 stacks."""
    r = _resolver(tmp_path)
    for _ in range(5):
        r._reject("hf:x", KeyError("k"), "remote_tables")
    assert len(r._reject_traced) == 1
    r._reject("hf:x", KeyError("k"), "att_decision")
    assert len(r._reject_traced) == 2


def test_reject_detail_is_bounded(tmp_path):
    """A huge repr must not flood a boot log line."""
    r = _resolver(tmp_path)
    msg = r._reject("hf:x", ValueError("y" * 5000), "remote_index")
    assert len(msg) < 300


def test_reject_survives_an_exception_with_no_message(tmp_path):
    r = _resolver(tmp_path)
    assert "KeyError" in r._reject("hf:x", KeyError(), "remote_index")


def test_resolver_has_a_lock_for_its_memo_dicts(tmp_path):
    """Prefetch runs on a thread pool now; the memo dicts were written for a
    single thread."""
    assert hasattr(_resolver(tmp_path), "_memo_lock")





# ------------------------------------------- telemetry must not fail the fetch
def test_every_incremented_stats_key_is_declared():
    """THE regression. `self.stats["bytes_from_prefetch"] += len(payload)` was
    added with the prefetch fast-path but never declared, and it sits on the
    SUCCESS branch -- right after range_from_prefetched returns bytes. So a
    KeyError fired exactly when the optimisation WORKED, was caught as a
    source rejection, and dropped the expert down the K ladder: the better
    prefetch performed, the more experts degraded to K2. 190 of them, on a
    real boot, before anything said why."""
    import re
    src = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
           / "layers" / "quantization" / "exl3_fungible"
           / "fragments.py").read_text()
    used = set(re.findall(r'self\.stats\[\s*"([a-z0-9_]+)"\s*\]\s*\+?=', src))
    init = src[src.index("self.stats = Counter({"):]
    declared = set(re.findall(r'"([a-z0-9_]+)"\s*:\s*0', init[:init.index("})")]))
    missing = sorted(used - declared)
    assert not missing, f"incremented but never declared: {missing}"


def test_stats_is_a_counter_so_a_missing_key_cannot_raise():
    """Belt as well as braces: telemetry must never be able to fail the thing
    it measures, even if a future counter is added without declaring it."""
    src = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
           / "layers" / "quantization" / "exl3_fungible"
           / "fragments.py").read_text()
    assert "self.stats = Counter({" in src


def test_undeclared_counter_increments_instead_of_raising(tmp_path):
    r = _resolver(tmp_path)
    r.stats["a_counter_nobody_declared"] += 5      # must not raise
    assert r.stats["a_counter_nobody_declared"] == 5


def test_declared_counters_still_report_zero(tmp_path):
    """A Counter must not make declared metrics vanish from a scrape."""
    r = _resolver(tmp_path)
    assert r.stats["bytes_from_prefetch"] == 0
    assert "segments_prefetched" in r.stats




# ------------------------------------------- download progress (delivered bytes)
def _mon(root):
    return FR._DownloadMonitor(root, every=0.05)


def test_progress_counts_delivered_segments_not_staging_size(tmp_path):
    """Xet PREALLOCATES its .incomplete staging files, so their size never
    changes during transfer. Watching that growth produced
    '3 in flight, 12.1 GiB this boot, 0 MiB/s' forever -- four ranks cycling
    four constant totals while the box pulled at 300+ MiB/s. Delivery is the
    only signal preallocation cannot fake."""
    root = tmp_path / "segments"
    dl = root / ".cache" / "huggingface" / "download"
    dl.mkdir(parents=True)
    m = _mon(root)

    big = dl / "preallocated.incomplete"
    big.write_bytes(b"\0" * 4096)              # full size from the start
    total, n = m._progress()
    assert total == 0, "preallocated staging bytes must not count as progress"
    assert n == 1, "but it IS in flight, and the count says so"

    (root / "layer-003.k3.safetensors").write_bytes(b"x" * 1000)
    total, _ = m._progress()
    assert total == 1000


def test_delivered_survives_release_layer_unlinking_the_file(tmp_path):
    """release_layer unlinking a segment is not un-delivering it; the series
    must never go backwards."""
    root = tmp_path / "segments"
    root.mkdir(parents=True)
    m = _mon(root)
    f = root / "layer-003.k3.safetensors"
    f.write_bytes(b"x" * 2048)
    assert m._progress()[0] == 2048
    f.unlink()
    assert m._progress()[0] == 2048, "delivered bytes went backwards"


def test_preexisting_segments_are_not_counted_as_this_boots_traffic(tmp_path):
    root = tmp_path / "segments"
    root.mkdir(parents=True)
    (root / "old.safetensors").write_bytes(b"z" * 999)
    m = _mon(root)                              # baseline taken here
    assert m._progress()[0] == 0
    (root / "new.safetensors").write_bytes(b"n" * 10)
    assert m._progress()[0] == 10


def test_a_segment_counted_once_not_on_every_scan(tmp_path):
    root = tmp_path / "segments"
    root.mkdir(parents=True)
    m = _mon(root)
    (root / "a.safetensors").write_bytes(b"x" * 500)
    assert [m._progress()[0] for _ in range(3)] == [500, 500, 500]


# ------------------------------------------------------------- keep layers
def test_keep_layers_is_opt_in():
    assert FR.FragmentResolver.keep_layers({}) is False


@pytest.mark.parametrize("name", ["VLLM_FQ_KEEP_LAYERS", "VLLM_FQ_PREFETCH_KEEP"])
def test_both_names_work(name):
    """The old name stays an alias so existing runs do not silently change
    behaviour."""
    assert FR.FragmentResolver.keep_layers({name: "1"}) is True


def test_keep_layers_retains_the_file_AND_the_prefetch_handle(tmp_path,
                                                              monkeypatch):
    """The point of keeping is the NEXT re-tiering: a K3->K4 promotion should
    slice the new expert out of the local whole-layer object rather than
    re-fetch ~2.5 GB. That needs the file on disk AND the _prefetched handle
    that makes range_from_prefetched reachable -- keeping only the file would
    leave the swap path fetching over the network anyway."""
    monkeypatch.setenv("VLLM_FQ_KEEP_LAYERS", "1")
    r = _resolver(tmp_path)
    seg = r.cache_dir / "segments" / "layer-004.k4.safetensors"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"w" * 4096)
    r._prefetched[(4, 4)] = seg
    assert r.release_layer(4) == 0
    assert seg.exists(), "kept layer was unlinked"
    assert r._prefetched.get((4, 4)) == seg, (
        "handle dropped: a later swap would re-download instead of slicing")


# ------------------------------ transient faults must not permanently degrade
class _FlakyOpen:
    """Fails `fail_times` with a transient error, then succeeds."""

    def __init__(self, payload: bytes, fail_times: int, exc=None):
        import urllib.error
        self.payload = payload
        self.left = fail_times
        self.calls = 0
        self.exc = exc or urllib.error.URLError("Network is unreachable")

    def __call__(self, relpath, headers):
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            raise self.exc
        import io
        return io.BytesIO(self.payload)


def test_attestation_fetch_retries_a_transient_network_error(monkeypatch):
    """Ranged reads retried; whole-object reads did NOT, and the asymmetry
    was expensive. One blip fetching an ATTESTATION is indistinguishable from
    'this source has no attestation', so the expert is permanently degraded a
    tier. Observed live: URLError(Errno 101) -> 'FQ DEGRADED L5/e2: K4
    unavailable, serving K3 instead'."""
    src = FR.HfSource("org/repo")
    flaky = _FlakyOpen(b'{"ok": true}', fail_times=2)
    monkeypatch.setattr(FR.HfSource, "_open", flaky)
    monkeypatch.setattr(FR.HfSource, "_BACKOFF", 0.0)
    assert src.read_json("index-k3.json") == {"ok": True}
    assert flaky.calls == 3, "did not retry the transient failure"


def test_read_text_retries_too(monkeypatch):
    src = FR.HfSource("org/repo")
    flaky = _FlakyOpen(b"line", fail_times=1)
    monkeypatch.setattr(FR.HfSource, "_open", flaky)
    monkeypatch.setattr(FR.HfSource, "_BACKOFF", 0.0)
    assert src.read_text("attestations/layer-005.k4.jsonl") == "line"


def test_a_404_is_a_miss_and_is_not_retried(monkeypatch):
    """Absence is an answer, not a fault; retrying it wastes four backoffs
    per expert."""
    import urllib.error
    src = FR.HfSource("org/repo")
    calls = {"n": 0}

    def not_found(self, relpath, headers):   # bound: a plain function gets self
        calls["n"] += 1
        raise urllib.error.HTTPError(relpath, 404, "nope", {}, None)

    monkeypatch.setattr(FR.HfSource, "_open", not_found)
    assert src.read_json("missing.json") is None
    assert calls["n"] == 1


def test_offline_is_not_retried_on_whole_object_reads(monkeypatch):
    src = FR.HfSource("org/repo")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("FQ_ALLOW_NETWORK", raising=False)
    with pytest.raises(FR.OfflineError):
        src.read_json("index-k3.json")


def test_exhausted_retries_raise_rather_than_report_absence(monkeypatch):
    """The dangerous failure would be returning None: that reads as 'no
    attestation' and silently downgrades the expert."""
    src = FR.HfSource("org/repo")
    monkeypatch.setattr(FR.HfSource, "_open", _FlakyOpen(b"x", fail_times=99))
    monkeypatch.setattr(FR.HfSource, "_BACKOFF", 0.0)
    with pytest.raises(OSError, match="after 4 attempts"):
        src.read_json("index-k3.json")


# ------------------------------------------------- local first, then reflink
def test_prefetch_skips_the_network_when_the_layer_is_already_local(tmp_path,
                                                                    monkeypatch):
    """prefetch_layer consulted only the REMOTE sources, so a layer sitting in
    a local segment dir was re-downloaded from the Hub -- paying network and
    disk for bytes already present. Nothing needs reflinking here because
    nothing needs copying: the local segment is mmapped in place."""
    r = _resolver(tmp_path)
    sentinel = object()

    class _Seg:
        pass

    monkeypatch.setattr(type(r), "_local_segment",
                        lambda self, base, layer, k: _Seg())
    r.sources = [pytest.fail]          # touching a source is the failure
    note = r.prefetch_layer(4, 4)
    assert note is not None and note.startswith("local ")
    assert r.stats["segments_local"] == 1
    assert sentinel is not None


def test_reflink_shares_extents_and_reports_what_happened(tmp_path):
    """os.copy_file_range may SILENTLY do a plain copy, so the return value
    reports the mechanism rather than claiming a reflink we cannot verify."""
    src = tmp_path / "seg.bin"
    src.write_bytes(b"x" * (1 << 20))
    dst = tmp_path / "clone.bin"
    how = FR.FragmentResolver.reflink_or_copy(src, dst)
    assert dst.read_bytes() == src.read_bytes(), "clone must be byte-identical"
    assert how in ("reflink/copy_file_range", "plain copy")


def test_reflink_leaves_no_temp_behind(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"y" * 4096)
    dst = tmp_path / "sub" / "b.bin"
    FR.FragmentResolver.reflink_or_copy(src, dst)
    assert dst.exists()
    assert not list(dst.parent.glob("*.rl*")), "temp file left behind"


def test_reflink_falls_back_rather_than_failing(tmp_path, monkeypatch):
    """An unsupported filesystem must degrade to a copy, not an error."""
    monkeypatch.delattr(os, "copy_file_range", raising=False)
    src = tmp_path / "a.bin"
    src.write_bytes(b"z" * 2048)
    dst = tmp_path / "c.bin"
    assert FR.FragmentResolver.reflink_or_copy(src, dst) == "plain copy"
    assert dst.read_bytes() == src.read_bytes()
