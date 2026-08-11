# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the activation-matrix endpoint (exl3_fungible/heatmap.py).

Covers the contract in ``runs/m5-serve/heatmap/ENDPOINT-SPEC.md`` §13:
shape, the cross-rank merge arithmetic (which must NOT sum — see
``heatmap.MERGE_RULE``), encoding round-trips and byte budgets, the ETag
revalidation path, single-flight, every degraded path (FQ off, collector
absent, rank failure, RPC timeout, aliased mass, DP), the three reset
scopes, and that the whole surface is off by default.

No GPU and no built ``vllm._C``. Fixtures are the REAL
``VLLM_FQ_DUMP_STATS`` dumps from ``results/k3-fq/``; the router is
driven with FastAPI's TestClient against a fake engine client that
executes the REAL ``worker_sample`` on fake workers, so the worker half
of the round trip is exercised too rather than mocked away.
"""
import asyncio
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        heatmap as H,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        policy as P,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
except ImportError:  # standalone: load by path with a stub package
    import importlib.util
    import sys

    _dir = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")
    _pkg_name = "vllm.model_executor.layers.quantization.exl3_fungible"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _pkg = sys.modules.get(_pkg_name)
    if _pkg is None or not hasattr(_pkg, "heatmap"):
        _pkg = _pkg or types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_dir)]
        sys.modules[_pkg_name] = _pkg
        for _sub in ("policy", "stats", "store", "decision_log",
                     "occupancy_table", "swap", "loop", "integration",
                     "admin", "heatmap"):
            if not hasattr(_pkg, _sub):
                setattr(_pkg, _sub,
                        _load(f"{_pkg_name}.{_sub}", _dir / f"{_sub}.py"))
    H, P = _pkg.heatmap, _pkg.policy
    FqStatsCollector = _pkg.stats.FqStatsCollector

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

DUMPS = Path("/home/mbelleau/protensors-work/vllm-voipmonitor/research"
             "/fungible-quant/runs/m5-serve/results/k3-fq")
ARCHIVED = ("stats.jsonl", "stats-code-axis.jsonl", "stats-synthetic.jsonl",
            "stats-INVALID-truncated-corpus.jsonl")


# ------------------------------------------------------------------ fixtures


def load_record(name: str = "stats.jsonl", index: int = 9) -> dict:
    """One line of a real dump.

    ``mass_is_real`` is ABSENT from every archived file (``loop.py:933``
    landed after they were taken), so it defaults to False here — and is
    never inferred from ``count == mass``, which a uniform router
    produces legitimately (``stats.py:302-318``).
    """
    path = DUMPS / name
    if not path.exists():
        pytest.skip(f"archived dump {path} not present")
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i == index:
                rec = json.loads(line)
                rec.setdefault("mass_is_real", False)
                return rec
    pytest.skip(f"{path} has no record {index}")


class FakeRouter:
    capture_fn = None
    capture_fn_wants_weights = False

    def set_capture_fn(self, fn):
        self.capture_fn = fn


class DumpCollector:
    """The collector read-surface ``heatmap.py`` uses, backed by a dump.

    Deliberately not a subclass: the point is to pin down exactly which
    attributes the endpoint depends on, so a change in ``stats.py``
    breaks a test here rather than the live endpoint.
    """

    def __init__(self, count, mass, *, layers, mass_is_real=False,
                 window_len=64, window_stride=32, decay=0.95,
                 rolled=596, win_pos=20, step=19100):
        count = np.asarray(count, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        self.num_experts = int(count.shape[1])
        self.window_len = window_len
        self.window_stride = window_stride
        self.decay = decay
        self._windows_rolled = rolled
        self._win_pos = win_pos
        self._step = step
        self._mass_real = bool(mass_is_real)
        self._rows = {}
        self.count_buf = {}
        self.mass_buf = {}
        self._count_win = {}
        self._mass_win = {}
        for row, lid in enumerate(layers):
            self._rows[int(lid)] = (torch.tensor(count[row]),
                                    torch.tensor(mass[row]))
            self.count_buf[int(lid)] = torch.zeros(self.num_experts + 1)
            self.mass_buf[int(lid)] = torch.zeros(self.num_experts + 1)
            self._count_win[int(lid)] = torch.zeros(
                (window_len, self.num_experts), dtype=torch.int64)
            self._mass_win[int(lid)] = torch.zeros(
                (window_len, self.num_experts), dtype=torch.float32)

    def mass_is_real(self, layer_id=None):
        return self._mass_real

    def decayed(self, layer_id):
        c, m = self._rows[int(layer_id)]
        return c, (m if self._mass_real else c)


class FakeLoopState:
    """The loop-state read-surface: layers, tier_of, clocks, policy sha."""

    def __init__(self, collector, layers, tier_of, *, rank=0,
                 policy_sha="9c1f4d0b7a2e5c31", apply_mode="atomic"):
        self.collector = collector
        self.layers = [int(x) for x in layers]
        self.tier_of = np.asarray(tier_of, dtype=np.int64)
        self.num_experts = collector.num_experts
        self.rank = rank
        self.policy_sha = policy_sha
        self.cfg = types.SimpleNamespace(apply_mode=apply_mode)
        self._collector_layer_map = {lid: lid for lid in self.layers}
        self._step = collector._step
        self._real_steps = collector._step - 6
        self._intervals_run = 191


class FakeWorker:
    def __init__(self, state, rank=0):
        self.model_runner = types.SimpleNamespace(fq_collector=state)
        self.rank = rank


class FakeEngineClient:
    """Runs the REAL ``worker_sample`` on N fake workers."""

    def __init__(self, workers, *, rpc_error=None, override=None, delay=0.0):
        self.workers = list(workers)
        self.calls = []
        self.rpc_error = rpc_error
        self.override = override
        self.delay = delay

    async def collective_rpc(self, method, timeout=None, args=(),
                             kwargs=None):
        self.calls.append((method, args, timeout))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.rpc_error is not None:
            raise self.rpc_error
        if self.override is not None:
            return list(self.override)
        payload = args[0] if args else "{}"
        return [H.worker_sample(w, payload) for w in self.workers]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Both gates ON in ``os.environ`` (the WORKER reads that), module
    poison/accumulator cleared. The router is always given an explicit
    ``environ=`` so the gate truth table stays independent of this."""
    monkeypatch.setenv(H.DEV_MODE_ENV, "1")
    monkeypatch.setenv(H.HEATMAP_API_ENV, "1")
    for env in (H.HEATMAP_TOKEN_ENV, H.HEATMAP_ALLOW_COLLECTOR_ZERO_ENV,
                H.HEATMAP_API_ALIAS_ENV):
        monkeypatch.delenv(env, raising=False)
    H.reset_state()
    yield
    H.reset_state()


def enabled_env(**extra):
    env = {H.DEV_MODE_ENV: "1", H.HEATMAP_API_ENV: "1"}
    env.update(extra)
    return env


def build_state(rec=None, *, ranks=1, mass_is_real=False, layers=None,
                tier=None, loop=True, rank_perturb=None):
    """``(workers, states)`` for ``ranks`` TP ranks over one dump record."""
    rec = rec or load_record()
    count = np.asarray(rec["count"], dtype=np.float64)
    mass = np.asarray(rec["mass"], dtype=np.float64)
    lids = [int(x) for x in (layers if layers is not None else rec["layers"])]
    if layers is not None:
        keep = [rec["layers"].index(x) for x in lids]
        count, mass = count[keep], mass[keep]
    tier_of = (np.asarray(tier) if tier is not None
               else np.asarray(rec["tier_of"], dtype=np.int64)[
                   [rec["layers"].index(x) for x in lids]])
    workers = []
    for r in range(ranks):
        c = count.copy()
        if rank_perturb is not None and r in rank_perturb:
            c = c + rank_perturb[r]
        coll = DumpCollector(c, mass, layers=lids, mass_is_real=mass_is_real,
                             step=int(rec.get("step") or 19100))
        state = (FakeLoopState(coll, lids, tier_of, rank=r) if loop else coll)
        workers.append(FakeWorker(state, rank=r))
    return workers


def make_client(env, workers=None, *, engine=None, dp=1):
    app = fastapi.FastAPI()
    app.state.engine_client = engine or FakeEngineClient(
        workers if workers is not None else build_state())
    if dp != 1:
        app.state.vllm_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(data_parallel_size=dp))
    router = None
    if H.heatmap_enabled(env):
        router = H.build_router(environ=env)
        app.include_router(router)
    attached = H.attach_router(fastapi.FastAPI(), environ=env)
    return TestClient(app), app.state.engine_client, router, attached


def get_json(client, url, **kw):
    resp = client.get(url, **kw)
    return resp, (resp.json() if resp.content else None)


# --------------------------------------------------------------- shape (§13)


def test_shape_75x256_and_every_array_decodes_to_cells():
    client, engine, _, _ = make_client(enabled_env(), build_state(ranks=4))
    resp, body = get_json(client, "/fq/heatmap?include=live")
    assert resp.status_code == 200, resp.text
    assert body["layers"] == list(range(3, 78))
    assert body["num_layers"] == 75
    assert body["num_experts"] == 256
    assert body["cells"] == 19200
    assert body["cells"] == body["num_layers"] * body["num_experts"]
    assert H.decode_floats(body["count"], "bf16", 19200).size == 19200
    assert H.decode_tier(body["tier"], 19200).size == 19200
    assert H.decode_floats(body["live_count"], "bf16", 19200).size == 19200
    # layer 78 (MTP) is not instrumented and must never appear
    assert 78 not in body["layers"]


def test_client_side_validator_rejects_a_short_array():
    short = H.encode_floats(np.zeros(19199, dtype=np.float32), "bf16")
    with pytest.raises(ValueError, match="19199 elements, expected 19200"):
        H.decode_floats(short, "bf16", 19200)


def test_server_refuses_to_serve_a_short_array():
    """A truncated blob must never be rendered as a shifted heatmap."""
    workers = build_state(ranks=1)
    good = json.loads(H.worker_sample(workers[0], '{"op":"sample"}'))
    good["count"] = H.encode_floats(np.zeros(19199, dtype=np.float32), "bf16")
    engine = FakeEngineClient([], override=[json.dumps(good)])
    client, _, _, _ = make_client(enabled_env(), engine=engine)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 503
    assert body["error"]["code"] == "fq_heatmap_worker_error"
    assert "19199 cells, expected 19200" in body["error"]["message"]


def test_layer_ids_are_data_not_row_plus_three():
    workers = build_state(layers=[5, 9, 40])
    client, _, _, _ = make_client(enabled_env(), workers)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 200, resp.text
    assert body["layers"] == [5, 9, 40]
    assert body["num_layers"] == 3
    assert body["cells"] == 3 * 256
    assert H.decode_floats(body["count"], "bf16", 3 * 256).size == 768


def test_layers_subset():
    client, _, _, _ = make_client(enabled_env(), build_state())
    resp, body = get_json(client, "/fq/heatmap?layers=20-22,40")
    assert resp.status_code == 200, resp.text
    assert body["layers"] == [20, 21, 22, 40]
    assert body["cells"] == 4 * 256


def test_layers_selector_errors():
    client, _, _, _ = make_client(enabled_env(), build_state())
    assert client.get("/fq/heatmap?layers=nope").status_code == 400
    resp, body = get_json(client, "/fq/heatmap?layers=999")
    assert resp.status_code == 400
    assert body["error"]["code"] == "bad_layers"


# --------------------------------------------------------------- mass (§3.4)


def test_mass_is_real_comes_from_the_flag_not_from_array_equality():
    """A uniform router makes ``count == mass`` legitimately."""
    rec = load_record()
    uniform = np.full((75, 256), 7.0)
    rec = dict(rec, count=uniform.tolist(), mass=uniform.tolist())
    workers = build_state(rec, mass_is_real=True)
    coll = workers[0].model_runner.fq_collector.collector
    c, m = coll.decayed(3)
    assert torch.equal(c, m), "fixture must have count == mass"
    client, _, _, _ = make_client(enabled_env(), workers)
    resp, body = get_json(client, "/fq/heatmap")
    assert body["mass_is_real"] is True
    assert body["mass"] is not None, "real mass must ship even when equal"
    assert not any("aliased" in w for w in body["warnings"])


def test_aliased_mass_is_omitted_with_a_warning_and_halves_the_bytes():
    real = build_state(mass_is_real=True)
    aliased = build_state(mass_is_real=False)
    c1, _, _, _ = make_client(enabled_env(), real)
    c2, _, _, _ = make_client(enabled_env(), aliased)
    _, big = get_json(c1, "/fq/heatmap")
    _, small = get_json(c2, "/fq/heatmap")
    assert big["mass"] is not None and small["mass"] is None
    assert small["mass_is_real"] is False
    assert any("VLLM_FQ_GATE_MASS=1" in w for w in small["warnings"])
    n_big = len(json.dumps(big).encode())
    n_small = len(json.dumps(small).encode())
    # the omitted array is one of two 51,200-char blobs
    assert n_small < n_big * 0.72, (n_small, n_big)


def test_archived_dumps_default_mass_is_real_to_false():
    for name in ARCHIVED:
        path = DUMPS / name
        if not path.exists():
            pytest.skip(f"{path} missing")
        with path.open() as fh:
            raw = json.loads(fh.readline())
        assert "mass_is_real" not in raw, name
        assert load_record(name, 0)["mass_is_real"] is False


# ------------------------------------------------------------ encoding (§5)


def test_bf16_roundtrip_on_the_real_record():
    rec = load_record()
    count = np.asarray(rec["count"], dtype=np.float64).reshape(-1)
    back = H.decode_floats(H.encode_floats(count, "bf16"), "bf16",
                           count.size).astype(np.float64)
    rel = np.abs(back - count) / np.maximum(np.abs(count), 1e-30)
    assert rel.max() < 0.005, rel.max()
    # round-to-nearest-even, measured 0.38898 % on this record
    assert 0.0030 < rel.max() < 0.0039, rel.max()
    f32 = H.decode_floats(H.encode_floats(count, "f32"), "f32",
                          count.size).astype(np.float64)
    relf = np.abs(f32 - count) / np.maximum(np.abs(count), 1e-30)
    assert relf.max() < 1e-6, relf.max()


def test_bf16_matches_the_spec_worked_example():
    given = np.array([27956.66712900402, 14229.637078354925,
                      10076.517826289375, 35942.17890498233,
                      20928.839368426034, 19015.02907982981])
    got = H.decode_floats(H.encode_floats(given, "bf16"), "bf16", 6)
    assert got.tolist() == [27904.0, 14208.0, 10048.0, 35840.0, 20992.0,
                            19072.0]


def test_bf16_survives_the_full_dynamic_range_and_nan():
    vals = np.array([0.0, 0.0437, 1.0, 337651.33, np.inf, np.nan],
                    dtype=np.float64)
    got = H.decode_floats(H.encode_floats(vals, "bf16"), "bf16", 6)
    assert got[0] == 0.0
    assert abs(got[1] - 0.0437) / 0.0437 < 0.004
    assert abs(got[3] - 337651.33) / 337651.33 < 0.004
    assert np.isinf(got[4]) and np.isnan(got[5])


def test_tier_encoding_is_u8_and_domain_checked():
    blob = H.encode_tier(np.full(19200, 3, dtype=np.int64))
    assert len(H.gzip_bytes(blob.encode())) < 200
    assert (H.decode_tier(blob, 19200) == 3).all()
    with pytest.raises(ValueError, match="occupancy_table.TIERS"):
        H.encode_tier(np.array([3, 4, 9]))


def test_u32_roundtrip():
    vals = np.array([0, 1, 2**31 - 1, 4_000_000_000], dtype=np.int64)
    assert H.decode_u32(H.encode_u32(vals), 4).tolist() == vals.tolist()


def test_byte_budget_regression_guards():
    """§5.3's numbers, as regression guards on the real record."""
    client, _, _, _ = make_client(enabled_env(), build_state(ranks=4))
    _, default = get_json(client, "/fq/heatmap")
    body = json.dumps(default, separators=(",", ":")).encode()
    gz = H.gzip_bytes(body, 1)
    assert default["mass"] is None and default["count"] is not None
    assert len(gz) < 40_000, len(gz)

    live = build_state(ranks=4, mass_is_real=True)
    c2, _, _, _ = make_client(enabled_env(), live)
    _, full = get_json(c2, "/fq/heatmap?include=live,mass")
    gz_full = H.gzip_bytes(
        json.dumps(full, separators=(",", ":")).encode(), 1)
    assert full["mass"] is not None and full["live_count"] is not None
    assert len(gz_full) < 110_000, len(gz_full)

    # ...against ~382 KB for the naive float64 JSON of the same sample
    naive = json.dumps(load_record(), separators=(",", ":")).encode()
    assert len(gz) < len(H.gzip_bytes(naive, 1)) / 8


def test_response_is_gzipped_and_honours_accept_encoding():
    client, _, _, _ = make_client(enabled_env(), build_state())
    resp = client.get("/fq/heatmap", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("content-encoding") == "gzip"
    plain = client.get("/fq/heatmap?max_age_ms=0",
                       headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert any("did not offer gzip" in w for w in plain.json()["warnings"])


# ---------------------------------------------------------------- ETag (§6)


def test_etag_is_stable_between_window_rolls_and_304s():
    workers = build_state()
    client, engine, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "0"}), workers)
    first = client.get("/fq/heatmap?max_age_ms=0")
    etag = first.headers["etag"]
    assert etag.startswith('W/"fqhm-1-')
    second = client.get("/fq/heatmap?max_age_ms=0")
    assert second.headers["etag"] == etag, "piecewise constant between rolls"
    revalidate = client.get("/fq/heatmap?max_age_ms=0",
                            headers={"If-None-Match": etag})
    assert revalidate.status_code == 304
    assert revalidate.content == b""
    assert revalidate.headers["etag"] == etag
    assert "x-fq-step" in {k.lower() for k in revalidate.headers}
    # a roll changes it
    for w in workers:
        w.model_runner.fq_collector.collector._windows_rolled += 1
    assert client.get("/fq/heatmap?max_age_ms=0").headers["etag"] != etag


def test_include_live_defeats_the_etag_by_design():
    workers = build_state()
    client, _, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "0"}), workers)
    a = client.get("/fq/heatmap?include=live&max_age_ms=0").headers["etag"]
    for w in workers:
        w.model_runner.fq_collector.collector._step += 1
    b = client.get("/fq/heatmap?include=live&max_age_ms=0").headers["etag"]
    assert a != b


def test_etag_basis_separates_query_variants():
    kw = dict(rolled=1, win_pos=2, tier_digest="ab", mass_is_real=False,
              layers=[3, 4], num_experts=256, include=["mass"],
              precision="bf16", reduce="rank0")
    base = H.compute_etag(**kw)
    assert base == H.compute_etag(**kw)
    assert base != H.compute_etag(**{**kw, "precision": "f32"})
    assert base != H.compute_etag(**{**kw, "reduce": "all"})
    assert base != H.compute_etag(**{**kw, "layers": [3, 5]})
    assert base != H.compute_etag(**{**kw, "rolled": 2})
    assert base != H.compute_etag(**{**kw, "include": ["mass", "live"]})


# ------------------------------------------------------- cross-rank (§4.3)


def test_merge_rule_is_rank0_canonical_never_a_sum():
    """The 4x lie this test exists to prevent.

    Under TP the gate is replicated (integration.py:149-162), so every
    rank histograms the SAME topk_ids: the ranks are replicas, not a
    partition. Summing four identical replicas would report 4x the real
    routing traffic.
    """
    rec = load_record()
    truth = np.asarray(rec["count"], dtype=np.float64).reshape(-1)
    client, _, _, _ = make_client(enabled_env(), build_state(ranks=4))
    _, body = get_json(client, "/fq/heatmap")
    got = H.decode_floats(body["count"], "bf16", 19200).astype(np.float64)
    rel = np.abs(got - truth) / np.maximum(truth, 1e-30)
    assert rel.max() < 0.005, "merged value must equal ONE rank's value"
    assert np.abs(got - 4 * truth).min() > 0, "must not be the 4-rank sum"
    assert body["ranks"] == {"count": 4, "canonical": 0, "agree": True,
                             "reduce": "rank0",
                             "merge_rule": "rank0-canonical",
                             "digests": body["ranks"]["digests"]}
    assert H.MERGE_RULE == "rank0-canonical"
    assert len({d["digest"] for d in body["ranks"]["digests"]}) == 1


def test_rank_divergence_is_reported_not_fatal():
    workers = build_state(ranks=4, rank_perturb={2: 1000.0})
    client, _, _, _ = make_client(enabled_env(), workers)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 200, "divergence must not deny the picture"
    assert body["ranks"]["agree"] is False
    assert body["count"] is not None
    assert any("DISAGREE" in w and "'rank': 2" in w
               for w in body["warnings"]), body["warnings"]


def test_reduce_all_returns_every_ranks_arrays():
    client, _, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "0"}),
        build_state(ranks=4))
    _, body = get_json(client, "/fq/heatmap?reduce=all&max_age_ms=0")
    assert len(body["per_rank"]) == 4
    assert all(r["count"] for r in body["per_rank"])
    # ...and rank0 mode ships arrays exactly once
    _, lean = get_json(client, "/fq/heatmap?max_age_ms=0")
    assert "per_rank" not in lean


def test_non_canonical_ranks_ship_a_digest_only():
    workers = build_state(ranks=4)
    r2 = json.loads(H.worker_sample(workers[2], '{"op":"sample"}'))
    assert r2["ok"] is True and r2["canonical"] is False
    assert "count" not in r2 and "tier" not in r2
    assert len(r2["digest"]) == 16
    assert len(json.dumps(r2)) < 700


# --------------------------------------------------- degraded paths (§4.4)


def test_rank_failure_is_503_with_the_message_forwarded_nothing_cached():
    bad = json.dumps({"ok": False, "error": {
        "code": "internal_error", "http_status": 500,
        "message": "RuntimeError: boom on rank 1", "details": {}}})
    good = json.dumps({"ok": True, "rank": 0, "digest": "x"})
    engine = FakeEngineClient([], override=[good, bad])
    client, _, router, _ = make_client(enabled_env(), engine=engine)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 503
    assert body["error"]["code"] == "fq_heatmap_worker_error"
    assert "boom on rank 1" in body["error"]["message"]
    assert router.fq_heatmap_cache == {}, "a failure must not be cached"
    client.get("/fq/heatmap")
    assert len(engine.calls) == 2, "the next poll retries"


def test_timeout_poisons_the_surface_and_poison_scope_clears_it():
    engine = FakeEngineClient([], rpc_error=TimeoutError(
        "RPC call to fq_heatmap_sample timed out."))
    client, _, _, _ = make_client(enabled_env(), engine=engine)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 504
    assert body["error"]["code"] == "fq_heatmap_timeout"
    assert "desynchronised" in body["error"]["details"]["guidance"]

    calls = len(engine.calls)
    resp2, body2 = get_json(client, "/fq/heatmap")
    assert resp2.status_code == 503
    assert body2["error"]["code"] == "fq_heatmap_poisoned"
    assert len(engine.calls) == calls, "poisoned must issue NO rpc"

    cleared = client.post("/fq/heatmap/reset", json={"scope": "poison"})
    assert cleared.status_code == 200
    assert cleared.json()["was_poisoned"] is True
    assert H._POISON.poisoned is False
    assert len(engine.calls) == calls, "clearing poison issues no rpc"


def test_engine_unavailable_is_503():
    engine = FakeEngineClient([], rpc_error=RuntimeError(
        "EngineCore encountered an issue: Server shutting down"))
    client, _, _, _ = make_client(enabled_env(), engine=engine)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 503
    assert body["error"]["code"] == "engine_unavailable"


def test_fq_not_active_is_404_not_an_empty_matrix():
    worker = FakeWorker(None, rank=0)
    client, _, _, _ = make_client(enabled_env(), [worker])
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 404
    assert body["error"]["code"] == "fq_not_active"
    assert "VLLM_FQ_ENABLE" in body["error"]["message"]


def test_meta_reports_fq_active_false_instead_of_404():
    """A health check that 404s is useless as a health check."""
    client, _, _, _ = make_client(enabled_env(), [FakeWorker(None)])
    resp, body = get_json(client, "/fq/heatmap/meta")
    assert resp.status_code == 200
    assert body["fq_active"] is False
    assert body["reason"]["code"] == "fq_not_active"
    assert body["cells"] is None


def test_collector_only_mode_still_serves_counts():
    workers = build_state(ranks=1, loop=False)
    client, _, _, _ = make_client(enabled_env(), workers)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 200, resp.text
    assert body["count"] is not None
    assert body["tier"] is None
    assert body["interval"] is None
    assert body["policy_sha"] is None
    assert body["real_steps"] is None
    assert body["num_layers"] == 75


def test_dp_greater_than_one_is_501():
    client, engine, _, _ = make_client(enabled_env(), build_state(), dp=2)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 501
    assert body["error"]["code"] == "dp_not_supported"
    assert "core_client.py:1449-1458" in body["error"]["message"]
    assert engine.calls == []


def test_bad_query_parameters():
    client, _, _, _ = make_client(enabled_env(), build_state())
    for url, code in (("/fq/heatmap?include=bogus", "unknown_include"),
                      ("/fq/heatmap?precision=fp8", "bad_precision"),
                      ("/fq/heatmap?reduce=sum", "bad_reduce"),
                      ("/fq/heatmap?max_age_ms=soon", "bad_max_age")):
        resp, body = get_json(client, url)
        assert resp.status_code == 400, url
        assert body["error"]["code"] == code, url


# ---------------------------------------------------------- gating (§3.3)


def test_the_endpoint_is_off_by_default():
    for env in ({}, {H.DEV_MODE_ENV: "1"}, {H.HEATMAP_API_ENV: "1"},
                {H.DEV_MODE_ENV: "0", H.HEATMAP_API_ENV: "1"}):
        client, engine, router, attached = make_client(env, build_state())
        assert attached is False, env
        assert router is None, env
        assert client.get("/fq/heatmap").status_code == 404, env
        assert client.get("/fq/heatmap/meta").status_code == 404, env
        assert client.post("/fq/heatmap/reset",
                           json={"scope": "client"}).status_code == 404, env
        assert engine.calls == []


def test_both_gates_on_attaches():
    _, _, _, attached = make_client(enabled_env(), build_state())
    assert attached is True
    _, _, _, aliased = make_client(
        {H.DEV_MODE_ENV: "1", H.HEATMAP_API_ALIAS_ENV: "1"}, build_state())
    assert aliased is True


def test_token_gate():
    env = enabled_env(**{H.HEATMAP_TOKEN_ENV: "hunter2"})
    client, _, _, _ = make_client(env, build_state())
    assert client.get("/fq/heatmap").status_code == 403
    assert client.get("/fq/heatmap",
                      headers={"X-FQ-Heatmap-Token": "nope"}).status_code \
        == 403
    ok = client.get("/fq/heatmap",
                    headers={"X-FQ-Heatmap-Token": "hunter2"})
    assert ok.status_code == 200
    # A latin-1 byte in the header must be a 403, never a 500: Starlette
    # decodes headers as latin-1 and hmac.compare_digest refuses str
    # operands with non-ASCII code points.
    assert client.get(
        "/fq/heatmap",
        headers={"X-FQ-Heatmap-Token": b"caf\xe9"}).status_code == 403


def test_worker_reenforces_the_gate_against_the_collective_rpc_bypass(
        monkeypatch):
    """dev mode alone attaches POST /collective_rpc, which can call
    ``fq_heatmap_sample`` by name — the worker must refuse."""
    workers = build_state()
    monkeypatch.delenv(H.HEATMAP_API_ENV, raising=False)
    out = json.loads(H.worker_sample(workers[0], '{"op":"sample"}'))
    assert out["ok"] is False
    assert out["error"]["code"] == "fq_heatmap_disabled"
    assert out["error"]["http_status"] == 404


# --------------------------------------------------- single-flight (§4.4)


def test_ttl_cache_collapses_repeat_polls_to_one_rpc():
    client, engine, _, _ = make_client(enabled_env(), build_state())
    for _ in range(10):
        assert client.get("/fq/heatmap?max_age_ms=1000").status_code == 200
    assert len(engine.calls) == 1, engine.calls
    assert client.get("/fq/heatmap").json()["cached"] is True


def test_single_flight_lock_collapses_concurrent_polls():
    app = fastapi.FastAPI()
    engine = FakeEngineClient(build_state(), delay=0.02)
    app.state.engine_client = engine
    app.include_router(H.build_router(environ=enabled_env()))

    async def hammer():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as ac:
            out = await asyncio.gather(
                *[ac.get("/fq/heatmap?max_age_ms=1000") for _ in range(10)])
        return [r.status_code for r in out]

    assert asyncio.run(hammer()) == [200] * 10
    assert len(engine.calls) == 1, engine.calls


def test_max_age_zero_is_clamped_to_the_floor_and_says_so():
    client, engine, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "500"}), build_state())
    _, body = get_json(client, "/fq/heatmap?max_age_ms=0")
    assert any("clamped up to the" in w and "500 ms" in w
               for w in body["warnings"]), body["warnings"]
    for _ in range(5):
        client.get("/fq/heatmap?max_age_ms=0")
    assert len(engine.calls) == 1, "the floor caps the engine-stall rate"


# ------------------------------------------------------------- rates (§5.4)


def _drive(collector, layers, hits_per_step, steps):
    for _ in range(steps):
        for lid in layers:
            collector.count_buf[lid][:collector.num_experts] += hits_per_step
        collector.step()


def test_rate_from_the_decayed_window_matches_a_constant_rate_source():
    coll = FqStatsCollector(4, window_len=8, window_stride=4, decay=0.95,
                            device="cpu")
    coll.bind_router(0, FakeRouter())
    _drive(coll, [0], 5.0, 8 * 4)
    count, _ = coll.decayed(0)
    denom = H.rate_denominator(coll.window_stride, coll.decay,
                               min(coll._windows_rolled, coll.window_len))
    got = float(count[0]) / denom
    assert abs(got - 5.0) / 5.0 < 0.01, got


def test_steps_since_roll_is_zero_right_after_a_roll():
    coll = FqStatsCollector(4, window_len=8, window_stride=4, device="cpu")
    coll.bind_router(0, FakeRouter())
    _drive(coll, [0], 1.0, 8)
    meta = H._window_meta(coll)
    assert meta["steps_since_roll"] == 0
    assert meta["rolled"] == 2
    assert meta["n_effective"] == 2
    _drive(coll, [0], 1.0, 1)
    assert H._window_meta(coll)["steps_since_roll"] == 1


def test_window_metadata_shape():
    meta = H._window_meta(DumpCollector(np.ones((1, 4)), np.ones((1, 4)),
                                        layers=[3]))
    assert meta["len"] == 64 and meta["stride"] == 32
    assert meta["decay"] == 0.95 and meta["n_effective"] == 64
    assert abs(meta["horizon_steps"] - 640.0) < 1e-9
    assert abs(meta["rate_denominator"] - 615.9845509) < 1e-6


# -------------------------------------------------------- cumulative (§7.4)


def _cum_collector(e=4, window_len=4, stride=2):
    coll = FqStatsCollector(e, window_len=window_len, window_stride=stride,
                            decay=0.9, device="cpu")
    coll.bind_router(0, FakeRouter())
    return coll


def test_cumulative_is_the_exact_int64_ring_sum():
    coll = _cum_collector()
    acc = H._CumAccumulator()
    acc.integrate(coll, [0], [0], coll._windows_rolled, coll._win_pos,
                  coll._step)
    assert acc.cum.sum() == 0, "the first sample starts at zero"
    _drive(coll, [0], 3.0, 6)          # 3 rolls x (2 steps x 3 hits) = 18
    acc.integrate(coll, [0], [0], coll._windows_rolled, coll._win_pos,
                  coll._step)
    exact = coll._count_win[0].sum(0).numpy().astype(np.int64)
    assert acc.cum[0].tolist() == exact.tolist()
    assert acc.cum[0].tolist() == [18, 18, 18, 18]
    assert acc.lossy is False


def test_cumulative_reports_loss_when_the_ring_wrapped():
    coll = _cum_collector(window_len=4, stride=2)
    acc = H._CumAccumulator()
    acc.integrate(coll, [0], [0], 0, 0, 0)
    _drive(coll, [0], 1.0, 2 * 10)                # 10 rolls into a 4-slot ring
    acc.integrate(coll, [0], [0], coll._windows_rolled, coll._win_pos,
                  coll._step)
    assert acc.lossy is True
    assert acc.dropped_slots == 6
    assert acc.cum[0].tolist() == [8, 8, 8, 8]    # only the surviving 4 slots


def test_cumulative_auto_rebases_before_u32_would_wrap():
    coll = _cum_collector()
    acc = H._CumAccumulator()
    acc.integrate(coll, [0], [0], 0, 0, 0)
    acc.cum += (1 << 31) - 1
    _drive(coll, [0], 1.0, 2)
    acc.integrate(coll, [0], [0], coll._windows_rolled, coll._win_pos, 999)
    assert acc.overflow_rebased is True
    assert acc.cum.max() == 0
    assert acc.since_step == 999


def test_include_cum_ships_a_u32_array_and_the_rebase_step():
    workers = build_state()
    client, _, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "0"}), workers)
    _, body = get_json(client, "/fq/heatmap?include=cum&max_age_ms=0")
    assert body["cum_count"] is not None
    assert H.decode_u32(body["cum_count"], 19200).size == 19200
    assert body["cum_since_step"] == 19100
    assert body["encoding"]["cum_count"] == "u32"


# ------------------------------------------------------------- reset (§7)


def test_reset_client_changes_nothing_server_side_and_issues_no_rpc():
    workers = build_state(ranks=2)
    coll = workers[0].model_runner.fq_collector.collector
    client, engine, _, _ = make_client(enabled_env(), workers)
    client.get("/fq/heatmap")
    before = (coll._count_win[3].clone(), coll._windows_rolled, coll._win_pos,
              workers[0].model_runner.fq_collector.tier_of.copy(),
              workers[0].model_runner.fq_collector.policy_sha)
    calls = len(engine.calls)

    resp = client.post("/fq/heatmap/reset",
                       json={"scope": "client", "reason": "before coder"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["baseline_sample_id"] == 1
    assert body["baseline_step"] == 19100
    assert "change since mark" in body["note"]
    assert len(engine.calls) == calls, "scope=client must issue no rpc"
    assert torch.equal(coll._count_win[3], before[0])
    assert (coll._windows_rolled, coll._win_pos) == before[1:3]
    assert np.array_equal(
        workers[0].model_runner.fq_collector.tier_of, before[3])
    assert workers[0].model_runner.fq_collector.policy_sha == before[4]


def test_reset_heatmap_rebases_cum_on_all_ranks_without_touching_the_window():
    workers = build_state(ranks=4)
    coll = workers[0].model_runner.fq_collector.collector
    client, engine, _, _ = make_client(
        enabled_env(**{H.HEATMAP_MIN_PERIOD_MS_ENV: "0"}), workers)
    client.get("/fq/heatmap?include=cum&max_age_ms=0")
    win_before = coll._count_win[3].clone()

    resp = client.post("/fq/heatmap/reset", json={"scope": "heatmap"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True and body["ranks"] == 4
    assert body["cum_since_step"] == 19100
    assert "untouched" in body["note"]
    assert engine.calls[-1][0] == "fq_heatmap_sample"
    assert json.loads(engine.calls[-1][1][0])["op"] == "reset_cum"
    assert torch.equal(coll._count_win[3], win_before)
    assert coll._windows_rolled == 596

    _, after = get_json(client, "/fq/heatmap?include=cum&max_age_ms=0")
    assert H.decode_u32(after["cum_count"], 19200).sum() == 0


def test_reset_collector_is_refused_without_the_third_gate():
    client, engine, _, _ = make_client(enabled_env(), build_state())
    resp = client.post("/fq/heatmap/reset", json={"scope": "collector"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "collector_zero_not_allowed"
    assert engine.calls == []


def test_reset_collector_with_the_gate_zeroes_every_rank(monkeypatch):
    monkeypatch.setenv(H.HEATMAP_ALLOW_COLLECTOR_ZERO_ENV, "1")
    colls = []
    workers = []
    for r in range(2):
        coll = FqStatsCollector(4, window_len=4, window_stride=2,
                                device="cpu")
        coll.bind_router(0, FakeRouter())
        _drive(coll, [0], 3.0, 8)
        colls.append(coll)
        workers.append(FakeWorker(
            FakeLoopState(coll, [0], np.full((1, 4), 3)), rank=r))
    assert float(colls[0].decayed(0)[0].sum()) > 0

    env = enabled_env(**{H.HEATMAP_ALLOW_COLLECTOR_ZERO_ENV: "1"})
    client, _, _, _ = make_client(env, workers)
    resp = client.post("/fq/heatmap/reset",
                       json={"scope": "collector", "reason": "operator"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True and body["ranks"] == 2
    assert any("pin-forced" in w for w in body["warnings"])
    for coll in colls:
        assert coll._windows_rolled == 0 and coll._win_pos == 0
        assert float(coll.decayed(0)[0].sum()) == 0.0
        assert float(coll._count_win[0].sum()) == 0


def test_zeroed_stats_hold_unpinned_swaps_but_pin_forced_pairs_still_emit():
    """§7.5, pinned in a test so nobody "fixes" the warning away.

    With an all-zero window the hysteresis guard is ``0 > 1.25*0`` ->
    False, so free swaps are suppressed — but ``policy.py:142-144``
    waives hysteresis for a pin-forced pair, and with every score equal
    the partner is chosen by ``lexsort((arange(E), -0))``, i.e. by index
    order rather than by traffic.
    """
    E = 8
    zero = {"count": np.zeros((1, E)), "mass": np.zeros((1, E))}
    eps = {P.K3: np.full((1, E), 1.0), P.K4: np.zeros((1, E))}
    tier = np.full((1, E), P.K3)
    tier[0, 6] = tier[0, 7] = P.K4                # K4 held by the top ids
    cfg = {"n_k4": [2]}

    assert P.decide(zero, eps, tier, cfg=cfg) == [], "no unpinned swaps"

    pins = np.zeros((1, E), dtype=np.int64)
    pins[0, 3] = P.PIN_K4                         # operator pin
    forced = P.decide(zero, eps, tier, pins=pins, cfg=cfg)
    assert forced == [(0, 6, 3)], forced
    assert forced[0][1] == 6, "partner picked by index order, not traffic"


def test_reset_bad_scope_and_bad_json():
    client, _, _, _ = make_client(enabled_env(), build_state())
    assert client.post("/fq/heatmap/reset",
                       json={"scope": "nuke"}).status_code == 400
    resp = client.post("/fq/heatmap/reset", content=b"{not json",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_json"


# ---------------------------------------------------------- no tearing (§4.2)


class RollingCollector(DumpCollector):
    """Bumps ``_windows_rolled`` during the first read only."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads = 0

    def decayed(self, layer_id):
        self.reads += 1
        if self.reads == 1:
            self._windows_rolled += 1
        return super().decayed(layer_id)


def test_a_roll_during_the_read_forces_exactly_one_retry():
    rec = load_record()
    coll = RollingCollector(np.asarray(rec["count"])[:2],
                            np.asarray(rec["mass"])[:2], layers=[3, 4])
    state = FakeLoopState(coll, [3, 4], np.full((2, 256), 3))
    out = json.loads(H.worker_sample(FakeWorker(state), '{"op":"sample"}'))
    assert out["ok"] is True
    assert any(w == "torn read retried 1x" for w in out["warnings"]), \
        out["warnings"]
    # 2 layers x 2 attempts
    assert coll.reads == 4
    assert out["window"]["rolled"] == coll._windows_rolled


# ------------------------------------------------------------- meta (§3.1)


def test_meta_is_small_and_cached_after_the_first_rpc():
    client, engine, _, _ = make_client(enabled_env(), build_state(ranks=4))
    resp, body = get_json(client, "/fq/heatmap/meta")
    assert resp.status_code == 200
    assert body["fq_active"] is True
    assert body["layers"] == list(range(3, 78))
    assert body["cells"] == 19200
    assert body["merge_rule"] == "rank0-canonical"
    assert body["cached"] is False
    assert len(json.dumps(body).encode()) < 2000
    assert len(engine.calls) == 1
    _, again = get_json(client, "/fq/heatmap/meta")
    assert again["cached"] is True
    assert len(engine.calls) == 1, "meta costs one rpc, ever"


def test_meta_stays_available_while_the_sampling_path_is_poisoned():
    client, engine, _, _ = make_client(enabled_env(), build_state())
    client.get("/fq/heatmap/meta")
    H._POISON.poison("test")
    resp, body = get_json(client, "/fq/heatmap/meta")
    assert resp.status_code == 200
    assert body["poisoned"] is True
    assert client.get("/fq/heatmap").status_code == 503


# ------------------------------------------------------- fixture shapes


@pytest.mark.parametrize("name", ARCHIVED)
def test_every_archived_dump_parses_into_a_valid_sample(name):
    """Shape only — ``stats-INVALID-truncated-corpus.jsonl`` came from a
    broken replay, so its VALUES are meaningless but its shape is not."""
    rec = load_record(name, 0)
    assert sorted(rec) == ["count", "interval", "layers", "mass",
                           "mass_is_real", "step", "tier_of"]
    workers = build_state(rec)
    client, _, _, _ = make_client(enabled_env(), workers)
    resp, body = get_json(client, "/fq/heatmap")
    assert resp.status_code == 200, resp.text
    assert body["cells"] == len(rec["layers"]) * 256
    assert H.decode_floats(body["count"], "bf16", body["cells"]).size \
        == body["cells"]
    assert H.decode_tier(body["tier"], body["cells"]).size == body["cells"]
    assert body["mass_is_real"] is False


def test_headers_carry_the_sample_identity():
    client, _, _, _ = make_client(enabled_env(), build_state())
    resp = client.get("/fq/heatmap")
    assert resp.headers["x-fq-sample-id"] == "1"
    assert resp.headers["x-fq-step"] == "19100"
    assert resp.headers["x-fq-merge-rule"] == "rank0-canonical"
    assert resp.headers["cache-control"] == "no-cache, must-revalidate"
    body = resp.json()
    assert body["schema"] == "fq-heatmap/1"
    assert len(body["server_boot_id"]) == 8
    assert body["encoding"]["layout"] == "layer-major"
    assert body["encoding"]["byte_order"] == "little"
