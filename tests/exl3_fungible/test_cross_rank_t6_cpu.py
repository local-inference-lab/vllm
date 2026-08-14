# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T6 — cross-rank agreement (03-testing-validation.md).

Claim under test: **every rank of a TP group computes the same swap list and
the same decision digest without any collective.** The mechanism is not the
hash — the hash was only ever evidence. The mechanism is that the policy
domain is *logical* (D4: `fq-policy/2` documents are keyed by logical expert
id and `store.validate_policy` refuses any rank/world_size/tp/device field)
and `policy.decide` is a pure, total-order-sorted function of
(stats, eps, membership, pins, dwell, cfg).

Evidence design — why 4 processes and not 4 GPUs. A TP4 serve run would
exercise the same pure function four times with the same inputs; the GPUs
contribute nothing to the property, and their presence would *weaken* the
test (a shared parent process, shared module state and one shared RNG make
accidental agreement easy). So this harness runs four SIMULATED ranks as
independent spawned interpreters, each hostile to the property in a way a
real TP4 run is not:

* a different ``PYTHONHASHSEED`` — any dict/set iteration-order dependence
  inside decide()/explain() diverges (str hashing is randomized per process);
* a different global RNG seeding (``random`` + ``numpy.random``) — any
  accidental use of global randomness diverges;
* a different ``VLLM_FQ_RANK``/``LOCAL_RANK``/``CUDA_VISIBLE_DEVICES`` in the
  environment — any topology leak into the decision diverges;
* the interval inputs are RECONSTRUCTED independently inside each process
  from one shared integer seed, so input construction is proven deterministic
  too (nothing is pickled across, except the seed and the transcript back).

The 50 synthetic intervals are a *trajectory*: each interval's swaps are
applied before the next one is decided, so a single divergence at interval i
cascades through every later interval and hash.

Runnable directly: ``python test_cross_rank_t6_cpu.py``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_DIR = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
        / "layers" / "quantization" / "exl3_fungible")
_PKG = "vllm.model_executor.layers.quantization.exl3_fungible"


def _load_tree_modules():
    """Load policy/store/decision_log from the WORKING TREE by path.

    A stub package entry makes ``decision_log``'s absolute import resolve to
    the same tree copy without importing vllm (which would cost seconds in
    every spawned rank, for nothing: the policy domain has no torch in it).
    """
    if _PKG in sys.modules and getattr(sys.modules[_PKG], "_fq_t6_stub", False):
        pkg = sys.modules[_PKG]
        return pkg.policy, pkg.store, pkg.decision_log

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg = types.ModuleType(_PKG)
    pkg._fq_t6_stub = True
    prev = sys.modules.get(_PKG)
    sys.modules[_PKG] = pkg
    try:
        pkg.policy = load(f"{_PKG}.policy", _DIR / "policy.py")
        pkg.store = load(f"{_PKG}.store", _DIR / "store.py")
        pkg.decision_log = load("fq_t6_decision_log", _DIR / "decision_log.py")
    except BaseException:
        if prev is not None:
            sys.modules[_PKG] = prev
        raise
    return pkg.policy, pkg.store, pkg.decision_log


P, S, DL = _load_tree_modules()

# ---------------------------------------------------------------- scenario

L, E, N_K4 = 6, 24, 8
INTERVALS = 50
RANKS = 4
SEED = 20260810


def interval_inputs(seed: int, i: int, tier_of: np.ndarray) -> dict:
    """Interval ``i``'s synthetic observation, from ``seed`` alone.

    Varied across intervals: routing counts/mass (a drifting hot set, so the
    desired set actually moves), eps curves, dwell readiness, pins, and the
    guard knobs (hysteresis, per-layer/total caps).
    """
    rng = np.random.default_rng(seed * 1_000_003 + i)
    hot = (np.arange(E)[None, :] + 3 * i) % E < 6  # the hot set walks
    count = rng.integers(0, 400, size=(L, E)).astype(np.float64)
    count += hot * rng.integers(1000, 9000, size=(L, E))
    mass = rng.uniform(0.0, 4.0, size=(L, E)) + hot * rng.uniform(5.0, 40.0,
                                                                  size=(L, E))
    eps = {3: rng.uniform(0.02, 0.30, size=(L, E)),
           4: rng.uniform(0.00, 0.02, size=(L, E))}
    dwell = rng.integers(0, 12000, size=(L, E)).astype(np.int64)
    pins = np.zeros((L, E), dtype=np.int64)
    if i % 7 == 3:  # a pinned pair every few intervals
        pins[i % L, (2 * i) % E] = P.PIN_K4
        pins[i % L, (2 * i + 1) % E] = P.PIN_K3
    cfg = {
        "n_k4": N_K4,
        "hysteresis": 1.05 + 0.05 * (i % 5),
        "dwell_steps": 6000 if i % 3 else 0,
        "max_swaps_per_layer": 1 + (i % 2),
        "max_swaps_total": 4 + (i % 9),
    }
    return {"stats": {"count": count, "mass": mass}, "eps": eps,
            "tier_of": tier_of, "pins": pins, "dwell": dwell, "cfg": cfg}


def initial_membership(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tier = np.full((L, E), P.K3, dtype=np.int64)
    for l in range(L):
        tier[l, rng.permutation(E)[:N_K4]] = P.K4
    return tier


def decision_sha(swaps) -> str:
    """The digest loop.FungibleQuantState.decision_sha logs per rank."""
    return hashlib.sha256(
        json.dumps(swaps, separators=(",", ":")).encode()).hexdigest()


def rank_transcript(seed: int, intervals: int = INTERVALS,
                    stats_jitter: float = 0.0) -> list[dict]:
    """One rank's full decision trajectory: the thing that must agree.

    Per interval: the ordered swap list, the resulting membership, the
    fq-policy/2 document hash (store.policy_hash — what gets persisted and
    compared at boot) and the decision digest, plus the explainability
    record's own digest (decision records are rank-invariant too, or the
    audit trail would not be comparable across ranks).
    """
    tier_of = initial_membership(seed)
    out: list[dict] = []
    for i in range(intervals):
        inp = interval_inputs(seed, i, tier_of)
        if stats_jitter:
            # Saboteur rank: it observed its OWN shard's routing sample
            # instead of the logical global one — the exact bug T6 exists to
            # catch. (A uniform *scale* would not diverge: score is linear in
            # count and the hysteresis test is a ratio, so both are
            # scale-invariant. Per-expert noise is the real hazard.)
            noise = np.random.default_rng(90000 + i).uniform(
                1.0 - stats_jitter, 1.0 + stats_jitter, size=(L, E))
            inp["stats"]["count"] = inp["stats"]["count"] * noise
        swaps = P.decide(inp["stats"], inp["eps"], inp["tier_of"],
                         pins=inp["pins"], dwell=inp["dwell"], cfg=inp["cfg"])
        tier_of = P.apply_swaps(tier_of, swaps)
        doc = {
            "schema": S.POLICY_SCHEMA,
            "manifest": "t6-synthetic",
            "budget": {"n_k4_per_layer": {str(l): N_K4 for l in range(L)}},
            "bits_per_expert": {str(l): [int(b) for b in tier_of[l]]
                                for l in range(L)},
        }
        S.validate_policy(doc, num_experts=E)  # D4: topology-neutral
        record = DL.explain(inp["stats"], inp["eps"], inp["tier_of"], swaps,
                            pins=inp["pins"], dwell=inp["dwell"],
                            cfg=inp["cfg"], step=i * 3000)
        out.append({
            "interval": i,
            "swaps": [[int(l), int(a), int(b)] for l, a, b in swaps],
            "decision_sha": decision_sha(swaps),
            "policy_sha": S.policy_hash(doc),
            "record_sha": hashlib.sha256(
                DL.to_json(record).encode()).hexdigest(),
            "n_k4": [int(x) for x in P.n_k4_of(tier_of)],
        })
    return out


def transcript_sha(transcript: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(transcript, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------- rank processes


def _rank_worker(rank: int, seed: int, queue) -> None:
    """Runs in a FRESH interpreter (spawn). Hostile-to-agreement setup:
    per-rank global RNG seeding on top of the per-rank PYTHONHASHSEED and
    topology env the parent set before starting this process."""
    import random

    random.seed(rank * 7919)
    np.random.seed(rank + 1)  # legacy global RNG, deliberately per-rank
    transcript = rank_transcript(seed)
    queue.put((rank, sys.flags.hash_randomization,
               os.environ.get("PYTHONHASHSEED"), transcript))


def _spawn_ranks(seed: int, ranks: int = RANKS) -> dict[int, list[dict]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    saved = {k: os.environ.get(k) for k in
             ("PYTHONHASHSEED", "VLLM_FQ_RANK", "LOCAL_RANK", "RANK",
              "CUDA_VISIBLE_DEVICES")}
    try:
        for rank in range(ranks):
            # Inherited by the spawned interpreter at startup: PYTHONHASHSEED
            # only takes effect that way, which is exactly what we want.
            os.environ["PYTHONHASHSEED"] = str(1 + rank * 104729)
            os.environ["VLLM_FQ_RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["RANK"] = str(rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            p = ctx.Process(target=_rank_worker, args=(rank, seed, queue))
            p.start()
            procs.append(p)
        results = {}
        seeds_seen = {}
        for _ in procs:
            rank, randomized, hashseed, transcript = queue.get(timeout=300)
            results[rank] = transcript
            seeds_seen[rank] = (randomized, hashseed)
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"rank process exited {p.exitcode}"
        assert len({s for _, s in seeds_seen.values()}) == RANKS, (
            f"ranks did not get distinct PYTHONHASHSEEDs: {seeds_seen}")
        return results
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def rank_transcripts():
    return _spawn_ranks(SEED)


# ------------------------------------------------------------------- tests


def test_scenario_is_non_vacuous():
    """The trajectory must actually swap, vary, and stay budget-legal —
    otherwise 'all ranks agree' would be agreement on nothing."""
    transcript = rank_transcript(SEED)
    total = sum(len(step["swaps"]) for step in transcript)
    moving = sum(1 for step in transcript if step["swaps"])
    assert total >= 50, f"only {total} swaps over {INTERVALS} intervals"
    assert moving >= INTERVALS // 3, f"only {moving} intervals moved"
    assert len({step["decision_sha"] for step in transcript}) >= 10
    for step in transcript:  # budget invariant holds all along (D1)
        assert step["n_k4"] == [N_K4] * L


def test_t6_four_ranks_agree_bit_for_bit(rank_transcripts):
    """T6 proper: 4 independent processes, identical transcripts."""
    assert sorted(rank_transcripts) == list(range(RANKS))
    reference = rank_transcripts[0]
    ref_sha = transcript_sha(reference)
    for rank in range(1, RANKS):
        other = rank_transcripts[rank]
        assert len(other) == len(reference) == INTERVALS
        for a, b in zip(reference, other):
            assert a["swaps"] == b["swaps"], (
                f"rank {rank} diverged on interval {a['interval']}: "
                f"{a['swaps']} vs {b['swaps']}")
            assert a["decision_sha"] == b["decision_sha"]
            assert a["policy_sha"] == b["policy_sha"]
            assert a["record_sha"] == b["record_sha"]
        assert transcript_sha(other) == ref_sha


def test_t6_digest_detects_divergence(rank_transcripts):
    """The evidence is not vacuous: a rank fed its own shard-local routing
    sample (±5% per-expert noise) must be caught by the digest — so equal
    digests really do mean equal decisions."""
    sabotaged = rank_transcript(SEED, stats_jitter=0.05)
    reference = rank_transcripts[0]
    assert transcript_sha(sabotaged) != transcript_sha(reference)
    diverged = [a["interval"] for a, b in zip(reference, sabotaged)
                if a["decision_sha"] != b["decision_sha"]]
    assert diverged, "divergent decisions produced identical digests"
    assert len(diverged) >= INTERVALS // 4, (
        f"only {len(diverged)}/{INTERVALS} intervals caught the divergence")


def test_t6_without_the_hash_outputs_still_identical(rank_transcripts):
    """'Then kill the debug collective and rerun — outputs must be identical'
    (03 T6): recompute the trajectory in-process with no hashing, no
    processes, no env games; the swap lists still match every rank."""
    tier_of = initial_membership(SEED)
    reference = rank_transcripts[0]
    for i in range(INTERVALS):
        inp = interval_inputs(SEED, i, tier_of)
        swaps = P.decide(inp["stats"], inp["eps"], inp["tier_of"],
                         pins=inp["pins"], dwell=inp["dwell"], cfg=inp["cfg"])
        assert [[int(l), int(a), int(b)] for l, a, b in swaps] == \
            reference[i]["swaps"], f"interval {i} diverged without the hash"
        tier_of = P.apply_swaps(tier_of, swaps)


def test_t6_policy_documents_are_topology_neutral():
    """D4 / store.validate_policy: nothing rank-shaped may enter the document
    the ranks compare — that is *why* they can agree without a collective."""
    doc = {
        "schema": S.POLICY_SCHEMA,
        "manifest": "t6-synthetic",
        "budget": {"n_k4_per_layer": {"0": N_K4}},
        "bits_per_expert": {"0": [4] * N_K4 + [3] * (E - N_K4)},
    }
    S.validate_policy(doc, num_experts=E)
    for banned in ("rank", "world_size", "tp", "device", "devices"):
        with pytest.raises(ValueError, match="topology-neutral"):
            S.validate_policy(dict(doc, **{banned: 0}), num_experts=E)


def test_t6_digest_matches_the_serving_loops_definition():
    """The digest proven here is the one the serve log prints per rank
    (loop.FungibleQuantState.decision_sha). Skipped if loop.py is not
    importable standalone (it is another agent's live file)."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        pkg = sys.modules[_PKG]
        if not hasattr(pkg, "stats"):  # loop.py's remaining sibling import
            pkg.stats = load(f"{_PKG}.stats", _DIR / "stats.py")
        loop_sha = load("fq_t6_loop_probe",
                        _DIR / "loop.py").FungibleQuantState.decision_sha
    except BaseException as exc:  # noqa: BLE001 — optional cross-check
        pytest.skip(f"loop.py not importable standalone here: {exc!r}")
    finally:
        sys.modules.pop("fq_t6_loop_probe", None)
    swaps = [(0, 3, 7), (2, 1, 9)]
    assert loop_sha(swaps) == decision_sha(swaps)
    assert loop_sha([]) == decision_sha([])


if __name__ == "__main__":
    sys.exit(pytest.main([
        __file__, "-v", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ]))
