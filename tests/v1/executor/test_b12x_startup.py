# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local startup progress and distributed preparation boundaries."""
from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm.v1.executor.abstract import Executor, _aggregate_b12x_progress
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator, _authorize_tuning

"""Cooperative control for the b12x warmup prelude."""



def _progress(*, done=False, pending=False, ready=()):
    return SimpleNamespace(
        done=done,
        pending_compilation=pending,
        ready_collectives=ready,
    )



class _Result:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("result-close")



class _Job:
    def __init__(self, events, progress):
        self.events = events
        self.progress = iter(progress)
        self.session = SimpleNamespace(_pool=None, cancel_tuning=self.cancel)
        self.events_session = None
        self.keys = []
        self.tunings = []

    def cancel(self):
        self.events.append("cancel")

    def advance(self, *, collective_key=None, tuning=None):
        self.keys.append(collective_key)
        self.tunings.append(tuning)
        self.events.append("advance")
        return next(self.progress)

    def result(self):
        self.events.append("result")
        return _Result(self.events)

    def close(self):
        self.events.append("job-close")



class _Session:
    def __init__(self, job, events):
        self._job = job
        self.events = events
        self.state = "OPEN"
        self._pool = None

    def begin(self, requests, *, autotune=None):
        self.events.append(("begin", autotune))
        return self._job

    def cancel_tuning(self):
        self.events.append("cancel")



def _batches(autotune=True):
    return [((object(),), autotune)]



def test_tuning_authorization_selects_once_across_disjoint_rank_shards() -> None:
    gathered = [
        {
            "global_rank": 0,
            "tuning": (("query", (0, 1), {"width": 4}, 3.0, 0),),
        },
        {
            "global_rank": 1,
            "tuning": (("query", (0, 1), {"width": 2}, 1.0, 1),),
        },
    ]

    assert _authorize_tuning(gathered, (0, 1)) == ((
        "query",
        (0, 1),
        {"width": 2},
        1.0,
        1,
    ),)



def test_progress_aggregates_work_totals_and_preserves_rank_local_batch_counts() -> None:
    from b12x.preparation import PreparationProgress

    outcomes = [
        {
            "global_rank": rank,
            "progress": PreparationProgress(
                running=True,
                pending_compilation=False,
                ready_collectives=(),
                done=False,
                phase="autotuning",
                component_id="sequence.gdn_prefill",
                request_name="shared-query",
                candidate_count=96,
                global_candidate_count=192,
                total_candidates=240,
                candidates_prepared=96,
                batch_index=rank + 1,
                batch_candidates=1,
                tuning_rank=rank,
                measured_candidates=96 * rank,
                compilations=300 + rank,
                active_compilations=rank,
                latest_round_us=(float(rank + 1),),
                elapsed_seconds=float(rank + 1),
                candidate_sharded=True,
            ),
        }
        for rank in range(2)
    ]

    progress = _aggregate_b12x_progress(outcomes)
    assert progress.candidate_count == 96
    assert progress.candidates_prepared == 96
    assert progress.global_candidate_count == 192
    assert progress.total_candidates == 480
    assert progress.measured_candidates == 96
    assert progress.compilations == 601
    assert progress.active_compilations == 1
    assert progress.latest_round_us == (1.0,)
    assert progress.batch_index == 1
    assert progress.batch_candidates == 1
    assert progress.tuning_rank == 0
    assert progress.elapsed_seconds == 2.0



def test_progress_does_not_finish_until_every_rank_finishes() -> None:
    from b12x.preparation import PreparationProgress

    ready = PreparationProgress(False, False, (), True, phase="ready")
    active = PreparationProgress(True, False, (), False, phase="priming")
    progress = _aggregate_b12x_progress(
        [
            {"global_rank": 0, "done": False, "progress": ready},
            {"global_rank": 1, "done": False, "progress": active},
        ]
    )
    assert progress.done is False
    assert progress.phase == "priming"



def test_progress_does_not_sum_replicated_fixed_candidate_count() -> None:
    from b12x.preparation import PreparationProgress

    fixed = PreparationProgress(
        True,
        True,
        (),
        False,
        phase="compiling",
        component_id="comm.pcie",
        request_name="fixed",
        candidate_count=1,
    )
    progress = _aggregate_b12x_progress(
        [
            {"global_rank": rank, "done": False, "progress": fixed}
            for rank in range(2)
        ]
    )
    assert progress.candidate_count == 1



def test_local_only_cancel_still_completes_and_primes() -> None:
    events = []
    job = _Job(events, [_progress(done=True)])
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    outcome = coordinator.advance(cancel_tuning=True)

    assert outcome["done"] is True
    assert outcome["cleanup_complete"] is True
    assert events == [
        ("begin", True),
        "cancel",
        "advance",
        "result",
        "result-close",
    ]



def test_default_only_batch_disables_tuning_for_job() -> None:
    events = []
    coordinator = B12xPreparationCoordinator(
        _Session(_Job(events, [_progress(done=True)]), events),
        _batches(autotune=False),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert events == [("begin", False)]
    coordinator.abort()



def test_batches_run_in_order_and_finish_after_the_last() -> None:
    events = []
    first = _Job(events, [_progress(done=True)])
    second = _Job(events, [_progress(done=True)])
    session = _Session(first, events)
    jobs = iter([first, second])
    session.begin = lambda requests, *, autotune=None: (
        events.append(("begin", autotune)) or next(jobs)
    )
    coordinator = B12xPreparationCoordinator(
        session,
        [((object(),), True), ((object(),), False)],
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.advance()["done"] is False
    assert coordinator.advance()["done"] is True
    assert events == [
        ("begin", True), "advance", "result", "result-close",
        ("begin", False), "advance", "result", "result-close",
    ]



def test_pending_compilation_wait_is_bounded() -> None:
    events = []
    job = _Job(events, [_progress(pending=True)])

    class _Pool:
        def wait_for_progress(self, *, timeout):
            events.append(("wait", timeout))

    job.session._pool = _Pool()
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.advance()["done"] is False
    assert ("wait", 0.05) in events



def test_abort_closes_active_job_once() -> None:
    events = []
    job = _Job(events, [_progress(pending=True)])
    coordinator = B12xPreparationCoordinator(
        _Session(job, events),
        _batches(),
        global_rank=0,
        world_group=None,
        process_local_only=True,
    )

    assert coordinator.abort()["cleanup_complete"] is True
    coordinator.abort()
    assert events.count("job-close") == 1



@pytest.mark.parametrize("variant", ("batched", "varlen"))
def test_attention_tuning_rendezvous_ignores_rank_local_device_ordinal(variant):
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from b12x.attention import varlen
    from b12x.preparation.session import PreparationJob

    mode = FakeTensorMode()
    ranks = (0, 1, 2, 3)
    gathered = []
    for rank in ranks:
        device = torch.device("cuda", rank)

        def metadata(shape, dtype):
            return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), device)

        q, k, v = (metadata((9216, 16, 64), torch.bfloat16) for _ in range(3))
        if variant == "varlen":
            plan = varlen.plan(
                q, k, v, metadata((2,), torch.int32),
                max_seqlen_q=9216, max_seqlen_k=9216, causal=False,
            )
        else:
            plan = varlen.plan_batched(q, k, v, causal=False)
        request = SimpleNamespace(plan=plan, dependencies=())
        configuration = plan.contract.configure(plan.query, device=None)
        obligation = SimpleNamespace(request=request, configuration=configuration)
        key = PreparationJob._choice_key(None, obligation, {})
        gathered.append({
            "global_rank": rank,
            "tuning": ((key, ranks, {"tile_m": 128, "tile_n": 64}, 10.0 + rank, rank),),
        })
    authorized = _authorize_tuning(gathered, ranks)
    assert authorized is not None
    assert authorized[0][1:] == (ranks, {"tile_m": 128, "tile_n": 64}, 10.0, 0)



def _coordinators(progress_by_rank):
    store = dist.HashStore()
    store.set_timeout(timedelta(seconds=5))
    ranks = tuple(range(len(progress_by_rank)))
    coordinators, jobs = [], []
    for rank, progress in enumerate(progress_by_rank):
        events = []
        job = _Job(events, progress)
        jobs.append(job)
        group = SimpleNamespace(store=store)
        # Prefixes must be shared across rank-local channel instances.
        coordinator = B12xPreparationCoordinator(
            _Session(job, events) if progress else None,
            _batches() if progress else [],
            global_rank=rank,
            world_group=SimpleNamespace(ranks=ranks, tcp_store_group=group),
        )
        coordinators.append(coordinator)
    return coordinators, jobs, store


def _finish(coordinator):
    for _ in range(30):
        outcome = coordinator.advance()
        if outcome["done"]:
            return outcome
    raise AssertionError("coordinator did not finish")


def _tuning_progress(rank, *, keys=("a", "b"), ranks=(0, 1)):
    from b12x.preparation import TuningRequirement

    return SimpleNamespace(
        done=False, pending_compilation=False, ready_collectives=(),
        ready_tuning=tuple(
            TuningRequirement(key, ranks, {"width": rank + 1}, float(2 - rank), rank)
            for key in keys
        ),
    )


def test_independent_work_never_waits_for_peer_progress():
    coordinators, jobs, store = _coordinators([
        [_progress() for _ in range(20)] + [_progress(done=True)],
        [_progress(done=True)],
    ])
    for _ in range(20):
        assert not coordinators[0].advance()["done"]
    assert jobs[1].keys == []
    assert not store.check(["stage-0/round-0/0"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_finish, coordinators))
    assert all(item["done"] for item in outcomes)
    assert [item["round"] for item in outcomes] == [1, 1]


def test_all_local_winners_consolidate_once_with_empty_world_rank():
    coordinators, jobs, store = _coordinators([
        [_tuning_progress(0), _progress(done=True)],
        [_tuning_progress(1), _progress(done=True)],
        [],
    ])
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(_finish, coordinators))
    assert all(item["done"] and not item["error"] for item in outcomes)
    assert [item["round"] for item in outcomes] == [2, 2, 2]
    for job in jobs[:2]:
        winners = job.tunings[1]
        assert [winner.key for winner in winners] == ["a", "b"]
        assert all(winner.assignment["width"] == 2 for winner in winners)
    decision = pickle.loads(store.get("stage-0/round-0/decision"))
    assert len(decision["tuning"]) == 2


def test_fixed_collective_authorization_waits_for_every_participant():
    required = SimpleNamespace(key="comm", ranks=(0, 1))
    coordinators, jobs, store = _coordinators([
        [_progress(ready=(required,)), _progress(done=True)],
        [_progress(ready=(required,)), _progress(done=True)],
    ])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_finish, coordinators[0])
        store.wait(["stage-0/round-0/0"], timedelta(seconds=2))
        assert jobs[0].keys == [None]
        assert not first.done()
        second = pool.submit(_finish, coordinators[1])
        assert first.result(timeout=3)["done"]
        assert second.result(timeout=3)["done"]
    assert all(job.keys == [None, "comm"] for job in jobs)


def test_cancelled_consolidation_discards_every_partial_winner():
    coordinators, jobs, store = _coordinators([
        [_tuning_progress(0), _progress(done=True)],
        [_progress(done=True)],
    ])
    store.set("stage-0/stop", b"1")
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_finish, coordinators))
    assert all(item["done"] and not item["error"] for item in outcomes)
    assert jobs[0].tunings[1] == ()
    assert "cancel" in jobs[0].events


def test_peer_failure_drains_every_rank_before_returning():
    coordinators, jobs, _ = _coordinators([
        [_progress() for _ in range(10)], [_progress(done=True)],
    ])
    peer_started = threading.Event()
    def local_progress(**kwargs):
        assert peer_started.wait(2)
        return _progress()
    def fail(**kwargs):
        peer_started.set()
        raise ValueError("candidate setup failed")
    jobs[0].advance = local_progress
    jobs[1].advance = fail
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(_finish, coordinators))
    assert all(item["done"] and item["cleanup_complete"] for item in outcomes)
    assert all(item["error"]["message"] == "candidate setup failed" for item in outcomes)
    assert all(job.events.count("job-close") == 1 for job in jobs)


def test_executor_runs_worker_on_calling_thread_and_cancels_without_step_rpcs():
    cancel = threading.Event()
    caller = threading.get_ident()
    calls, cancellations = [], []
    class Worker:
        rank = 0
        def advance_b12x_preparation(self, *, cancel_tuning=False):
            assert threading.get_ident() == caller
            cancellations.append(cancel_tuning)
            cancel.set()
            time.sleep(0.02)
            assert len(cancellations) < 100
            return {"global_rank": 0, "done": cancel_tuning, "error": None}
    worker = Worker()
    def rpc(method, kwargs=None):
        calls.append(method)
        if method == "begin_b12x_preparation":
            return [{"native": False, "global_rank": 0, "done": False}]
        assert method == "run_b12x_preparation"
        return [WorkerBase.run_b12x_preparation(worker, **kwargs)]
    executor = SimpleNamespace(collective_rpc=rpc, _b12x_autotuning_cancel=cancel)
    Executor._run_b12x_preparation(executor, stage="weights")
    assert calls == ["begin_b12x_preparation", "run_b12x_preparation"]
    assert cancellations[0] is False and cancellations[-1] is True


def test_candidate_progress_waits_for_every_expected_rank_to_finish_planning():
    from dataclasses import replace
    from b12x.preparation import PreparationProgress

    known = PreparationProgress(False, False, (), False, total_candidates=120, measured_candidates=30)
    rank_zero = {"global_rank": 0, "progress": known}
    assert _aggregate_b12x_progress([rank_zero], expected_ranks=(0, 1)).total_candidates is None
    rank_one = {"global_rank": 1, "progress": replace(known, total_candidates=None)}
    assert _aggregate_b12x_progress([rank_zero, rank_one], expected_ranks=(0, 1)).total_candidates is None
    rank_one["progress"] = replace(known, total_candidates=80, measured_candidates=20)
    combined = _aggregate_b12x_progress([rank_zero, rank_one], expected_ranks=(0, 1))
    assert combined.total_candidates == 200
    assert combined.measured_candidates == 50


@pytest.mark.parametrize("measured", [(3900, 3725, 3725, 3725), (11901,) * 4])
def test_four_rank_candidate_fraction_uses_the_same_scope_for_both_counts(measured):
    from b12x.preparation import PreparationProgress

    outcomes = [
        {
            "global_rank": rank,
            "progress": PreparationProgress(
                False, False, (), False, phase="autotuning",
                tuning_rank=rank, candidate_sharded=True,
                measured_candidates=count, total_candidates=11901,
            ),
        }
        for rank, count in enumerate(measured)
    ]
    progress = _aggregate_b12x_progress(outcomes, expected_ranks=(0, 1, 2, 3))
    assert progress.measured_candidates == sum(measured)
    assert progress.total_candidates == 47604
    assert progress.measured_candidates / progress.total_candidates == pytest.approx(
        sum(measured) / 47604,
    )
    assert progress.measured_candidates <= progress.total_candidates


def test_cancel_drains_pending_winners_then_orders_collective_warmup():
    required = SimpleNamespace(key="comm", ranks=(0, 1))
    coordinators, jobs, store = _coordinators([
        [_tuning_progress(0), _progress(ready=(required,)), _progress(done=True)],
        [_progress(), _progress(ready=(required,)), _progress(done=True)],
        [],
    ])
    # A blocked job retains its collective requirement until authorization.
    for job in jobs[:2]:
        advance = job.advance
        pending = [None]
        def blocked_advance(*, collective_key=None, tuning=None, advance=advance, pending=pending):
            if pending[0] is not None and collective_key != pending[0].ready_collectives[0].key:
                return pending[0]
            progress = advance(collective_key=collective_key, tuning=tuning)
            pending[0] = progress if progress.ready_collectives else None
            return progress
        job.advance = blocked_advance
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(_finish, coordinators[0])
        store.wait(["stage-0/round-0/0"], timedelta(seconds=2))
        assert not first.done()
        empty = pool.submit(_finish, coordinators[2])
        def cancel_peer():
            coordinators[1].advance(cancel_tuning=True)
            return _finish(coordinators[1])
        second = pool.submit(cancel_peer)
        outcomes = [future.result(timeout=4) for future in (first, second, empty)]
    assert all(item["done"] and not item["error"] for item in outcomes)
    assert jobs[0].tunings[1] == ()
    assert all("cancel" in job.events for job in jobs[:2])
    assert all(job.keys[-1] == "comm" for job in jobs[:2])
    decision = pickle.loads(store.get("stage-0/round-0/decision"))
    assert decision["stop"] and decision["tuning"] == ()


def test_heuristic_warmup_is_local_on_each_rank(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from b12x.preparation import (
        CollectiveRequirement, DetectedDevice, MemoryRequirements, Plan,
        PreparationSession, PreparedCall,
    )
    from b12x.preparation.tuning import Knob, TuningContract

    @dataclass(frozen=True)
    class Query:
        rows: int
    @dataclass(frozen=True)
    class Config:
        width: int
    def forbidden(*args, **kwargs):
        raise AssertionError("heuristic warmup entered search or compilation planning")
    contract = TuningContract(
        component_id="test.rank_warmup", query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({"rows"}), config_fields=frozenset({"width"}),
        encode_query=lambda q: {"rows": q.rows}, encode_config=lambda c: {"width": c.width},
        decode_config=lambda value: Config(value["width"]),
        validate_query=lambda *_: None, validate_config=lambda *_: None,
        default_config=lambda *_: Config(7), knobs=(Knob(name="width", values=(1, 2, 4)),),
        parameters=forbidden,
    )
    monkeypatch.setattr(PreparationSession, "_selection_cache", forbidden)
    monkeypatch.setattr(PreparationSession, "_compiler", forbidden)
    store = dist.HashStore()
    store.set_timeout(timedelta(seconds=5))
    def run(rank):
        calls = []
        with PreparationSession(device=DetectedDevice(None, None), autotune=False) as session:
            session.configure_tuning_shard(rank, (0, 1))
            plan = Plan(
                contract=contract, query=Query(3), _compile_jobs=forbidden,
                _memory_requirements=lambda *_: MemoryRequirements(),
                _materialize=lambda selection, device: selection.config.width,
            )
            request = plan.request(
                name="collective", collective=CollectiveRequirement("comm", (0, 1)),
                prepare_call=lambda state: PreparedCall(run=lambda: calls.append(state)),
            )
            coordinator = B12xPreparationCoordinator(
                session, [((request,), False)], global_rank=rank,
                world_group=SimpleNamespace(ranks=(0, 1), tcp_store_group=SimpleNamespace(store=store)),
            )
            outcome = _finish(coordinator)
            assert not outcome["error"]
            assert plan.selection.source == "default"
            assert calls == [7]
            assert not outcome["progress"].ready_tuning
            return outcome
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, (0, 1)))
    assert all(outcome["done"] for outcome in outcomes)
    decision = pickle.loads(store.get("stage-0/round-0/decision"))
    assert decision["tuning"] == ()
    assert decision["collective"] == ("comm", (0, 1))
