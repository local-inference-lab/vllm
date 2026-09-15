# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Units, workloads, and the preparation driver for native b12x kernels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.b12x_blockscaled import B12xBlockscaledLinear
from vllm.model_executor.warmup import b12x_prepare
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    _B12X_UNIT_PROVIDERS,
    b12x_layer,
    b12x_layer_prefix,
    b12x_preparation_token_counts,
    register_b12x_layer,
    register_b12x_unit_provider,
    scope_b12x_unit_calls,
)
from vllm.v1.worker.workspace import _workspace_lane


@dataclass
class _Request:
    name: str
    collective: object = None
    prepare_call: object = None
    benchmark_call: object = None
    dependencies: tuple = ()
    plan: object = None

    def __post_init__(self):
        if self.plan is None:
            self.plan = SimpleNamespace(name=self.name, prepared=None)


@dataclass
class _Call:
    run: object
    produce: object = None
    reset: object = None
    restore: object = None
    close: object = None


def _workload(stage="weights", **overrides):
    values = dict(
        stage=stage, token_counts=(1, 2, 4, 32), fixed_token_counts=(1, 2, 4),
        output_dtype=torch.bfloat16, max_tokens=32, max_seqs=2, max_model_len=32,
    )
    values.update(overrides)
    return B12xWorkload(**values)


def _unit(name, *requests, stage="weights", autotune=True, key=None):
    return B12xPreparationUnit(
        name=name, key=key or name, requests=tuple(requests), stage=stage, autotune=autotune,
    )


class _Provider:
    def __init__(self, stage="weights", *, record=None):
        self.stage = stage
        self.record = [] if record is None else record

    def get_b12x_preparation_units(self, layer, workload):
        self.record.append((layer, workload))
        return (_unit(f"family-{id(layer):x}", _Request(f"{id(layer):x}"), stage=self.stage),)


def _worker(model, *, draft=None, draft_lane=0):
    return SimpleNamespace(
        get_model=lambda: model,
        get_draft_model=lambda: draft,
        model_runner=SimpleNamespace(mm_registry=None, _draft_workspace_lane=draft_lane),
    )


@pytest.mark.parametrize(
    "register",
    (
        ("vllm.model_executor.layers.vocab_parallel_embedding", "_register_b12x_embedding_collective", (object(), "", 128, 2)),
        ("vllm.model_executor.layers.linear", "_register_b12x_row_parallel_collective", (object(), "", 128, True)),
    ),
)
def test_collective_describers_reject_process_local_identity(monkeypatch, register) -> None:
    import importlib

    module_name, function_name, arguments = register
    function = getattr(importlib.import_module(module_name), function_name)
    describers = []
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.register_b12x_collective_describer",
        lambda _owner, describe: describers.append(describe),
    )

    function(*arguments)

    assert len(describers) == 1
    workload = SimpleNamespace(token_counts=(1,), output_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="stable module prefix"):
        describers[0](workload)


def test_preparation_token_counts_cover_every_serving_regime() -> None:
    counts = b12x_preparation_token_counts(
        max_tokens=32,
        cudagraph_capture_sizes=(1, 2),
        compile_sizes=(4, 16),
        compile_range_endpoints=(16, 23, 32),
        speculative_tokens=3,
    )
    assert counts == (1, 2, 4, 16, 23, 29, 32)
    workload = _workload(token_counts=counts, fixed_token_counts=(1, 2, 4, 16))
    assert workload.fixed_token_counts == (1, 2, 4, 16)


def test_workload_rejects_invalid_stage_and_counts() -> None:
    with pytest.raises(ValueError, match="invalid native preparation stage"):
        _workload(stage="loaded")
    with pytest.raises(ValueError, match="within capacity"):
        _workload(token_counts=(1, 33))
    with pytest.raises(ValueError, match="sorted subset"):
        _workload(fixed_token_counts=(3,))
    with pytest.raises(ValueError, match="below capacity"):
        _workload(fixed_token_counts=(32,))


def test_unit_rejects_duplicate_request_names() -> None:
    with pytest.raises(ValueError, match="duplicate request names"):
        _unit("dup", _Request("a"), _Request("a"))


def test_batches_group_timed_requests_before_default_only() -> None:
    timed_a, timed_b, default = _Request("a"), _Request("b"), _Request("c")
    batches = b12x_prepare.b12x_batches([
        _unit("x", timed_a), _unit("y", default, autotune=False), _unit("z", timed_b),
    ])
    assert batches == [((timed_a, timed_b), True), ((default,), False)]
    assert b12x_prepare.b12x_batches([]) == []


def test_collect_units_filters_by_stage_and_forces_eager_defaults() -> None:
    record = []
    weights, state, tower = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
    weights.b12x_preparation_provider = _Provider("weights", record=record)
    state.b12x_preparation_provider = _Provider("state", record=record)
    tower.b12x_preparation_provider = _Provider("weights", record=record)
    tower.b12x_eager_token_counts = (65_536,)
    tower.b12x_eager_only = True
    model = torch.nn.Module()
    model.a, model.b, model.visual = weights, state, tower
    worker = _worker(model)

    units = b12x_prepare.collect_b12x_units(worker, _workload("weights"))
    names = {unit.requests[0].name: unit for unit in units}
    assert set(names) == {f"{id(weights):x}", f"{id(tower):x}"}
    assert names[f"{id(tower):x}"].autotune is False
    seen = {id(layer): workload for layer, workload in record}
    assert seen[id(tower)].token_counts == (65_536,)
    assert seen[id(tower)].fixed_token_counts == ()
    assert seen[id(tower)].eager_only is True
    assert seen[id(weights)].eager_only is False

    # The state stage carries the weights units again (prepared plans are
    # skipped by the session) but never the eager-only encoder shapes.
    record.clear()
    units = b12x_prepare.collect_b12x_units(worker, _workload("state"))
    assert [unit.requests[0].name for unit in units] == [f"{id(state):x}", f"{id(weights):x}"]
    assert id(tower) not in {id(layer) for layer, _ in record}


def test_collect_units_asks_unit_providers_and_rejects_duplicate_names() -> None:
    class _Comm:
        def get_b12x_preparation_units(self, owner, workload):
            assert owner is self
            return (_unit("comm", _Request("shared")),)

    comm = _Comm()
    register_b12x_unit_provider(comm)
    try:
        model = torch.nn.Module()
        assert [unit.name for unit in b12x_prepare.collect_b12x_units(_worker(model), _workload())] == ["comm"]
        layer = torch.nn.Module()
        layer.b12x_preparation_provider = SimpleNamespace(
            get_b12x_preparation_units=lambda module, workload: (_unit("dup", _Request("shared")),)
        )
        model.layer = layer
        with pytest.raises(ValueError, match="conflicting b12x preparation request names"):
            b12x_prepare.collect_b12x_units(_worker(model), _workload())
    finally:
        _B12X_UNIT_PROVIDERS[:] = [ref for ref in _B12X_UNIT_PROVIDERS if ref() is not comm]


def test_draft_model_units_run_their_callbacks_in_the_draft_lane() -> None:
    observed = []

    class _LaneProvider:
        def get_b12x_preparation_units(self, layer, workload):
            observed.append(("collect", workload.lane, _workspace_lane.get()))
            request = _Request("draft")

            def prepare(state):
                observed.append(("prepare", _workspace_lane.get()))
                return _Call(run=lambda: observed.append(("run", _workspace_lane.get())))

            request.prepare_call = prepare
            return (_unit("draft", request),)

    draft = torch.nn.Module()
    draft.b12x_preparation_provider = _LaneProvider()
    worker = _worker(torch.nn.Module(), draft=draft, draft_lane=1)

    (unit,) = b12x_prepare.collect_b12x_units(worker, _workload())
    call = unit.requests[0].prepare_call(object())
    call.run()
    assert observed == [("collect", 1, 1), ("prepare", 1), ("run", 1)]
    assert _workspace_lane.get() == 0


def test_scope_unit_calls_wraps_mapping_factories() -> None:
    request = _Request("composite")
    request.prepare_call = {2: lambda state: _Call(run=lambda: _workspace_lane.get())}
    scoped = scope_b12x_unit_calls(_unit("u", request), 1)
    assert scoped.requests[0].prepare_call[2](object()).run() == 1


def test_blockscaled_holder_declares_provided_workspace_under_the_cap(monkeypatch) -> None:
    queries, regimes = [], []

    class _Query:
        def __init__(self, **kwargs):
            queries.append(kwargs)

    class _Plan:
        token_counts = (1, 2, 32)

        def request(self, *, name, prepare_calls, benchmark_calls):
            assert set(prepare_calls) == set(self.token_counts) == set(benchmark_calls)
            return _Request(name)

    def plan_regimes(query, *, exact_m):
        regimes.append(exact_m)
        return _Plan()

    api = SimpleNamespace(BlockscaledQuery=_Query, plan_regimes=plan_regimes)
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.b12x_blockscaled.get_b12x_blockscaled", lambda: api
    )
    packed = SimpleNamespace(
        in_features=128, padded_in_features=128, out_features=64,
        weight=SimpleNamespace(values=torch.zeros(1), scale_mma=torch.zeros(1)),
    )
    holder = B12xBlockscaledLinear(packed, recipe="mxfp8", activation_mode="auto", layer_name="layer")
    with pytest.raises(PreparationResourceUnavailableError, match="no declared plan"):
        holder.run(torch.zeros(1, 128), None)

    unit = holder.unit(_workload(fixed_token_counts=(1, 2)), name="linear.mxfp8.layer")
    assert unit.stage == "weights" and unit.autotune is True
    assert regimes == [(1, 2)]
    assert queries[-1]["workspace_form"] == "provided"
    assert queries[-1]["workspace_nbytes"] == envs.VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES
    assert queries[-1]["num_tokens"] == 32 and queries[-1]["global_scale_kind"] == "none"

    # The declared plan is never replaced: more exact-M regimes are served by
    # the capacity regime, and a different capacity is a declaration error.
    plan = holder.plan
    again = holder.unit(_workload(fixed_token_counts=(1, 2, 4)), name="linear.mxfp8.layer")
    assert holder.plan is plan and len(regimes) == 1 and again.stage == "weights"
    with pytest.raises(ValueError, match="capacity changed"):
        holder.unit(_workload(token_counts=(65_536,), fixed_token_counts=(), max_tokens=65_536, eager_only=True), name="x")

    tower = B12xBlockscaledLinear(packed, recipe="mxfp8", activation_mode="auto", layer_name="tower")
    eager = tower.unit(_workload(token_counts=(65_536,), fixed_token_counts=(), max_tokens=65_536, eager_only=True), name="x")
    assert eager.autotune is False and regimes[-1] == ()


@pytest.mark.parametrize("fail", [False, True])
def test_profile_prepares_collectives_and_releases_only_fresh_plans(monkeypatch, fail) -> None:
    from b12x.preparation import CollectiveRequirement
    from vllm.distributed import parallel_state

    released, prepared, authorized = [], [], []
    requirement = CollectiveRequirement(key="profile-collective", ranks=(0,))
    plain, collective = _Request("plain"), _Request("collective", collective=requirement)
    installed = _Request("installed", plan=SimpleNamespace(name="installed", prepared=object()))

    class Job:
        session = SimpleNamespace(_pool=None)
        waiting = True

        def advance(self, *, collective_key=None, tuning=None):
            if self.waiting:
                self.waiting = False
                return SimpleNamespace(done=False, pending_compilation=False, ready_collectives=(requirement,))
            authorized.append(collective_key)
            assert collective_key == requirement.key
            if fail:
                raise RuntimeError("profiling collective failed")
            return SimpleNamespace(done=True, pending_compilation=False, ready_collectives=())

        def result(self):
            return SimpleNamespace(close=lambda: None)

        def close(self):
            pass

    session = SimpleNamespace(
        state="OPEN",
        begin=lambda requests, *, autotune: prepared.append((requests, autotune)) or Job(),
        release=lambda plan: released.append(plan.name),
        cancel_tuning=lambda: None,
    )
    monkeypatch.setattr(b12x_prepare, "b12x_native_supported", lambda worker: True)
    monkeypatch.setattr(b12x_prepare, "b12x_workload", lambda worker, *, stage, lane=0: _workload(stage))
    monkeypatch.setattr(
        b12x_prepare, "collect_b12x_units",
        lambda worker, workload: [
            _unit("a", plain, stage="state"), _unit("b", collective, stage="state"),
            _unit("c", installed, stage="weights"),
        ],
    )
    monkeypatch.setattr(b12x_prepare, "get_b12x_session", lambda worker: session)
    world = SimpleNamespace(ranks=(0,), tcp_store_group=SimpleNamespace(all_gather_obj=lambda payload: [payload]))
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    worker = SimpleNamespace(rank=0)
    if fail:
        with pytest.raises(RuntimeError, match="profiling collective failed"):
            b12x_prepare.prepare_b12x_profile(worker, stage="state")
    else:
        batch = b12x_prepare.prepare_b12x_profile(worker, stage="state")
        batch.release()
        batch.release()
    assert prepared == [((plain, collective, installed), False)]
    assert authorized == [requirement.key]
    assert released == ["collective", "plain"]


def test_layer_registry_resolves_live_layers_by_name() -> None:
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp"
    name = b12x_layer_prefix(layer)
    assert name == "model.layers.0.mlp"
    register_b12x_layer(name, layer)
    assert b12x_layer(name) is layer
    other = torch.nn.Module()
    other.prefix = "model.layers.0.mlp"
    assert b12x_layer_prefix(other).startswith("model.layers.0.mlp#")
    with pytest.raises(ValueError, match="already registered"):
        register_b12x_layer(name, other)
    anonymous = torch.nn.Module()
    assert b12x_layer_prefix(anonymous).startswith("Module#")
    with pytest.raises(PreparationResourceUnavailableError):
        b12x_layer("model.layers.99.mlp")


def _config_worker(*, capture_sizes, max_seqs, speculative_tokens, decode_query_len=None, manager=None):
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=list(capture_sizes), compile_sizes=[],
        get_compile_ranges=lambda: [], max_cudagraph_capture_size=max(capture_sizes),
    )
    spec = None if not speculative_tokens else SimpleNamespace(num_speculative_tokens=speculative_tokens)
    runner = SimpleNamespace(mm_registry=None, _draft_workspace_lane=0, cudagraph_manager=manager)
    if decode_query_len is not None:
        runner.decode_query_len = decode_query_len
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation, speculative_config=spec),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=max_seqs),
        model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=4096),
        model_runner=runner,
    )


def test_workload_declares_the_same_decode_counts_in_both_stages() -> None:
    sizes = (1, 2, 4, 8, 16)
    before = _config_worker(capture_sizes=sizes, max_seqs=2, speculative_tokens=3, decode_query_len=4)
    weights = b12x_prepare.b12x_workload(before, stage="weights")
    # Uniform decode graphs round the capture sizes up to the query length and
    # keep the ones that fit the request budget: 4 (one request) and 8 (two).
    assert weights.token_counts == (1, 2, 4, 8, 16, 125, 128)
    assert weights.fixed_token_counts == (1, 2, 4, 8, 16)
    manager = SimpleNamespace(planned_token_counts=lambda: [4, 8, 1, 2, 16])
    after = _config_worker(capture_sizes=sizes, max_seqs=2, speculative_tokens=3, decode_query_len=4, manager=manager)
    state = b12x_prepare.b12x_workload(after, stage="state")
    assert state.token_counts == weights.token_counts
    assert state.fixed_token_counts == weights.fixed_token_counts

    # Without a runner query length the speculative configuration decides.
    implicit = _config_worker(capture_sizes=(1, 2, 4), max_seqs=1, speculative_tokens=2)
    assert b12x_prepare.b12x_workload(implicit, stage="weights").token_counts == (1, 2, 3, 4, 126, 128)
    plain = _config_worker(capture_sizes=(1, 2, 4), max_seqs=1, speculative_tokens=0)
    assert b12x_prepare.b12x_workload(plain, stage="weights").token_counts == (1, 2, 4, 128)


def test_provider_attachment_never_registers_a_child_module() -> None:
    from vllm.utils.b12x import set_b12x_preparation_provider

    layer = torch.nn.Module()
    layer.inner = torch.nn.Linear(2, 2)
    set_b12x_preparation_provider(layer, layer)
    assert layer.b12x_preparation_provider is layer
    names = [name for name, _ in layer.named_modules(remove_duplicate=False)]
    assert names == ["", "inner"]
    provider = torch.nn.Module()
    set_b12x_preparation_provider(layer, provider)
    assert layer.b12x_preparation_provider is provider
    assert "b12x_preparation_provider" not in dict(layer.named_children())


def test_declaration_dump_flattens_metadata_and_never_raises(tmp_path, monkeypatch) -> None:
    from b12x.preparation import FrozenMapping

    path = tmp_path / "queries.jsonl"
    monkeypatch.setenv("VLLM_B12X_DUMP_QUERIES", str(path))
    contract = SimpleNamespace(
        component_id="test.family", query_schema_version=3,
        encode_query=lambda query: {"rows": query.rows, "dtype": torch.bfloat16, "recipe": FrozenMapping({"tile": (64, 128)})},
    )
    plan = SimpleNamespace(
        contract=contract, query=SimpleNamespace(rows=4),
        invocation=FrozenMapping({"operation": "pre", "nested": FrozenMapping({"k": 1})}), shared=True,
    )
    request = _Request("family.m4", plan=plan)
    b12x_prepare._dump_declarations([_unit("family", request)], _workload())
    import json
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["query"] == {"rows": 4, "dtype": "bfloat16", "recipe": {"tile": [64, 128]}}
    assert record["invocation"] == {"operation": "pre", "nested": {"k": 1}}
    assert record["shared"] is True and record["component"] == "test.family"

    broken = SimpleNamespace(contract=SimpleNamespace(
        component_id="x", query_schema_version=1, encode_query=lambda q: (_ for _ in ()).throw(RuntimeError("boom"))),
        query=None, invocation=FrozenMapping(), shared=False)
    b12x_prepare._dump_declarations([_unit("broken", _Request("b", plan=broken))], _workload())


def test_b12x_batches_with_autotune_disabled_time_nothing():
    timed_a, timed_b, default = (SimpleNamespace(name=n) for n in ("a", "b", "d"))
    units = [_unit("x", timed_a), _unit("y", default, autotune=False), _unit("z", timed_b)]

    assert b12x_prepare.b12x_batches(units) == [
        ((timed_a, timed_b), True), ((default,), False),
    ]
    assert b12x_prepare.b12x_batches(units, autotune=False) == [
        ((timed_a, default, timed_b), False),
    ]


@pytest.mark.parametrize("shared_module", [True, False])
@pytest.mark.parametrize("draft_lane", [0, 1])
def test_collect_units_handles_target_and_draft_embedding_aliases(shared_module, draft_lane):
    from vllm.models.deepseek_v4_1.b12x_layers import B12xEmbeddingMethod

    target = torch.nn.Module()
    target.embed_tokens = torch.nn.Module()
    target.embed_tokens.weight = torch.nn.Parameter(
        torch.ones(32, 64, dtype=torch.bfloat16), requires_grad=False,
    )
    method = B12xEmbeddingMethod()
    method.process_weights_after_loading(target.embed_tokens)
    draft = torch.nn.Module()
    if shared_module:
        draft.embed_tokens = target.embed_tokens
    else:
        draft.embed_tokens = torch.nn.Module()
        draft.embed_tokens.weight = target.embed_tokens.weight
        method.process_weights_after_loading(draft.embed_tokens)
    worker = _worker(target, draft=draft, draft_lane=draft_lane)
    units = b12x_prepare.collect_b12x_units(worker, _workload())
    requests = [request for unit in units for request in unit.requests]
    assert len(requests) == (2 if shared_module else 4)
    assert len({request.name for request in requests}) == len(requests)
    assert len({id(request.plan) for request in requests}) == len(requests)
    assert {request.plan.query.id_dtype for request in requests} == {"int32", "int64"}


@pytest.mark.parametrize("query_len", [7, 8])
def test_parallel_draft_counts_reach_owners_and_collectives(monkeypatch, query_len):
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.b12x_preparation_provider = _Provider()
    draft.b12x_preparation_provider = _Provider()
    collective = _Provider()
    monkeypatch.setattr(b12x_prepare, "b12x_unit_providers", lambda: (collective,))
    worker = _worker(target, draft=draft, draft_lane=1)
    worker.model_runner.speculator = SimpleNamespace(
        num_query_per_req=query_len, query_cudagraph_manager=None,
    )
    workload = _workload(
        token_counts=(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 4089, 4096),
        fixed_token_counts=(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64),
        max_tokens=4096, max_seqs=8, max_model_len=1048576,
    )
    b12x_prepare.collect_b12x_units(worker, workload)
    target_workload = target.b12x_preparation_provider.record[-1][1]
    draft_workload = draft.b12x_preparation_provider.record[-1][1]
    collective_workload = collective.record[-1][1]
    assert target_workload.token_counts == workload.token_counts
    for reqs in range(1, 9):
        for rows in (reqs, reqs * query_len):
            assert rows in draft_workload.token_counts
            assert rows in draft_workload.fixed_token_counts
            assert rows in collective_workload.token_counts
    assert draft_workload.lane == 1 and collective_workload.lane == 0
    worker.model_runner.speculator.query_cudagraph_manager = SimpleNamespace(
        planned_token_counts=lambda: [query_len * reqs for reqs in range(1, 9)],
    )
    b12x_prepare.collect_b12x_units(worker, replace(workload, stage="state"))
    assert draft.b12x_preparation_provider.record[-1][1].token_counts == draft_workload.token_counts


def test_workload_prepares_every_adaptive_verification_profile_shape():
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager

    capture_sizes = (1, 2, 4, 8, 16, 24, 32, 48, 64)
    worker = _config_worker(capture_sizes=capture_sizes, max_seqs=8, speculative_tokens=7)
    worker.scheduler_config.max_num_batched_tokens = 4096
    worker.vllm_config.speculative_config.enable_adaptive_verification = True
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096)
    profiled = {batch["num_tokens"] for batch in manager.batches_to_profile(list(capture_sizes))}
    tail = {96, 128, 256, 512, 1024, 2048, 4096}
    assert profiled == {*capture_sizes, *tail}
    for stage in ("weights", "state"):
        workload = b12x_prepare.b12x_workload(worker, stage=stage)
        assert profiled <= set(workload.token_counts)
        assert not tail.intersection(workload.fixed_token_counts)


def test_workload_covers_sampler_warmup_prefill_request_limits():
    from vllm.v1.worker.gpu.warmup import warmup_prefill_shape

    worker = _config_worker(
        capture_sizes=(1, 2, 4, 8, 16, 24, 32, 48, 64),
        max_seqs=8, speculative_tokens=7, decode_query_len=8,
    )
    worker.use_v2_model_runner = True
    worker.scheduler_config.max_num_batched_tokens = 4096
    prompt_len, max_reqs = warmup_prefill_shape(
        max_num_seqs=8, max_num_batched_tokens=4096, decode_query_len=8,
    )
    assert (prompt_len, max_reqs) == (9, 8)
    for stage in ("weights", "state"):
        workload = b12x_prepare.b12x_workload(worker, stage=stage)
        assert all(prompt_len * reqs in workload.token_counts for reqs in range(1, max_reqs + 1))
        assert 72 not in workload.fixed_token_counts
