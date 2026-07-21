# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import nullcontext
from types import ModuleType

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.v1.attention.backends.mla import b12x_sparse_ckv_decode
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
from vllm.v1.attention.backends.mla.b12x_sparse_ckv_decode import (
    build_dense_union_remap,
    dense_union_remap_reference,
    owner_and_local_ordinal,
    plan_sparse_ckv_decode,
    sparse_decode_batch_eligible,
    sparse_decode_prefetch_targets,
    try_exchange_sparse_decode_shared_layers,
)


def _layout(*, dcp: int = 4, requests: int = 8, pool: int = 0):
    return plan_sparse_ckv_decode(
        dcp_world_size=dcp,
        topk=2048,
        rows_per_request=4,
        max_requests=requests,
        pool_records=pool,
        record_bytes=432,
        prefetch_depth=3,
    )


def test_bulk_shared_prefetch_exchanges_three_layers_once_without_new_workspace():
    class Exchange:
        def __init__(self):
            self.calls = []

        def exchange_layers(self, **kwargs):
            self.calls.append(kwargs)

    layout = _layout()
    exchange = Exchange()
    records = (object(), object(), object())
    outputs = (object(), object(), object())
    transport_indices = object()
    workspace_geometry = (layout.workspace_slots, layout.workspace_bytes)

    used_bulk = try_exchange_sparse_decode_shared_layers(
        enabled=True,
        transport="ce",
        exchange=exchange,
        records_by_layer=records,
        local_indices_by_destination=transport_indices,
        outputs_by_layer=outputs,
        active_records=layout.active_records(1),
        pool_records=layout.pool_records,
    )

    assert used_bulk
    assert exchange.calls == [
        {
            "records_by_layer": records,
            "local_indices_by_destination": transport_indices,
            "outputs_by_layer": outputs,
        }
    ]
    assert (layout.workspace_slots, layout.workspace_bytes) == workspace_geometry


def test_bulk_shared_prefetch_is_default_off(monkeypatch):
    name = "VLLM_B12X_MLA_SPARSE_DECODE_BULK_PREFETCH"
    monkeypatch.delenv(name, raising=False)

    assert b12x_sparse_ckv_decode.envs.environment_variables[name]() is False

    monkeypatch.setenv(name, "1")
    assert b12x_sparse_ckv_decode.envs.environment_variables[name]() is True


@pytest.mark.parametrize("value", ["enabled", "1 ", ""])
def test_bulk_shared_prefetch_rejects_invalid_value(monkeypatch, value):
    name = "VLLM_B12X_MLA_SPARSE_DECODE_BULK_PREFETCH"
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="Valid options"):
        b12x_sparse_ckv_decode.envs.environment_variables[name]()


def test_sparse_decode_transport_rejects_unknown_value(monkeypatch):
    name = "VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT"
    monkeypatch.setenv(name, "unknown")

    with pytest.raises(ValueError, match="Valid options"):
        b12x_sparse_ckv_decode.envs.environment_variables[name]()


@pytest.mark.parametrize(
    ("enabled", "transport"),
    [(False, "ce"), (True, "direct")],
)
def test_bulk_shared_prefetch_preserves_per_layer_fallback(enabled, transport):
    class Exchange:
        pass

    used_bulk = try_exchange_sparse_decode_shared_layers(
        enabled=enabled,
        transport=transport,
        exchange=Exchange(),
        records_by_layer=(object(), object(), object()),
        local_indices_by_destination=object(),
        outputs_by_layer=(object(), object(), object()),
        active_records=8192,
        pool_records=65536,
    )

    assert not used_bulk


def test_bulk_shared_prefetch_old_b12x_falls_back_cleanly():
    class LegacyCopyExchange:
        pass

    used_bulk = try_exchange_sparse_decode_shared_layers(
        enabled=True,
        transport="ce",
        exchange=LegacyCopyExchange(),
        records_by_layer=(object(), object(), object()),
        local_indices_by_destination=object(),
        outputs_by_layer=(object(), object(), object()),
        active_records=8192,
        pool_records=65536,
    )

    assert not used_bulk


def test_bulk_shared_prefetch_requires_exact_depth_three():
    class Exchange:
        def exchange_layers(self, **kwargs):
            raise AssertionError("partial groups must use the per-layer path")

    used_bulk = try_exchange_sparse_decode_shared_layers(
        enabled=True,
        transport="ce",
        exchange=Exchange(),
        records_by_layer=(object(), object()),
        local_indices_by_destination=object(),
        outputs_by_layer=(object(), object()),
        active_records=8192,
        pool_records=65536,
    )

    assert not used_bulk


def test_bulk_shared_prefetch_falls_back_above_layered_capacity():
    class Exchange:
        layered_max_records = 8191

        def exchange_layers(self, **kwargs):
            raise AssertionError("oversized batches must use the per-layer path")

    used_bulk = try_exchange_sparse_decode_shared_layers(
        enabled=True,
        transport="ce",
        exchange=Exchange(),
        records_by_layer=(object(), object(), object()),
        local_indices_by_destination=object(),
        outputs_by_layer=(object(), object(), object()),
        active_records=8192,
        pool_records=65536,
    )

    assert not used_bulk


def test_bulk_shared_prefetch_rejects_pool_overflow():
    class Exchange:
        def exchange_layers(self, **kwargs):
            raise AssertionError("overflow must fail before exchange")

    with pytest.raises(ValueError, match="exceed pool"):
        try_exchange_sparse_decode_shared_layers(
            enabled=True,
            transport="ce",
            exchange=Exchange(),
            records_by_layer=(object(), object(), object()),
            local_indices_by_destination=object(),
            outputs_by_layer=(object(), object(), object()),
            active_records=65537,
            pool_records=65536,
        )


def test_bulk_shared_prefetch_reuses_existing_sparse_state(monkeypatch):
    layout = plan_sparse_ckv_decode(
        dcp_world_size=4,
        topk=2,
        rows_per_request=2,
        max_requests=1,
        pool_records=4,
        record_bytes=8,
        prefetch_depth=3,
    )

    class Output:
        def __init__(self, slot):
            self.slot = slot
            self.streams = []

        def record_stream(self, stream):
            self.streams.append(stream)

    outputs = [Output(slot) for slot in range(layout.workspace_slots)]

    class Workspace:
        def __getitem__(self, key):
            slot, records = key
            assert records == slice(None, layout.active_records(1), None)
            return outputs[slot]

    class Event:
        def __init__(self):
            self.streams = []

        def record(self, stream):
            self.streams.append(stream)

    class Stream:
        def __init__(self):
            self.waited_for = []

        def wait_stream(self, stream):
            self.waited_for.append(stream)

    class Exchange:
        layered_max_records = layout.pool_records

        def __init__(self):
            self.calls = []

        def exchange_layers(self, **kwargs):
            self.calls.append(kwargs)

    class State:
        payload_workspace = Workspace()
        transport_local_slots = {1: object()}
        complete_events = [Event() for _ in range(layout.workspace_slots)]

    class PrefetchState:
        def get_sparse_decode_state(self, actual_layout, actual_exchange):
            assert actual_layout is layout
            assert actual_exchange is exchange
            return state

    class Metadata:
        num_reqs = 1

    exchange = Exchange()
    state = State()
    stream = Stream()
    current_stream = object()
    monkeypatch.setitem(
        b12x_sparse_ckv_decode._SELECTED_RECORD_STREAMS,
        id(exchange),
        stream,
    )
    monkeypatch.setattr(
        b12x_sparse_ckv_decode.envs,
        "VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT",
        "ce",
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    impl = object.__new__(B12xMLASparseImpl)
    impl._sparse_decode_bulk_prefetch = True
    impl._sparse_decode_layout = layout
    impl._sparse_decode_exchange = exchange
    impl.block_size = 1
    impl.cp_kv_cache_interleave_size = 1
    targets = []
    target_kv_caches = []
    for target_idx in (1, 2, 3):
        target_impl = object.__new__(B12xMLASparseImpl)
        target_impl._sparse_decode_layout = layout
        target_impl._sparse_decode_exchange = exchange
        target_impl.block_size = 1
        target_impl.cp_kv_cache_interleave_size = 1
        target_kv = torch.zeros((1, 1, layout.record_bytes), dtype=torch.uint8)
        target_kv_caches.append(target_kv)
        targets.append((target_idx, target_impl, target_kv))

    def fail_allocation(*args, **kwargs):
        raise AssertionError("bulk prefetch must reuse allocated sparse state")

    for allocator in ("empty", "empty_like", "zeros", "zeros_like"):
        monkeypatch.setattr(torch, allocator, fail_allocation)
    result = impl._dcp_gather_sparse_decode_shared_layers(
        targets,
        Metadata(),
        PrefetchState(),
    )

    assert result is not None
    layer_slots, complete = result
    assert layer_slots == {1: 1, 2: 2, 3: 3}
    assert stream.waited_for == [current_stream]
    assert all(output.streams == [stream] for output in outputs[1:4])
    assert complete is state.complete_events[1]
    assert complete.streams == [stream]
    assert len(exchange.calls) == 1
    call = exchange.calls[0]
    assert all(
        actual.data_ptr() == expected.data_ptr()
        and actual.dtype == expected.dtype
        and actual.shape[-1] == layout.record_bytes
        for actual, expected in zip(
            call["records_by_layer"], target_kv_caches, strict=True
        )
    )
    assert call["outputs_by_layer"] == tuple(outputs[1:4])
    assert call["local_indices_by_destination"] is state.transport_local_slots[1]


def _install_fake_b12x_transport(monkeypatch, *, ce_error=None, include_ce=True):
    calls = []

    class InitializationError(RuntimeError):
        pass

    class DirectExchange:
        @classmethod
        def from_process_group(cls, **kwargs):
            calls.append("direct")
            return cls()

        def close(self):
            pass

    class CopyExchange:
        @classmethod
        def from_process_group(cls, **kwargs):
            calls.append("ce")
            if ce_error is not None:
                raise InitializationError(ce_error)
            return cls()

        def close(self):
            pass

    package = ModuleType("sparkinfer")
    package.__path__ = []
    comm = ModuleType("sparkinfer.comm")
    comm.__path__ = []
    pcie = ModuleType("sparkinfer.comm.pcie")
    pcie.SelectedRecordExchange = DirectExchange
    pcie.SelectedRecordExchangeInitializationError = InitializationError
    if include_ce:
        pcie.SelectedRecordCopyExchange = CopyExchange
    package.comm = comm
    comm.pcie = pcie
    monkeypatch.setitem(sys.modules, "sparkinfer", package)
    monkeypatch.setitem(sys.modules, "sparkinfer.comm", comm)
    monkeypatch.setitem(sys.modules, "sparkinfer.comm.pcie", pcie)
    return calls, DirectExchange, CopyExchange


def _prepare_transport_factory_test(monkeypatch, transport):
    monkeypatch.setattr(
        b12x_sparse_ckv_decode.envs,
        "VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT",
        transport,
    )
    monkeypatch.setattr(b12x_sparse_ckv_decode, "_SELECTED_RECORD_EXCHANGES", {})
    monkeypatch.setattr(b12x_sparse_ckv_decode, "_SELECTED_RECORD_STREAMS", {})
    stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    return stream


def test_selected_record_transport_auto_prefers_copy_engine(monkeypatch):
    stream = _prepare_transport_factory_test(monkeypatch, "auto")
    calls, _, CopyExchange = _install_fake_b12x_transport(monkeypatch)

    exchange = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=object(),
        device=torch.device("cuda", 0),
        layout=_layout(),
        lane_key=("target", 1),
    )

    assert isinstance(exchange, CopyExchange)
    assert calls == ["ce"]
    assert b12x_sparse_ckv_decode.get_selected_record_stream(exchange) is stream


def test_selected_record_transport_auto_falls_back_to_direct(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "auto")
    calls, DirectExchange, _ = _install_fake_b12x_transport(
        monkeypatch, ce_error="CE unavailable"
    )

    exchange = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=object(),
        device=torch.device("cuda", 0),
        layout=_layout(),
        lane_key=("target", 1),
    )

    assert isinstance(exchange, DirectExchange)
    assert calls == ["ce", "direct"]


def test_selected_record_transport_auto_supports_older_b12x(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "auto")
    calls, DirectExchange, _ = _install_fake_b12x_transport(
        monkeypatch, include_ce=False
    )

    exchange = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=object(),
        device=torch.device("cuda", 0),
        layout=_layout(),
        lane_key=("target", 1),
    )

    assert isinstance(exchange, DirectExchange)
    assert calls == ["direct"]


def test_selected_record_transport_direct_never_constructs_copy_engine(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "direct")
    calls, DirectExchange, _ = _install_fake_b12x_transport(monkeypatch)

    exchange = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=object(),
        device=torch.device("cuda", 0),
        layout=_layout(),
        lane_key=("target", 1),
    )

    assert isinstance(exchange, DirectExchange)
    assert calls == ["direct"]


def test_selected_record_transport_isolates_target_and_draft_lanes(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "direct")
    calls, _, _ = _install_fake_b12x_transport(monkeypatch)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    process_group = object()
    layout = _layout()

    target = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=process_group,
        device=torch.device("cuda", 0),
        layout=layout,
        lane_key=("target", 1),
    )
    draft = b12x_sparse_ckv_decode.get_selected_record_exchange(
        process_group=process_group,
        device=torch.device("cuda", 0),
        layout=layout,
        lane_key=("draft", 1),
    )

    assert target is not draft
    assert b12x_sparse_ckv_decode.get_selected_record_stream(target) is not (
        b12x_sparse_ckv_decode.get_selected_record_stream(draft)
    )
    assert calls == ["direct", "direct"]


def test_selected_record_transport_strict_ce_does_not_fallback(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "ce")
    calls, _, _ = _install_fake_b12x_transport(monkeypatch, ce_error="CE unavailable")

    with pytest.raises(RuntimeError, match="strict copy-engine"):
        b12x_sparse_ckv_decode.get_selected_record_exchange(
            process_group=object(),
            device=torch.device("cuda", 0),
            layout=_layout(),
            lane_key=("target", 1),
        )

    assert calls == ["ce"]


def test_selected_record_transport_rejects_unknown_mode(monkeypatch):
    _prepare_transport_factory_test(monkeypatch, "mystery")

    with pytest.raises(ValueError, match="must be auto, ce, or direct"):
        b12x_sparse_ckv_decode.get_selected_record_exchange(
            process_group=object(),
            device=torch.device("cuda", 0),
            layout=_layout(),
            lane_key=("target", 1),
        )


@pytest.mark.parametrize("dcp", [2, 3, 4, 5, 6, 7, 8])
def test_sparse_decode_layout_supports_dcp_two_through_eight(dcp):
    layout = _layout(dcp=dcp)

    assert layout.dcp_world_size == dcp
    assert layout.per_request_capacity == 8192
    assert layout.pool_records == 65536
    assert layout.max_fast_requests == 8
    assert layout.record_bytes == 432
    assert layout.workspace_slots == 4


@pytest.mark.parametrize("dcp", [1, 9])
def test_sparse_decode_layout_rejects_unsupported_dcp(dcp):
    with pytest.raises(ValueError, match="DCP world sizes 2-8"):
        _layout(dcp=dcp)


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8])
def test_sparse_decode_batch_policy_accepts_c1_through_c8(concurrency):
    layout = _layout()
    rows = concurrency * layout.rows_per_request

    assert sparse_decode_batch_eligible(
        layout,
        num_requests=concurrency,
        num_rows=rows,
        max_query_len=layout.rows_per_request,
        num_actual_tokens=rows,
        has_required_metadata=True,
    )
    assert layout.active_records(concurrency) == concurrency * 8192


def test_sparse_decode_pool_overflow_falls_back_before_transport():
    layout = _layout(pool=3 * 8192)

    assert layout.max_fast_requests == 3
    assert sparse_decode_batch_eligible(
        layout,
        num_requests=3,
        num_rows=12,
        max_query_len=4,
        num_actual_tokens=12,
        has_required_metadata=True,
    )
    assert not sparse_decode_batch_eligible(
        layout,
        num_requests=4,
        num_rows=16,
        max_query_len=4,
        num_actual_tokens=16,
        has_required_metadata=True,
    )
    with pytest.raises(ValueError, match="outside sparse capacity"):
        layout.active_records(4)


def test_mtp3_union_is_per_sequence_dense_stable_and_exact():
    indices = torch.tensor(
        [
            [
                [10, 11, 10, -1],
                [12, 11, 13, 10],
                [13, 14, 12, -1],
                [10, 15, 14, 15],
            ],
            [
                [10, 20, 10, -1],
                [21, 20, 22, 10],
                [22, 23, 21, -1],
                [10, 24, 23, 24],
            ],
        ],
        dtype=torch.int32,
    )

    union, remap, counts = dense_union_remap_reference(indices)

    assert union[0, :6].tolist() == [10, 11, 12, 13, 14, 15]
    assert union[1, :6].tolist() == [10, 20, 21, 22, 23, 24]
    assert counts.tolist() == [6, 6]
    for request in range(indices.shape[0]):
        for row in range(indices.shape[1]):
            for column in range(indices.shape[2]):
                token = int(indices[request, row, column])
                slot = int(remap[request, row, column])
                if token < 0:
                    assert slot == -1
                else:
                    assert int(union[request, slot]) == token


def test_destination_unions_and_remaps_are_independent_and_exact():
    indices = torch.tensor(
        [
            [[[1, 2], [2, 3]], [[10, 11], [11, 12]]],
            [[[4, 5], [5, 6]], [[10, 13], [13, 14]]],
        ],
        dtype=torch.int32,
    )

    union, remap, counts = dense_union_remap_reference(indices)

    assert counts.tolist() == [[3, 3], [3, 3]]
    assert union[0, 0, :3].tolist() == [1, 2, 3]
    assert union[1, 0, :3].tolist() == [4, 5, 6]
    assert union[0, 1, :3].tolist() == [10, 11, 12]
    assert union[1, 1, :3].tolist() == [10, 13, 14]
    reconstructed = torch.gather(
        union.unsqueeze(-2).expand(*indices.shape[:-2], indices.shape[-2], -1),
        -1,
        remap.to(torch.int64),
    )
    assert torch.equal(reconstructed, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_union_accepts_runtime_rows_below_reserved_mtp_rows():
    device = torch.device("cuda")
    indices = torch.tensor([[[[1, 2]]], [[[3, 4]]]], dtype=torch.int32, device=device)
    union = torch.empty((2, 2, 8), dtype=torch.int32, device=device)
    remap = torch.empty((2, 2, 4, 2), dtype=torch.int32, device=device)
    counts = torch.empty((2, 2), dtype=torch.int32, device=device)
    hash_keys = torch.empty((2, 2, 16), dtype=torch.int32, device=device)
    hash_first = torch.empty_like(hash_keys)
    first_to_dense = torch.empty_like(union)

    build_dense_union_remap(
        indices,
        union,
        remap,
        counts,
        hash_keys,
        hash_first,
        first_to_dense,
        num_requests=1,
    )
    torch.cuda.synchronize()

    assert counts[:, 0].cpu().tolist() == [2, 2]
    assert union[0, 0, :2].cpu().tolist() == [1, 2]
    assert union[1, 0, :2].cpu().tolist() == [3, 4]
    assert remap[:, 0, 0, :].cpu().tolist() == [[0, 1], [0, 1]]


def test_union_preflight_caches_rank_consistent_success(monkeypatch):
    process_group = object()
    loads = 0
    reductions = 0
    monkeypatch.setattr(b12x_sparse_ckv_decode, "_UNION_PREFLIGHT_RESULTS", {})

    def load_extension():
        nonlocal loads
        loads += 1

    def all_reduce(status, *, op, group):
        nonlocal reductions
        reductions += 1
        assert int(status.item()) == 1
        assert op is torch.distributed.ReduceOp.MIN
        assert group is process_group

    monkeypatch.setattr(
        b12x_sparse_ckv_decode, "preload_dense_union_extension", load_extension
    )
    monkeypatch.setattr(b12x_sparse_ckv_decode.dist, "all_reduce", all_reduce)

    for _ in range(2):
        b12x_sparse_ckv_decode.preload_dense_union_extension_consistently(
            process_group, torch.device("cpu")
        )

    assert loads == 1
    assert reductions == 1


def test_union_preflight_propagates_remote_rank_failure(monkeypatch):
    process_group = object()
    reductions = 0
    monkeypatch.setattr(b12x_sparse_ckv_decode, "_UNION_PREFLIGHT_RESULTS", {})

    def all_reduce(status, *, op, group):
        nonlocal reductions
        reductions += 1
        status.zero_()

    monkeypatch.setattr(
        b12x_sparse_ckv_decode, "preload_dense_union_extension", lambda: None
    )
    monkeypatch.setattr(b12x_sparse_ckv_decode.dist, "all_reduce", all_reduce)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="another DCP rank"):
            b12x_sparse_ckv_decode.preload_dense_union_extension_consistently(
                process_group, torch.device("cpu")
            )

    assert reductions == 1


def test_selected_record_cleanup_closes_once_and_clears_registries(monkeypatch):
    class Exchange:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    first = Exchange()
    second = Exchange()
    monkeypatch.setattr(
        b12x_sparse_ckv_decode,
        "_SELECTED_RECORD_EXCHANGES",
        {("target",): first, ("target-alias",): first, ("draft",): second},
    )
    monkeypatch.setattr(
        b12x_sparse_ckv_decode,
        "_SELECTED_RECORD_STREAMS",
        {id(first): object(), id(second): object()},
    )
    monkeypatch.setattr(
        b12x_sparse_ckv_decode,
        "_UNION_PREFLIGHT_RESULTS",
        {(1, "cuda", 0): (True, "")},
    )

    b12x_sparse_ckv_decode.close_selected_record_exchanges()

    assert first.close_calls == 1
    assert second.close_calls == 1
    assert b12x_sparse_ckv_decode._SELECTED_RECORD_EXCHANGES == {}
    assert b12x_sparse_ckv_decode._SELECTED_RECORD_STREAMS == {}
    assert b12x_sparse_ckv_decode._UNION_PREFLIGHT_RESULTS == {}


def test_model_parallel_cleanup_runs_before_group_destroy(monkeypatch):
    events = []

    class Group:
        def destroy(self):
            events.append("group")

    monkeypatch.setattr(
        parallel_state,
        "_MODEL_PARALLEL_CLEANUP_HOOKS",
        [lambda: events.append("cleanup")],
    )
    monkeypatch.setattr(parallel_state, "_TP", Group())
    for name in (
        "_DCP",
        "_QUERY_SPLIT",
        "_DCP_CKV_PREFETCH",
        "_PCP",
        "_PP",
        "_DP",
        "_EP",
        "_EPLB",
    ):
        monkeypatch.setattr(parallel_state, name, None)

    parallel_state.destroy_model_parallel()

    assert events == ["cleanup", "group"]


@pytest.mark.parametrize("concurrency", [2, 4, 8])
def test_multi_sequence_union_never_aliases_request_local_positions(concurrency):
    rows = []
    for request in range(concurrency):
        base = request * 1000
        rows.append(
            [
                [0, base + 1, base + 2],
                [base + 2, base + 3, 0],
                [base + 3, base + 4, base + 1],
                [base + 4, base + 5, 0],
            ]
        )
    indices = torch.tensor(rows, dtype=torch.int32)

    union, remap, counts = dense_union_remap_reference(indices)

    assert counts.tolist() == [6] * concurrency
    pooled_union = union.reshape(-1)
    for request in range(concurrency):
        reconstructed = torch.gather(
            union[request].expand(indices.shape[1], -1),
            1,
            remap[request].to(torch.int64),
        )
        assert torch.equal(reconstructed, indices[request])
        pooled_slots = remap[request].to(torch.int64) + request * union.shape[-1]
        assert torch.equal(pooled_union[pooled_slots], indices[request])


def test_shared_layer_prefetch_is_exactly_full_to_s1_s2_s3():
    # GLM: Full 0-2, Shared 3-5, then Full 6.
    emits_topk = [True, True, True, False, False, False, True, False]

    assert sparse_decode_prefetch_targets(2, 3, emits_topk) == [3, 4, 5]
    assert sparse_decode_prefetch_targets(1, 3, emits_topk) == []
    assert sparse_decode_prefetch_targets(6, 3, emits_topk) == [7]


@pytest.mark.parametrize("dcp", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_owner_arithmetic_round_trips_global_token(dcp, interleave):
    for logical_token in range(1024):
        owner, local = owner_and_local_ordinal(
            logical_token,
            dcp_world_size=dcp,
            interleave=interleave,
        )
        group = local // interleave
        lane = local % interleave
        reconstructed = (group * dcp + owner) * interleave + lane

        assert 0 <= owner < dcp
        assert reconstructed == logical_token


def test_sparse_decode_requires_uniform_rows_and_complete_metadata():
    layout = _layout()

    assert not sparse_decode_batch_eligible(
        layout,
        num_requests=2,
        num_rows=7,
        max_query_len=4,
        num_actual_tokens=7,
        has_required_metadata=True,
    )
    assert not sparse_decode_batch_eligible(
        layout,
        num_requests=2,
        num_rows=8,
        max_query_len=4,
        num_actual_tokens=8,
        has_required_metadata=False,
    )
