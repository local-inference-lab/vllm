# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CompilationMode
from vllm.config.vllm import VllmConfig
from vllm.v1.kv_cache_interface import KVarNFullAttentionSpec, MLAAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_kvarn_mla_live_block_trackers,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestDraftModelSpeculator(DraftModelSpeculator):
    def load_draft_model(self, *args, **kwargs):
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        raise NotImplementedError

    def capture(self, *args, **kwargs):
        raise NotImplementedError

    def init_cudagraph_manager(self, *args, **kwargs):
        raise NotImplementedError


class _MetadataBuilder:
    supports_exact_metadata_reuse = False

    def build(self, *, common_attn_metadata, **kwargs):
        return common_attn_metadata


class _AttentionGroup:
    kv_cache_spec = object()

    def __init__(self, layer_name: str) -> None:
        self.layer_names = [layer_name]

    def get_metadata_builder(self, index: int) -> _MetadataBuilder:
        assert index == 0
        return _MetadataBuilder()


@pytest.mark.parametrize(
    ("cache_dtype", "expected"),
    [
        ("kvarn_mla_k5_g64", []),
        ("kvarn_k5v5_g64", ["KVarN KV cache"]),
        ("kvarn_k4v2_g128", ["KVarN KV cache"]),
    ],
)
def test_v2_gate_admits_only_mla_kvarn(cache_dtype: str, expected: list[str]) -> None:
    config = SimpleNamespace(
        model_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            distributed_executor_backend="mp",
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            kv_sharing_fast_prefill=False,
        ),
        ec_transfer_config=None,
    )

    assert VllmConfig._get_v2_model_runner_unsupported_features(config) == expected


def test_tracker_initialization_selects_only_mla_kvarn_cache_groups() -> None:
    mla_kvarn = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="kvarn_mla_k5_g64",
    )
    mla_ordinary = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    generic_kvarn_v1 = KVarNFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tile_size=1024,
        quant_group_size=64,
    )
    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=mla_kvarn),
            SimpleNamespace(kv_cache_spec=mla_ordinary),
            SimpleNamespace(kv_cache_spec=generic_kvarn_v1),
        ]
    )

    trackers = init_kvarn_mla_live_block_trackers(
        cache_config,
        dcp_size=2,
        dcp_rank=1,
        cp_interleave=1,
    )

    assert set(trackers) == {0}
    assert trackers[0].group_configs[0] == (64, 128, 1, 2, 1, 1)


def test_target_metadata_propagates_physical_fills_per_cache_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attn_utils, "exact_attention_metadata_cache_key", lambda *args: ()
    )
    fills = [{0: 64, 1: None}, None, {8: 32, 9: 17}]
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(dcp_replicated=False))
            for _ in fills
        ]
    )

    metadata = build_attn_metadata(
        attn_groups=[
            [_AttentionGroup("target.group0")],
            [_AttentionGroup("target.group1")],
            [_AttentionGroup("target.group2")],
        ],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([65], dtype=torch.int32),
        max_seq_len=65,
        block_tables=[
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([[4, 5]], dtype=torch.int32),
            torch.tensor([[8, 9]], dtype=torch.int32),
        ],
        slot_mappings=torch.tensor([[64], [0], [32]], dtype=torch.int64),
        kv_cache_config=kv_cache_config,
        kvarn_mla_block_fills=fills,
    )

    assert metadata["target.group0"].kvarn_mla_block_fills is fills[0]
    assert metadata["target.group1"].kvarn_mla_block_fills is None
    assert metadata["target.group2"].kvarn_mla_block_fills is fills[2]


def test_draft_metadata_receives_same_physical_fills_as_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {"draft": kwargs["kvarn_mla_block_fills"]}

    monkeypatch.setattr(
        speculator_module, "build_attn_metadata", fake_build_attn_metadata
    )
    speculator = object.__new__(_TestDraftModelSpeculator)
    speculator.rebuild_prefill_attn_metadata = False
    speculator.arange = torch.arange(2, dtype=torch.int32)
    speculator.block_tables = SimpleNamespace(
        input_block_tables=[
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[8]], dtype=torch.int32),
        ],
        slot_mappings=torch.tensor([[0], [8]], dtype=torch.int64),
        cp_size=1,
    )
    speculator.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    speculator.attn_groups = [[], []]
    speculator.draft_max_seq_len = 1
    speculator.kv_cache_config = SimpleNamespace()
    target_fills = ({0: 1}, {8: None})

    speculator.set_kvarn_mla_block_fills(target_fills)
    metadata = speculator._build_draft_attn_metadata(
        num_reqs=1,
        num_reqs_padded=1,
        num_tokens_padded=1,
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
        step=1,
    )

    assert speculator.rebuild_prefill_attn_metadata
    assert metadata == {"draft": target_fills}
    assert captured["kvarn_mla_block_fills"] == target_fills


class _OwnershipTracker:
    def __init__(self, group_id: int, fills: dict[int, int | None]) -> None:
        self.group_id = group_id
        self.fills = fills
        self.pending_blocks: dict[str, object] = {}
        self.updated = False

    def block_fills(self, group_id: int) -> dict[int, int | None]:
        assert group_id == self.group_id
        return self.fills

    def update(self, requests, scheduled, removed, skipped, rollback_tokens) -> None:
        assert set(requests) == {"request:1"}
        assert scheduled == {"request:1": 1}
        assert rollback_tokens == {"request:1": 0}
        self.updated = True


def test_runner_propagates_identical_group_fills_to_target_and_draft() -> None:
    group0 = _OwnershipTracker(0, {0: 64})
    group2 = _OwnershipTracker(2, {8: None})
    request = model_runner_module._KVarNMLARequestState(
        req_id="request",
        generation=1,
        block_ids=([0], [4], [8]),
        num_computed_tokens=64,
    )
    speculator = object.__new__(_TestDraftModelSpeculator)
    speculator.rebuild_prefill_attn_metadata = False
    runner = object.__new__(GPUModelRunner)
    runner._kvarn_mla_live_block_trackers = {0: group0, 2: group2}
    runner._kvarn_mla_requests = {"request": request}
    runner._kvarn_mla_removed_tracker_keys = set()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()] * 3)
    runner.num_speculative_steps = 0
    runner.speculator = speculator
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request": 1},
        scheduled_spec_decode_tokens={},
    )

    runner._update_kvarn_mla_ownership(scheduler_output)

    expected = ({0: 64}, None, {8: None})
    assert group0.updated and group2.updated
    assert runner._kvarn_mla_target_block_fills == expected
    assert speculator.kvarn_mla_block_fills == expected


class _Tracker:
    def __init__(self) -> None:
        self.resolved: list[tuple[str, int]] = []

    def resolve_async(self, key, request, actual_end_tokens) -> None:
        self.resolved.append((key, actual_end_tokens))


def _make_resolution_runner() -> tuple[
    GPUModelRunner, _Tracker, model_runner_module._KVarNMLARequestState
]:
    tracker = _Tracker()
    request = model_runner_module._KVarNMLARequestState(
        req_id="request",
        generation=7,
        block_ids=([0, 1],),
        num_computed_tokens=61,
        rollback_tokens=4,
        ownership_step=3,
        ownership_start_tokens=61,
        ownership_scheduled_tokens=5,
    )
    runner = object.__new__(GPUModelRunner)
    runner._kvarn_mla_live_block_trackers = {0: tracker}
    runner.num_speculative_steps = 4
    runner._kvarn_mla_requests = {"request": request}
    return runner, tracker, request


def test_resolution_callback_is_disabled_without_speculation() -> None:
    runner, _, _ = _make_resolution_runner()
    runner.num_speculative_steps = 0

    assert runner._make_kvarn_mla_resolution_callback(["request"]) is None


@pytest.mark.parametrize("stale_dimension", ["generation", "ownership_step"])
def test_resolution_callback_ignores_stale_request_state(stale_dimension: str) -> None:
    runner, tracker, request = _make_resolution_runner()
    callback = runner._make_kvarn_mla_resolution_callback(["request"])
    assert callback is not None
    if stale_dimension == "generation":
        runner._kvarn_mla_requests["request"] = (
            model_runner_module._KVarNMLARequestState(
                req_id="request",
                generation=8,
                block_ids=([8, 9],),
                num_computed_tokens=0,
            )
        )
    else:
        request.ownership_step += 1

    callback(np.array([1], dtype=np.int32), np.array([4], dtype=np.int32))

    assert tracker.resolved == []


class _CompletedEvent:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


def _make_async_output(completion_callback) -> AsyncOutput:
    output = object.__new__(AsyncOutput)
    output.copy_event_recorded = True
    output.copy_event = _CompletedEvent()
    output.completion_callback = completion_callback
    output.completion_lock = Lock()
    output.num_rejected_tokens_np = np.array([3], dtype=np.int32)
    output.num_sampled_tokens_np = np.array([2], dtype=np.int32)
    output.sampled_token_ids = np.array([[101, 102]], dtype=np.int64)
    output.model_runner_output = ModelRunnerOutput(
        req_ids=["request"],
        req_id_to_index={"request": 0},
    )
    output.draft_token_ids_np = None
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output.routed_experts_cpu = None
    output._has_fault = None
    return output


def test_synchronous_get_output_resolves_pending_ownership_once() -> None:
    runner, tracker, request = _make_resolution_runner()
    callback = runner._make_kvarn_mla_resolution_callback(["request"])
    assert callback is not None
    output = _make_async_output(callback)

    model_output = output.get_output()

    assert output.copy_event.synchronized
    assert model_output.sampled_token_ids == [[101, 102]]
    assert tracker.resolved == [("request:7", 63)]
    assert request.rollback_tokens == 0
    assert output.completion_callback is None


def test_async_next_step_resolves_before_ownership_update() -> None:
    runner, tracker, request = _make_resolution_runner()
    callback = runner._make_kvarn_mla_resolution_callback(["request"])
    assert callback is not None
    output = _make_async_output(callback)
    runner._kvarn_mla_pending_resolution = output.resolve_completion

    runner._resolve_pending_kvarn_mla_output()

    assert output.copy_event.synchronized
    assert tracker.resolved == [("request:7", 63)]
    assert request.rollback_tokens == 0
    assert runner._kvarn_mla_pending_resolution is None

    output.get_output()
    assert tracker.resolved == [("request:7", 63)]
