# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction and execution tests for Qwen3.8-Flash-Next."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
import vllm.distributed.utils as distributed_utils
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module
import vllm.model_executor.offloader as offloader
import vllm.models.qwen3_8_flash_next.hyperconnection as hyperconnection_module
import vllm.models.qwen3_8_flash_next.model as model_module
import vllm.models.qwen3_8_flash_next.nvidia.qsa as qsa_module
import vllm.models.qwen3_8_flash_next.ple_layer as ple_layer_module
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    _resolve_gdn_decode_kernel,
)
from vllm.models.qwen3_8_flash_next.ple_layer import Qwen3_8FlashNextPLELayer
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

_ALIGNED_PAGE_SIZE_BYTES = 818_176
_ALIGNED_BLOCK_SIZE = 752


def test_ple_embedding_preparation_runs_materialized_state() -> None:
    binding = object()
    owner = SimpleNamespace(
        requires_disk_preparation=False,
        max_total_tokens=8,
        eos_token_id=7,
        _token_ids=torch.zeros(8, dtype=torch.int64),
        _query_start_loc=torch.zeros(2, dtype=torch.int32),
        _committed_history=torch.zeros((1, 3), dtype=torch.int64),
        _num_seqs=torch.zeros(1, dtype=torch.int32),
        _num_tokens=torch.zeros(1, dtype=torch.int32),
        _embedding_out=torch.zeros((8, 16), dtype=torch.bfloat16),
        _bind_embedding_state=lambda state, capacity: binding,
    )
    state = Mock()
    factory = ple_layer_module.Qwen3_8FlashNextNGramEmbedding._embedding_prepare_call(
        owner, 8
    )

    call = factory(state)
    call.run()

    state.run.assert_called_once_with(binding, token_count=1)


def test_ple_profile_without_attention_metadata_preserves_live_dataflow(
    monkeypatch,
) -> None:
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer._out = torch.full((5, 2, 3), float("nan"))
    residual = torch.empty((3, 2, 3))
    key = torch.empty_like(residual)
    value = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        ple_layer_module,
        "_b12x_module",
        lambda _name: pytest.fail("profile path invoked the stateful PLE kernel"),
    )

    layer._run_ple(
        residual,
        key,
        value,
        torch.tensor([0, 3], dtype=torch.int32),
    )

    torch.testing.assert_close(layer._out[:3], value[:, None, :].expand_as(residual))
    assert torch.count_nonzero(layer._out[3:]) == 0


def test_hyperconnection_consumes_projection_views_without_staging(monkeypatch):
    """Low-rank and injection readers share the immutable projection owner."""
    layer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(layer)
    layer.use_combine = True
    layer.lora_rank, layer.hc_count = 8, 4
    merged = torch.arange(64, dtype=torch.bfloat16).reshape(4, 16)
    layer.input_mix_weight_down_block_inject = lambda x: merged
    layer.input_mix_weight_up = lambda x: x
    layer._binding = lambda _state, _operation: binding
    normalized = torch.ones(4, 8)
    binding = object()
    seen = {}

    def silu(projected, *, binding):
        seen["down"] = projected
        return projected + 1

    api = SimpleNamespace(
        run_scaled_silu=silu,
        run_gate_mean=lambda normalized, logits, **kwargs: normalized + logits,
    )
    monkeypatch.setattr(hyperconnection_module, "_hyperconnection_api", lambda: api)
    block, injection = layer._mix_normalized(normalized)
    torch.testing.assert_close(block, normalized + merged[:, :8] + 1)
    torch.testing.assert_close(injection, merged[:, 8:12], rtol=0, atol=0)
    for view in (seen["down"], injection):
        assert view.untyped_storage().data_ptr() == merged.untyped_storage().data_ptr()
        assert view.stride(0) == merged.stride(0)


def test_hyperconnection_benchmark_reproduces_activation_inputs() -> None:
    layer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(params_dtype=torch.bfloat16)
    layer.hc_count = 2
    layer.hidden_size = 4
    layer.lora_rank = 3
    object.__setattr__(
        layer,
        "_workspace",
        SimpleNamespace(
            device=torch.device("cpu"),
            bottleneck=torch.empty((4, 3), dtype=torch.bfloat16),
        ),
    )

    call = layer._benchmark_call("scaled_silu", 4)(object())
    assert call.produce is not None
    activation, template = call.owners[:2]
    call.produce()
    torch.testing.assert_close(activation, template)
    activation.zero_()
    call.produce()
    torch.testing.assert_close(activation, template)


def test_final_hyperconnection_declares_the_combine_norm_it_consumes(monkeypatch):
    from vllm.utils.b12x import B12xWorkload

    class Declaration:
        def request(self, **kwargs):
            return SimpleNamespace(name=kwargs["name"])

    layer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(layer)
    layer.use_combine = False
    layer._preparation_prefix = "model.hyper_connection_mixer"
    layer.hc_norm = SimpleNamespace(weight=torch.empty(1))
    layer.config = SimpleNamespace(rms_norm_eps=1e-6)
    workspace = SimpleNamespace(max_tokens=8, caps=lambda _tokens: object())
    object.__setattr__(layer, "_workspace", workspace)
    monkeypatch.setattr(
        hyperconnection_module,
        "_hyperconnection_api",
        lambda: SimpleNamespace(plan=lambda *args, **kwargs: Declaration()),
    )
    workload = B12xWorkload(
        stage="weights", token_counts=(8,), fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=8, max_seqs=1, max_model_len=8,
    )

    (unit,) = layer.get_b12x_preparation_units(layer, workload)

    assert {request.name for request in unit.requests} == {
        f"model.hyper_connection_mixer.hc.{operation}.m8"
        for operation in (
            "grouped_rmsnorm",
            "scaled_silu",
            "gate_mean",
            "combine",
            "combine_norm",
        )
    }


def test_hyperconnection_declares_one_capacity_per_operation(monkeypatch):
    from vllm.utils.b12x import B12xWorkload

    class Declaration:
        def request(self, **kwargs):
            return SimpleNamespace(name=kwargs["name"])

    layer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(layer)
    layer.use_combine = True
    layer._preparation_prefix = "model.layers.0.hyperconnection"
    layer.hc_norm = SimpleNamespace(weight=torch.empty(1))
    layer.config = SimpleNamespace(rms_norm_eps=1e-6)
    workspace = SimpleNamespace(max_tokens=32, caps=lambda _tokens: object())
    object.__setattr__(layer, "_workspace", workspace)
    monkeypatch.setattr(
        hyperconnection_module,
        "_hyperconnection_api",
        lambda: SimpleNamespace(plan=lambda *args, **kwargs: Declaration()),
    )
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 2, 4, 16, 32), fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=32, max_seqs=1, max_model_len=32,
    )

    (unit,) = layer.get_b12x_preparation_units(layer, workload)

    assert len(unit.requests) == 5
    assert all(request.name.endswith(".m32") for request in unit.requests)


def test_mtp_compaction_selector_reuses_aot_callable_across_draft_phases():
    from vllm.models.qwen3_8_flash_next.mtp import (
        Qwen3_8FlashNextMultiTokenPredictor,
    )

    predictor = SimpleNamespace(
        _decode_output_indices=torch.zeros(1, dtype=torch.int64)
    )
    select = Qwen3_8FlashNextMultiTokenPredictor.set_prefill_output_indices
    source = torch.arange(32).reshape(4, 8)
    tail = torch.tensor([3])
    select(predictor, tail)

    def compact(x):
        return x[predictor._prefill_output_indices]

    compiled = torch.compile(compact, fullgraph=True).aot_compile(((source,), {}))
    for index in (3, 1, 2):
        tail.fill_(index)
        select(predictor, tail)
        torch.testing.assert_close(compiled(source), source[index : index + 1])
        select(predictor, None)
        torch.testing.assert_close(compiled(source), source[:1])


def test_mtp_feedback_candidates_share_bounded_trial_storage(monkeypatch):
    from vllm.models.qwen3_8_flash_next.mtp import (
        Qwen3_8FlashNextMultiTokenPredictor,
    )

    class FakeWorkspaceManager:
        def __init__(self):
            self.buffer = torch.empty(64, dtype=torch.uint8)

        def get_simultaneous(self, *specs):
            assert specs == (((64,), torch.uint8),)
            return [self.buffer]

    def state():
        calls = []
        layout = SimpleNamespace(
            caps=SimpleNamespace(max_tokens=4),
            scratch_specs=lambda: (
                SimpleNamespace(
                    shape=(64,), dtype=torch.uint8, device=torch.device("cpu")
                ),
            ),
            output_storage_shape=lambda: (4, 16),
        )

        def run_tensors(*args, **kwargs):
            calls.append((args, kwargs))

        return SimpleNamespace(layout=layout, run_tensors=run_tensors), calls

    manager = FakeWorkspaceManager()
    monkeypatch.setattr(
        "vllm.v1.worker.workspace.current_workspace_manager", lambda: manager
    )
    predictor = Qwen3_8FlashNextMultiTokenPredictor.__new__(
        Qwen3_8FlashNextMultiTokenPredictor
    )
    nn.Module.__init__(predictor)
    predictor.hidden_size = 16
    predictor.hc_count = 2
    predictor.config = SimpleNamespace(rms_norm_eps=1e-6)
    for name in (
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "fc_embedding",
        "fc_hidden",
    ):
        setattr(predictor, name, SimpleNamespace(weight=torch.empty(1)))

    factory = predictor._feedback_benchmark_factory()
    first_state, first_runs = state()
    second_state, second_runs = state()
    first_call = factory(first_state)
    second_call = factory(second_state)
    first_call.produce()
    second_call.produce()
    first_call.run()
    second_call.run()

    first_args = first_runs[0][0]
    second_args = second_runs[0][0]
    assert first_args[0] is second_args[0]
    assert first_args[1] is second_args[1]
    assert first_args[6] is second_args[6]
    assert first_args[7] is second_args[7]
    assert not first_call.capture_safe
    assert not second_call.capture_safe


@pytest.mark.parametrize("indices", [[0], [3], [0, 3]])
def test_mtp_compaction_preserves_attention_rows_and_selected_outputs(indices):
    """Cache-producing attention sees every row; tokenwise MLP sees only tails."""
    layer = model_module.Qwen3_8FlashNextDecoderLayer.__new__(
        model_module.Qwen3_8FlashNextDecoderLayer
    )
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = "full_attention"
    rows = {}

    def attention(*, hidden_states, positions):
        rows["attention"] = hidden_states.clone()
        return hidden_states + positions[:, None]

    def mlp(x):
        rows["mlp"] = x.shape[0]
        return x.square()

    layer.attn_hyper_connection = SimpleNamespace(mix=lambda x: (x, x * 2, x / 2))
    layer.mlp_hyper_connection = SimpleNamespace(
        combine_and_mix=lambda x, attn, injection: (x + attn, attn + injection, x)
    )
    layer.self_attn = attention
    layer.mlp = mlp
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    kwargs = dict(
        hidden_states=hidden,
        prev_block_output=None,
        prev_injection=None,
        positions=torch.arange(4),
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )
    expected = layer(**kwargs)
    selection = torch.tensor(indices)
    actual = layer(**kwargs, output_indices=selection)
    torch.testing.assert_close(rows["attention"], hidden * 2)
    assert rows["mlp"] == len(indices)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference[selection], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
@pytest.mark.parametrize("kind", ["gdn", "qsa"])
def test_attention_projection_overlap_replays_with_changed_inputs(
    monkeypatch, num_tokens: int, kind: str
) -> None:
    """The fork/join must precede consumers on every graph replay."""
    torch.manual_seed(42)
    qkvz = nn.Linear(2560, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    ba = nn.Linear(2560, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        in_proj_qkvz=lambda x: (qkvz(x), None),
        in_proj_ba=lambda x: (ba(x), None),
        qkv_proj=lambda x: (qkvz(x), None),
        indexer=SimpleNamespace(index_qk_proj=lambda x: (ba(x), None)),
    )
    module = gdn_module if kind == "gdn" else qsa_module
    op = (
        torch.ops.vllm.qwen_gdn_input_projections
        if kind == "gdn"
        else torch.ops.vllm.qwen3_8_flash_next_qsa_input_projections
    )
    stream = torch.cuda.Stream()
    monkeypatch.setattr(module, "aux_stream", lambda: stream)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test.gdn": layer}),
    )

    @torch.compile(backend="eager", fullgraph=True)
    def project(x):
        qkvz_out, ba_out = op(x, 512, 64, "test.gdn")
        return qkvz_out, ba_out.sigmoid()

    x = torch.randn(num_tokens, 2560, device="cuda", dtype=torch.bfloat16)
    main_stream = torch.cuda.Stream()
    main_stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(main_stream):
        for _ in range(3):
            project(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main_stream):
            actual = project(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            expected = (qkvz(x), ba(x).sigmoid())
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("kind", ["gdn", "qsa"])
@torch.inference_mode()
def test_b12x_projection_overlap_preserves_scratch(monkeypatch, num_tokens, kind):
    """Concurrent projections must not overwrite activation or split-K scratch."""
    pytest.importorskip("b12x")
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession
    from vllm.model_executor.kernels.linear.b12x_blockscaled import (
        B12xBlockscaledLinear,
    )
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )

    torch.manual_seed(43)
    device = torch.device("cuda", torch.cuda.current_device())
    if not blockscaled.is_supported(device):
        pytest.skip("requires b12x block-scaled kernels on SM120/SM121")
    init_workspace_manager(device)
    module = gdn_module if kind == "gdn" else qsa_module
    op = (
        torch.ops.vllm.qwen_gdn_input_projections
        if kind == "gdn"
        else torch.ops.vllm.qwen3_8_flash_next_qsa_input_projections
    )
    widths = (4096, 24) if kind == "gdn" else (3584, 640)
    x = torch.randn(num_tokens, 2560, device=device, dtype=torch.bfloat16)
    linears = []
    configs = (
        blockscaled.BlockscaledConfig(mode="quantized")
        if num_tokens == 1
        else blockscaled.BlockscaledConfig(
            mode="a16", tile_n=64, tile_k=64, split_k=2
        ),
        blockscaled.BlockscaledConfig(
            mode="a16", tile_n=64, tile_k=128, split_k=4
        ),
    )
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            for n, config in zip(widths, configs):
                weight = (torch.randn(n, 2560, device=device) * 0.05).to(
                    torch.float8_e4m3fn
                )
                scales = torch.full((n, 80), 127, device=device, dtype=torch.uint8)
                packed = blockscaled.pack_weight(weight, scales)
                holder = B12xBlockscaledLinear(
                    packed, recipe="mxfp8", activation_mode="auto",
                    layer_name=f"projection.{n}",
                )
                query = blockscaled.BlockscaledQuery(
                    recipe="mxfp8", num_tokens=num_tokens,
                    in_features=2560, padded_in_features=2560,
                    out_features=n, expected_m=num_tokens,
                    workspace_form="provided",
                )
                holder.plan = blockscaled.plan(query, override=config)
                session.prepare((holder.plan.request(
                    name=str(n), prepare_call=holder._call_factory(num_tokens),
                ),))
                linears.append(holder)
            session.freeze()

            class Projection:
                def __init__(self, holder):
                    self.b12x_linear = holder

                def __call__(self, value):
                    return self.b12x_linear.run(value, None), None

            first, second = map(Projection, linears)
            layer = SimpleNamespace(
                in_proj_qkvz=first, in_proj_ba=second, qkv_proj=first,
                indexer=SimpleNamespace(index_qk_proj=second),
            )
            monkeypatch.setattr(
                module, "get_forward_context",
                lambda: SimpleNamespace(no_compile_layers={"test.projection": layer}),
            )
            monkeypatch.setattr(module, "aux_stream", lambda: side_stream)
            side_stream = torch.cuda.Stream()
            main_stream = torch.cuda.Stream()
            main_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(main_stream):
                expected = op(x, *widths, "test.projection")
                assert all(value.isfinite().all() for value in expected)
                assert all(value.count_nonzero() for value in expected)
                manager = current_workspace_manager()
                manager.lock()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=main_stream):
                    actual = op(x, *widths, "test.projection")
                try:
                    for _ in range(3):
                        x.normal_()
                        reference = tuple(
                            value.clone() for value in op(x, *widths, "test.projection")
                        )
                        allocations = torch.cuda.memory_stats(device)[
                            "allocation.all.allocated"
                        ]
                        graph.replay()
                        torch.cuda.synchronize(device)
                        assert torch.cuda.memory_stats(device)[
                            "allocation.all.allocated"
                        ] == allocations
                        for value, ref in zip(actual, reference):
                            torch.testing.assert_close(value, ref, rtol=0, atol=0)
                finally:
                    del graph
    finally:
        torch.cuda.synchronize(device)
        reset_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [4, 32])
def test_ple_prefetch_joins_before_embedding_consumers(monkeypatch, num_tokens) -> None:
    embedding = ple_layer_module.Qwen3_8FlashNextNGramEmbedding.__new__(
        ple_layer_module.Qwen3_8FlashNextNGramEmbedding
    )
    nn.Module.__init__(embedding)
    embedding.owner_prefix = "test.ple"
    embedding.requires_disk_preparation = False
    embedding._embedding_out = torch.empty(32, 32, device="cuda")
    table = torch.randn(512, 32, device="cuda")

    def lookup(ids, query_start_loc, history):
        rows = (ids + history[0, 0]) % table.shape[0]
        embedding._embedding_out[: ids.numel()].copy_(table[rows])

    embedding._run_embedding = lookup
    stream = torch.cuda.Stream()
    monkeypatch.setattr(ple_layer_module, "_get_prefetch_stream", lambda: stream)
    monkeypatch.setattr(
        ple_layer_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"test.ple": SimpleNamespace(ple_embedding=embedding)}
        ),
    )

    @torch.compile(fullgraph=True)
    def forward(ids, query_start_loc, history, hidden):
        embedding.prefetch(ids, query_start_loc, history)
        hidden = hidden * 2
        return embedding(ids, query_start_loc, history, wait_for=hidden) + hidden

    ids = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")
    history = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    hidden = torch.randn(num_tokens, 32, device="cuda")
    main_stream = torch.cuda.Stream()
    main_stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(main_stream):
        for _ in range(3):
            forward(ids, query_start_loc, history, hidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main_stream):
            actual = forward(ids, query_start_loc, history, hidden)
        for _ in range(3):
            ids.random_(0, 512)
            history.random_(0, 512)
            hidden.normal_()
            graph.replay()
            expected = table[(ids + history[0, 0]) % 512] + hidden * 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disk_ple_preparation_refreshes_graph_output(tmp_path, monkeypatch) -> None:
    """Real file reads precede replay, including accepted history and padding."""
    pytest.importorskip("b12x.sequence.ple_embedding")
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_utils import (
        file_source_tensor,
        safetensors_file_sources,
    )
    from vllm.models.qwen3_8_flash_next.model_state import (
        Qwen3_8FlashNextModelState,
    )
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    monkeypatch.setattr(
        ple_layer_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(ple_layer_module, "get_tensor_model_parallel_rank", lambda: 0)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=0,
        split_ngram_parts=4,
        ple_embedding_dtype="bfloat16",
        vocab_size=128,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=16,
    )

    def make_embedding(memory):
        return ple_layer_module.Qwen3_8FlashNextNGramEmbedding(
            config,
            64,
            0,
            32,
            2,
            f"test.{memory}.ple",
            f"test.{memory}.embedding",
            torch.bfloat16,
            memory,
        )

    embedding = make_embedding("io_uring")
    resident = make_embedding("device")
    rows = embedding._table_layout.padded_vocab_size
    table = torch.arange(rows * 32).reshape(rows, 32).remainder(251).to(torch.bfloat16)
    shard_rows = (rows + 3) // 4
    weights = {
        f"ngram_embedding.shard_{index}.weight": table[
            index * shard_rows : min((index + 1) * shard_rows, rows)
        ].contiguous()
        for index in range(4)
    }
    path = tmp_path / "ple.safetensors"
    save_file(weights, str(path))
    embedding.load_weights(
        (name, file_source_tensor(source))
        for name, source in safetensors_file_sources(str(path)).items()
    )
    resident.load_weights(weights.items())
    ple_layer_module.flush_weight_transfers()
    from b12x.preparation import PreparationSession
    from vllm.utils.b12x import B12xWorkload

    counts = (1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32)
    workload = B12xWorkload(
        stage="weights", token_counts=counts, fixed_token_counts=(),
        output_dtype=torch.bfloat16, max_tokens=32, max_seqs=2, max_model_len=32,
    )
    session = PreparationSession(device="cuda")
    for candidate in (embedding, resident):
        units = candidate.get_b12x_preparation_units(candidate, workload)
        requests = tuple(request for unit in units for request in unit.requests)
        session.prepare(requests)

    state = Qwen3_8FlashNextModelState.__new__(Qwen3_8FlashNextModelState)
    state.uses_ngram_embedding = True
    state.disk_embeddings = (embedding,)
    state.ngram_context = torch.empty(2, 2, dtype=torch.int64, device="cuda")
    state.ngram_context_offsets = torch.tensor([-2, -1], device="cuda")
    state.ngram_eos_token_id = 0
    state.ple_query_start_loc = torch.empty(3, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(MambaHybridModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(MambaHybridModelState, "prepare_dummy_inputs", lambda *args: {})
    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=2,
        input_ids=torch.zeros(4, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32, device="cuda"),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([2], dtype=torch.int32, device="cuda")
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[5, 9, 13, 17, 19, 23]], device="cuda")
        ),
    )
    with pytest.raises(RuntimeError, match="not prepared"):
        embedding(batch.input_ids, batch.query_start_loc, state.ngram_context)

    @torch.compile(backend="eager", fullgraph=True)
    def consume(ids, query_start_loc, ngram_context):
        return embedding(ids, query_start_loc, ngram_context) + 1

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        dummy = state.prepare_dummy_inputs(2, 4)
        for _ in range(3):
            consume(batch.input_ids, **dummy)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = consume(batch.input_ids, **dummy)
        graph.replay()
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
        for ids, accepted, live in (
            ([11, 12, 99, 99], 2, 2),
            ([21, 22, 23, 99], 3, 3),
            ([31, 99, 99, 99], 1, 1),
        ):
            batch.input_ids.copy_(torch.tensor(ids, dtype=torch.int32, device="cuda"))
            batch.query_start_loc[1:].fill_(live)
            req_states.num_computed_tokens.gpu.fill_(accepted)
            prepared = state.prepare_inputs(batch, req_states)
            graph.replay()
            expected = resident(batch.input_ids, **prepared) + 1
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual[live:], torch.ones_like(actual[live:]), rtol=0, atol=0
            )
        # Changing batch sizes must not specialize on Python preparation counts.
        for count in (4, 8, 3, 7, 2, 6, 12, 16, 24, 32):
            batch.input_ids = torch.arange(count, dtype=torch.int32, device="cuda")
            batch.query_start_loc[1:].fill_(count)
            prepared = state.prepare_inputs(batch, req_states)
            expected = resident(batch.input_ids, **prepared) + 1
            torch.testing.assert_close(
                consume(batch.input_ids, **prepared), expected, rtol=0, atol=0
            )
    torch.cuda.current_stream().wait_stream(stream)


class _RecordingPlan:
    def __init__(self) -> None:
        self.bind_kwargs: dict[str, Any] | None = None
        self.request_kwargs: dict[str, Any] | None = None
        self.binding = object()

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.binding

    def request(self, **kwargs):
        self.request_kwargs = kwargs
        return ("request", kwargs["name"])

    def scratch_specs(self):
        return (
            SimpleNamespace(
                shape=(1,),
                dtype=torch.uint8,
                device=torch.device("cpu"),
            ),
        )


def _allocate_aligned_mamba_cache(
    *,
    layer_name: str,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[torch.dtype, ...],
    mamba_type: MambaAttentionBackendEnum,
    num_blocks: int = 2,
) -> torch.Tensor:
    spec = MambaSpec(
        shapes=shapes,
        dtypes=dtypes,
        block_size=_ALIGNED_BLOCK_SIZE,
        page_size_padded=_ALIGNED_PAGE_SIZE_BYTES,
        mamba_type=mamba_type,
        num_speculative_blocks=2,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                layers=[layer_name],
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec([layer_name], spec)],
    )
    return allocate_kv_cache(
        config,
        torch.device("cpu"),
        KVCacheLayout.BLHNC,
    )[layer_name]


@pytest.mark.parametrize(
    ("enabled", "backend"), [("0", "device"), ("1", "mapped_host")]
)
def test_ple_cpu_offload_env_alias(monkeypatch, enabled, backend) -> None:
    monkeypatch.delenv("VLLM_PLE_TABLE_MEMORY", raising=False)
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", enabled)
    assert ple_layer_module._resolve_ple_table_memory(None) == backend


@pytest.mark.parametrize(
    ("policy", "legacy_flag", "backend"),
    [("ram", "0", "mapped_host"), ("disk", "1", "io_uring")],
)
def test_ple_table_memory_env_overrides_cpu_offload_flag(
    monkeypatch, policy, legacy_flag, backend
) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", legacy_flag)
    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", policy)
    assert ple_layer_module._resolve_ple_table_memory(None) == backend


def test_ple_table_memory_env_rejects_backend_names(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", "io_uring")
    with pytest.raises(ValueError, match="VLLM_PLE_TABLE_MEMORY"):
        ple_layer_module._resolve_ple_table_memory(None)


def test_qwen3_8_prefers_b12x_gdn_unless_explicitly_overridden(monkeypatch) -> None:
    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next_text")
        ),
    )
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    assert _resolve_gdn_decode_kernel(config) == ("b12x", False)
    config.model_config.hf_text_config.model_type = "qwen3_next"
    assert _resolve_gdn_decode_kernel(config) == ("cuda", False)

    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    config.model_config.hf_text_config.model_type = "qwen3_8_flash_next_text"
    assert _resolve_gdn_decode_kernel(config) == ("triton", True)

    config.additional_config["gdn_decode_kernel"] = "b12x"
    assert _resolve_gdn_decode_kernel(config) == ("b12x", True)


def test_explicit_ple_table_memory_overrides_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", "disk")
    assert (
        ple_layer_module._resolve_ple_table_memory({"ple_table_memory": "device"})
        == "device"
    )


def test_ple_registers_request_dependent_piecewise_splitting_ops_once() -> None:
    compilation_config = SimpleNamespace(
        static_forward_context={},
        splitting_ops=[],
    )

    ple_layer_module._register_ple_compilation_context(
        compilation_config,
        "model.layers.1.ple",
        nn.Identity(),
    )
    ple_layer_module._register_ple_compilation_context(
        compilation_config,
        "model.layers.5.ple",
        nn.Identity(),
    )

    assert compilation_config.splitting_ops == list(ple_layer_module._PLE_SPLITTING_OPS)
    assert set(compilation_config.static_forward_context) == {
        "model.layers.1.ple",
        "model.layers.5.ple",
    }




def test_decoder_layer_factory_accepts_make_layers_prefix(monkeypatch) -> None:
    created_layers: list[tuple[str, str]] = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeWorkspace(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeDecoderLayer(nn.Module):
        def __init__(
            self,
            _vllm_config,
            layer_type: str,
            _workspace,
            *,
            prefix: str,
        ) -> None:
            super().__init__()
            created_layers.append((prefix, layer_type))

    class FakePPGroup:
        rank_in_group = 0
        world_size = 1
        is_last_rank = False

    class FakeOffloader:
        @staticmethod
        def wrap_modules(modules):
            return list(modules)

    pp_group = FakePPGroup()
    monkeypatch.setattr(model_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(model_module, "HyperConnectionWorkspace", FakeWorkspace)
    monkeypatch.setattr(model_module, "Qwen3_8FlashNextDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        distributed_utils,
        "get_pp_indices",
        lambda num_layers, _rank, _world_size: (0, num_layers),
    )
    monkeypatch.setattr(offloader, "get_offloader", lambda: FakeOffloader())

    text_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
        indexer_n_heads=None,
        hc_count=4,
        hc_lowrank=2,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=None,
    )

    model = model_module.Qwen3_8FlashNextModel(
        vllm_config=vllm_config,
        prefix="model",
    )

    assert len(model.layers) == 2
    assert created_layers == [
        ("model.layers.0", "full_attention"),
        ("model.layers.1", "linear_attention"),
    ]


def test_ple_bind_uses_installed_execution_and_exact_aligned_page_stride(
    monkeypatch,
) -> None:
    shapes = ((10_240, 11),)
    dtypes = (torch.bfloat16,)
    raw_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.1.ple",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer._state_caps = object()
    layer._preparation_prefix = "model.layers.1.ple"
    plan = _RecordingPlan()
    layer._state_plans = {4: plan}
    layer.max_tokens = 4
    import vllm.v1.worker.workspace as workspace

    # No workspace manager is initialized on the host; get_b12x_scratch_buffers
    # falls back to allocating the plan's declared scratch directly.
    monkeypatch.setattr(workspace, "is_workspace_manager_initialized", lambda: False)
    layer._residual = torch.empty(4, 1, 1)
    layer._key = torch.empty(4, 1, 1)
    layer._value = torch.empty(4, 1)
    layer.norm_key = SimpleNamespace(weight=torch.empty(1))
    layer.norm_query = SimpleNamespace(weight=torch.empty(1))
    layer.norm_conv = SimpleNamespace(weight=torch.empty(1))
    layer.conv1d = SimpleNamespace(weight=torch.empty(1, 1, 1))
    layer._query_start_loc = torch.empty(1, dtype=torch.int32)
    layer._state_slot_ids = torch.empty(1, dtype=torch.int64)
    layer._state_is_fresh = torch.empty(1, dtype=torch.bool)
    layer._num_accepted_tokens = torch.empty(1, dtype=torch.int32)
    layer._num_seqs = torch.empty(1, dtype=torch.int32)
    layer._num_tokens = torch.empty(1, dtype=torch.int32)
    layer._out = torch.empty(4, 1, 1)
    layer._request_is_prefill = torch.empty(1, dtype=torch.bool)
    monkeypatch.setattr(layer, "get_state_shape", lambda: shapes)
    monkeypatch.setattr(layer, "get_state_dtype", lambda: dtypes)
    layer.kv_cache = (
        ple_layer_module.MambaBase.bind_kv_cache(layer, raw_cache)
        or layer.kv_cache[0],
    )
    calls = []
    monkeypatch.setattr(
        ple_layer_module,
        "_b12x_module",
        lambda _name: SimpleNamespace(
            bind=lambda bound_plan, **kwargs: calls.append((bound_plan, kwargs))
            or object()
        ),
    )

    layer._bind_ple(4)

    (conv_state,) = layer.kv_cache
    assert raw_cache.shape == (2, 1, 1, 225_280)
    assert conv_state.stride() == (409_088, 11, 1)
    assert calls[0][0] is plan
    assert calls[0][1]["conv_state"] is conv_state
    assert calls[0][1]["residual"].shape[0] == 4

    layer._bind_ple(3)
    assert calls[1][0] is plan
    assert calls[1][1]["residual"].shape[0] == 4
    assert calls[1][1]["out"].shape[0] == 4
    assert layer._state_plans == {4: plan}
    with pytest.raises(ValueError, match="exceeds capacity"):
        layer._bind_ple(5)


def test_ple_preparation_invocation_preserves_aligned_page_stride(
    monkeypatch,
) -> None:
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer._residual = torch.empty(4, 2, 3)
    layer._key = torch.empty_like(layer._residual)
    layer._value = torch.empty(4, 3)
    layer.norm_key = SimpleNamespace(weight=torch.empty(3))
    layer.norm_query = SimpleNamespace(weight=torch.empty(3))
    layer.norm_conv = SimpleNamespace(weight=torch.empty(3))
    layer.conv1d = SimpleNamespace(weight=torch.empty(6, 1, 2))
    layer._query_start_loc = torch.empty(2, dtype=torch.int32)
    layer._state_slot_ids = torch.empty(1, dtype=torch.int64)
    layer._state_is_fresh = torch.empty(1, dtype=torch.bool)
    layer._num_accepted_tokens = torch.empty(1, dtype=torch.int32)
    layer._request_is_prefill = torch.empty(1, dtype=torch.bool)
    layer._num_seqs = torch.empty(1, dtype=torch.int32)
    layer._num_tokens = torch.empty(1, dtype=torch.int32)
    layer._out = torch.empty_like(layer._residual)
    storage = torch.empty(2 * 64)
    layer.kv_cache = (torch.as_strided(storage, (2, 6, 4), (32, 4, 1)),)
    captured = {}

    def invocation_from_tensors(**tensors):
        captured.update(tensors)
        return {"state_strides": tuple(tensors["conv_state"].stride())}

    monkeypatch.setattr(
        ple_layer_module,
        "_b12x_module",
        lambda _name: SimpleNamespace(
            invocation_from_tensors=invocation_from_tensors,
        ),
    )

    invocation = layer._state_invocation(2)

    assert invocation == {"state_strides": (32, 4, 1)}
    assert captured["conv_state"] is layer.kv_cache[0]
    assert captured["residual"].shape[0] == 2


def test_b12x_gdn_bind_preserves_exact_aligned_page_stride(monkeypatch) -> None:
    shapes = ((2_560, 5), (12, 128, 128))
    dtypes = (torch.bfloat16, torch.float32)
    raw_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.0.linear_attn",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    plan = _RecordingPlan()

    def fake_bind(bound_plan, **kwargs):
        assert bound_plan is plan
        plan.bind_kwargs = kwargs
        return plan.binding

    bind_mock = Mock(wraps=fake_bind)
    planned_slots: list[int] = []

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "get_state_shape", lambda _: shapes)
    monkeypatch.setattr(QwenGatedDeltaNetAttention, "get_state_dtype", lambda _: dtypes)

    def make_plan(_self, max_state_slots: int):
        planned_slots.append(max_state_slots)
        return plan

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "_make_b12x_gdn_plan", make_plan)
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.gdn_decode_kernel = "b12x"
    layer.gdn_prefill_backend = "triton"
    layer._b12x_gdn_api = SimpleNamespace(bind=bind_mock)
    layer._b12x_prefill_api = None
    layer._b12x_decode_plan = None
    layer.A_log = nn.Parameter(torch.empty(0))
    layer.dt_bias = nn.Parameter(torch.empty(0))
    layer.norm = SimpleNamespace(weight=torch.empty(0))
    # The decode plan's declared scratch is drawn through
    # get_b12x_scratch_buffers, so _RecordingPlan.scratch_specs() suffices;
    # every remaining activation/metadata tensor comes from the staging
    # buffers a real bind_kv_cache would build lazily via
    # _ensure_b12x_gdn_decode_staging.
    staging = SimpleNamespace(
        mixed_qkv=torch.empty(0), a=torch.empty(0), b=torch.empty(0), z=torch.empty(0),
        output=torch.empty(0), query_start_loc=torch.empty(0),
        num_accepted_tokens=torch.empty(0), state_indices=torch.empty(0),
        num_seqs=torch.empty(0), num_tokens=torch.empty(0),
    )
    layer._b12x_decode_staging = staging

    layer.bind_kv_cache(raw_cache)
    bind_mock.assert_not_called()
    binding = layer._bind_b12x_gdn_decode()
    assert binding is plan.binding
    layer._bind_b12x_gdn_decode()
    assert bind_mock.call_count == 2

    conv_state, recurrent_state = layer.kv_cache
    assert conv_state.stride() == (409_088, 5, 1)
    assert recurrent_state.stride() == (204_544, 16_384, 128, 1)
    assert recurrent_state.storage_offset() == 6_400
    assert recurrent_state.data_ptr() == raw_cache.data_ptr() + 25_600
    assert not recurrent_state.is_contiguous()
    assert recurrent_state[0].is_contiguous()
    assert planned_slots == [2]
    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["recurrent_state"] is recurrent_state
    assert plan.bind_kwargs["mixed_qkv"] is staging.mixed_qkv
    assert not hasattr(layer, "_b12x_binding")

    layer.unbind_kv_cache()

    assert layer.kv_cache == ()
    assert layer._b12x_decode_plan is None
    with pytest.raises(RuntimeError, match="not prepared for the current KV generation"):
        layer._bind_b12x_gdn_decode()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_b12x_gdn_binds_live_projections_and_stages_rollback_metadata(
    monkeypatch,
) -> None:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._b12x_max_tokens = 6
    layer._b12x_max_seqs = 2
    layer._b12x_state_index_columns = 3
    plan = _RecordingPlan()
    bind_mock = Mock(wraps=plan.bind)
    monkeypatch.setattr(plan, "bind", bind_mock)
    layer._b12x_plan = plan
    layer._b12x_scratch = torch.empty(1, device="cuda")
    layer.A_log = torch.empty(2, device="cuda")
    layer.dt_bias = torch.empty(2, device="cuda")
    layer.norm = SimpleNamespace(weight=torch.empty(4, device="cuda"))
    layer.kv_cache = (torch.empty(0), torch.empty(10, 2, 4, 4, device="cuda"))
    layer.layer_norm_epsilon = 1e-6
    layer.head_k_dim = 128
    layer._b12x_mixed_qkv = torch.empty(6, 8, device="cuda")
    layer._b12x_a = torch.empty(6, 2, device="cuda")
    layer._b12x_b = torch.empty(6, 2, device="cuda")
    layer._b12x_z = torch.empty(6, 2, 4, device="cuda")
    layer._b12x_output = torch.full((6, 2, 4), 17.0, device="cuda")
    layer._b12x_query_start_loc = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    layer._b12x_num_accepted_tokens = torch.full(
        (2,), -1, dtype=torch.int32, device="cuda"
    )
    layer._b12x_state_indices = torch.full((2, 3), -1, dtype=torch.int32, device="cuda")
    layer._b12x_num_seqs = torch.zeros(1, dtype=torch.int32, device="cuda")
    layer._b12x_num_tokens = torch.zeros(1, dtype=torch.int32, device="cuda")
    calls: list[tuple[object, float, float]] = []

    def run(binding, *, eps: float, scale: float) -> None:
        calls.append((binding, eps, scale))
        assert plan.bind_kwargs is not None
        plan.bind_kwargs["output"].fill_(17.0)

    layer._b12x_gdn_api = SimpleNamespace(run=run)
    mixed_qkv = torch.arange(40, dtype=torch.float32, device="cuda").reshape(5, 8)
    a = torch.arange(10, dtype=torch.float32, device="cuda").reshape(5, 2)
    b = a + 20
    output_gate = torch.arange(40, dtype=torch.float32, device="cuda").reshape(5, 2, 4)
    state_indices = torch.tensor(
        [[7, 8, 9], [4, 5, 6]], dtype=torch.int32, device="cuda"
    )
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    core_attn_out = torch.zeros(5, 2, 4, device="cuda")

    layer._run_b12x_gdn_decode_post_conv(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        output_gate=output_gate,
        core_attn_out=core_attn_out,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        num_requests=2,
    )

    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["mixed_qkv"] is mixed_qkv
    assert plan.bind_kwargs["a"] is a
    assert plan.bind_kwargs["b"] is b
    assert plan.bind_kwargs["z"] is output_gate
    assert plan.bind_kwargs["output"].data_ptr() == core_attn_out.data_ptr()
    torch.testing.assert_close(layer._b12x_query_start_loc, query_start_loc)
    torch.testing.assert_close(layer._b12x_num_accepted_tokens, accepted)
    torch.testing.assert_close(layer._b12x_state_indices, state_indices)
    torch.testing.assert_close(
        layer._b12x_num_seqs, torch.tensor([2], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        layer._b12x_num_tokens, torch.tensor([5], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(core_attn_out, torch.full_like(core_attn_out, 17.0))
    assert calls == [(plan.binding, 1e-6, 128**-0.5)]
    bind_mock.assert_called_once()
    assert plan.bind_kwargs["recurrent_state"] is layer.kv_cache[1]
    assert plan.bind_kwargs["scratch"] is layer._b12x_scratch


@pytest.mark.parametrize("head_dim, expected", [(128, "flashinfer"), (64, "triton")])
def test_sm120_gdn_prefill_selects_supported_flashinfer_geometry(
    monkeypatch, head_dim, expected
):
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability=lambda _cap: False,
        is_device_capability_family=lambda cap: cap == 120,
        get_cuda_runtime_major=lambda: 13,
    )
    monkeypatch.setattr(gdn_module, "current_platform", platform)
    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=head_dim)
        ),
    )
    assert gdn_module._resolve_gdn_prefill_backend(config) == ("auto", expected)


def test_sm120_flashinfer_sequence_offsets_preserve_values(monkeypatch):
    monkeypatch.setattr(
        gdn_module,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda cap: cap == 120),
    )
    offsets = torch.tensor([0, 1, 129, 6019], dtype=torch.int32)
    converted = gdn_module._prepare_flashinfer_cu_seqlens(offsets)
    assert converted.dtype == torch.int64
    torch.testing.assert_close(converted, offsets.to(torch.int64))
    assert gdn_module._prepare_flashinfer_cu_seqlens(converted) is converted
    assert gdn_module._prepare_flashinfer_cu_seqlens(None) is None


@pytest.mark.skipif(
    not gdn_module.current_platform.is_device_capability_family(120),
    reason="requires SM12x FlashInfer GDN",
)
@pytest.mark.parametrize("boundaries", [[0, 129], [0, 1, 130, 259]])
def test_sm120_flashinfer_gdn_int32_offsets_match_int64(boundaries):
    """The scheduler's int32 offsets preserve the supported kernel result."""
    torch.manual_seed(312)
    rows = boundaries[-1]
    shape = (1, rows, 4, 128)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    gate_shape = (1, rows, 4)
    g = -torch.rand(gate_shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(gate_shape, device="cuda", dtype=torch.float32)
    state = torch.randn(len(boundaries) - 1, 4, 128, 128, device="cuda")
    offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int64)
    expected = gdn_module.fi_chunk_gated_delta_rule(
        q, k, v, g, beta, state.clone(), True, offsets
    )
    actual = gdn_module.fi_chunk_gated_delta_rule(
        q, k, v, g, beta, state.clone(), True, offsets.to(torch.int32)
    )
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all() and torch.count_nonzero(result)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_ple_embedding_reuses_capacity_for_live_token_counts(
    monkeypatch,
) -> None:
    layer = ple_layer_module.Qwen3_8FlashNextNGramEmbedding.__new__(
        ple_layer_module.Qwen3_8FlashNextNGramEmbedding
    )
    nn.Module.__init__(layer)
    layer.requires_disk_preparation = False
    layer.max_total_tokens = 128
    layer.owner_prefix = "model.layers.1.ple.ple_embedding"
    planned = _RecordingPlan()
    capacity = _RecordingPlan()
    layer._plans = {8: planned, 128: capacity}
    layer._scratch = torch.empty(1)
    layer.ngram_embedding = SimpleNamespace(
        weight=torch.empty(1), weight_scale=None, weight_scale_2=None, disk_table=None
    )
    layer._token_ids = torch.empty(128, dtype=torch.int64)
    layer._query_start_loc = torch.empty(2, dtype=torch.int32)
    layer._committed_history = torch.empty(1, 2, dtype=torch.int64)
    layer._num_seqs = torch.zeros(1, dtype=torch.int32)
    layer._num_tokens = torch.zeros(1, dtype=torch.int32)
    layer._embedding_out = torch.empty(128, 4)
    calls = []
    monkeypatch.setattr(
        ple_layer_module,
        "_b12x_module",
        lambda _name: SimpleNamespace(
            bind=lambda bound_plan, **kwargs: calls.append((bound_plan, kwargs))
            or object()
        ),
    )

    layer._bind_embedding(8)
    layer._bind_embedding(11)
    layer._bind_embedding(11)

    assert [call[0] for call in calls] == [planned, capacity, capacity]
    assert [call[1]["token_ids"].shape[0] for call in calls] == [8, 128, 128]
    assert [call[1]["out"].shape[0] for call in calls] == [8, 128, 128]
    assert layer._plans == {8: planned, 128: capacity}
    with pytest.raises(ValueError, match="exceeds capacity"):
        layer._bind_embedding(129)



def test_ple_state_prepare_call_restores_the_staging_buffers_it_overwrites(monkeypatch):
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer.eps = 1e-6
    layer._state_caps = object()
    layer.kv_cache = (torch.arange(6.0).reshape(2, 3),)
    layer._residual = torch.full((4, 1, 2), 7.0)
    layer._key = torch.full((4, 1, 2), 8.0)
    layer._value = torch.full((4, 2), 9.0)
    layer._out = torch.full((4, 1, 2), 3.0)
    layer._query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    layer._state_slot_ids = torch.tensor([1, 0], dtype=torch.int64)
    layer._state_is_fresh = torch.tensor([False, True])
    layer._num_accepted_tokens = torch.tensor([2, 3], dtype=torch.int32)
    layer._request_is_prefill = torch.tensor([True])
    layer._num_seqs = torch.tensor([2], dtype=torch.int32)
    layer._num_tokens = torch.tensor([4], dtype=torch.int32)
    layer.norm_key = SimpleNamespace(weight=torch.empty(1))
    layer.norm_query = SimpleNamespace(weight=torch.empty(1))
    layer.norm_conv = SimpleNamespace(weight=torch.empty(1))
    layer.conv1d = SimpleNamespace(weight=torch.empty(1, 1, 1))
    live = {
        name: getattr(layer, name).clone()
        for name in (
            "_residual", "_key", "_value", "_out", "_query_start_loc", "_state_slot_ids",
            "_state_is_fresh", "_num_accepted_tokens", "_request_is_prefill",
            "_num_seqs", "_num_tokens",
        )
    }
    conv_state = layer.kv_cache[0].clone()
    runs = []
    state = SimpleNamespace(
        layout=SimpleNamespace(scratch_specs=lambda: (
            SimpleNamespace(shape=(1,), dtype=torch.uint8, device=torch.device("cpu")),
        )),
        bind=lambda **kwargs: kwargs,
        run=lambda binding, **kwargs: (
            runs.append(kwargs), binding["conv_state"][0].fill_(-1.0), binding["out"].fill_(5.0)
        ),
    )

    call = layer._state_prepare_call(4)(state)
    call.reset()
    assert float(layer._residual.min()) == 1.0 and int(layer._num_seqs) == 1
    call.run()
    assert runs == [{"eps": 1e-6, "token_count": 4}]
    call.restore()

    for name, saved in live.items():
        torch.testing.assert_close(getattr(layer, name), saved, msg=name)
    torch.testing.assert_close(layer.kv_cache[0], conv_state)
