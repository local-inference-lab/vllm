# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED orchestration: full encoder context, bounded decoder, original output ABI.

The real model/layer forwards drive a small deterministic mathematical model;
attention kernels and mapped metadata are qualified by the GPU CED tests.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4_1.nvidia import model as native


def test_compacted_prefill_graph_excludes_prompt_logprobs_and_other_batch_shapes():
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.single_request_prefill_cudagraph_tokens = 4096
    state._ced_prompt_logprobs = {"full-output"}
    assert state.can_use_single_request_prefill_graph(1, 4096, ["compact"])
    assert not state.can_use_single_request_prefill_graph(1, 4096, ["full-output"])
    assert not state.can_use_single_request_prefill_graph(1, 2048, ["compact"])
    assert not state.can_use_single_request_prefill_graph(2, 4096, ["a", "b"])


def test_piecewise_capture_refreshes_compacted_indices_after_metadata_staging():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState

    indices = torch.arange(128)
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = SimpleNamespace(get_indices=lambda: indices)
    inputs = {"ced_indices": None}
    state.finalize_cudagraph_inputs(inputs, CUDAGraphMode.FULL)
    assert inputs["ced_indices"] is None
    state.finalize_cudagraph_inputs(inputs, CUDAGraphMode.PIECEWISE)
    assert inputs["ced_indices"] is indices


@pytest.mark.parametrize("rows", [1, 8, 19])
@pytest.mark.parametrize("strided", [False, True])
def test_attention_returns_owned_projection_without_copy(monkeypatch, rows, strided):
    """The opaque wrapper preserves result ownership and reads each live input."""
    from vllm.models.deepseek_v4_1 import attention

    results = []

    def project(positions, hidden):
        output = (hidden.float() * 0.25 + positions[:, None] * 0.5).to(hidden.dtype)
        results.append(output)
        return output

    layer = SimpleNamespace(prefix="owned_projection", _forward=project)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={layer.prefix: layer}),
    )
    hidden = torch.arange(rows * 32, dtype=torch.bfloat16).reshape(rows, 32)
    if strided:
        hidden = hidden[:, ::2]
    positions = torch.arange(rows, dtype=torch.int64)
    ready = torch.ones(1, dtype=torch.uint8)
    saved_hidden = hidden.clone()
    first = attention.DeepseekV4Attention.forward(
        layer, positions, hidden, global_kv_ready=ready
    )
    assert first.data_ptr() == results[-1].data_ptr()
    assert first.data_ptr() != hidden.data_ptr()
    torch.testing.assert_close(hidden, saved_hidden, rtol=0, atol=0)
    saved_first = first.clone()
    hidden.add_(4)
    second = attention.DeepseekV4Attention.forward(
        layer, positions, hidden, global_kv_ready=ready
    )
    assert second.data_ptr() == results[-1].data_ptr()
    assert second.data_ptr() != first.data_ptr()
    torch.testing.assert_close(first, saved_first, rtol=0, atol=0)
    torch.testing.assert_close(
        second,
        (hidden.float() * 0.25 + positions[:, None] * 0.5).to(hidden.dtype),
        rtol=0,
        atol=0,
    )
    checks = torch.library.opcheck(
        attention._attention, (hidden, positions, layer.prefix, ready)
    )
    assert all(value == "SUCCESS" for value in checks.values()), checks


def test_attention_empty_batch_does_not_enter_cache_or_collectives():
    from vllm.models.deepseek_v4_1 import attention

    hidden = torch.empty((0, 32), dtype=torch.bfloat16)
    output = attention.DeepseekV4Attention.forward(
        SimpleNamespace(), torch.empty(0, dtype=torch.int64), hidden
    )
    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_attention_owned_output_graph_reads_live_inputs(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention

    def project(positions, hidden):
        return (hidden.float() * 0.25 + positions[:, None] * 0.5).to(hidden.dtype)

    layer = SimpleNamespace(prefix="graph_projection", _forward=project)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={layer.prefix: layer}),
    )
    for rows in (1, 8, 19):
        hidden = torch.arange(rows * 32, dtype=torch.bfloat16, device="cuda").reshape(
            rows, 32
        )
        positions = torch.arange(rows, dtype=torch.int64, device="cuda")
        for _ in range(3):
            attention.DeepseekV4Attention.forward(layer, positions, hidden)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = attention.DeepseekV4Attention.forward(layer, positions, hidden)
        assert output.data_ptr() != hidden.data_ptr()
        output_pointer = output.data_ptr()
        for _ in range(3):
            hidden.add_(4)
            positions.add_(1)
            graph.replay()
            torch.testing.assert_close(
                output, project(positions, hidden), rtol=0, atol=0
            )
            assert output.data_ptr() == output_pointer


@pytest.mark.parametrize("ced", [False, True])
@pytest.mark.parametrize("context", [32768, 131072, 1048576])
def test_indexer_short_scan_reservation_preserves_long_scan_and_decode(
    monkeypatch, ced, context
):
    from vllm.models.deepseek_v4_1 import attention

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        attention.mla,
        "plan",
        lambda caps: SimpleNamespace(
            caps=caps,
            shapes_and_dtypes=lambda: (),
        ),
    )
    monkeypatch.setattr(
        attention.dsa_indexer, "plan", lambda caps: SimpleNamespace(caps=caps)
    )
    monkeypatch.setattr(attention, "_scratch", lambda plan: None)
    monkeypatch.setattr(
        attention,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: None,
        ),
    )
    state = SimpleNamespace(
        _ready=False,
        INDEX_CHUNK=256,
        DECODE_CHUNK=64,
        config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=256),
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=128),
        ),
        max_model_len=context,
        compress_ratio=2,
        capacity=4096,
        n_local_heads=16,
        swa_width=128,
        is_ced_decoder=ced,
        layer_id=2,
        candidate_source_layer=20,
        topk_indices_buffer=None,
        is_index_source=True,
        indexer=SimpleNamespace(heads=32),
        compressor=None,
    )
    attention.DeepseekV4Attention._prepare(state, torch.device("cpu"))
    assert state._index_plans["decode"].caps.max_q_rows == 64
    assert state._index_plans["prefill"].caps.max_q_rows == 256
    assert state._index_plans["prefill"].caps.max_page_table_width == context // 256
    if ced:
        assert state._short_index_plan is None
    else:
        caps = state._short_index_plan.caps
        assert caps.max_q_rows == 1024
        assert caps.max_page_table_width * caps.page_size == 16384
        assert state._short_index_pages.shape == (1024, 128)


def _gather(source, indices):
    result = source[indices.clamp_min(0)].clone()
    result[indices < 0] = 0
    return result


def _scatter(source, indices, full_rows):
    result = source.new_zeros((full_rows, *source.shape[1:]))
    valid = indices >= 0
    result[indices[valid]] = source[valid]
    return result


class _Mix:
    def post(self, x, residual, post, comb):
        return residual + x[:, None, :] * 0.25

    def pre(self, residual, fn, scale, base, norm, pre, **previous):
        if previous.get("previous_output") is not None:
            residual = self.post(previous["previous_output"], residual, None, None)
        if residual.ndim == 2:
            residual = residual[:, None, :].expand(-1, 4, -1).clone()
        rows = residual.shape[0]
        mix = residual.new_zeros((rows, 4))
        mix[:, 0] = 1
        return (
            residual,
            residual.new_full((rows, 4), 0.25),
            torch.eye(4).expand(rows, 4, 4),
            residual[:, 0, :],
            mix,
        )

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(self.post(x, residual, post, comb), fn, scale, base, norm, pre)


class _Attention(nn.Module):
    def __init__(self, index, shared, rows):
        super().__init__()
        self.index, self.shared, self.rows = index, shared, rows

    def prepare_global_kv(self, positions, hidden):
        self.shared["global"] = hidden.mean(dim=0)
        self.shared["global_rows"] = hidden.shape[0]

    def forward(self, positions, hidden, scaling, *, global_kv_ready=None):
        if self.index == 2 and "global" not in self.shared:
            self.prepare_global_kv(positions, hidden)
        self.rows.append(("attention", self.index, hidden.shape[0]))
        result = hidden + self.index + 1 + positions[:, None].float() * 0.01
        if self.index >= 2:
            result = result + self.shared["global"]
        return result


class _Experts(nn.Module):
    def __init__(self, index, rows):
        super().__init__()
        self.index, self.rows = index, rows

    def forward(self, hidden, ids):
        self.rows.append(("experts", self.index, hidden.shape[0]))
        return hidden * 0.5 + ids[:, None].float() * 0.02


class _Layer(nn.Module):
    forward = native.DeepseekV4DecoderLayer.forward

    def __init__(self, index, shared, rows):
        super().__init__()
        self.engram = None
        self.hc_mult = 4
        self._b12x_mhc = _Mix()
        self.attn = _Attention(index, shared, rows)
        self.ffn = _Experts(index, rows)
        for name in (
            "hc_attn_fn_broadcast",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
        ):
            setattr(self, name, torch.ones(1))
        self.attn_norm = self.ffn_norm = SimpleNamespace(weight=torch.ones(2))


@pytest.mark.parametrize("compact", [False, True])
def test_ced_global_context_and_full_row_output_abi(monkeypatch, compact):
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(native, "gather_rows", _gather)
    monkeypatch.setattr(native, "scatter_rows", _scatter)
    monkeypatch.setattr(native, "collapse", lambda hidden, mix: hidden[:, 0])
    monkeypatch.setattr(native, "stream_mean", lambda hidden: hidden.mean(dim=1))
    ids = torch.arange(8)
    positions = torch.arange(100, 108)
    inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    indices = torch.tensor([2, 3, 6, 7, -1, -1]) if compact else None
    shared, row_work = {}, []
    buffer = torch.empty(8, 8)
    target = SimpleNamespace(
        embed_input_ids=lambda _: inputs,
        use_mega_moe=False,
        use_sequence_parallel=False,
        disk_engram=False,
        engram_hash=None,
        start_layer=0,
        end_layer=4,
        ced_decoder_start=2,
        layers=nn.ModuleList(_Layer(i, shared, row_work) for i in range(4)),
        aux_hidden_state_layers={3, 4},
        _mtp_hidden_buffer=buffer,
        norm=lambda hidden: hidden,
    )
    output, aux = native.DeepseekV4Model.forward(
        target, ids, positions, None, ced_indices=indices
    )

    # Independent arithmetic oracle; decoder global context includes encoder
    # rows that will never enter decoder attention or experts.
    expected = inputs.clone()
    expected_aux = []
    global_context = None
    for index in range(4):
        if index == 2:
            global_context = expected.mean(dim=0)
        attention = expected + index + 1 + positions[:, None].float() * 0.01
        if index >= 2:
            attention = attention + global_context
        after_attention = expected + attention * 0.25
        expected = (
            after_attention
            + (after_attention * 0.5 + ids[:, None].float() * 0.02) * 0.25
        )
        if index >= 2:
            expected_aux.append(expected.clone())
    if compact:
        keep = torch.zeros(8, dtype=torch.bool)
        keep[indices[indices >= 0]] = True
        expected[~keep] = 0
        for value in expected_aux:
            value[~keep] = 0
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    for actual, value in zip(aux, expected_aux, strict=True):
        torch.testing.assert_close(actual, value, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        buffer.view(8, 4, 2),
        expected[:, None, :].expand(-1, 4, -1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert shared["global_rows"] == 8
    decoder_rows = len(indices) if compact else 8
    assert row_work == [
        (kind, i, 8 if i < 2 else decoder_rows)
        for i in range(4)
        for kind in ("attention", "experts")
    ]
