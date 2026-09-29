# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


class _Accumulator:
    input_width = 8
    slice_width = 4
    max_tokens = 2048

    def __init__(self, output: torch.Tensor):
        self.output = output
        self.tokens = 0
        self.inputs: list[torch.Tensor] = []

    def begin(self, tokens: int) -> None:
        self.tokens = tokens
        self.inputs.clear()

    def append(self, source: torch.Tensor) -> None:
        self.inputs.append(source.clone())

    def finish(self) -> torch.Tensor:
        self.output[: self.tokens].fill_(7)
        return self.output[: self.tokens]


def test_padded_auxiliary_projection_trims_after_one_gather(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash

    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=5)
    model._aux_projection_tp_size = 2
    local = torch.arange(6, dtype=torch.bfloat16).view(2, 3)
    calls = []

    def gather(value, dim):
        calls.append(value)
        return torch.cat((value, value), dim=dim)

    monkeypatch.setattr(qwen3_dflash, "tensor_model_parallel_all_gather", gather)
    result = model._gather_auxiliary_projection(local)
    assert len(calls) == 1
    torch.testing.assert_close(result, torch.cat((local, local), dim=-1)[:, :5])


def test_padded_auxiliary_projection_trims_small_inputs():
    fc = nn.Linear(8, 6, bias=False)
    fc.input_size = 8
    wrapper = object.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=5),
        use_aux_hidden_state=True,
        fc=fc,
        _streamed_aux_accumulator=None,
    )
    source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    torch.testing.assert_close(wrapper.combine_hidden_states(source), fc(source)[:, :5])


def test_dflash_streams_auxiliary_states_and_claims_result_once() -> None:
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.use_aux_hidden_state = True
    model._target_hidden_size = 4
    model._streamed_aux_layer_ids = (2, 4)
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    scratch = torch.empty(2048, 4)
    accumulator = _Accumulator(scratch)
    model.bind_auxiliary_stream(accumulator, scratch)

    first = torch.ones(1024, 4)
    second = torch.full((1024, 4), 2.0)
    residual = torch.full((1024, 4), 3.0)
    assert model.can_stream_auxiliary_states((2, 4), first)

    model.begin_auxiliary_stream(first)
    model.accumulate_auxiliary_state(first, None)
    model.accumulate_auxiliary_state(second, residual)
    result = model.finish_auxiliary_stream()

    torch.testing.assert_close(accumulator.inputs[0], first)
    torch.testing.assert_close(accumulator.inputs[1], second + residual)
    assert model.is_streamed_context_states([result])
    assert not model.is_streamed_context_states([result])


def test_dflash_narrows_output_scratch_for_target_residual_staging() -> None:
    """A narrower target state may reuse the draft output buffer safely."""
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=6)
    model.use_aux_hidden_state = True
    model._target_hidden_size = 4
    model._streamed_aux_layer_ids = (2, 4)
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    scratch = torch.empty(2048, 6)
    accumulator = _Accumulator(scratch)
    model.bind_auxiliary_stream(accumulator, scratch)

    primary = torch.full((1024, 4), 2.0)
    residual = torch.full((1024, 4), 3.0)
    model.begin_auxiliary_stream(primary)
    model.accumulate_auxiliary_state(primary, residual)

    assert len(accumulator.inputs) == 1
    torch.testing.assert_close(
        accumulator.inputs[0],
        torch.full((1024, 4), 5.0),
    )


def test_dflash_rejects_output_scratch_narrower_than_target_state() -> None:
    """Streaming must fall back when target residuals cannot fit the scratch."""
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model._target_hidden_size = 6
    scratch = torch.empty(2048, 4)

    with pytest.raises(ValueError, match="at least 6 wide"):
        model.bind_auxiliary_stream(_Accumulator(scratch), scratch)


@pytest.mark.parametrize("rows", [1, 8, 1023, 1024, 2048])
def test_dflash_large_concatenated_inputs_reuse_staged_projection(rows) -> None:
    """Every large fallback uses staging; its FC needs only small-row scratch."""
    fc = nn.Linear(8, 4, bias=False)
    fc.input_size = 8
    accumulator = _Accumulator(torch.empty(2048, 4))
    wrapper = object.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
        use_aux_hidden_state=True,
        fc=fc,
        _streamed_aux_accumulator=accumulator,
        _gather_auxiliary_projection=lambda output: output,
    )
    source = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    output = wrapper.combine_hidden_states(source)
    if rows < 1024:
        assert not accumulator.inputs
        torch.testing.assert_close(output, fc(source))
    else:
        assert len(accumulator.inputs) == 2
        torch.testing.assert_close(torch.cat(accumulator.inputs, dim=-1), source)
        assert output.data_ptr() == accumulator.output.data_ptr()


def test_dflash_preparation_bounds_nonstaged_fc_capacity() -> None:
    """Prepared FC capacity covers exactly the rows not handled by staging."""
    from vllm.utils.b12x import B12xWorkload

    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8, 1024, 4096),
        fixed_token_counts=(1, 8, 1024),
        output_dtype=torch.bfloat16,
        max_tokens=4096,
        max_seqs=128,
        max_model_len=950000,
    )
    accumulator = SimpleNamespace(preparation_unit=lambda name: name)
    linear = SimpleNamespace(layer_name="draft.fc", unit=lambda w, name: w)
    wrapper = SimpleNamespace(
        model=SimpleNamespace(
            _streamed_aux_accumulator=accumulator,
            fc=SimpleNamespace(b12x_linear=linear),
        )
    )
    small, staged = DFlashQwen3ForCausalLM.get_b12x_preparation_units(
        wrapper, wrapper, workload
    )
    assert small.max_tokens == 1023
    assert small.token_counts == (1, 8, 1023)
    assert small.fixed_token_counts == (1, 8)
    assert staged == "dflash.auxiliary"


def test_dflash_bf16_staging_keeps_local_projection_and_single_gather(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash

    monkeypatch.setattr(qwen3_dflash.envs, "VLLM_DFLASH_AUX_BF16_STAGING", True)
    monkeypatch.setattr(qwen3_dflash.envs, "VLLM_DFLASH_AUX_MXFP8_STREAMING", False)
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.use_aux_hidden_state = True
    model._target_hidden_size = 4
    model._aux_projection_tp_size = 2
    model._streamed_aux_layer_ids = (2, 4)
    model._streamed_aux_generation = 0
    weight = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 16
    calls = []

    def local_linear(layer, x, bias):
        calls.append("linear")
        return torch.nn.functional.linear(x, weight, bias)

    def gather(local_output, dim):
        calls.append("gather")
        assert local_output.shape[1] == 2
        return torch.cat([local_output, local_output], dim=dim)

    monkeypatch.setattr(qwen3_dflash, "tensor_model_parallel_all_gather", gather)
    model.fc = SimpleNamespace(
        input_size=8, bias=None, quant_method=SimpleNamespace(apply=local_linear)
    )
    wrapper = object.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    bound = []
    target = SimpleNamespace(set_aux_hidden_state_projector=bound.append)
    scratch = torch.empty(2048, 4, dtype=torch.bfloat16)
    wrapper.bind_target_auxiliary_stream(target, scratch)
    assert bound == [model]
    first = torch.full((1024, 4), 0.125, dtype=torch.bfloat16)
    second = torch.full_like(first, -0.25)
    model.begin_auxiliary_stream(first)
    model.accumulate_auxiliary_state(first, None)
    model.accumulate_auxiliary_state(second, None)
    actual = model.finish_auxiliary_stream()
    expected_local = torch.nn.functional.linear(torch.cat([first, second], -1), weight)
    torch.testing.assert_close(
        actual, torch.cat([expected_local, expected_local], -1), rtol=0, atol=0
    )
    assert calls == ["linear", "gather"]
    assert (
        wrapper.get_b12x_preparation_units(wrapper, SimpleNamespace(stage="weights"))
        == ()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "slices,width,output_width,capacity",
    [(2, 128, 64, 257), (6, 7168, 7168, 4096), (6, 7168, 448, 4096)],
)
def test_prepared_mxfp8_staging_matches_concatenation_and_replays(
    slices,
    width,
    output_width,
    capacity,
):
    """Slice reuse preserves MXFP8 bytes, one-GEMM math and graph ownership."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession

    from vllm.model_executor.kernels.linear.mxfp8.staged import (
        B12xMxfp8InputAccumulator,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(314159)
    device = torch.device("cuda:0")
    weight = (
        torch.randn(output_width, slices * width, device=device, dtype=torch.bfloat16)
        / 64
    )
    values, scales = mxfp8_e4m3_quantize(weight)
    packed = blockscaled.pack_weight(values, scales)
    del weight, values, scales
    layer = SimpleNamespace(
        b12x_mxfp8_packed_weight=packed, b12x_activation_mode="quantized"
    )
    output = torch.empty(capacity, output_width, dtype=torch.bfloat16, device=device)
    accumulator = B12xMxfp8InputAccumulator(layer, output, width)
    unit = accumulator.preparation_unit("staged-test")
    source = (
        torch.randn(slices, capacity, width, dtype=torch.bfloat16, device=device) / 8
    )

    def run(rows):
        accumulator.begin(rows)
        for i in range(slices):
            accumulator.append(source[i, :rows])
        return accumulator.finish()

    def check(rows, expected=None):
        joined = torch.cat([source[i, :rows] for i in range(slices)], dim=-1)
        ref_values, ref_scales = mxfp8_e4m3_quantize(joined)
        torch.testing.assert_close(
            accumulator.values[:rows].view(torch.uint8),
            ref_values.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            accumulator.scales[:rows], ref_scales, rtol=0, atol=0
        )
        actual = output[:rows]
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            dequantized_input = ref_values.float() * torch.exp2(
                ref_scales.float() - 127
            ).repeat_interleave(32, dim=1)
            dequantized_weight = packed.weight.values.float() * torch.exp2(
                packed.weight.scale_rows.view(torch.uint8)
                .reshape(output_width, -1)
                .float()
                - 127
            ).repeat_interleave(32, dim=1)
            reference = dequantized_input @ dequantized_weight.t()
            torch.testing.assert_close(actual.float(), reference, rtol=0.01, atol=0.003)

    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(unit.requests)
        session.freeze()
        pointers = [t.data_ptr() for t in (output, accumulator.values, accumulator.mma)]
        for rows in (1, 8, 129, capacity):
            with kernel_resolution_guard("staged input live counts"):
                run(rows)
            check(rows)

        rows = 129
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(rows)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(rows)
        source.mul_(0.5)
        run(rows)
        expected = output[:rows].clone()
        torch.accelerator.synchronize()
        output.fill_(float("nan"))
        accumulator.values.view(torch.uint8).fill_(0xFF)
        accumulator.mma.fill_(0xFF)
        allocated = torch.accelerator.memory_allocated()
        with kernel_resolution_guard("staged input graph replay"):
            graph.replay()
            torch.accelerator.synchronize()
        assert torch.accelerator.memory_allocated() == allocated
        check(rows, expected)
        assert pointers == [
            t.data_ptr() for t in (output, accumulator.values, accumulator.mma)
        ]
