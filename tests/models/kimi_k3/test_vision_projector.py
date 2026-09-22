# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.models.kimi_k25_vit import (
    MLP2,
    KimiK25MultiModalProjector,
    MoonViTEncoderLayer,
    _project_vision_intermediate,
    apply_rope_into_packed_qk,
    mm_projector_forward,
)


@torch.inference_mode()
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reuse_projection_storage", [False, True])
def test_vision_attention_releases_packed_qkv_before_output_projection(
    device, reuse_projection_storage, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.v1.worker import workspace

    monkeypatch.setattr(
        workspace, "_manager", workspace.WorkspaceManager(torch.device(device))
    )
    inputs = torch.randn(17, 64, device=device, dtype=torch.bfloat16)
    original_inputs = inputs.clone()
    bank = torch.empty(17 * 192, device=device, dtype=inputs.dtype)
    weight = torch.randn(192, 64, device=device, dtype=torch.bfloat16)
    phases = torch.randn(17, 16, device=device)
    frequencies = torch.polar(torch.ones_like(phases), phases)
    packed_refs = []
    normalized_refs = []

    def project(x):
        normalized_refs.append(weakref.ref(x))
        packed = F.linear(x, weight)
        packed_refs.append(weakref.ref(packed))
        return packed, None

    def attend(q, k, v, **kwargs):
        assert all(reference() is None for reference in normalized_refs)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def output_projection(x):
        assert all(reference() is None for reference in packed_refs)
        return x + 1, None

    layer = SimpleNamespace(
        wqkv=project,
        wo=output_projection,
        attn=attend,
        num_attention_heads_per_partition=2,
        hidden_size_per_attention_head=32,
        norm0=lambda x: x + 2,
        norm1=lambda x: x + 3,
        mlp=lambda x, **kwargs: x + 4,
    )
    layer.attention_qkvpacked = MethodType(
        MoonViTEncoderLayer.attention_qkvpacked, layer
    )
    cu_seqlens = torch.tensor([0, 17], dtype=torch.int32, device=device)

    def execute():
        return MoonViTEncoderLayer.forward(
            layer,
            inputs,
            cu_seqlens,
            frequencies,
            max_seqlen=17,
            projection_workspace=bank if reuse_projection_storage else None,
        )

    def reference():
        packed = F.linear(inputs + 2, weight).view(17, 3, 2, 32)
        q, k, v = packed.unbind(1)
        q, k = apply_rope_into_packed_qk(q, k, frequencies)
        attention = (
            attend(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).reshape(17, 64) + 1
        )
        hidden = inputs + attention
        return hidden + ((hidden + 3) + 4)

    torch.testing.assert_close(execute(), reference(), rtol=0, atol=0)
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = execute()
        for _ in range(3):
            inputs.normal_()
            expected = reference()
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scratch_rows", [None, 1, 127])
def test_vision_rope_preserves_packed_storage_and_complex_arithmetic(
    device, dtype, scratch_rows
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(87)
    packed = torch.randn(257, 3, 12, 128, device=device, dtype=dtype)
    query, key, value = packed.unbind(1)
    original_value = value.clone()
    phases = torch.randn(257, 64, device=device)
    frequencies = torch.polar(torch.ones_like(phases), phases)

    def reference(x):
        converted = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        rotated = converted * frequencies.unsqueeze(-2)
        return torch.view_as_real(rotated).flatten(-2).to(dtype)

    expected_q, expected_k = reference(query), reference(key)
    scratch = (
        None
        if scratch_rows is None
        else torch.empty(scratch_rows * 12 * 128, dtype=torch.float32, device=device)
    )
    actual_q, actual_k = apply_rope_into_packed_qk(
        query, key, frequencies, workspace=scratch
    )
    assert actual_q is query and actual_k is key
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(value, original_value, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vision_rope_caller_scratch_has_no_temporary_allocation():
    packed = torch.randn(16385, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
    query, key, value = packed.unbind(1)
    phases = torch.randn(16385, 64, device="cuda")
    frequencies = torch.polar(torch.ones_like(phases), phases)
    scratch = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.float32)
    apply_rope_into_packed_qk(query, key, frequencies, workspace=scratch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    apply_rope_into_packed_qk(query, key, frequencies, workspace=scratch)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == allocated
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_rope_into_packed_qk(query, key, frequencies, workspace=scratch)
    for _ in range(3):
        packed.normal_()
        expected = packed.clone()
        expected_q, expected_k, expected_v = expected.unbind(1)
        apply_rope_into_packed_qk(expected_q, expected_k, frequencies)
        scratch.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vision_rope_bounds_fp32_storage_and_replays_changed_images():
    packed = torch.randn(16385, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
    query, key, value = packed.unbind(1)
    phases = torch.randn(16385, 64, device="cuda")
    frequencies = torch.polar(torch.ones_like(phases), phases)

    def reference(x):
        converted = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        return (
            torch.view_as_real(converted * frequencies.unsqueeze(-2))
            .flatten(-2)
            .to(x.dtype)
        )

    apply_rope_into_packed_qk(query, key, frequencies)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    apply_rope_into_packed_qk(query, key, frequencies)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated <= 64 * 1024 * 1024

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_rope_into_packed_qk(query, key, frequencies)
    for _ in range(3):
        packed.normal_()
        expected_q, expected_k, expected_v = (
            reference(query),
            reference(key),
            value.clone(),
        )
        graph.replay()
        torch.testing.assert_close(query, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(key, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_v, rtol=0, atol=0)


class _PackedProjector(nn.Module):
    def __init__(self, storage_dtype, norm_name):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(1, dtype=storage_dtype), requires_grad=False
        )
        setattr(self, norm_name, nn.LayerNorm(4, dtype=torch.bfloat16))
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs)
        return inputs + 1


@pytest.mark.parametrize("storage_dtype", [torch.int32, torch.float8_e4m3fn])
@pytest.mark.parametrize("norm_name", ["pre_norm", "post_norm"])
def test_projector_preserves_bf16_activations_with_packed_weights(
    storage_dtype, norm_name
):
    projector = _PackedProjector(storage_dtype, norm_name)
    inputs = [torch.randn(2, 4), torch.randn(3, 4)]
    outputs = mm_projector_forward(projector, inputs)
    assert [value.shape for value in projector.inputs] == [(2, 4), (3, 4)]
    assert len(outputs) == len(inputs)
    for result, features in zip(outputs, inputs):
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, features.bfloat16() + 1, rtol=0, atol=0)


def test_projector_rejects_empty_image_features():
    with pytest.raises(ValueError, match="requires at least one image"):
        mm_projector_forward(_PackedProjector(torch.int32, "post_norm"), [])


def test_image_wise_projection_matches_combined_batch():
    torch.manual_seed(1729)
    projector = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 3))
    inputs = [torch.randn(2, 4), torch.randn(3, 4)]
    expected = projector(torch.cat(inputs)).split([2, 3])
    outputs = mm_projector_forward(projector, inputs)
    for result, reference in zip(outputs, expected):
        torch.testing.assert_close(result, reference)


def _projector_for_norm_test(device, width):
    projector = KimiK25MultiModalProjector.__new__(KimiK25MultiModalProjector)
    nn.Module.__init__(projector)
    projector.mm_projector_type = "patchmergerv2"
    # Isolate the normalization lifetime while preserving a fresh projection
    # output: callers must retain their original image features unchanged.
    projector.linear_1 = lambda x: (x.clone(), None)
    projector.linear_2 = lambda x: (x, None)
    projector.act = nn.Identity()
    projector.post_norm = nn.RMSNorm(
        width, eps=1e-6, device=device, dtype=torch.bfloat16
    )
    return projector


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [17, 1024, 1025, 10000])
def test_projector_norm_bounds_rows_and_preserves_exact_graph_output(rows):
    projector = _projector_for_norm_test("cuda", 7168)
    inputs = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)
    original = inputs.clone()
    expected = projector.post_norm(inputs)
    chunk_rows = []
    handle = projector.post_norm.register_forward_pre_hook(
        lambda module, args: chunk_rows.append(args[0].shape[0])
    )
    output = projector(inputs)
    handle.remove()
    assert sum(chunk_rows) == rows and max(chunk_rows) <= 1024
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = projector(inputs)
    for _ in range(3):
        inputs.normal_()
        expected = projector.post_norm(inputs)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_projector_norm_retains_differentiable_path():
    projector = _projector_for_norm_test("cpu", 8)
    inputs = torch.randn(1025, 8, dtype=torch.bfloat16, requires_grad=True)
    projector(inputs).float().square().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all() and inputs.grad.count_nonzero() > 0


class _TupleLinear(nn.Linear):
    def forward(self, x):
        output = super().forward(x)
        self.output = output
        return output, None


def _mlp_with_local_linears(activation, device, dtype):
    # Only the parallel linear plumbing is replaced; exercise MLP2.forward.
    mlp = MLP2.__new__(MLP2)
    nn.Module.__init__(mlp)
    mlp.fc0 = _TupleLinear(32, 64, device=device, dtype=dtype)
    mlp.fc1 = _TupleLinear(64, 32, device=device, dtype=dtype)
    mlp.activation = activation
    return mlp


@torch.inference_mode()
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("activation", [F.gelu, nn.GELU(), nn.GELU("tanh"), nn.SiLU()])
def test_vision_mlp_consumed_gelu_matches_activation(device, dtype, activation):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(79)
    mlp = _mlp_with_local_linears(activation, device, dtype)
    inputs = torch.randn(257, 32, device=device, dtype=dtype)
    original_inputs = inputs.clone()
    intermediate, _ = mlp.fc0(inputs)
    expected_activation = activation(intermediate)
    expected, _ = mlp.fc1(expected_activation)
    actual = mlp(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    if activation is F.gelu or type(activation) is nn.GELU:
        torch.testing.assert_close(mlp.fc0.output, expected_activation, rtol=0, atol=0)


def test_vision_mlp_keeps_differentiable_activation():
    mlp = _mlp_with_local_linears(nn.GELU(), "cpu", torch.float32)
    inputs = torch.randn(3, 32, requires_grad=True)
    output = mlp(inputs, consume_input=True)
    output.square().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all() and inputs.grad.count_nonzero() > 0


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vision_mlp_gelu_graph_replays_changed_images():
    mlp = _mlp_with_local_linears(nn.GELU(), "cuda", torch.bfloat16)
    inputs = torch.randn(257, 32, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mlp(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mlp(inputs)
    pointer = output.data_ptr()
    for _ in range(3):
        inputs.normal_()
        intermediate, _ = mlp.fc0(inputs)
        expected, _ = mlp.fc1(F.gelu(intermediate))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert output.data_ptr() == pointer


def _marlin_vision_projection(input_size, output_size):
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_layer_for_marlin,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    layer = nn.Module()
    layer.input_size_per_partition = input_size
    layer.output_size_per_partition = output_size
    layer.disable_tp, layer.skip_bias_add = True, False
    layer.bias = None
    weight = (
        torch.randn(output_size, input_size, device="cuda", dtype=torch.bfloat16) / 32
    )
    packed, scales = mxfp8_e4m3_quantize(weight)
    layer.weight = nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = nn.Parameter(scales, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)
    layer.quant_method = SimpleNamespace(kernel=object.__new__(MarlinMxfp8LinearKernel))
    layer.forward = lambda x: (
        layer.quant_method.kernel.apply_weights(layer, x, layer.bias),
        None,
    )
    return layer


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [17, 257, 20160])
def test_vision_marlin_intermediate_reuses_disjoint_caller_storage(rows):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )

    torch.manual_seed(1743)
    layer = _marlin_vision_projection(1024, 4608)
    inputs = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    bank = torch.empty(rows * 4608 + 256, device="cuda", dtype=inputs.dtype)

    def reference():
        return apply_mxfp8_marlin_linear(
            inputs, layer.weight, layer.weight_scale, layer.workspace, 4608, 1024
        )

    expected = reference()
    original_inputs = inputs.clone()
    bank.fill_(float("nan"))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    output = _project_vision_intermediate(layer, inputs, bank)
    torch.cuda.synchronize()
    # Only Marlin's bounded FP32 reduction scratch remains, not the 178 MiB
    # Q/K/V output of the 20,160-patch case.
    assert torch.cuda.max_memory_allocated() - allocated <= 16 * 2**20
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    assert output.data_ptr() == bank.data_ptr()
    assert bank[-256:].isnan().all()

    with pytest.raises(ValueError, match="disjoint"):
        apply_mxfp8_marlin_linear(
            inputs,
            layer.weight,
            layer.weight_scale,
            layer.workspace,
            4608,
            1024,
            output=inputs,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _project_vision_intermediate(layer, inputs, bank)
    for _ in range(3):
        inputs.normal_()
        expected = reference()
        bank.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert output.data_ptr() == bank.data_ptr()
        assert bank[-256:].isnan().all()


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [17, 257, 20160, 40960])
def test_vision_mlp_replaces_only_consumed_normalized_input(rows):
    torch.manual_seed(1744)
    mlp = MLP2.__new__(MLP2)
    nn.Module.__init__(mlp)
    mlp.fc0 = _marlin_vision_projection(1024, 4096)
    mlp.fc1 = _marlin_vision_projection(4096, 1024)
    mlp.activation = F.gelu
    inputs = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    original = inputs.clone()
    bank = torch.empty(rows * 4096, device="cuda", dtype=inputs.dtype)
    expected = mlp(inputs, projection_workspace=bank)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    output = mlp(inputs, projection_workspace=bank, consume_input=True)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated <= 16 * 2**20
    assert output.data_ptr() == inputs.data_ptr()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mlp(inputs, projection_workspace=bank, consume_input=True)
    for _ in range(3):
        inputs.normal_()
        expected = mlp(inputs, projection_workspace=bank)
        bank.fill_(float("nan"))
        graph.replay()
        assert output.data_ptr() == inputs.data_ptr()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_vision_intermediate_preserves_unsupported_linear_path():
    layer = _TupleLinear(32, 64, dtype=torch.bfloat16)
    inputs = torch.randn(17, 32, dtype=torch.bfloat16)
    scratch = torch.full((17 * 64,), float("nan"), dtype=torch.bfloat16)
    expected = layer(inputs)[0]
    actual = _project_vision_intermediate(layer, inputs, scratch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert scratch.isnan().all()


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [17, 257, 20160, 40960])
def test_vision_block_preserves_attention_residual_during_mlp(rows, monkeypatch):
    """Borrowed attention output must not overlap the following MLP's fc0."""
    from vllm.v1.worker import workspace

    monkeypatch.setattr(
        workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda"))
    )
    torch.manual_seed(1745)
    layer = MoonViTEncoderLayer.__new__(MoonViTEncoderLayer)
    nn.Module.__init__(layer)
    layer.wqkv = _marlin_vision_projection(1024, 4608)
    layer.wo = _marlin_vision_projection(1536, 1024)
    layer.num_attention_heads_per_partition = 12
    layer.hidden_size_per_attention_head = 128
    layer.norm0 = nn.LayerNorm(1024, device="cuda", dtype=torch.bfloat16)
    layer.norm1 = nn.LayerNorm(1024, device="cuda", dtype=torch.bfloat16)
    layer.mlp = MLP2.__new__(MLP2)
    nn.Module.__init__(layer.mlp)
    layer.mlp.fc0 = _marlin_vision_projection(1024, 4096)
    layer.mlp.fc1 = _marlin_vision_projection(4096, 1024)
    layer.mlp.activation = F.gelu
    # Keep actual quantized projections and block lifetimes without a
    # quadratic attention workload in this storage-ownership test.
    layer.attn = lambda q, k, v, **kwargs: v.clone()
    inputs = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    original = inputs.clone()
    bank = torch.empty(rows * 5632 + 256, device="cuda", dtype=inputs.dtype)
    frequencies = torch.ones(rows, 64, device="cuda", dtype=torch.complex64)
    cu_seqlens = torch.tensor([0, rows], device="cuda", dtype=torch.int32)

    def run(scratch):
        return layer(
            inputs,
            cu_seqlens,
            frequencies,
            max_seqlen=rows,
            projection_workspace=scratch,
        )

    expected = run(None)
    bank.fill_(float("nan"))
    actual = run(bank)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    assert bank[-256:].isnan().all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(bank)
    for _ in range(3):
        inputs.normal_()
        expected = run(None)
        bank.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bank[-256:].isnan().all()
