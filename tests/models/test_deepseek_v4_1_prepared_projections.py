# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native compressor projection and stream-collapse preparation boundaries."""

from types import SimpleNamespace

import pytest
import torch


def _device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x integration requires SM12x")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("ratio", [1, 2])
@torch.no_grad()
def test_compressor_split_projection_precision_and_replay(ratio):
    from b12x.preparation import PreparationSession
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    torch.manual_seed(411)
    module = DeepseekCompressor.__new__(DeepseekCompressor)
    torch.nn.Module.__init__(module)
    module.prefix, module.compress_ratio, module.capacity = "compressor", ratio, 257
    module._projection_plans = {}
    weight = torch.randn(ratio * 512, 5120, dtype=torch.bfloat16, device=device) / 5120**0.5
    module.fused_wkv_wgate = SimpleNamespace(weight=weight)
    dtype = torch.float32 if ratio == 2 else torch.bfloat16
    module._values = torch.empty(257, 512, dtype=dtype, device=device)
    module._gates = torch.empty_like(module._values) if ratio == 2 else None
    source = torch.randn(257, 5120, dtype=torch.bfloat16, device=device)
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, 257), fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=257, max_seqs=8, max_model_len=257,
    )
    allocated = torch.cuda.memory_allocated(device)
    unit = module._projection_unit(workload)
    assert torch.cuda.memory_allocated(device) == allocated
    assert unit.stage == "weights"
    outputs = (module._values, module._gates)[:ratio]

    def check(rows):
        for index, output in enumerate(outputs):
            expected = (source[:rows].float() @ weight[index * 512:(index + 1) * 512].float().T).to(dtype)
            torch.testing.assert_close(output[:rows], expected, rtol=0.008 if ratio == 1 else 0.002, atol=1e-4)
            assert torch.isfinite(output[:rows]).all() and torch.count_nonzero(output[:rows]) > 0

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(unit.requests)
        session.freeze()
        for rows in (1, 8, 129, 257):
            for output in outputs:
                output.fill_(float("nan"))
            module._project(source[:rows])
            check(rows)
        graph = torch.cuda.CUDAGraph()
        pointers = tuple(output.data_ptr() for output in outputs)
        try:
            with session.capture(), torch.cuda.graph(graph):
                module._project(source)
            source.neg_()
            weight.mul_(0.5)
            for output in outputs:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert pointers == tuple(output.data_ptr() for output in outputs)
            check(257)
        finally:
            graph.reset()


@torch.no_grad()
def test_mhc_weighted_collapse_and_mean_use_prepared_plans(monkeypatch):
    from b12x.preparation import PreparationSession
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    monkeypatch.setattr(b12x_layers, "_execution_capacities", lambda: (1, 8, 4096))
    module = b12x_layers.B12xMHC(SimpleNamespace(
        hidden_size=5120, rms_norm_eps=1e-20, hc_eps=1e-6,
        hc_sinkhorn_iters=20, hc_mult=4,
    ))
    owner = SimpleNamespace(
        hc_attn_fn=torch.empty(24, 5120, device=device),
        hc_attn_fn_broadcast=None,
        hc_ffn_fn=torch.empty(24, 20480, device=device),
        hc_attn_scale=torch.empty(3, device=device), hc_ffn_scale=torch.empty(3, device=device),
        hc_attn_base=torch.empty(24, device=device), hc_ffn_base=torch.empty(24, device=device),
        attn_norm=SimpleNamespace(weight=torch.ones(5120, device=device, dtype=torch.bfloat16)),
        ffn_norm=SimpleNamespace(weight=torch.ones(5120, device=device, dtype=torch.bfloat16)),
    )
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, 4096), fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=4096, max_seqs=8, max_model_len=4096,
    )
    (unit,) = module.get_b12x_preparation_units(owner, workload)
    requests = tuple(request for request in unit.requests if ".collapse." in request.name)
    assert len(requests) == 2
    source = torch.randn(4096, 4, 5120, dtype=torch.bfloat16, device=device)
    mix = torch.randn(4096, 4, dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        session.freeze()
        for weighted in (False, True):
            def reference(rows):
                if weighted:
                    return (source[:rows].float() * mix[:rows, :, None]).sum(1).bfloat16()
                return source[:rows].float().mean(1).bfloat16()

            for rows in (1, 8, 257, 4096):
                actual = module.collapse(source[:rows], mix[:rows] if weighted else None)
                torch.testing.assert_close(actual, reference(rows), rtol=0.008, atol=1e-5)
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    actual = module.collapse(source, mix if weighted else None)
                pointer = actual.data_ptr()
                source.neg_()
                mix.mul_(0.5)
                actual.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated and actual.data_ptr() == pointer
                torch.testing.assert_close(actual, reference(4096), rtol=0.008, atol=1e-5)
            finally:
                graph.reset()


@torch.no_grad()
def test_dspark_context_projection_is_ready_before_pool_preparation(monkeypatch, workspace_init):
    from b12x.gemm import block_fp8_linear
    from b12x.preparation import PreparationSession, PreparedCall
    from vllm.models.deepseek_v4_1.nvidia import dspark
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    torch.manual_seed(419)
    counts, rank, hidden = (1, 8, 4096), 1280, 5120
    monkeypatch.setattr(dspark, "_execution_capacities", lambda: counts)
    weight = torch.randn(rank + 512, hidden, device=device).to(torch.float8_e4m3fn)
    scales = torch.randint(123, 128, ((rank + 512) // 32, hidden // 32),
                           dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
    fused = SimpleNamespace(weight=weight, weight_scale_inv=scales)
    projection = dspark._ContextKVProjection(SimpleNamespace(fused_wqa_wkv=fused, q_lora_rank=rank), 4096)
    packed = block_fp8_linear.pack_weight(weight, scales, block_size=(32, 32))
    workload = B12xWorkload(
        stage="weights", token_counts=counts, fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=4096, max_seqs=8, max_model_len=1048576,
    )
    (unit,) = projection.get_b12x_preparation_units(projection, workload)
    assert unit.stage == "weights"
    source = torch.randn(4096, hidden, dtype=torch.bfloat16, device=device)
    references = {}
    reference_requests = []
    for rows in counts:
        plan = block_fp8_linear.plan(block_fp8_linear.Caps(
            device=device, max_tokens=rows, in_features=hidden,
            out_features=rank + 512, output_mode="provided", block_size=(32, 32),
        ))
        def prime(state, rows=rows):
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                            for spec in state.scratch.scratch_specs())
            output = torch.empty(rows, rank + 512, 1, dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=source[:rows],
                                 packed_weight=packed, output=output)
            references[rows] = (state, binding)
            return PreparedCall(run=lambda: state.run_binding(binding))
        reference_requests.append(plan.request(name=f"fused-reference.m{rows}", prepare_call=prime))
    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(unit.requests)
        session.prepare(reference_requests, autotune=False)
        for rows in (1, 7, 8, 96, 257, 1023, 4096):
            expected_state, expected_binding = references[next(count for count in counts if rows <= count)]
            expected_state.run_binding(expected_binding)
            actual = projection(source[:rows])
            expected = expected_binding.output[:rows, rank:, 0]
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        session.freeze()
        for rows in (96, 4096):
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    actual = projection(source[:rows])
                pointer = actual.data_ptr()
                source.neg_()
                actual.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated and actual.data_ptr() == pointer
                expected_state, expected_binding = references[4096]
                expected_state.run_binding(expected_binding)
                torch.testing.assert_close(actual, expected_binding.output[:rows, rank:, 0], rtol=0.008, atol=1e-5)
            finally:
                graph.reset()


@torch.no_grad()
def test_weights_collection_prepares_shared_and_markov_logits_heads(
    default_vllm_config, dist_init, monkeypatch,
):
    from b12x.preparation import PreparationSession
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    monkeypatch.setattr(default_vllm_config.kernel_config, "linear_backend", "b12x")
    torch.manual_seed(411)
    target, draft = torch.nn.Module(), torch.nn.Module()
    with torch.device(device):
        target.lm_head = ParallelLMHead(32768, 5120, params_dtype=torch.bfloat16)
    target.lm_head.weight.normal_(std=5120**-0.5)
    target.logits_processor = LogitsProcessor(32768)
    draft.logits_processor = LogitsProcessor(32768)
    draft.lm_head = target.lm_head
    assert not target.logits_processor._b12x_vocab_heads
    assert not draft.logits_processor._b12x_vocab_heads
    with torch.device(device):
        markov = ParallelLMHead(32768, 256, params_dtype=torch.bfloat16, disable_tp=True)
    markov.weight.normal_(std=256**-0.5)
    draft.logits_processor.prepare_b12x_vocab_projection(markov)
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8), fixed_token_counts=(1,),
        output_dtype=torch.bfloat16, max_tokens=8, max_seqs=8, max_model_len=1048576,
    )
    seen = set()
    units = tuple(_units_from_modules(target, workload, seen=seen)) + tuple(
        _units_from_modules(draft, workload, seen=seen)
    )
    processors = (target.logits_processor, draft.logits_processor)
    assert len(units) == 3 and all(unit.stage == "weights" for unit in units)
    source = torch.randn(8, 5120, dtype=torch.bfloat16, device=device)
    markov_source = torch.randn(8, 256, dtype=torch.bfloat16, device=device)
    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        session.freeze()
        for rows in (1, 7, 8):
            expected = torch.nn.functional.linear(source[:rows], target.lm_head.weight)
            for processor in processors:
                actual = processor(target.lm_head, source[:rows])
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-4)
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            bias = draft.logits_processor(markov, markov_source[:rows])
            expected_bias = torch.nn.functional.linear(markov_source[:rows], markov.weight)
            torch.testing.assert_close(bias, expected_bias, rtol=0.008, atol=1e-4)
            assert torch.isfinite(bias).all() and torch.count_nonzero(bias) > 0
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                actual = draft.logits_processor(target.lm_head, source)
                bias = draft.logits_processor(markov, markov_source)
            pointer = actual.data_ptr()
            bias_pointer = bias.data_ptr()
            source.neg_()
            target.lm_head.weight.mul_(0.5)
            markov_source.neg_()
            markov.weight.mul_(0.5)
            bias.fill_(float("nan"))
            actual.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated and actual.data_ptr() == pointer
            expected = torch.nn.functional.linear(source, target.lm_head.weight)
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-4)
            assert bias.data_ptr() == bias_pointer
            expected_bias = torch.nn.functional.linear(markov_source, markov.weight)
            torch.testing.assert_close(bias, expected_bias, rtol=0.008, atol=1e-4)
        finally:
            graph.reset()
