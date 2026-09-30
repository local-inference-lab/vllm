# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GLM-5.3 TP3 FP8 decode and low-latency GEMM selection.

Layers are built on the meta device; only flag parsing, TP/shape guards and
method selection are exercised. The kernels themselves need a GPU.
"""

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.glm5next.nvidia.glm53_fp8_dense as fp8
import vllm.models.glm5next.nvidia.glm53_low_latency_gemm as llg
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform

FP8_ENVS = (
    "VLLM_GLM53_FP8_DENSE",
    "VLLM_GLM53_FP8_LM_HEAD",
    "VLLM_GLM53_FP8_FFN",
    "VLLM_GLM53_FP8_MTP",
    "VLLM_GLM53_FP8_PREFILL",
    "VLLM_GLM53_FP8_DENSE_MAX_M",
    "VLLM_GLM53_LOW_LATENCY_GEMM",
)

# Per-rank BF16 projection shapes (N, K) of one GLM-5.3 decoder slice.
TP3_SHAPES = {
    "kda_in_proj": (8598, 4096),
    "kda_o_proj": (4096, 2816),
    "kda_g_a_proj": (128, 4096),
    "dsa_fused_qkv_a": (2048, 4096),
    "dsa_q_b_proj": (6144, 1536),
    "dsa_o_proj": (4096, 6144),
    "ffn_gate_up": (8192, 4096),
    "ffn_down": (4096, 4096),
    "shared_down": (4096, 704),
    "indexer_weights_proj": (32, 4096),
}
# At TP4 the DSA o_proj is (4096, 4096), the TP3 dense FFN down_proj shape.
TP4_SHAPES = {
    "kda_in_proj": (6288, 4096),
    "dsa_q_b_proj": (4096, 1536),
    "dsa_o_proj": (4096, 4096),
    "ffn_gate_up": (6144, 4096),
    "dsa_fused_qkv_a": (2048, 4096),
    "indexer_weights_proj": (32, 4096),
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in FP8_ENVS:
        monkeypatch.delenv(name, raising=False)
    # Replicated layers built on CPU still ask for the TP rank.
    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture
def tp_size(monkeypatch):
    def set_tp(size: int) -> None:
        for module in (fp8, llg):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_world_size", lambda: size
            )

    set_tp(3)
    return set_tp


def _toy_model(shapes: dict[str, tuple[int, int]]) -> nn.Module:
    model = nn.Module()
    with torch.device("meta"):
        for name, (n, k) in shapes.items():
            setattr(
                model,
                name,
                ReplicatedLinear(k, n, bias=False, params_dtype=torch.bfloat16),
            )
    return model


def _methods(model: nn.Module) -> dict[str, type]:
    return {
        name: type(child.quant_method)
        for name, child in model.named_children()
        if hasattr(child, "quant_method")
    }


def _all_plain(model: nn.Module) -> bool:
    return all(cls is UnquantizedLinearMethod for cls in _methods(model).values())


# ---------------------------------------------------------------------------
# FP8 dense decode
# ---------------------------------------------------------------------------


def test_fp8_dense_is_off_by_default(tp_size) -> None:
    model = _toy_model(TP3_SHAPES)
    assert not fp8.enabled()
    assert fp8.enable_glm53_fp8_dense(model) == 0
    assert fp8.enable_glm53_fp8_dense(model, draft=True) == 0
    assert _all_plain(model)


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_fp8_dense_skips_non_tp3(monkeypatch, tp_size, size) -> None:
    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE", "1")
    tp_size(size)
    model = _toy_model(TP4_SHAPES)
    assert fp8.enable_glm53_fp8_dense(model) == 0
    assert fp8.enable_glm53_fp8_dense(model, draft=True) == 0
    assert _all_plain(model)


def test_fp8_dense_wraps_tp3_projections(monkeypatch, tp_size) -> None:
    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE", "1")
    model = _toy_model(TP3_SHAPES)
    assert fp8.enable_glm53_fp8_dense(model) == 6
    methods = _methods(model)
    wrapped = {
        name for name, cls in methods.items() if cls is fp8.GLM53Fp8DecodeLinearMethod
    }
    assert wrapped == {
        "kda_in_proj",
        "kda_o_proj",
        "dsa_q_b_proj",
        "dsa_o_proj",
        "ffn_gate_up",
        "ffn_down",
    }
    # Small, shared-expert and fused_qkv_a projections stay BF16.
    for name in ("kda_g_a_proj", "dsa_fused_qkv_a", "shared_down"):
        assert methods[name] is UnquantizedLinearMethod
    # Wrapping twice is a no-op.
    assert fp8.enable_glm53_fp8_dense(model) == 0


def test_fp8_dense_sub_flags(monkeypatch, tp_size) -> None:
    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE", "1")
    monkeypatch.setenv("VLLM_GLM53_FP8_FFN", "0")
    model = _toy_model(TP3_SHAPES)
    assert fp8.enable_glm53_fp8_dense(model) == 4
    assert _methods(model)["ffn_down"] is UnquantizedLinearMethod

    monkeypatch.setenv("VLLM_GLM53_FP8_MTP", "0")
    draft = _toy_model({"dsa_o_proj": TP3_SHAPES["dsa_o_proj"]})
    assert fp8.enable_glm53_fp8_dense(draft, draft=True) == 0
    monkeypatch.setenv("VLLM_GLM53_FP8_MTP", "1")
    assert fp8.enable_glm53_fp8_dense(draft, draft=True) == 1


def test_fp8_prefill_and_batch_flags(monkeypatch) -> None:
    assert fp8.prefill_mode() == "bf16"
    assert fp8.max_m() == 32
    monkeypatch.setenv("VLLM_GLM53_FP8_PREFILL", " W8A8 ")
    assert fp8.prefill_mode() == "w8a8"
    monkeypatch.setenv("VLLM_GLM53_FP8_PREFILL", "fp8")
    with pytest.raises(ValueError, match="bf16 or w8a8"):
        fp8.prefill_mode()
    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE_MAX_M", "0")
    with pytest.raises(ValueError):
        fp8.max_m()


def test_fp8_decode_eligibility() -> None:
    x = torch.empty(32, 8, dtype=torch.bfloat16)
    assert fp8._decode_eligible(x, None, 32)
    assert not fp8._decode_eligible(torch.empty(33, 8, dtype=torch.bfloat16), None, 32)
    assert not fp8._decode_eligible(x.float(), None, 32)
    assert not fp8._decode_eligible(x, torch.empty(8), 32)
    assert not fp8._decode_eligible(torch.empty(0, 8, dtype=torch.bfloat16), None, 32)


def _lm_head(monkeypatch, size: int) -> ParallelLMHead:
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: size
    )
    with torch.device("meta"):
        return ParallelLMHead(154880, 4096, params_dtype=torch.bfloat16)


def test_fp8_lm_head_only_at_tp3(monkeypatch, tp_size) -> None:
    head = _lm_head(monkeypatch, 3)
    assert tuple(head.weight.shape) == fp8.GLM53_TP3_LM_HEAD_SHAPE
    assert not fp8.enable_glm53_fp8_lm_head(head)

    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE", "1")
    monkeypatch.setenv("VLLM_GLM53_FP8_LM_HEAD", "0")
    assert not fp8.enable_glm53_fp8_lm_head(head)
    monkeypatch.setenv("VLLM_GLM53_FP8_LM_HEAD", "1")
    assert fp8.enable_glm53_fp8_lm_head(head)
    method = head.quant_method
    assert isinstance(method, fp8.GLM53Fp8DecodeLMHeadMethod)
    # LogitsProcessor must route through apply(), not its raw-weight branch.
    assert not isinstance(method, UnquantizedEmbeddingMethod | UnquantizedLinearMethod)
    # The BF16 weight stays for the MTP draft head copy and large batches.
    assert head.weight.dtype == torch.bfloat16

    tp_size(4)
    head4 = _lm_head(monkeypatch, 4)
    assert not fp8.enable_glm53_fp8_lm_head(head4)
    assert type(head4.quant_method) is UnquantizedEmbeddingMethod


# ---------------------------------------------------------------------------
# Low-latency GEMM plan
# ---------------------------------------------------------------------------


@pytest.fixture
def sm120(monkeypatch):
    requested: list = []
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 120
    )
    monkeypatch.setattr(llg.shape_dynamic_skinny_gemm, "is_available", lambda: True)
    monkeypatch.setattr(
        llg.shape_dynamic_skinny_gemm,
        "request_warmup_configs",
        lambda dtype, configs: requested.append((dtype, set(configs))),
    )
    return requested


def test_low_latency_plan_configs_fit_the_kernel() -> None:
    for (n, k), ms in llg.GLM53_TP3_PLAN.items():
        plan = llg.plan_for(n, k)
        assert set(plan) == set(ms), (n, k)
        for m, config in plan.items():
            assert llg._config_fits(config, m, n, k)


def test_low_latency_gemm_swaps_tp3_shapes(tp_size, sm120) -> None:
    model = _toy_model(TP3_SHAPES)
    assert llg.enable_glm53_low_latency_gemm(model, torch.bfloat16) == 7
    methods = _methods(model)
    for name in ("ffn_gate_up", "ffn_down", "shared_down"):
        assert methods[name] is UnquantizedLinearMethod
    assert methods["kda_in_proj"] is llg.GLM53LowLatencyLinearMethod
    assert len(sm120) == 1 and sm120[0][0] is torch.bfloat16


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_low_latency_gemm_is_tp3_only(monkeypatch, tp_size, sm120, size) -> None:
    monkeypatch.setenv("VLLM_GLM53_LOW_LATENCY_GEMM", "1")
    tp_size(size)
    model = _toy_model(TP4_SHAPES)
    assert llg.enable_glm53_low_latency_gemm(model, torch.bfloat16) == 0
    assert _all_plain(model)
    assert not sm120


def test_low_latency_gemm_opt_out_and_guards(monkeypatch, tp_size, sm120) -> None:
    model = _toy_model(TP3_SHAPES)
    assert llg.enable_glm53_low_latency_gemm(model, torch.float16) == 0
    monkeypatch.setenv("VLLM_GLM53_LOW_LATENCY_GEMM", "0")
    assert llg.enable_glm53_low_latency_gemm(model, torch.bfloat16) == 0
    monkeypatch.delenv("VLLM_GLM53_LOW_LATENCY_GEMM")
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: False
    )
    assert llg.enable_glm53_low_latency_gemm(model, torch.bfloat16) == 0
    assert _all_plain(model)


def test_low_latency_dispatch_uses_measured_batch_sizes(monkeypatch) -> None:
    method = llg.GLM53LowLatencyLinearMethod(llg.plan_for(2048, 4096))
    calls: list[int] = []

    def fake_skinny_gemm(x, weight, config):
        calls.append(config.num_rows)
        return "skinny"

    monkeypatch.setattr(llg, "shape_dynamic_skinny_gemm", fake_skinny_gemm)
    monkeypatch.setattr(
        UnquantizedLinearMethod, "apply", lambda self, layer, x, bias=None: "cublas"
    )
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.zeros(2048, 4096, dtype=torch.bfloat16))
    for m, expected in ((1, "skinny"), (2, "skinny"), (3, "cublas"), (4, "cublas")):
        x = torch.zeros(m, 4096, dtype=torch.bfloat16)
        assert method.apply(layer, x) == expected
    x = torch.zeros(1, 4096, dtype=torch.bfloat16)
    assert method.apply(layer, x, torch.zeros(2048)) == "cublas"
    assert calls == [1, 2]


def test_fp8_wraps_low_latency_projections(monkeypatch, tp_size, sm120) -> None:
    monkeypatch.setenv("VLLM_GLM53_FP8_DENSE", "1")
    model = _toy_model(TP3_SHAPES)
    llg.enable_glm53_low_latency_gemm(model, torch.bfloat16)
    fp8.enable_glm53_fp8_dense(model)
    method = model.kda_in_proj.quant_method
    assert isinstance(method, fp8.GLM53Fp8DecodeLinearMethod)
    # Batches above the FP8 threshold fall back to the wrapped method.
    assert isinstance(method._inner, llg.GLM53LowLatencyLinearMethod)
    assert type(model.dsa_fused_qkv_a.quant_method) is llg.GLM53LowLatencyLinearMethod
