# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

PROMPT = "On the surface of Mars, we found"

MODELS_QUANT = [
    ("ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head", True),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", False),
]


@pytest.fixture
def mxfp8_head_config(monkeypatch, default_vllm_config):
    from vllm.model_executor import parameter

    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, head_dtype=None
    )
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    return default_vllm_config


@pytest.mark.cpu_test
@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_mxfp8_only_selects_lm_head(monkeypatch, mxfp8_head_config, enabled):
    from vllm.model_executor.layers.quantization.online import mxfp8

    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", str(int(enabled)))
    monkeypatch.setattr(mxfp8, "init_mxfp8_linear_kernel", lambda: None)
    head = ParallelLMHead(256, 128, params_dtype=torch.bfloat16, disable_tp=True)
    embedding = VocabParallelEmbedding(
        256, 128, params_dtype=torch.bfloat16, disable_tp=True
    )

    expected = mxfp8.Mxfp8OnlineLinearMethod if enabled else UnquantizedEmbeddingMethod
    assert isinstance(head.quant_method, expected)
    assert isinstance(embedding.quant_method, UnquantizedEmbeddingMethod)
    assert head.weight.dtype == embedding.weight.dtype == torch.bfloat16
    assert head.weight.shape == embedding.weight.shape == (256, 128)


@pytest.mark.cpu_test
def test_runtime_mxfp8_rejects_quantized_checkpoint(monkeypatch, mxfp8_head_config):
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "1")
    quant_config = SimpleNamespace(get_quant_method=lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="requires an unquantized LM head"):
        ParallelLMHead(256, 128, quant_config=quant_config, disable_tp=True)


@pytest.mark.cpu_test
@pytest.mark.parametrize("recipe", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize(
    "fallback", ["dtype", "shape", "backend", "platform", "tied", "quantized"]
)
def test_runtime_lm_head_defaults_preserve_ineligible_heads(
    monkeypatch, mxfp8_head_config, recipe, fallback
):
    """Automatic quantization must preserve unsupported checkpoint/head paths."""
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab

    for name in ("VLLM_MXFP8_LM_HEAD", "VLLM_MTP_NVFP4_LM_HEAD", "VLLM_LM_HEAD_A16"):
        monkeypatch.delenv(name, raising=False)
    mxfp8_head_config.kernel_config.linear_backend = "b12x"
    mxfp8_head_config.model_config.hf_text_config = SimpleNamespace(
        tie_word_embeddings=fallback == "tied"
    )
    if fallback in ("tied", "quantized"):
        monkeypatch.setattr(
            vocab, "_supports_default_lm_head_quantization", lambda *a: True
        )
    if fallback == "backend":
        mxfp8_head_config.kernel_config.linear_backend = "cutlass"
    if fallback == "platform":
        monkeypatch.setattr(vocab.current_platform, "is_cuda", lambda: False)
    existing = SimpleNamespace(create_weights=lambda *args, **kwargs: None)
    quant_config = (
        SimpleNamespace(get_quant_method=lambda *a, **kw: existing)
        if fallback == "quantized"
        else None
    )
    dtype = torch.float16 if fallback == "dtype" else torch.bfloat16
    head = ParallelLMHead(
        256,
        64 if fallback == "shape" else 128,
        params_dtype=dtype,
        disable_tp=True,
        quant_config=quant_config,
        lm_head_quantization="nvfp4" if recipe == "nvfp4" else None,
    )
    assert head.runtime_lm_head_quantization is None
    if fallback == "quantized":
        assert head.quant_method is existing
    else:
        assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)
        assert head.weight.dtype == dtype


@pytest.mark.parametrize("use_a16", [False, True, None])
def test_draft_nvfp4_head_preserves_verifier_and_dynamic_graph_scales(
    monkeypatch, mxfp8_head_config, use_a16
):
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        break_fp4_bytes,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    if not current_platform.is_device_capability_family(120):
        pytest.skip("Requires b12x on SM120/SM121")
    if use_a16 is None:
        for name in (
            "VLLM_MXFP8_LM_HEAD",
            "VLLM_LM_HEAD_A16",
            "VLLM_MTP_NVFP4_LM_HEAD",
        ):
            monkeypatch.delenv(name, raising=False)
        use_a16 = True
    else:
        monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "1")
        monkeypatch.setenv("VLLM_LM_HEAD_A16", str(int(use_a16)))
    mxfp8_head_config.kernel_config.linear_backend = "b12x"
    torch.manual_seed(42)
    weight = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda") * 0.02
    with torch.device("cuda"):
        verifier = ParallelLMHead(
            256, 128, params_dtype=torch.bfloat16, disable_tp=True
        )
        draft = ParallelLMHead(
            256,
            128,
            params_dtype=torch.bfloat16,
            disable_tp=True,
            lm_head_quantization="nvfp4",
        )
    for head in (verifier, draft):
        head.weight.weight_loader(head.weight, weight)
        head.quant_method.process_weights_after_loading(head)
    assert isinstance(draft.quant_method, Nvfp4OnlineLinearMethod)
    verifier_values = verifier.b12x_mxfp8_packed_weight.weight.values.clone()
    assert not _should_share(
        SimpleNamespace(has_own_lm_head=True), "has_own_lm_head", draft, verifier
    )
    packed_ptr = draft.weight.data_ptr()
    draft.quant_method.process_weights_after_loading(draft)
    assert draft.weight.data_ptr() == packed_ptr
    w_inv = 2688.0 / weight.abs().amax().float().clamp_min(1e-8)
    qw, sw = scaled_fp4_quant(weight, w_inv, is_sf_swizzled_layout=False)
    torch.testing.assert_close(draft.weight, qw, rtol=0, atol=0)
    w_ref = (
        break_fp4_bytes(qw, torch.float32) * sw.float().repeat_interleave(16, 1) / w_inv
    )
    if use_a16:
        w_ref = (
            break_fp4_bytes(qw, torch.float32) * sw.float().repeat_interleave(16, 1)
        ).bfloat16().float() / w_inv
        draft.b12x_warmup_provider.get_b12x_warmup_unit(
            draft, (1, 4), torch.bfloat16
        ).compile()
    for tokens in (1, 4):
        x = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")

        def project(x=x):
            return draft.quant_method.apply(draft, x)

        for _ in range(3):
            project()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = project()
        for amplitude in (0.0, 0.01, 1.0, 32.0):
            x.normal_().mul_(amplitude)
            allocated_before = torch.accelerator.memory_allocated()
            graph.replay()
            assert torch.accelerator.memory_allocated() == allocated_before
            x_inv = 2688.0 / x.abs().amax().float().clamp_min(1e-8)
            qx, sx = scaled_fp4_quant(x, x_inv, is_sf_swizzled_layout=False)
            x_ref = (
                break_fp4_bytes(qx, torch.float32)
                * sx.float().repeat_interleave(16, 1)
                / x_inv
            )
            if use_a16:
                x_ref = x.float()
            expected = (x_ref @ w_ref.T).to(torch.bfloat16)
            torch.testing.assert_close(result, expected, rtol=0.016, atol=0.001)
            assert torch.isfinite(result).all()
    torch.testing.assert_close(
        verifier.b12x_mxfp8_packed_weight.weight.values, verifier_values, rtol=0, atol=0
    )


@pytest.mark.parametrize("use_a16", [False, True, None])
def test_runtime_mxfp8_b12x_shard_loading_and_graph(
    monkeypatch, mxfp8_head_config, use_a16
):
    from vllm.model_executor.layers import vocab_parallel_embedding
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )
    from vllm.platforms import current_platform
    from vllm.utils.b12x import get_b12x_mxfp8_linear

    b12x = get_b12x_mxfp8_linear()
    if not current_platform.is_device_capability_family(120) or b12x is None:
        pytest.skip("Requires b12x on SM120/SM121")

    if use_a16 is None:
        for name in (
            "VLLM_MXFP8_LM_HEAD",
            "VLLM_LM_HEAD_A16",
            "VLLM_MTP_NVFP4_LM_HEAD",
        ):
            monkeypatch.delenv(name, raising=False)
        use_a16 = True
    else:
        monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "1")
        monkeypatch.setenv("VLLM_LM_HEAD_A16", str(int(use_a16)))
    mxfp8_head_config.kernel_config.linear_backend = "b12x"
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: 2
    )
    torch.manual_seed(0)
    weight = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda") * 0.02
    with torch.device("cuda"):
        head = ParallelLMHead(256, 128, params_dtype=torch.bfloat16)
        processor = LogitsProcessor(256, lm_head=head)
    head.weight.weight_loader(head.weight, weight)
    head.quant_method.process_weights_after_loading(head)
    packed = head.b12x_mxfp8_packed_weight
    assert packed.out_features == 128
    assert head.weight.numel() == 0
    head.quant_method.process_weights_after_loading(head)
    assert head.b12x_mxfp8_packed_weight is packed

    values, scales = mxfp8_e4m3_quantize(weight[128:])
    torch.testing.assert_close(
        packed.weight.values.view(torch.uint8), values.view(torch.uint8)
    )
    reference_weight = dequant_mxfp8_to_bf16(values, scales).float()
    if use_a16:
        head.b12x_warmup_provider.get_b12x_warmup_unit(
            head, (1, 3), torch.bfloat16
        ).compile()
    for tokens in (1, 3):
        x = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")

        def project(x=x):
            return processor(head, x, skip_gather=True)

        for _ in range(2):
            project()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = project()
        for _ in range(2):
            x.normal_()
            graph.replay()
            x_values, x_scales = mxfp8_e4m3_quantize(x)
            reference_x = dequant_mxfp8_to_bf16(x_values, x_scales).float()
            if use_a16:
                reference_x = x.float()
            expected = torch.nn.functional.linear(reference_x, reference_weight)
            torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("model_id, lm_head_quantized", MODELS_QUANT)
def test_lm_head(
    vllm_runner,
    model_id: str,
    lm_head_quantized: bool,
    monkeypatch,
) -> None:
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        model_id, dtype=torch.float16, max_model_len=2048, enforce_eager=True
    ) as vllm_model:

        def check_model(model):
            lm_head_layer = model.lm_head
            if lm_head_quantized:
                assert isinstance(
                    lm_head_layer.quant_method,
                    AutoGPTQLinearMethod,
                )
            else:
                assert isinstance(
                    lm_head_layer.quant_method, UnquantizedEmbeddingMethod
                )

        vllm_model.apply_model(check_model)

        print(vllm_model.generate_greedy(["Hello my name is"], max_tokens=4)[0][1])
