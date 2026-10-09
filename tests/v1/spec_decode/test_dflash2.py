# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import (
    _add_global_draft_layer_exclusions,
)
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize(
    ("target_format", "draft_format"), [("fastsafetensors", "auto"), ("b12x", "b12x")]
)
def test_dflash_loader_honors_draft_load_config(
    monkeypatch, target_format, draft_format
):
    from vllm.config import LoadConfig

    draft_load_config = LoadConfig(load_format=target_format)
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    speculative_config = SimpleNamespace(
        attention_backend=None,
        draft_load_config=draft_load_config,
        draft_model_config=draft_model_config,
        kv_cache_dtype=None,
    )
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        load_config=LoadConfig(load_format=target_format),
        speculative_config=speculative_config,
    )
    loaded = SimpleNamespace(model=SimpleNamespace())
    captured = {}

    def fake_replace(config, **changes):
        values = vars(config).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return loaded

    monkeypatch.setattr(dflash_utils, "replace", fake_replace)
    monkeypatch.setattr(dflash_utils, "get_model", fake_get_model)
    monkeypatch.setattr(dflash_utils, "maybe_share_target_embed", lambda *_args: None)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag",
        lambda _tag: nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _config: True,
    )

    assert dflash_utils.load_dflash_model(SimpleNamespace(), vllm_config) is loaded
    assert captured["load_config"].load_format == draft_format
    assert captured["vllm_config"].load_config.load_format == draft_format
    assert vllm_config.load_config.load_format == target_format


def test_dflash_reset_attn_releases_cache_layout_state():
    speculator = object.__new__(DFlashSpeculator)
    cache_derived_fields = (
        "model_state",
        "kv_cache_config",
        "attn_groups",
        "attn_cg_support",
        "block_tables",
        "target_attn_groups",
        "draft_kv_cache_group_ids",
        "draft_kv_cache_group_id",
        "_context_slot_mappings",
        "_layer_group_idx",
        "_group_causal",
    )
    for name in cache_derived_fields:
        setattr(speculator, name, object())
    speculator.query_cudagraph_manager = object()

    speculator.reset_attn()

    assert all(not hasattr(speculator, name) for name in cache_derived_fields)
    assert speculator.query_cudagraph_manager is None


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_groups,group_size,block_size,taps",
    [
        (7, 13, 11, 5, 1),
        (7, 13, 11, 5, 3),
        (7, 13, 11, 8, 2),
        (16, 64, 16, 8, 2),
        (16, 160, 16, 8, 2),
    ],
)
def test_grouped_conv_triton_matches_reference(
    dtype: torch.dtype,
    batch: int,
    num_groups: int,
    group_size: int,
    block_size: int,
    taps: int,
):
    torch.manual_seed(0)
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size, device="cuda", dtype=dtype)
    base = torch.randn(taps, num_groups * group_size, device="cuda", dtype=dtype)
    projected = torch.randn(rows, 2, taps, num_groups, device="cuda", dtype=dtype)
    delta = projected[:, 1]

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )

    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    if dtype is torch.bfloat16 and taps == 2 and group_size == 16 and block_size == 8:
        expected = _grouped_conv(
            hidden.cpu(),
            delta.cpu(),
            base.cpu(),
            block_size,
            num_groups,
            group_size,
            taps,
        ).to(actual.device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    torch.testing.assert_close(
        actual,
        expected.flatten(0, 1).flatten(-2).to(dtype),
        rtol=1e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype is torch.bfloat16 else 1e-5,
    )


def test_draft_quant_exclusions_include_global_layer_indices():
    quant_config = SimpleNamespace(
        exclude_modules=[
            "layers.0.mlp_conv*",
            "*layers.4.self_attn.q_proj",
            "layers.88.already_global",
            "lilicorr.layers.0.mlp.0",
        ]
    )

    _add_global_draft_layer_exclusions(quant_config, 88, 5)

    assert "layers.88.mlp_conv*" in quant_config.exclude_modules
    assert "*layers.92.self_attn.q_proj" in quant_config.exclude_modules
    assert quant_config.exclude_modules.count("layers.88.already_global") == 1
    assert "lilicorr.layers.88.mlp.0" not in quant_config.exclude_modules


def test_dflash2_auxiliary_linears_use_draft_quantization(
    monkeypatch, default_vllm_config
):
    """Every checkpoint-serialized DFlash2 linear receives its quant config."""
    from torch import nn

    import vllm.model_executor.models.qwen3_dflash2 as dflash2

    class StubLinear(nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            self.quant_config = quant_config

    monkeypatch.setattr(dflash2, "ReplicatedLinear", StubLinear)
    quant_config = object()
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(default_vllm_config):
        grouped_conv = dflash2.DFlashGroupedConv(
            hidden_size=16,
            taps=2,
            group_size=4,
            block_size=8,
            params_dtype=torch.float32,
            quant_config=quant_config,
            prefix="attention_conv",
        )
        selector = dflash2.CandidateSelector(
            hidden_size=16,
            vocab_size=32,
            rank=4,
            top_k=3,
            params_dtype=torch.float32,
            quant_config=quant_config,
            prefix="candidate_selector",
        )

    assert grouped_conv.kernel_projection.quant_config is quant_config
    assert selector.hidden_projection.quant_config is quant_config


@pytest.mark.parametrize("mxfp8_layer", [0, 1])
def test_dflash_context_projection_rejects_mixed_quantization(mxfp8_layer: int):
    from torch import nn

    from vllm.model_executor.layers.quantization.modelopt import (
        build_linear_method,
    )
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    class UnreadableWeight:
        def __getitem__(self, _key):
            raise AssertionError("weights must not be read before validation")

    mxfp8_method = build_linear_method(None, "MXFP8", "")
    methods = [build_linear_method(None, "FP8", ""), None]
    methods[mxfp8_layer] = mxfp8_method
    layers_attn = [
        SimpleNamespace(
            qkv_proj=SimpleNamespace(
                quant_method=method,
                q_size=1,
                weight=UnreadableWeight(),
            )
        )
        for method in methods
    ]
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)

    with pytest.raises(ValueError, match="Every DFlash attention layer"):
        model._build_context_kv_buffers(layers_attn, has_bias=False)


def test_dflash_bf16_context_projection_preserves_weights():
    from torch import nn

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    torch.manual_seed(7)
    layers_attn = [
        SimpleNamespace(
            q_size=4,
            qkv_proj=SimpleNamespace(
                quant_method=UnquantizedLinearMethod(),
                weight=torch.randn(8, 32, dtype=torch.bfloat16),
            ),
            k_norm=SimpleNamespace(weight=torch.ones(2, dtype=torch.bfloat16)),
        )
        for _ in range(2)
    ]
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(weight=torch.ones(32, dtype=torch.bfloat16))
    model.layers = [SimpleNamespace(self_attn=attn) for attn in layers_attn]
    model._build_context_kv_buffers(layers_attn, has_bias=False)
    expected = torch.cat([attn.qkv_proj.weight[4:] for attn in layers_attn])
    torch.testing.assert_close(model._fused_kv_weight, expected, rtol=0, atol=0)
    assert model._fused_kv_weight_scale is None
    model.process_weights_after_loading()
    torch.testing.assert_close(model._fused_kv_weight, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dflash_serialized_mxfp8_context_has_independent_kernel(monkeypatch):
    from torch import nn

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import (
        build_linear_method,
    )
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.utils.torch_utils import set_default_torch_dtype

    monkeypatch.setenv(
        "VLLM_DISABLED_KERNELS",
        "B12xMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel",
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    torch.manual_seed(9)
    layers_attn = []
    expected_weights = []
    with (
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        for _ in range(2):
            method = build_linear_method(None, "MXFP8", "")
            linear = nn.Module()
            linear.quant_method = method
            method.create_weights(linear, 512, [512], 512, 512, torch.bfloat16)
            linear.weight.data.copy_(torch.randn(512, 512).to(torch.float8_e4m3fn))
            linear.weight_scale.data.fill_(127)
            expected_weights.append(linear.weight[256:].to(torch.bfloat16))
            layers_attn.append(
                SimpleNamespace(
                    q_size=256,
                    qkv_proj=linear,
                    k_norm=SimpleNamespace(weight=torch.ones(128)),
                )
            )
        model = object.__new__(DFlashQwen3Model)
        nn.Module.__init__(model)
        model.hidden_norm = SimpleNamespace(weight=torch.ones(512))
        model.layers = [SimpleNamespace(self_attn=attn) for attn in layers_attn]
        model._fused_kv_linear = nn.Module()
        model._build_context_kv_buffers(layers_attn, has_bias=False)
        for attn in layers_attn:
            attn.qkv_proj.quant_method.process_weights_after_loading(attn.qkv_proj)
        source_kernel = layers_attn[0].qkv_proj.quant_method.kernel
        source_weight = layers_attn[0].qkv_proj.weight.clone()
        model.process_weights_after_loading()
        fused_method = model._fused_kv_quant_method
        assert fused_method is not None
        assert fused_method.kernel is not source_kernel
        torch.testing.assert_close(
            layers_attn[0].qkv_proj.weight, source_weight, rtol=0, atol=0
        )
        expected_weight = torch.cat(expected_weights)
        for rows in (1, 7, 256):
            x = torch.randn(rows, 512)

            def run(x=x):
                return fused_method.apply(model._fused_kv_linear, x)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for _ in range(2):
                x.normal_()
                graph.replay()
                expected = torch.nn.functional.linear(x, expected_weight)
                assert torch.isfinite(actual).all() and actual.abs().max() > 0
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(), expected.float().flatten(), dim=0
                )
                assert cosine > 0.9999


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_context_projection_owns_its_linear_method(monkeypatch, mxfp8, has_bias):
    """Fused K/V packing must not reconfigure the query projection's kernel."""
    from torch import nn

    import vllm.model_executor.layers.quantization.modelopt as modelopt
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    kernels = []

    def select_kernel(spec, layer, runtime_dtypes, **kwargs):
        kernel = SimpleNamespace(
            process_weights_after_loading=lambda layer: None,
            input_quant_key=lambda: None,
        )
        kernels.append((kernel, layer.output_size_per_partition))
        assert runtime_dtypes.input_dtype == torch.bfloat16
        assert runtime_dtypes.out_dtype == torch.bfloat16
        assert runtime_dtypes.marlin_input_dtype == torch.float16
        assert layer.has_bias is has_bias
        return kernel

    monkeypatch.setattr(modelopt, "select_linear_kernel", select_kernel)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    method = (
        modelopt.build_linear_method(None, "MXFP8", "")
        if mxfp8
        else UnquantizedLinearMethod()
    )
    original_kernel = object()
    if mxfp8:
        assert isinstance(method, modelopt.ModelOptLinearMethod)
        method.kernel = original_kernel
        method.input_dtype = method.out_dtype = torch.bfloat16
        method.marlin_input_dtype = torch.float16
    layers = []
    for index in range(2):
        projection = nn.Module()
        projection.quant_method = method
        weight = torch.full((192, 64), index + 1, dtype=torch.bfloat16)
        if mxfp8:
            weight = weight.to(torch.float8_e4m3fn)
            projection.weight_scale = nn.Parameter(
                torch.full((192, 2), 127, dtype=torch.uint8), requires_grad=False
            )
        projection.weight = nn.Parameter(weight, requires_grad=False)
        if has_bias:
            projection.bias = nn.Parameter(torch.full((192,), float(index + 1)))
        attention = SimpleNamespace(
            qkv_proj=projection,
            q_size=128,
            k_norm=SimpleNamespace(weight=torch.ones(32)),
        )
        layers.append(SimpleNamespace(self_attn=attention))
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.layers = layers
    model.hidden_norm = SimpleNamespace(weight=torch.ones(64, dtype=torch.bfloat16))
    model._fused_kv_linear = nn.Module()
    model._fused_kv_quant_method = None
    model._build_context_kv_buffers([layer.self_attn for layer in layers], has_bias)
    assert model._fused_kv_weight is not None
    expected = model._fused_kv_weight.clone()
    model.process_weights_after_loading()
    if mxfp8:
        assert isinstance(method, modelopt.ModelOptLinearMethod)
        assert model._fused_kv_quant_method is not None
        assert model._fused_kv_quant_method is not method
        assert method.kernel is original_kernel
        assert kernels == [(model._fused_kv_quant_method.kernel, 128)]
        assert model._fused_kv_linear.weight_block_size == [1, 32]
        torch.testing.assert_close(
            model._fused_kv_linear.weight.float(), expected.float()
        )
        assert model._fused_kv_weight is None
        assert model._fused_kv_weight_scale is None
    else:
        assert not kernels
        assert model._fused_kv_quant_method is None
        torch.testing.assert_close(model._fused_kv_weight, expected)


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("variant", "aux_staging"),
    [
        ("dflash2", None),
        ("lilicorr", None),
        ("lilicorr_plain", None),
        ("dflash2", "mxfp8"),
        ("dflash2", "bf16"),
    ],
)
def test_candidate_model_decoder_layer_cls(monkeypatch, variant, aux_staging):
    from types import SimpleNamespace

    from torch import nn

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )
    from vllm.model_executor.models import qwen3_dflash
    from vllm.model_executor.models.lilicorr import LiLiCorr
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    monkeypatch.setenv(
        "VLLM_DFLASH_AUX_MXFP8_STREAMING", str(int(aux_staging == "mxfp8"))
    )
    monkeypatch.setenv("VLLM_DFLASH_AUX_BF16_STAGING", str(int(aux_staging == "bf16")))
    monkeypatch.setenv("VLLM_DFLASH_SHARD_AUX_PROJECTION", "0")
    if aux_staging is not None:

        class AuxiliaryProjection(nn.Module):
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(output_size, input_size))
                self.quant_method = Mxfp8OnlineLinearMethod()

        monkeypatch.setattr(qwen3_dflash, "ReplicatedLinear", AuxiliaryProjection)
        monkeypatch.setattr(qwen3_dflash, "_get_dflash_fc_input_size", lambda _: 512)

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 0 if variant == "lilicorr_plain" else 3,
            "conv_group_size": 0 if variant == "lilicorr_plain" else 2,
            "block_size": 5,
            "lilicorr_candidate_topk": 4,
            "lilicorr_hidden_size": 8,
            "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 2,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 4,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 3.0,
            "use_aux_hidden_state": aux_staging is not None,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model_cls = DFlash2Qwen3Model if variant == "dflash2" else LiLiCorr
        model = model_cls(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    expected = (
        DFlashQwen3DecoderLayer
        if variant == "lilicorr_plain"
        else DFlash2Qwen3DecoderLayer
    )
    assert type(model.layers[0]) is expected
    if aux_staging is not None:
        method = model.fc.quant_method
        assert isinstance(method, Mxfp8OnlineLinearMethod)
        kernel = method.kernel
        assert kernel.config.weight_shape == (256, 512)
        expected_kernel = (
            "B12xMxfp8LinearKernel"
            if aux_staging == "mxfp8"
            else "MarlinMxfp8LinearKernel"
        )
        assert type(kernel).__name__ == expected_kernel


def test_conv_projections_use_draft_quant_config(monkeypatch):
    from torch import nn

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        DFlashQwen3DecoderLayer,
        "__init__",
        lambda self, *a, **kw: nn.Module.__init__(self),
    )
    # Exercise the real quantized parameter allocation without selecting a GPU kernel.
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    layer = DFlash2Qwen3DecoderLayer(
        SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        config=SimpleNamespace(
            hidden_size=16,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 2},
        ),
        layer_idx=0,
        prefix="model.layers.0",
        quant_config=quant_config,
    )
    for name in ("attention_conv", "mlp_conv"):
        module = getattr(layer, name)
        projection = module.kernel_projection
        assert projection.weight.dtype == torch.uint8
        assert projection.weight.shape == (32, 8)
        weight = torch.ones_like(projection.weight)
        AutoWeightsLoader(module).load_weights([("kernel_projection.weight", weight)])
        torch.testing.assert_close(projection.weight, weight)
        assert module.base_kernel.dtype == torch.bfloat16


def test_context_kv_uses_quantized_projection_fallback(monkeypatch):
    from torch import nn

    from vllm.model_executor.models import qwen3_dflash

    class Projection(nn.Module):
        def __init__(self, packed_weight):
            super().__init__()
            self.register_buffer("packed_weight", packed_weight)
            self.quant_method = object()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.packed_weight), None

    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, hidden_states, weight, eps: output.copy_(hidden_states),
    )
    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projections = [
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        ),
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        ),
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        _rms_norm_eps=1e-6,
        _fused_kv_quant_method=None,
    )
    layers_attn = [
        SimpleNamespace(
            qkv_proj=projection,
            q_size=2,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        )
        for projection in projections
    ]
    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(
        model,
        layers_attn,
        has_bias=False,
    )
    assert model._fused_kv_weight is None
    assert all(not hasattr(projection, "weight") for projection in projections)

    actual_k, actual_v = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        context_states,
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
    )

    expected_k = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).unsqueeze(2)
    expected_v = torch.tensor(
        [[[3.0, -1.0], [7.0, -1.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert [projection.calls for projection in projections] == [1, 1]


def test_fused_context_projection_dequantizes_block_fp8_rows():
    """Online ``fp8_per_block`` drafters quantize while loading, so the fused
    context projection receives FP8 rows; it must run on their dequantized
    values (with their block scales), not on raw FP8 bytes."""
    from torch import nn

    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    torch.manual_seed(0)
    layers, expected = [], []
    for _ in range(2):
        projection = nn.Module()
        projection.quant_method = None
        scale = torch.rand(2, 2) + 0.5
        full_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        dense = torch.randn(256, 256)
        weight = (dense / full_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        projection.weight = nn.Parameter(weight, requires_grad=False)
        projection.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
        projection.weight_block_size = [128, 128]
        layers.append(
            SimpleNamespace(
                qkv_proj=projection,
                q_size=128,
                k_norm=SimpleNamespace(weight=torch.ones(32)),
            )
        )
        expected.append((weight.float() * full_scale)[128:].to(torch.bfloat16))
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = SimpleNamespace(weight=torch.ones(256, dtype=torch.bfloat16))
    model._build_context_kv_buffers(layers, has_bias=False)
    assert model._fused_kv_weight is not None
    assert model._fused_kv_weight.dtype == torch.bfloat16
    assert torch.equal(model._fused_kv_weight, torch.cat(expected))
