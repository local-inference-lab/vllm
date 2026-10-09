# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The adapter preserves vLLM's indexed source selection and owned inputs."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("b12x")

from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._progress import CheckpointDisplay

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.b12x_loader import B12xModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@pytest.mark.parametrize("register_plugin", [False, True])
def test_load_format_selects_native_b12x_with_or_without_plugin(
    monkeypatch, register_plugin
):
    from vllm.model_executor import model_loader

    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(
        model_loader,
        "_LOAD_FORMAT_TO_MODEL_LOADER",
        dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER),
    )
    model_loader._LOAD_FORMAT_TO_MODEL_LOADER.pop("b12x", None)
    if register_plugin:
        # FlashInfer's B12X ships without the standalone vLLM plugin package.
        plugin = pytest.importorskip("b12x.integration.vllm.loader")
        plugin.register_b12x_loader()
    config = LoadConfig(
        load_format="b12x", model_loader_extra_config={"read_mode": "bounce"}
    )
    loader = model_loader.get_model_loader(config)
    assert isinstance(loader, B12xModelLoader)
    assert loader.read_mode == "bounce"
    assert config.load_format == "b12x"


def test_direct_loader_reports_missing_host_hooks_when_selected(monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.delattr(weight_utils, "file_source_tensor", raising=False)
    with pytest.raises(
        RuntimeError, match="requires vLLM file-source hooks.*file_source_tensor"
    ):
        B12xModelLoader(LoadConfig(load_format="b12x"))


@pytest.mark.parametrize("read_mode", ["gds", "bounce"])
@pytest.mark.parametrize("expandable", [False, True])
@pytest.mark.parametrize("fail_loading", [False, True])
def test_shared_loading_preserves_allocator_settings(
    monkeypatch, read_mode, expandable, fail_loading
):
    import vllm.model_executor.model_loader.b12x_loader as adapter
    from vllm.distributed import parallel_state

    original = f"max_split_size_mb:64,expandable_segments:{expandable}"
    settings = [original]
    monkeypatch.setattr(
        torch._C, "_accelerator_getAllocatorSettings", lambda: settings[0]
    )
    monkeypatch.setattr(
        torch._C,
        "_accelerator_setAllocatorSettings",
        lambda value: settings.__setitem__(0, value),
    )
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(cpu_group=object())
    )
    monkeypatch.setattr(adapter, "SharedReadGroup", lambda *_: object())

    @contextmanager
    def session(*args, **kwargs):
        yield SimpleNamespace(stats=lambda: {"payload_bytes": 0})

    monkeypatch.setattr(adapter, "DirectWeightSession", session)

    def load(loader, *args):
        expected = original.replace("True", "False") if read_mode == "gds" else original
        assert settings[0] == expected
        if fail_loading:
            raise RuntimeError("checkpoint read failed")
        loader.counter_before_loading_weights = 1.0
        loader.counter_after_loading_weights = 2.0
        return torch.nn.Module()

    monkeypatch.setattr(DefaultModelLoader, "load_model", load)
    loader = B12xModelLoader(
        LoadConfig(
            load_format="b12x",
            model_loader_extra_config={"read_mode": read_mode},
        )
    )
    config = SimpleNamespace(device_config=SimpleNamespace(device="cuda:0"))
    model_config = SimpleNamespace(enable_cumem_allocator=False)
    try:
        if fail_loading:
            with pytest.raises(RuntimeError, match="checkpoint read failed"):
                loader.load_model(config, model_config)
        else:
            loader.load_model(config, model_config)
    finally:
        assert settings[0] == original


def test_shared_gds_rejects_cumem_before_allocating_weights():
    loader = B12xModelLoader(
        LoadConfig(load_format="b12x", model_loader_extra_config={"read_mode": "gds"})
    )
    config = SimpleNamespace(device_config=SimpleNamespace(device="cuda:0"))
    with pytest.raises(ValueError, match="does not support the CuMem allocator"):
        loader.load_model(config, SimpleNamespace(enable_cumem_allocator=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC")
def test_shared_weight_allocations_export_with_expandable_segments_enabled():
    from vllm.model_executor.model_loader.b12x_loader import _ipc_weight_allocations

    original = torch._C._accelerator_getAllocatorSettings()
    configured = "max_split_size_mb:64,expandable_segments:True"
    try:
        torch._C._accelerator_setAllocatorSettings(configured)
        with DirectWeightSession(read_mode="gds") as session:
            native = session._gds
            executor = native.owner_create(
                session.device,
                session.io_threads,
                *(program.function for program in session._copy_programs),
            )
            try:
                with _ipc_weight_allocations():
                    assert torch._C._accelerator_getAllocatorSettings() == (
                        "max_split_size_mb:64,expandable_segments:False"
                    )
                    weights = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
                assert torch._C._accelerator_getAllocatorSettings() == configured
                base, size, handle = native.owner_export(
                    executor, weights.data_ptr(), weights.nbytes
                )
                assert base <= weights.data_ptr()
                assert weights.data_ptr() + weights.nbytes <= base + size
                assert len(handle) == 64
            finally:
                native.owner_close(executor)
    finally:
        torch._C._accelerator_setAllocatorSettings(original)


@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("read_mode", ["auto", "bounce"])
def test_draft_iterator_uses_index_and_retained_tensors_keep_their_bytes(
    tmp_path, capsys, show_progress, read_mode
):
    """Draft loading must not open unrelated target shards or reuse buffers."""
    save_file({"mtp.weight": torch.arange(16)}, tmp_path / "draft.safetensors")
    save_file({"mtp.bias": torch.arange(4) + 100}, tmp_path / "bias.safetensors")
    (tmp_path / "target.safetensors").write_bytes(b"must not be opened")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.weight": "draft.safetensors",
                    "mtp.bias": "bias.safetensors",
                    "model.weight": "target.safetensors",
                }
            }
        )
    )
    config = LoadConfig(
        load_format="b12x",
        use_tqdm_on_load=show_progress,
        model_loader_extra_config={"read_mode": read_mode},
    )
    loader = B12xModelLoader(config)
    source = DefaultModelLoader.Source(
        str(tmp_path), revision=None, prefix="draft.", weight_name_prefixes=("mtp.",)
    )
    with (
        DirectWeightSession(read_mode=loader.read_mode) as session,
        CheckpointDisplay(enabled=show_progress) as display,
    ):
        loader._session = session
        loader._progress = display
        retained = dict(loader._get_weights_iterator(source))
        assert set(retained) == {"draft.mtp.weight", "draft.mtp.bias"}
        values = {}
        for name, descriptor in retained.items():
            values[name] = torch.empty_like(descriptor, device="cuda")
            assert session(values[name], descriptor)
        loader._session = None
        loader._progress = None
    torch.testing.assert_close(values["draft.mtp.weight"].cpu(), torch.arange(16))
    torch.testing.assert_close(values["draft.mtp.bias"].cpu(), torch.arange(4) + 100)
    assert config.load_format == "b12x"
    assert config.model_loader_extra_config == {"read_mode": read_mode}
    progress = capsys.readouterr().err
    if show_progress:
        assert "b12x routing checkpoint shards" in progress
        assert "2/2 shards routed" in progress
        assert "0.00 GB selected on rank 0" in progress
        assert "it/s" not in progress
    else:
        assert "shards routed" not in progress


def test_gdn_convolution_shards_read_into_final_parameter_slices(tmp_path):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    path = tmp_path / "conv.safetensors"
    checkpoint = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    save_file({"conv.weight": checkpoint}, path)
    loader = mamba_v2_sharded_weight_loader(
        [(8, 0, 0), (4, 2, 1)], tp_size=2, tp_rank=1
    )
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        source = dict(session.weights([path]))["conv.weight"]
        target = torch.full((6, 2), -1.0, device="cuda")
        loader(target, source)
    torch.testing.assert_close(target.cpu(), checkpoint[4:])


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_hyperconnection_weights_load_without_allocator_hooks(
    tmp_path, monkeypatch, source_dtype
):
    """Loading the norm must preserve separate workspace contents."""
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.qwen4_exp.nvidia.hyperconnection import (
        GroupedGemmaRMSNorm,
        HyperConnectionConfig,
        HyperConnectionWorkspace,
    )

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.hyperconnection.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    checkpoint = torch.linspace(-0.01739, 0.01917, 128).to(source_dtype)
    expected = checkpoint.to(torch.bfloat16)
    path = tmp_path / "norm.safetensors"
    save_file({"weight": checkpoint}, path)
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        norm = GroupedGemmaRMSNorm(128, eps=1e-6, group_size=32, dtype=expected.dtype)
        workspace = HyperConnectionWorkspace(
            HyperConnectionConfig(
                hc_count=4,
                hidden_size=32,
                params_dtype=expected.dtype,
                hc_lowrank=16,
                rms_norm_eps=1e-6,
                hc_per_branch_norm=True,
            ),
            8,
        )
        loaded = AutoWeightsLoader(norm).load_weights(session.weights([path]))
        assert loaded == {"weight"}
    torch.testing.assert_close(norm.weight.cpu(), expected, rtol=0, atol=0)
    for buffer in workspace.buffers():
        buffer.fill_(7)
    torch.accelerator.synchronize()
    torch.testing.assert_close(norm.weight.cpu(), expected)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tp_size,rank", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
)
def test_dflash_sink_shards_preserve_file_sources(tmp_path, monkeypatch, tp_size, rank):
    """Route per-head sinks into final parameters, including padded TP tails."""
    from vllm.model_executor.models import qwen3_dflash as dflash
    from vllm.model_executor.weight_transfer import weight_transfer

    monkeypatch.setattr(dflash, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(dflash, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setenv("VLLM_DFLASH_COMPACT_ROPE", "0")
    for name in (
        "QKVParallelLinear",
        "RowParallelLinear",
        "DFlashAttention",
        "RMSNorm",
        "get_rope",
    ):
        monkeypatch.setattr(dflash, name, lambda *args, **kwargs: torch.nn.Identity())
    heads = 8
    padded_heads = ((heads + tp_size - 1) // tp_size) * tp_size
    expected = torch.arange(heads, dtype=torch.float32) / 4
    name = "layers.0.self_attn.attention_sink_bias"
    path = tmp_path / "dflash-sinks.safetensors"
    save_file({name: expected}, path)
    model = dflash.DFlashQwen3Model.__new__(dflash.DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_attention_heads=padded_heads)
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        model.layers[0].self_attn = dflash.DFlashQwen3Attention(
            hidden_size=padded_heads * 16,
            num_heads=padded_heads,
            num_kv_heads=1,
            head_dim=16,
            rope_parameters={},
            add_swa_attention_sink_bias=True,
        )
        assert model.load_weights(session.weights([path])) == {name}
    width = padded_heads // tp_size
    padded = torch.nn.functional.pad(expected, (0, padded_heads - heads))
    torch.testing.assert_close(
        model.layers[0].self_attn.attention_sink_bias.cpu(),
        padded[rank * width : (rank + 1) * width],
    )


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("quantization", ["mxfp8", "block_fp8"])
def test_glm_attention_dequantization_reads_owned_checkpoint_inputs(
    tmp_path, scale_first, quantization
):
    """Numerical projection transforms must consume payloads, not meta views."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.glm5next.common.model import Glm5NextModel

    prefix = "layers.3.self_attn"
    if quantization == "mxfp8":
        projection = target = "indexer.weights_proj"
        scale_name = "weight_scale"
        scale = torch.full((1, 1), 128, dtype=torch.uint8)
    else:
        projection, target = "q_a_proj", "fused_qkv_a_proj"
        scale_name = "weight_scale_inv"
        scale = torch.full((1, 1), 2.0, dtype=torch.float32)
    parameter_name = f"{prefix}.{target}.weight"
    weight = torch.arange(32).reshape(1, 32).to(torch.float8_e4m3fn)
    expected = weight.float() * 2
    path = tmp_path / "attention.safetensors"
    save_file(
        {
            f"{prefix}.{projection}.weight": weight,
            f"{prefix}.{projection}.{scale_name}": scale,
        },
        path,
    )

    class Projection(torch.nn.Module):
        is_fused_shared_expert_enabled = False
        config = SimpleNamespace(
            n_routed_experts=None,
            layer_types=["linear_attention"] * 4,
            mla_use_nope=False,
            qk_rope_head_dim=0,
            num_nextn_predict_layers=0,
            num_hidden_layers=4,
        )

        def named_parameters(self):
            return iter([(parameter_name, param)])

    with (
        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.nn.Parameter(
            torch.empty((1, 32), dtype=torch.float32, device="cuda")
        )
        param.weight_loader = lambda p, value, shard_id=None: default_weight_loader(
            p, value
        )
        sources = sorted(
            session.weights([path]),
            key=lambda pair: pair[0].endswith(".weight") == scale_first,
        )
        loaded = Glm5NextModel.load_weights(Projection(), iter(sources))
        assert loaded == {parameter_name}
    torch.testing.assert_close(param.cpu(), expected)


@pytest.mark.parametrize("rank", [0, 1])
def test_kda_convolution_loads_each_tp_shard_into_fused_weights(tmp_path, rank):
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        _make_fused_conv1d_weight_loader,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    path = tmp_path / "kda.safetensors"
    weights = {
        name: (torch.arange(32).reshape(8, 1, 4) + i * 64).to(torch.bfloat16)
        for i, name in enumerate(("q", "k", "v"))
    }
    save_file(weights, path)
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.empty((12, 1, 4), device="cuda")
        loader = _make_fused_conv1d_weight_loader([8, 8, 8], 2, rank)
        sources = dict(session.weights([path]))
        for i, name in enumerate(("q", "k", "v")):
            loader(param, sources[name], i)
    expected = torch.cat([weights[name][rank * 4 : (rank + 1) * 4] for name in weights])
    torch.testing.assert_close(param.cpu(), expected.float())


@pytest.mark.parametrize(
    "tp_size,rank", [(tp, rank) for tp in (2, 4) for rank in range(tp)]
)
def test_deepseek_sink_shards_are_flushed_before_derived_weights(
    tmp_path, monkeypatch, tp_size, rank
):
    """Padded sinks keep -inf and model post-load hooks consume completed reads."""
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.deepseek_v4.nvidia import model as ds4

    monkeypatch.setattr(ds4, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(ds4, "get_tensor_model_parallel_rank", lambda: rank)
    expected = torch.arange(64, dtype=torch.float32) / 4
    path = tmp_path / "sinks.safetensors"
    name = "layers.0.attn.attn_sink"
    save_file({name: expected}, path)

    class Model(torch.nn.Module):
        config = SimpleNamespace(num_attention_heads=64)
        quant_config = None
        use_sequence_parallel = False

        def __init__(self):
            super().__init__()
            self.sink = torch.nn.Parameter(
                torch.full((64,), -torch.inf, device="cuda"),
                requires_grad=False,
            )
            self.derived = None

        def named_parameters(self):
            return iter([(name, self.sink)])

        def get_expert_mapping(self):
            return []

        def finalize_mega_moe_weights(self):
            pass

        def finalize_mhc_broadcast_weights(self):
            self.derived = self.sink[: 64 // tp_size] * 2

        def process_b12x_weights_after_loading(self):
            pass

    with (
        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        model = Model()
        assert ds4.DeepseekV4Model.load_weights(model, session.weights([path])) == {
            name
        }
        ds4.DeepseekV4ForCausalLM.process_weights_after_loading(
            SimpleNamespace(model=model)
        )
    width = 64 // tp_size
    assert model.derived is not None
    torch.testing.assert_close(
        model.derived.cpu(), expected[rank * width : (rank + 1) * width] * 2
    )
    assert torch.isneginf(model.sink[width:]).all()


@pytest.mark.parametrize("scale_first", [False, True])
def test_dsa_indexer_dequantization_owns_inputs_across_checkpoint_shards(
    tmp_path, scale_first
):
    """Full GLM's fused WK projection reads real FP8 values before dequantizing."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
    from vllm.model_executor.weight_transfer import weight_transfer

    prefix = "layers.0.self_attn.indexer"
    weight = (torch.arange(128 * 256).reshape(128, 256) % 64).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0]])
    paths = [tmp_path / "weight.safetensors", tmp_path / "scale.safetensors"]
    save_file({f"{prefix}.wk.weight": weight}, paths[0])
    save_file({f"{prefix}.wk.weight_scale_inv": scale}, paths[1])
    if scale_first:
        paths.reverse()
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.nn.Parameter(
            torch.zeros((160, 256), device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        param.weight_loader = lambda p, value, shard_id: default_weight_loader(
            p[:128], value
        )
        name = f"{prefix}.wk_weights_proj.weight"
        pending: dict[str, dict[str, torch.Tensor]] = {}
        loaded: set[str] = set()
        for source_name, source in session.weights(paths):
            assert _try_load_fp8_indexer_wk(
                source_name, source, pending, {name: param}, loaded, []
            )
        assert not pending
        assert loaded == {name}
    expected = (weight.float() * scale.repeat_interleave(128, dim=1)).bfloat16()
    torch.testing.assert_close(param[:128].cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(param[128:]) == 0


def test_dspark_markov_embedding_reads_checkpoint_into_weight_storage(
    tmp_path, monkeypatch
):
    """A plain nn.Embedding loads without allocation hooks."""
    from vllm import envs
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    from vllm.model_executor.weight_transfer import weight_transfer

    monkeypatch.setattr(envs, "VLLM_MXFP8_LM_HEAD", False)
    expected = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    path = tmp_path / "markov.safetensors"
    save_file({"markov_w1.weight": expected}, path)
    with (
        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
        source = dict(session.weights([path]))["markov_w1.weight"]
        default_weight_loader(head.markov_w1.weight, source)
        session.flush()
        result = head.embed(torch.tensor([0, 17, 127]))
    torch.testing.assert_close(result.cpu(), expected[[0, 17, 127]])


def test_glm_mtp_projection_loads_from_main_shard_without_sharing_runtime_buffers(
    tmp_path, monkeypatch
):
    """MTP's plain Linear loads without a target-model allocation scope."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.glm5next.common import mtp

    class Decoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.topk_indices_buffer = kwargs["topk_indices_buffer"]
            self.pool_topk_indices_buffer = kwargs["pool_topk_indices_buffer"]

    monkeypatch.setattr(mtp, "Glm5NextDecoderLayer", Decoder)
    monkeypatch.setattr(mtp, "SharedHead", lambda **kwargs: torch.nn.Identity())
    config = SimpleNamespace(
        hidden_size=8, rms_norm_eps=1e-6, index_topk=16, index_kpool=4
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_text_config=config)
        ),
        quant_config=None,
        attention_config=SimpleNamespace(backend=None),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )
    name = "model.language_model.layers.45.eh_proj.weight"
    expected = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {name: expected, "model.language_model.layers.0.weight": torch.ones(8)}, path
    )
    with (
        set_current_vllm_config(VllmConfig()),
        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        layer = mtp.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.45")
        source = dict(
            session.weights([path], prefixes=("model.language_model.layers.45.",))
        )
        assert set(source) == {name}
        default_weight_loader(layer.eh_proj.weight, source[name])
        session.flush()
        result = layer.eh_proj(torch.ones(1, 16))
    torch.testing.assert_close(layer.eh_proj.weight.cpu(), expected)
    torch.testing.assert_close(result.cpu(), expected.sum(dim=1).unsqueeze(0))
