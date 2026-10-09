# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


def test_dspark_parallel_config_disables_eplb_atomically():
    target_config = _FakeParallelConfig()

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert not draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 0
    assert not draft_config.enable_elastic_ep
    assert draft_config.eplb_config is not target_config.eplb_config


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)


@pytest.mark.parametrize("draft_format", [None, "auto", "safetensors"])
def test_dspark_selects_explicit_draft_loader(monkeypatch, draft_format):
    """A CPU-backed draft loader must not inherit target GPU staging buffers."""
    from vllm.config import AttentionConfig, CacheConfig, LoadConfig
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import utils as model_utils
    from vllm.v1.worker.gpu.spec_decode import utils as spec_utils
    from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    @dataclass
    class Config:
        speculative_config: object
        parallel_config: object
        kernel_config: object
        attention_config: object
        cache_config: object
        load_config: object
        model_config: object

    target_load = LoadConfig(load_format="instanttensor")
    draft_load = LoadConfig(load_format=draft_format) if draft_format else None
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3", num_hidden_layers=1),
        get_vocab_size=lambda: 32,
    )
    config = Config(
        speculative_config=SimpleNamespace(
            draft_model_config=model_config,
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=4),
            attention_backend=None,
            moe_backend=None,
            kv_cache_dtype=None,
            draft_load_config=draft_load,
        ),
        parallel_config=_FakeParallelConfig(),
        kernel_config=None,
        attention_config=AttentionConfig(),
        cache_config=CacheConfig(),
        load_config=target_load,
        model_config=model_config,
    )
    target, draft = nn.Module(), nn.Module()
    target.model, draft.model = nn.Module(), nn.Module()
    selected_loaders = []

    def select_loader(load_config):
        selected_loaders.append(load_config)
        return SimpleNamespace(load_model=lambda **kwargs: draft)

    monkeypatch.setattr(model_loader, "get_model_loader", select_loader)
    monkeypatch.setattr(model_utils, "get_draft_quant_config", lambda config: None)
    monkeypatch.setattr(
        spec_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    assert load_dspark_model(target, config) is draft
    assert selected_loaders == [draft_load or target_load]
    assert config.load_config is target_load
    assert config.load_config.load_format == "instanttensor"
