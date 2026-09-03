# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only construction tests for Qwen3.8-Flash-Next."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
import vllm.distributed.utils as distributed_utils
import vllm.model_executor.offloader as offloader
import vllm.models.qwen3_8_flash_next.hyperconnection as hyperconnection_module
import vllm.models.qwen3_8_flash_next.model as model_module
import vllm.models.qwen3_8_flash_next.ple_layer as ple_layer_module
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.mamba.gdn import (
    qwen_gdn_linear_attn as qwen_gdn_module,
)
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


class _RecordingPlan:
    def __init__(self) -> None:
        self.bind_kwargs: dict[str, Any] | None = None
        self.binding = object()

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self.binding

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


def _set_tensor_attributes(module: nn.Module, *names: str) -> None:
    for name in names:
        setattr(module, name, torch.empty(0))


def test_ple_cpu_offload_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    assert ple_layer_module._resolve_ple_table_memory(None) == "mapped_host"


def test_qwen3_8_prefers_b12x_gdn_unless_explicitly_overridden(monkeypatch) -> None:
    config = SimpleNamespace(additional_config={})
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=True) == ("b12x", False)
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=False) == ("cuda", False)

    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=True) == ("triton", True)

    config.additional_config["gdn_decode_kernel"] = "b12x"
    assert _resolve_gdn_decode_kernel(config, prefer_b12x=False) == ("b12x", True)


def test_explicit_ple_table_memory_overrides_env_alias(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
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


def test_gated_residual_uses_canonical_combine_ops(monkeypatch) -> None:
    combined = torch.full((2, 8), 3.0, dtype=torch.bfloat16)
    normalized = torch.full((2, 8), 5.0, dtype=torch.bfloat16)
    plan = object()
    binding = object()
    workspace = SimpleNamespace(plan=plan, bind=Mock(return_value=binding))
    api = SimpleNamespace(
        run_combine_norm=Mock(return_value=(combined, normalized)),
        run_combine=Mock(return_value=combined),
    )

    mixer = hyperconnection_module.GatedResidual.__new__(
        hyperconnection_module.GatedResidual
    )
    nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(rms_norm_eps=1e-6)
    mixer.hc_norm = SimpleNamespace(weight=torch.empty(8, dtype=torch.bfloat16))
    object.__setattr__(mixer, "_workspace", workspace)
    monkeypatch.setattr(hyperconnection_module, "_hyperconnection_api", lambda: api)
    monkeypatch.setattr(
        hyperconnection_module.GatedResidual,
        "_mix_normalized",
        lambda _self, value, _binding: (value[:, :2], None),
    )

    hidden_states = torch.empty((2, 8), dtype=torch.bfloat16)
    block_output = torch.empty((2, 2), dtype=torch.bfloat16)
    injection = torch.empty((2, 4), dtype=torch.bfloat16)
    actual_combined, block_input, actual_injection = mixer.combine_and_mix(
        hidden_states,
        block_output,
        injection,
    )
    final_combined = mixer.combine(hidden_states, block_output, injection)

    assert actual_combined is combined
    assert block_input.data_ptr() == normalized.data_ptr()
    assert actual_injection is None
    assert final_combined is combined
    api.run_combine_norm.assert_called_once_with(
        hidden_states,
        block_output,
        injection,
        mixer.hc_norm.weight,
        eps=1e-6,
        plan=plan,
    )
    workspace.bind.assert_called_once_with(2)
    api.run_combine.assert_called_once_with(
        hidden_states,
        block_output,
        injection,
        plan=plan,
    )


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


def test_ple_bind_preserves_exact_aligned_page_stride(monkeypatch) -> None:
    shapes = ((10_240, 11),)
    dtypes = (torch.bfloat16,)
    raw_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.1.ple",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    plan = _RecordingPlan()
    planned_slots: list[int] = []

    monkeypatch.setattr(
        Qwen3_8FlashNextPLELayer, "get_state_shape", lambda _self: shapes
    )
    monkeypatch.setattr(
        Qwen3_8FlashNextPLELayer, "get_state_dtype", lambda _self: dtypes
    )

    def make_plan(_self, max_state_slots: int):
        planned_slots.append(max_state_slots)
        return plan

    monkeypatch.setattr(Qwen3_8FlashNextPLELayer, "_make_plan", make_plan)
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer.conv_state_len = 9
    layer.num_spec_tokens = 2
    layer.hc_hidden_size = 10_240
    _set_tensor_attributes(
        layer,
        "_scratch",
        "_residual",
        "_key",
        "_value",
        "_query_start_loc",
        "_state_slot_ids",
        "_state_is_fresh",
        "_num_accepted_tokens",
        "_num_seqs",
        "_num_tokens",
        "_out",
        "_request_is_prefill",
    )
    layer.norm_key = SimpleNamespace(weight=torch.empty(0))
    layer.norm_query = SimpleNamespace(weight=torch.empty(0))
    layer.norm_conv = SimpleNamespace(weight=torch.empty(0))
    layer.conv1d = SimpleNamespace(weight=torch.empty(1, 1, 1))

    layer.bind_kv_cache(raw_cache)

    (conv_state,) = layer.kv_cache
    assert raw_cache.shape == (2, 1, 1, 225_280)
    assert raw_cache.stride(0) == _ALIGNED_PAGE_SIZE_BYTES
    assert conv_state.stride() == (409_088, 11, 1)
    assert not conv_state.is_contiguous()
    assert conv_state[0].is_contiguous()
    assert conv_state.data_ptr() == raw_cache.data_ptr()
    assert (
        conv_state.untyped_storage().data_ptr()
        == raw_cache.untyped_storage().data_ptr()
    )
    assert planned_slots == [2]
    assert plan.bind_kwargs is not None
    assert plan.bind_kwargs["conv_state"] is conv_state
    assert layer._binding is plan.binding

    layer.unbind_kv_cache()

    assert layer.kv_cache == ()
    assert layer._plan is None
    assert layer._binding is None

    replacement_cache = _allocate_aligned_mamba_cache(
        layer_name="model.layers.1.ple",
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    layer.bind_kv_cache(replacement_cache)

    assert planned_slots == [2, 2]
    assert layer.kv_cache[0] is not conv_state
    assert plan.bind_kwargs["conv_state"] is layer.kv_cache[0]
    assert layer._binding is plan.binding


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
    layer._b12x_binding = None
    layer._b12x_plan = None
    _set_tensor_attributes(
        layer,
        "_b12x_scratch",
        "_b12x_mixed_qkv",
        "_b12x_a",
        "_b12x_b",
        "_b12x_z",
        "_b12x_query_start_loc",
        "_b12x_num_accepted_tokens",
        "_b12x_state_indices",
        "_b12x_num_seqs",
        "_b12x_num_tokens",
        "_b12x_output",
    )
    layer.A_log = nn.Parameter(torch.empty(0))
    layer.dt_bias = nn.Parameter(torch.empty(0))
    layer.norm = SimpleNamespace(weight=torch.empty(0))

    layer.bind_kv_cache(raw_cache)

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
    assert layer._b12x_binding is plan.binding

    layer.unbind_kv_cache()

    assert layer.kv_cache == ()
    assert layer._b12x_plan is None
    assert layer._b12x_binding is None


def test_b12x_gdn_plan_trusts_scheduler_metadata(monkeypatch) -> None:
    captured_caps: dict[str, Any] = {}

    class FakeApi:
        @staticmethod
        def Caps(**kwargs):
            captured_caps.update(kwargs)
            return kwargs

        @staticmethod
        def plan(caps):
            return caps

    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._b12x_gdn_api = FakeApi()
    layer._b12x_max_tokens = 16
    layer._b12x_max_seqs = 4
    layer._b12x_state_index_columns = 4
    layer._b12x_local_key_heads = 4
    layer._b12x_local_value_heads = 4
    layer.head_k_dim = 128
    layer.head_v_dim = 128
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.norm = SimpleNamespace(activation="silu")

    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "get_state_dtype",
        lambda _: (torch.bfloat16, torch.float32),
    )
    monkeypatch.setattr(
        qwen_gdn_module,
        "current_platform",
        SimpleNamespace(current_device=lambda: "cuda:0"),
    )

    plan = layer._make_b12x_gdn_plan(max_state_slots=32)

    assert plan == captured_caps
    assert captured_caps["qwen_metadata_validation"] == "trusted"


def test_b12x_gdn_stages_speculative_rollback_metadata() -> None:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._b12x_max_tokens = 6
    layer._b12x_max_seqs = 2
    layer._b12x_state_index_columns = 3
    layer._b12x_binding = object()
    layer.layer_norm_epsilon = 1e-6
    layer.head_k_dim = 128
    layer._b12x_mixed_qkv = torch.empty(6, 8)
    layer._b12x_a = torch.empty(6, 2)
    layer._b12x_b = torch.empty(6, 2)
    layer._b12x_z = torch.empty(6, 2, 4)
    layer._b12x_output = torch.full((6, 2, 4), 17.0)
    layer._b12x_query_start_loc = torch.full((3,), -1, dtype=torch.int32)
    layer._b12x_num_accepted_tokens = torch.full((2,), -1, dtype=torch.int32)
    layer._b12x_state_indices = torch.full((2, 3), -1, dtype=torch.int32)
    layer._b12x_num_seqs = torch.zeros(1, dtype=torch.int32)
    layer._b12x_num_tokens = torch.zeros(1, dtype=torch.int32)
    calls: list[tuple[object, float, float]] = []

    def run(binding, *, eps: float, scale: float) -> None:
        calls.append((binding, eps, scale))

    layer._b12x_gdn_api = SimpleNamespace(run=run)
    mixed_qkv = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    a = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    b = a + 20
    output_gate = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4)
    state_indices = torch.tensor([[7, 8, 9], [4, 5, 6]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    accepted = torch.tensor([1, 2], dtype=torch.int32)
    core_attn_out = torch.zeros(5, 2, 4)

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

    torch.testing.assert_close(layer._b12x_mixed_qkv[:5], mixed_qkv)
    torch.testing.assert_close(layer._b12x_a[:5], a)
    torch.testing.assert_close(layer._b12x_b[:5], b)
    torch.testing.assert_close(layer._b12x_z[:5], output_gate)
    torch.testing.assert_close(layer._b12x_query_start_loc, query_start_loc)
    torch.testing.assert_close(layer._b12x_num_accepted_tokens, accepted)
    torch.testing.assert_close(layer._b12x_state_indices, state_indices)
    torch.testing.assert_close(
        layer._b12x_num_seqs, torch.tensor([2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        layer._b12x_num_tokens, torch.tensor([5], dtype=torch.int32)
    )
    torch.testing.assert_close(core_attn_out, torch.full_like(core_attn_out, 17.0))
    assert calls == [(layer._b12x_binding, 1e-6, 128**-0.5)]
