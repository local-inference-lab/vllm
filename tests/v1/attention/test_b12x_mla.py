# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import b12x_mla
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_b12x_mla_is_registered_with_k3_envelope() -> None:
    assert AttentionBackendEnum.B12X_MLA.get_class() is B12xMLABackend
    assert B12xMLABackend.get_name() == "B12X_MLA"
    assert B12xMLABackend.get_supported_head_sizes() == [576]
    assert B12xMLABackend.supports_block_size(944)
    assert not B12xMLABackend.supports_block_size(936)
    assert B12xMLABackend.supports_compute_capability(DeviceCapability(12, 0))
    assert not B12xMLABackend.supports_compute_capability(DeviceCapability(10, 0))


class _FakePlan:
    def shapes_and_dtypes(self):
        return (((256,), torch.uint8),)


class _FakeDenseMLA:
    def __init__(self) -> None:
        self.bindings: list[SimpleNamespace] = []
        self.compile_count = 0

    def bind(self, plan, **kwargs):
        binding = SimpleNamespace(plan=plan, **kwargs)
        self.bindings.append(binding)
        return binding

    def compile(self, *, binding) -> None:
        self.compile_count += 1

    def run(self, *, binding):
        lse = torch.zeros(
            binding.output.shape[:2], dtype=torch.float32, device=binding.output.device
        )
        return binding.output, lse


def _fake_impl(monkeypatch) -> tuple[B12xMLAImpl, _FakeDenseMLA]:
    monkeypatch.setattr(b12x_mla, "is_workspace_manager_initialized", lambda: False)
    impl = object.__new__(B12xMLAImpl)
    impl.num_heads = 8
    impl.kv_lora_rank = 512
    impl.scale = 192**-0.5
    impl._compiled_bindings = set()
    impl._fallback_scratch = {}
    dense_mla = _FakeDenseMLA()
    impl._dense_mla = dense_mla
    return impl, dense_mla


def test_b12x_mla_adapter_binds_common_decode_metadata(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    q_nope = torch.randn(batch, 8, 512, dtype=torch.bfloat16)
    q_rope = torch.randn(batch, 8, 64, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)
    output_2, _ = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)

    assert output.shape == (batch, 8, 512)
    assert output.dtype == torch.bfloat16
    assert lse is not None and lse.dtype == torch.float32
    assert output_2.shape == output.shape
    assert dense_mla.compile_count == 1
    binding = dense_mla.bindings[0]
    assert binding.q.shape == (batch, 8, 576)
    assert binding.q.is_contiguous()
    assert binding.kv_cache is cache
    assert binding.page_table is metadata.decode.block_table
    assert binding.cache_seqlens is metadata.decode.seq_lens
    assert binding.q_scale is None
    assert binding.kv_scale is None
    assert binding.sm_scale == impl.scale


def test_b12x_mla_adapter_passes_fp8_scales(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    q = torch.empty(1, 8, 576, dtype=torch.float8_e4m3fn)
    cache = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.q_scale is layer._q_scale
    assert binding.kv_scale is layer._k_scale
