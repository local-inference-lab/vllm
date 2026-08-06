# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

import vllm.config as config_module
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl

_WRITER_MODULE = "sparkinfer.attention._shared.mla.kv_cache"
_WRITER_NAME = "concat_and_cache_nvfp4_mla_fp8_rope"


class _StopAfterWriterBinding(Exception):
    pass


class _WriterInitializationFailure(RuntimeError):
    pass


def _initialize_writer_seam(
    impl: B12xMLASparseImpl,
    *,
    kv_cache_dtype: str = "nvfp4_ds_mla",
) -> None:
    B12xMLASparseImpl.__init__(
        impl,
        num_heads=8,
        head_size=576,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=None,
        attn_type=b12x_mla_sparse.AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=torch.empty((1, 1), dtype=torch.int32),
        kv_lora_rank=512,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=512,
    )


def _construct_through_writer_binding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
) -> B12xMLASparseImpl:
    monkeypatch.setattr(b12x_mla_sparse, "_KV_FP8_ROPE_REQUESTED", enabled)
    monkeypatch.setattr(b12x_mla_sparse, "_IS_GLM_MOE_DSA_CACHE", True)

    def stop_after_writer_binding():
        raise _StopAfterWriterBinding

    monkeypatch.setattr(
        config_module,
        "get_current_vllm_config",
        stop_after_writer_binding,
    )
    impl = object.__new__(B12xMLASparseImpl)
    with pytest.raises(_StopAfterWriterBinding):
        _initialize_writer_seam(impl)
    return impl


def _install_fake_writer_package(
    monkeypatch: pytest.MonkeyPatch,
    writer,
) -> None:
    sparkinfer_module = types.ModuleType("sparkinfer")
    sparkinfer_module.__path__ = []
    attention_module = types.ModuleType("sparkinfer.attention")
    attention_module.__path__ = []
    shared_module = types.ModuleType("sparkinfer.attention._shared")
    shared_module.__path__ = []
    mla_module = types.ModuleType("sparkinfer.attention._shared.mla")
    mla_module.__path__ = []
    kv_cache_module = types.ModuleType(_WRITER_MODULE)

    sparkinfer_module.attention = attention_module
    attention_module._shared = shared_module
    shared_module.mla = mla_module
    mla_module.kv_cache = kv_cache_module
    if writer is not None:
        setattr(kv_cache_module, _WRITER_NAME, writer)

    monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer_module)
    monkeypatch.setitem(sys.modules, "sparkinfer.attention", attention_module)
    monkeypatch.setitem(sys.modules, "sparkinfer.attention._shared", shared_module)
    monkeypatch.setitem(
        sys.modules,
        "sparkinfer.attention._shared.mla",
        mla_module,
    )
    monkeypatch.setitem(sys.modules, _WRITER_MODULE, kv_cache_module)


def _enabled_impl(writer) -> B12xMLASparseImpl:
    impl = object.__new__(B12xMLASparseImpl)
    impl._kv_fp8_rope = True
    impl._concat_and_cache_nvfp4_mla_fp8_rope = writer
    return impl


def test_disabled_mode_uses_stock_writer_without_loading_compact_writer(
    monkeypatch: pytest.MonkeyPatch,
):
    compact_imports = []
    compact_calls = []
    stock_calls = []
    real_import = __import__

    def guard_compact_writer_import(name, *args, **kwargs):
        if name == _WRITER_MODULE:
            compact_imports.append(name)
            raise AssertionError("disabled mode must not import the compact writer")
        return real_import(name, *args, **kwargs)

    def compact_writer(*args, **kwargs):
        compact_calls.append((args, kwargs))

    def stock_writer(
        kv_c_arg,
        k_pe_arg,
        kv_cache_arg,
        slot_mapping_arg,
        *,
        kv_cache_dtype,
        scale,
    ):
        stock_calls.append(
            (
                kv_c_arg,
                k_pe_arg,
                kv_cache_arg,
                slot_mapping_arg,
                kv_cache_dtype,
                scale,
            )
        )

    monkeypatch.setattr("builtins.__import__", guard_compact_writer_import)
    monkeypatch.setattr(
        B12xMLASparseImpl,
        "_concat_and_cache_nvfp4_mla_fp8_rope",
        compact_writer,
        raising=False,
    )
    impl = _construct_through_writer_binding(monkeypatch, enabled=False)
    monkeypatch.setattr(ops, "concat_and_cache_mla", stock_writer)

    kv_c = torch.zeros((2, 512), dtype=torch.bfloat16)
    k_pe = torch.arange(128).to(torch.bfloat16).reshape(2, 1, 64)
    kv_cache = torch.empty((1, 2, 432), dtype=torch.uint8)
    slot_mapping = torch.tensor([[11], [7]], dtype=torch.int64)
    k_scale = torch.tensor(0.25)

    impl.do_kv_cache_update(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        "nvfp4_ds_mla",
        k_scale,
    )

    assert compact_imports == []
    assert compact_calls == []
    assert len(stock_calls) == 1
    (
        actual_kv_c,
        actual_k_pe,
        actual_kv_cache,
        actual_slots,
        actual_dtype,
        actual_scale,
    ) = stock_calls[0]
    assert actual_kv_c is kv_c
    assert torch.equal(actual_k_pe, k_pe.squeeze(1))
    assert actual_kv_cache is kv_cache
    assert torch.equal(actual_slots, slot_mapping.flatten())
    assert actual_dtype == "nvfp4_ds_mla"
    assert actual_scale is k_scale


def test_enabled_mode_calls_bound_public_writer_with_flattened_slots_and_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    writer_calls = []

    def public_writer(kv_c_arg, k_pe_arg, kv_cache_arg, slot_mapping_arg, scale_arg):
        writer_calls.append(
            (kv_c_arg, k_pe_arg, kv_cache_arg, slot_mapping_arg, scale_arg)
        )

    def reject_stock_writer(*args, **kwargs):
        pytest.fail("enabled mode fell back to the stock MLA writer")

    _install_fake_writer_package(monkeypatch, public_writer)
    monkeypatch.setattr(ops, "concat_and_cache_mla", reject_stock_writer)
    impl = _construct_through_writer_binding(monkeypatch, enabled=True)

    kv_c = torch.zeros((2, 512), dtype=torch.bfloat16)
    k_pe = torch.arange(128).to(torch.bfloat16).reshape(2, 1, 64)
    kv_cache = torch.empty((1, 2, 368), dtype=torch.uint8)
    slot_mapping = torch.tensor([[9, 3]], dtype=torch.int64)
    k_scale = torch.tensor(0.125)

    impl.do_kv_cache_update(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        "nvfp4_ds_mla",
        k_scale,
    )

    assert len(writer_calls) == 1
    actual_kv_c, actual_k_pe, actual_kv_cache, actual_slots, actual_scale = (
        writer_calls[0]
    )
    assert actual_kv_c is kv_c
    assert actual_k_pe.shape == (2, 64)
    assert torch.equal(actual_k_pe, k_pe.squeeze(1))
    assert actual_kv_cache is kv_cache
    assert actual_slots.shape == (2,)
    assert torch.equal(actual_slots, slot_mapping.flatten())
    assert actual_scale is k_scale


def test_enabled_mode_rejects_wrong_cache_dtype_before_calling_writer():
    writer_calls = []

    def writer(*args):
        writer_calls.append(args)

    impl = _enabled_impl(writer)

    with pytest.raises(
        RuntimeError,
        match="KV_FP8_ROPE writer reached a non-NVFP4 cache: 'fp8_ds_mla'",
    ):
        impl.do_kv_cache_update(
            torch.empty((1, 512)),
            torch.empty((1, 1, 64)),
            torch.empty((1, 1, 368), dtype=torch.uint8),
            torch.tensor([[4]], dtype=torch.int64),
            "fp8_ds_mla",
            torch.tensor(1.0),
        )

    assert writer_calls == []


def test_enabled_mode_empty_cache_is_noop_before_dtype_validation():
    writer_calls = []

    def writer(*args):
        writer_calls.append(args)

    impl = _enabled_impl(writer)
    impl.do_kv_cache_update(
        torch.empty((0, 512)),
        torch.empty((0, 1, 64)),
        torch.empty((0, 368), dtype=torch.uint8),
        torch.empty((0, 1), dtype=torch.int64),
        "not-a-supported-cache-dtype",
        torch.tensor(1.0),
    )

    assert writer_calls == []


def test_enabled_mode_missing_public_writer_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_writer_package(monkeypatch, writer=None)
    monkeypatch.setattr(b12x_mla_sparse, "_KV_FP8_ROPE_REQUESTED", True)
    monkeypatch.setattr(b12x_mla_sparse, "_IS_GLM_MOE_DSA_CACHE", True)

    def reject_fallback_initialization():
        raise _StopAfterWriterBinding

    monkeypatch.setattr(
        config_module,
        "get_current_vllm_config",
        reject_fallback_initialization,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "KV_FP8_ROPE=1 requires a SparkInfer build with "
            "concat_and_cache_nvfp4_mla_fp8_rope package API support"
        ),
    ):
        _initialize_writer_seam(object.__new__(B12xMLASparseImpl))


def test_enabled_mode_writer_initialization_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = __import__

    def fail_writer_initialization(name, *args, **kwargs):
        if name == _WRITER_MODULE:
            raise _WriterInitializationFailure("compact writer initialization failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_writer_initialization)
    monkeypatch.setattr(b12x_mla_sparse, "_KV_FP8_ROPE_REQUESTED", True)
    monkeypatch.setattr(b12x_mla_sparse, "_IS_GLM_MOE_DSA_CACHE", True)

    def reject_fallback_initialization():
        raise _StopAfterWriterBinding

    monkeypatch.setattr(
        config_module,
        "get_current_vllm_config",
        reject_fallback_initialization,
    )

    with pytest.raises(
        _WriterInitializationFailure,
        match="compact writer initialization failed",
    ):
        _initialize_writer_seam(object.__new__(B12xMLASparseImpl))
