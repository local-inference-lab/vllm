# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import filelock
import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    _load_online_encoding_with_retry,
)
from vllm.model_executor.layers.quantization.exl3_online_cache import (
    Exl3OnlineCacheKey,
    Exl3OnlineNonFiniteError,
    cache_mode,
    cache_path,
    cache_root,
    load_or_quantize,
    resolve_encoder_identity,
    resolve_model_identity,
)


def _key() -> Exl3OnlineCacheKey:
    return Exl3OnlineCacheKey(
        model_identity="model-identity",
        encoder_identity="encoder-identity",
        prefix="model.layers.0.self_attn.o_proj",
        bits=6,
        seed=123,
        tp_world_size=4,
        tp_rank=2,
        input_size=128,
        output_size=256,
    )


def _tensors(key: Exl3OnlineCacheKey) -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.arange(
            key.input_size // 16 * key.output_size // 16 * key.bits * 16,
            dtype=torch.int16,
        ).reshape(key.input_size // 16, key.output_size // 16, key.bits * 16),
        "suh": torch.ones(key.input_size, dtype=torch.float16),
        "svh": torch.full((key.output_size,), 2, dtype=torch.float16),
    }


def test_cache_round_trip_skips_second_quantization(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
    key = _key()
    calls = 0

    def quantize():
        nonlocal calls
        calls += 1
        return _tensors(key), 0.125

    first = load_or_quantize(key, device=torch.device("cpu"), quantize=quantize)
    second = load_or_quantize(key, device=torch.device("cpu"), quantize=quantize)

    assert calls == 1
    assert not first.hit
    assert second.hit
    assert second.proxy_error == pytest.approx(0.125)
    assert second.path == cache_path(key)
    for name, expected in _tensors(key).items():
        assert torch.equal(second.tensors[name], expected)


def test_corrupt_cache_is_replaced_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
    key = _key()
    path = cache_path(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a safetensors file")
    calls = 0

    def quantize():
        nonlocal calls
        calls += 1
        return _tensors(key), None

    result = load_or_quantize(key, device=torch.device("cpu"), quantize=quantize)
    reloaded = load_or_quantize(
        key,
        device=torch.device("cpu"),
        quantize=lambda: pytest.fail("valid replacement must be reused"),
    )

    assert calls == 1
    assert not result.hit
    assert reloaded.hit


def test_cache_publish_failure_keeps_completed_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
    key = _key()
    calls = 0

    def quantize():
        nonlocal calls
        calls += 1
        return _tensors(key), 0.25

    def fail_save(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.exl3_online_cache._save",
        fail_save,
    )

    result = load_or_quantize(key, device=torch.device("cpu"), quantize=quantize)

    assert calls == 1
    assert not result.hit
    assert result.path is None
    assert result.proxy_error == pytest.approx(0.25)


def test_cache_lock_timeout_falls_back_to_uncached_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
    key = _key()
    observed_timeout = None

    class HeldLock:
        def __init__(self, lock_file, *, timeout):
            nonlocal observed_timeout
            self.lock_file = lock_file
            observed_timeout = timeout

        def __enter__(self):
            raise filelock.Timeout(self.lock_file)

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(filelock, "FileLock", HeldLock)

    result = load_or_quantize(
        key,
        device=torch.device("cpu"),
        quantize=lambda: (_tensors(key), 0.375),
    )

    assert observed_timeout == 600.0
    assert not result.hit
    assert result.path is None
    assert result.proxy_error == pytest.approx(0.375)
    assert not cache_path(key).exists()


def test_cache_root_uses_vllm_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_ONLINE_CACHE_DIR", raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

    assert cache_root() == tmp_path / "exl3_online"


@pytest.mark.parametrize("bad_field", ["suh", "svh", "proxy_error"])
def test_nonfinite_encoding_is_rejected(tmp_path, monkeypatch, bad_field):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
    key = _key()

    def quantize():
        tensors = _tensors(key)
        proxy_error = 0.125
        if bad_field == "proxy_error":
            proxy_error = float("nan")
        else:
            tensors[bad_field][0] = float("nan")
        return tensors, proxy_error

    with pytest.raises(Exl3OnlineNonFiniteError, match="non-finite"):
        load_or_quantize(key, device=torch.device("cpu"), quantize=quantize)

    assert not cache_path(key).exists()


def test_online_encoding_retries_one_nonfinite_result(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "off")
    key = _key()
    calls = 0

    def quantize():
        nonlocal calls
        calls += 1
        tensors = _tensors(key)
        if calls == 1:
            tensors["svh"][0] = float("nan")
        return tensors, 0.125

    result = _load_online_encoding_with_retry(
        key, device=torch.device("cpu"), quantize=quantize
    )

    assert calls == 2
    assert torch.isfinite(result.tensors["svh"]).all()


def test_online_encoding_fails_after_second_nonfinite_result(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "off")
    key = _key()
    calls = 0

    def quantize():
        nonlocal calls
        calls += 1
        tensors = _tensors(key)
        tensors["suh"][0] = float("nan")
        return tensors, 0.125

    with pytest.raises(Exl3OnlineNonFiniteError, match="non-finite"):
        _load_online_encoding_with_retry(
            key, device=torch.device("cpu"), quantize=quantize
        )

    assert calls == 2


def test_readonly_miss_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "readonly")
    key = _key()

    result = load_or_quantize(
        key,
        device=torch.device("cpu"),
        quantize=lambda: (_tensors(key), 0.5),
    )

    assert not result.hit
    assert result.path is None
    assert not cache_path(key).exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", "off"),
        ("none", "off"),
        ("readonly", "readonly"),
        ("read-only", "readonly"),
        ("readwrite", "readwrite"),
        ("rw", "readwrite"),
    ],
)
def test_cache_mode_aliases(raw, expected, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", raw)
    assert cache_mode() == expected


def test_cache_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "sometimes")
    with pytest.raises(ValueError, match="must be off, readonly, or readwrite"):
        cache_mode()


def test_off_mode_neither_reads_nor_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_ONLINE_CACHE_MODE", "off")
    key = _key()
    cache_path(key).parent.mkdir(parents=True)
    cache_path(key).write_bytes(b"must not be read")

    result = load_or_quantize(
        key,
        device=torch.device("cpu"),
        quantize=lambda: (_tensors(key), 0.5),
    )

    assert not result.hit
    assert result.path is None
    assert cache_path(key).read_bytes() == b"must not be read"


def test_cache_key_covers_every_rank_local_encoding_input():
    key = _key()
    variants = [
        replace(key, model_identity="other-model"),
        replace(key, encoder_identity="other-encoder"),
        replace(key, prefix="model.layers.1.self_attn.o_proj"),
        replace(key, bits=5),
        replace(key, seed=456),
        replace(key, tp_world_size=8),
        replace(key, tp_rank=3),
        replace(key, input_size=256),
        replace(key, output_size=128),
    ]

    assert len({key.digest(), *(variant.digest() for variant in variants)}) == 10


def test_local_model_identity_tracks_metadata_and_shards(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    shard = model / "model-00001-of-00001.safetensors"
    config.write_text('{"model_type":"test"}', encoding="utf-8")
    shard.write_bytes(b"weight payload")

    before = resolve_model_identity(str(model))
    config.write_text('{"model_type":"changed"}', encoding="utf-8")
    after_config = resolve_model_identity(str(model))
    shard.write_bytes(b"changed weight payload")
    after_shard = resolve_model_identity(str(model))

    assert before != after_config
    assert after_config != after_shard


def test_hub_model_identity_tracks_resolved_revision():
    first = resolve_model_identity("org/model", revision="commit-a")
    second = resolve_model_identity("org/model", revision="commit-b")

    assert first != second

    resolved = resolve_model_identity(
        "org/model",
        revision="main",
        hf_config=type("Config", (), {"_commit_hash": "commit-a"})(),
    )
    assert resolved == first


def test_hub_model_identity_rejects_unresolved_revision():
    with pytest.raises(ValueError, match="requires a resolved revision"):
        resolve_model_identity("org/model")


def test_encoder_identity_tracks_source_without_explicit_revision(tmp_path):
    package = tmp_path / "exllamav3"
    package.mkdir()
    source = package / "quantize.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = resolve_encoder_identity(package)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = resolve_encoder_identity(package)

    assert before != after
    explicit = resolve_encoder_identity(package, revision="abc")
    assert explicit == resolve_encoder_identity(package, revision="abc")
