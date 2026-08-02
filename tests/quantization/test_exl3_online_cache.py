# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3_online_cache import (
    Exl3OnlineCacheKey,
    cache_path,
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
    assert not cache_path(key).exists()


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
    after = resolve_model_identity(str(model))

    assert before != after


def test_hub_model_identity_tracks_resolved_revision():
    first = resolve_model_identity("org/model", revision="commit-a")
    second = resolve_model_identity("org/model", revision="commit-b")

    assert first != second


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
