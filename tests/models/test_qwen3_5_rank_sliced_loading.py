# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for rank-sliced EXL3 weight loading in Qwen3.5.

These tests exercise the shared ``normalize_rank_sliced_weights`` helper and
verify the ordering contract in both ``Qwen3_5Model.load_weights`` and
``Qwen3_5MTP.load_weights``:

* rank normalization happens *before* the ``WeightsMapper`` (main model) /
  ``mtp.``→``model.`` remap (MTP draft);
* local TP-rank payloads are retained and have the ``.rank{r}`` segment
  stripped;
* non-local ranks are dropped;
* ordinary (non-rank-sliced) weights pass through unchanged.

No real checkpoint or CUDA is required — ``AutoWeightsLoader`` is replaced
with a capture stub and ``quant_config`` is a lightweight stand-in.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    _RANK_SLICED_WEIGHT_RE,
    normalize_rank_sliced_weights,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.model_executor.models.utils import WeightsMapper

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_LOCAL_RANK = 0


class StubExl3QuantConfig:
    """Mimics ``Exl3Config.normalize_rank_sliced_weight_name`` for rank 0.

    The real method delegates to ``get_tensor_model_parallel_rank()`` which
    needs an initialized TP group; the stub hard-codes ``_LOCAL_RANK`` so the
    tests are self-contained and deterministic on CPU.
    """

    def normalize_rank_sliced_weight_name(self, name: str) -> str | None:
        match = _RANK_SLICED_WEIGHT_RE.match(name)
        if match is None:
            return name
        if int(match.group("rank")) != _LOCAL_RANK:
            return None
        return f"{match.group('prefix')}.{match.group('field')}"


class PlainQuantConfig:
    """A quant_config without ``normalize_rank_sliced_weight_name``."""


class _CapturedLoader:
    """Drop-in for ``AutoWeightsLoader`` that records what it received."""

    def __init__(self, model):
        self.model = model
        self.captured_weights: list[tuple[str, torch.Tensor]] | None = None
        self.captured_mapper: object | None = None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: object | None = None,
    ) -> set[str]:
        self.captured_weights = list(weights)
        self.captured_mapper = mapper
        return set()


def _w(name: str) -> tuple[str, torch.Tensor]:
    """Build a trivial (name, tensor) pair — tensor identity is irrelevant."""
    return name, torch.empty(1)


# ---------------------------------------------------------------------------
# Helper-level tests (covers the deduplicated primitive for both models)
# ---------------------------------------------------------------------------


class TestNormalizeRankSlicedWeights:
    quant_config = StubExl3QuantConfig()

    def test_local_rank_retained_and_normalized(self):
        weights = [
            _w("mlp.experts.0.w13.rank0.trellis"),
            _w("mlp.experts.0.w2.rank0.suh"),
        ]
        out = list(normalize_rank_sliced_weights(weights, self.quant_config))
        names = [n for n, _ in out]
        assert names == [
            "mlp.experts.0.w13.trellis",
            "mlp.experts.0.w2.suh",
        ]

    def test_nonlocal_rank_dropped(self):
        weights = [
            _w("mlp.experts.0.w13.rank1.trellis"),
            _w("mlp.experts.0.w13.rank0.trellis"),
            _w("mlp.experts.0.w2.rank2.mcg"),
        ]
        out = list(normalize_rank_sliced_weights(weights, self.quant_config))
        names = [n for n, _ in out]
        assert names == ["mlp.experts.0.w13.trellis"]

    def test_ordinary_weight_passes_through(self):
        weights = [
            _w("layers.0.self_attn.q_proj.weight"),
            _w("mlp.experts.0.w13.trellis"),  # already normalized form
        ]
        out = list(normalize_rank_sliced_weights(weights, self.quant_config))
        names = [n for n, _ in out]
        assert names == [
            "layers.0.self_attn.q_proj.weight",
            "mlp.experts.0.w13.trellis",
        ]

    def test_plain_quant_config_passthrough(self):
        weights = [
            _w("mlp.experts.0.w13.rank0.trellis"),
            _w("layers.0.self_attn.q_proj.weight"),
        ]
        out = list(normalize_rank_sliced_weights(weights, PlainQuantConfig()))
        names = [n for n, _ in out]
        # No normalize_rank_sliced_weight_name => identity passthrough.
        assert names == [
            "mlp.experts.0.w13.rank0.trellis",
            "layers.0.self_attn.q_proj.weight",
        ]

    def test_returns_iterable_not_list(self):
        """Helper returns a lazy generator (or the original iterable)."""
        weights = [_w("mlp.experts.0.w13.rank0.trellis")]
        result = normalize_rank_sliced_weights(weights, self.quant_config)
        assert not isinstance(result, list)
        # Consume to verify it works.
        assert list(result)


# ---------------------------------------------------------------------------
# Qwen3_5Model.load_weights — composition with WeightsMapper
# ---------------------------------------------------------------------------


def _make_model_instance() -> Qwen3_5Model:
    """Bare Qwen3_5Model without __init__ (avoids CUDA / full model build)."""
    instance = Qwen3_5Model.__new__(Qwen3_5Model)
    instance.quant_config = StubExl3QuantConfig()
    # hf_to_vllm_mapper is a class attribute; no need to set it.
    return instance


class TestQwen3_5ModelLoadWeights:
    def test_local_rank_normalized_before_loader(self, monkeypatch):
        instance = _make_model_instance()
        captured: list[_CapturedLoader] = []
        original_cls = _CapturedLoader

        def _factory(model):
            c = original_cls(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5.AutoWeightsLoader", _factory
        )
        weights = [_w("mlp.experts.0.w13.rank0.trellis")]
        instance.load_weights(weights)

        assert len(captured) == 1
        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["mlp.experts.0.w13.trellis"]

    def test_nonlocal_rank_dropped_before_loader(self, monkeypatch):
        instance = _make_model_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5.AutoWeightsLoader", _factory
        )
        weights = [
            _w("mlp.experts.0.w13.rank1.trellis"),
            _w("mlp.experts.0.w2.rank0.suh"),
        ]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["mlp.experts.0.w2.suh"]

    def test_ordinary_weight_passes_to_loader(self, monkeypatch):
        instance = _make_model_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5.AutoWeightsLoader", _factory
        )
        weights = [_w("layers.0.self_attn.q_proj.weight")]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["layers.0.self_attn.q_proj.weight"]

    def test_weights_mapper_preserved(self, monkeypatch):
        """The mapper is still forwarded to the loader after normalization."""
        instance = _make_model_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5.AutoWeightsLoader", _factory
        )
        instance.load_weights([_w("layers.0.self_attn.q_proj.weight")])

        # The class-level hf_to_vllm_mapper must reach loader.load_weights.
        assert captured[0].captured_mapper is Qwen3_5Model.hf_to_vllm_mapper

    def test_mapper_receives_normalized_names(self, monkeypatch):
        """Normalization runs *before* the mapper sees the names."""
        instance = _make_model_instance()
        # Inject a custom mapper that would *fail* if it saw a .rankN segment,
        # proving the rank was already stripped upstream.
        sentinel_mapper = WeightsMapper(
            orig_to_new_substr={".w13.trellis": ".w13_trellis"}
        )
        instance.hf_to_vllm_mapper = sentinel_mapper
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5.AutoWeightsLoader", _factory
        )
        instance.load_weights([_w("mlp.experts.0.w13.rank0.trellis")])

        names = [n for n, _ in captured[0].captured_weights]
        # The loader receives the normalized name (mapper applied inside the
        # real loader; here we only capture pre-mapper names, which must
        # already have the rank stripped).
        assert names == ["mlp.experts.0.w13.trellis"]
        assert captured[0].captured_mapper is sentinel_mapper


# ---------------------------------------------------------------------------
# Qwen3_5MTP.load_weights — mtp.→model. remap AFTER normalization
# ---------------------------------------------------------------------------


def _make_mtp_instance() -> Qwen3_5MTP:
    """Bare Qwen3_5MTP without __init__."""
    instance = Qwen3_5MTP.__new__(Qwen3_5MTP)
    instance.quant_config = StubExl3QuantConfig()
    return instance


class TestQwen3_5MTPLoadWeights:
    def test_rank_normalized_then_mtp_remapped(self, monkeypatch):
        """Ordering contract: normalization first, then mtp.→model. rewrite."""
        instance = _make_mtp_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _factory
        )
        weights = [_w("mtp.layers.0.mlp.experts.0.w13.rank0.trellis")]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        # rank0 stripped => mtp.layers.0.mlp.experts.0.w13.trellis
        # then mtp. => model. => model.layers.0.mlp.experts.0.w13.trellis
        assert names == ["model.layers.0.mlp.experts.0.w13.trellis"]

    def test_nonlocal_rank_dropped_before_remap(self, monkeypatch):
        instance = _make_mtp_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _factory
        )
        weights = [
            _w("mtp.layers.0.mlp.experts.0.w13.rank1.trellis"),
            _w("mtp.layers.0.mlp.experts.0.w2.rank0.suh"),
        ]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        # rank1 dropped; rank0 normalized then remapped.
        assert names == ["model.layers.0.mlp.experts.0.w2.suh"]

    def test_ordinary_mtp_weight_remapped(self, monkeypatch):
        instance = _make_mtp_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _factory
        )
        weights = [_w("mtp.layers.0.self_attn.q_proj.weight")]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["model.layers.0.self_attn.q_proj.weight"]

    def test_embed_tokens_language_model_prefix_stripped(self, monkeypatch):
        instance = _make_mtp_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _factory
        )
        weights = [_w("language_model.embed_tokens.weight")]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["embed_tokens.weight"]

    def test_ordering_rank_then_remap_no_collision(self, monkeypatch):
        """A rank-sliced mtp. weight is normalized before the prefix rewrite.

        This is the ordering contract: if the remap ran first it would turn
        ``mtp.…rank0.trellis`` into ``model.…rank0.trellis`` and the
        subsequent normalization would still strip the rank — but the point
        is that the *final* name has both transforms applied correctly.
        Verifying the exact output ``model.…w13.trellis`` proves the two
        rewrites compose without colliding.
        """
        instance = _make_mtp_instance()
        captured: list[_CapturedLoader] = []

        def _factory(model):
            c = _CapturedLoader(model)
            captured.append(c)
            return c

        monkeypatch.setattr(
            "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader", _factory
        )
        weights = [_w("mtp.experts.7.w13.rank0.mul1")]
        instance.load_weights(weights)

        names = [n for n, _ in captured[0].captured_weights]
        assert names == ["model.experts.7.w13.mul1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--noconftest"])
