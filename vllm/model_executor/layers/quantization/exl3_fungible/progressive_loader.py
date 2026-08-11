# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--load-format progressive`` — boot mixed-K EXL3 from Progressive
Tensors segments + a policy, with no materialized assembled checkpoint.

The loader synthesizes the tensor stream an assembled checkpoint would
provide (see :mod:`.progressive`): dense/attention/shared tensors from the
dense-source checkpoint dir, routed experts resolved per-policy through the
:class:`~.fragments.FragmentResolver` (local segment dirs -> HF sources).
The mixed ``hybrid_tr3_tail`` / ``quantization_config`` metadata reaches
the model config through the ``--hf-overrides`` JSON that
``python -m ...exl3_fungible.progressive`` emits at serve time.

Configuration: ``VLLM_FQ_MANIFEST_DIR`` (+ ``VLLM_FQ_POLICY`` /
``VLLM_FQ_DENSE_SOURCE`` / ``VLLM_FQ_CACHE`` / ``VLLM_FQ_VERIFY`` /
``VLLM_FQ_SOURCES``), overridable per-serve via
``--model-loader-extra-config '{"manifest_dir": ..., "policy": ...,
"dense_source": ...}'``.
"""
from __future__ import annotations

import time

from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.exl3_fungible.progressive import (
    ProgressiveSpec,
    progressive_weights_iterator,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

_EXTRA_CONFIG_KEYS = {"manifest_dir", "policy", "dense_source"}


class ProgressiveModelLoader(BaseModelLoader):
    """Stream mixed-K EXL3 weights from Progressive Tensors segments."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                "load_format=progressive"
            )
        unexpected = set(extra) - _EXTRA_CONFIG_KEYS
        if unexpected:
            raise ValueError(
                f"Unexpected extra config keys for load_format=progressive: "
                f"{unexpected} (allowed: {sorted(_EXTRA_CONFIG_KEYS)})"
            )
        self._extra = extra

    def _spec(self, model_config: ModelConfig) -> ProgressiveSpec:
        return ProgressiveSpec.from_env(
            model_config.model, overrides=self._extra
        )

    def download_model(self, model_config: ModelConfig) -> None:
        # Validates that manifest dir / policy / dense source resolve; all
        # fragment IO is lazy and rank-scoped, so there is nothing to
        # prefetch here.
        self._spec(model_config)

    @staticmethod
    def _tp_rank() -> int | None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        except (ImportError, AssertionError, RuntimeError, ValueError):
            return None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        spec = self._spec(model_config)
        resolver = spec.make_resolver()
        tp_rank = self._tp_rank()
        logger.info(
            "Progressive boot: manifest_dir=%s policy=%s (digest %s) "
            "dense_source=%s k_values=%s tp_rank=%s",
            spec.manifest_dir,
            spec.policy_path or "<policy store>",
            spec.policy_digest[:12],
            spec.dense_source,
            spec.k_values,
            tp_rank,
        )
        # State the network posture BEFORE the nine-minute compile, not after
        # a failure. A run that silently fetched when the operator asked for
        # offline is not reproducible; a run that cannot fetch should say so
        # while there is still time to prime the cache.
        try:
            from vllm.model_executor.layers.quantization.exl3_fungible.\
                fragments import HfSource
            if HfSource.offline():
                logger.warning(
                    "Progressive boot: OFFLINE (HF_HUB_OFFLINE set) — Hub "
                    "sources disabled; only local segment dirs and the primed "
                    "cache will be used. Set FQ_ALLOW_NETWORK=1 to override.")
        except Exception:  # noqa: BLE001 — posture logging must not fail a boot
            pass
        start = time.perf_counter()
        model.load_weights(
            progressive_weights_iterator(
                spec,
                resolver,
                tp_rank=tp_rank,
                log=logger.info,
            )
        )
        logger.info_once(
            "Loading weights took %.2f seconds (progressive)",
            time.perf_counter() - start,
        )
