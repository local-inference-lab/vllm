# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.deepseek_v2 import _should_skip_index_topk


def test_index_topk_pattern_only_skips_backbone_layers() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        index_topk_pattern="FSSF",
    )

    assert not _should_skip_index_topk(config, 0)
    assert _should_skip_index_topk(config, 1)
    assert _should_skip_index_topk(config, 2)
    assert not _should_skip_index_topk(config, 3)
    assert not _should_skip_index_topk(config, 4)
    assert not _should_skip_index_topk(config, 5)


def test_index_topk_frequency_does_not_skip_nextn_layer() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        index_topk_pattern=None,
        index_topk_freq=3,
        index_skip_topk_offset=2,
    )

    assert _should_skip_index_topk(config, 2)
    assert not _should_skip_index_topk(config, 4)
