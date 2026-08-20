# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def test_kimi_dspark_binds_target_auxiliary_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = torch.empty(16, 8)
    binder = Mock()
    model = SimpleNamespace(
        bind_target_auxiliary_stream=binder,
        draft_id_to_target_id=None,
    )
    speculator = SimpleNamespace(
        vllm_config=object(),
        hidden_states=scratch,
        use_draft_token_capacity=False,
        draft_logits=None,
        _draft_topk=None,
        _use_local_draft_argmax=False,
        _capture_sharded_markov=False,
        _markov_outside_cudagraph=False,
    )
    target_model = object()
    monkeypatch.delenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", raising=False)
    monkeypatch.setattr(
        speculator_module,
        "load_dspark_model",
        lambda _target_model, _vllm_config: model,
    )

    loaded = DSparkSpeculator.load_draft_model(
        speculator,
        target_model=target_model,
        target_attn_layer_names=set(),
    )

    assert loaded is model
    binder.assert_called_once_with(target_model, scratch)


def test_sharded_markov_model_selects_local_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        draft_id_to_target_id=None,
    )
    speculator = SimpleNamespace(
        vllm_config=object(),
        use_draft_token_capacity=False,
        draft_logits=None,
        _draft_topk=None,
        _use_local_draft_argmax=False,
        _capture_sharded_markov=False,
        _markov_outside_cudagraph=False,
    )
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setattr(
        speculator_module,
        "load_dspark_model",
        lambda _target_model, _vllm_config: model,
    )

    loaded = DSparkSpeculator.load_draft_model(
        speculator,
        target_model=object(),
        target_attn_layer_names=set(),
    )

    assert loaded is model
    assert speculator._use_local_draft_argmax


def test_sharded_markov_model_rejects_reduced_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        draft_id_to_target_id=None,
    )
    speculator = SimpleNamespace(
        vllm_config=object(),
        use_draft_token_capacity=False,
        draft_logits=None,
        _draft_topk=32,
        _use_local_draft_argmax=False,
        _capture_sharded_markov=False,
        _markov_outside_cudagraph=False,
    )
    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setattr(
        speculator_module,
        "load_dspark_model",
        lambda _target_model, _vllm_config: model,
    )

    with pytest.raises(ValueError, match="does not support dspark_draft_topk"):
        DSparkSpeculator.load_draft_model(
            speculator,
            target_model=object(),
            target_attn_layer_names=set(),
        )


def test_sequential_greedy_sampling_keeps_vocabulary_shards_local() -> None:
    base_logits = torch.tensor(
        [
            [1.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )
    local_bias = torch.zeros((1, 4), dtype=torch.bfloat16)
    model = SimpleNamespace(
        compute_local_draft_logits=Mock(return_value=base_logits),
        markov_embed=Mock(side_effect=lambda token_ids: token_ids[:, None].float()),
        compute_local_markov_bias=Mock(return_value=local_bias),
        sample_local_draft_logits=Mock(
            side_effect=lambda base, bias: (base + bias).argmax(dim=-1)
        ),
    )
    speculator = SimpleNamespace(
        sample_indices=torch.tensor([0, 1]),
        model=model,
        _use_local_draft_argmax=True,
        draft_logits=None,
        _draft_topk=None,
        sample_idx_mapping=torch.tensor([0, 1]),
        sample_pos=torch.tensor([1, 2]),
        draft_token_confidence_logits=torch.empty((1, 2)),
        min_survival_probability=0.0,
        use_draft_token_capacity=False,
        input_buffers=SimpleNamespace(input_ids=torch.tensor([3, 0])),
        device=torch.device("cpu"),
        vocab_size=16,
        draft_token_valid_lengths=torch.empty((1,), dtype=torch.int32),
        draft_tokens=torch.empty((1, 2), dtype=torch.int64),
        draft_token_capacity=torch.empty((1,), dtype=torch.int32),
        _sample_logits=Mock(side_effect=AssertionError("full logits were sampled")),
    )

    DSparkSpeculator._sample_sequential(
        speculator,
        num_reqs=1,
        head_hidden=torch.randn((2, 8)),
        num_speculative_steps=2,
        num_query_per_req=2,
    )

    assert speculator.draft_tokens.tolist() == [[1, 2]]
    model.compute_local_draft_logits.assert_called_once()
    assert model.compute_local_markov_bias.call_count == 2
    assert model.sample_local_draft_logits.call_count == 2
    speculator._sample_logits.assert_not_called()
