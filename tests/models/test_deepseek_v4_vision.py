# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.common.ops import sequence_parallel
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    DeepseekV4VLProcessingInfo,
)
from vllm.models.deepseek_v4.nvidia import mtp, vl_model


@pytest.mark.parametrize("vocabulary", [{}, {IMAGE_PLACEHOLDER: 42}])
def test_image_placeholder_requires_exact_vocabulary_entry(vocabulary):
    tokenizer = SimpleNamespace(
        get_vocab=lambda: vocabulary,
        convert_tokens_to_ids=lambda token: vocabulary.get(token, 17),
    )
    info = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    if not vocabulary:
        with pytest.raises(ValueError, match="Token not found in tokenizer"):
            DeepseekV4VLProcessingInfo.get_image_placeholder_token_id(info)
    else:
        assert DeepseekV4VLProcessingInfo.get_image_placeholder_token_id(info) == 42


def test_interleaved_vision_weights_are_streamed_and_finalized_once(monkeypatch):
    events = []
    language_model = SimpleNamespace(process_weights_after_loading=Mock())
    model = SimpleNamespace(
        language_model=language_model,
        hf_to_vllm_mapper=SimpleNamespace(apply=iter),
    )
    finalize = vl_model.DeepseekV4ForConditionalGeneration.process_weights_after_loading
    model.process_weights_after_loading = lambda: finalize(model)

    class Loader:
        def __init__(self, module):
            self.module = module

        def load_weights(self, weights):
            loaded = set()
            for name, _ in weights:
                events.append(("load", name))
                loaded.add(name)
            return loaded

    monkeypatch.setattr(vl_model, "AutoWeightsLoader", Loader)
    names = (
        "vision.patch_embed.weight",
        "language_model.model.embed.weight",
        "aligner.w1.weight",
        "language_model.model.layers.0.weight",
    )

    def weights():
        for name in names:
            events.append(("yield", name))
            yield name, torch.empty(1)

    loaded = vl_model.DeepseekV4ForConditionalGeneration.load_weights(model, weights())
    assert loaded == set(names)
    assert events == [
        event
        for name in names
        for event in (("yield", name), ("load", name.removeprefix("language_model.")))
    ]
    model.process_weights_after_loading()
    language_model.process_weights_after_loading.assert_called_once_with()


@pytest.mark.parametrize("num_tokens", [3, 4])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("use_sp", [False, True])
def test_mtp_routing_ids_match_local_hidden_rows(monkeypatch, num_tokens, rank, use_sp):
    monkeypatch.setattr(
        sequence_parallel, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        sequence_parallel, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(mtp.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(mtp, "fused_mtp_input_rmsnorm", lambda e, p, h, *args: (e, h))
    monkeypatch.setattr(mtp, "sp_all_gather", lambda x: torch.cat([x, x]))
    ids = torch.arange(1, num_tokens + 1)
    expected = sequence_parallel.sp_shard(ids) if use_sp else ids

    class Block:
        use_sequence_parallel = use_sp

        def __call__(self, *, positions, x, input_ids):
            torch.testing.assert_close(input_ids, expected)
            assert input_ids.shape[0] == x.shape[0]
            torch.testing.assert_close(x[:, 0, 0], input_ids.float())
            return x, None, None, None

        @staticmethod
        def mhc_post(x, *args):
            return x

    norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    layer = SimpleNamespace(
        config=SimpleNamespace(hidden_size=2),
        hc_mult=1,
        enorm=norm,
        hnorm=norm,
        h_proj=lambda x: x,
        e_proj=lambda x: x,
        mtp_block=Block(),
    )
    result = mtp.DeepSeekV4MultiTokenPredictorLayer.forward(
        layer,
        input_ids=ids,
        positions=torch.arange(num_tokens),
        previous_hidden_states=ids.float().view(-1, 1).expand(-1, 2),
        inputs_embeds=torch.zeros(num_tokens, 2),
    )
    assert result.shape == (num_tokens, 2)
