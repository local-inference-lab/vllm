# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.deepseek_v2 import (
    _skip_disabled_mtp_weight,
    get_spec_layer_idx_from_weight_name,
)


def test_disabled_mtp_checkpoint_weights_are_skipped():
    config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=0)

    assert not _skip_disabled_mtp_weight(config, "model.layers.77.self_attn.weight")
    assert _skip_disabled_mtp_weight(config, "model.layers.78.self_attn.weight")


def test_spec_layer_index_uses_valid_layer_counts():
    config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=3)

    assert get_spec_layer_idx_from_weight_name(config, "model.layers.79.weight") == 79
    assert get_spec_layer_idx_from_weight_name(config, "layers.80.weight") == 80
    assert get_spec_layer_idx_from_weight_name(config, "model.layers.81.weight") is None


def test_spec_layer_index_does_not_scan_configured_layer_count():
    config = SimpleNamespace(
        num_hidden_layers=78,
        num_nextn_predict_layers=10**12,
    )

    assert get_spec_layer_idx_from_weight_name(config, "model.layers.80.weight") == 80
    assert (
        get_spec_layer_idx_from_weight_name(config, "model.embed_tokens.weight") is None
    )


@pytest.mark.parametrize("invalid", [None, True, "3", 3.0, -1, [3]])
def test_malformed_mtp_layer_counts_fail_closed(invalid):
    valid = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=3)
    bad_nextn = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=invalid)
    bad_hidden = SimpleNamespace(num_hidden_layers=invalid, num_nextn_predict_layers=3)

    for config in (bad_nextn, bad_hidden):
        assert not _skip_disabled_mtp_weight(config, "model.layers.78.weight")
        assert (
            get_spec_layer_idx_from_weight_name(config, "model.layers.78.weight")
            is None
        )

    assert get_spec_layer_idx_from_weight_name(valid, "model.layers.78.weight") == 78
