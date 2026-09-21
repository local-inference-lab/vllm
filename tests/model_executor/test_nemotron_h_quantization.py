# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import torch


def test_puzzle_config_preserves_layer_geometry_without_mtp():
    from vllm.transformers_utils.configs.nemotron_h_puzzle import NemotronHPuzzleConfig

    blocks = [
        {"block_type": "mamba"},
        {
            "block_type": "moe",
            "n_routed_experts": 512,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1280,
            "moe_latent_size": 1024,
            "moe_shared_expert_intermediate_size": 5376,
        },
        {
            "block_type": "moe",
            "n_routed_experts": 512,
            "num_experts_per_tok": 18,
            "moe_intermediate_size": 2816,
            "moe_latent_size": 1024,
            "moe_shared_expert_intermediate_size": 5376,
        },
    ]
    config = NemotronHPuzzleConfig(
        block_configs=blocks,
        mtp_block_configs=[],
        num_nextn_predict_layers=0,
        chunk_size=128,
        ssm_state_size=96,
        n_groups=8,
    )
    restored = NemotronHPuzzleConfig.from_dict(config.to_dict())
    assert restored.hybrid_override_pattern == "MEE"
    assert restored.num_nextn_predict_layers == 0
    assert restored.chunk_size == 128
    assert restored.ssm_state_size == 96
    assert restored.get_nemotron_h_config_for_layer(1).moe_intermediate_size == 1280
    assert restored.get_nemotron_h_config_for_layer(2).moe_intermediate_size == 2816
    assert restored.get_nemotron_h_config_for_layer(2).num_experts_per_tok == 18
    assert not hasattr(restored, "n_routed_experts")


def test_mamba_convolution_view_tracks_checkpoint_parameter_storage():
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    mixer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.conv1d = torch.nn.Module()
    mixer.conv1d.weight = torch.nn.Parameter(torch.randn(8, 1, 4))
    assert mixer.conv_weights.data_ptr() == mixer.conv1d.weight.data_ptr()
    mixer.conv1d.weight = torch.nn.Parameter(torch.randn(8, 1, 4))
    assert mixer.conv_weights.data_ptr() == mixer.conv1d.weight.data_ptr()
    assert tuple(mixer.conv_weights.shape) == (8, 4)
    assert "conv_weights" not in dict(mixer.named_buffers())


def test_nemotron_h_lm_head_receives_quant_config():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_relu2_fp8_fusion_uses_registry():
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused
