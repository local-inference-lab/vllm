# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import CacheConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.models.glm5next.nvidia.kda import (
    Glm5NextRecoverKDAMetadataBuilder,
)
from vllm.models.glm5next.nvidia.model import Glm5NextForCausalLM
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.fixture(autouse=True)
def disable_pinned_host_staging(monkeypatch):
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    monkeypatch.setattr("vllm.v1.attention.backends.utils.PIN_MEMORY", False)


def _recovery_config(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr(
        "vllm.platforms.current_platform.get_device_capability",
        lambda: SimpleNamespace(major=12),
    )
    monkeypatch.setattr(
        "vllm.utils.b12x.get_b12x_gdn_decode",
        lambda: SimpleNamespace(bind_kda_commit=Mock(), is_supported=lambda: True),
    )
    return SimpleNamespace(
        cache_config=CacheConfig(mamba_cache_mode="align"),
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
            supports_replayssm=True,
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(linear_head_dim=128),
        ),
        speculative_config=SimpleNamespace(method="mtp"),
        num_speculative_tokens=3,
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        mamba_config=SimpleNamespace(
            backend=MambaBackendEnum.TRITON, enable_stochastic_rounding=False
        ),
        kv_transfer_config=None,
        use_request_boundary_checkpoints=True,
    )


@pytest.mark.parametrize("method,depth", [("mtp", 3), ("dflash", 7)])
@pytest.mark.parametrize("external", [False, True])
def test_glm_recovery_defaults_to_native_with_atomic_cache(
    monkeypatch, method, depth, external
):
    config = _recovery_config(monkeypatch)
    config.speculative_config.method = method
    config.num_speculative_tokens = depth
    if external:
        config.kv_transfer_config = SimpleNamespace(is_kv_transfer_instance=True)
    VllmConfig.validate_mamba_cached_kernel(config)
    assert config.cache_config.use_replayssm is True
    assert config.cache_config.use_kda_recoverssm is True


@pytest.mark.parametrize("unsupported", ["depth", "connector", "state", "v1", "b12x"])
def test_glm_recovery_leaves_unsupported_configs_unchanged(monkeypatch, unsupported):
    config = _recovery_config(monkeypatch)
    if unsupported == "depth":
        config.num_speculative_tokens = 10
    elif unsupported == "connector":
        config.kv_transfer_config = SimpleNamespace(is_kv_transfer_instance=True)
        config.use_request_boundary_checkpoints = False
    elif unsupported == "state":
        config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    elif unsupported == "v1":
        config.use_v2_model_runner = False
    else:
        monkeypatch.setattr("vllm.utils.b12x.get_b12x_gdn_decode", lambda: None)
    VllmConfig.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm
    config.cache_config.use_replayssm = True
    with pytest.raises(ValueError, match="GLM KDA recovery requires"):
        VllmConfig.validate_mamba_cached_kernel(config)


def test_glm_recovery_respects_opt_out_and_other_model_defaults(monkeypatch):
    config = _recovery_config(monkeypatch)
    config.cache_config.use_replayssm = False
    VllmConfig.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm
    config.cache_config.use_replayssm = None
    config.model_config.architecture = "KimiLinearForCausalLM"
    VllmConfig.validate_mamba_cached_kernel(config)
    assert not config.cache_config.use_kda_recoverssm


def test_glm_recoverssm_reserves_one_recurrent_state_per_draft_window():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                linear_num_heads=64,
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            use_kda_recoverssm=False,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
    )
    baseline = MambaSpec(
        block_size=256,
        shapes=Glm5NextForCausalLM.get_mamba_state_shape_from_config(config),
        dtypes=Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    config.cache_config.use_kda_recoverssm = True
    recovered = MambaSpec(
        block_size=256,
        shapes=Glm5NextForCausalLM.get_mamba_state_shape_from_config(config),
        dtypes=Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )
    assert len(baseline.shapes) == 2
    assert len(recovered.shapes) == 4
    assert recovered.dtypes[1] == torch.float32
    assert recovered.dtypes[2] == torch.float32
    assert baseline.page_size_bytes == 2_342_912
    assert recovered.page_size_bytes == 2_605_056
    assert (
        baseline.max_memory_usage_bytes(config)
        - recovered.max_memory_usage_bytes(config)
        == 15_876_096
    )


def _builder() -> Glm5NextRecoverKDAMetadataBuilder:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm5_next")
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            use_kda_recoverssm=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=2,
            parallel_drafting=False,
            use_eagle_block_drop=Mock(return_value=False),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
            static_forward_context={},
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256),
        additional_config={},
        num_speculative_tokens=2,
        use_v2_model_runner=False,
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
        num_speculative_blocks=0,
    )
    builder = Glm5NextRecoverKDAMetadataBuilder(
        spec,
        ["layer.0"],
        config,
        torch.device("cpu"),
    )
    builder._recoverssm_context = Mock()
    return builder


def test_glm_recoverssm_mixed_prefill_and_decode_keeps_one_state_slot():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 65, 20], query_lens=[1, 1, 3]),
        16,
        torch.device("cpu"),
    ).replace(is_prefilling=torch.tensor([True, True, False]))
    metadata = _builder().build(
        0,
        common,
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1, 2]),
        num_accepted_tokens=torch.tensor([3, 2, 2]),
    )
    assert metadata.num_spec_decodes == 1
    assert metadata.num_prefills == 2
    assert metadata.spec_state_indices_tensor is not None
    assert metadata.spec_state_indices_tensor.shape == (1, 1)
    torch.testing.assert_close(
        metadata.num_accepted_tokens, torch.ones(1, dtype=torch.int32)
    )
    assert metadata.recoverssm_commit is not None
    torch.testing.assert_close(
        metadata.recoverssm_commit.request_indices, torch.tensor([2], dtype=torch.int32)
    )
    sampled = torch.tensor([1, 1, 3], dtype=torch.int32)
    assert metadata.commit_recoverssm_state(sampled) is None
    metadata.recoverssm_context.commit.assert_called_once()


def test_glm_recoverssm_keeps_draftless_decode_out_of_prefill():
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20], query_lens=[1]),
        16,
        torch.device("cpu"),
    ).replace(is_prefilling=torch.tensor([False]))
    metadata = _builder().build(
        0,
        common,
        num_decode_draft_tokens_cpu=torch.tensor([-1]),
        num_accepted_tokens=torch.tensor([1]),
    )
    assert metadata.num_spec_decodes == 1
    assert metadata.num_prefills == 0
    assert metadata.spec_state_indices_tensor is not None
    assert metadata.spec_state_indices_tensor.shape == (1, 1)
