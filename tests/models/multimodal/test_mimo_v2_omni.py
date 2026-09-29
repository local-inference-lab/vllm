# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 checkpoint loading and vision window attention."""

import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = 64
WINDOW = 8
# One sequence shorter than the window, one longer.
SEQ_LENS = [5, 37]


@pytest.mark.skip_global_cleanup
def test_omni_configures_dflash_auxiliary_layers():
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.model_executor.models.mimo_v2 import MiMoV2FlashForCausalLM, MiMoV2Model
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniForCausalLM

    model = MiMoV2OmniForCausalLM.__new__(MiMoV2OmniForCausalLM)
    torch.nn.Module.__init__(model)
    model.language_model = MiMoV2FlashForCausalLM.__new__(MiMoV2FlashForCausalLM)
    torch.nn.Module.__init__(model.language_model)
    backbone = MiMoV2Model.__new__(MiMoV2Model)
    torch.nn.Module.__init__(backbone)
    model.language_model.model = backbone

    assert supports_eagle3(model)
    model.set_aux_hidden_state_layers((1, 16, 32, 48, 70))
    assert backbone.aux_hidden_state_layers == (1, 16, 32, 48, 70)


@pytest.fixture
def vision_attn_env():
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        backend="nccl",
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    torch.set_default_dtype(default_dtype)


def _reference(q, k, v, cu_seqlens, sinks, scale):
    """Dense windowed softmax with an extra zero-valued sink per head."""
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:end].float()
        ks = k[start:end].float().repeat_interleave(groups, dim=1)
        vs = v[start:end].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        pos = torch.arange(end - start, device=q.device)
        outside = (pos.view(-1, 1) - pos.view(1, -1)).abs() > WINDOW
        scores.masked_fill_(outside, -torch.inf)
        sink_logits = sinks.float().view(-1, 1, 1).expand(-1, end - start, 1)
        probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(-1)[..., :-1]
        out[start:end] = torch.einsum("hqk,khd->qhd", probabilities, vs)
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires flash-attn")
@pytest.mark.parametrize("num_kv_heads", [NUM_HEADS, NUM_HEADS // 2])
def test_window_attention_applies_sinks(vision_attn_env, num_kv_heads):
    from vllm.model_executor.models.mimo_v2_omni import MiMoVisionAttention

    torch.manual_seed(0)
    attn = MiMoVisionAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        qk_channels=HEAD_DIM,
        kv_channels=HEAD_DIM,
        use_sink=True,
        visual_token_window_size=WINDOW,
    ).cuda()
    attn.sinks.data.normal_()

    total = sum(SEQ_LENS)
    opts = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, **opts)
    k = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    v = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    out = attn._forward_window_attn(q, k, v, cu_seqlens, max(SEQ_LENS))
    ref = _reference(q, k, v, cu_seqlens, attn.sinks, attn.scale)

    # bf16 attention lands at ~2e-3 here; dropping the sinks lands at ~1e-1.
    error = ((out.float() - ref).norm() / ref.norm()).item()
    assert error < 1e-2, f"sink-corrected output is off by {error:.2e}"


@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_fp8_qkv_merges_training_shards(num_kv_heads, tp_rank, metadata_only):
    from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj
    from vllm.model_executor.weight_transfer import weight_transfer

    checkpoint_tp = 4
    q_rows, k_rows, v_rows = 16 * 192, num_kv_heads // 4 * 192, num_kv_heads // 4 * 128
    rows = q_rows + k_rows + v_rows
    scale_rows = (rows + 127) // 128
    shards = []
    for group in range(checkpoint_tp):
        shards.append(
            torch.cat(
                [
                    torch.full((count, 128), 2.0 ** (group + kind))
                    for kind, count in enumerate((q_rows, k_rows, v_rows))
                ]
            )
        )
    weight = torch.cat(shards).to(torch.float8_e4m3fn)
    scale = (2.0 ** (torch.arange(checkpoint_tp * scale_rows) % 4)).view(-1, 1)
    shards = [
        shard * group_scale.repeat_interleave(128, 0)[:rows]
        for shard, group_scale in zip(shards, scale.chunk(checkpoint_tp))
    ]

    class Reader:
        def __init__(self):
            self.sources = {}

        def source(self, tensor):
            if not metadata_only:
                return tensor
            source = torch.empty_like(tensor, device="meta")
            self.sources[source.untyped_storage()._cdata] = tensor
            return source

        def materialize(self, source):
            if not source.is_meta:
                return source.clone()
            tensor = self.sources[source.untyped_storage()._cdata]
            return tensor.as_strided(
                source.shape, source.stride(), source.storage_offset()
            ).clone()

    reader = Reader()
    with weight_transfer(reader):
        actual_weight, actual_scale = _shard_fp8_qkv_proj(
            reader.source(weight),
            reader.source(scale),
            num_heads=64,
            num_kv_heads=num_kv_heads,
            head_dim=192,
            v_head_dim=128,
            tp_rank=tp_rank,
            tp_size=2,
            checkpoint_tp_size=checkpoint_tp,
        )
    actual = actual_weight.float() * actual_scale.repeat_interleave(128, 0)
    rank_shards = [
        shard.split((q_rows, k_rows, v_rows))
        for shard in shards[tp_rank * 2 : (tp_rank + 1) * 2]
    ]
    expected = torch.cat([part for parts in zip(*rank_shards) for part in parts])
    torch.testing.assert_close(actual, expected)


def _mimo_processor():
    from vllm.transformers_utils.processors.mimo_v2_omni import MiMoVLProcessor

    return MiMoVLProcessor(
        tokenizer=None,
        patch_size=16,
        image_min_pixels=8192,
        image_max_pixels=8388608,
        video_min_pixels=8192,
        video_max_pixels=8388608,
        video_total_max_pixels=268435456,
        fps=1.0,
    )


def test_audio_features_without_torchaudio(monkeypatch):
    """Audio preprocessing works without torchaudio and matches it."""
    import vllm.transformers_utils.processors.mimo_v2_omni as processor_module
    from vllm.multimodal.audio import MelSpectrogram

    wave = torch.randn(32000, generator=torch.Generator().manual_seed(7))
    audio = (wave, 16000)  # resampled to 24 kHz inside the processor

    reference = None
    if processor_module._HAS_TORCHAUDIO:
        reference = _mimo_processor().preprocess_audio(audio)

    monkeypatch.setattr(processor_module, "_MelSpectrogram", MelSpectrogram)
    monkeypatch.setattr(processor_module, "_HAS_TORCHAUDIO", False)
    spec, token_len = _mimo_processor().preprocess_audio(audio)

    # 48000 samples at 24 kHz, hop 240: 201 frames of 128 log-mel bins.
    assert spec.shape == (201, 128)
    assert token_len == 13
    if reference is not None:
        assert torch.equal(spec, reference[0])
        assert token_len == reference[1]
