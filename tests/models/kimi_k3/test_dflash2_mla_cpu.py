# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the DFlash2 draft's config lifting, name mapping and conv.

The checkpoint tensor inventory below is `lightseekorg/kimi-k3-dflash2`
(`model.safetensors`, 87 tensors); the test asserts every tensor maps to a
parameter the colocated model owns, that the shared embedding is skipped, and
that the grouped convolution matches a direct per-block evaluation.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import regex as re
import torch
from torch import nn

from vllm.models.kimi_k3.nvidia import dflash2_mla, dspark_mla
from vllm.models.kimi_k3.nvidia.dflash2_mla import (
    DFlash2ForCausalLM,
    DFlash2Model,
    _grouped_conv,
    is_dflash2_draft,
    normalize_dflash2_config,
    rename_dflash2_checkpoint_name,
)

CHECKPOINT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_act": "silu",
    "hidden_size": 7168,
    "intermediate_size": 14336,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 163840,
    "rms_norm_eps": 1e-05,
    "num_target_layers": 93,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "is_causal": False,
    "use_sliding_window": True,
    "sliding_window": 4096,
    "dflash_config": {
        "block_size": 8,
        "attention_mode": "mla",
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [19, 37, 66, 78, 90],
        "mask_token_id": 163592,
    },
    "draft_vocab_size": 163840,
}

CHECKPOINT_TENSORS = [
    "candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "embed_tokens.weight",
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
] + [
    f"layers.{layer}.{name}"
    for layer in range(5)
    for name in (
        "attention_conv.base_kernel",
        "attention_conv.kernel_projection.weight",
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp_conv.base_kernel",
        "mlp_conv.kernel_projection.weight",
        "post_attention_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
    )
]


def _config() -> SimpleNamespace:
    return SimpleNamespace(**copy.deepcopy(CHECKPOINT_CONFIG))


def test_checkpoint_is_recognized_and_lifted() -> None:
    config = _config()
    assert is_dflash2_draft(config)
    normalize_dflash2_config(config)
    assert config.target_layer_ids == [19, 37, 66, 78, 90]
    assert config.num_target_layers == 5
    assert config.target_hidden_size == 7168
    assert config.draft_vocab_size == 163840
    assert config.sample_from_anchor is False
    # Idempotent.
    normalize_dflash2_config(config)
    assert config.num_target_layers == 5


def test_non_mla_dflash_is_not_dflash2() -> None:
    config = _config()
    config.architectures = ["DFlashDraftModel"]
    assert not is_dflash2_draft(config)
    config.architectures = ["DFlash2DraftModel"]
    config.dflash_config["attention_mode"] = "gqa"
    assert not is_dflash2_draft(config)


def test_every_checkpoint_tensor_maps_to_a_backbone_parameter() -> None:
    assert len(CHECKPOINT_TENSORS) == 87
    mapped = {name: rename_dflash2_checkpoint_name(name) for name in CHECKPOINT_TENSORS}
    assert mapped["embed_tokens.weight"] is None
    assert mapped["fc.weight"] == "context_proj.weight"
    assert mapped["hidden_norm.weight"] == "context_norm.weight"
    assert mapped["norm.weight"] == "final_norm.weight"
    assert (
        mapped["layers.3.input_layernorm.weight"] == "layers.3.input_layernorm.weight"
    )
    assert mapped["candidate_selector.successor_codebook"] == (
        "candidate_selector.successor_codebook"
    )
    kept = [name for name, target in mapped.items() if target is not None]
    assert len(kept) == 86
    # Names the DSpark loader fuses or stacks keep their checkpoint form.
    assert "layers.0.self_attn.q_a_proj.weight" in kept
    assert "layers.0.self_attn.kv_a_proj_with_mqa.weight" in kept
    assert "layers.0.mlp.gate_proj.weight" in kept
    assert rename_dflash2_checkpoint_name("model.norm.weight") == "final_norm.weight"
    assert (
        rename_dflash2_checkpoint_name("layers.1.self_attn.rotary_emb.inv_freq") is None
    )


class _ReferenceGroupedConv(torch.nn.Module):
    """Verbatim ``DFlashGroupedConv`` of TorchSpec (MIT, LightSeek Foundation;
    ``torchspec/models/draft/dflash2.py`` at revision fc28d35), the authors'
    batched ``[batch, length, hidden]`` formulation the flat-row port must
    reproduce."""

    def __init__(self, hidden_size, block_size, kernel_size, group_size):
        super().__init__()
        self.block_size = int(block_size)
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size
        base_kernel = torch.zeros(2, self.kernel_size, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = torch.nn.Parameter(base_kernel)
        self.kernel_projection = torch.nn.Linear(
            hidden_size, 2 * self.kernel_size * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, dynamic_kernel, side):
        batch, length, hidden_size = hidden_states.shape
        blocks = hidden_states.reshape(
            -1, self.block_size, self.num_groups, self.group_size
        )
        dynamic = dynamic_kernel.reshape(
            -1, self.block_size, self.kernel_size, self.num_groups
        )
        base = self.base_kernel[side].reshape(
            1, 1, self.kernel_size, self.num_groups, self.group_size
        )
        coefficients = base.to(hidden_states.dtype) + dynamic.unsqueeze(-1)
        output = torch.zeros_like(blocks)
        for offset in range(self.kernel_size):
            values = (
                blocks
                if offset == 0
                else torch.nn.functional.pad(
                    blocks[:, :-offset], (0, 0, 0, 0, offset, 0)
                )
            )
            output = output + coefficients[:, :, offset] * values
        return output.reshape(batch, length, hidden_size)

    def prepare(self, hidden_states):
        dynamic = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.kernel_size, self.num_groups
        )
        return (
            self._convolve(hidden_states, dynamic[..., 0, :, :], 0),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden_states, dynamic_kernel):
        return self._convolve(hidden_states, dynamic_kernel, 1)


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_torchspec_reference(block_size: int) -> None:
    """Flat-row convolution equals the authors' batched formulation."""
    torch.manual_seed(20260908)
    hidden, taps, group_size, blocks = 64, 2, 16, 3
    reference = _ReferenceGroupedConv(hidden, block_size, taps, group_size)
    with torch.no_grad():
        reference.base_kernel.copy_(torch.randn_like(reference.base_kernel))
        reference.kernel_projection.weight.copy_(
            torch.randn_like(reference.kernel_projection.weight) * 0.05
        )
    x = torch.randn(blocks, block_size, hidden)
    with torch.no_grad():
        expected, expected_kernel = reference.prepare(x)
        expected_finish = reference.finish(torch.sin(expected), expected_kernel)

        flat = x.reshape(blocks * block_size, hidden)
        dynamic = (flat @ reference.kernel_projection.weight.T).reshape(
            flat.shape[0], 2, taps, hidden // group_size
        )
        got = _grouped_conv(
            flat,
            dynamic[:, 0],
            reference.base_kernel[0],
            block_size,
            hidden // group_size,
            group_size,
            taps,
        )
        torch.testing.assert_close(got, expected.reshape_as(flat), rtol=1e-5, atol=1e-5)
        got_finish = _grouped_conv(
            torch.sin(got),
            dynamic[:, 1],
            reference.base_kernel[1],
            block_size,
            hidden // group_size,
            group_size,
            taps,
        )
        torch.testing.assert_close(
            got_finish, expected_finish.reshape_as(flat), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_per_block_reference(block_size: int) -> None:
    torch.manual_seed(20260908)
    hidden, groups, group_size, taps, blocks = 64, 4, 16, 2, 3
    tokens = blocks * block_size
    x = torch.randn(tokens, hidden)
    delta = torch.randn(tokens, taps, groups) * 0.1
    base = torch.randn(taps, hidden)
    got = _grouped_conv(x, delta, base, block_size, groups, group_size, taps)

    # Reference: per block, row t = sum_tap coef[t, tap] * row[t - tap].
    xb = x.view(blocks, block_size, groups, group_size)
    db = delta.view(blocks, block_size, taps, groups)
    expected = torch.zeros_like(xb)
    for b in range(blocks):
        for t in range(block_size):
            for tap in range(taps):
                if t - tap < 0:
                    continue
                coef = base[tap].view(groups, group_size) + db[b, t, tap][:, None]
                expected[b, t] += coef * xb[b, t - tap]
    torch.testing.assert_close(got, expected.view(tokens, hidden), rtol=1e-5, atol=1e-5)


class _DummyModule(nn.Module):
    """Stand-in for TP-sharded primitives: records its arguments, owns one
    zero-size weight so parameter paths still resolve."""

    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        type(self).calls.append((args, kwargs))
        self.weight = nn.Parameter(torch.empty(0))


def _build_dflash2_model(monkeypatch: pytest.MonkeyPatch) -> DFlash2Model:
    """Construct DFlash2Model on CPU with sharded primitives replaced."""

    class _Linear(_DummyModule):
        calls = []

    class _Attention(_DummyModule):
        calls = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.o_proj = SimpleNamespace(reduce_results=True)
            self.rotary_emb = None

    monkeypatch.delenv("VLLM_DSPARK_COMPACT_ROPE", raising=False)
    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", _Linear)
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", _Linear)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", _Linear)
    monkeypatch.setattr(dspark_mla, "RMSNorm", _DummyModule)
    monkeypatch.setattr(dspark_mla, "MultiHeadLatentAttention", _Attention)
    monkeypatch.setattr(dspark_mla, "KimiMLP", _DummyModule)
    monkeypatch.setattr(dflash2_mla, "ReplicatedLinear", _Linear)

    config = _config()
    config.hidden_size = 64
    config.intermediate_size = 128
    config.vocab_size = 256
    config.draft_vocab_size = 256
    config.num_hidden_layers = 2
    config.dflash_config["selector_rank"] = 8
    config.dflash_config["selector_top_k"] = 4
    normalize_dflash2_config(config)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config),
            num_speculative_tokens=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(dtype=torch.float32),
        cache_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=9),
    )
    model = DFlash2Model(vllm_config=vllm_config, start_layer_id=93, prefix="model")
    model._linear_calls = _Linear.calls  # type: ignore[attr-defined]
    return model


@pytest.mark.cpu_test
def test_dflash2_model_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Markov head; convolutions and selector sized from the config."""
    model = _build_dflash2_model(monkeypatch)
    params = dict(model.named_parameters())
    assert model.markov_head is None
    assert params["candidate_selector.predecessor_codebook"].shape == (256, 8)
    assert params["candidate_selector.successor_codebook"].shape == (256, 8)
    assert model.candidate_selector.top_k == 4
    for layer in range(2):
        for conv in ("attention_conv", "mlp_conv"):
            assert params[f"layers.{layer}.{conv}.base_kernel"].shape == (2, 2, 64)
            block = getattr(model.layers[layer], conv)
            assert block.block_size == 5  # anchor + four mask tokens
            assert block.num_groups == 4
    projection_sizes = [
        (args[0], args[1])
        for args, _ in model._linear_calls  # type: ignore[attr-defined]
        if isinstance(args[1], int)
    ]
    # kernel_projection: 2 sides x 2 taps x 4 groups; selector: rank 8;
    # context projection: 5 taps x target hidden (= hidden when unspecified).
    assert (64, 16) in projection_sizes
    assert (64, 8) in projection_sizes
    assert (5 * 64, 64) in projection_sizes


@pytest.mark.cpu_test
def test_checkpoint_names_resolve_to_model_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every DFlash2-specific checkpoint tensor lands on a backbone parameter.

    MLA and MLP internals are stand-ins here; their names are covered by the
    DSpark backbone's own mapping tests.
    """
    model = _build_dflash2_model(monkeypatch)
    params = set(dict(model.named_parameters()))
    mapper = DFlash2ForCausalLM.hf_to_vllm_mapper
    for name in CHECKPOINT_TENSORS:
        if ".self_attn." in name or ".mlp." in name:
            continue
        # The CPU model has two layers; fold the checkpoint's five onto them.
        folded = re.sub(
            r"layers\.(\d+)\.", lambda m: f"layers.{int(m.group(1)) % 2}.", name
        )
        target = rename_dflash2_checkpoint_name(folded)
        if target is None:
            assert name == "embed_tokens.weight"
            continue
        mapped, shard_id = mapper._map_name_with_shard(target)
        assert shard_id is None
        assert mapped.startswith("model.")
        assert mapped.removeprefix("model.") in params, (name, mapped)


def test_dflash2_draft_uses_the_k3_dspark_virtual_tp_plan() -> None:
    """Nine-rank serving pads the draft's 64 heads like the K3 DSpark draft."""
    from vllm.config.virtual_tp import (
        _is_dflash_draft_config,
        _is_kimi_k3_dspark_config,
    )

    hf_config = _config()
    hf_config.model_type = "eagle"  # the EAGLEConfig wrapper of method dflash
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    assert _is_kimi_k3_dspark_config(model_config)
    assert not _is_dflash_draft_config(model_config)
