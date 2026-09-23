# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo fused-QKV FP8 checkpoints load exactly at every TP size that splits them.

MiMo V2.5/V2.6 store fused QKV as ``num_key_value_heads`` [Q | K | V] chunks
(the training TP layout), each with its own 128x128 block-scale grid.  A rank
that owns several chunks must not requantize: global layers put K and V rows
(192 + 128) in a shared scale block, so requantizing their de-interleaved
layout changes the stored FP8 values.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.mimo_v2 as mimo

pytestmark = pytest.mark.cpu_test

BLOCK = 128
CHUNKS = 4
GEOMETRY = {
    # name: (num_heads, num_kv_heads, head_dim, v_head_dim)
    "global": (64, 4, 192, 128),
    "sliding": (64, 8, 192, 128),
}


def _checkpoint(num_heads, num_kv_heads, head_dim, v_head_dim, hidden=256):
    gen = torch.Generator().manual_seed(0)
    rows = (num_heads * head_dim + num_kv_heads * (head_dim + v_head_dim)) // CHUNKS
    scale_rows = -(-rows // BLOCK)
    weight = (torch.randn(CHUNKS * rows, hidden, generator=gen) * 64).clamp(-448, 448)
    scale = torch.rand(CHUNKS * scale_rows, hidden // BLOCK, generator=gen) + 0.5
    return weight.to(torch.float8_e4m3fn), scale, rows, scale_rows


def _dequant(weight, scale):
    expanded = scale.double().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return weight.double() * expanded[: weight.shape[0], : weight.shape[1]]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("kind", ["global", "sliding"])
def test_fused_qkv_shards_keep_checkpoint_values(kind, tp_size):
    heads, kv_heads, head_dim, v_head_dim = GEOMETRY[kind]
    weight, scale, rows, scale_rows = _checkpoint(*GEOMETRY[kind])
    kv_chunk_rows = mimo._fused_qkv_kv_chunk_rows(
        heads, kv_heads, head_dim, v_head_dim, tp_size, CHUNKS
    )
    # Only global layers (K/V share a scale block) on ranks owning several
    # chunks need the padded layout.
    assert bool(kv_chunk_rows) == (kind == "global" and tp_size < CHUNKS)

    q_rows = heads // CHUNKS * head_dim
    k_rows = kv_heads // CHUNKS * head_dim
    q_size = heads // tp_size * head_dim
    k_size = kv_heads // tp_size * head_dim
    v_size = kv_heads // tp_size * v_head_dim
    per_rank = CHUNKS // tp_size
    for rank in range(tp_size):
        w_rank, s_rank = mimo._shard_fp8_qkv_proj(
            weight,
            scale,
            heads,
            kv_heads,
            head_dim,
            v_head_dim,
            rank,
            tp_size,
            CHUNKS,
            kv_chunk_rows=kv_chunk_rows,
        )
        out_rows = q_size + (
            per_rank * kv_chunk_rows if kv_chunk_rows else k_size + v_size
        )
        assert w_rank.shape == (out_rows, weight.shape[1])
        assert s_rank.shape == (-(-out_rows // BLOCK), scale.shape[1])

        # Split the rank's dequantized rows as the forward splits its output.
        q, k, v = mimo._split_qkv(
            _dequant(w_rank, s_rank).T, q_size, k_size, v_size, kv_chunk_rows
        )
        chunks = [
            _dequant(
                weight[c * rows : (c + 1) * rows],
                scale[c * scale_rows : (c + 1) * scale_rows],
            )
            for c in range(rank * per_rank, (rank + 1) * per_rank)
        ]
        # Bit-exact: the same FP8 values times the same block scales.
        assert torch.equal(q, torch.cat([c[:q_rows] for c in chunks]).T)
        assert torch.equal(
            k, torch.cat([c[q_rows : q_rows + k_rows] for c in chunks]).T
        )
        assert torch.equal(v, torch.cat([c[q_rows + k_rows :] for c in chunks]).T)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("kind", ["global", "sliding"])
@pytest.mark.parametrize("fused", [True, False])
def test_attention_projection_matches_rank_layout(monkeypatch, kind, tp_size, fused):
    created = {}

    def recorder(name):
        def build(*args, **kwargs):
            created[name] = (args, kwargs)
            return torch.nn.Identity()

        return build

    class Backend:
        name = "STUB_DIFFKV"

        @staticmethod
        def get_class():
            return Backend

        @staticmethod
        def set_head_size_v(size):
            pass

        @staticmethod
        def get_name():
            return "STUB_DIFFKV"

    monkeypatch.setattr(mimo, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(mimo, "QKVParallelLinear", recorder("qkv"))
    monkeypatch.setattr(mimo, "MergedColumnParallelLinear", recorder("merged"))
    monkeypatch.setattr(mimo, "RowParallelLinear", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(mimo, "get_rope", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(mimo, "Attention", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(
        mimo,
        "get_current_vllm_config",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(backend=Backend)),
    )
    heads, kv_heads, head_dim, v_head_dim = GEOMETRY[kind]
    attn = mimo.MiMoV2Attention(
        hidden_size=256,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        fused_qkv_chunks=CHUNKS if fused else 0,
        prefix="model.layers.0.self_attn",
    )
    if fused and kind == "global" and tp_size < CHUNKS:
        assert set(created) == {"merged"}
        _, output_sizes = created["merged"][0]
        # Q rows, then every chunk's K|V rows padded to whole scale blocks.
        assert output_sizes == [heads * head_dim, CHUNKS * 3 * BLOCK]
        assert attn.kv_chunk_rows == 3 * BLOCK
    else:
        assert set(created) == {"qkv"}
        assert attn.kv_chunk_rows == 0
