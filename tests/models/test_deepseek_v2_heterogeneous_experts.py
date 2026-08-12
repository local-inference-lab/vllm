# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for heterogeneous per-layer routed-expert widths.

These tests exercise the pure helpers that ``DeepseekV2MoE`` and
``DeepseekV2Model.load_weights`` use to resolve a layer's routed-expert
count and to map ROCm AITER fused shared-expert weights. Importing
``deepseek_v2`` itself works on a CPU-only host (it only lazy-loads CUDA
extensions), so we import the helpers directly rather than constructing the
full model, which would require CUDA/ROCm and a real checkpoint.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.exl3 import _experts_per_layer
from vllm.model_executor.models.deepseek_v2 import _layer_routed_expert_count
from vllm.model_executor.models.utils import _parse_layer_index


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

def _hetero_config():
    """A heterogeneous config: layer 2 is wide (256), the rest narrow (4).

    The global scalar ``n_routed_experts`` is deliberately set to the wide
    value (256) so that any code path that falls back to it for a NARROW
    layer would produce the wrong width -- this is exactly the B5 hazard.
    """
    return SimpleNamespace(
        n_routed_experts=256,
        n_routed_experts_per_layer=[4, 4, 256, 4],
        n_shared_experts=1,
    )


def _uniform_config():
    """A uniform (non-list) config: every layer has the same expert count."""
    return SimpleNamespace(n_routed_experts=8, n_shared_experts=1)


def _shared_expert_slot_name(name, config, j):
    """Mirror the mapping performed in ``DeepseekV2Model.load_weights``:

    the fused shared-expert chunk ``j`` is routed to expert slot
    ``mlp.experts.{<this layer's routed-expert count> + j}``.
    """
    offset = _layer_routed_expert_count(name, config)
    return name.replace("mlp.shared_experts", f"mlp.experts.{offset + j}")


# ---------------------------------------------------------------------------
# C11: the unified layer-index parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.3", 3),                 # prefix ending at layers.<N>
        ("model.layers.3.mlp.experts", 3),     # trailing components
        ("model.layers.77.mlp.shared_experts.gate_proj.weight", 77),
        ("layers.0", 0),                       # bare prefix
    ],
)
def test_parse_layer_index_matches(prefix, expected):
    assert _parse_layer_index(prefix) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "model.sublayers.3",   # 'layers' must be preceded by ^ or '.'
        "model.foo.bar",       # no layer index at all
        "model.layers",        # no digit
        "",
    ],
)
def test_parse_layer_index_rejects(prefix):
    assert _parse_layer_index(prefix) is None


# ---------------------------------------------------------------------------
# Per-layer width selection (B5 / C10 / C15)
# ---------------------------------------------------------------------------

def test_per_layer_width_narrow_layer():
    cfg = _hetero_config()
    # Layer 0 is narrow even though the global scalar is 256.
    assert _layer_routed_expert_count(
        "model.layers.0.mlp.experts", cfg) == 4


def test_per_layer_width_wide_layer():
    cfg = _hetero_config()
    assert _layer_routed_expert_count(
        "model.layers.2.mlp.experts", cfg) == 256


def test_per_layer_width_uses_global_scalar_name_not_list_index():
    """The list is indexed by the GLOBAL model layer number parsed from the
    prefix, not by the position of 'experts' in the string."""
    cfg = _hetero_config()
    # An expert weight name still resolves to the owning layer's width.
    assert _layer_routed_expert_count(
        "model.layers.3.mlp.experts.5.gate_proj.weight", cfg) == 4


# ---------------------------------------------------------------------------
# B5 regression: ROCm AITER fused shared-expert name mapping
# ---------------------------------------------------------------------------

def test_shared_expert_mapping_narrow_layer():
    """B5: a NARROW layer's shared expert must land at slot <narrow width>+j,
    NOT at the global scalar ``n_routed_experts`` (256). The per-layer
    FusedMoE is built with num_experts=<layer width>, so the global offset
    would point at a non-existent / wrong slot."""
    cfg = _hetero_config()
    name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
    # The fix's offset for this layer...
    assert _layer_routed_expert_count(name, cfg) == 4
    # ...must differ from the pre-fix global-scalar offset.
    assert cfg.n_routed_experts == 256
    assert _shared_expert_slot_name(name, cfg, 0) == \
        "model.layers.0.mlp.experts.4.gate_proj.weight"


def test_shared_expert_mapping_wide_layer():
    cfg = _hetero_config()
    name = "model.layers.2.mlp.shared_experts.gate_proj.weight"
    assert _layer_routed_expert_count(name, cfg) == 256
    assert _shared_expert_slot_name(name, cfg, 0) == \
        "model.layers.2.mlp.experts.256.gate_proj.weight"


def test_shared_expert_mapping_multiple_shared_experts():
    """With n_shared_experts > 1, each chunk j maps to <width>+j."""
    cfg = SimpleNamespace(
        n_routed_experts=256,
        n_routed_experts_per_layer=[4, 4, 256, 4],
        n_shared_experts=2,
    )
    name = "model.layers.0.mlp.shared_experts.down_proj.weight"
    assert _shared_expert_slot_name(name, cfg, 0) == \
        "model.layers.0.mlp.experts.4.down_proj.weight"
    assert _shared_expert_slot_name(name, cfg, 1) == \
        "model.layers.0.mlp.experts.5.down_proj.weight"


def test_mapping_table_size_bounds_all_layer_offsets():
    """The name-mapping table is sized at max(widths)+n_shared; every layer's
    shared-expert slot (width+j) must fall within that range so the table has
    an entry for it. This is the 'consistency' half of the B5 fix."""
    cfg = _hetero_config()
    widths = cfg.n_routed_experts_per_layer
    table_size = max(widths) + cfg.n_shared_experts
    for layer_idx, width in enumerate(widths):
        for j in range(cfg.n_shared_experts):
            assert width + j < table_size, (layer_idx, j)


# ---------------------------------------------------------------------------
# C10: unparseable prefix raises (no silent fallback to global scalar)
# ---------------------------------------------------------------------------

def test_unparseable_prefix_raises():
    cfg = _hetero_config()
    with pytest.raises(ValueError, match="layer index cannot be parsed"):
        _layer_routed_expert_count("model.attention.dense", cfg)


# ---------------------------------------------------------------------------
# C15: out-of-range index raises
# ---------------------------------------------------------------------------

def test_out_of_range_layer_index_raises():
    cfg = _hetero_config()  # 4 entries -> valid indices 0..3
    with pytest.raises(ValueError, match="requires one"):
        _layer_routed_expert_count("model.layers.99.mlp.experts", cfg)


def test_experts_per_layer_bounds_check():
    metadata = {"experts_per_layer": [4, 4, 256, 4]}
    with pytest.raises(ValueError, match="out of range"):
        _experts_per_layer(metadata, 99)
    # valid index works
    assert _experts_per_layer(metadata, 2) == 256


def test_experts_per_layer_scalar_unchanged():
    metadata = {"experts_per_layer": 8}
    assert _experts_per_layer(metadata, 0) == 8
    assert _experts_per_layer(metadata, 99) == 8  # scalar ignores index


def test_experts_per_layer_none_index_returns_widest():
    metadata = {"experts_per_layer": [4, 4, 256, 4]}
    assert _experts_per_layer(metadata, None) == 256


# ---------------------------------------------------------------------------
# Uniform (non-list) config is unchanged
# ---------------------------------------------------------------------------

def test_uniform_config_returns_scalar_for_any_layer():
    cfg = _uniform_config()
    for prefix in (
        "model.layers.0.mlp.experts",
        "model.layers.42.mlp.shared_experts.gate_proj.weight",
        "model.layers.0",
    ):
        assert _layer_routed_expert_count(prefix, cfg) == 8


def test_uniform_config_shared_expert_mapping():
    cfg = _uniform_config()
    name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
    assert _shared_expert_slot_name(name, cfg, 0) == \
        "model.layers.0.mlp.experts.8.gate_proj.weight"
