# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing.

Every cache group packs its layers densely into one block; groups overlay each other
(a block ID is owned by one group at a time), so the packed block stride is the
largest group's packing. The layout decides whether the layer dim sits outside the
block dim (a contiguous region per layer) or inside it (all layers' pages within each
block); the allocation is the same either way.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _pool_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

MEMORY = 8 * 1024 * 1024


def _mla(head_size: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=head_size, dtype=torch.uint8
    )


def _full() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )


def _uniform_group(specs: dict) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
    )


def _mixed_page_groups(n_mla=3, n_idx=3, n_swa=5):
    """A DeepSeek-V4-style hybrid: two groups with different page mixes."""
    g1 = {f"mla.{i}": _mla(512) for i in range(n_mla)}
    g1.update({f"idx.{i}": _mla(128) for i in range(n_idx)})
    g2 = {f"swa.{i}": _mla(512) for i in range(n_swa)}
    return [_uniform_group(g1), _uniform_group(g2)], g1, g2


def _mock_vllm_config(layout: str | None):
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    return config


def _pages(groups) -> dict[str, int]:
    return {
        name: group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec.page_size_bytes
        for group in groups
        for name in group.layer_names
    }


def _expected_bytes_per_block(groups) -> int:
    pages = _pages(groups)
    return max(sum(pages[n] for n in g.layer_names) for g in groups)


def _bind(config, layout: str):
    return allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout], None)


def test_v41_mixed_cache_pages_preserve_request_partial_states(monkeypatch):
    from vllm import envs
    from vllm.models.deepseek_v4_1.compressor import (
        CompressorBackend,
        CompressorStateCache,
    )
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )

    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", None)
    vllm_config = _mock_vllm_config(None)
    vllm_config.kv_transfer_config = None
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.speculative_config = None
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.scheduler_config.max_num_batched_tokens = 1024
    vllm_config.max_in_flight_tokens = 1024
    vllm_config.model_config.max_model_len = 32768
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    partial = CompressorStateCache(vllm_config, "partial.0")
    partial_spec = partial.get_kv_cache_spec(vllm_config)
    specs = {
        "swa": SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=528,
            dtype=torch.uint8,
            sliding_window=128,
            alignment=None,
        ),
        "index": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=68,
            state_content_bytes=68,
            dtype=torch.uint8,
            alignment=None,
        ),
        "main2": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=288,
            dtype=torch.uint8,
            tokens_per_state=2,
            alignment=None,
        ),
        "main1": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=288,
            dtype=torch.uint8,
            alignment=None,
        ),
    }
    supported = get_supported_kv_cache_layouts(
        [DeepseekV41B12xBackend, CompressorBackend]
    )
    # Profiling resolves the backend preference before the full specs arrive.
    layout = resolve_kv_cache_layout(vllm_config, [[item.name for item in supported]])
    # Exercise the real group planner, not prebuilt groups that bypass its
    # logical-block-size requirements.
    groups = get_kv_cache_groups(
        vllm_config,
        {
            **specs,
            "partial.0": partial_spec,
            "partial.1": partial_spec,
        },
    )
    config = get_kv_cache_config_from_groups(vllm_config, groups, MEMORY)
    views = _bind(config, layout.name)
    blocks = {
        name: group_id + 1
        for group_id, group in enumerate(groups)
        for name in group.layer_names
    }
    for view in views.values():
        assert view.data_ptr() % 16 == 0
        assert view.stride(0) * view.element_size() % 16 == 0
    partial.bind_kv_cache(views["partial.0"])
    partial_block = blocks["partial.0"]
    partial.kv_cache[partial_block].fill_(4)
    views["partial.1"][blocks["partial.1"]].fill_(5)
    # Components sharing a block table coexist; independent groups own
    # distinct block IDs. Ring rows must also retain independent values.
    partial.kv_cache[partial_block, 1024:2048].fill_(7)
    views["swa"][blocks["swa"]].fill_(17)
    views["index"][blocks["index"]].fill_(23)
    views["main2"][blocks["main2"]].fill_(31)
    assert (partial.kv_cache[partial_block, :1024] == 4).all()
    assert (partial.kv_cache[partial_block, 1024:2048] == 7).all()
    assert (partial.kv_cache[partial_block, 2048:] == 4).all()
    assert (views["partial.1"][blocks["partial.1"]] == 5).all()
    assert (views["swa"][blocks["swa"]] == 17).all()
    assert (views["index"][blocks["index"]] == 23).all()


def test_v41_full_context_packs_shared_global_cache_without_page_inflation():
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention, _Cache
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache
    from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

    config = _mock_vllm_config("BLHNC")
    config.cache_config.block_size = 256
    config.kv_transfer_config = None
    config.compilation_config.static_forward_context = {}
    config.speculative_config = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.scheduler_config.max_num_batched_tokens = 4096
    config.max_in_flight_tokens = 8192
    config.model_config.max_model_len = 1048576
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    specs = {}
    global_names = []
    for layer in range(43):
        prefix = f"model.layers.{layer}.self_attn"
        swa = _Cache(
            config, prefix + ".swa_cache", kind="swa", window=128, draft=layer >= 40
        )
        specs[swa.prefix] = swa.get_kv_cache_spec(config)
        if layer not in (2, 8, 14, 20):
            continue
        ratio = 1 if layer == 20 else 2
        specs[prefix] = DeepseekV4Attention.get_kv_cache_spec(
            SimpleNamespace(is_kv_source=True, compress_ratio=ratio), config
        )
        index = _Cache(config, prefix + ".indexer.k_cache", kind="index", ratio=ratio)
        specs[index.prefix] = index.get_kv_cache_spec(config)
        global_names.extend((prefix, index.prefix))
        if ratio == 2:
            state = CompressorStateCache(config, prefix + ".compressor.state_cache")
            specs[state.prefix] = state.get_kv_cache_spec(config)

    groups = get_kv_cache_groups(config, specs)
    # Global payload is 890 MiB at 1M. Allow bounded in-flight SWA/partial-state
    # storage, but not the former 11.20 GiB largest-page allocation inflation.
    required = _max_memory_usage_bytes_from_groups(config, groups)
    assert required < 2 * 1024**3
    global_groups = [
        group for group in groups if set(group.layer_names).intersection(global_names)
    ]
    charged_global_bytes = sum(
        _get_kv_cache_bytes_per_block(groups)
        * (
            (1048576 + group.kv_cache_spec.block_size - 1)
            // group.kv_cache_spec.block_size
        )
        for group in global_groups
    )
    assert charged_global_bytes == 890 * 1024**2

    allocation = get_kv_cache_config_from_groups(config, groups, MEMORY)
    views = _bind(allocation, "BLHNC")
    # All components of the shared global history coexist in one logical
    # block; filling any component must not overwrite another component.
    for value, name in enumerate(global_names, 1):
        views[name][1].fill_(value)
    for value, name in enumerate(global_names, 1):
        assert (views[name][1] == value).all()


class TestDensePacking:
    def test_bytes_per_block_is_largest_group(self):
        groups, g1, g2 = _mixed_page_groups()
        assert _get_kv_cache_bytes_per_block(groups) == _expected_bytes_per_block(
            groups
        )

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        # Groups overlay: both start at offset 0.
        assert [tensor.offset for tensor in config.kv_cache_tensors].count(0) == 2
        assert [tensor.layers for tensor in config.kv_cache_tensors] == [
            list(g1)[:3],
            list(g1)[3:],
            list(g2),
        ]

    def test_glm53_split_env_does_not_pad_other_mla_models(self, monkeypatch):
        groups, _, _ = _mixed_page_groups()
        expected = _expected_bytes_per_block(groups)
        assert expected % (64 * 132) != 0

        monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "512")

        assert _get_kv_cache_bytes_per_block(groups) == expected

    def test_layers_within_a_group_are_dense(self):
        groups, _, _ = _mixed_page_groups()
        pages = _pages(groups)
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        offsets = {
            name: tensor.offset + i * tensor.layer_stride
            for tensor in config.kv_cache_tensors
            for i, name in enumerate(tensor.layers)
        }
        for group in groups:
            expected = 0
            for name in group.layer_names:
                assert offsets[name] == expected
                expected += pages[name]

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_allocation_is_layout_invariant(self, layout):
        specs = {f"l.{i}": _full() for i in range(4)}
        groups = [KVCacheGroupSpec(list(specs), _full())]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        (tensor,) = config.kv_cache_tensors
        page = _full().page_size_bytes
        assert tensor.layers == list(specs)
        assert config.num_blocks == MEMORY // (4 * page)
        assert tensor.size == 4 * page * config.num_blocks
        if layout == "LBNHC":
            assert (tensor.layer_stride, tensor.block_stride) == (
                page * config.num_blocks,
                page,
            )
        else:
            assert (tensor.layer_stride, tensor.block_stride) == (page, 4 * page)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_single_group_mixed_pages_follows_layout(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        block_stride = _expected_bytes_per_block(groups)
        mla_tensor, idx_tensor = config.kv_cache_tensors
        assert mla_tensor.layers == ["mla.0", "mla.1"]
        assert idx_tensor.layers == ["idx.0"]
        assert {t.size for t in config.kv_cache_tensors} == {
            block_stride * config.num_blocks
        }
        if layout == "LBNHC":
            assert (
                idx_tensor.offset == 2 * _mla(512).page_size_bytes * config.num_blocks
            )
            assert mla_tensor.block_stride == _mla(512).page_size_bytes
        else:
            assert idx_tensor.offset == 2 * _mla(512).page_size_bytes
            assert mla_tensor.block_stride == block_stride

    def test_overlaid_groups_alias_and_stay_isolated(self):
        groups, g1, g2 = _mixed_page_groups()
        # Overlay models resolve to a block-outer layout at backend selection (the
        # model's backend declares it); mirror that here.
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLNHC"), groups, MEMORY
        )
        assert config.num_blocks == MEMORY // _expected_bytes_per_block(groups)
        assert _pool_bytes_per_block(groups) == _expected_bytes_per_block(groups)

        views = _bind(config, "BLNHC")
        assert set(views) == set(g1) | set(g2)
        assert views["swa.0"].data_ptr() == views["mla.0"].data_ptr()

        # A block is owned by one group at a time: writes to group-owned
        # blocks never disturb the other group's blocks.
        for i, name in enumerate(g1):
            views[name][0].fill_(i + 1)
            views[name][2].fill_(i + 1)
        for i, name in enumerate(g2):
            views[name][1].fill_(100 + i)
            views[name][3].fill_(100 + i)
        for i, name in enumerate(g1):
            assert (views[name][0].to(torch.int32) == i + 1).all()
            assert (views[name][2].to(torch.int32) == i + 1).all()
        for i, name in enumerate(g2):
            assert (views[name][1].to(torch.int32) == 100 + i).all()
            assert (views[name][3].to(torch.int32) == 100 + i).all()
        # Layers within a group are disjoint.
        views["mla.0"][0].fill_(77)
        for i, name in enumerate(list(g1)[1:], start=1):
            assert (views[name][0].to(torch.int32) == i + 1).all()

    def test_layer_compact_layout_rejected_for_overlaid_groups(self):
        # The layout has a single writer (backend-selection resolution); a layer-
        # compact layout reaching an overlay model's allocation is an error, not a
        # silent flip.
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LBNHC"), groups, MEMORY)

    def test_unresolved_layout_rejected(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(ValueError, match="has not been resolved"):
            get_kv_cache_config_from_groups(_mock_vllm_config(None), groups, MEMORY)

    def test_head_outer_layout_rejected_for_mixed_pages(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LHBNC"), groups, MEMORY)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_bound_views_round_trip(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        views = _bind(config, layout)
        for i, name in enumerate(specs):
            views[name].fill_(i + 1)
        for i, name in enumerate(specs):
            assert (views[name].to(torch.int32) == i + 1).all(), name
            assert views[name].shape[0] == config.num_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
