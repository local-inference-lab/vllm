# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.hisparse.binding as attn_utils_module
from tests.v1.attention.utils import dense_kv_cache_views
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MultipleOf
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.binding import allocate_hisparse_kv_caches
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
    init_attn_backend,
    synchronize_attention_impl_kv_cache_layout,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


@pytest.mark.parametrize(
    ("enabled", "block_size", "main_sizes", "indexer_sizes", "expected"),
    [
        (True, 256, [64], [64], 64),
        (True, 64, [32, 64], [16, 32], 32),
        (True, 64, [MultipleOf(16)], [32], 32),
        (True, 64, [64], [32], None),
        (False, 256, [64], [64], 256),
    ],
)
def test_get_kv_cache_spec_resolves_hisparse_block_size(
    monkeypatch, enabled, block_size, main_sizes, indexer_sizes, expected
):
    """Resolve shared MLA geometry before planning; leave other specs alone."""
    specs = {
        "main": MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        ),
        "indexer": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
        "dense": FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
    }
    layers = {}
    for name, sizes in zip(specs, [main_sizes, indexer_sizes, [block_size]]):
        backend = SimpleNamespace(
            customize_spec=AttentionBackend.customize_spec,
            get_supported_kernel_block_sizes=lambda sizes=sizes: sizes,
        )
        layers[name] = SimpleNamespace(
            get_kv_cache_spec=lambda _, spec=specs[name]: spec,
            get_attn_backend=lambda backend=backend: backend,
        )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object() if enabled else None)
    )
    if expected is None:
        with pytest.raises(ValueError, match="supported by every sparse"):
            attn_utils.get_kv_cache_spec(config)
        return

    resolved = attn_utils.get_kv_cache_spec(config)
    assert resolved["main"].block_size == resolved["indexer"].block_size == expected
    assert resolved["dense"] is specs["dense"]
    assert all(spec.block_size == block_size for spec in specs.values())


@pytest.mark.parametrize("offset", range(4))
def test_mm_prefix_ranges_include_aligned_image_sentinels(offset):
    is_embed = torch.zeros(384, dtype=torch.bool)
    is_embed[8:-8] = True
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="image",
        mm_position=PlaceholderRange(offset=offset, length=384, is_embed=is_embed),
    )
    features = {"request": [feature]}

    assert compute_mm_prefix_ranges(["request"], features, sliding_window=128) == {
        0: []
    }
    assert compute_mm_prefix_ranges(
        ["request"],
        features,
        sliding_window=128,
        clamp_sliding_window=True,
        span_leading_pad_modulus=4,
    ) == {0: [(3, offset + 383)]}


def test_mm_prefix_ranges_preserve_unaligned_embed_ranges():
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="image",
        mm_position=PlaceholderRange(offset=7, length=32),
    )
    assert compute_mm_prefix_ranges(
        ["request"], {"request": [feature]}, sliding_window=128
    ) == {0: [(7, 38)]}


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport):
        self.support = support

    def get_cudagraph_support(self, *_args):
        return self.support


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


class _CachingMetadataBuilder:
    supports_update_block_table = True

    def __init__(self):
        self.num_builds = 0
        self.num_updates = 0
        self.state = torch.zeros(1, dtype=torch.int32)
        self.token_mapping = torch.full((2,), -1, dtype=torch.int32)

    def build(self, common_prefix_len, common_attn_metadata, **_kwargs):
        self.num_builds += 1
        self.state.fill_(self.num_builds)
        return SimpleNamespace(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            is_prefilling=common_attn_metadata.is_prefilling,
            state=self.state,
            token_mapping=common_attn_metadata.token_to_req_indices(self.token_mapping),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def update_block_table(self, metadata, block_table, slot_mapping):
        self.num_updates += 1
        return SimpleNamespace(
            block_table=block_table,
            slot_mapping=slot_mapping,
            reused=metadata,
            is_prefilling=metadata.is_prefilling,
            state=metadata.state,
            token_mapping=metadata.token_mapping,
        )


def test_attention_impl_cache_layout_preserves_model_specific_dtype():
    target_cache_config = SimpleNamespace(
        cache_dtype="fp8_ds_mla", kv_cache_layout=None
    )
    draft_cache_config = SimpleNamespace(cache_dtype="fp8", kv_cache_layout=None)
    layers = {
        "target": SimpleNamespace(
            impl=SimpleNamespace(cache_config=target_cache_config)
        ),
        "draft": SimpleNamespace(impl=SimpleNamespace(cache_config=draft_cache_config)),
    }

    synchronize_attention_impl_kv_cache_layout(layers, "BLHNC")

    assert target_cache_config.kv_cache_layout == "BLHNC"
    assert draft_cache_config.kv_cache_layout == "BLHNC"
    assert target_cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache_config.cache_dtype == "fp8"


def test_attention_builders_keep_target_and_draft_model_configs(monkeypatch):
    import vllm.v1.worker.gpu.attn_utils as attn_utils

    class Builder(_FakeMetadataBuilder):
        requires_block_table_width = False

        def __init__(self, spec, layer_names, config, device):
            super().__init__(AttentionCGSupport.ALWAYS)
            self.config = config

        def set_kernel_block_size(self, block_size):
            self.kernel_block_size = block_size

    class Backend:
        @staticmethod
        def full_cls_name():
            return (__name__, "Backend")

        @staticmethod
        def get_supported_kernel_block_sizes():
            return [16]

        @staticmethod
        def get_builder_cls():
            return Builder

    configs = [
        SimpleNamespace(
            cache_config=SimpleNamespace(
                kv_cache_layout=None, kv_sharing_fast_prefill=False
            ),
            parallel_config=SimpleNamespace(use_ubatching=False),
        )
        for _ in range(2)
    ]
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: Backend, num_heads=8)
        for name in ("target", "draft")
    }
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _: {})
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda config, layer_type, names: {name: layers[name] for name in names},
    )
    spec = FullAttentionSpec(
        block_size=32, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    cache = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(layers), spec)],
    )

    groups, _, kernel_sizes = init_attn_backend(
        cache,
        configs[0],
        torch.device("cpu"),
        layer_vllm_configs={"draft": configs[1]},
    )

    assert kernel_sizes == [16]
    assert len(groups[0]) == 2
    for group, expected_config in zip(groups[0], configs):
        assert group.get_metadata_builder(0).config is expected_config


@pytest.mark.parametrize("num_ubatches", [1, 2])
@pytest.mark.parametrize("num_target_groups,num_draft_groups", [(4, 1), (8, 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mla_prefill_scratch_shares_groups_but_isolates_execution_lanes(
    monkeypatch, num_ubatches, num_target_groups, num_draft_groups, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class Builder(MLACommonMetadataBuilder):
        def __init__(self, spec, layer_names, config, device):
            self.chunked_prefill_workspace = torch.empty(
                (128, spec.head_size), dtype=spec.dtype, device=device
            )

        def get_cudagraph_support(self, *_args):
            return AttentionCGSupport.ALWAYS

    class Backend:
        @staticmethod
        def full_cls_name():
            return (__name__, "MLABackend")

        @staticmethod
        def get_supported_kernel_block_sizes():
            return [16]

        @staticmethod
        def get_builder_cls():
            return Builder

    configs = [
        SimpleNamespace(
            cache_config=SimpleNamespace(
                kv_cache_layout=None, kv_sharing_fast_prefill=False
            ),
            parallel_config=SimpleNamespace(
                use_ubatching=num_ubatches > 1, num_ubatches=num_ubatches
            ),
        )
        for _ in range(2)
    ]
    names = [f"target.{i}" for i in range(num_target_groups)] + [
        f"draft.{i}" for i in range(num_draft_groups)
    ]
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: Backend, num_heads=6)
        for name in names
    }
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _: {})
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda config, layer_type, names: {name: layers[name] for name in names},
    )
    spec = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    cache = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([name], spec) for name in names],
    )
    groups, _, _ = init_attn_backend(
        cache,
        configs[0],
        torch.device(device),
        layer_vllm_configs={
            name: configs[1] for name in names if name.startswith("draft")
        },
    )
    target = [group[0] for group in groups[:num_target_groups]]
    draft = [group[0] for group in groups[num_target_groups:]]
    pointers = set()
    for lane, lane_groups in enumerate((target, draft)):
        for ubatch in range(num_ubatches):
            scratch = (
                lane_groups[0].get_metadata_builder(ubatch).chunked_prefill_workspace
            )
            pointers.add(scratch.data_ptr())
            scratch.fill_(1 + lane * num_ubatches + ubatch)
            assert all(
                group.get_metadata_builder(ubatch).chunked_prefill_workspace is scratch
                for group in lane_groups
            )
    assert len(pointers) == 2 * num_ubatches
    for lane, lane_groups in enumerate((target, draft)):
        for ubatch in range(num_ubatches):
            scratch = (
                lane_groups[0].get_metadata_builder(ubatch).chunked_prefill_workspace
            )
            assert torch.all(scratch == 1 + lane * num_ubatches + ubatch)


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,  # type: ignore[arg-type]
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,  # type: ignore[arg-type]
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)  # type: ignore[arg-type]
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_get_kv_sharing_fast_prefill_eligible_layers(monkeypatch: pytest.MonkeyPatch):
    """Fast prefill applies to the contiguous suffix of KV-sharing layers.

    Draft-model layers register after the target model's and may share KV, so
    they must not extend (or break) the target's eligible suffix.
    """

    def check(
        layer_names: list[str],
        shared: dict[str, str],
        draft_layer_names: set[str] | None = None,
    ) -> set[str]:
        monkeypatch.setattr(
            attn_utils,
            "get_layers_from_vllm_config",
            lambda *a, **k: {name: None for name in layer_names},
        )
        monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda *a: shared)
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=True)
        )
        return attn_utils.get_kv_sharing_fast_prefill_eligible_layers(
            vllm_config, draft_layer_names
        )

    # No KV sharing: nothing is eligible.
    assert check(["t0", "t1"], {}) == set()

    # Trailing run of sharing layers (YOCO-style second half).
    assert check(["t0", "t1", "t2", "t3"], {"t2": "t1", "t3": "t1"}) == {"t2", "t3"}

    # A non-sharing layer after a sharing one breaks the suffix.
    assert check(["t0", "t1", "t2", "t3"], {"t1": "t0", "t3": "t0"}) == {"t3"}

    # KV-sharing draft layers at the end are collected without an exclusion...
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
    ) == {"t2", "t3", "d0", "d1"}

    # ...so the runner excludes them: skipped, not collected, and they do not
    # break the target's trailing run.
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
        draft_layer_names={"d0", "d1"},
    ) == {"t2", "t3"}

    # Feature flag off: nothing is eligible even with sharing layers.
    monkeypatch.setattr(
        attn_utils, "get_layers_from_vllm_config", lambda *a, **k: {"t0": None}
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda *a: {"t0": "t0"}
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(vllm_config) == set()


class _FakeSharedHostRegion:
    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.base_tensor = torch.empty(1, dtype=torch.int8)

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_profiling_cleanup_releases_tp_shared_region_once(monkeypatch):
    """TP-shared profiling pools must use region-aware chunk cleanup."""
    region = _FakeSharedHostRegion()
    runtime = SimpleNamespace(
        _host_cache=object(),
        registered_host_pool=region.base_tensor,
        hot_backing=object(),
        shared_host_region=region,
    )
    forward_context = {
        "layer": SimpleNamespace(
            hisparse_cache=SimpleNamespace(runtime=runtime),
        )
    }
    released = []

    def release_pinned_state(runtimes, pinned_host_pools, shared_host_region):
        released.append((runtimes, pinned_host_pools, shared_host_region))

    monkeypatch.setattr(
        attn_utils_module,
        "release_pinned_state",
        release_pinned_state,
    )

    attn_utils_module.release_hisparse_profiling_cache(forward_context)

    assert released == [([runtime], [], region)]


@pytest.mark.parametrize("failure_phase", ["allocation", "binding", "buffers"])
def test_init_hisparse_rolls_back_shared_region(monkeypatch, failure_phase):
    """A failure after mmap allocation must not leak the shared registration."""
    region = _FakeSharedHostRegion()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
    )

    def allocate(*args):
        args[-1].shared_region = region
        if failure_phase == "allocation":
            raise RuntimeError("initialization failed")
        return {}

    def bind(**kwargs):
        if failure_phase == "binding":
            raise RuntimeError("initialization failed")
        return []

    def buffers(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(attn_utils_module, "allocate_hisparse_kv_caches", allocate)
    monkeypatch.setattr(attn_utils_module, "bind_hisparse_kv_caches", bind)
    monkeypatch.setattr(
        attn_utils_module, "initialize_hisparse_runtime_buffers", buffers
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        attn_utils_module.init_hisparse_kv_cache(
            SimpleNamespace(),
            torch.device("cpu"),
            [],
            vllm_config,
            {},
            SimpleNamespace(),
        )
    assert region.cleanup_calls == 1


@pytest.mark.parametrize("for_capture", [False, True])
@pytest.mark.parametrize("same_spec", [False, True])
def test_build_attn_metadata_reuses_equivalent_cache_group_builds(
    for_capture, same_spec
):
    builders = [_CachingMetadataBuilder(), _CachingMetadataBuilder()]
    groups = []
    cache_groups = []
    for group_id, builder in enumerate(builders):
        spec = FullAttentionSpec(
            block_size=16 if same_spec or group_id == 0 else 32,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        layer_name = f"layer.{group_id}"
        group = AttentionGroup(
            _TargetBackend,  # type: ignore[arg-type]
            [layer_name],
            spec,
            group_id,
        )
        group.metadata_builders = [builder]  # type: ignore[list-item]
        groups.append([group])
        cache_groups.append(KVCacheGroupSpec([layer_name], spec))

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=cache_groups,
    )
    block_tables = [
        torch.full((2, 1), group_id, dtype=torch.int32) for group_id in range(2)
    ]
    slot_mappings = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    is_prefilling = torch.ones(2, dtype=torch.bool)
    model_metadata = SimpleNamespace(
        get_extra_common_attn_kwargs=Mock(
            side_effect=lambda *_: {"is_prefilling": is_prefilling}
        ),
        get_extra_attn_kwargs=Mock(return_value={}),
    )

    build_kwargs = dict(
        attn_groups=groups,
        num_reqs=2,
        num_tokens=2,
        query_start_loc_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        model_specific_attn_metadata=model_metadata,
    )
    metadata = build_attn_metadata(**build_kwargs, for_cudagraph_capture=for_capture)

    assert [builder.num_builds for builder in builders] == [1, 0 if same_spec else 1]
    assert [builder.num_updates for builder in builders] == [0, 1 if same_spec else 0]
    assert model_metadata.get_extra_common_attn_kwargs.call_count == (
        1 if same_spec else 2
    )
    assert metadata["layer.0"].block_table is block_tables[0]
    assert metadata["layer.1"].block_table is block_tables[1]
    for group_id in range(2):
        torch.testing.assert_close(
            metadata[f"layer.{group_id}"].slot_mapping, slot_mappings[group_id]
        )
    if same_spec:
        assert metadata["layer.1"].reused is metadata["layer.0"]
    # Distinct KV geometries still share the batch's query-to-request mapping.
    for item in metadata.values():
        assert item.token_mapping.data_ptr() == builders[0].token_mapping.data_ptr()
        assert item.token_mapping.tolist() == [0, 1]
    assert builders[1].token_mapping.tolist() == [-1, -1]
    assert all(item.is_prefilling is is_prefilling for item in metadata.values())
    captured_state = metadata["layer.1"].state
    build_kwargs["query_start_loc_cpu"][1] = 2
    build_kwargs["query_start_loc_gpu"][1] = 2
    build_kwargs["max_query_len"] = 2
    runtime = build_attn_metadata(**build_kwargs)
    assert captured_state.data_ptr() == runtime["layer.1"].state.data_ptr()
    assert captured_state.item() == 2
    assert all(item.token_mapping.tolist() == [0, 0] for item in runtime.values())


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "storage_block_size",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, None, 4, 64),
        ([256], None, 4, 64),
        ([64], None, 16, 16),
        ([64], 256, 4, 64),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    storage_block_size: int | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        storage_block_size=storage_block_size,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )


def test_allocate_hisparse_kv_caches_host_pool_and_view_less_specs():
    """Host tensors get their own backing; view-less specs keep the raw one."""
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    page = spec.page_size_bytes
    resident_spec = HiSparseResidentSpec(block_size=2, page_size=page)
    device_size = 4 * page
    config = KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=3 * page,
                layers=["source"],
                layer_stride=3 * page,
                block_stride=page,
                host_resident=True,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["indexer"],
                layer_stride=device_size,
                block_stride=page,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["resident"],
                layer_stride=device_size,
                block_stride=page,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec),
            KVCacheGroupSpec(["resident"], resident_spec),
        ],
    )
    host_buffers: list[torch.Tensor] = []

    def host_allocator(size: int) -> torch.Tensor:
        host_buffers.append(torch.zeros(size, dtype=torch.int8))
        return host_buffers[-1]

    caches = allocate_hisparse_kv_caches(
        config,
        torch.device("cpu"),
        KVCacheLayout.LBHNC,
        [2, 2, 2],
        SimpleNamespace(allocate=host_allocator),
    )
    assert len(config.kv_cache_tensors) == 3

    assert [buf.numel() for buf in host_buffers] == [3 * page]
    assert caches["source"].shape[0] == 3
    assert (
        caches["source"].untyped_storage().data_ptr()
        == host_buffers[0].untyped_storage().data_ptr()
    )
    assert caches["indexer"].shape[0] == 4
    backing = caches["resident"]
    assert backing.dtype == torch.int8 and backing.numel() >= device_size
    assert (
        backing.untyped_storage().data_ptr()
        == caches["indexer"].untyped_storage().data_ptr()
    )
