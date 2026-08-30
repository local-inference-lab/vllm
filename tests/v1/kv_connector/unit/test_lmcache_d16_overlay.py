# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[4]
DCP_LAYOUT = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/dcp_layout.py"
)
TRANSFER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
)
GROUPS = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/kv_cache_groups.py"
)
GROUP_EDITS = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/kv_cache_group_edits.py"
)
ADAPTER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/vllm_multi_process_adapter.py"
)
CONNECTOR = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/lmcache_mp_connector.py"
)
METADATA = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/lmcache_mp_metadata.py"
)
CORE_KV_CACHE_MANAGER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "vllm/v1/core/kv_cache_manager.py"
)
CORE_SINGLE_TYPE_MANAGER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "vllm/v1/core/single_type_kv_cache_manager.py"
)
CUDA_SLOT_STRIDE_PATCH = (
    ROOT
    / "docker/glm53-flash/lmcache-d16-overlay/lmcache_cuda_ops_slot_stride.patch"
)
SLOT_STRIDE_PATCHER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/patch_lmcache_slot_stride.py"
)
PADDED_STRIDE_PATCHER = (
    ROOT
    / "docker/glm53-flash/lmcache-d16-overlay/"
    "patch_lmcache_padded_packed_stride.py"
)


class AttentionSpec:
    def __init__(self, block_size: int):
        self.block_size = block_size


class MambaSpec:
    def __init__(self, block_size: int):
        self.block_size = block_size


def _load_layout():
    spec = importlib.util.spec_from_file_location("lmcache_dcp_layout", DCP_LAYOUT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exact_mamba_store_helper():
    tree = ast.parse(METADATA.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_exact_mamba_store_blocks"
    )
    namespace = {"LMCacheMPRequestTracker": object}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
            str(METADATA),
            "exec",
        ),
        namespace,
    )
    return namespace["apply_exact_mamba_store_blocks"]


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_group_converter(monkeypatch):
    def normalize(caches, _groups, _engine_type, _hints):
        return caches, [cache.kernel for cache in caches]

    def group_layers(_caches, formats, per_layer_group_idx):
        grouped: dict[tuple[int, str], list[int]] = {}
        for layer_idx, kernel in enumerate(formats):
            key = (per_layer_group_idx[layer_idx], kernel)
            grouped.setdefault(key, []).append(layer_idx)
        return [
            (SimpleNamespace(engine_group_idx=group_idx), indices)
            for (group_idx, _), indices in grouped.items()
        ]

    def engine_group_info(**kwargs):
        return SimpleNamespace(**kwargs)

    modules = {
        "lmcache": _module("lmcache"),
        "lmcache.logging": _module(
            "lmcache.logging", init_logger=lambda _: logging.getLogger(__name__)
        ),
        "lmcache.utils": _module(
            "lmcache.utils", EngineType=SimpleNamespace(VLLM="vllm")
        ),
        "lmcache.v1": _module("lmcache.v1"),
        "lmcache.v1.multiprocess": _module("lmcache.v1.multiprocess"),
        "lmcache.v1.multiprocess.group_view": _module(
            "lmcache.v1.multiprocess.group_view",
            EngineGroupInfo=engine_group_info,
        ),
        "lmcache.v1.gpu_connector": _module("lmcache.v1.gpu_connector"),
        "lmcache.v1.gpu_connector.utils": _module(
            "lmcache.v1.gpu_connector.utils",
            normalize_and_discover_per_layer_formats=normalize,
        ),
        "lmcache.v1.kv_layer_groups": _module(
            "lmcache.v1.kv_layer_groups",
            EXCLUDED_ENGINE_GROUP=-1,
            group_layers_by_identity=group_layers,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("lmcache_d16_groups", GROUPS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_engine_group_infos_from_vllm


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="zai-org/GLM-5.3-FP8"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=4,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=4,
            world_size=4,
        ),
        cache_config=SimpleNamespace(block_size=256, mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
    )


def test_dcp_layout_resolves_hybrid_geometry_and_cache_identity() -> None:
    layout = _load_layout()
    config = _config()
    kv_cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=AttentionSpec(2304)),
            SimpleNamespace(kv_cache_spec=MambaSpec(2304)),
        ]
    )

    assert layout.get_group_tokens_per_block(config, kv_cache) == [9216, 2304]
    assert layout.get_lmcache_scheduler_block_size(config, kv_cache) == 9216
    identity = layout.get_lmcache_model_name(config)
    assert identity.endswith("##lmcache-dcp-layout-v1-d4-interleave4")
    assert layout.get_lmcache_base_model_name(identity) == config.model_config.model
    layout.validate_dcp_support(config, n_servers=1, kv_cache_config=kv_cache)
    layout.validate_mamba_step_alignment(config, kv_cache)


def test_dcp_layout_identity_includes_noninterleaved_dcp() -> None:
    layout = _load_layout()
    config = _config()
    config.parallel_config.cp_kv_cache_interleave_size = 1

    assert layout.get_lmcache_model_name(config).endswith(
        "##lmcache-dcp-layout-v1-d4-interleave1"
    )


def test_dcp_layout_fails_closed_on_partial_shard_ownership() -> None:
    layout = _load_layout()
    config = _config()
    config.parallel_config.world_size = 8
    with pytest.raises(ValueError, match="shards"):
        layout.validate_dcp_support(config, n_servers=4)

    config.parallel_config.world_size = 12
    with pytest.raises(ValueError, match="whole-number set"):
        layout.validate_dcp_support(config, n_servers=2)


def test_registration_preserves_authoritative_engine_group_spans(
    monkeypatch,
) -> None:
    convert = _load_group_converter(monkeypatch)
    groups = [
        SimpleNamespace(
            layer_names=[f"layer{index}"],
            kv_cache_spec=SimpleNamespace(block_size=2304),
        )
        for index in range(4)
    ]
    caches = {
        f"layer{index}": SimpleNamespace(kernel=f"kernel{index}") for index in range(4)
    }
    infos = convert(
        SimpleNamespace(kv_cache_groups=groups),
        caches,
        group_tokens_per_block=[2304, 2304, 2304, 9216],
    )

    assert [info.engine_group_id for info in infos] == [0, 1, 2, 3]
    assert [info.tokens_per_block for info in infos] == [
        2304,
        2304,
        2304,
        9216,
    ]
    assert [9216 // info.tokens_per_block for info in infos] == [4, 4, 4, 1]


def test_scheduler_starts_each_heartbeat_once() -> None:
    tree = ast.parse(ADAPTER.read_text())
    adapter_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheMPSchedulerAdapter"
    )
    ensure_started = next(
        node
        for node in adapter_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_ensure_heartbeat_started"
    )
    probe_class = ast.ClassDef(
        name="AdapterProbe",
        bases=[],
        keywords=[],
        body=[ensure_started],
        decorator_list=[],
        type_params=[],
    )
    namespace: dict[str, Any] = {}
    started: list[str] = []

    class Heartbeat:
        def __init__(self, mq_client, health_event, interval):
            self.client = mq_client

        def start(self):
            started.append(self.client)

    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[probe_class], type_ignores=[])),
            str(ADAPTER),
            "exec",
        ),
        {"HeartbeatThread": Heartbeat},
        namespace,
    )
    adapter = namespace["AdapterProbe"]()
    adapter._heartbeats_started = False
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0
    adapter._heartbeats = {}
    adapter._health_events = {"a": object(), "b": object()}
    adapter.mq_clients = {"a": "client-a", "b": "client-b"}

    adapter._ensure_heartbeat_started()
    adapter._ensure_heartbeat_started()

    assert started == ["client-a", "client-b"]


def test_connector_prefix_is_bounded_by_lagging_mamba_state() -> None:
    tree = ast.parse(CORE_KV_CACHE_MANAGER.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager"
    )
    method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_computed_blocks_for_connector"
    )
    probe = ast.ClassDef(
        name="ManagerProbe",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
        type_params=[],
    )

    class HybridKVCacheCoordinator:
        pass

    namespace = {
        "Request": object,
        "KVCacheBlocks": object,
        "HybridKVCacheCoordinator": HybridKVCacheCoordinator,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])),
            str(CORE_KV_CACHE_MANAGER),
            "exec",
        ),
        namespace,
    )
    manager_probe = namespace["ManagerProbe"]()
    coordinator = HybridKVCacheCoordinator()
    coordinator.full_attention_group_id = 0
    coordinator.find_longest_cache_hit_per_group = lambda *_: (
        ("fa", "mamba"),
        [8, 4],
    )
    manager_probe.coordinator = coordinator
    manager_probe.kv_cache_config = SimpleNamespace(has_mamba_layers=True)
    manager_probe.prefix_cache_lookup_enabled = lambda _request: True
    manager_probe.create_kv_cache_blocks = lambda computed: ("unsafe", computed)
    manager_probe.get_computed_blocks = lambda _request: ("reconciled", 4, 0)
    request = SimpleNamespace(block_hashes=["h"], num_tokens=9)

    assert manager_probe.get_computed_blocks_for_connector(request) == (
        "reconciled",
        4,
        0,
        True,
    )


def test_mamba_cache_blocks_emits_every_exact_committed_boundary() -> None:
    tree = ast.parse(CORE_SINGLE_TYPE_MANAGER.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MambaManager"
    )
    cache_blocks = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "cache_blocks"
    )
    source = ast.unparse(cache_blocks)

    assert "_pending_partial_tail_offloads.append" in source
    assert "(idx + 1) * self.block_size" in source
    assert "block.is_null or block.block_hash is None" in source


def test_tracker_initializes_exact_mamba_boundary_map() -> None:
    tree = ast.parse(METADATA.read_text())
    tracker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheMPRequestTracker"
    )
    constructor = next(
        node
        for node in tracker.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    assert "self.exact_mamba_boundary_blocks = {}" in ast.unparse(constructor)


def test_exact_mamba_boundaries_replace_only_mamba_source_ids() -> None:
    apply = _load_exact_mamba_store_helper()
    tracker = SimpleNamespace(
        exact_mamba_boundary_blocks={0: {9216: 104, 18432: 108}}
    )
    block_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]

    apply(
        tracker,
        block_ids,
        group_tokens_per_block=[2304, 9216],
        mamba_group_ids={0},
        lmcache_tokens_per_chunk=9216,
        start_token_idx=0,
        end_token_idx=18432,
    )
    assert block_ids == [[104, 108], [9, 10]]


def test_unavailable_mamba_boundaries_use_sparse_null_placeholders() -> None:
    apply = _load_exact_mamba_store_helper()
    tracker = SimpleNamespace(
        exact_mamba_boundary_blocks={0: {18432: 108}},
    )
    block_ids = [[1] * 8, [9, 10]]

    apply(
        tracker,
        block_ids,
        group_tokens_per_block=[2304, 9216],
        mamba_group_ids={0},
        lmcache_tokens_per_chunk=9216,
        start_token_idx=0,
        end_token_idx=18432,
    )
    assert block_ids == [[0, 108], [9, 10]]


def test_connector_ingests_core_mamba_handoffs_before_stores() -> None:
    tree = ast.parse(CONNECTOR.read_text())
    connector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheMPConnector"
    )
    build_meta = next(
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_connector_meta"
    )
    process_new = next(
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_process_new_requests"
    )
    build_source = ast.unparse(build_meta)
    process_source = ast.unparse(process_new)

    assert build_source.index("_ingest_exact_mamba_boundary_blocks") < (
        build_source.index("_process_new_requests")
    )
    assert "self._mamba_group_ids" in process_source


def test_cuda_slot_stride_rejects_unaligned_xwords_before_conversion() -> None:
    patch = CUDA_SLOT_STRIDE_PATCH.read_text()

    check = "TORCH_CHECK(block_stride_elems % elements_per_xword == 0"
    conversion = "block_stride_elems / elements_per_xword"
    assert patch.index(check) < patch.index(conversion)


def test_group_edits_unwrap_uniform_specs_per_layer() -> None:
    tree = ast.parse(GROUP_EDITS.read_text())
    apply_edits = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_kv_cache_group_edits"
    )

    class UniformTypeKVCacheSpecs:
        def __init__(self, specs):
            self.kv_cache_specs = specs

    layer_spec = object()
    seen_specs = []

    class Edit:
        name = "probe"

        def matches(self, spec, _cache):
            seen_specs.append(spec)
            return True

        def apply(self, spec, cache, _layout_hints):
            assert spec is layer_spec
            return f"edited-{cache}"

    namespace = {
        "Counter": __import__("collections").Counter,
        "KVCacheConfig": object,
        "KVCacheSpec": object,
        "LayoutHints": dict,
        "Mapping": dict,
        "RegisteredKVCache": object,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "_EDITS": (Edit(),),
        "logger": SimpleNamespace(info=lambda *_args: None),
        "validate_kv_cache_groups": lambda _config: None,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[apply_edits], type_ignores=[])
            ),
            str(GROUP_EDITS),
            "exec",
        ),
        namespace,
    )
    config = SimpleNamespace(
        has_mamba_layers=True,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs({"layer": layer_spec}),
                layer_names=["layer"],
            )
        ],
    )

    assert namespace["apply_kv_cache_group_edits"](
        config, {"layer": "cache"}, {}
    ) == {"layer": "edited-cache"}
    assert seen_specs == [layer_spec]


def test_subpaged_mla_docstring_describes_three_dimensional_view() -> None:
    tree = ast.parse(GROUP_EDITS.read_text())
    edit = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_SubpagedMLAAttentionViewEdit"
    )
    apply = next(
        node
        for node in edit.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )
    docstring = ast.get_docstring(apply)

    assert docstring is not None
    normalized = " ".join(docstring.split())
    assert "(num_kernel_pages, kernel_block_size, entry_size)" in normalized
    assert "(num_logical_blocks, spec.block_size, -1)" in normalized


def test_exact_mamba_handoffs_skip_unknown_requests() -> None:
    tree = ast.parse(CONNECTOR.read_text())
    connector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheMPConnector"
    )
    ingest = next(
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_ingest_exact_mamba_boundary_blocks"
    )
    namespace = {"SchedulerOutput": object}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[ingest], type_ignores=[])),
            str(CONNECTOR),
            "exec",
        ),
        namespace,
    )
    tracker = SimpleNamespace(exact_mamba_boundary_blocks={})
    probe = SimpleNamespace(
        request_trackers={"known": tracker},
        _mamba_group_ids={1},
    )
    scheduler_output = SimpleNamespace(
        partial_tail_offloads={
            "unknown": [(1, 10, 64)],
            "known": [(1, 11, 128), (2, 12, 128), (1, -1, 192)],
        }
    )

    namespace["_ingest_exact_mamba_boundary_blocks"](probe, scheduler_output)

    assert tracker.exact_mamba_boundary_blocks == {1: {128: 11}}


def test_finished_store_dedup_state_is_pruned_after_engine_converges() -> None:
    tree = ast.parse(ADAPTER.read_text())
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheMPWorkerAdapter"
    )
    methods = [
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_process_finished_stores", "_update_and_get_finished_store"}
    ]
    probe_class = ast.ClassDef(
        name="AdapterProbe",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    namespace: dict[str, Any] = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[probe_class], type_ignores=[])
            ),
            str(ADAPTER),
            "exec",
        ),
        namespace,
    )
    probe = namespace["AdapterProbe"]()
    probe.finished_stores = set()
    probe.previously_finished = set()
    probe.store_futures = {"request": object()}
    probe._returned_finished = set()

    assert probe._process_finished_stores(set(), {"request"}) == set()
    probe.store_futures = {}
    assert probe._process_finished_stores({"request"}, set()) == {"request"}
    assert probe._returned_finished == {"request"}
    assert probe._process_finished_stores(set(), set()) == set()
    assert probe._returned_finished == {"request"}
    assert probe._process_finished_stores(set(), {"request"}) == set()
    assert probe._returned_finished == set()


def test_sink_manager_forwards_zeroing_constructor_argument() -> None:
    tree = ast.parse(CORE_SINGLE_TYPE_MANAGER.read_text())
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SinkFullAttentionManager"
    )
    constructor = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    source = ast.unparse(constructor)

    assert "needs_kv_cache_zeroing: bool=False" in source
    assert "needs_kv_cache_zeroing=needs_kv_cache_zeroing" in source


def test_flashinfer_pins_match_docker_metadata() -> None:
    assert "ARG FLASHINFER_VERSION=0.6.18" in (
        ROOT / "docker/Dockerfile"
    ).read_text()
    assert '"FLASHINFER_VERSION": {\n      "default": "0.6.18"' in (
        ROOT / "docker/versions.json"
    ).read_text()
    cuda_requirements = (ROOT / "requirements/cuda.txt").read_text()
    assert "flashinfer-python==0.6.18" in cuda_requirements
    assert "flashinfer-cubin==0.6.18" in cuda_requirements


def test_slot_stride_patcher_validates_output_before_write(
    monkeypatch, tmp_path
) -> None:
    patcher = _load_module("slot_stride_patcher", SLOT_STRIDE_PATCHER)
    connector = Path("connector.py")
    torch_ops = Path("torch_ops.py")
    originals = {connector: b"connector", torch_ops: b"torch"}
    for relative, content in originals.items():
        (tmp_path / relative).write_bytes(content)

    monkeypatch.setattr(patcher, "CONNECTORS", connector)
    monkeypatch.setattr(patcher, "TORCH_OPS", torch_ops)
    monkeypatch.setattr(
        patcher,
        "EXPECTED_BEFORE",
        {relative: patcher.digest(content) for relative, content in originals.items()},
    )
    monkeypatch.setattr(
        patcher,
        "EXPECTED_AFTER",
        {connector: patcher.digest(b"changed connector"), torch_ops: "wrong"},
    )
    monkeypatch.setattr(
        patcher, "patch_connectors", lambda _text, _path: "changed connector"
    )
    monkeypatch.setattr(
        patcher, "patch_torch_ops", lambda _text, _path: "changed torch"
    )

    with pytest.raises(RuntimeError, match="patched source hash mismatch"):
        patcher.patch(tmp_path)
    assert (tmp_path / connector).read_bytes() == originals[connector]
    assert (tmp_path / torch_ops).read_bytes() == originals[torch_ops]


def test_padded_stride_patcher_validates_output_before_write(
    monkeypatch, tmp_path
) -> None:
    patcher = _load_module("padded_stride_patcher", PADDED_STRIDE_PATCHER)
    relative = Path("utils.py")
    original = b"allowlist\nerror\n"
    path = tmp_path / relative
    path.write_bytes(original)
    monkeypatch.setattr(patcher, "RELATIVE", relative)
    monkeypatch.setattr(patcher, "EXPECTED_BEFORE", patcher.digest(original))
    monkeypatch.setattr(patcher, "EXPECTED_AFTER", "wrong")
    monkeypatch.setattr(patcher, "OLD_ALLOWLIST", "allowlist")
    monkeypatch.setattr(patcher, "NEW_ALLOWLIST", "new allowlist")
    monkeypatch.setattr(patcher, "OLD_ERROR", "error")
    monkeypatch.setattr(patcher, "NEW_ERROR", "new error")

    with pytest.raises(RuntimeError, match="patched source hash mismatch"):
        patcher.patch(tmp_path)
    assert path.read_bytes() == original
