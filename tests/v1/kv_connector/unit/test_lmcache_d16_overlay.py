# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

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
ADAPTER = (
    ROOT / "docker/glm53-flash/lmcache-d16-overlay/overlay/"
    "lmcache/integration/vllm/vllm_multi_process_adapter.py"
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
