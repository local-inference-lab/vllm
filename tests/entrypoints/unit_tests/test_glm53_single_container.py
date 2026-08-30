# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = ROOT / "docker/glm53-flash/single-container/serve_glm53_with_lmcache.py"
RECIPE = ROOT / "docker/glm53-flash/single-container/README.md"
DOCKERFILE = ROOT / "docker/glm53-flash/single-container/Dockerfile"
VLLM_OVERLAY = (
    ROOT / "docker/glm53-flash/single-container/vllm-startup-overlay/vllm"
)
GPU_WORKER = VLLM_OVERLAY / "v1/worker/gpu_worker.py"
GPU_WARMUP = VLLM_OVERLAY / "v1/worker/gpu/warmup.py"
CUDA_REQUIREMENTS = ROOT / "requirements/cuda.txt"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("glm53_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(
        statement.value, ast.Call
    ):
        return None
    return ast.unparse(statement.value.func)


def _statement_lists(node: ast.AST):
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            statements = [item for item in value if isinstance(item, ast.stmt)]
            if statements:
                yield statements
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _statement_lists(item)
        elif isinstance(value, ast.AST):
            yield from _statement_lists(value)


def test_single_container_defaults_match_qualified_geometry(tmp_path) -> None:
    supervisor = _load_supervisor()
    lmcache = tmp_path / "lmcache"
    vllm = tmp_path / "vllm"
    lmcache.touch(mode=0o755)
    vllm.touch(mode=0o755)
    broker = tmp_path / "broker"
    config = supervisor.load_config(
        {
            "GLM53_LMCACHE_EXECUTABLE": str(lmcache),
            "GLM53_VLLM_EXECUTABLE": str(vllm),
            "GLM53_LMCACHE_CUMEM_BROKER_DIR": str(broker),
        }
    )

    assert config.chunk_size == 9216
    assert config.l1_size_gb == 48
    assert config.max_gpu_workers == 4
    assert supervisor.lmcache_argv(config) == [
        str(lmcache),
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "5555",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        "8080",
        "--l1-size-gb",
        "48",
        "--l1-init-size-gb",
        "20",
        "--l1-align-bytes",
        "16384",
        "--max-gpu-workers",
        "4",
        "--max-cpu-workers",
        "8",
        "--chunk-size",
        "9216",
        "--eviction-trigger-watermark",
        "0.85",
        "--eviction-ratio",
        "0.1",
        "--eviction-policy",
        "LRU",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--separate-object-groups",
    ]
    supervisor.ensure_broker_directory(broker)
    assert stat.S_IMODE(broker.stat().st_mode) == 0o700
    assert supervisor.child_environment(config, {})["LMCACHE_CUMEM_BROKER_DIR"] == str(
        broker
    )


def test_single_container_rejects_unknown_runtime_variable() -> None:
    supervisor = _load_supervisor()
    with pytest.raises(ValueError, match="unknown GLM53"):
        supervisor.load_config({"GLM53_LMCACHE_UNSAFE_OPTION": "1"})


def test_single_container_recipe_preserves_qualified_d16_contract() -> None:
    recipe = RECIPE.read_text()
    requirements = CUDA_REQUIREMENTS.read_text()

    assert "flashinfer-python==0.6.18" in requirements
    assert "flashinfer-cubin==0.6.18" in requirements
    for argument in (
        "--tensor-parallel-size 4",
        "--decode-context-parallel-size 4",
        "--cp-kv-cache-interleave-size 4",
        "--quantization modelopt_mixed",
        "--attention-backend B12X",
        "--linear-backend b12x",
        "--max-model-len 1048576",
        "--max-num-seqs 4",
        "--max-num-batched-tokens 32768",
        '"cudagraph_mode":"FULL"',
        '"num_speculative_tokens":3',
    ):
        assert argument in recipe


def test_docker_installs_startup_fences_into_vllm_package() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert (
        "COPY single-container/vllm-startup-overlay/ "
        "/opt/glm53-vllm-startup-overlay/" in dockerfile
    )
    assert (
        'cp -a /opt/glm53-vllm-startup-overlay/vllm/. "${python_dir}/vllm/";'
        in dockerfile
    )
    assert GPU_WORKER.is_file()
    assert GPU_WARMUP.is_file()


def test_scheduler_warmup_stages_are_rank_fenced() -> None:
    warmup = _function(ast.parse(GPU_WARMUP.read_text()), "warmup_kernels")
    fence = _function(warmup, "_fence")
    assert [_call_name(statement) for statement in fence.body] == [
        "torch.accelerator.synchronize",
        "get_tp_group().barrier",
    ]

    stage_counts = {"worker_execute_model": 0, "worker_sample_tokens": 0}
    for statements in _statement_lists(warmup):
        for index, statement in enumerate(statements):
            stage = _call_name(statement)
            if stage not in stage_counts:
                continue
            stage_counts[stage] += 1
            assert _call_name(statements[index + 1]) == "_fence"
            if stage == "worker_execute_model":
                assert _call_name(statements[index - 1]) == "_fence"

    assert stage_counts == {"worker_execute_model": 3, "worker_sample_tokens": 2}
    direct_calls = [
        _call_name(statement) for statement in warmup.body if _call_name(statement)
    ]
    assert direct_calls[-2:] == [
        "torch.accelerator.synchronize",
        "get_tp_group().barrier",
    ]


def test_kernel_warmup_is_fenced_before_full_graph_capture() -> None:
    method = _function(
        ast.parse(GPU_WORKER.read_text()), "compile_or_warm_up_model"
    )
    calls = sorted(
        (node.lineno, node.col_offset, ast.unparse(node))
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    )
    call_sources = [source for _, _, source in calls]

    warmup = call_sources.index("kernel_warmup(self)")
    sync = call_sources.index("torch.accelerator.synchronize()", warmup)
    barrier = call_sources.index("get_tp_group().barrier()", sync)
    capture = call_sources.index("self.model_runner.capture_model()", barrier)
    assert warmup < sync < barrier < capture


def test_lmcache_failure_prevents_vllm_start(tmp_path) -> None:
    supervisor = _load_supervisor()
    marker = tmp_path / "vllm-started"
    lmcache = tmp_path / "lmcache"
    lmcache.write_text("#!/bin/sh\nexit 7\n")
    lmcache.chmod(0o755)
    vllm = tmp_path / "vllm"
    vllm.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    vllm.chmod(0o755)

    status = supervisor.supervise(
        ["serve", "model"],
        {
            "GLM53_LMCACHE_EXECUTABLE": str(lmcache),
            "GLM53_VLLM_EXECUTABLE": str(vllm),
            "GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker"),
            "GLM53_LMCACHE_STARTUP_TIMEOUT_SECONDS": "1",
            "GLM53_LMCACHE_HEALTH_POLL_SECONDS": "0.01",
            "GLM53_LMCACHE_SHUTDOWN_GRACE_SECONDS": "0.1",
        },
    )

    assert status == 7
    assert not marker.exists()
