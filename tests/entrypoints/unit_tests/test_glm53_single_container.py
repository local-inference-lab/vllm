# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = ROOT / "docker/glm53-flash/single-container/serve_glm53_with_lmcache.py"
RECIPE = ROOT / "docker/glm53-flash/single-container/README.md"
CUDA_REQUIREMENTS = ROOT / "requirements/cuda.txt"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("glm53_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
