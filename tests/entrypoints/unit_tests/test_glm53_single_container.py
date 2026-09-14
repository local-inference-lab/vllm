# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = ROOT / "docker/glm53-flash/single-container/serve_glm53_with_lmcache.py"
RECIPE = ROOT / "docker/glm53-flash/single-container/README.md"
DOCKERFILE = ROOT / "docker/glm53-flash/single-container/Dockerfile"
LOGGING_CONFIG = ROOT / "docker/glm53-flash/single-container/glm53-logging.json"
VLLM_OVERLAY = ROOT / "docker/glm53-flash/single-container/vllm-startup-overlay/vllm"
GPU_WORKER = VLLM_OVERLAY / "v1/worker/gpu_worker.py"
GPU_WARMUP = VLLM_OVERLAY / "v1/worker/gpu/warmup.py"
CUDA_REQUIREMENTS = ROOT / "requirements/cuda.txt"
METRICS_SAMPLE = (
    "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="2304",cache_dtype="fp8_e4m3",'
    'enable_prefix_caching="True",gpu_memory_utilization="0.9",'
    'kv_cache_max_concurrency="1.4728683471679688",'
    'kv_cache_size_tokens="1544414",num_gpu_blocks="670",'
    'swap_space_bytes="4294967296"} 1.0\n'
)
KV_POOL_SUMMARY = (
    "KV pool: 1,544,414 tokens | max-context concurrency: 1.47x "
    "| block: 2304 | dtype: fp8_e4m3"
)


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
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
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
    child_env = supervisor.child_environment(config, {})
    assert child_env["LMCACHE_CUMEM_BROKER_DIR"] == str(broker)
    assert child_env["VLLM_LOGGING_CONFIG_PATH"] == supervisor.VLLM_LOGGING_CONFIG
    assert "VLLM_LOGGING_LEVEL" not in child_env
    assert child_env["LMCACHE_LOG_LEVEL"] == "WARNING"


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_vllm_log_level_override_prevents_default_config(tmp_path, level) -> None:
    supervisor = _load_supervisor()
    config = supervisor.load_config(
        {"GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker")}
    )

    child_env = supervisor.child_environment(
        config,
        {
            "VLLM_LOGGING_LEVEL": level,
            "LMCACHE_LOG_LEVEL": "INFO",
        },
    )

    assert child_env["VLLM_LOGGING_LEVEL"] == level
    assert "VLLM_LOGGING_CONFIG_PATH" not in child_env
    assert child_env["LMCACHE_LOG_LEVEL"] == "INFO"


def test_vllm_logging_config_override_is_preserved(tmp_path) -> None:
    supervisor = _load_supervisor()
    config = supervisor.load_config(
        {"GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker")}
    )
    custom_config = "/operator/logging.json"

    child_env = supervisor.child_environment(
        config, {"VLLM_LOGGING_CONFIG_PATH": custom_config}
    )

    assert child_env["VLLM_LOGGING_CONFIG_PATH"] == custom_config
    assert "VLLM_LOGGING_LEVEL" not in child_env
    assert child_env["LMCACHE_LOG_LEVEL"] == "WARNING"


def test_packaged_logging_config_selects_only_periodic_metrics() -> None:
    config = json.loads(LOGGING_CONFIG.read_text())
    loggers = config["loggers"]
    info_loggers = {
        name for name, settings in loggers.items() if settings["level"] == "INFO"
    }

    assert config["handlers"]["vllm"]["level"] == "INFO"
    assert loggers["vllm"] == {
        "handlers": ["vllm"],
        "level": "WARNING",
        "propagate": False,
    }
    assert info_loggers == {
        "vllm.v1.metrics.loggers",
        "vllm.v1.spec_decode.metrics",
    }
    for name in info_loggers:
        assert "handlers" not in loggers[name]
        assert loggers[name]["propagate"] is True


def test_packaged_logging_config_emits_selected_info_once() -> None:
    script = """
import json
import logging
import logging.config
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    logging.config.dictConfig(json.load(config_file))
logging.getLogger("vllm.other").info("hidden-info")
logging.getLogger("vllm.other").warning("general-warning")
logging.getLogger("vllm.v1.metrics.loggers").info("request-metrics")
logging.getLogger("vllm.v1.spec_decode.metrics").info("spec-metrics")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(LOGGING_CONFIG)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "hidden-info" not in result.stderr
    for message in ("general-warning", "request-metrics", "spec-metrics"):
        assert result.stderr.count(message) == 1


def test_single_container_rejects_unknown_runtime_variable() -> None:
    supervisor = _load_supervisor()
    with pytest.raises(ValueError, match="unknown GLM53"):
        supervisor.load_config({"GLM53_LMCACHE_UNSAFE_OPTION": "1"})


def test_single_container_appends_prompt_tokens_details_flag() -> None:
    supervisor = _load_supervisor()

    args = supervisor._with_prompt_tokens_details(["serve", "model", "--port", "8000"])

    assert args == [
        "serve",
        "model",
        "--port",
        "8000",
        "--enable-prompt-tokens-details",
    ]


def test_single_container_keeps_one_prompt_tokens_details_flag() -> None:
    supervisor = _load_supervisor()
    flag = "--enable-prompt-tokens-details"

    args = supervisor._with_prompt_tokens_details(
        ["serve", "model", flag, "--port", "8000", flag]
    )

    assert args == ["serve", "model", flag, "--port", "8000"]


def test_kv_pool_summary_reports_only_capacity_fields() -> None:
    supervisor = _load_supervisor()

    summary = supervisor.kv_pool_summary(METRICS_SAMPLE)

    assert summary == KV_POOL_SUMMARY
    for unrelated in (
        "gpu_memory_utilization",
        "num_gpu_blocks",
        "swap_space_bytes",
        "enable_prefix_caching",
        "0.9",
    ):
        assert unrelated not in summary


def test_kv_pool_summary_falls_back_to_max_model_len() -> None:
    supervisor = _load_supervisor()
    without_concurrency = METRICS_SAMPLE.replace(
        'kv_cache_max_concurrency="1.4728683471679688",', ""
    )

    assert supervisor.kv_pool_summary(without_concurrency) is None
    assert supervisor.kv_pool_summary(without_concurrency, 1048576) == KV_POOL_SUMMARY
    assert (
        supervisor._max_model_len(["serve", "model", "--max-model-len", "1048576"])
        == 1048576
    )
    assert (
        supervisor._max_model_len(["serve", "model", "--max-model-len=1048576"])
        == 1048576
    )
    assert supervisor._max_model_len(["serve", "model"]) is None


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n",
        'vllm:cache_config_info{block_size="2304"} 1.0\n',
        'vllm:cache_config_info{kv_cache_size_tokens="many",block_size="2304",'
        'cache_dtype="fp8_e4m3",kv_cache_max_concurrency="1.47"} 1.0\n',
        'vllm:cache_config_info{kv_cache_size_tokens="1544414",block_size="0",'
        'cache_dtype="fp8_e4m3",kv_cache_max_concurrency="1.47"} 1.0\n',
        'vllm:cache_config_info{kv_cache_size_tokens="1544414",block_size="2304",'
        'cache_dtype="",kv_cache_max_concurrency="1.47"} 1.0\n',
        'vllm:cache_config_info{kv_cache_size_tokens="1544414",block_size="2304",'
        'cache_dtype="fp8_e4m3",kv_cache_max_concurrency="nan"} 1.0\n',
        'vllm:cache_config_info{kv_cache_size_tokens="1544414",block_size="2304",'
        'cache_dtype="fp8_e4m3",kv_cache_max_concurrency="1.47"\n',
        "<html><body>404 not found</body></html>\n",
    ],
)
def test_kv_pool_summary_rejects_malformed_metrics(payload) -> None:
    supervisor = _load_supervisor()

    assert supervisor.kv_pool_summary(payload) is None


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (503, b"unavailable"),
        (200, b"x" * ((1 << 20) + 1)),
        (200, b"\xff"),
    ],
)
def test_metrics_response_failures_return_none(
    tmp_path, monkeypatch, status, body
) -> None:
    supervisor = _load_supervisor()
    config = supervisor.load_config(
        {"GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker")}
    )

    class Response:
        def read(self, amount: int) -> bytes:
            assert amount == supervisor.METRICS_BODY_LIMIT_BYTES + 1
            return body[:amount]

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def request(self, method: str, path: str) -> None:
            assert (method, path) == ("GET", "/metrics")

        def getresponse(self) -> Response:
            response = Response()
            response.status = status
            return response

        def close(self) -> None:
            pass

    monkeypatch.setattr(supervisor.http.client, "HTTPConnection", Connection)

    assert supervisor._fetch_metrics(config, 5.0) is None


def test_unavailable_metrics_warn_once(monkeypatch, capsys, tmp_path) -> None:
    supervisor = _load_supervisor()
    config = supervisor.load_config(
        {"GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker")}
    )
    monkeypatch.setattr(supervisor, "_fetch_metrics", lambda *_args: None)

    supervisor.report_kv_pool(config, ["serve", "model"])

    stderr = capsys.readouterr().err
    assert stderr.count("WARNING KV pool capacity unavailable from /metrics") == 1
    assert "KV pool:" not in stderr


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
        "COPY single-container/glm53-logging.json "
        "/etc/vllm/glm53-logging.json" in dockerfile
    )
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
    method = _function(ast.parse(GPU_WORKER.read_text()), "compile_or_warm_up_model")
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


def test_supervisor_launches_vllm_with_prompt_tokens_details(
    tmp_path, monkeypatch
) -> None:
    supervisor = _load_supervisor()
    lmcache = tmp_path / "lmcache"
    vllm = tmp_path / "vllm"
    lmcache.touch(mode=0o755)
    vllm.touch(mode=0o755)
    launched: list[list[str]] = []

    class Process:
        pid = 1

        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            launched.append(argv)

        def poll(self) -> int | None:
            return 0 if self is vllm_process else None

    def popen(argv: list[str], **kwargs: object) -> Process:
        nonlocal vllm_process
        process = Process(argv, **kwargs)
        if argv[0] == str(vllm):
            vllm_process = process
        return process

    vllm_process: Process | None = None
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "_healthy", lambda *_args: True)
    monkeypatch.setattr(supervisor, "_stop_processes", lambda *_args: None)

    status = supervisor.supervise(
        ["serve", "model", "--port", "8000"],
        {
            "GLM53_LMCACHE_EXECUTABLE": str(lmcache),
            "GLM53_VLLM_EXECUTABLE": str(vllm),
            "GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker"),
        },
    )

    assert status == 0
    assert launched[1] == [
        str(vllm),
        "serve",
        "model",
        "--port",
        "8000",
        "--enable-prompt-tokens-details",
    ]


def test_supervisor_fetches_metrics_once_after_readiness(tmp_path, monkeypatch) -> None:
    supervisor = _load_supervisor()
    lmcache = tmp_path / "lmcache"
    vllm = tmp_path / "vllm"
    lmcache.touch(mode=0o755)
    vllm.touch(mode=0o755)
    events: list[str] = []

    class Process:
        pid = 1

        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            self.is_vllm = argv[0] == str(vllm)

        def poll(self) -> int | None:
            if self.is_vllm and "metrics" in events:
                return 0
            return None

    def readiness(*_args) -> bool:
        events.append("readiness")
        return True

    def fetch_metrics(*_args) -> str:
        events.append("metrics")
        return METRICS_SAMPLE

    monkeypatch.setattr(supervisor.subprocess, "Popen", Process)
    monkeypatch.setattr(supervisor, "_healthy", lambda *_args: True)
    monkeypatch.setattr(supervisor, "_vllm_healthy", readiness)
    monkeypatch.setattr(supervisor, "_fetch_metrics", fetch_metrics)
    monkeypatch.setattr(supervisor, "_stop_processes", lambda *_args: None)

    status = supervisor.supervise(
        ["serve", "model", "--max-model-len=1048576"],
        {
            "GLM53_LMCACHE_EXECUTABLE": str(lmcache),
            "GLM53_VLLM_EXECUTABLE": str(vllm),
            "GLM53_LMCACHE_CUMEM_BROKER_DIR": str(tmp_path / "broker"),
        },
    )

    assert status == 0
    assert events == ["readiness", "metrics"]
