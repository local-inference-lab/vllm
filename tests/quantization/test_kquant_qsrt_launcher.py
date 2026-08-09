# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _pid_is_live(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        state = stat_path.read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_qsrt_image_and_launcher_share_immutable_interpreter() -> None:
    repository = Path(__file__).resolve().parents[2]
    dockerfile = (repository / "Dockerfile.fruit-qsrt").read_text()
    launcher = (repository / "serve-glm52-fruit-qsrt.sh").read_text()

    assert "ENV PYTHON_BIN=" not in dockerfile
    assert "/opt/venv/bin/python3 -c" in dockerfile
    assert 'PYTHON_BIN="/opt/venv/bin/python3"' in launcher


@pytest.mark.parametrize(
    "trust",
    [
        {},
        {
            "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256": "a" * 64,
            "FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256": "b" * 64,
        },
    ],
    ids=["unanchored", "mixed"],
)
def test_qsrt_launcher_requires_exactly_one_trust_mode(
    trust: dict[str, str],
) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHON_BIN", None)
    environment.pop("B12X_ROOT", None)
    environment.pop("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", None)
    environment.pop("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", None)
    environment.update(trust)

    result = subprocess.run(
        ["/bin/bash", "-p", str(repository / "serve-glm52-fruit-qsrt.sh")],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == (
        "exactly one of FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 or "
        "FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256 is required\n"
    )


def test_qsrt_launcher_rejects_relative_runtime_evidence_path() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHON_BIN", None)
    environment.pop("B12X_ROOT", None)
    environment.pop("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", None)
    environment.update(
        {
            "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256": "0" * 64,
            "VLLM_KQUANT_RUNTIME_EVIDENCE": "relative/runtime-paths.json",
        }
    )

    result = subprocess.run(
        ["/bin/bash", "-p", str(repository / "serve-glm52-fruit-qsrt.sh")],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "VLLM_KQUANT_RUNTIME_EVIDENCE must be an absolute path\n"


def test_qsrt_launcher_rejects_relative_kld_capture_path() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHON_BIN", None)
    environment.pop("B12X_ROOT", None)
    environment.pop("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", None)
    environment.update(
        {
            "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256": "0" * 64,
            "VLLM_KLD_CAPTURE_DIR": "relative/kld-captures",
        }
    )

    result = subprocess.run(
        ["/bin/bash", "-p", str(repository / "serve-glm52-fruit-qsrt.sh")],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "VLLM_KLD_CAPTURE_DIR must be an absolute path\n"


def test_qsrt_launcher_rejects_python_interpreter_override() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256": "0" * 64,
            "PYTHON_BIN": "/bin/false",
        }
    )

    result = subprocess.run(
        ["/bin/bash", "-p", str(repository / "serve-glm52-fruit-qsrt.sh")],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == (
        "PYTHON_BIN is fixed by the Fruit runtime image and must not be overridden\n"
    )


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repository: Path) -> str:
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fruit Launcher Test",
            "-c",
            "user.email=fruit-launcher@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return _git(repository, "rev-parse", "HEAD")


def _tracked_source_sha256(repository: Path) -> str:
    entries = []
    for line in _git(repository, "ls-files", "--stage").splitlines():
        metadata, relative = line.split("\t", 1)
        mode, _, stage = metadata.split()
        assert stage == "0"
        entries.append((relative.encode(), mode.encode()))
    digest = hashlib.sha256(b"kquant-tracked-worktree-sha256-v1\0")
    for relative, mode in sorted(entries):
        content = (repository / relative.decode()).read_bytes()
        digest.update(mode)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _publication_bytes(
    *,
    vllm_revision: str,
    b12x_revision: str,
    vllm_source_sha256: str,
    b12x_source_sha256: str,
) -> tuple[bytes, bytes]:
    manifest = {
        "producer": {
            "encoder": {"kquant_revision": "5efc5fb924be67367279c79b2708c3b9465ecb58"},
            "runtime": {
                "vllm_revision": vllm_revision,
                "b12x_revision": b12x_revision,
                "vllm_source_sha256": vllm_source_sha256,
                "b12x_source_sha256": b12x_source_sha256,
            },
        }
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    marker_bytes = json.dumps(
        {"package_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return manifest_bytes, marker_bytes


@pytest.fixture(scope="module", params=["complete", "candidate"])
def secured_launcher_run(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    trust_mode = str(request.param)
    root = tmp_path_factory.mktemp(f"qsrt-launcher-{trust_mode}")
    b12x_repository = root / "b12x-fruit"
    _write(b12x_repository / "b12x/__init__.py", "SOURCE_VALUE = 'verified-b12x'\n")
    _write(b12x_repository / "b12x/moe/__init__.py", "")
    _write(b12x_repository / "b12x/moe/fused_moe.py", "")
    _write(b12x_repository / "b12x/attention/__init__.py", "")
    _write(b12x_repository / "b12x/attention/sparse_mla.py", "")
    b12x_revision = _commit(b12x_repository)

    vllm_repository = root / "vllm-fruit"
    python_bin = vllm_repository / ".venv/bin/python"
    cache_parent = root / "cache"
    cache_parent.mkdir()
    runtime_manifest = root / "MANIFEST.sha256"
    launcher = (
        Path(__file__).resolve().parents[2] / "serve-glm52-fruit-qsrt.sh"
    ).read_text()
    launcher = launcher.replace(
        'process_group_alive() {\n  if [[ -z "${server_pgid}" ]]; then',
        "process_group_alive() {\n"
        '  if [[ "${TEST_STICKY_PROCESS_GROUP:-0}" == "1" '
        '&& -n "${server_pgid}" ]]; then\n'
        "    return 0\n"
        "  fi\n"
        '  if [[ -z "${server_pgid}" ]]; then',
    )
    assert "${TEST_STICKY_PROCESS_GROUP:-0}" in launcher
    launcher = launcher.replace("{1..50}", "{1..2}")
    launcher = launcher.replace("{1..200}", "{1..2}")
    launcher = launcher.replace("/usr/bin/sleep 0.1", "/usr/bin/sleep 0.001")
    launcher = launcher.replace("/usr/bin/sleep 0.05", "/usr/bin/sleep 0.001")
    launcher = re.sub(
        r'EXPECTED_B12X_REVISION="[0-9a-f]{40}"',
        f'EXPECTED_B12X_REVISION="{b12x_revision}"',
        launcher,
    )
    launcher = launcher.replace(
        'PYTHON_BIN="/opt/venv/bin/python3"',
        f'PYTHON_BIN="{python_bin}"',
    )
    launcher = launcher.replace(
        'B12X_ROOT="/opt/b12x-fruit"',
        f'B12X_ROOT="{b12x_repository}"',
    )
    launcher = launcher.replace(
        "/usr/bin/mktemp -d -p /cache fruit-qsrt.XXXXXXXXXX",
        f'/usr/bin/mktemp -d -p "{cache_parent}" fruit-qsrt.XXXXXXXXXX',
    )
    launcher = launcher.replace(
        'Path("/opt/fruit-runtime/MANIFEST.sha256")',
        f'Path("{runtime_manifest}")',
    )
    launcher = launcher.replace(
        '  "${trust_name}=${QSRT_EXPECTED_MARKER_SHA256}"\n)',
        """  "${trust_name}=${QSRT_EXPECTED_MARKER_SHA256}"
  "TEST_B12X_REVISION=${TEST_B12X_REVISION}"
  "TEST_B12X_SOURCE_SHA256=${TEST_B12X_SOURCE_SHA256}"
  "TEST_DESCENDANT_PID=${TEST_DESCENDANT_PID}"
  "TEST_KQUANT_REVISION=${TEST_KQUANT_REVISION}"
  "TEST_MUTABLE_B12X_ROOT=${TEST_MUTABLE_B12X_ROOT}"
  "TEST_MUTABLE_VLLM_ROOT=${TEST_MUTABLE_VLLM_ROOT}"
  "TEST_REPORT=${TEST_REPORT}"
  "TEST_PUBLICATION_ARGS=${TEST_PUBLICATION_ARGS}"
  "TEST_STICKY_PROCESS_GROUP=${TEST_STICKY_PROCESS_GROUP}"
  "TEST_VLLM_REVISION=${TEST_VLLM_REVISION}"
  "TEST_VLLM_SOURCE_SHA256=${TEST_VLLM_SOURCE_SHA256}"
)""",
    )
    _write(vllm_repository / "serve-glm52-fruit-qsrt.sh", launcher)
    (vllm_repository / "serve-glm52-fruit-qsrt.sh").chmod(0o755)
    _write(vllm_repository / ".gitignore", ".venv/\n*.so\n")
    _write(vllm_repository / "vllm/__init__.py", "SOURCE_VALUE = 'verified-vllm'\n")
    _write(
        vllm_repository / "vllm/compiled_runtime.so",
        "verified-extension\n",
    )
    _write(vllm_repository / "vllm/entrypoints/__init__.py", "")
    _write(vllm_repository / "vllm/entrypoints/cli/__init__.py", "")
    _write(
        vllm_repository / "vllm/entrypoints/cli/main.py",
        """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import b12x
import vllm

report = {
    "compiled_runtime": (
        Path(vllm.__file__).parent / "compiled_runtime.so"
    ).read_text(),
    "compiled_runtime_path": str(
        Path(vllm.__file__).parent / "compiled_runtime.so"
    ),
    "b12x_file": b12x.__file__,
    "b12x_value": b12x.SOURCE_VALUE,
    "fixed": {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "VLLM_USE_B12X_MOE",
        )
    },
    "kld_capture_dir": os.environ.get("VLLM_KLD_CAPTURE_DIR"),
    "hostile": {
        name: os.environ.get(name)
        for name in (
            "B12X_FAKE_TOGGLE",
            "TORCH_COMPILE_DISABLE",
            "VLLM_USE_V1",
        )
    },
    "vllm_file": vllm.__file__,
    "vllm_value": vllm.SOURCE_VALUE,
}
Path(os.environ["TEST_REPORT"]).write_text(json.dumps(report))
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import os, signal, time;"
        "signal.signal(signal.SIGTERM, lambda *_: None);"
        "open(os.environ['TEST_DESCENDANT_PID'], 'w').write(str(os.getpid()));"
        "time.sleep(60)",
    ],
    env=os.environ.copy(),
)
for _ in range(500):
    if Path(os.environ["TEST_DESCENDANT_PID"]).exists():
        break
    if child.poll() is not None:
        raise RuntimeError("test descendant exited before recording its PID")
    time.sleep(0.01)
else:
    raise RuntimeError("test descendant did not start")
""".lstrip(),
    )
    _write(
        vllm_repository
        / "vllm/model_executor/layers/quantization/kquant_qsrt_publication.py",
        """
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
expected_digest = sys.argv[3]
candidate_arguments = sys.argv[4:]
if candidate_arguments not in ([], ["--candidate-mode"]):
    raise RuntimeError("fixture received invalid publication arguments")
candidate_mode = candidate_arguments == ["--candidate-mode"]
Path(os.environ["TEST_PUBLICATION_ARGS"]).write_text(
    json.dumps(candidate_arguments)
)
shutil.copytree(source, destination)
manifest = {
    "producer": {
        "encoder": {"kquant_revision": os.environ["TEST_KQUANT_REVISION"]},
        "runtime": {
            "vllm_revision": os.environ["TEST_VLLM_REVISION"],
            "b12x_revision": os.environ["TEST_B12X_REVISION"],
            "vllm_source_sha256": os.environ["TEST_VLLM_SOURCE_SHA256"],
            "b12x_source_sha256": os.environ["TEST_B12X_SOURCE_SHA256"],
        },
    }
}
manifest_bytes = json.dumps(
    manifest, sort_keys=True, separators=(",", ":")
).encode()
marker_bytes = json.dumps(
    {"package_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
    sort_keys=True,
    separators=(",", ":"),
).encode()
if hashlib.sha256(marker_bytes).hexdigest() != expected_digest:
    raise RuntimeError("fixture publication anchor mismatch")
(destination / "qsrt-manifest.json").write_bytes(manifest_bytes)
marker_name = "QSRT_CANDIDATE.json" if candidate_mode else "QSRT_COMPLETE.json"
(destination / marker_name).write_bytes(marker_bytes)
(Path(os.environ["TEST_MUTABLE_VLLM_ROOT"]) / "vllm/__init__.py").write_text(
    "SOURCE_VALUE = 'mutated-vllm'\\n"
)
(Path(os.environ["TEST_MUTABLE_VLLM_ROOT"]) / "vllm/compiled_runtime.so").write_text(
    "mutated-extension\\n"
)
(Path(os.environ["TEST_MUTABLE_B12X_ROOT"]) / "b12x/__init__.py").write_text(
    "SOURCE_VALUE = 'mutated-b12x'\\n"
)
""".lstrip(),
    )
    vllm_revision = _commit(vllm_repository)
    vllm_source_sha256 = _tracked_source_sha256(vllm_repository)
    b12x_source_sha256 = _tracked_source_sha256(b12x_repository)

    manifest_lines = []
    for namespace, package_root in (
        ("vllm", vllm_repository / "vllm"),
        ("b12x", b12x_repository / "b12x"),
    ):
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                name = f"{namespace}/{path.relative_to(package_root).as_posix()}"
                manifest_lines.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}"
                )
    runtime_manifest.write_text("\n".join(manifest_lines) + "\n")

    _, marker_bytes = _publication_bytes(
        vllm_revision=vllm_revision,
        b12x_revision=b12x_revision,
        vllm_source_sha256=vllm_source_sha256,
        b12x_source_sha256=b12x_source_sha256,
    )
    model_source = root / "model-source"
    _write(model_source / "weights.bin", "fixture")
    report = root / "server-report.json"
    capture_dir = root / "kld-captures"
    capture_dir.mkdir()
    descendant_pid = root / "descendant.pid"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to(Path(sys.executable).resolve())
    _write(root / "hostile-bash-env", "exit 92\n")
    hostile_git = root / "hostile-bin/git"
    _write(hostile_git, "#!/bin/sh\nexit 93\n")
    hostile_git.chmod(0o755)
    hostile_home = root / "hostile-home"
    fsmonitor = root / "hostile-fsmonitor"
    _write(fsmonitor, "#!/bin/sh\nexit 94\n")
    fsmonitor.chmod(0o755)
    _write(
        hostile_home / ".gitconfig",
        f"[core]\n\tfsmonitor = {fsmonitor}\n"
        f"[include]\n\tpath = {root / 'hostile-include'}\n",
    )
    environment = os.environ.copy()
    environment.pop("PYTHON_BIN", None)
    environment.pop("B12X_ROOT", None)
    environment.pop("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256", None)
    environment.pop("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256", None)
    environment.pop("VLLM_KLD_CAPTURE_DIR", None)
    environment.update(
        {
            (
                "FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256"
                if trust_mode == "candidate"
                else "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256"
            ): hashlib.sha256(marker_bytes).hexdigest(),
            "VLLM_KLD_CAPTURE_DIR": str(capture_dir),
            "MODEL": str(model_source),
            "TEST_B12X_REVISION": b12x_revision,
            "TEST_B12X_SOURCE_SHA256": b12x_source_sha256,
            "TEST_DESCENDANT_PID": str(descendant_pid),
            "TEST_KQUANT_REVISION": "5efc5fb924be67367279c79b2708c3b9465ecb58",
            "TEST_MUTABLE_B12X_ROOT": str(b12x_repository),
            "TEST_MUTABLE_VLLM_ROOT": str(vllm_repository),
            "TEST_REPORT": str(report),
            "TEST_PUBLICATION_ARGS": str(root / "publication-arguments.json"),
            "TEST_STICKY_PROCESS_GROUP": "1",
            "TEST_VLLM_REVISION": vllm_revision,
            "TEST_VLLM_SOURCE_SHA256": vllm_source_sha256,
            "B12X_FAKE_TOGGLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "VLLM_USE_V1": "1",
            "BASH_ENV": str(root / "hostile-bash-env"),
            "BASH_FUNC_git%%": "() { exit 91; }",
            "PATH": str(root / "hostile-bin"),
            "HOME": str(hostile_home),
        }
    )
    result = subprocess.run(
        ["/bin/bash", "-p", str(vllm_repository / "serve-glm52-fruit-qsrt.sh")],
        cwd=vllm_repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "capture_dir": capture_dir,
        "b12x_repository": b12x_repository,
        "cache_parent": cache_parent,
        "descendant_pid": descendant_pid,
        "report": report,
        "result": result,
        "publication_arguments": root / "publication-arguments.json",
        "trust_mode": trust_mode,
        "vllm_repository": vllm_repository,
    }


def test_qsrt_launcher_uses_private_runtime_snapshot_after_live_tree_mutation(
    secured_launcher_run: dict[str, object],
) -> None:
    result = secured_launcher_run["result"]
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0, result.stderr
    report_path = secured_launcher_run["report"]
    assert isinstance(report_path, Path)
    report = json.loads(report_path.read_text())
    assert report["vllm_value"] == "verified-vllm"
    assert report["b12x_value"] == "verified-b12x"
    assert report["compiled_runtime"] == "verified-extension\n"
    assert report["fixed"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "VLLM_USE_B12X_MOE": "1",
    }
    capture_dir = secured_launcher_run["capture_dir"]
    assert isinstance(capture_dir, Path)
    assert report["kld_capture_dir"] == str(capture_dir.resolve())
    assert report["hostile"] == {
        "B12X_FAKE_TOGGLE": None,
        "TORCH_COMPILE_DISABLE": None,
        "VLLM_USE_V1": None,
    }
    vllm_repository = secured_launcher_run["vllm_repository"]
    b12x_repository = secured_launcher_run["b12x_repository"]
    assert isinstance(vllm_repository, Path)
    assert isinstance(b12x_repository, Path)
    assert not Path(report["vllm_file"]).is_relative_to(vllm_repository)
    assert not Path(report["b12x_file"]).is_relative_to(b12x_repository)
    assert not Path(report["compiled_runtime_path"]).is_relative_to(vllm_repository)
    assert "mutated-vllm" in (vllm_repository / "vllm/__init__.py").read_text()
    assert "mutated-b12x" in (b12x_repository / "b12x/__init__.py").read_text()


def test_qsrt_launcher_selects_candidate_snapshot_only_for_candidate_trust(
    secured_launcher_run: dict[str, object],
) -> None:
    arguments_path = secured_launcher_run["publication_arguments"]
    assert isinstance(arguments_path, Path)
    arguments = json.loads(arguments_path.read_text())
    expected = (
        ["--candidate-mode"]
        if secured_launcher_run["trust_mode"] == "candidate"
        else []
    )
    assert arguments == expected


def test_qsrt_launcher_kills_process_group_descendants_before_return(
    secured_launcher_run: dict[str, object],
) -> None:
    result = secured_launcher_run["result"]
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0, result.stderr
    pid_path = secured_launcher_run["descendant_pid"]
    assert isinstance(pid_path, Path)
    descendant_pid = int(pid_path.read_text())
    for _ in range(100):
        if not _pid_is_live(descendant_pid):
            break
        time.sleep(0.01)
    else:
        pytest.fail(
            f"process-group descendant {descendant_pid} survived launcher return"
        )


def test_qsrt_launcher_bounds_post_sigkill_process_group_wait(
    secured_launcher_run: dict[str, object],
) -> None:
    result = secured_launcher_run["result"]
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0, result.stderr
    assert "did not reap after SIGKILL" in result.stderr
    cache_parent = secured_launcher_run["cache_parent"]
    assert isinstance(cache_parent, Path)
    assert not tuple(cache_parent.iterdir())
