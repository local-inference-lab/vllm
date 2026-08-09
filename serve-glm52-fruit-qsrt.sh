#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$(pwd -P)"
MODEL="${MODEL:-/model}"
if [[ "${MODEL}" != /* ]]; then
  MODEL="${CALLER_DIR}/${MODEL}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export VLLM_PLUGINS=""

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
B12X_ROOT="${B12X_ROOT:-${SCRIPT_DIR}/../b12x-fruit}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
if [[ -z "${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256:-}" ]]; then
  echo "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 is required" >&2
  exit 1
fi
if [[ ! "${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 1
fi
export FRUIT_QSRT_EXPECTED_COMPLETE_SHA256
if (($#)); then
  echo "Fruit QSRT launcher accepts no additional vLLM arguments" >&2
  exit 1
fi
if [[ "${MAX_NUM_SEQS}" != "1" ]]; then
  echo "Fruit QSRT has only been qualified at MAX_NUM_SEQS=1" >&2
  exit 1
fi
if [[ "${TENSOR_PARALLEL_SIZE}" != "1" ]]; then
  echo "Fruit QSRT has only been qualified at tensor parallel size 1" >&2
  exit 1
fi
if [[ "${MAX_MODEL_LEN}" != "4096" ]]; then
  echo "Fruit QSRT has only been qualified at MAX_MODEL_LEN=4096" >&2
  exit 1
fi
if [[ "${MAX_NUM_BATCHED_TOKENS}" != "4096" ]]; then
  echo "Fruit QSRT has only been qualified at MAX_NUM_BATCHED_TOKENS=4096" >&2
  exit 1
fi

EXPECTED_B12X_REVISION="89876a54b2e61bc41d844cc9ca5af040c9ad2f07"
EXPECTED_KQUANT_REVISION="b79ab03c4423d791a8d67a3e7eb1d94d2e46fc82"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing vLLM Python: ${PYTHON_BIN}" >&2
  exit 1
fi
# Preserve the venv launcher basename: resolving its final symlink bypasses
# pyvenv.cfg discovery and silently switches to the system site-packages.
python_dir="$(readlink -e -- "$(dirname -- "${PYTHON_BIN}")")" || {
  echo "Unable to resolve vLLM Python directory: ${PYTHON_BIN}" >&2
  exit 1
}
PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
resolved_python="$(readlink -f -- "${PYTHON_BIN}")" || {
  echo "Unable to resolve vLLM Python: ${PYTHON_BIN}" >&2
  exit 1
}
if [[ ! -f "${resolved_python}" || ! -x "${resolved_python}" ]]; then
  echo "vLLM Python is not an executable file: ${resolved_python}" >&2
  exit 1
fi

if [[ -L "${B12X_ROOT}" ]]; then
  echo "Fruit B12X root must not be a symbolic link: ${B12X_ROOT}" >&2
  exit 1
fi
resolved_b12x="$(readlink -e -- "${B12X_ROOT}")" || {
  echo "Unable to resolve the Fruit B12X root: ${B12X_ROOT}" >&2
  exit 1
}
if [[ ! -d "${resolved_b12x}/b12x" ]]; then
  echo "Missing Fruit B12X checkout: ${resolved_b12x}" >&2
  exit 1
fi
B12X_ROOT="${resolved_b12x}"

if [[ -L "${MODEL}" ]]; then
  echo "MODEL must not be a symbolic link: ${MODEL}" >&2
  exit 1
fi
resolved_model="$(readlink -e -- "${MODEL}")" || {
  echo "Unable to resolve the Fruit model: ${MODEL}" >&2
  exit 1
}
if [[ ! -d "${resolved_model}" ]]; then
  echo "Fruit model is not a directory: ${resolved_model}" >&2
  exit 1
fi
MODEL="${resolved_model}"

actual_b12x_revision="$(git -C "${B12X_ROOT}" rev-parse HEAD)" || {
  echo "Unable to resolve the Fruit B12X revision: ${B12X_ROOT}" >&2
  exit 1
}
if [[ "${actual_b12x_revision}" != "${EXPECTED_B12X_REVISION}" ]]; then
  echo "Fruit B12X revision mismatch: got ${actual_b12x_revision}, expected ${EXPECTED_B12X_REVISION}" >&2
  exit 1
fi
if [[ -n "$(git -C "${B12X_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Fruit B12X checkout has uncommitted files: ${B12X_ROOT}" >&2
  exit 1
fi
actual_vllm_revision="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)" || {
  echo "Unable to resolve the Fruit vLLM revision: ${SCRIPT_DIR}" >&2
  exit 1
}
if [[ -n "$(git -C "${SCRIPT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Fruit vLLM checkout has uncommitted files: ${SCRIPT_DIR}" >&2
  exit 1
fi

cache_id="fruit-qsrt-${actual_vllm_revision:0:12}-${actual_b12x_revision:0:12}"
cache_root="/cache/${cache_id}"
export LOCAL_INFERENCE_CACHE_FINGERPRINT="${cache_id}"
export XDG_CACHE_HOME="${cache_root}"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
export VLLM_CACHE_DIR="${cache_root}/vllm"
export TRITON_CACHE_DIR="${cache_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${cache_root}/flashinfer-autotune"
export TVM_FFI_CACHE_DIR="${cache_root}/tvm-ffi"
export TVM_CACHE_DIR="${cache_root}/tvm"
export TILELANG_CACHE_DIR="${cache_root}/tilelang"
export TILELANG_TMP_DIR="${cache_root}/tilelang/tmp"
export CUTE_DSL_CACHE_DIR="${cache_root}/cute-dsl"
export B12X_CUTE_COMPILE_CACHE_DIR="${cache_root}/b12x-cute"
export B12X_COMPILE_CACHE_DIR="${cache_root}/b12x/compile"
export SPARKINFER_COMPILE_CACHE_DIR="${cache_root}/b12x/compile"
export DG_JIT_CACHE_DIR="${cache_root}/deep-gemm"
export MM_SPARSE_ATTN_AOT_CACHE="${cache_root}/minfer/mm-sparse-attn"
export MINFER_FMHA_CACHE_DIR="${cache_root}/minfer/fmha"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export CUPY_CACHE_DIR="${cache_root}/cupy"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export VLLM_EXL3_ONLINE_CACHE_DIR="${cache_root}/exl3-online"

"${PYTHON_BIN}" -I -S -c '
import hashlib
import json
import os
import subprocess
import stat
import sys
from pathlib import Path, PurePosixPath

RUNTIME_MANIFEST = Path("/opt/fruit-runtime/MANIFEST.sha256")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_stable_regular(path, max_bytes):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise RuntimeError(f"runtime input is not a linked regular file: {path}")
        if before.st_size > max_bytes:
            raise RuntimeError(f"runtime input exceeds its size bound: {path}")
        chunks = []
        bytes_read = 0
        while chunk := os.read(descriptor, min(1 << 20, max_bytes + 1 - bytes_read)):
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise RuntimeError(f"runtime input exceeds its size bound: {path}")
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        bytes_read != before.st_size
        or source_stat_identity(before) != source_stat_identity(after)
        or source_stat_identity(after) != source_stat_identity(path_after)
    ):
        raise RuntimeError(f"runtime input changed while reading: {path}")
    return b"".join(chunks)


def verify_runtime_manifest(vllm_root, b12x_root):
    if RUNTIME_MANIFEST.is_symlink() or not RUNTIME_MANIFEST.is_file():
        raise RuntimeError("Fruit runtime manifest is missing or symbolic")
    expected = {}
    for line in RUNTIME_MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise RuntimeError("Fruit runtime manifest entry is malformed")
        digest, name = fields
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or name in expected
            or relative.parts[0] not in {"vllm", "b12x"}
        ):
            raise RuntimeError("Fruit runtime manifest entry is invalid")
        expected[name] = digest
    actual = {}
    for namespace, root in (
        ("vllm", vllm_root / "vllm"),
        ("b12x", b12x_root / "b12x"),
    ):
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"Fruit runtime package path is symbolic: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(f"Fruit runtime package path is not regular: {path}")
            name = f"{namespace}/{path.relative_to(root).as_posix()}"
            actual[name] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"Fruit runtime inventory mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for name, path in actual.items():
        if sha256(path) != expected[name]:
            raise RuntimeError(f"Fruit runtime file digest mismatch: {name}")

def git_source_output(root, *args):
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect runtime source tree: {root}") from exc


def nul_records(output, description):
    if not output or output[-1:] != b"\0":
        raise RuntimeError(f"runtime source has malformed {description}")
    records = output.split(b"\0")
    records.pop()
    if not records or any(not record for record in records):
        raise RuntimeError(f"runtime source has malformed {description}")
    return records


def tracked_source_index(root):
    index = {}
    for record in nul_records(
        git_source_output(root, "ls-files", "--stage", "-v", "-z"),
        "Git index",
    ):
        if len(record) < 3 or record[1:2] != b" " or b"\t" not in record:
            raise RuntimeError("runtime source has malformed Git index")
        tag = record[:1]
        if tag == b"h":
            raise RuntimeError("runtime source has assume-unchanged files")
        if tag in (b"S", b"s"):
            raise RuntimeError("runtime source has skip-worktree files")
        if tag != b"H":
            raise RuntimeError("runtime source has anomalous tracked files")
        metadata, relative = record[2:].split(b"\t", 1)
        fields = metadata.split(b" ")
        if (
            len(fields) != 3
            or fields[2] != b"0"
            or len(fields[1]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[1])
        ):
            raise RuntimeError("runtime source has malformed or unmerged Git index")
        mode, object_id = fields[:2]
        if mode not in (b"100644", b"100755"):
            raise RuntimeError("runtime source tracks a non-regular entry")
        if (
            not relative
            or relative.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in relative.split(b"/"))
            or relative in index
        ):
            raise RuntimeError("runtime source has anomalous tracked paths")
        index[relative] = (mode, object_id)
    return index


def tracked_source_head(root):
    head = {}
    for record in nul_records(
        git_source_output(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        "HEAD tree",
    ):
        if b"\t" not in record:
            raise RuntimeError("runtime source has malformed HEAD tree")
        metadata, relative = record.split(b"\t", 1)
        fields = metadata.split(b" ")
        if (
            len(fields) != 3
            or len(fields[2]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[2])
        ):
            raise RuntimeError("runtime source has malformed HEAD tree")
        mode, object_type, object_id = fields
        if object_type != b"blob" or mode not in (b"100644", b"100755"):
            raise RuntimeError("runtime source tracks a non-regular entry")
        if not relative or relative in head:
            raise RuntimeError("runtime source has anomalous HEAD paths")
        head[relative] = (mode, object_id)
    return head


def source_stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def require_git_status_clean(root):
    if git_source_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise RuntimeError(f"runtime source checkout is not clean: {root}")


def tracked_tree_sha256(root_value):
    root = Path(root_value).resolve(strict=True)
    index = tracked_source_index(root)
    if index != tracked_source_head(root):
        raise RuntimeError("runtime source index does not match HEAD")
    require_git_status_clean(root)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    digest = hashlib.sha256(b"kquant-tracked-worktree-sha256-v1\0")
    root_fd = os.open(root, directory_flags)
    try:
        for relative in sorted(index):
            expected_mode, expected_object_id = index[relative]
            file_fd = os.open(relative, flags, dir_fd=root_fd)
            try:
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError("tracked runtime source is not regular")
                actual_mode = b"100755" if before.st_mode & 0o111 else b"100644"
                if actual_mode != expected_mode:
                    raise RuntimeError("tracked runtime source mode differs from Git")
                digest.update(actual_mode)
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(before.st_size.to_bytes(8, "big"))
                blob_digest = hashlib.sha1(
                    b"blob " + str(before.st_size).encode("ascii") + b"\0"
                )
                bytes_read = 0
                while chunk := os.read(file_fd, 1 << 20):
                    bytes_read += len(chunk)
                    digest.update(chunk)
                    blob_digest.update(chunk)
                after = os.fstat(file_fd)
                path_after = os.stat(
                    relative,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if (
                    bytes_read != before.st_size
                    or source_stat_identity(before) != source_stat_identity(after)
                    or source_stat_identity(after) != source_stat_identity(path_after)
                ):
                    raise RuntimeError("tracked runtime source changed while hashing")
                if blob_digest.hexdigest().encode("ascii") != expected_object_id:
                    raise RuntimeError("tracked runtime source bytes differ from Git")
            finally:
                os.close(file_fd)
    finally:
        os.close(root_fd)
    require_git_status_clean(root)
    return digest.hexdigest()

model = sys.argv[1]
expected_complete_sha256 = sys.argv[2]
expected_kquant = sys.argv[3]
expected_b12x = sys.argv[4]
actual_vllm = sys.argv[5]
b12x_root = Path(sys.argv[6]).resolve(strict=True)
vllm_root = Path(sys.argv[7]).resolve(strict=True)
verify_runtime_manifest(vllm_root, b12x_root)
model_root = Path(model).resolve(strict=True)
marker_bytes = read_stable_regular(model_root / "QSRT_COMPLETE.json", 1 << 20)
if hashlib.sha256(marker_bytes).hexdigest() != expected_complete_sha256:
    raise RuntimeError("Fruit completion marker disagrees with the external anchor")
try:
    marker = json.loads(marker_bytes)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise RuntimeError("Fruit completion marker is not valid JSON") from exc
manifest_bytes = read_stable_regular(model_root / "qsrt-manifest.json", 16 << 20)
if hashlib.sha256(manifest_bytes).hexdigest() != marker.get(
    "package_manifest_sha256"
):
    raise RuntimeError("Fruit package manifest disagrees with the completion marker")
try:
    manifest = json.loads(manifest_bytes)
    producer = manifest["producer"]
    encoder = producer["encoder"]
    runtime = producer["runtime"]
except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
    raise RuntimeError("Fruit package manifest has invalid producer metadata") from exc
if encoder.get("kquant_revision") != expected_kquant:
    raise RuntimeError("Fruit package KQuant revision is unsupported")
if runtime.get("b12x_revision") != expected_b12x:
    raise RuntimeError("Fruit package B12X revision is unsupported")
if runtime.get("vllm_revision") != actual_vllm:
    raise RuntimeError("Fruit package vLLM revision does not match the runtime")
if runtime.get("b12x_source_sha256") != tracked_tree_sha256(b12x_root):
    raise RuntimeError("Fruit package B12X source fingerprint is unsupported")
if runtime.get("vllm_source_sha256") != tracked_tree_sha256(vllm_root):
    raise RuntimeError("Fruit package vLLM source fingerprint is unsupported")
' \
  "${MODEL}" \
  "${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}" \
  "${EXPECTED_KQUANT_REVISION}" \
  "${EXPECTED_B12X_REVISION}" \
  "${actual_vllm_revision}" \
  "${B12X_ROOT}" \
  "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${B12X_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUTE_DSL_ARCH="sm_120a"
export CUDA_DEVICE_MAX_CONNECTIONS="32"
export SAFETENSORS_FAST_GPU="1"
export VLLM_USE_B12X_MOE="1"
export VLLM_USE_B12X_SPARSE_INDEXER="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

"${PYTHON_BIN}" -P -c '
import sys
from pathlib import Path

import b12x
import vllm

b12x_root = Path(sys.argv[1]).resolve(strict=True)
vllm_root = Path(sys.argv[2]).resolve(strict=True)
if not Path(b12x.__file__).resolve().is_relative_to(b12x_root / "b12x"):
    raise RuntimeError("imported B12X does not come from the sealed source root")
if not Path(vllm.__file__).resolve().is_relative_to(vllm_root / "vllm"):
    raise RuntimeError("imported vLLM does not come from the sealed source root")

import b12x.moe.fused_moe
import b12x.attention.sparse_mla
' "${B12X_ROOT}" "${SCRIPT_DIR}"


HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-QSRT-Fruit-Instruct}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

exec "${PYTHON_BIN}" -P -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size 1 \
  --quantization kquant_hybrid \
  --load-format fastsafetensors \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --kv-cache-dtype nvfp4_ds_mla \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,48,64]}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --generation-config vllm
