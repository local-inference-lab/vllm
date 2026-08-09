#!/bin/bash -p
set -euo pipefail
builtin unset -v BASH_ENV ENV SHELLOPTS CDPATH GLOBIGNORE 2>/dev/null || true
while IFS= read -r inherited_function; do
  builtin unset -f "${inherited_function}"
done < <(builtin compgen -A function)
builtin readonly PATH="/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
builtin readonly LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
builtin export PATH LD_LIBRARY_PATH
CALLER_DIR="$(pwd -P)"
MODEL="${MODEL:-/model}"
if [[ "${MODEL}" != /* ]]; then
  MODEL="${CALLER_DIR}/${MODEL}"
fi

SCRIPT_DIR="$(cd -- "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
unset PYTHONHOME PYTHONPATH

if [[ -n "${PYTHON_BIN:-}" ]]; then
  echo "PYTHON_BIN is fixed by the Fruit runtime image and must not be overridden" >&2
  exit 1
fi
PYTHON_BIN="/opt/venv/bin/python3"
if [[ -n "${B12X_ROOT:-}" ]]; then
  echo "B12X_ROOT is fixed by the Fruit runtime image and must not be overridden" >&2
  exit 1
fi
B12X_ROOT="/opt/b12x-fruit"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
complete_trust_set=0
candidate_trust_set=0
if [[ -v FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 ]]; then
  complete_trust_set=1
fi
if [[ -v FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256 ]]; then
  candidate_trust_set=1
fi
if ((complete_trust_set + candidate_trust_set != 1)); then
  echo "exactly one of FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 or FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256 is required" >&2
  exit 1
fi
if ((candidate_trust_set)); then
  QSRT_TRUST_MODE="candidate"
  QSRT_EXPECTED_MARKER_SHA256="${FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256}"
  trust_name="FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256"
else
  QSRT_TRUST_MODE="complete"
  QSRT_EXPECTED_MARKER_SHA256="${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}"
  trust_name="FRUIT_QSRT_EXPECTED_COMPLETE_SHA256"
fi
if [[ ! "${QSRT_EXPECTED_MARKER_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "${trust_name} must be a lowercase SHA-256 digest" >&2
  exit 1
fi
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
if [[ "${GPU_MEMORY_UTILIZATION}" != "0.80" ]]; then
  echo "Fruit QSRT has only been qualified at GPU_MEMORY_UTILIZATION=0.80" >&2
  exit 1
fi
if [[ -n "${VLLM_KQUANT_RUNTIME_EVIDENCE:-}" ]]; then
  if [[ "${VLLM_KQUANT_RUNTIME_EVIDENCE}" != /* ]]; then
    echo "VLLM_KQUANT_RUNTIME_EVIDENCE must be an absolute path" >&2
    exit 1
  fi
  if [[ -L "${VLLM_KQUANT_RUNTIME_EVIDENCE}" ]]; then
    echo "VLLM_KQUANT_RUNTIME_EVIDENCE must not be a symbolic link" >&2
    exit 1
  fi
  if [[ -e "${VLLM_KQUANT_RUNTIME_EVIDENCE}" && ! -f "${VLLM_KQUANT_RUNTIME_EVIDENCE}" ]]; then
    echo "VLLM_KQUANT_RUNTIME_EVIDENCE must name a regular file" >&2
    exit 1
  fi
  evidence_parent="$(/usr/bin/readlink -e -- "$(/usr/bin/dirname -- "${VLLM_KQUANT_RUNTIME_EVIDENCE}")")" || {
    echo "VLLM_KQUANT_RUNTIME_EVIDENCE parent does not exist" >&2
    exit 1
  }
  if [[ ! -d "${evidence_parent}" || ! -w "${evidence_parent}" ]]; then
    echo "VLLM_KQUANT_RUNTIME_EVIDENCE parent is not writable" >&2
    exit 1
  fi
  VLLM_KQUANT_RUNTIME_EVIDENCE="${evidence_parent}/$(/usr/bin/basename -- "${VLLM_KQUANT_RUNTIME_EVIDENCE}")"
  export VLLM_KQUANT_RUNTIME_EVIDENCE
fi

EXPECTED_B12X_REVISION="56d5a9063e7726d6799c87760e2070c38e479677"
EXPECTED_KQUANT_REVISION="ea07fea0f5e93a0e321c3176bf11b654c645f489"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing vLLM Python: ${PYTHON_BIN}" >&2
  exit 1
fi
# Preserve the venv launcher basename: resolving its final symlink bypasses
# pyvenv.cfg discovery and silently switches to the system site-packages.
python_dir="$(/usr/bin/readlink -e -- "$(/usr/bin/dirname -- "${PYTHON_BIN}")")" || {
  echo "Unable to resolve vLLM Python directory: ${PYTHON_BIN}" >&2
  exit 1
}
PYTHON_BIN="${python_dir}/$(/usr/bin/basename -- "${PYTHON_BIN}")"
resolved_python="$(/usr/bin/readlink -f -- "${PYTHON_BIN}")" || {
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
resolved_b12x="$(/usr/bin/readlink -e -- "${B12X_ROOT}")" || {
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
resolved_model="$(/usr/bin/readlink -e -- "${MODEL}")" || {
  echo "Unable to resolve the Fruit model: ${MODEL}" >&2
  exit 1
}
if [[ ! -d "${resolved_model}" ]]; then
  echo "Fruit model is not a directory: ${resolved_model}" >&2
  exit 1
fi
MODEL="${resolved_model}"

SOURCE_ENV=(
  /usr/bin/env -i
  "GIT_CONFIG_COUNT=0"
  "GIT_CONFIG_GLOBAL=/dev/null"
  "GIT_CONFIG_NOSYSTEM=0"
  "GIT_NO_LAZY_FETCH=1"
  "GIT_NO_REPLACE_OBJECTS=1"
  "GIT_OPTIONAL_LOCKS=0"
  "GIT_TERMINAL_PROMPT=0"
  "HOME=/nonexistent"
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  "PATH=${PATH}"
)
actual_b12x_revision="$("${SOURCE_ENV[@]}" /usr/bin/git -C "${B12X_ROOT}" rev-parse HEAD)" || {
  echo "Unable to resolve the Fruit B12X revision: ${B12X_ROOT}" >&2
  exit 1
}
if [[ "${actual_b12x_revision}" != "${EXPECTED_B12X_REVISION}" ]]; then
  echo "Fruit B12X revision mismatch: got ${actual_b12x_revision}, expected ${EXPECTED_B12X_REVISION}" >&2
  exit 1
fi
if [[ -n "$("${SOURCE_ENV[@]}" /usr/bin/git -C "${B12X_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Fruit B12X checkout has uncommitted files: ${B12X_ROOT}" >&2
  exit 1
fi
actual_vllm_revision="$("${SOURCE_ENV[@]}" /usr/bin/git -C "${SCRIPT_DIR}" rev-parse HEAD)" || {
  echo "Unable to resolve the Fruit vLLM revision: ${SCRIPT_DIR}" >&2
  exit 1
}
if [[ -n "$("${SOURCE_ENV[@]}" /usr/bin/git -C "${SCRIPT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Fruit vLLM checkout has uncommitted files: ${SCRIPT_DIR}" >&2
  exit 1
fi

private_root="$(/usr/bin/mktemp -d -p /cache fruit-qsrt.XXXXXXXXXX)" || {
  echo "Unable to create a private Fruit runtime namespace under /cache" >&2
  exit 1
}
/usr/bin/chmod 0700 "${private_root}"
server_pid=""
server_pgid=""
process_group_alive() {
  if [[ -z "${server_pgid}" ]]; then
    return 1
  fi
  local proc_stat suffix state pgrp
  for proc_stat in /proc/[0-9]*/stat; do
    [[ -r "${proc_stat}" ]] || continue
    suffix="$(<"${proc_stat}")" || continue
    suffix="${suffix##*) }"
    read -r state _ pgrp _ <<<"${suffix}" || continue
    if [[ "${pgrp}" == "${server_pgid}" && "${state}" != "Z" ]]; then
      return 0
    fi
  done
  return 1
}
terminate_server_group() {
  if [[ -z "${server_pgid}" ]]; then
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill -s TERM "${server_pid}" 2>/dev/null || true
      for _ in {1..50}; do
        kill -0 "${server_pid}" 2>/dev/null || break
        /usr/bin/sleep 0.1
      done
      if kill -0 "${server_pid}" 2>/dev/null; then
        kill -s KILL "${server_pid}" 2>/dev/null || true
      fi
      wait "${server_pid}" 2>/dev/null || true
    fi
    server_pid=""
    return
  fi
  if process_group_alive; then
    kill -s TERM -- "-${server_pgid}" 2>/dev/null || true
    for _ in {1..50}; do
      process_group_alive || break
      /usr/bin/sleep 0.1
    done
  fi
  if process_group_alive; then
    kill -s KILL -- "-${server_pgid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]]; then
    wait "${server_pid}" 2>/dev/null || true
  fi
  while process_group_alive; do
    /usr/bin/sleep 0.05
  done
  server_pid=""
  server_pgid=""
}
cleanup_private_root() {
  status=$?
  trap - EXIT HUP INT TERM
  terminate_server_group
  /usr/bin/rm -rf -- "${private_root}"
  exit "${status}"
}
forward_to_server() {
  signal="$1"
  status="$2"
  trap - "${signal}"
  if process_group_alive; then
    kill -s "${signal}" -- "-${server_pgid}" 2>/dev/null || true
  elif [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -s "${signal}" "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup_private_root EXIT
trap 'forward_to_server HUP 129' HUP
trap 'forward_to_server INT 130' INT
trap 'forward_to_server TERM 143' TERM

cache_root="${private_root}/cache"
/usr/bin/mkdir -m 0700 "${cache_root}"
tmp_root="${private_root}/tmp"
/usr/bin/mkdir -m 0700 "${tmp_root}"
export TMPDIR="${tmp_root}"
export TMP="${tmp_root}"
export TEMP="${tmp_root}"
home_root="${private_root}/home"
/usr/bin/mkdir -m 0700 "${home_root}"
export HOME="${home_root}"
export LOCAL_INFERENCE_CACHE_FINGERPRINT="${private_root##*/}"
export XDG_CACHE_HOME="${cache_root}"
export HF_HOME="${cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${cache_root}/huggingface/hub"
export TRANSFORMERS_CACHE="${cache_root}/huggingface/transformers"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
export TORCH_HOME="${cache_root}/torch"
export HF_DATASETS_CACHE="${cache_root}/huggingface/datasets"
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



model_source="${MODEL}"
MODEL="${private_root}/model"
export FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT="${MODEL}"
runtime_root="${private_root}/runtime"
/usr/bin/mkdir -m 0700 "${runtime_root}"
FRUIT_QSRT_RUNTIME_ENVIRONMENT_JSON='{"B12X_COMPILE_CACHE_DIR":"<PRIVATE_ROOT>/cache/b12x/compile","B12X_CUTE_COMPILE_CACHE_DIR":"<PRIVATE_ROOT>/cache/b12x-cute","B12X_ROOT":"<PRIVATE_ROOT>/runtime/b12x-source","CUDA_CACHE_PATH":"<PRIVATE_ROOT>/cache/cuda","CUDA_DEVICE_MAX_CONNECTIONS":"32","CUDA_VISIBLE_DEVICES":"0","CUPY_CACHE_DIR":"<PRIVATE_ROOT>/cache/cupy","CUTE_DSL_ARCH":"sm_120a","CUTE_DSL_CACHE_DIR":"<PRIVATE_ROOT>/cache/cute-dsl","DG_JIT_CACHE_DIR":"<PRIVATE_ROOT>/cache/deep-gemm","FLASHINFER_WORKSPACE_BASE":"<PRIVATE_ROOT>/cache/flashinfer","FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT":"<PRIVATE_ROOT>/model","GIT_OPTIONAL_LOCKS":"0","HF_DATASETS_CACHE":"<PRIVATE_ROOT>/cache/huggingface/datasets","HF_DATASETS_OFFLINE":"1","HF_HOME":"<PRIVATE_ROOT>/cache/huggingface","HF_HUB_OFFLINE":"1","HOME":"<PRIVATE_ROOT>/home","HUGGINGFACE_HUB_CACHE":"<PRIVATE_ROOT>/cache/huggingface/hub","LD_LIBRARY_PATH":"/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64","LOCAL_INFERENCE_CACHE_FINGERPRINT":"<PRIVATE_ROOT_ID>","MINFER_FMHA_CACHE_DIR":"<PRIVATE_ROOT>/cache/minfer/fmha","MM_SPARSE_ATTN_AOT_CACHE":"<PRIVATE_ROOT>/cache/minfer/mm-sparse-attn","NUMBA_CACHE_DIR":"<PRIVATE_ROOT>/cache/numba","PATH":"/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","PYTHONPATH":"<PRIVATE_ROOT>/runtime/vllm-source:<PRIVATE_ROOT>/runtime/b12x-source","PYTHONSAFEPATH":"1","SAFETENSORS_FAST_GPU":"1","SPARKINFER_COMPILE_CACHE_DIR":"<PRIVATE_ROOT>/cache/b12x/compile","TEMP":"<PRIVATE_ROOT>/tmp","TILELANG_CACHE_DIR":"<PRIVATE_ROOT>/cache/tilelang","TILELANG_TMP_DIR":"<PRIVATE_ROOT>/cache/tilelang/tmp","TMP":"<PRIVATE_ROOT>/tmp","TMPDIR":"<PRIVATE_ROOT>/tmp","TORCHINDUCTOR_CACHE_DIR":"<PRIVATE_ROOT>/cache/torchinductor","TORCH_EXTENSIONS_DIR":"<PRIVATE_ROOT>/cache/torch-extensions","TORCH_HOME":"<PRIVATE_ROOT>/cache/torch","TRANSFORMERS_CACHE":"<PRIVATE_ROOT>/cache/huggingface/transformers","TRANSFORMERS_OFFLINE":"1","TRITON_CACHE_DIR":"<PRIVATE_ROOT>/cache/triton","TVM_CACHE_DIR":"<PRIVATE_ROOT>/cache/tvm","TVM_FFI_CACHE_DIR":"<PRIVATE_ROOT>/cache/tvm-ffi","VLLM_CACHE_DIR":"<PRIVATE_ROOT>/cache/vllm","VLLM_CACHE_ROOT":"<PRIVATE_ROOT>/cache/vllm","VLLM_EXL3_ONLINE_CACHE_DIR":"<PRIVATE_ROOT>/cache/exl3-online","VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR":"<PRIVATE_ROOT>/cache/flashinfer-autotune","VLLM_PLUGINS":"","VLLM_USE_B12X_MOE":"1","VLLM_USE_B12X_SPARSE_INDEXER":"1","VLLM_WORKER_MULTIPROC_METHOD":"spawn","XDG_CACHE_HOME":"<PRIVATE_ROOT>/cache"}'
FRUIT_QSRT_RUNTIME_ENVIRONMENT_SHA256="377f72e1663fb8afa2fc69f33ca1a56ff8d7998d5c396988d0103c7baa751e1d"
RUNTIME_ENV=(
  /usr/bin/env -i
  "B12X_COMPILE_CACHE_DIR=${cache_root}/b12x/compile"
  "B12X_CUTE_COMPILE_CACHE_DIR=${cache_root}/b12x-cute"
  "B12X_ROOT=${runtime_root}/b12x-source"
  "CUDA_CACHE_PATH=${cache_root}/cuda"
  "CUDA_DEVICE_MAX_CONNECTIONS=32"
  "CUDA_VISIBLE_DEVICES=0"
  "CUPY_CACHE_DIR=${cache_root}/cupy"
  "CUTE_DSL_ARCH=sm_120a"
  "CUTE_DSL_CACHE_DIR=${cache_root}/cute-dsl"
  "DG_JIT_CACHE_DIR=${cache_root}/deep-gemm"
  "FLASHINFER_WORKSPACE_BASE=${cache_root}/flashinfer"
  "FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT=${MODEL}"
  "FRUIT_QSRT_RUNTIME_ENVIRONMENT_JSON=${FRUIT_QSRT_RUNTIME_ENVIRONMENT_JSON}"
  "FRUIT_QSRT_RUNTIME_ENVIRONMENT_SHA256=${FRUIT_QSRT_RUNTIME_ENVIRONMENT_SHA256}"
  "GIT_OPTIONAL_LOCKS=0"
  "HF_DATASETS_CACHE=${cache_root}/huggingface/datasets"
  "HF_DATASETS_OFFLINE=1"
  "HF_HOME=${cache_root}/huggingface"
  "HF_HUB_OFFLINE=1"
  "HOME=${home_root}"
  "HUGGINGFACE_HUB_CACHE=${cache_root}/huggingface/hub"
  "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
  "LOCAL_INFERENCE_CACHE_FINGERPRINT=${private_root##*/}"
  "MINFER_FMHA_CACHE_DIR=${cache_root}/minfer/fmha"
  "MM_SPARSE_ATTN_AOT_CACHE=${cache_root}/minfer/mm-sparse-attn"
  "NUMBA_CACHE_DIR=${cache_root}/numba"
  "PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONNOUSERSITE=1"
  "PYTHONPATH=${runtime_root}/vllm-source:${runtime_root}/b12x-source"
  "PYTHONSAFEPATH=1"
  "SAFETENSORS_FAST_GPU=1"
  "SPARKINFER_COMPILE_CACHE_DIR=${cache_root}/b12x/compile"
  "TEMP=${tmp_root}"
  "TILELANG_CACHE_DIR=${cache_root}/tilelang"
  "TILELANG_TMP_DIR=${cache_root}/tilelang/tmp"
  "TMP=${tmp_root}"
  "TMPDIR=${tmp_root}"
  "TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor"
  "TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions"
  "TORCH_HOME=${cache_root}/torch"
  "TRANSFORMERS_CACHE=${cache_root}/huggingface/transformers"
  "TRANSFORMERS_OFFLINE=1"
  "TRITON_CACHE_DIR=${cache_root}/triton"
  "TVM_CACHE_DIR=${cache_root}/tvm"
  "TVM_FFI_CACHE_DIR=${cache_root}/tvm-ffi"
  "VLLM_CACHE_DIR=${cache_root}/vllm"
  "VLLM_CACHE_ROOT=${cache_root}/vllm"
  "VLLM_EXL3_ONLINE_CACHE_DIR=${cache_root}/exl3-online"
  "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${cache_root}/flashinfer-autotune"
  "VLLM_PLUGINS="
  "VLLM_USE_B12X_MOE=1"
  "VLLM_USE_B12X_SPARSE_INDEXER=1"
  "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  "XDG_CACHE_HOME=${cache_root}"
  "${trust_name}=${QSRT_EXPECTED_MARKER_SHA256}"
)
if [[ -n "${VLLM_KQUANT_RUNTIME_EVIDENCE:-}" ]]; then
  RUNTIME_ENV+=(
    "VLLM_KQUANT_RUNTIME_EVIDENCE=${VLLM_KQUANT_RUNTIME_EVIDENCE}"
  )
fi

"${RUNTIME_ENV[@]}" "${PYTHON_BIN}" -I -S -c '
import hashlib
import json
import os
import subprocess
import stat
import sys
from pathlib import Path, PurePosixPath

RUNTIME_MANIFEST = Path("/opt/fruit-runtime/MANIFEST.sha256")


def sha256(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise RuntimeError(f"runtime package input is not regular: {path}")
        while chunk := os.read(descriptor, 8 << 20):
            bytes_read += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        bytes_read != before.st_size
        or source_stat_identity(before) != source_stat_identity(after)
        or source_stat_identity(after) != source_stat_identity(path_after)
    ):
        raise RuntimeError(f"runtime package input changed while hashing: {path}")
    return digest.hexdigest()


def read_stable_regular(path, max_bytes):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"runtime input is not singly linked: {path}")
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
    try:
        manifest_lines = read_stable_regular(
            RUNTIME_MANIFEST, 16 << 20
        ).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Fruit runtime manifest is not UTF-8") from exc
    for line in manifest_lines:
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



def open_tracked_file(root_fd, relative, directory_flags, file_flags):
    components = relative.split(b"/")
    parent_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(components[-1], file_flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, file_fd

def create_snapshot_file(root_fd, relative, mode, directory_flags):
    components = relative.split(b"/")
    parent_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(
            components[-1],
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, file_fd


def write_all(descriptor, value):
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written == 0:
            raise RuntimeError("short write while sealing runtime source")
        offset += written


def snapshot_tracked_tree(root_value, snapshot):
    root = Path(root_value).resolve(strict=True)
    index = tracked_source_index(root)
    if index != tracked_source_head(root):
        raise RuntimeError("runtime source index does not match HEAD")
    require_git_status_clean(root)
    snapshot.mkdir(mode=0o700)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    digest = hashlib.sha256(b"kquant-tracked-worktree-sha256-v1\0")
    root_fd = os.open(root, directory_flags)
    snapshot_root_fd = os.open(snapshot, directory_flags)
    try:
        for relative in sorted(index):
            expected_mode, expected_object_id = index[relative]
            parent_fd, file_fd = open_tracked_file(
                root_fd, relative, directory_flags, flags
            )
            snapshot_parent_fd = None
            snapshot_fd = None
            try:
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError("tracked runtime source is not regular")
                actual_mode = b"100755" if before.st_mode & 0o111 else b"100644"
                if actual_mode != expected_mode:
                    raise RuntimeError("tracked runtime source mode differs from Git")
                snapshot_parent_fd, snapshot_fd = create_snapshot_file(
                    snapshot_root_fd,
                    relative,
                    0o700 if actual_mode == b"100755" else 0o600,
                    directory_flags,
                )
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
                    write_all(snapshot_fd, chunk)
                after = os.fstat(file_fd)
                path_after = os.stat(
                    relative.rsplit(b"/", 1)[-1],
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    bytes_read != before.st_size
                    or source_stat_identity(before) != source_stat_identity(after)
                    or source_stat_identity(after) != source_stat_identity(path_after)
                ):
                    raise RuntimeError("tracked runtime source changed while snapshotting")
                if blob_digest.hexdigest().encode("ascii") != expected_object_id:
                    raise RuntimeError("tracked runtime source bytes differ from Git")
                if os.fstat(snapshot_fd).st_size != bytes_read:
                    raise RuntimeError("sealed runtime source has an invalid size")
            finally:
                if snapshot_fd is not None:
                    os.close(snapshot_fd)
                if snapshot_parent_fd is not None:
                    os.close(snapshot_parent_fd)
                os.close(file_fd)
                os.close(parent_fd)
    finally:
        os.close(snapshot_root_fd)
        os.close(root_fd)
    require_git_status_clean(root)
    return digest.hexdigest()


model_source = sys.argv[1]
model = sys.argv[2]
expected_marker_sha256 = sys.argv[3]
expected_kquant = sys.argv[4]
expected_b12x = sys.argv[5]
actual_vllm = sys.argv[6]
b12x_root = Path(sys.argv[7]).resolve(strict=True)
vllm_root = Path(sys.argv[8]).resolve(strict=True)
runtime_root = Path(sys.argv[9]).resolve(strict=True)
python_bin = sys.argv[10]
trust_mode = sys.argv[11]
if trust_mode not in {"complete", "candidate"}:
    raise RuntimeError("Fruit publication trust mode is invalid")
candidate_mode = trust_mode == "candidate"
marker_name = "QSRT_CANDIDATE.json" if candidate_mode else "QSRT_COMPLETE.json"
marker_kind = "candidate" if candidate_mode else "completion"
b12x_snapshot = runtime_root / "b12x-source"
vllm_snapshot = runtime_root / "vllm-source"
b12x_source_sha256 = snapshot_tracked_tree(b12x_root, b12x_snapshot)
vllm_source_sha256 = snapshot_tracked_tree(vllm_root, vllm_snapshot)
verify_runtime_manifest(vllm_snapshot, b12x_snapshot)
publication_tool = (
    vllm_snapshot
    / "vllm/model_executor/layers/quantization/kquant_qsrt_publication.py"
)
publication_arguments = (
    python_bin,
    "-I",
    "-S",
    str(publication_tool),
    model_source,
    model,
    expected_marker_sha256,
)
if candidate_mode:
    publication_arguments += ("--candidate-mode",)
try:
    subprocess.run(publication_arguments, check=True)
except (OSError, subprocess.CalledProcessError) as exc:
    raise RuntimeError("cannot create an authenticated Fruit model snapshot") from exc
model_root = Path(model).resolve(strict=True)
marker_bytes = read_stable_regular(model_root / marker_name, 1 << 20)
if hashlib.sha256(marker_bytes).hexdigest() != expected_marker_sha256:
    raise RuntimeError(
        f"Fruit {marker_kind} marker disagrees with the external anchor"
    )
try:
    marker = json.loads(marker_bytes)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"Fruit {marker_kind} marker is not valid JSON") from exc
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
if runtime.get("b12x_source_sha256") != b12x_source_sha256:
    raise RuntimeError("Fruit package B12X source fingerprint is unsupported")
if runtime.get("vllm_source_sha256") != vllm_source_sha256:
    raise RuntimeError("Fruit package vLLM source fingerprint is unsupported")
' \
  "${model_source}" \
  "${MODEL}" \
  "${QSRT_EXPECTED_MARKER_SHA256}" \
  "${EXPECTED_KQUANT_REVISION}" \
  "${EXPECTED_B12X_REVISION}" \
  "${actual_vllm_revision}" \
  "${B12X_ROOT}" \
  "${SCRIPT_DIR}" \
  "${runtime_root}" \
  "${PYTHON_BIN}" \
  "${QSRT_TRUST_MODE}"

VLLM_ROOT="${runtime_root}/vllm-source"
B12X_ROOT="${runtime_root}/b12x-source"

"${RUNTIME_ENV[@]}" "${PYTHON_BIN}" -P -c '
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
' "${B12X_ROOT}" "${VLLM_ROOT}"


HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-QSRT-Fruit-Instruct}"

session_ready="${private_root}/server-session.ready"
"${RUNTIME_ENV[@]}" "${PYTHON_BIN}" -I -S -c '
import os
import sys

os.setsid()
ready = os.open(
    sys.argv[1],
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o600,
)
os.close(ready)
os.execv(sys.argv[2], sys.argv[2:])
' "${session_ready}" "${PYTHON_BIN}" -P -m vllm.entrypoints.cli.main serve "${MODEL}" \
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
  --compilation-config '{"backend":"inductor","cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,48,64]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --generation-config vllm &
server_pid=$!
for _ in {1..500}; do
  if [[ -f "${session_ready}" && ! -L "${session_ready}" ]]; then
    server_pgid="${server_pid}"
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    if wait "${server_pid}"; then
      server_status=0
    else
      server_status=$?
    fi
    exit "${server_status}"
  fi
  /usr/bin/sleep 0.01
done
if [[ -z "${server_pgid}" ]]; then
  echo "vLLM server did not establish its private process session" >&2
  exit 1
fi
if wait "${server_pid}"; then
  server_status=0
else
  server_status=$?
fi
exit "${server_status}"
