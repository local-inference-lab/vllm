#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
B12X_ROOT="${B12X_ROOT:-${SCRIPT_DIR}/../b12x-fruit}"
MODEL="${MODEL:-/mnt/vault/llm/fruit-pilot/output/GLM-5.2-SIQ-Fruit-QSRT-exact}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
if [[ "${MAX_NUM_SEQS}" != "1" ]]; then
  echo "Fruit QSRT has only been qualified at MAX_NUM_SEQS=1" >&2
  exit 1
fi
if [[ "${TENSOR_PARALLEL_SIZE}" != "1" ]]; then
  echo "Fruit QSRT has only been qualified at tensor parallel size 1" >&2
  exit 1
fi

EXPECTED_B12X_REVISION="de50e8622a8695e9829c83ad9f8c96f9b3be573a"
EXPECTED_KQUANT_REVISION="f1ce7c8f4a9564194ea7067e1a88282a8e39135c"

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
if [[ -n "$(git -C "${B12X_ROOT}" status --porcelain -- b12x)" ]]; then
  echo "Fruit B12X source tree has uncommitted changes: ${B12X_ROOT}" >&2
  exit 1
fi
actual_vllm_revision="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)" || {
  echo "Unable to resolve the Fruit vLLM revision: ${SCRIPT_DIR}" >&2
  exit 1
}
if [[ -n "$(git -C "${SCRIPT_DIR}" status --porcelain -- vllm)" ]]; then
  echo "Fruit vLLM source tree has uncommitted changes: ${SCRIPT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${B12X_ROOT}"

"${PYTHON_BIN}" -P -c '
import hashlib
import sys
from pathlib import Path

import b12x
import vllm
from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
    verify_qsrt_publication,
)

def source_tree_sha256(root_value):
    root = Path(root_value).resolve(strict=True)
    files = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"runtime source must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise RuntimeError(f"runtime source tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()

seal = verify_qsrt_publication(sys.argv[1])
producer = seal.manifest["producer"]
encoder = producer["encoder"]
runtime = producer["runtime"]
if encoder.get("kquant_revision") != sys.argv[2]:
    raise RuntimeError("Fruit package KQuant revision is unsupported")
if runtime.get("b12x_revision") != sys.argv[3]:
    raise RuntimeError("Fruit package B12X revision is unsupported")
if runtime.get("vllm_revision") != sys.argv[4]:
    raise RuntimeError("Fruit package vLLM revision does not match the runtime")
b12x_root = Path(sys.argv[5]).resolve(strict=True)
vllm_root = Path(sys.argv[6]).resolve(strict=True)
if not Path(b12x.__file__).resolve().is_relative_to(b12x_root / "b12x"):
    raise RuntimeError("imported B12X does not come from the sealed source root")
if not Path(vllm.__file__).resolve().is_relative_to(vllm_root / "vllm"):
    raise RuntimeError("imported vLLM does not come from the sealed source root")
if runtime.get("b12x_source_sha256") != source_tree_sha256(b12x_root / "b12x"):
    raise RuntimeError("Fruit package B12X source fingerprint is unsupported")
if runtime.get("vllm_source_sha256") != source_tree_sha256(vllm_root / "vllm"):
    raise RuntimeError("Fruit package vLLM source fingerprint is unsupported")

import b12x.moe.fused_moe
import b12x.attention.sparse_mla
' \
  "${MODEL}" \
  "${EXPECTED_KQUANT_REVISION}" \
  "${EXPECTED_B12X_REVISION}" \
  "${actual_vllm_revision}" \
  "${B12X_ROOT}" \
  "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-QSRT-Fruit}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

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
  --generation-config vllm \
  "$@"
