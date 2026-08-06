#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

action=${1:-env}

verify_python() {
  local root=$1
  "${root}/.venv/bin/python" - <<'PY'
import importlib
import json
import site

import torch

modules = [
    "vllm",
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm._qutlass_C",
    "flashinfer",
    "b12x",
]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "imported")

result = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "capability": torch.cuda.get_device_capability(),
    "modules": versions,
    "user_site_enabled": site.ENABLE_USER_SITE,
}
assert result["capability"] == (12, 1), result
print(json.dumps(result, sort_keys=True))
PY
}

verify_nccl_link() {
  local expected site_packages actual
  expected=$(readlink -f "$(nccl_so)")
  site_packages=$("${VENV_PYTHON}" -c \
    'import site; print(site.getsitepackages()[0])')
  actual=$(readlink -f \
    "${site_packages}/nvidia/nccl/lib/libnccl.so.2")
  [[ "${actual}" == "${expected}" ]] || \
    die "Python NCCL resolves to ${actual}; expected ${expected}"
}

verify_env() {
  assert_runtime_inputs
  ensure_uv
  verify_python "${VLLM_ROOT}"
  verify_nccl_link
  uv_pip_check
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_VLLM_ROOT}' && tools/spark/verify.sh remote-env"
}

verify_remote_env() {
  ensure_uv
  verify_python "${VLLM_ROOT}"
  verify_nccl_link
  uv_pip_check
}

run_pytests() {
  local tests=(
    tests/tokenizers_/test_deepseek_v4.py
    tests/parser/engine/test_deepseek_v4.py
    tests/tool_parsers/test_deepseekv4_tool_parser.py
    tests/models/deepseek_v4/test_b12x_mhc_expected_m.py
    tests/v1/attention/test_b12x_attn.py
    tests/model_executor/layers/test_sparse_attn_indexer_b12x.py
    tests/distributed/test_b12x_fused_all_reduce.py
  )
  local existing=()
  local test
  for test in "${tests[@]}"; do
    [[ -e "${VLLM_ROOT}/${test}" ]] && existing+=("${VLLM_ROOT}/${test}")
  done
  [[ ${#existing[@]} -gt 0 ]] || die "none of the targeted tests exist"
  "${VENV_PYTHON}" -m pytest -v "${existing[@]}"
}

run_nccl_smoke() {
  require_command timeout
  local library
  library=$(nccl_so) || die "custom NCCL library is missing"
  local log_dir="${STATE_ROOT}/logs"
  local head_log="${log_dir}/nccl-head.log"
  local worker_log="${log_dir}/nccl-worker.log"
  mkdir -p "${log_dir}"
  rm -f "${head_log}" "${worker_log}"

  local common_env=(
    CUDA_VISIBLE_DEVICES=0
    NCCL_DEBUG=INFO
    NCCL_DEBUG_SUBSYS=INIT,NET
    NCCL_IB_DISABLE=0
    NCCL_IB_HCA="${PRIMARY_HCA},${SECONDARY_HCA}"
    NCCL_NET_PLUGIN=none
    NCCL_SOCKET_IFNAME="${PRIMARY_IFACE}"
    GLOO_SOCKET_IFNAME="${PRIMARY_IFACE}"
    MASTER_ADDR="${HEAD_ADDR}"
    MASTER_PORT=29601
    WORLD_SIZE=2
    LD_LIBRARY_PATH="$(dirname "${library}"):${CUDA_HOME:-/usr/local/cuda}/lib64"
    LD_PRELOAD="${library}"
  )
  local worker_env=("${common_env[@]}" RANK=1)
  local head_env=("${common_env[@]}" RANK=0)
  local remote_command
  printf -v remote_command '%q ' env "${worker_env[@]}" timeout 180 \
    "${REMOTE_VLLM_ROOT}/.venv/bin/python" \
    "${REMOTE_VLLM_ROOT}/tools/spark/nccl_smoke.py"

  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_VLLM_ROOT}' && ${remote_command}" \
    >"${worker_log}" 2>&1 &
  local remote_pid=$!
  trap 'kill "${remote_pid}" 2>/dev/null || true' RETURN
  sleep 2

  local head_status worker_status
  set +e
  env "${head_env[@]}" timeout 180 "${VENV_PYTHON}" \
    "${SPARK_DIR}/nccl_smoke.py" >"${head_log}" 2>&1
  head_status=$?
  wait "${remote_pid}"
  worker_status=$?
  set -e
  trap - RETURN

  cat "${head_log}"
  cat "${worker_log}"
  [[ ${head_status} -eq 0 && ${worker_status} -eq 0 ]] || \
    die "NCCL smoke failed; inspect ${head_log} and ${worker_log}"
  if ! grep -Eq 'NET/IB|Using.*IB' "${head_log}" "${worker_log}"; then
    die "NCCL smoke passed without evidence of the IB transport"
  fi
  log "two-node custom-NCCL all-reduce passed over IB"
}

smoke_api() {
  local response
  curl -fsS http://127.0.0.1:8000/health >/dev/null
  curl -fsS http://127.0.0.1:8000/v1/models
  response=$(curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply with exactly: two sparks ready"}],"temperature":0,"max_tokens":128}')
  printf '%s\n' "${response}"
  SMOKE_RESPONSE="${response}" "${VENV_PYTHON}" - <<'PY'
import json
import os

response = json.loads(os.environ["SMOKE_RESPONSE"])
content = response["choices"][0]["message"]["content"]
if content.strip() != "two sparks ready":
    raise SystemExit(f"unexpected smoke response: {content!r}")
PY
}

case "${action}" in
  env) verify_env ;;
  remote-env) verify_remote_env ;;
  pytest) run_pytests ;;
  nccl) run_nccl_smoke ;;
  network) "${SPARK_DIR}/network.sh" verify ;;
  smoke) smoke_api ;;
  *) die "verify target must be env, pytest, nccl, network, or smoke" ;;
esac
