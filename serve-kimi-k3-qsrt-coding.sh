#!/usr/bin/env bash
# Serve the current Kimi-K3 QSRT coding-specialist checkpoint at TP12.
set -euo pipefail

K3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K3_BASE_LAUNCHER="${K3_SCRIPT_DIR}/serve-kimi-k3-exl3-3p09-tp12.sh"

export K3_MODEL_DIR="${K3_MODEL_DIR:-/models/Kimi-K3-QSRT-CHEB-Q8H4-ROUTED-X4T-3p11-KLD-v1-serve}"
export K3_SERVED_MODEL_NAME="${K3_SERVED_MODEL_NAME:-kimi-k3-qsrt-coding}"

if [[ ! -d "${K3_MODEL_DIR}" ]]; then
  echo "QSRT checkpoint directory not found: ${K3_MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -x "${K3_BASE_LAUNCHER}" ]]; then
  echo "Base Kimi-K3 TP12 launcher not found or not executable: ${K3_BASE_LAUNCHER}" >&2
  exit 1
fi

exec "${K3_BASE_LAUNCHER}" "$@"
