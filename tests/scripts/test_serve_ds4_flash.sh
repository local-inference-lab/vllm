#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="${repo_root}/serve-ds4-flash.sh"
vision_model=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp

capture() {
  env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    PYTHON_BIN=/bin/true \
    DRY_RUN=1 \
    MODE=dspark \
    MODEL="${vision_model}" \
    "$@" \
    "${launcher}" 2>&1
}

assert_contains() {
  local output=$1 expected=$2
  if [[ "${output}" != *"${expected}"* ]]; then
    printf 'Missing expected launcher fragment: %s\n%s\n' \
      "${expected}" "${output}" >&2
    exit 1
  fi
}

default_output="$(capture)"
assert_contains "${default_output}" 'variant=vision mode=dspark'
assert_contains "${default_output}" 'DeepSeek-V4-Flash-Vision-Exp'
assert_contains "${default_output}" '--revision 6821d6ad3681a4b137b066b76094fa82ebd0a380'
assert_contains "${default_output}" 'num_speculative_tokens\":3'
assert_contains "${default_output}" '--gpu-memory-utilization 0.975'

lmcache_output="$(capture LMCACHE_MODE=ram)"
assert_contains "${lmcache_output}" '--gpu-memory-utilization 0.96'

engine_driven_output="$(capture \
  LMCACHE_MODE=ram LMCACHE_TRANSFER_MODE=engine_driven)"
assert_contains "${engine_driven_output}" '--gpu-memory-utilization 0.975'

explicit_output="$(capture \
  LMCACHE_MODE=ram GPU_MEMORY_UTILIZATION=0.965 DSPARK_TOKENS=2)"
assert_contains "${explicit_output}" '--gpu-memory-utilization 0.965'
assert_contains "${explicit_output}" 'num_speculative_tokens\":2'
