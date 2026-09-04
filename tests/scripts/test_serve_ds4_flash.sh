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
assert_contains "${default_output}" '--max-model-len 1048576'

text_output="$(capture \
  MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 DS4_MODEL_VARIANT=text)"
assert_contains "${text_output}" 'variant=text mode=dspark'
assert_contains "${text_output}" '--max-model-len 131072'

lmcache_output="$(capture LMCACHE_MODE=ram)"
assert_contains "${lmcache_output}" '--gpu-memory-utilization 0.951'
assert_contains "${lmcache_output}" '--max-model-len 900000'

engine_driven_output="$(capture \
  LMCACHE_MODE=ram LMCACHE_TRANSFER_MODE=engine_driven)"
assert_contains "${engine_driven_output}" '--gpu-memory-utilization 0.975'
assert_contains "${engine_driven_output}" '--max-model-len 1048576'

explicit_output="$(capture \
  LMCACHE_MODE=ram GPU_MEMORY_UTILIZATION=0.965 DSPARK_TOKENS=2)"
assert_contains "${explicit_output}" '--gpu-memory-utilization 0.965'
assert_contains "${explicit_output}" 'num_speculative_tokens\":2'

explicit_high_capacity_output="$(capture \
  LMCACHE_MODE=ram GPU_MEMORY_UTILIZATION=0.96 MAX_MODEL_LEN=1048576)"
assert_contains "${explicit_high_capacity_output}" '--gpu-memory-utilization 0.96'
assert_contains "${explicit_high_capacity_output}" '--max-model-len 1048576'

if unsafe_high_capacity_output="$(capture \
    LMCACHE_MODE=ram MAX_MODEL_LEN=1048576)"; then
  printf 'Vision direct LMCache accepted an unqualified 1M memory profile:\n%s\n' \
    "${unsafe_high_capacity_output}" >&2
  exit 1
fi
assert_contains "${unsafe_high_capacity_output}" \
  'MAX_MODEL_LEN above 900000 requires an explicit GPU_MEMORY_UTILIZATION override'
