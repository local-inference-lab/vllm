#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

glm_mla="$root/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
mla_indexer="$root/vllm/v1/attention/backends/mla/indexer.py"
attn_backend="$root/vllm/v1/attention/backend.py"
attn_utils="$root/vllm/v1/attention/backends/utils.py"
fusion_matcher="$root/vllm/compilation/passes/fusion/matcher_utils.py"
ar_fusion="$root/vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
synthetic_sampler="$root/vllm/v1/worker/gpu/spec_decode/synthetic_rejection_sampler_utils.py"
forward_context="$root/vllm/forward_context.py"
attn_registry="$root/vllm/v1/attention/backends/registry.py"
speculative_config="$root/vllm/config/speculative.py"
spec_dflash="$root/vllm/v1/spec_decode/dflash.py"
llm_base_proposer="$root/vllm/v1/spec_decode/llm_base_proposer.py"
gpu_model_runner="$root/vllm/v1/worker/gpu_model_runner.py"
gpu_input_batch="$root/vllm/v1/worker/gpu_input_batch.py"
ubatch_utils="$root/vllm/v1/worker/ubatch_utils.py"
kimi_run="$root/scripts/run-kimi26-vllm"
glm_run="$root/scripts/run-glm51-vllm"

if [[ ! -f "$kimi_run" && -f /usr/local/bin/run-kimi26-vllm ]]; then
  kimi_run=/usr/local/bin/run-kimi26-vllm
fi
if [[ ! -f "$glm_run" && -f /usr/local/bin/run-glm51-vllm ]]; then
  glm_run=/usr/local/bin/run-glm51-vllm
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_marker() {
  local path="$1"
  local marker="$2"
  if ! grep -Fq "$marker" "$path"; then
    echo "missing marker in $path: $marker" >&2
    exit 1
  fi
}

reject_marker() {
  local path="$1"
  local marker="$2"
  if grep -Fq "$marker" "$path"; then
    echo "forbidden marker in $path: $marker" >&2
    exit 1
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

require_file "$glm_mla"
require_file "$mla_indexer"
require_file "$attn_backend"
require_file "$attn_utils"
require_file "$fusion_matcher"
require_file "$ar_fusion"
require_file "$synthetic_sampler"
require_file "$forward_context"
require_file "$attn_registry"
require_file "$speculative_config"
require_file "$spec_dflash"
require_file "$llm_base_proposer"
require_file "$gpu_model_runner"
require_file "$gpu_input_batch"
require_file "$ubatch_utils"
require_file "$kimi_run"
require_file "$glm_run"

# GLM DCP long-context coherence fix from the verified 2026-05-08 image.
require_marker "$glm_mla" "_b12x_split_decode_final_lse_kernel"
require_marker "$glm_mla" "_sparse_mla_split_decode_forward_with_lse_vllm_metadata"
require_marker "$glm_mla" "decode_inline_lse"
require_marker "$glm_mla" "nsa_cu_seqlens_k"
require_marker "$glm_mla" "can_return_lse_for_decode: bool = True"
require_marker "$mla_indexer" "class DeepseekV4IndexerBackend"
require_marker "$mla_indexer" "DEEPSEEK_V4_INDEXER"
require_marker "$fusion_matcher" "class MatcherFusedAddRMSNorm"
require_marker "$fusion_matcher" "ir.ops.fused_add_rms_norm"
require_marker "$ar_fusion" "VllmPatternReplacement"
require_marker "$ar_fusion" "CustomAllreduce"
require_marker "$ar_fusion" "rocm_aiter_ops"
require_marker "$synthetic_sampler" "compute_synthetic_rejection_sampler_params"
require_marker "$synthetic_sampler" "MIN_ACCEPTANCE_DECAY_FACTOR"
require_marker "$forward_context" "def static_forward_context"
require_marker "$forward_context" "BOB_DISABLE_STATIC_HOIST"
require_marker "$gpu_input_batch" "is_spec_decode"

# CPU/GPU sync avoidance from glm51-b12x-a16-padfix-cpuhangfix-20260511.
# This carries an exact CPU-side upper-bound shadow for seq_lens so DCP/B12X
# metadata does not fall back to D2H seq_lens reads in async/spec paths.
require_marker "$attn_backend" "seq_lens_cpu_upper_bound"
require_marker "$attn_utils" "seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound"
require_marker "$spec_dflash" "cad.seq_lens_cpu_upper_bound + num_query_per_req"
require_marker "$llm_base_proposer" "or seq_lens_cpu_upper_bound to avoid D2H sync"
require_marker "$gpu_model_runner" "seq_lens_cpu_upper_bound = seq_lens_cpu"
require_marker "$ubatch_utils" "seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound"
require_marker "$mla_indexer" "seq_lens_cpu_upper_bound"

# Kimi K2.6 launch defaults from the verified 2026-05-10 image.
require_marker "$kimi_run" 'MODEL="${MODEL:-moonshotai/Kimi-K2.6}"'
require_marker "$kimi_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_MLA}"'
require_marker "$kimi_run" 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"'
require_marker "$kimi_run" "lightseekorg/kimi-k2.6-eagle3-mla"
reject_marker "$kimi_run" '"draft_attention_backend":"TRITON_MLA"'

# GLM launch defaults.
require_marker "$glm_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"'
require_marker "$glm_run" "lukealonso/GLM-5.1-NVFP4-MTP"
require_marker "$glm_run" '"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"'

require_sha256 "$glm_mla" "1fa71fc3a934831077b90dea555254d68366371e5a3969766e79cfec222ce418"
require_sha256 "$kimi_run" "08b3d317be09c32bf2c68c2ba5f74f38ebed84e9c012c2a3213a45f7852ae0dc"
require_sha256 "$glm_run" "1bd2eae9ae22534d96bc37f5ad7180bbe943cbd3fbf0c79f482b9a632a454f6a"

installed_glm_mla="/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
if [[ -f "$installed_glm_mla" ]]; then
  require_sha256 "$installed_glm_mla" "1fa71fc3a934831077b90dea555254d68366371e5a3969766e79cfec222ce418"
fi

echo "GLM/Kimi canonical source checks passed."
