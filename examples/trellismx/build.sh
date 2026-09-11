#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
revision=$(git rev-parse HEAD)
core_image=${JOVIAN_IMAGE:-local/trellismx-jovian-core:${revision:0:12}}
release_image=${TRELLISMX_IMAGE:-local/trellismx-jovian:review}
docker build -f docker/Dockerfile --target vllm-openai \
  --build-arg VLLM_BUILD_COMMIT="$revision" \
  -t "$core_image" "$@" .
docker build -f examples/trellismx/Dockerfile \
  --build-arg JOVIAN_IMAGE="$core_image" -t "$release_image" .
