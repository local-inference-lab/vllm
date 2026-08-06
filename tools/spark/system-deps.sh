#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

host=${1:-both}
packages=(
  build-essential
  ccache
  cmake
  curl
  git
  iputils-ping
  libcudnn9-cuda-13
  libcudnn9-dev-cuda-13
  libibverbs-dev
  libopenmpi-dev
  libprotobuf-dev
  network-manager
  ninja-build
  perftest
  pkg-config
  protobuf-compiler
  python3-dev
  rdma-core
  rsync
  shellcheck
)

usage() {
  cat <<'EOF'
Usage: tools/spark/system-deps.sh [local|luxon|both]

Install the host packages used by the non-container Spark build. This does not
replace the NVIDIA driver, CUDA toolkit, or system Python.
EOF
}

install_local() {
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends "${packages[@]}"
  sudo -n loginctl enable-linger "$(id -un)"
}

install_remote() {
  local package_args remote_command
  printf -v package_args ' %q' "${packages[@]}"
  remote_command="sudo -n apt-get update && sudo -n env "
  remote_command+="DEBIAN_FRONTEND=noninteractive apt-get install -y "
  remote_command+="--no-install-recommends${package_args}"
  remote_command+=" && sudo -n loginctl enable-linger \"\$(id -un)\""
  # The package arguments were escaped with printf %q above.
  # shellcheck disable=SC2029
  ssh "${REMOTE_HOST}" "${remote_command}"
}

case "${host}" in
  local) install_local ;;
  luxon) install_remote ;;
  both)
    install_local
    install_remote
    ;;
  -h|--help) usage ;;
  *) usage; die "host must be local, luxon, or both" ;;
esac

log "system dependencies are ready on ${host}"
