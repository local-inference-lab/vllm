#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

action=${1:-verify}
PROFILE_PRIMARY=spark-rdma-primary
PROFILE_SECONDARY=spark-rdma-secondary

usage() {
  cat <<'EOF'
Usage: tools/spark/network.sh configure|verify|bench|rollback

Manage only the two direct Spark links. The management LAN is never modified.
EOF
}

nmcli_local() {
  sudo -n nmcli "$@"
}

nmcli_remote() {
  ssh "${REMOTE_HOST}" sudo -n nmcli "$@"
}

configure_profile() {
  local side=$1
  local profile=$2
  local iface=$3
  local address=$4
  local runner=nmcli_local
  [[ "${side}" == remote ]] && runner=nmcli_remote

  if "${runner}" -t -f NAME connection show | grep -Fxq "${profile}"; then
    "${runner}" connection modify "${profile}" \
      connection.interface-name "${iface}" \
      connection.autoconnect yes \
      ipv4.method manual \
      ipv4.addresses "${address}/24" \
      ipv4.never-default yes \
      ipv6.method link-local \
      ipv6.never-default yes \
      802-3-ethernet.mtu 9000
  else
    "${runner}" connection add type ethernet \
      con-name "${profile}" ifname "${iface}" \
      connection.autoconnect yes \
      ipv4.method manual ipv4.addresses "${address}/24" \
      ipv4.never-default yes \
      ipv6.method link-local ipv6.never-default yes \
      802-3-ethernet.mtu 9000
  fi
  "${runner}" connection up "${profile}" ifname "${iface}"
}

configure() {
  require_command nmcli
  sudo -n true || die "passwordless local sudo is required"
  ssh "${REMOTE_HOST}" sudo -n true || \
    die "passwordless sudo is required on ${REMOTE_HOST}"

  configure_profile local "${PROFILE_PRIMARY}" \
    "${PRIMARY_IFACE}" "${HEAD_ADDR}"
  configure_profile local "${PROFILE_SECONDARY}" \
    "${SECONDARY_IFACE}" "${HEAD_ADDR_SECONDARY}"
  configure_profile remote "${PROFILE_PRIMARY}" \
    "${PRIMARY_IFACE}" "${WORKER_ADDR}"
  configure_profile remote "${PROFILE_SECONDARY}" \
    "${SECONDARY_IFACE}" "${WORKER_ADDR_SECONDARY}"
  verify
}

verify() {
  log "checking direct-link addresses and jumbo frames"
  ip -br address show "${PRIMARY_IFACE}"
  ip -br address show "${SECONDARY_IFACE}"
  ssh "${REMOTE_HOST}" \
    "ip -br address show '${PRIMARY_IFACE}'; ip -br address show '${SECONDARY_IFACE}'"
  ping -c 3 -M do -s 8972 "${WORKER_ADDR}"
  ping -c 3 -M do -s 8972 "${WORKER_ADDR_SECONDARY}"

  log "checking both RoCE HCAs"
  ibv_devinfo -d "${PRIMARY_HCA}" | grep -E 'hca_id|state:|phys_state:'
  ibv_devinfo -d "${SECONDARY_HCA}" | grep -E 'hca_id|state:|phys_state:'
  ssh "${REMOTE_HOST}" \
    "ibv_devinfo -d '${PRIMARY_HCA}' | grep -E 'hca_id|state:|phys_state:'; ibv_devinfo -d '${SECONDARY_HCA}' | grep -E 'hca_id|state:|phys_state:'"
}

bench_hca() {
  local hca=$1
  local address=$2
  local port=$3
  local remote_log="${STATE_ROOT}/ib-write-bw-${hca}.log"
  ssh "${REMOTE_HOST}" \
    "mkdir -p '${STATE_ROOT}'; nohup timeout 30 ib_write_bw -d '${hca}' -F --report_gbits -D 5 -p '${port}' >'${remote_log}' 2>&1 </dev/null &"
  for _ in {1..20}; do
    if ib_write_bw -d "${hca}" -F --report_gbits -D 5 -p "${port}" \
      "${address}"; then
      ssh "${REMOTE_HOST}" "cat '${remote_log}'"
      return
    fi
    sleep 1
  done
  ssh "${REMOTE_HOST}" "cat '${remote_log}'" || true
  die "ib_write_bw failed for ${hca}"
}

bench() {
  verify
  bench_hca "${PRIMARY_HCA}" "${WORKER_ADDR}" 18515
  bench_hca "${SECONDARY_HCA}" "${WORKER_ADDR_SECONDARY}" 18516
}

rollback() {
  for profile in "${PROFILE_PRIMARY}" "${PROFILE_SECONDARY}"; do
    nmcli_local connection delete "${profile}" 2>/dev/null || true
    nmcli_remote connection delete "${profile}" 2>/dev/null || true
  done
  log "removed Spark-managed direct-link profiles; management networking was untouched"
}

case "${action}" in
  configure) configure ;;
  verify) verify ;;
  bench) bench ;;
  rollback) rollback ;;
  -h|--help) usage ;;
  *) usage; die "unknown network action: ${action}" ;;
esac
