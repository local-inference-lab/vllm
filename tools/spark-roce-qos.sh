#!/usr/bin/env bash
# Install as /etc/NetworkManager/dispatcher.d/90-spark-roce-qos (root, mode 0755).
set -euo pipefail

case "${1:-}" in
  enp1s0f0np0|enP2p1s0f0np0|enp1s0f1np1|enP2p1s0f1np1) ;;
  *) exit 0 ;;
esac
case "${2:-}" in
  up|reapply) ;;
  *) exit 0 ;;
esac

/usr/bin/mlnx_qos -i "$1" --dcbx=os --trust=dscp --pfc=0,0,0,1,0,0,0,0
/usr/bin/mlnx_qos -i "$1" --dscp2prio=set,26,3
/usr/bin/mlnx_qos -i "$1" --dscp2prio=set,48,6
