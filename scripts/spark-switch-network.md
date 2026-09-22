# Four-Spark switched RoCE network

The Sparks connect through a MikroTik CRS812-8DS-2DQ-2DDQ. Each connected
QSFP port exposes two Ethernet/RoCE interfaces, one for each PCIe path.
Use a separate `/24` subnet per interface and the same host suffix throughout.

| Host | Management | `enp1s0f0np0` | `enP2p1s0f0np0` | Switch port |
| --- | --- | --- | --- | --- |
| Tachyon | `192.168.42.223` | `10.200.0.10` | `10.200.1.10` | `qsfp56-dd-2-1` |
| Luxon | `192.168.42.110` | `10.200.0.11` | `10.200.1.11` | `qsfp56-dd-1-1` |
| Graviton | `192.168.42.55` | `10.200.0.12` | `10.200.1.12` | `qsfp56-2-1` |
| Chroniton | `192.168.42.78` | `10.200.0.13` | `10.200.1.13` | `qsfp56-1-1` |

The unconnected QSFP port reserves `10.200.2.10–13` on `enp1s0f1np1`
and `10.200.3.10–13` on `enP2p1s0f1np1`.

The switch has `10.200.0.1/24` on its existing hardware-offloaded bridge.
From a Spark, use `ssh admin@10.200.0.1`; from the management LAN, use
`ssh -J luke@192.168.42.223 admin@10.200.0.1`.

Use the management addresses above for automation. During validation,
`graviton` resolved to stale LAN address `192.168.42.91`; its verified
management address is `192.168.42.55`. The launchers use explicit addresses.

## Host configuration

Persistent configuration is `/etc/netplan/40-cx7-switch.yaml` on each host,
using NetworkManager. The four profile names remain `netplan-cx7p0a`,
`netplan-cx7p0b`, `netplan-cx7p1a`, and `netplan-cx7p1b`.
All RDMA interfaces use static addresses, MTU 9000, disabled DHCP, and no
gateway or DNS settings. Management and the default route remain on `enP7s7`.
The former ring profile is backed up under
`/etc/netplan/backup-switch-20260918/` and is outside Netplan's active files.

Install `scripts/spark-roce-qos.sh` as
`/etc/NetworkManager/dispatcher.d/90-spark-roce-qos`, owned by root with mode
0755. It reapplies DSCP trust and PFC on priority 3 when a ConnectX interface
comes up. DSCP 26 maps to priority 3; DSCP 48 maps to priority 6. To apply it
immediately, run it as root with the interface name and `up` as its arguments.

## Serving

The TP2 and TP4 launchers select `rocep1s0f0,roceP2p1s0f0` on every rank.
ROCEnante is the default for eligible all-reduce and all-gather operations;
NCCL handles operations outside its configured limits. `ALLREDUCE=nccl`
selects NCCL exclusively.

The TP4 launcher no longer forces the physical ring's rank order, skips tree
connections, or uses `/30` subnet routing. Management networking supplies
bootstrap; data collectives use RDMA. Both interfaces must have populated
IPv4 RoCE v2 GIDs at index 3, as selected by the launchers.

Both transports default to IP traffic class 106: DSCP 26 with ECN-capable
marking. `NCCL_IB_TC` overrides the NCCL value; `B12X_ROCE_TRAFFIC_CLASS`
overrides ROCEnante and otherwise follows `NCCL_IB_TC`. These settings must
match the NIC and switch QoS configuration.

For DeepSeek V4.1 TP4 with DSpark 7:

```bash
scripts/serve-ds41-flash-dspark-tp4-rdma.sh --detach -- --enable-prompt-tokens-details
docker logs -f --tail 100 vllm_ds41_flash_tp4
```

The API is `http://192.168.42.223:8000/v1`, with served model name
`DeepSeek-V4.1-Flash`.

## Switch and validation

RouterOS and `switch-marvell` are 7.23.7 long-term. CRS8xx ECN/PFC support
requires 7.23 or newer; the old 7.20.8 release exposed QoS settings without
complete support for this hardware.

All four Spark ports use `auto-negotiation=no speed=200G-baseCR4`,
`l2mtu=9500`, and `mtu=9000`. Automatic negotiation selected only 100 Gb/s
with the supplied Amphenol cables. This manual speed setting follows
[NVIDIA's switched Spark guide](https://build.nvidia.com/spark/multi-sparks-through-switch/multi-sparks).
The two Linux interfaces share one physical 200 Gb/s link; they give the NIC
access to both PCIe Gen5 x4 paths, rather than doubling the physical link rate.

The switch uses DSCP 26 / traffic class 3 for RoCE and DSCP 48 / traffic
class 6 for congestion notifications. Queue 3 has ECN and PFC enabled,
with `egress-rate-queue3=200G` matching the physical rate. Queue 6 uses strict
priority. The four ports trust Layer 3 markings with `trust-l3=keep`, and
LLDP DCBX is enabled. Buffer allocation uses RouterOS's automatic settings.
See [MikroTik's RoCE guide](https://help.mikrotik.com/docs/spaces/ROS/pages/189497483/Quality%2Bof%2BService).

Verify jumbo frames with `ping -M do -s 8972` on both interfaces in both
directions between every pair of hosts. Successful small pings alone do
not establish jumbo-frame readiness.

After network changes, validate bidirectional RDMA on all six host pairs and
both interfaces, then run the existing b12x ROCEnante correctness tests with
four ranks, including CUDA graph replay, before restarting the model server.

### Validation results, 2026-09-18

After rebooting all four hosts with their final cabling connected, all
24 directed jumbo-ping paths passed. Every host pair sustained
196.02–196.04 Gb/s combined in each direction, using both interfaces
simultaneously. Each `ib_write_bw` process used one QP, 64 KiB messages,
RDMA MTU 4096, traffic class 106, and a seven-second duration. Disjoint
pairs were tested concurrently. Switch counters recorded no dropped packets.

The four-rank ROCEnante suite passed 67 cases per rank after reboot,
including graph replay and both-interface striping. One oversized-gather
case was skipped because it exceeds that fixture's configured capacity.
Static addresses, MTU 9000, DSCP trust, and priority-3 PFC persisted on all
four hosts. Evidence is under `runlogs/spark-switch-20260918/`.

DeepSeek V4.1 TP4 with DSpark 7 started successfully after recovery, with
ROCEnante enabled for TP collectives. All 15 serving smoke requests passed:
text, vision, OCR, multiple images, prefix reuse, concurrency, chunked
prefill, changed prompt tails, longer decode, and default reasoning.
The repeated 12,119-token prompt reused 11,776 cached tokens.

The four-rank ROCEnante benchmark also passed its before/after timing
correctness checks. CUDA-graph all-reduce timings, in microseconds, were:

| Payload | Before reboot | After reboot |
| --- | --- | --- |
| 8 KiB | 18.0 | 11.9 |
| 32 KiB | 43.6 | 17.5 |
| 64 KiB | 80.5 | 24.2 |
| 256 KiB | 274.2 | 55.1 |
| 1 MiB | 1043.6 | 166.7 |

These are collective microbenchmarks, not model token-throughput results.
Both runs used BF16, both interfaces, 40 samples in four interleaved blocks,
and 20 operations per graph. The reported statistic is the median of the
maximum latency across ranks. Raw receipts are `bench200g-dual.json` and
`bench200g-post-reboot.json` in the evidence directory.

### Recovery after QSFP cable changes

Before reboot, links reported 200 Gb/s but actual payload throughput was
only about 13 Gb/s per interface. Traffic classes 0 and 106 performed
equally poorly, with no switch drops or congestion notifications. Rebooting
with the final cabling connected recovered full throughput, consistent with
[firsthand Spark reports](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/10).

Testing mixed pairs during recovery isolated the cap to transmission from
unrebooted hosts: Tachyon and Chroniton could receive at 196 Gb/s while
still transmitting at about 26.5 Gb/s combined. All hosts recovered after
their respective reboots.

After changing QSFP cabling, test actual payload bandwidth on both
interfaces simultaneously in both directions. If the link reports 200 Gb/s
but an interface is capped around 13 Gb/s, reboot the affected endpoints
with their final cabling connected and retest before changing switch QoS.

Hardware forwarding is active on all four switch ports; bridge IP firewall
processing is disabled. RouterOS's published Prestera configuration does
not document a cut-through toggle for this model. Queue priority and ECN
can improve latency under congestion, but should be evaluated separately
from the payload-throughput recovery described above.
