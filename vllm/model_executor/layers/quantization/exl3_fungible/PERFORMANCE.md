# exl3_fungible — performance & instrumentation contract

Standing rule (operator directive, 2026-08-10): every change in this
package that can influence **correctness (KLD)**, **runtime memory**, or
**performance (prefill PP / token-gen TG)** must be:

1. **Performant by construction** — CUDA-graph-compatible hot paths (pure
   tensor ops on persistent buffers; no `.item()`/host branches/allocation
   in capture or apply paths), work moved off the forward path wherever
   possible (windowing in `step()`, policy on host, staging pre-quiesce).
2. **Instrumented and called out** — the landing commit/report states:
   - measured PP and TG delta vs `VLLM_FQ_ENABLE=0` baseline (same rig),
   - KLD probe result whenever weights/routing are touched
     (`tools/fq_probe.py` reference-leg methodology),
   - runtime memory delta (`torch.cuda.memory_allocated` steady-state;
     zero cudaMalloc in steady state per T7),
   - Prometheus metrics per 03-testing-validation §Instrumentation
     (`fq_swaps_total`, `fq_probe_kld`, per-tier occupancy, …) shipping
     with M2, not later.

Milestone gates remain the hard floors (M1: <0.5% decode overhead at cc8;
T7: within 1% of baseline between swap intervals; swap stall < 1 engine
step in atomic mode). Disk/HF space is explicitly not a constraint.
