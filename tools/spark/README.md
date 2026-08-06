# Live two-Spark vLLM development

This workflow reproduces the relevant `spark-vllm-docker` behavior without a
container. Tachyon builds native artifacts once; luxon receives those artifacts
and the live vLLM/B12X sources. The regular DeepSeek V4 Flash checkpoint runs as
TP=2 through vLLM's multiprocessing backend. Decode context parallelism defaults
to DCP=2, with MTP disabled.

Tachyon's live vLLM and B12X working trees are authoritative. External
dependencies, checkpoint revisions, and CUDA architecture remain pinned in
`versions.env`. Every build records the exact live source state and package
provenance in `.spark-artifacts/manifest.json`.

The cluster workflow uses `serve-ds4-flash-spark.sh`. The repository-root
`serve-ds4-flash.sh` remains the local workstation launcher.

## Initial setup

```bash
tools/spark/system-deps.sh both
tools/spark/network.sh configure
tools/spark/network.sh bench
tools/spark/bootstrap.sh --host local --recreate
tools/spark/build.sh all
tools/spark/sync.sh --all
tools/spark/bootstrap.sh --host luxon --recreate
tools/spark/verify.sh env
tools/spark/verify.sh nccl
tools/spark/model.sh download
tools/spark/model.sh sync
```

System setup enables user lingering on both hosts so the transient rank
services survive after SSH sessions close. Cluster preflight verifies it.

The official DSpark checkpoint is pinned separately. Fetch and mirror it with:

```bash
tools/spark/model.sh download dspark
tools/spark/model.sh sync dspark
```

All Python package operations use `uv` and `.venv/bin/python`. The scripts do
not replace the system Python, driver, CUDA toolkit, or system NCCL.
On arm64, NVIDIA publishes cuSPARSELt with an `sbsa` platform tag that
`uv pip check` does not recognize. The gate accepts that one metadata warning only after
confirming that the installed shared library is an AArch64 ELF.

## Development loop

`cluster.sh start` automatically mirrors tachyon's live vLLM and B12X source
trees to luxon before launching either rank. To synchronize without starting:

```bash
tools/spark/sync.sh --source-only
```

For vLLM C++/CUDA changes:

```bash
tools/spark/build.sh incremental
tools/spark/sync.sh --artifacts-only
```

Use `tools/spark/build.sh vllm` when a new distributable wheel and manifest are
required. Source sync always overwrites luxon's live mirror with tachyon's
working trees; environments, build products, and profiler output are excluded.
Set `SPARK_SYNC_SOURCE_ON_START=0` only when intentionally launching sources
that were synchronized separately.

To arm Torch profiling on both ranks, launch with `--profile`, wait for the
server, and bracket the workload with the profile actions:

```bash
VLLM_ENABLE_DSPARK=1 DCP_SIZE=1 NUM_SPECULATIVE_TOKENS=5 \
  tools/spark/cluster.sh start --profile
tools/spark/cluster.sh wait
tools/spark/cluster.sh profile-start
# Send the requests to profile.
tools/spark/cluster.sh profile-stop
```

Traces default to timestamped `rank-0` and `rank-1` directories beneath
`~/.local/state/vllm-spark/profiles`. Use `--profile-dir /absolute/path` to
choose another base directory. The `VLLM_TORCH_PROFILER_*` variables control
stack capture, shapes, memory, FLOPs, compression, and iteration scheduling.

## Serving

```bash
tools/spark/cluster.sh preflight
tools/spark/cluster.sh start
tools/spark/cluster.sh wait
tools/spark/verify.sh smoke
tools/spark/cluster.sh logs
tools/spark/cluster.sh stop
```

The initial profile uses 500K maximum context, four sequences, FP8 KV cache,
and 80 percent GPU-memory utilization. DCP uses NCCL A2A through 64 scheduled
tokens and NCCL AG/RS for larger batches. This favors A2A's lower collective
count during decode and AG/RS's bandwidth efficiency during prefill. The B12X
CUDA-IPC A2A transport is disabled because the ranks are on separate hosts.

Override launcher settings with environment variables such as `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `DCP_SIZE`, and
`DCP_COMM_BACKEND`. Set `DCP_COMM_BACKEND=ag_rs` to compare pure AG/RS, or set
`VLLM_DCP_A2A_MAX_TOKENS=0` for pure A2A. To enable MTP, set
`VLLM_ENABLE_MTP=1` and optionally `NUM_SPECULATIVE_TOKENS`. One-million-token
context remains an explicit validation step.

DSpark uses the official checkpoint's embedded draft module and requires DCP
to be off. Its checkpoint block size supports at most five draft tokens. The
Spark launcher defaults the draft block to B12X sparse attention; set
`DSPARK_DRAFT_ATTENTION_BACKEND=auto` to use vLLM's automatic selection:

```bash
VLLM_ENABLE_DSPARK=1 DCP_SIZE=1 NUM_SPECULATIVE_TOKENS=5 \
  tools/spark/cluster.sh start
tools/spark/cluster.sh wait
```

When DSpark is enabled, the launcher profiles KV-cache memory instead of
reusing the regular model's explicit cache allocation. Set
`KV_CACHE_MEMORY_BYTES` only after profiling the DSpark configuration.

`tools/spark/network.sh rollback` removes only the two NetworkManager profiles
created by this workflow; it never changes the management LAN.
