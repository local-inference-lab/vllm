# UVA token-table update qualification

Status: **implemented; qualified for the recorded module tests and update
measurements**. The measurement date is 2026-09-08.

The V2 runner keeps its request token table in pinned host memory, exposed to
the GPU through Unified Virtual Addressing (UVA). A staged update supplied its
contents through a temporary CUDA tensor before the write kernel stored them
into that host-backed target. The implementation from
[upstream vLLM #55819](https://github.com/vllm-project/vllm/pull/55819), revision
`8d96e8e2498a`, instead stages those contents in reusable UVA buffers.
GPU-backed targets retain the H2D contents path.

Each contents slot grows to a power-of-two capacity. Slot rotation uses the
runner's existing maximum in-flight batch bound. The caller must retire GPU
readers before reusing a slot; buffers are not overwritten or released while
their permitted consumers remain in flight. The destination token-table
address and write-kernel ordering are retained.

## Recorded Kimi geometry

The source composition is vLLM `f9ac0209a2ed`, B12X `0bf9f177b237`, and
LMCache `9f8514c680e7`. Its frozen source ID is `6e5fff3911585dfd`. This
composition retains the Kimi CPU token-table accessor, a separate integration
extension not required by the staged-write implementation.

The fixed-input comparison runs on physical GPU 4, an RTX PRO 6000 Blackwell
Max-Q. Both arms use the same runtime and an `int32[4, 1048576]` host-backed
target. Four updates warm both rotating slots and the allocator/kernel caches.
Thirty samples alternate reference and candidate order. Timing starts after
`stage_write()` and an initial device synchronization; it includes
`apply_write()` and completion synchronization.

| Rows | Elements per row | H2D contents, median µs | UVA contents, median µs |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 35.863 | 24.046 |
| 1 | 4,608 | 239.887 | 126.296 |
| 4 | 4,608 | 835.425 | 408.067 |
| 1 | 65,536 | 2,912.337 | 1,466.932 |
| 1 | 131,072 | 5,783.501 | 2,895.931 |
| 1 | 1,048,576 | 46,530.364 | 22,916.654 |

All destination bytes match. The record in `uva_staged_token_updates.json`
contains per-case digests and every timing sample. Before and after the
accepted interval, the serving endpoint reports no running/waiting requests
and unchanged token/completion counters. GPU mode is P1, graphics clock
2,580 MHz, memory clock 16,365 MHz, and temperature 59°C at both boundaries.
An interval with concurrent service activity is excluded.

These are synchronized update-completion measurements. They do not measure
model throughput or establish a 50% decode gain. Token-table admission and
prefill updates differ from a steady decode step with no staged host writes.

## Correctness and source scope

The serving-composition and public-port modules pass 12 single-GPU tests from
`tests/kernels/core/test_uva.py`. They cover host/device write visibility,
growing and shrinking contents, list/NumPy/Tensor inputs, all supported dtypes,
and in-flight consumers. The public port has identical buffer code except
that the composition also exposes its inherited CPU token-table accessor.
The GPU-write test explicitly synchronizes its device before host assertions;
mapped storage does not make CPU reads wait for asynchronous GPU execution.

The deployed composition passes seven cache/conversation checks, including
9,216- and 4,608-token external cache reuse, and four concurrent request
canaries. All sampled log probabilities are finite. These checks qualify
serving correctness under their request conditions, not a throughput gain.

A four-iteration rank-zero Torch decode capture at approximately 64 Ki context
measures target graph replay at 30.840 ms median versus 30.848 ms in the
reference composition, and draft proposal at 2.281 versus 2.285 ms. Steady
target execution is preserved in these captures. Input-preparation scope
medians are 0.033 and 0.018 ms respectively; the traces contain profiler-start
outliers and use different sampled continuations. They do not establish a
model-throughput improvement. The qualified UVA-write composition is retained
in service.

The benchmark is `benchmarks/kernels/benchmark_staged_uva_writes.py`. Supply
the unmodified `buffer_utils.py` as `--reference`. Run in an environment with
the matching vLLM native extension. Correctness reads occur outside the timed
boundary. No model weights or private request data are required.

```bash
python benchmarks/kernels/benchmark_staged_uva_writes.py \
  --reference /path/to/reference/buffer_utils.py \
  --output staged-writes.json --iterations 30
```

## Allocation-policy boundaries

The Kimi deployment's LMCache pool is one 48 GiB POSIX shared-memory object
registered in each of nine worker processes. Its cross-process ownership and
byte-offset contract remain unchanged. QSRT expert weights are prepared into
separate CUDA allocations. The InstantTensor loader's host-buffer retention
option is disabled in the observed worker environments.

A separate 4 KiB probe uses the installed vLLM CUDA-view helper with default
pinned, mapped pinned, write-combined mapped, and managed allocations. The
three pinned modes preserve CPU/GPU update visibility. The managed allocation
reports `is_pinned=false`; the helper's unpinned fallback copies into another
host allocation, and neither direction's later writes reaches the original
buffer. Managed allocation is therefore **unsupported as a drop-in UVA
replacement for this native runtime**. That finding does not assert that CUDA
managed memory cannot support other designs.

Write-combined memory permits CPU reads but makes them inefficient, so a
CPU-write/GPU-read staging result does not qualify a CPU-readable cache pool.
Managed memory also changes residency and migration behavior; CUDA IPC does
not export managed allocations. See the
[CUDA host-allocation API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902)
and [Unified Memory IPC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#inter-process-communication-ipc-with-unified-memory).

Removing `pin_memory()` globally is also outside this change. The rationale
in [upstream #37006](https://github.com/vllm-project/vllm/pull/37006) was
corrected during review, and pinning was restored by
[#45424](https://github.com/vllm-project/vllm/pull/45424) to avoid possible
GPU/CPU stream synchronization. The pinned allocator's recycling behavior
must be distinguished from direct `cudaHostAlloc`/`cudaFreeHost` calls.
