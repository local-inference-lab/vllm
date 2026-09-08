# Kimi-K3 verification optimization applicability

Status: **research-only**, except for explicitly qualified entries below.
The review date is 2026-09-08. The reference serving composition is vLLM
`62468755f795`, B12X `04f246d7e5e6`, and LMCache `9f8514c680e7`.
The packed-reader composition is vLLM `fa6ea71c01fd`, B12X `0bf9f177b237`,
and the same LMCache backport. Configuration and measured conditions are in
[Kimi-K3 DFlash2 serving qualification](kimi_k3_dflash2_qualification.md).

An all-state collection completed at 06:02:56 UTC covers metadata and bodies
for 75,914 PRs in seven repositories. Indexed shortlists, 14 selected discussion
snapshots, and complete review-thread snapshots for B12X #311 and vLLM #644
support this applicability review. This is collection coverage, not a claim
that every implementation received a line-by-line review. The raw corpus is
not part of this repository.

The related `local-inference-lab/llm-inference-bench` inventory covers 11 PRs.
Its #9 handles missing timestamps and a retrieval gate. Request-owned throughput
is addressed by [benchmark #16](https://github.com/local-inference-lab/llm-inference-bench/pull/16):
zero client output cannot borrow another request's throughput, and stream
totals do not use a server-wide counter. Three upstream regressions pass;
the resume-capable deployment passes four cases and discards saved
`prometheus_fallback` rows.

## Packed attention and target verification

| PR | Mechanism and relevance | Disposition |
| --- | --- | --- |
| [B12X #311](https://github.com/local-inference-lab/b12x/pull/311) | Balance live chunk ranges, optionally retain FP32 split partials, vectorize packed KV staging and shared-memory value loads, optionally pack queries. | **Implemented; kernel qualified** in the packed-reader composition. Seventeen GPU regressions pass. The serving candidate enables the vector path while keeping static ranges, BF16 partials, and software residual reconstruction, preserving baseline arithmetic. |
| [lab vLLM #644](https://github.com/local-inference-lab/vllm/pull/644) | Adapter controls for the packed reader and query gather, plus cache-resume correctness. | **Implemented; kernel qualified** for the three adapter commits present in the packed-reader composition. The reference serving tree has the initial packed reader but lacks these controls, even though its environment requests them. The frozen composition passes 47 adapter tests and the two-GPU gather test. The head-padding port at `2c99648d7185` passes 63 adapter tests overlaid on the frozen runtime; this is module correctness, not standalone PR performance. |
| [B12X #271](https://github.com/local-inference-lab/b12x/pull/271), [lab vLLM #565](https://github.com/local-inference-lab/vllm/pull/565) | Reuse KV loads across four verify queries. | **Implemented for ordinary E4M3 KV; unsupported for the packed serving path.** The packed cache plans return `uint8`, while the four-query-plan gate requires `float8_e4m3fn`. Trace evidence confirms one query per packed-reader CTA. A packed multi-query kernel remains a structural candidate. Sparse-history policies from these PRs are excluded by the precision contract. |
| [B12X #312](https://github.com/local-inference-lab/b12x/pull/312) | Guard online-softmax rescaling and physical-page byte offsets. | **Qualified conditions**: the imported reader includes the S4 return-state correction. Its late-maximum tests pass. A separate frozen-candidate check at byte offset 2,149,244,928 produces bit-identical output and LSE versus low pages. |
| [sglang #36821](https://github.com/sgl-project/sglang/pull/36821) | Fused convolution/gating/KDA verification writes replay operands instead of every state snapshot; commit replays the accepted prefix. | **Research-only**. Transferable to SM120 through the Triton design, but requires a Kimi accepted-state/cache-restore integration. Trace measures about 0.36 ms/step in recurrent KDA and 0.36 ms in convolution; the entire model cannot gain the PR's per-kernel ratio. |
| [sglang #38399](https://github.com/sgl-project/sglang/pull/38399) | ReplaySSM benchmarks include the flush phase and advancing ring position. | **Adopted measurement constraint**. A replay benchmark must time a complete flush cycle; measuring only phase zero overstates gains. |
| [sglang #37587](https://github.com/sgl-project/sglang/pull/37587) | Pad hybrid verify batches in complete request groups using an LCM alignment. | **Research-only applicability review**. This protects physical/logical recurrent-row alignment under DP/TP MLP synchronization. It does not imply adding eight target rows because a draft was trained with eight rows. |
| [sglang #33291](https://github.com/sgl-project/sglang/pull/33291) | Deterministic decode/verify/rollback with accepted-state reconstruction. | **Research-only**. Useful state-contract design; deterministic kernel policy and its performance must be assessed separately from speculative acceptance. |

The packed-reader trace also exposes a TP9-specific launch cost not removed by
the imported split policy. Ninety-nine effective heads are padded to 104 by
the eight-head dense adapter; the packed reader executes six full 16-head tiles
and an eight-head remainder in two successive launches. The packed-reader adapter
pads to 112, executes one launch, and removes the padding before DCP reduction.
Six four-row cases at 2,048/8,192/16,384 local tokens preserve every valid output
and LSE byte. With vector loads, one call measures approximately 108–213 µs
versus 296–426 µs in the reference reader. These are isolated kernel timings.

## QSRT expert execution and communication

| PR | Mechanism and relevance | Disposition |
| --- | --- | --- |
| [B12X #328](https://github.com/local-inference-lab/b12x/pull/328) | Shorter trellis address/decode chain, cross-tile prefetch, one coupled-Hadamard rotation per token. | **Implemented in the reference source**, including the epilogue scratch-span repair. Do not count these as additional gains. The PR records incomplete sanitizer coverage for its prefill scheduling switches; their presence does not establish that qualification. |
| [B12X #339](https://github.com/local-inference-lab/b12x/pull/339) | Posted-write paired gather, register top-16 selection, clipped output widths, fused quantization shape selection, direct BF16 route-sum store. | **Implemented in the reference source** for gather/selection/width/store changes. Shape-selective fused quantization remains disabled: the recorded four-row measurements did not establish a winning shape. |
| [B12X #335](https://github.com/local-inference-lab/b12x/pull/335) | Posted-write query gather and LSE reduce-scatter on PCIe. | **Implemented and previously qualified** under `B12X_PCIE_DCP_A2A_TRANSPORT=push`. Rank-zero traces still include collective latency; it includes peer waits and is not a direct fabric-bandwidth measurement. |
| [B12X #334](https://github.com/local-inference-lab/b12x/pull/334) | Pipeline-depth and CTA-count controls for the QSRT MoE kernel. | **Research-only controls; measured alternatives rejected**. The PR discussion reports ratios 1.010 for three stages, 1.157 for two, and 1.001 for two CTAs/SM at M=4. The direct table is valuable; deeper pipelines can evict it. Do not repeat these settings as unmeasured improvements. |
| [B12X #300](https://github.com/local-inference-lab/b12x/pull/300) | Bulk-load the shared direct trellis reconstruction table. | **Implementation inventory**. Compare actual source/compiled objects before attributing an additional benefit; direct-table use is already enabled in the serving environment. |
| [B12X #280](https://github.com/local-inference-lab/b12x/pull/280) | Parallelize independent token-route packing for GLM NVFP4 M8. | **Research-only transfer**. Independent work decomposition is relevant, but GLM NVFP4 activation quantization is not the QSRT BF16-activation contract. No copied GLM throughput claim applies to Kimi. |
| [B12X #279](https://github.com/local-inference-lab/b12x/pull/279) | Programmatic dependent launch between PCIe all-reduce and a normalization/residual consumer. | **Research-only transfer**. Kimi uses AttnRes, not the PR's MHC consumer. Publication ordering must be preserved when adapting producer release and consumer wait. |
| [B12X #198](https://github.com/local-inference-lab/b12x/pull/198) | Exact paired top-16 routing, narrow-output GEMM stride handling, and vocabulary-parallel selection. | **Architecture reference**. The TP16 measurements do not predict TP9 speed. Exact routing and stride checks carry over; several related mechanisms already exist in the TP9 source. |
| [lab vLLM #592](https://github.com/local-inference-lab/vllm/pull/592), [#586](https://github.com/local-inference-lab/vllm/pull/586) | Prefetch upcoming dense weights on a side stream during other work. | **Implemented in the reference source**. Summed prefetch kernel time overlaps other streams and must not be subtracted directly from step latency. |
| [FlashInfer #4361](https://github.com/flashinfer-ai/flashinfer/pull/4361) | Task-scheduled persistent MoE kernels, fused epilogues, PDL. | **Research-only architecture reference**. The documented kernel families target SM100/103 and do not consume QSRT K2 payloads. Scheduling ideas can transfer; replacing the quantization format would violate the contract. |

The B12X source exposes `B12X_W4A16_M8_CTA_THREADS=512` at commit
`04f246d7e5e6`. It retains each warp's K-slice and fold order while increasing
N parallelism. The reference trace uses 256 threads. The extent benchmark in
[B12X #339](https://github.com/local-inference-lab/b12x/pull/339) allocates output
using the configured dtype, including BF16 route-sum output. Revision
`ab269462fcfc` already contains that allocation contract.

Sixteen actual-weight cases (two widths, four row counts, two CTA sizes) have
matching cross-thread output digests and exact graph/eager results. Three of
four replay intervals show service activity. The rank-0 512-thread interval
has unchanged counters and idle endpoint gauges, but its 256-thread reference
overlaps service. The thread-count comparison remains **research-only**, and
512-thread execution is not selected. Raw timing samples, launch attestation,
GPU operating state, and overlap evidence are published with #339 in
`docs/evidence/kimi_qsrt_extent_diagnostics.json`.

## Cache and runtime state

| PR | Mechanism and relevance | Disposition |
| --- | --- | --- |
| [lab LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44) | Respect physical page stride in engine-driven KV transfers. | **Qualified** in the reference serving candidate and retained in the packed-reader composition. This is the established DFlash2 output-corruption fix. |
| [lab LMCache #63](https://github.com/local-inference-lab/LMCache/pull/63) | Production-size padded-page transfer regressions and optional layout-audit suppression. | **Qualified API cases** on the `dev`-based PR: 103 transfer/layout tests pass with native extensions built from that checkout. The serving composition uses the release-line backport; this does not qualify a full `dev` deployment. |
| [lab vLLM #707](https://github.com/local-inference-lab/vllm/pull/707) | Stable, single-row compaction of GLM's 2,051 sparse selections. | **Unsupported as a direct Kimi optimization**. Kimi's target MLA is dense. Stable-order metadata compaction is a useful design principle, but sparse selection would change target attention semantics. |
| [FlashInfer #4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) | Retain workspaces referenced by CUDA graphs after cache growth or clearing. | **Adopted lifetime constraint**. Caller-owned storage and graph-pinned addresses are required for every candidate. The B12X serving path must be checked on its own lifetime implementation rather than assuming this FlashInfer patch applies. |
| [FlashInfer #5012](https://github.com/flashinfer-ai/flashinfer/pull/5012) | Register and compile fused GDN decode for SM121 separately from SM120. | **Architecture reference**. The deployment uses SM120; SM121-versus-SM120 registry fixes are not a direct speedup. Avoid comparing the PR's composable Torch baseline with optimized serving kernels. |
| [lab vLLM #709](https://github.com/local-inference-lab/vllm/pull/709), [lab LMCache #62](https://github.com/local-inference-lab/LMCache/pull/62) | Publish target, recurrent, and draft auxiliary checkpoints only after all ranks finish; pin destinations and transfer leases through cancellation. | **Research-only integration candidate**. These two APIs form one ownership contract and do not replace the aligned connector used by this deployment. Their GLM TP4 transfer results do not establish Kimi TP9 acceptance parity. Relevant to full-history draft restoration and concurrent cache work, rather than the measured steady-state MLA cost. |
| [upstream vLLM #55818](https://github.com/vllm-project/vllm/pull/55818) | Bound sliding-window physical allocation while retaining absolute logical block-table positions and speculative extra retention. | **Research-only allocator reference**. Preserve logical positions and null blocks when separating DFlash2's sliding-window and full-history groups. The serving arm already admits a one-million-token request capacity; no blanket allocator replacement is justified by the decode trace. |
| [upstream vLLM #55627](https://github.com/vllm-project/vllm/pull/55627) | Exact Mamba2 prefill/decode bits through FP32 state and replay from chunk boundaries. | **Unsupported as a direct optimization**. The PR excludes speculation, TP above one, prefix caching, and connectors, and leaves CUDA graphs. Its reproducibility contract is informative; enabling it would not be a valid Kimi KDA speed change. |

## Precision decisions

1. Retain QSRT K2 payloads, finite E4M3 reconstruction values, scale semantics,
   coupled transforms, and target attention visibility.
2. Qualify bit-preserving staging, padding, and query transport separately from
   split reassociation or arithmetic changes.
3. FP32 partials improve measured reference error in the sampled packed-reader
   cases, but the resulting bits differ. They are available for research and
   are not enabled in the bit-preserving serving candidate.
4. Keep hardware residual reconstruction disabled in that candidate: it repairs
   a software treatment of E4M3 subnormals but changes arithmetic and needs a
   separate precision qualification.
5. Do not infer model speed from a PR's isolated kernel ratio, foreign hardware,
   different speculative depth, or unoptimized reference baseline.
