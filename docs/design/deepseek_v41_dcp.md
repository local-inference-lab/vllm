# DeepSeek V4.1 native B12X DCP (experimental)

This feature branch includes local-inference-lab `dev/jovian-judgement`
through `82250ef0135b76dd60239773812e7947f2d25596`. Native DCP exchange and
sharded-index preparation require the companion yatesdr/b12x feature branch
through `e811430fe08d34793a6fdbb30c06254735522b8c`. No upstream PR is submitted.

## Layout and execution

- Compressed main KV is owner-sharded. Ownership uses original-token stripes
  divisible by the compression ratio; the tested stripe is 128 tokens.
- In the currently qualified path, index KV, SWA and private compressor state
  are replicated. This is a safe implementation baseline, not proof that every
  state is inherently unshardable. Existing native global Full/Reindex/Reuse
  candidate selection is unchanged.
- Tuple packing keeps replicated index and sharded main MLA in distinct full
  groups. Stripe validation uses physical sharded pages, not the replicated
  compressor ring's scheduling granularity. Shared exchange preparation is
  exposed by one stable layer helper in both preparation stages.
- Global selected IDs are masked/remapped into each owner's local main cache.
- Native B12X PCIe head gather expands TP-local queries within the DCP group.
  Native compressed MLA returns partial output and natural-log LSE. B12X LSE
  reduce-scatter returns each TP rank's own heads.
- Replicated SWA and the attention sink enter the attention union only on DCP
  rank zero. This prevents duplicated probability mass.
- One serial channel is shared across layers, with a 256-row capacity. Prefill
  attention is chunked within that capacity. The current layout does not yield
  a fourfold increase in total KV capacity because index and SWA are replicated
  and the single shared block pool introduces cross-group stride padding.
- Owned selected IDs are compacted in global selection order with true local
  lengths. DCP WO replay uses opt-in native variable-row capacity declarations;
  it does not pad rows, change GEMM precision, or create a serving-time plan.

### All-index sharding under qualification

`VLLM_DS41_DCP_SHARD_INDEX=1` changes all four DS4.1 index caches from the
replicated baseline to striped DCP ownership. The feature is deliberately
all-or-nothing: sharding only the three early index caches increases the current
single-pool stride because the remaining replicated index cache still sets the
group width.

Each index layer scores its local cache and selects 512 local candidates. An
exact all-gather/global-top-k merge returns global compressed-token IDs. At the
candidate source layer, the same construction merges 2,048 coarse blocks before
expanding them to the global 16,384-token reuse set. Only the rank owning the
globally newest block forces that block into its local selection. Later layers
compact the global reuse set into their local shard without changing its order,
score it locally, and repeat the exact global merge. SWA ownership and numerical
precision are unchanged.

The bounded four-rank oracle passes exact 512-token and 2,048-block selection,
candidate ownership and three CUDA-graph replays with stable allocation on all
four SM120 GPUs. Focused cache-spec, local-length, source-selection and metadata
tests pass. This is component qualification only; full-model startup, output
parity, prefix reuse, long-context admission and performance remain required
before enabling the option by default.

## Tests performed on cn4

Four RTX PRO 6000 Workstation GPUs, SM120, existing 300 W power limits and
13365 MHz memory clocks preserved; cn3 was untouched.

```bash
/opt/venv/bin/python -m pytest \
  tests/v1/attention/test_b12x_v41_ced_metadata.py -k dcp -q
# 14 passed: DCP2/4 ownership, nonowner queries, padding, selected-ID mapping.

/opt/venv/bin/python -m torch.distributed.run --nproc-per-node=4 \
  --master-port=29564 tests/distributed/deepseek_v41_dcp_oracle.py
```

The four-rank component oracle passes decode and extend, eager and three
CUDA-graph replays each. It compares against independent packed-cache/Torch
attention semantics, includes empty owners and a physical address span above
2 GiB, and asserts stable allocated memory during replay. Maximum observed
absolute output difference was 0.007812 (BF16 output, rtol 0.03/atol 0.01).
This does not qualify full-model output, long-context quality or performance.

Additional regression gates: 15 ownership/metadata graph tests, five shared
channel/declaration tests, and 12 allocator/configuration tests passed. The
allocator suite includes target-plus-draft layouts and existing GLM cases.
The final startup-reservation regression run passes 18 focused cases, including
all declared index regimes after workspace locking. Full-model API checks pass
arithmetic, explicit history, vision and repeated-prefix reuse (27648 cached
tokens of a 28725-token request). Initial no-speculation LIL decode rates are
C1 100.3 and C4 221.2 aggregate tok/s; these are not DSpark results or a matched
DCP1 comparison.

The initial K5 image served arithmetic/history/cold-prefix requests but failed
warm-prefix replay: a 1078-token suffix caused an undeclared WO packer JIT.
The native capacity fix passes four singleton/production-grouped geometry
regressions, including that exact suffix, independent exact-M comparisons and
frozen graph replay. The corrected p12 full-model K5 image passes arithmetic,
explicit history, vision and cold/warm prefix replay with 27648 cached tokens.
The failure is not hidden by disabling the JIT guard.

The short LIL v0.6.2 K5 sweep measured C1 121.1 tok/s, C4 208.6 aggregate tok/s
(20-second cells, requested context zero, temperature 1, reasoning high).
Effective acceptance lengths were 2.14/1.89; no request errors were reported.
One uncached 32770-token sample measured 1725 client / 1732 server prefill tok/s
and 19.002s TTFT. This is a failed performance gate, not a matched DCP1 speedup.
The engine reports an 8038903-token hybrid pool, 10.25 GiB per rank. Neither
the estimate nor this unmatched run demonstrates a capacity gain over DCP1.

The generic upstream GSM8K chat stop strings cut off reasoning on `Question`:
the initial 16-question, 8-shot subset returned 9 correct/seven empty answers;
a full-response-logged replay returned eight correct/eight empty answers, all
empty responses explicitly stopped on `Question`. The same fixed subset with
generic stop strings removed returned 14 correct and zero empty answers.
This diagnostic does not establish full quality or deterministic DCP1 parity.

A separate 10-second worker-stack profile collected 994 samples, with 42.9%
inside MoE workspace-shape calculation and 43.9% inside memory requirements
(overlapping categories). Repeated CPU plan lowering is an optimization target;
the profile is not a GPU/communication latency breakdown or scored benchmark.
MoE workspace sizing now reuses only the current prepared family's envelope,
invalidating on parent/child preparation changes and holding only weak
references. Its focused CPU contract regression passes. No native math or
precision is changed.

The post-change profile shows workspace sizing 42.9% -> 0.2% of samples and
scratch-plan lowering 39.9% -> zero. One uncached prefill sample remained
essentially unchanged at 1743 client / 1751 server tok/s (18.801s TTFT); this is
not a throughput gain. Live Nsight counters, filtering samples to GR Active
>=95% to exclude the idle tail, show SM activity 14.2-15.4%, tensor activity
1.4-1.6%, DRAM reads 2.3-2.5% and PCIe RX/TX approximately 9.1-9.2% of metric
peaks. An underfilled/wait-heavy collective is a hypothesis, not a measured
per-kernel timing attribution.

The existing `B12X_PCIE_DCP_BLOCK_LIMIT=64` setting was compared against
the default 16-CTA cap. The oracle's 32 rows exercise all 64 CTAs in both
collectives, eager and three frozen graph replays, preserving the independent
reference tolerance and 0.007812 maximum BF16 error. This changes launch
capacity only, not native kernel math or precision. One uncached 32770-token
sample measured 1767 client / 1775 server tok/s and 18.54s TTFT, versus 1743
client tok/s at 16 CTAs. The approximately 1.4% difference does not establish a
gain from single samples. The serving configuration returns to 16 CTAs; no
additional benchmark sweep was run. Per-kernel gather/attention/reduce timing
is the next performance investigation, not more model-test coverage.

The p12 K5 instance completed a fresh 525000-token prompt with zero cached
tokens in 301.41s and returned the final record correctly. This qualifies
admission/final-record retrieval only, not long-range conversation quality.

## Serving configuration under qualification

### Profile-driven packed-KV encoder prefill

A bounded Kineto capture of two 4096-token worker iterations records 616 head
gathers and 616 LSE reductions on each rank. Their combined durations are
85-87% of summed kernel work; the kernel union is approximately 4.31s across
a 4.42s span. Sparse prefill attention is 2.8-4.3% of summed kernel work.
Instrumented durations are not uninstrumented benchmark results.

CED changes the traffic estimate: only 18 compressed encoder layers process
all prompt rows; the later 20 layers process compact queries. At 32K, encoder
head exchange alone reads 54 GiB of peer data per rank. The all-38-layer
estimate of 114 GiB is an upper bound, not the actual CED serving layout.

The new DCP4 encoder-extend path replicates packed 288-byte indexed records
after each KV-source write. Sources 2/8/14 reuse their replica until the next
source write. Global index selection and NVFP4 bytes are unchanged. Native
B12X attention uses the rank's original 16 heads, SWA and sink. CED and decode
retain the existing owner-sharded gather/partial-attention/reduce path.

Replica storage is declared before memory profiling. Sources 2/8/14 have
nonoverlapping encoder-layer intervals (2-7/8-13/14-19), so they share one
output buffer, accounted once by its transport declaration. One replica
plus one shared local staging slab require approximately 0.36 GiB per rank
at max-seqs4/540672 context. No full-model KV pool capacity gain is inferred.
A standalone one-CTA peer barrier follows completion of previous consumers;
another follows local staging before peer reads. This avoids unsafe per-block
reuse when live grid geometry changes, needs no host-patched graph epoch, and
keeps one stream-ordered channel. Page/record offsets remain Int64.
Startup preparation and model profiling may use different logical streams;
handoffs insert a CUDA stream dependency at the previous stream's tail,
including its attention consumers. No serving-time host synchronization is
added. Independent overlapping replay and microbatching are unsupported;
microbatching fails closed. Capture/replay must use the warmed logical stream.
The source cache's physical page stride is a runtime scalar: pooled/padded
pages need not be contiguous across page boundaries. Only semantic 288-byte
records are copied, never allocator padding.

The first native oracle passes byte-exact reconstruction, >2 GiB physical
pages, multiple live counts and serial graph replay without allocation growth.
The expanded oracle additionally covers changing CTA grids, cached prefixes
and recycled nonsequential page mappings. A serial-stream-handoff regression
is added after p15 rejected vLLM's preparation-to-profiling transition.
The handoff oracle and four attention/workspace declaration cases pass in p16.
p16 then rejected an incorrect FrozenMapping constructor in the shared-output
wrapper before model profiling. Its mapping-form correction passes one CPU-only
constructor/residency regression before loading weights. The corrected p17
image is healthy on cn4. Two uncached32770-token LIL samples measured6963
client /7084 server tok/s, TTFT4.706s, versus the p13 single sample1743/1751
and18.801s. A focused full-history prefix probe returned the final record:
cold4.102s/cached0, warm0.869s/27648 cached of28724 prompt tokens. One20-second
C1 cell measured111.131 tok/s,59.8 verifier steps/s and acceptance1.866;
decode has not established a gain. These are not matched current-dev DCP1
comparisons or full-model/long-range production qualification.

The p17 bounded trace contains initial67-token prefill plus one6-row decode
step. The decode timestamp window includes draft/sampling tails, not exact
CPU/GPU phase correlation. Each rank has38 native head gathers (~3ms summed)
and88 oneshot all-reduces (~4.6-6.6ms). Target/draft graphs are active. Active
PCIe links remain Gen4x16 at existing300 W limits; idle Gen1 is normal.

B12X offers an experimental opt-in posted-write head gather using the same
global-head output layout, IPC slab, per-CTA system barrier and graph epoch.
The declaration captures the transport mode; bind/replay does not read an
environment variable. Incoming loads bypass L1. The default remains pull.
The bounded head-gather-only oracle passes exact bytes at live1/6/32 rows
and three allocation-stable graph replays with mutated inputs and poisoned
outputs. Its plan-time CPU guard passes freezing and unsupported-geometry
rejection. This does not establish a serving speed gain.

The p18 push-gather image serves successfully. Its two6-row decode steps have
76 gathers totaling1.675ms on rank0 (~22us/call); the prior mixed-trace decode
window was ~80us/call. This is a scoped instrumented comparison. One20-second
C1 cell measured123.472 tok/s,55.260 verifier steps/s, acceptance2.234; verifier
throughput did not improve over p17. Plain TP all-reduce is now31.7% of rank0
summed kernel work (176 calls,11.628ms), motivating the next source change.

The opt-in B12X TP4 transport extends to plain DS4.1 BF16 widths5120/1280 at
1-8 rows. It reuses the existing four-shard IPC allocation and graph epoch,
uses posted writes/local L1-bypassing reads, and preserves the original
rotating-rank FP32 accumulation/one BF16 rounding. Other plain routes and the
default-off behavior are unchanged. A CPU routing/frozen-policy gate passes,
and the bounded four-rank oracle passes eager1/6/8-row cases and three fresh,
allocation-stable graph replays, byte-exact against both native pull and an
independent same-order FP32 sum. No serving speedup is claimed before baking
and measuring this identical source.

The p19 image includes both posted-write controls and serves successfully.
One20-second LIL C1 cell measures138.441 tok/s,61.018 verifier steps/s and
effective acceptance2.269, zero errors. p18 measured123.472/55.260; this is
not a repeated-median or matched DCP1 speed claim. The same seeded prompt
passes (finish=stop); the bounded capture contains two6-row target steps.
Plain-allreduce medians across ranks fall from47.8-56.2us to14.1-21.5us.
A12.290ms first-collective outlier on rank0 coincides with the other ranks
entering that first profiled step approximately12ms later. Profiler-entry
skew is a plausible explanation, not proof of production tail stability.
The hybrid logical KV pool reports7,724,715 tokens, rank0 budget9.85 GiB.
No new prefill/quality/context sweep was run for this decode-only change.
The expanded four-rank oracle also passes mixed small-push/large-pull
declarations sharing each runtime's graph slots/epochs: exact reference
outputs and three poisoned/mutated, allocation-stable replays. This targeted
serving invariant is not a full-model quality or long-running stability pass.

TP4/DCP4, 540672 context, 4096 batch budget, max-seqs 4, main/SWA pages 256/128,
SSD Engram, native prefix caching, decode graphs enabled, prefill graphs off.
Initial full-model bring-up disabled speculation to isolate native correctness;
the corrected serving instance uses DSpark K5. LMCache and startup search
(`enable_b12x_autotune=false`) remain disabled. Faster DCP serving, matched
current-dev DCP1 benchmarks, full model/long-range quality and LMCache DCP restore
remain required before calling this a production-qualified configuration.

LMCache must use a separate DCP-layout namespace and a chunk divisible by
the main cache's global block span (256 * 4 = 1024), not reuse DCP1 objects.
Do not enable it based only on the component oracle results.

PCP, multi-node DCP, overlapping independent channel replay and DCP sizes other
than 2/4 are not supported by this implementation. Human review, relevant model
evaluations and DCO confirmation are required before an upstream PR.

AI assistance was used in implementation and test development.
