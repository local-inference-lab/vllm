# Kimi-K3 DFlash2 serving qualification

Status: **implemented and qualified for the conditions recorded here**.
The measurement date is 2026-09-08. The source-locked serving composition is
vLLM `fa6ea71c01fdaef3a664d083a536cee00a848b1d`, B12X
`0bf9f177b237`, and LMCache `9f8514c680e77f8d5d026e05037bc29b603a5bf4`.
The eight-row serving configuration has frozen source ID `5ee97cdc6d995067`.
It combines the DFlash2 port and rejection guards in
[vLLM #704](https://github.com/local-inference-lab/vllm/pull/704) with the
packed-reader integration of
[vLLM #644](https://github.com/local-inference-lab/vllm/pull/644) and
[B12X #311](https://github.com/local-inference-lab/b12x/pull/311).
These are composition measurements, not a claim that any one PR independently
produces the whole serving result.
The 99-to-112 head padding and DCP self-copy avoidance from the measured
vLLM candidate are also ported to PR #644 at `2c99648d7185`. Its 63 adapter
tests pass with those modules overlaid on the frozen serving runtime; that
check establishes module correctness, not standalone PR serving performance.

## Configuration and precision contract

| Property | Qualified setting |
| --- | --- |
| Hardware | Nine RTX PRO 6000 Blackwell GPUs; TP9/DCP9 |
| Target | Kimi-K3, native QSRT K2 routed-expert payloads |
| Target KV | `fp8_ds_mla`, 656-byte records |
| Draft | `lightseekorg/kimi-k3-dflash2`, colocated and TP-sharded |
| Proposals | Three, sampled with the candidate selector |
| Draft query width | Eight rows: one anchor and seven mask rows |
| Target verification width | Four rows |
| Draft KV | FP8, replicated bounded window 4,608 |
| Auxiliary stream | Target prefix-sum stream; taps 19, 37, 66, 78, 90 |
| Maximum model length | 1,048,576 tokens |
| Packed reader | Vector loads, static split ranges, BF16 partials and queries |
| Residual reconstruction | Software path retained; hardware override disabled |

The target weight payload, KV precision, visible history and valid-head
arithmetic are retained. Selected packed-reader changes preserve output and
LSE bytes in the recorded fixed-input comparisons. Balanced FP32 partials and
hardware residual reconstruction are separate numerical choices and are not
enabled in this composition.

`VLLM_DFLASH2_FULL_BLOCK=1` selects the trained query width. K remains three:
only the first three mask rows produce proposals. The common lookahead property
reserves `max(K + 1, draft_query_rows)` KV slots beyond scheduled target tokens,
so the eight-row arm reserves eight. Both scheduler allocation and warmup use
that property. It does not require eight additional target verification rows
or an independent Mamba padding adjustment.

The draft's full-history layer remains restricted by the uniform 4,608-token
window. This is a documented approximation of the draft, not full fidelity to
the authors' attention pattern. Per-layer full-history retention needs separate
KV-group, restoration and memory qualification. The target's rejection sampler
receives the actual proposal distribution.

## Restored-state corruption

The deployed engine-driven transfer path omitted the physical page stride when
constructing gather/scatter descriptors. A draft MLA page contains
`1536 * 576 = 884736` logical bytes, but the shared hybrid pool gives it a
1,007,616-byte physical stride. Tight-page addressing can read or overwrite
another cache group's page.

A reproduced restore had a finite input to target layer 13 and non-finite
BF16 convolution state. The first 64 bytes of that state matched a draft cache
object at byte offset 2,260,992; the recurrent-state prefix matched the same
object at 2,316,288. The 55,296-byte separation equals the convolution-state
size. Raw request text and tensor values are not published.

The native stride correction is covered by
[LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44). The
deployment backport is `9f8514c680e7`. A namespace-only control was insufficient:
a cold request succeeded, then a 9,216-token restore corrupted target state.
Correct stride handling is required in addition to namespace separation.

Undefined verifier ratios were a second defect. The block verifier could
approve every proposal and emit token zero (`!` for this tokenizer). Invalid
blocks now discard all proposal rows and sample the target directly. A finite
first row is also discarded when a later row invalidates the block, preventing
an incorrect residual subtraction. Sequential verification retains its valid
prefix and recovers at the invalid row. These guards do not repair corrupted
target KV; the transfer fix supplies that correction.

Configuration-specific namespace selection must be explicit. Merely skipping
registration leaves a previously accepted fingerprint mapped to its shared
namespace. Use a namespace derived from the complete configuration fingerprint
and keep entries from faulty transfers quarantined. The stock and DFlash2
deployments in this record expose 16 and 20 cache groups, respectively, so
their cache entries are not interchangeable.

## Numerical and serving checks

| Check | Result and scope |
| --- | --- |
| Rejection sampler GPU suite | 63 pass, including invalid distributions at every proposal position |
| DFlash2/DSpark model CPU suites | 42 pass |
| Native padded-page transfer cases | Four pass; independent gather/scatter, two geometries, adjacent pages and padding |
| Transport/recurrent/async suites | 65 pass; one unregister mock failure also occurs on the unmodified baseline |
| Packed MLA GPU cases | 17 pass, including graph replay and late-maximum rescaling |
| vLLM MLA adapter | 47 pass, including 64/99-head packed-query equivalence |
| Packed query gather | Two-GPU eager and graph case passes |
| High physical page | Byte offset 2,149,244,928 matches low-page output/LSE exactly |
| Sampled cache and conversation checks | Seven pass, including 9,216- and 4,608-token external reuse; finite logprobs |
| Four concurrent request canaries | All return their own code; finite logprobs |

The physical-page regressions and transfer-audit suppression are also published
against LMCache `dev` in
[LMCache #63](https://github.com/local-inference-lab/LMCache/pull/63). Its
103 transfer/layout tests pass with matching native extensions built from that
checkout. This is a separate API qualification, not a deployment of `dev`.

The high-page and packed-reader tools and numerical records are published
with B12X #311. The tests establish their stated numerical and state contracts;
they do not establish equality of every sampled model continuation.

## Decode measurement

The client is a frozen llm decode bench 0.6.2 source with SHA-256
`053989edff8c9c93e2b96e61342b2ffbd9851e03deba17e6d3fc96fcd6694c1e`.
It uses the default mathematics-generation prompt, unseeded default sampling,
one concurrent request and 30-second cells. The reference composition uses
vLLM `62468755f795`, B12X `04f246d7e5e6`, and the same LMCache stride backport.

| Requested context | Reference tokens/s | Packed reader, four draft rows | Selected eight draft rows | Reference steps/s | Selected steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 71.43 | 71.05 | 78.29 | 31.23 | 31.75 |
| 16 Ki | 61.14 | 64.21 | 73.09 | 26.50 | 29.98 |
| 32 Ki | 61.53 | 70.42 | 70.04 | 26.46 | 29.95 |
| 64 Ki | 62.28 | 72.81 | 67.49 | 26.40 | 29.82 |
| 128 Ki | invalid | 68.90 | 66.24 | invalid | 28.03 |

The selected arm's inferred step rate improves about 13% at 16–64 Ki. There
is no valid 128 Ki reference comparison. Relative to four draft rows, eight
rows cost 0.5–0.7% in step rate and yield a 2.1% higher throughput geometric
mean in these individual sweeps. Contexts move in both directions, so no
universal acceptance advantage is claimed.

The client derives step rate from throughput and speculative-counter ratios.
Independent rank-zero traces at approximately 64 Ki show target graph duration
35.35 to 30.85 ms, packed MLA kernel sum 7.50 to 2.83 ms, and four-row draft
duration about 2.14 ms. Eight draft rows take 2.285 ms while the target remains
30.848 ms. Cross-stream kernel sums are not critical-path durations. The
reference trace contains about 66 Ki total tokens and the optimized traces
about 64 Ki; fixed-input kernel comparisons provide a separate control.

The ordinary E4M3 four-query MLA kernel is present in the source but does not
consume `fp8_ds_mla`. This packed path originally padded 99 effective heads to
104, producing a separate eight-head tail launch. Padding to 112 reduces
192 reader launches to 96 across four steps and 24 MLA layers. This launch
change and vector shared-memory loads are consistent with the reduction in
the fixed-input kernel measurements. The serving traces combine both changes
and do not isolate their individual contributions.

### Invalid server-counter attribution

The reference 128 Ki cell has zero client tokens, zero requests with output,
zero TTFT, and a queue fraction of 1.0. Its request remains queued after a
188-second warmup timeout. `prometheus_fallback` attributes 1,443 tokens from
another request to this cell and reports 48.09 tokens/s and 23.23 steps/s.
Both values are invalid for the benchmark request and supply no gain estimate.

[llm-inference-bench #16](https://github.com/local-inference-lab/llm-inference-bench/pull/16)
requires output from the benchmark streams, removes the live/final global-rate
fallback, and uses observed chunks for totals when usage counts are absent.
The unmodified client fails two ownership regressions; the fixed upstream
client passes three, and the resume-capable deployment passes four. Saved
`prometheus_fallback` rows are excluded from resumed comparisons.

The operator separately reported about 72 tokens/s at 128 Ki without a recorded
row. The available data cannot establish that observation's cause. Changes in
acceptance can also legitimately make output throughput non-monotonic even
when verification step rate is stable.

## Research coverage and limitations

An all-state snapshot contains PR metadata and bodies for 75,914 pull requests:
vLLM 37,013, SGLang 30,628, FlashInfer 3,712, lab vLLM 621, B12X 274,
lab LMCache 62, and upstream LMCache 3,604. Collection completed at
2026-09-08 06:02:56 UTC. This is collection coverage, not line-by-line review
of every implementation. The related benchmark repository adds 11 PR records.

- KDA ReplaySSM designs in SGLang #36821 can reduce snapshot writes, but the
  measured recurrent/convolution kernels account for about 0.36 ms each per
  step. Accepted-state commit and cache restoration must be integrated.
- SGLang #38399 demonstrates why replay benchmarks must include flush phases.
- B12X #279 supplies a PDL producer/consumer pattern; Kimi's AttnRes consumer
  differs from that PR's MHC operation.
- The QSRT target already decodes native two-bit payloads. Its reconstruction
  law and coupled transforms are preserved; no lower-precision target codec
  or sparse attention policy is selected.
- The 512-thread MoE option preserves the recorded output digests, but its
  instrumented timing runs overlap live service requests. Performance is
  unqualified, and the option remains disabled. B12X #339 contains the output
  dtype correction and timing/launch evidence tooling.
- Full-history draft restoration and atomic multi-rank checkpoint designs in
  lab vLLM #709 and LMCache #62 remain separate integration work. Foreign
  hardware or transfer-only results do not establish Kimi acceptance parity.
