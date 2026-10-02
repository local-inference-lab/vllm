# GLM Spark projection selection and online MTP quantization: validation

The implementation converts the BF16 projections selected by the GLM Spark
checkpoint to MXFP8 while loading the `mtp-bf16` source checkpoint. It also
converts the BF16 routed experts in the MTP layer to NVFP4 W4A16. The source
checkpoint is read without modification. MXFP8 scales retain their native
representation.

## Behavior and compatibility

- **Implemented:** exact target coverage, MXFP8 attention/shared projections,
  online NVFP4 MoE with BF16 activations, independent draft quantization, and
  separate main/draft FP4 scale-compression policies.
- **Qualified:** target-set coverage on both TP2 ranks, GPU encoder tests,
  bounded loader ownership, startup, C1/C8 generation and 4K prefill under the
  configurations recorded below. Qualification is limited to these conditions.
- **Research-only:** short throughput measurements and the English corpus
  quality sample. These do not establish downstream model quality or long-context
  capacity.
- **Unsupported by this report:** MXFP8 scale compression, checkpoint export,
  byte parity with the serialized Spark checkpoint, million-token serving, and
  hardware or parallelism configurations outside the recorded tests.

`strict_targets` is opt-in and defaults to `false`. Exact targets in the example
set it to `true`; missing projections, incomplete fused groups and extra online
quantizers fail before checkpoint loading. The selected 531 source projections
comprise 402 attention matrices (34 KDA layers and 12 MLA layers, including MTP)
and 129 shared-expert matrices. Fusion produces 331 main-model linear modules
from 520 source targets and nine MTP linear modules from 11 source targets.
One additional MTP target selects the routed MoE with `nvfp4_a16`.

Embeddings, the LM head, the vision tower, routers, norms, convolutions and the
first three dense feed-forward layers retain their source representation.
Independent head, vision and dense quantization environment controls are disabled
in the tested configuration. Main NVFP4 expert activation precision is unchanged.

The loader releases Python references to replaced BF16 parameters before packed
weights are prepared. The NVFP4 encoder computes per-expert global scales across
TP shards and quantizes the original BF16 values directly into preallocated
outputs. It does not first round globally rescaled BF16 weights or retain a list
of complete intermediate expert tensors. Draft configuration uses a shallow
configuration copy to preserve target-model recurrent-state planning while
selecting the draft's own quantization method.

The NVFP4 packed layout does not change. Existing `nvfp4_per_token` encoding can
produce different values because eliminating the intermediate BF16/FP16 rounding
changes code selection; original-input GPU parity tests cover the intended
quantizer contract. Its SM100 restriction remains. `nvfp4_a16` explicitly rejects
FP16 activations and requires BF16 with a supported W4A16 backend.

## Artifacts and conditions

| Artifact | Immutable identity |
| --- | --- |
| Source checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4`, revision `cfd47bd7680e68408924df09b179d5bed25b2ae9` (`mtp-bf16`) |
| Projection-selection reference | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision `a608241037e4c2565356bff7ca293f2133888f88` |
| Container | `ghcr.io/local-inference-lab/vllm@sha256:2230db60afb4fd06dc2ef2b7f7f27dc7f78d7a50be70a08461b4df9fa5e90732` |
| vLLM base | `93dabce32fd4d5355662608296e64d720711cdcc` plus the source changes in this PR |
| B12X | Unmodified installed `a77b3f85e5e2a81315ff4a90088912f187708005` |
| Hardware | Frank1, RTX PRO 6000 Blackwell 96 GB, TP2, 600 W limits |
| Runtime limits | FP8 KV, 3 GiB KV allocation per rank, 8192 model length, 1024 batched tokens, eight sequences |

Performance uses physical GPUs 12/13 for the no-MTP pair and 14/15 for the MTP
scale-policy pair. Loaded memory clocks were 16365 MHz. Graphics clocks were
dynamic, approximately 2602–2910 MHz; active telemetry reported the 0x400 clock
event mask. No task-owned compilation or second benchmark ran concurrently with
the throughput measurements. Separate already-running services on GPUs 0–9 were
preserved, so these are not isolated-host measurements.

Main FP4 CSF is enabled in every performance row. CUDA graph mode is
`FULL_AND_PIECEWISE`, with capture sizes 1, 2, 4, 8, 16, 32 and 64. The online
weights are generated during loading; the deferred MXFP8-CSF prototype is absent
from the runtime import path. Performance runtimes contain a disabled-unless-
requested prompt-logit capture helper; benchmark requests do not enable it.

## Memory and preparation

| Measurement | Result | Interpretation |
| --- | ---: | --- |
| Main model allocation before KV, BF16 selected projections | 92.071 GB/rank | PyTorch allocated bytes at the loader audit |
| Main model allocation before KV, MXFP8 selected projections | 88.548 GB/rank | Same no-MTP audit point |
| Main projection allocation reduction | 3.524 GB/rank; 7.047 GB/TP2 | Includes backend representation at this audit point |
| MTP NVFP4 native prepared expert storage | 2,038,438,216 bytes/rank | Real 288-expert packed-tensor census |
| Standalone MTP CSF additional storage with dedicated scratch | 117,862,208 / 117,865,424 bytes/rank | Applies when a separate reconstruction buffer is required |
| CSF reconstruction scratch | 226,492,416 bytes/rank | Main and MTP share it in the measured full model |
| Full-model MTP CSF allocation saving | 108,628,480 / 108,624,896 bytes/rank | 217,253,376 bytes saved across TP2 with shared scratch |
| Whole-expert BF16-to-NVFP4 encoding | 0.0894 seconds | GPU encoding after staging; excludes I/O and serving initialization |

GB denotes decimal bytes. The packed-tensor census is not an end-to-end server
VRAM comparison. Main allocation in the MTP configurations is 88.564 GB/rank;
different graph/buffer preparation accounts for the small difference from the
no-MTP audit. End-to-end NVML memory includes CUDA context, caches, graph pools and
temporary allocations and must not be equated with checkpoint size.

The recommended draft setting inherits the main model's CSF policy. A full-model
audit verifies identical reconstruction-buffer addresses within each rank's
process for all 42 main layers and the MTP layer. The draft therefore adds only
its compressed payload, saving about 217 MB across TP2. A standalone MTP census
that charges the entire reconstruction buffer to that layer does not describe
this serving configuration. `mtp-shared-scale-storage.json` records the pointer
identity checks and paired loader allocations. Both native and compressed draft
scales completed TP2 serving.
The MXFP8 scale-compression investigation is deferred: its separate tensor
census estimated approximately 185 MB total saving across TP2, not a measured
full-server saving.

## Throughput

Decode entries are medians of three 30-second sustained runs. C8 is aggregate
throughput across eight requests. Input prompts are approximately 78 tokens and
decode requests use up to 2048 output tokens. Prefill uses four samples with
approximately 4170 actual prompt tokens, validated against server counters with
zero cached tokens.

| Weight configuration | C1 output tok/s | C8 output tok/s | 4K prefill tok/s |
| --- | ---: | ---: | ---: |
| BF16 selected projections, no MTP | 121.705 | 511.332 | 6545 |
| MXFP8 selected projections, no MTP | 145.071 | 558.631 | 5115 |
| MXFP8 + NVFP4 MTP, draft CSF scales | 210.500 | 711.354 | 4782 |

The native-draft run with a 65536 model limit measured medians of 212.881 tok/s
at C1 and 712.667 tok/s at C8, with 4977 tok/s 4K prefill and 5865 tok/s 32K
prefill. All three accepted decode repetitions sustained the requested
concurrency without errors, underfilling or queue flags. A preceding attempt
with one transient queued-request sample was retained as diagnostic evidence
and excluded under the benchmark's existing validity rule; the rule was not
changed. Its raw output remains in
`serving/weights-mtp-native-scales-10/benchmark/`; the complete accepted set is
in `benchmark-verified/` beside it.

With the same 65536 model limit, the compressed-draft run measures 215.569 tok/s
at C1, 707.298 tok/s at C8, 4848 tok/s 4K prefill and 5792 tok/s 32K prefill.
Compared with native draft scales, these are +1.26%, -0.75%, -2.59% and -1.24%,
respectively. The runs use different GPU pairs with matching memory clocks;
graphics clocks remain dynamic, and three decode repetitions do not establish
sub-percent performance parity. Draft acceptance is 52.59% with CSF. Combined
with the measured 217 MB TP2 allocation saving and shared-buffer audit, these
results support inheriting CSF for the draft in this configuration.

With a 1024-token scheduler budget, the paired no-MTP comparison improves
decode by 19.2% at C1 and 9.25% at C8, but reduces prefill throughput by 21.8%.
The prefill result depends strongly on that budget; the 3072-token comparison
below reverses the throughput difference.

Across the draft-CSF benchmark, 64,929 of 123,627 proposed tokens were accepted:
52.52%, or 2.576 output tokens per draft cycle including the target token.
These counters include benchmark warmups and prefill requests. They do not
compare NVFP4 draft quality against a BF16 draft or prove downstream capability
retention.

### 32K prefill

A separate no-MTP pair raises `max_model_len` to 65536 while retaining the same
3 GiB KV allocation, 1024-token chunk size and normal performance kernels.
Both servers run on GPUs with 16365 MHz loaded memory clocks, with BF16 on 14/15
and MXFP8 on 12/13. The two benchmarks run sequentially after model loading.

| Selected projection weights | Actual prompt tokens | Server prefill tok/s | Client TTFT | Samples |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 32,328 | 7796 | 4.20 s | 2 |
| MXFP8 | 32,312 | 6022 | 5.41 s | 2 |

MXFP8 prefill throughput is 22.8% lower and TTFT is 28.8% higher in this bounded
sample. Server counters report zero cached tokens. The benchmark estimates the
text length, so the prompts are approximately 32K rather than exactly 32768
tokens. In the same runs, 4K server throughput is 6618 versus 5039 tok/s.
Raw results are in `serving/{teacher,weights}-prefill32k-9/benchmark/`.

### Prefill scheduler budget

On the same physical GPUs 12/13, raising `--max-num-batched-tokens` to 3072
gives 12119 tok/s with BF16 selected projections and 12681 tok/s with online
MXFP8, a 4.6% increase for MXFP8. Both runs use the source revision above,
TP2/DCP1, no MTP, main FP4 CSF, 3 GiB FP8 KV per rank, 65536 model length,
eight sequences, normal fast MoE reduction, and frozen runtime12 on the same
pinned image. The measurements run sequentially with no second benchmark or
model compilation active. Memory clocks are 16365 MHz and graphics clocks
remain dynamic under 600 W limits. BF16 on GPUs 14/15 independently measures
12075 tok/s with the same settings. An independent reload of online MXFP8 on
GPUs 12/13 measures 12760 tok/s, compared with 12681 in the first run. Each
32K row contains two samples, so these figures do not establish a narrow
confidence interval.

The budget changes both the number of model executions needed for a prompt and
the MXFP8 GEMM dispatch. In installed B12X revision `a77b3f85`,
`b12x/gemm/blockscaled/_preparation.py` prepares a separate large-prefill
program only when the declared capacity is at least 2048 rows. At execution,
that program serves calls with at least 2048 rows; smaller calls retain the
short-row program. `b12x/_lib/dense_gemm.py` uses the large-row hint to select
different matrix tiles for supported geometries. A 1024-token scheduler
budget cannot enter that regime. The measurements establish the combined
effect of batching and dispatch; they do not isolate the contribution of
each mechanism.

The separate profiler requests contain approximately 4490 prompt tokens.
The 1024-token run performs 1545 packed-linear calls and 210 routed-MoE
calls; the 3072-token run performs 618 and 84, respectively. This is the
expected five-versus-two model executions. Inclusive CPU time in the
packed-linear operation is 200.6 versus 84.9 ms under profiling. Profiling
adds overhead, so these CPU totals explain call amplification and must not
be substituted for the unprofiled server throughput or interpreted as a
complete critical-path decomposition.

Online weight encoding completes during loading. Both online MXFP8 and
serialized ModelOpt MXFP8 use `B12xMxfp8LinearKernel` in this runtime. No encoder
or B12X source change is required to obtain the 3072-token result. The
1024-token slowdown remains a valid measurement for that configuration and
must not be generalized to MXFP8 with other scheduler budgets.

The serialized Spark comparison uses revision
`a608241037e4c2565356bff7ca293f2133888f88` with the same image, overlay and
serving settings. Its 44 index-referenced LFS object identities and byte
counts match locally cached blobs; configuration and tokenizer files are
resolved at the stated revision. It measures 13360 tok/s at 32K on GPUs 14/15
with a 3072-token budget. With a 1024-token budget, serialized Spark instead
measures 6009 tok/s on GPUs 12/13, while a repeated online MXFP8 run measures
6031 tok/s on GPUs 14/15. The approximately 0.4% difference is below the
precision justified by these short runs. Serialized weights therefore
reproduce the small-budget slowdown: the online weight encoder is not its
cause. Spark contains different weight values from the
`mtp-bf16` source, so its throughput difference is not an isolated comparison
of offline and online encoding implementations.

| Weight configuration | 1024-token budget, 32K tok/s | 3072-token budget, 32K tok/s |
| --- | ---: | ---: |
| Source BF16 projections | 7796 | 12119 |
| Online MXFP8 projections | 6031 | 12681 |
| Serialized Spark MXFP8 | 6009 | 13360 |

The source BF16 and online MXFP8 3072-token rows use the same GPUs 12/13.
The 1024-token BF16/online rows use GPUs 14/15. Spark uses 12/13 for 1024 and
14/15 for 3072. Both pairs have matching memory clocks; the independent BF16
3072-token repeat differs by 0.36% between pairs. Each row has two or three
32K samples with no prefix-cache hits. Model values, dynamic graphics clocks
and short samples prevent interpreting the Spark/online gap as an encoding
implementation cost.

Raw artifacts are under `serving/prefill-bf16-c3072`,
`serving/prefill-bf16-c3072-g12`, `serving/prefill-mxfp8-c3072` and
`serving/prefill-spark-c3072`; the online reload is
`serving/prefill-mxfp8-c3072-repeat`. Their benchmark JSON records zero prefix-cache
hits and both client TTFT and server prefill counters. GPU profiler captures
are separate requests and are excluded from throughput measurements.
The 1024-token repeat artifacts are `serving/prefill-mxfp8-c1024-profile` and
`serving/prefill-spark-c1024`.

With three-token MTP and compressed main/draft scales, the 3072-token budget
also fits the same 3 GiB KV allocation per rank and completes all serving
checks. Three accepted decode repetitions measure medians of 212.874 tok/s
at C1 and 704.562 tok/s at C8. Prefill measures 11831, 12148 and 12295 tok/s
at approximately 2K, 8K and 32K; 32K client TTFT is 2.68 seconds. No request
errors, underfilled decode cells, queue flags or prefix-cache hits are reported.
These measurements use GPUs 14/15 and are recorded in
`serving/prefill-mtp-c3072`. Weight audits retain 331 main linear modules and
nine draft linear modules plus one draft MoE on each rank.

The `prefill_budget_comparison` object in the adjacent machine-readable
results records every accepted row, exact configurations, repeated decode
samples and artifact hashes. The separate local diagnostic artifact
`prefill-budget-diagnosis.json` contains the detailed profiler operations and
launch records. Source code and weight encoding remain unchanged throughout
these scheduler-budget comparisons.

## Quality sample

Quality uses a 1024-token scheduler budget. The 3072-token performance
measurements do not repeat the corpus quality evaluation.
Quality compares the source checkpoint's BF16 selected projections with their
online MXFP8 replacements, with MTP disabled on both sides. The main experts
remain NVFP4 with CSF in both runs. This is not a comparison against a fully
BF16 original model. The comparison includes the selected MXFP8 linear
backend's activation arithmetic as well as quantized weights.

The sample contains 16 distinct English WikiText document prefixes of 1024
tokens each, selected with seed 53145 from `Salesforce/wikitext`,
`wikitext-2-raw-v1`, test split, dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The tokenizer uses the pinned source
checkpoint. The input-suite SHA256 is
`dece045c0b507d3288d667e281a76b12cbe75c2e4afc16e91f26657e234e821a`.

Full-vocabulary FP32 logits cover all 154880 vocabulary entries. The comparison
normalizes in FP64 and computes `KL(P_source || P_online)` over 16384 positions;
next-token negative log-likelihood uses the 1023 known successor tokens per
document. Confidence intervals resample documents, not individual tokens.

For quality capture, both servers use `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`,
eager execution, TP2 on GPUs 10/11 and 2 GiB KV per rank. The normal fast MoE
accumulation path produced approximately 0.09 KL even when repeating an unchanged
model. Repeated hidden-state captures were identical through layers 0–2 and
first differed in the routed MoE output at layer 3. The existing deterministic
reduction removes that confound: repeated four-document captures of both the
source and MXFP8 model have exactly zero KL and identical top-1 predictions.
Performance tables use the normal fast path, not this quality configuration.

| Measurement | Result |
| --- | ---: |
| Mean source-to-MXFP8 KL | 0.09906274 nats/token |
| Document-bootstrap 95% interval | 0.08623–0.11260 |
| Top-1 agreement | 90.271% |
| Source next-token NLL | 1.08259615 |
| MXFP8 next-token NLL | 1.08448085 |
| NLL increase | 0.00188470 nats/token |

The probability distributions change measurably. The mean next-token NLL change
is small on this corpus sample, but it does not establish multilingual,
reasoning, tool-use or long-context accuracy. It also does not qualify MTP draft
quality against a BF16 draft. The deterministic result is recorded in
`deterministic-target-weight-kld.json`; repeat checks are
`teacher-deterministic-repeat-kld.json` and
`weights-deterministic-repeat-kld.json`. Nondeterministic captures must not be
used as the isolated quantization-error measurement.

## Component validation

The final combined source check ran 29 focused tests covering exact target rejection and
fused coverage, the `nvfp4_a16` shorthand, original-input NVFP4 parity, kernel
reuse, caller-owned workspace forwarding, released BF16 parameter ownership,
draft configuration, owning-model scale policy, BF16 activation validation and
pooled-indexer FP32 dequantization. All 29 passed.
Earlier broader component checks passed 43 online/configuration tests, 32 loader
unit tests, six GPU NVFP4/TP tests, 18 scale-policy tests and a compilation-hash
test. Counts overlap and must not be added as independent coverage.

The loader lifetime reproducer failed before the ownership correction and passed
for both parameter load orders afterwards. GPU NVFP4 tests compare values and
scales with the original-input quantizer and verify TP-consistent global scales.
Full model serving audited the constructed online methods on both TP ranks;
checking configuration strings alone was insufficient.

Eight additional GPU scale-policy tests verify that each expert retains its
owning model's policy even when deferred preparation runs inside the other
model's configuration context. The reproducer fails six of eight cases with
ambient-context selection and passes all eight with the policy stored in
`FusedMoEConfig`.

The final activation-contract check passes four tests covering FP16 rejection,
original-input encoding and kernel reuse for both activation modes. Frozen
runtime12 differs from runtime11's production code only by that early dtype
validation; the serving measurements use BF16 and do not enter its rejection
branch.

A chat smoke returned Paris and 323 for the requested factual/arithmetic values,
but included surrounding text despite a JSON-only instruction. It is a bounded
generation check, not structured-output qualification. The full repository test
suite and downstream evaluation suites were not run.

## Evidence location

Frank1 evidence is under
`/data/trellis-quant/glm53-online-mxfp8-csf-20261002/`:

- `weight-serving-summary.json` records raw throughput repetitions, speculative
  counters and loader allocations.
- `serving/{teacher-nospec-1,weights-nospec-5,weights-mtp-load-6,weights-mtp-native-scales-7}/`
  contains exact launch arguments/environment, weight audits, benchmark JSON,
  server metrics and GPU telemetry.
- `runtime/online-weights-{5,6,9,10}/source-manifest.json` records frozen Python
  hashes, image identity, installed binary links and diagnostic instrumentation.
- `mtp-nvfp4-csf-storage.json` records the standalone prepared-tensor census and
  timing; `mtp-shared-scale-storage.json` verifies full-model buffer sharing and
  the corresponding allocation saving.
- `qualified-source-tests.log` and the component test logs preserve pytest outcomes.

The adjacent [machine-readable results](glm53_spark_online_results.json) contain
all accepted repetitions, telemetry ranges, quality statistics, memory audits
and SHA256 identities of the detailed artifacts. Runtime11 adds diagnostic
buffer-ownership capture; runtime12 adds the final activation dtype validation.

The [serving guide](glm53_spark_online.md) contains a complete launch command;
the exact projection manifest and generator are in
`examples/features/quantization/glm53_spark_targets.json` and
`examples/features/quantization/glm53_spark_online.py`.
