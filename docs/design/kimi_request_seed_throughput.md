# Kimi request seed and decode throughput

Status: qualified observations for three short requests; the reported
95-to-70/75-token/s startup difference is not fully reproduced.

The observed serving package used vLLM `f9ac0209a2ed`, B12X `0bf9f177b237`,
TP9/DCP9, a colocated DFlash2 draft with K=3, and context capacity 1,048,576.
Requests used the mathematics-article prompt from `llm_decode_bench.py`,
CC1, context-padding length zero, no temperature override, `ignore_eos=true`,
and streamed cumulative token usage. The actual chat prompt contained 155
tokens. Each request was observed for approximately 15 seconds after its
first streamed token and then cancelled.

The explicit diagnostic seeds were A=`5778815928437199162` and
B=`7613996499038379085`. They were derived from the signed-int64 fallback
seed stream in the V2 sampler, not selected by a throughput search.

| Request | Client tokens/s | Tokens per speculative step | Estimated engine-step interval |
| --- | --- | --- | --- |
| B, repetition 1 | 82.397 | 2.590 | 31.367 ms |
| B, repetition 2 | 82.427 | 2.590 | 31.356 ms |
| A | 76.525 | 2.402 | 31.319 ms |

The B repetitions produced identical output bytes. Their output SHA-256 was
`fe397186ff023312e3115388ae33ad6ca443f62c7ff0368bcd2aa44f2ac1bf0d`.
The A/B throughput difference follows speculative acceptance while the
estimated step interval remains nearly constant. This does not establish the
cause of the operator-reported 95-token/s sample, which was not captured.

An initial A sample was excluded because server counters included another
request: 1,122 prompt tokens instead of 155 and 151 generated tokens beyond
the probe's stream usage. Further replay attempts stopped when live traffic
prevented an idle measurement boundary. No result from those attempts is
qualified as CC1 evidence.

## Source mechanism

The observed V2 runtime's `vllm/v1/worker/gpu/sample/states.py` maintains a
private `np.random.default_rng(model_seed)` stream. A request without an
explicit seed draws an int64 from `[INT64_MIN, INT64_MAX)`. Different request
seeds affect both target rejection sampling and DFlash2's probabilistic draft
sampling. Process startup resets the private stream, so request order can
reproduce the same sequence of different workloads after each restart.

Sampler warmup also consumes fallback seeds. In the four-request warmup
configuration, the first external request is expected to use draw five;
the actual warmup count must be verified before treating that prediction as
an observed seed. The default benchmark additionally performs a hidden decode
warmup and a context scout, so its first displayed cell need not be the first
external request. Explicit seeds were used only for diagnosis; the production
sampler and its default seed policy were not changed.

## Interpretation and limits

Token throughput combines execution rate and accepted tokens per step.
The interval above is inferred from elapsed time and speculative/generation
counter deltas. It is not a GPU-only verify-kernel duration. Kernel traces,
matched output workloads and hardware-state checks remain necessary to
attribute a performance change to an implementation.

At about 31.3 ms per step, acceptance near three emitted tokens per step
would yield about 96 tokens/s. That arithmetic makes acceptance a plausible
explanation for a fast sample; it is not a measurement of the missing sample.
The observations do not prove or disprove a one-time post-request runtime
transition. Full benchmark and trace comparisons were deferred to preserve
interactive serving, which remained on the qualified code package.
