# GLM-5.3-Flash D16 qualification

Source: `llm-inference-bench` commit
`42c38fdd0476cf86940268f4e098581e5b61542e`.
The supplied artifact SHA-256 was
`0fd4f7609962ee5b7cb13780c7ce23c9820f5e851e27463bb21264f3662f21c6`.
The committed artifact removes the machine hostname from two startup fields
and has SHA-256
`946b22d4cb9f44051ab21af82e7c979f378d6506ad192a3bb7cce82921e94c07`.

> **Corrected readiness status:** D16 remains performance evidence only.
> Delayed lockhandle corruption invalidated its production-readiness claim.
> The measured benchmark results below are unchanged.

## Sustained aggregate decode tok/s

| Context | C1 | C2 | C4 | Requested C8 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 153.1 | 142.4 | 229.8 | 227.1 (effective C4) |
| 16k | 141.5 | 126.5 | 220.4 | 189.0 (effective C4) |
| 32k | 140.7 | 127.3 | 214.7 | 192.4 (effective C4) |
| 64k | 70.7 | 128.5 | 208.2 | 202.5 (effective C4) |
| 128k | 69.4 | 123.7 | 204.2 | 199.0 (effective C4) |
| 256k | 66.1 | 125.7 | 202.1 | skipped |
| 512k | 66.0 | 122.7 | skipped | skipped |

## Burst/E2E aggregate tok/s

| Context | C1 | C2 | C4 | Requested C8 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 77.3 | 142.6 | 228.1 | 227.4 (effective C4) |
| 16k | 71.2 | 127.1 | 196.5 | 197.5 (effective C4) |
| 32k | 70.6 | 127.7 | 197.5 | 198.7 (effective C4) |
| 64k | 70.8 | 128.9 | 201.6 | 203.9 (effective C4) |
| 128k | 69.3 | 124.0 | 195.5 | 196.6 (effective C4) |
| 256k | 65.9 | 118.8 | 187.2 | skipped |
| 512k | 61.1 | 110.2 | skipped | skipped |

## Prefill scout

| Prompt | Client tok/s | Prometheus tok/s | Prometheus seconds |
| --- | ---: | ---: | ---: |
| 8k | 5,832 | unavailable | unavailable |
| 16k | 1,822 | 1,868 | 8.771 |
| 32k | 2,946 | 3,013 | 10.875 |
| 64k | 3,984 | 4,056 | 16.158 |
| 128k | 6,486 | 6,624 | 19.787 |

The integrated scout uses client `prompt_tokens / TTFT`. Prometheus
`kv_computed` values are validation where available.

## Run facts

- 4× NVIDIA RTX PRO 6000 Blackwell Max-Q, driver 595.71.05
- 2h18m42s runtime; 10,078 hardware samples
- 96.23% average and 100% maximum GPU utilization
- 63C maximum GPU temperature
- 748.39 W average and 956.46 W maximum aggregate power
- 99.08% maximum VRAM used
- 82.8/88.8 GB/s peak PCIe RX/TX
- 25 measured sustained cells and 25 burst cells
- capacity skips: `(256k,C8)`, `(512k,C4)`, `(512k,C8)`
- zero request errors
- 1,544,414-token KV pool, 1.47× exact-1M concurrency

D16's immediate correctness validation retrieved a unique buried record at
199,974 tokens while storing 4.919 GB, immediately replayed the same prefix
correctly, passed a four-turn chat, and passed one-container stop/start. The run
had no OOM, CUDA error, `EngineDead`, or process restart. Those immediate checks
did not expose delayed lockhandle corruption and therefore do not establish
production readiness.

## Corrected D22 correctness qualification

The root mechanism invalidating D16 readiness was positional/stale Mamba
recurrent state stored under valid prefix keys. External H2D reload then
silently corrupted live state while HTTP health stayed green.

D22 corrects this with:

- vLLM exact committed Mamba boundary handoffs
- connector boundary reconciliation
- sparse null placeholders for unavailable historical Mamba checkpoints
- separated LMCache object groups
- effective window-count validation and transfer

Runtime code ownership is split across
[#525](https://github.com/local-inference-lab/vllm/pull/525) for exact
geometry and boundaries,
[#526](https://github.com/local-inference-lab/vllm/pull/526) for transfer
normalization, and
[#527](https://github.com/local-inference-lab/vllm/pull/527) for lifecycle and
object separation.

D22 production qualification on 4× RTX PRO 6000 used a unique 131,041-token C1
store. L1 held 60 objects totaling 993,329,152 bytes. C2 external reload
processed 258,048 tokens total (2×129,024); both responses were coherent, and
an unrelated post-reload raw probe was also coherent. The production container
remained healthy with restart count 0 and OOM false. There was no lockhandle,
CUDA error, `EngineDead`, or new Xid.

The direct Bifrost smoke route `vllm/glm-5.3-flash` returned exactly `OK` for
the exact-OK prompt, with 578 ms displayed latency and 20 input / 53 output
tokens.

## Interpretation

Sustained decode uses `ignore_eos=true`; forced continuation text is not an
answer-quality evaluation. Requested C8 is generally effective C4 because the
qualified server uses `max_num_seqs=4`. Effective-concurrency labels must be
retained when citing these results.
