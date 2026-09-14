# DeepSeek V4.1 native B12X DCP (experimental)

This feature branch starts at local-inference-lab `dev/jovian-judgement`
`ab03e87100efa9536ec87e01994828b459c956ff`. Native DCP exchange preparation
requires the companion yatesdr/b12x feature branch, commit `3395626b` (based
on `9e90d60f0cc8f204aa2fd219ed9b6abee32de7d8`). No upstream PR is submitted.

## Layout and execution

- Compressed main KV is owner-sharded. Ownership uses original-token stripes
  divisible by the compression ratio; the tested stripe is 128 tokens.
- Index KV, SWA and private compressor state are replicated. Existing native
  global Full/Reindex/Reuse candidate selection is unchanged.
- Global selected IDs are masked/remapped into each owner's local main cache.
- Native B12X PCIe head gather expands TP-local queries within the DCP group.
  Native compressed MLA returns partial output and natural-log LSE. B12X LSE
  reduce-scatter returns each TP rank's own heads.
- Replicated SWA and the attention sink enter the attention union only on DCP
  rank zero. This prevents duplicated probability mass.
- One serial channel is shared across layers, with a 256-row capacity. Prefill
  attention is chunked within that capacity. This is not a fourfold increase
  in total KV capacity, because the index and SWA remain replicated.

## Tests performed on cn4

Four RTX PRO 6000 Max-Q, SM120, stock operating settings; cn3 was untouched.

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

## Serving configuration under qualification

TP4/DCP4, 540672 context, 4096 batch budget, max-seqs 4, main/SWA pages 256/128,
SSD Engram, native prefix caching, decode graphs enabled, prefill graphs off.
Initial full-model bring-up disables speculation, LMCache and startup search
(`enable_b12x_autotune=false`) to isolate native DCP correctness. DSpark K5,
matched serving benchmarks, long-context qualification and LMCache DCP restore
remain required before calling this a production-qualified configuration.

LMCache must use a separate DCP-layout namespace and a chunk divisible by
the main cache's global block span (256 * 4 = 1024), not reuse DCP1 objects.
Do not enable it based only on the component oracle results.

PCP, multi-node DCP, overlapping independent channel replay and DCP sizes other
than 2/4 are not supported by this implementation. Human review, relevant model
evaluations and DCO confirmation are required before an upstream PR.

AI assistance was used in implementation and test development.
