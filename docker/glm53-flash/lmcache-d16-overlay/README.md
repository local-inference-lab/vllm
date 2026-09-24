# GLM-5.3 LMCache D16 overlay

This directory contains the packaged LMCache and vLLM core overlays for the
qualified GLM-5.3 integration. Apply the files under `overlay/` to the runtime
site-packages, use LMCache 0.5.4, and rebuild its CUDA extension with the
included patches at LMCache commit
`3e11b8ed191631e6f098b8038235823f1a410b24`.

The overlay provides:

- DCP/interleave-aware cache identity and logical group spans;
- BLHNC/BLNHC padded packed-stride transfers;
- one- and two-argument Mamba alignment validation;
- authoritative engine-group registration geometry;
- exact committed Mamba boundary handoffs from vLLM core;
- sparse LMCache Mamba chunk sources, where retained checkpoints use their
  physical block IDs and unavailable historical checkpoints use null block 0,
  while attention groups retain their normal dense block history;
- lifecycle-safe cuMem POSIX-FD IPC without `cudaDeviceReset`; and
- legacy D2H stores for compressed logical geometry while retaining native
  H2D retrieval and native D2H for uncompressed object groups.

The exact-boundary behavior depends on the scheduler and connector semantics
from [#526](https://github.com/local-inference-lab/vllm/pull/526) and
[#527](https://github.com/local-inference-lab/vllm/pull/527). In particular,
connector reconciliation cannot advertise a prefix beyond the lagging Mamba
state, and the scheduler must forward each core-selected
`(request_id, group_id, physical_block_id, boundary_tokens)` handoff.

The sibling `single-container/` directory documents and supervises the
production one-container deployment.
