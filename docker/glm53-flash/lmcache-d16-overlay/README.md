# GLM-5.3 LMCache D16 overlay

This directory contains the LMCache-owned portion of the qualified GLM-5.3
integration. It is intentionally separate from vLLM source. Apply the files
under `overlay/` to LMCache 0.5.4 and rebuild its CUDA extension with the
included patches at LMCache commit
`3e11b8ed191631e6f098b8038235823f1a410b24`.

The overlay provides:

- DCP/interleave-aware cache identity and logical group spans;
- BLHNC/BLNHC padded packed-stride transfers;
- one- and two-argument Mamba alignment validation;
- authoritative engine-group registration geometry;
- lifecycle-safe cuMem POSIX-FD IPC without `cudaDeviceReset`; and
- legacy D2H stores for compressed logical geometry while retaining native
  H2D retrieval and native D2H for uncompressed object groups.

The sibling `single-container/` directory documents and supervises the
production one-container deployment.
