# GLM-5.3 D16 lineage disposition

This matrix accounts for the 81 reference changes in
`30df0f8^..8c61ef3`. “Already Jovian” means the behavior is present in the
`dev/jovian-judgement` base. “Ported (source+test)” identifies the production
source or recipe and focused regression test in this change. “Excluded”
identifies fleet-only qualification, image-build, or diagnostic behavior that
does not belong in the production vLLM/LMCache integration.

## D22 readiness correction

D16 remains performance evidence only. Its production-readiness claim was
invalidated by delayed lockhandle corruption: positional/stale Mamba recurrent
state was stored under valid prefix keys, then external H2D reload silently
corrupted live state while HTTP health stayed green. The historical
dispositions below remain unchanged.

| D22 area | Corrected disposition and evidence |
| --- | --- |
| Mamba boundary state | Proven — vLLM exact committed Mamba boundary handoffs, connector boundary reconciliation, and sparse null placeholders for unavailable historical Mamba checkpoints. Runtime ownership: [#525](https://github.com/local-inference-lab/vllm/pull/525) exact geometry/boundaries. |
| Transfer and storage | Proven — separated LMCache object groups plus effective window-count validation/transfer. Runtime ownership: [#526](https://github.com/local-inference-lab/vllm/pull/526) transfer normalization and [#527](https://github.com/local-inference-lab/vllm/pull/527) lifecycle/object separation. |
| Production qualification | Proven on 4× RTX PRO 6000 — unique 131,041-token C1 store; L1 60 objects / 993,329,152 bytes; C2 external reload 258,048 tokens total (2×129,024); both responses coherent; unrelated post-reload raw probe coherent. Production container healthy, restart count 0, OOM false; no lockhandle, CUDA error, `EngineDead`, or new Xid. |
| Direct Bifrost smoke | Proven — route `vllm/glm-5.3-flash` returned exactly `OK` for the exact-OK prompt, with 578 ms displayed latency and 20 input / 53 output tokens. |

| # | Reference | Behavior | Disposition and evidence |
| ---: | --- | --- | --- |
| 1 | `30df0f8` | GLM qualification track | Excluded — fleet qualification orchestration; the portable recipe is `single-container/README.md`. |
| 2 | `1e737d9` | SM120 sparse MLA | Already Jovian — B12X sparse MLA dispatch is in `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`. |
| 3 | `871c192` | BF16 sparse-MLA query | Already Jovian — GLM B12X attention preserves the model query path. |
| 4 | `4c6f16e` | FlashInfer prerequisite pin | Ported (source+test) — `requirements/cuda.txt`; `test_glm53_single_container.py::test_single_container_recipe_preserves_qualified_d16_contract`. |
| 5 | `389ec56` | Matching FlashInfer release set | Ported (source+test) — Python and cubin are both 0.6.18; recipe-contract test checks both pins. |
| 6 | `c1ea629` | No-RoPE MLA routing | Already Jovian — GLM model/attention configuration selects the supported no-RoPE path. |
| 7 | `ba41dd9` | Native SM120 sparse MLA | Already Jovian — native B12X sparse backend is present. |
| 8 | `e37b054` | Owned SM120 overlay | Excluded — overlay ownership/build plumbing, superseded by source-integrated Jovian code. |
| 9 | `a43f527` | Qualification gates | Excluded — fleet verifier gates; production capability gates are rows 60–61. |
| 10 | `d3d7cf4` | Runtime verifier repair | Excluded — image qualification verifier, not serving behavior. |
| 11 | `3066acb` | Exact token boundaries | Excluded — qualification-corpus construction. |
| 12 | `5d934ad` | Token repair alternatives | Excluded — exploratory qualification tooling. |
| 13 | `6a45004` | Reasoning-effort setting | Excluded — benchmark client option, unrelated to serving correctness. |
| 14 | `af3cca5` | Compressed DCP | Already Jovian — sparse DCP cache and index handling are in the base. |
| 15 | `a1a8599` | DCP sparse LSE merge | Already Jovian — B12X sparse DCP LSE merge is in the base backend. |
| 16 | `703b8f8` | Malformed LSE verifier case | Excluded — fleet image verifier test. |
| 17 | `33cd03b` | Gathered DCP decode | Already Jovian — DCP4 B12X decode path is in the base. |
| 18 | `ddaad43` | Flat MLA DCP metadata | Already Jovian — current sparse metadata supports the resolved layout. |
| 19 | `aad3480` | Proven serving lane | Excluded — fleet deployment lane; portable launch is documented locally. |
| 20 | `700fd25` | Exact-1M profile | Ported (source+test) — recipe uses `--max-model-len 1048576`; recipe-contract test asserts it. |
| 21 | `9926e8e` | DCP4 serving profile | Ported (source+test) — recipe carries TP4/DCP4/interleave4; recipe-contract test asserts all three. |
| 22 | `02f84b0` | Image provenance binding | Excluded — downstream image identity is environment-specific and intentionally absent from the sanitized generic recipe. |
| 23 | `1de96a2` | DCP config import shadowing | Excluded — fleet overlay import-order fix; source integration has no patch-module shadowing. |
| 24 | `112dbfb` | Unsupported DCP4 fail-closed | Ported (source+test) — `dcp_layout.py::validate_dcp_support`; DCP ownership tests. |
| 25 | `15545f6` | B12X DCP4 lane | Ported (source+test) — B12X/DCP4 recipe and exact capability-gate tests. |
| 26 | `8a23426` | NGC base identity | Excluded — downstream base-image provenance, not a portable repository dependency. |
| 27 | `a16bfc7` | Deferred image-label check | Excluded — image-build verifier sequencing. |
| 28 | `f096fc5` | SM120 block-FP8 fallback | Already Jovian — modelopt mixed quantization and B12X fallback are in the base. |
| 29 | `e1290cf` | B12X routing for MTP | Already Jovian — GLM MTP accepts B12X attention/MoE backends. |
| 30 | `626ef33` | GLM FP8 performance lane | Already Jovian — GLM-5.3 model, selective ModelOpt quantization, B12X attention/linear, Humming MoE, and PCIe all-reduce are present; recipe selects them. |
| 31 | `b093e99` | JIT toolchain verification | Excluded — image verifier behavior. |
| 32 | `f7a337b` | Device-free image verification | Excluded — image verifier behavior. |
| 33 | `3e6dd75` | Offline Torch verification | Excluded — image verifier process isolation. |
| 34 | `38df3e0` | DCP4/selective B12X linear | Already Jovian — selective quantization and B12X linear routing are in the base; recipe preserves both. |
| 35 | `5b2c471` | Verifier teardown handling | Excluded — verifier-only false-failure suppression. |
| 36 | `d049c20` | Request-shaped B12X metadata | Already Jovian — current B12X metadata builders retain request geometry. |
| 37 | `c0e07b1` | DCP1 speed profile | Excluded — alternate fleet profile; this artifact is specifically D16/DCP4. |
| 38 | `8a70f16` | Official LMCache payload | Ported (source+test) — `lmcache-d16-overlay/`; overlay CPU tests. |
| 39 | `44ec03c` | Complete sidecar payload | Ported (source+test) — all required integration, transfer, CUDA IPC, and cache-context modules are included and compile-tested. |
| 40 | `5ea3c37` | Unified Mamba cache views | Ported (source+test) — `kv_cache_group_edits.py`; overlay geometry tests. |
| 41 | `5097cbf` | Padded packed strides | Ported (source+test) — `patch_lmcache_padded_packed_stride.py` and CUDA patch; overlay source/geometry tests. |
| 42 | `e58dab9` | Slot-transfer physical strides | Ported (source+test) — `patch_lmcache_slot_stride.py` and CUDA patch; overlay tests. |
| 43 | `ea5c2e7` | CuPy registration dependency | Excluded — supplied by the downstream pinned LMCache base image, not added to vLLM requirements. |
| 44 | `17814ef` | Connector-only Mamba view | Ported (source+test) — edits are applied only during LMCache registration; overlay tests. |
| 45 | `1142a2e` | Hybrid engine-driven groups | Ported (source+test) — group-aware adapters and `kv_cache_groups.py`; authoritative-span test. |
| 46 | `80c9817` | Abort failed SHM stores | Ported (source+test) — `worker_transfer.py` aborts failed prepared stores; overlay source tests. |
| 47 | `b3da9d9` | SHM store backpressure | Ported (source+test) — engine-driven admission/commit flow is retained in `worker_transfer.py`; overlay source tests. |
| 48 | `731e79e` | Abort orphaned SHM writes | Ported (source+test) — exception and unsuccessful-commit paths call `abort_store`; overlay source tests. |
| 49 | `a3c0129` | cuMem KV sharing | Ported (source+test) — `cumem_ipc.py`, `ipc_wrapper.py`, and interposer; CPU descriptor/lifecycle tests. |
| 50 | `4858c66` | Cross-container cuMem broker | Ported (source+test) — same-path mode-0700 broker and SCM_RIGHTS transport; supervisor and overlay tests. |
| 51 | `0b55a21` | Imported mapping release | Ported (source+test) — refcounted `ImportedCuMemRegistry` and wrapper close; lifecycle tests. |
| 52 | `56965f8` | Preserve CUDA contexts | Ported (source+test) — cleanup unmaps/releases without `cudaDeviceReset`; artifact scan and overlay tests. |
| 53 | `5126fb6` | GDN FULL fail-closed gate | Ported (source+test) — `glm5next_cudagraph.py`, GDN capability gate, and bounded-gate tests. |
| 54 | `426bd7d` | Native FULL rejection record | Ported (source+test) — non-qualified configs return `UNIFORM_BATCH`; capability tests cover rejection. |
| 55 | `621c3a6` | Draft metadata under DCP | Already Jovian — current MTP/DCP scheduling and metadata rebuild are in the base. |
| 56 | `e0f8f9d` | Reject prefill from decode FULL | Ported (source+test) — legacy runner forces non-uniform routing and asserts against decode FULL; runner test. |
| 57 | `eb39710` | Prefill via mixed graph routing | Ported (source+test) — FULL/FULL_DECODE_ONLY/PIECEWISE dispatch honors prefill branch; graph/runner tests. |
| 58 | `95e8533` | Stable GDN metadata addresses | Ported (source+test) — `_PersistentGDNMetadataArena`; fixed-address/capacity test. |
| 59 | `9c9d73c` | Stable pooled selector replay | Ported (source+test) — `_PersistentPooledSelectorArena` plus vectorized all-row selection; selector arena test. |
| 60 | `65e9877` | LMCache DCP geometry | Ported (source+test) — `dcp_layout.py`; 9216/2304 group-geometry tests. |
| 61 | `bbf1efb` | Mixed FULL capability gate | Ported (source+test) — exact GLM/B12X/TP4/DCP4/MTP3 gate; exact-gate test. |
| 62 | `e5c0a70` | FULL capture capacity | Ported (source+test) — four-request/32768-token bounds; capacity test. |
| 63 | `e83f736` | KDA full-capture sizing | Ported (source+test) — KDA token arena includes maximum capture size; KDA plumbing tests. |
| 64 | `3d39898` | KDA plan/live capacities | Ported (source+test) — separate plan and live-request capacities with fail-closed checks; KDA tests. |
| 65 | `719313f` | Diagnostic target-only FULL | Excluded — target-only diagnostic mode is not a production serving mode. |
| 66 | `3c4e5b1` | Reuse KDA chunk indices | Ported (source+test) — precomputed indices thread through KDA; focused test. |
| 67 | `42aee72` | Reuse KDA chunk offsets | Ported (source+test) — precomputed offsets thread through every KDA layer; focused test. |
| 68 | `520aa0c` | Diagnostic speculator graphs | Excluded — diagnostic target/speculator isolation, not production routing. |
| 69 | `cc1cb5d` | Replay instrumentation | Excluded — fleet diagnostic telemetry. |
| 70 | `9df7bc5` | Replay checkpoint labels | Excluded — diagnostic checkpoint naming. |
| 71 | `51aef6d` | Capture-contract instrumentation | Excluded — fleet diagnostic instrumentation; production contract is enforced by gates/tests. |
| 72 | `3527013` | FULL execution-branch identity | Ported (source+test) — `TargetExecutionBranch` keys prefill/spec-decode/decode separately; compatibility tests. |
| 73 | `96598dc` | Mamba validator ABI | Ported (source+test) — one- and two-argument `validate_mamba_step_alignment`; overlay tests call both-compatible form. |
| 74 | `fb931d2` | Exact decode request count | Ported (source+test) — branch-specialized decode/spec graphs require exact live request counts; graph tests. |
| 75 | `a71bd4f` | Registration geometry | Ported (source+test) — authoritative per-group spans flow into `EngineGroupInfo`; registration-span test. |
| 76 | `cdcd087` | Single-container lifecycle | Ported (source+test) — PID-1 supervisor, health gates, process groups, sibling teardown; supervisor tests. |
| 77 | `56dce74` | Provenance environment | Excluded — concrete image IDs are deployment-specific; the generic supervisor only permits immutable provenance variables without embedding private values. |
| 78 | `7b12ac3` | Bounded vLLM readiness | Ported (source+test) — configurable readiness deadline and health URL; supervisor failure/startup tests. |
| 79 | `a429ae0` | Diagnostic store suppression | Excluded — fault-injection-only store suppression. |
| 80 | `1f1c771` | Isolated LMCache D2H path | Ported (source+test) — D2H policy is isolated per object group; focused policy test. |
| 81 | `8c61ef3` | Safe default D2H routing | Ported (source+test) — compressed groups default to legacy D2H, while H2D and uncompressed D2H remain native; focused policy test. |

## Overlap audit

All listed PRs target the same Jovian base, so this change avoids copying their
independent optimizations:

| PR | State at audit | Overlap disposition |
| ---: | --- | --- |
| #488 | Open | Broad B12X CKV/DCP/MTP work. This change touches the sparse metadata builder only to add the GLM FULL capability gate. |
| #498 | Open | mHC token-batch dispatch; no shared files. |
| #505 | Open | C4 prefill-pool kernel optimization. Shared selector files are changed here only for fixed-address, graph-safe metadata and all-row dispatch; no kernel changes are copied. |
| #510 | Open | Hybrid KV unbounded-anchor policy; no shared files. |
| #513 | Open | DFlash replicated DCP; shared B12X file change here is limited to the GLM capability gate. |
| #515 | Open | CUDA-graph profiling resource lifetime. Shared graph-manager files here implement branch/request identity only. |
| #517 | Open | Full C4 gather for DCP prefill. Shared B12X file change here is limited to the GLM capability gate. |
| #519 | Open | DFlash hybrid cache groups; no shared files. |
| #520 | Open | DFlash replicated-page alignment; no shared files. |
| #521 | Merged | Live-tensor B12X KDA is already in the base. This change only adds capture sizing, null-state initialization, and precomputed chunk metadata. |
