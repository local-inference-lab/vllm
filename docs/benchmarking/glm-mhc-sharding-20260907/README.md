# Token-sharded mHC prefill: serving evidence

Status: **research-only**. Four GB10 GPUs running GLM-5.3-Flash with
TP4/DCP4 were tested under all four combinations of continuation coalescing
and token-sharded mHC. [Results](report.md) summarize 36 cold prefill samples,
20 exact-answer/cache checks, and 16 decode cells. [Measurements](evidence.json)
retain individual timings, activation witnesses, settings, and artifact hashes.

[Source composition](manifest.json) identifies the standalone feature revision
and the combined runtime used for measurements. Combining the pinned feature
revisions in that manifest reproduces the measured runtime and test subtrees.
The separately supplied benchmark client and its tests are reproduction tools,
not part of the measured server image.

These observations cover bounded serving checks, not full model-quality
qualification. The table includes disabled and enabled feature combinations
on the same image. Native 512-token split-page geometry, diagnostic logging,
and fused SparkRing transport are constant. Reboots between configurations and
short, sequential measurement windows limit small-difference conclusions.

The JSON preserves private-source artifact hashes for audit while excluding
request text, private addresses and deployment credentials. The component
implementation does not require the private deployment tooling.

[Reproduction instructions](reproduction.md) pin the model and benchmark inputs.
