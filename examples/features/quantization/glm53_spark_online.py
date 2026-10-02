# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write exact GLM Spark projection and MTP online-quantization configurations.

The adjacent manifest records the selected projections in the pinned Spark
checkpoint. This generates configuration only; vLLM quantizes BF16 weights
during loading and does not rewrite checkpoint files.
"""

import argparse
import json
from pathlib import Path

SOURCE_MODEL = "local-inference-lab/GLM-5.3-Flash-NVFP4"
SOURCE_REVISION = "cfd47bd7680e68408924df09b179d5bed25b2ae9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=SOURCE_MODEL)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument(
        "--mtp-scale-compression",
        choices=("inherit", "native", "csf"),
        default="inherit",
    )
    args = parser.parse_args()
    manifest = json.loads(
        Path(__file__).with_name("glm53_spark_targets.json").read_text()
    )
    main_targets, mtp_targets = {}, {}
    source_prefix = "model.language_model."
    for name in manifest["targets"]:
        if not name.startswith(source_prefix):
            raise ValueError(f"Unexpected source projection name: {name}")
        if name.startswith(source_prefix + "layers.45."):
            mtp_targets[name.replace(source_prefix, "model.", 1)] = "mxfp8"
        else:
            main_targets[name.replace(source_prefix, "language_model.model.", 1)] = (
                "mxfp8"
            )
    if len(main_targets) != 520 or len(mtp_targets) != 11:
        raise ValueError(
            "The Spark manifest must define 520 main and 11 MTP projections"
        )
    for name in manifest["nvfp4_mtp_targets"]:
        mtp_targets[name.replace(source_prefix, "model.", 1)] = "nvfp4_a16"
    configurations = {
        "main.json": {"targets": main_targets, "strict_targets": True},
        "mtp.json": {
            "method": "mtp",
            "model": args.model,
            "revision": args.revision,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard",
            "moe_backend": "b12x",
            "attention_backend": "B12X",
            "moe_scale_compression": (
                None
                if args.mtp_scale_compression == "inherit"
                else args.mtp_scale_compression
            ),
            "quantization_config": {"targets": mtp_targets, "strict_targets": True},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in configurations.items():
        (args.output_dir / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(f"Wrote {args.output_dir}: 531 MXFP8 projections and NVFP4 W4A16 MTP experts")


if __name__ == "__main__":
    main()
