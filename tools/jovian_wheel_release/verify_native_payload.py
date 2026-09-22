#!/usr/bin/env python3
"""Reject missing, added or changed native payloads in a Python-only wheel."""

import argparse
import hashlib
import zipfile
from pathlib import Path


def native_payload(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("vllm/")
            and (name.endswith(".so") or name == "vllm/vllm-rs")
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = native_payload(args.reference)
    candidate = native_payload(args.candidate)
    if not reference or candidate != reference:
        changed = sorted(
            name
            for name in reference.keys() | candidate.keys()
            if reference.get(name) != candidate.get(name)
        )
        raise RuntimeError(f"Native wheel payload differs: {changed}")
    print(f"Verified identical native payloads: {len(reference)} files")


if __name__ == "__main__":
    main()
