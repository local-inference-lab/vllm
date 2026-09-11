# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only validation for the versioned TrellisMX routed overlay."""

import hashlib
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re

_PREFIX = re.compile(
    r"(?:model\.language_model|language_model\.model|model)"
    r"\.layers\.(\d+)\.mlp\.experts"
)


def routed_layer(prefix: str) -> int | None:
    match = _PREFIX.fullmatch(prefix)
    if match is None:
        return None
    layer = int(match[1])
    return layer if 3 <= layer <= 44 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file(root: Path, relative: str) -> Path:
    path = root / relative
    # Symlinked local checkpoints are supported, but traversal in manifests is not.
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe TrellisMX path: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing TrellisMX artifact: {path}")
    return path


@dataclass(frozen=True)
class Overlay:
    root: Path
    records: dict
    transform_hash: str

    def sidecar(self, layer: int, rank: int, *, verify: bool = True) -> Path:
        record = self.records[layer, rank]
        path = _local_file(self.root, record["path"])
        if verify and sha256(path) != record["sha256"]:
            raise ValueError(f"TrellisMX weight hash mismatch: {path}")
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if not 2 <= length <= 16 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(stream.read(length))
        metadata = header.get("__metadata__", {})
        expected = {
            "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
            "layer": str(layer),
            "rank": str(rank),
            "world_size": "4",
            "bits": str(record["bits"]),
            "alphabet": "e4m3",
            "scale": "ue8m0-k32",
            "law": "procedural-mcg-alpha2",
            "source_design_sha256": record["source_design_sha256"],
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"TrellisMX {path}: incompatible {name}")
        return path


@lru_cache(maxsize=4)
def load_overlay(directory: str) -> Overlay:
    root = Path(directory).absolute()
    manifest = json.loads(_local_file(root, "trellismx-manifest.json").read_text())
    if manifest.get("schema") != "trellismx.hf-overlay-release.v1":
        raise ValueError("Unsupported TrellisMX overlay schema")
    if manifest.get("carrier") != {
        "repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4",
        "revision": "520de24eabf507659eaef7c70f14fd584527facc",
    }:
        raise ValueError("Unsupported TrellisMX carrier identity")
    allocation = manifest.get("allocation", {})
    if set(allocation) != {str(n) for n in range(3, 45)}:
        raise ValueError("GLM TrellisMX requires every routed layer 3..44")
    if any(type(bits) is not int or bits not in (4, 5) for bits in allocation.values()):
        raise ValueError("This TrellisMX adapter supports K4/K5 only")
    designs = {
        sha256(_local_file(root, f"design/design-{index}.json")) for index in range(3)
    }
    transform = _local_file(root, "design/transform.json")
    transform_config = json.loads(transform.read_text())
    if (
        transform_config.get("boundary") != "coupled-h512-h128-suh-svh-v1"
        or transform_config.get("sign_draw") != 0
        or transform_config.get("activation") != "silu-cap10"
    ):
        raise ValueError("Unsupported TrellisMX coupled transform")
    records = {}
    for record in manifest.get("files", []):
        key = record["layer"], record["rank"]
        if key in records or key[0] not in range(3, 45) or key[1] not in range(4):
            raise ValueError("Duplicate or invalid TrellisMX layer/rank")
        expected_path = f"sidecars/p8-layer-{key[0]:03d}-tp4-rank-{key[1]}.safetensors"
        if record["path"] != expected_path:
            raise ValueError("Unexpected TrellisMX sidecar path")
        if record["bits"] != allocation[str(key[0])]:
            raise ValueError("TrellisMX allocation and sidecar rate disagree")
        if record["source_design_sha256"] not in designs:
            raise ValueError("TrellisMX source design outside allowlist")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("Invalid TrellisMX weight hash")
        if _local_file(root, record["path"]).stat().st_size != record["bytes"]:
            raise ValueError("TrellisMX sidecar size mismatch")
        records[key] = record
    if set(records) != {(layer, rank) for layer in range(3, 45) for rank in range(4)}:
        raise ValueError("Incomplete TrellisMX TP4 inventory")
    return Overlay(root, records, sha256(transform))
