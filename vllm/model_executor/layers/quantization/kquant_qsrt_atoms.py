# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reader for canonical TP-independent QSRT atom containers.

Each checkpoint layer stores one padded safetensors atom slab. Tensor
parallelism selects a contiguous whole-row extent at load time; the serialized
file never names a TP size or rank. Runtime loading narrows InstantTensor's
unopened I/O layout to that extent so no unrelated atom bytes cross the
host/device boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

KIMI_K3_SCHEMA = "kquant_kimi_k3_qsrt_atoms_v1"
FRUIT_SCHEMA = "kquant_fruit_qsrt_atoms_v1"
SUPPORTED_SCHEMAS = frozenset({KIMI_K3_SCHEMA, FRUIT_SCHEMA})
ENCODING = "qsrt_sqg_e4m3"
VERSION = 1
ATOM_CHANNELS = 32
ATOMS_PER_PAIR = 8
STORAGE_ALIGNMENT = 4096
FORMAT_X4T = 0xFF

FORMAT_TENSOR = "_qsrt_format_section"
SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
ATOM_TENSOR = "qsrt_atoms"
TENSOR_INVENTORY = {FORMAT_TENSOR, SHARED_SCALE_TENSOR, ATOM_TENSOR}

_SHA256_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_MARKER = "QSRT_COMPLETE.json"
_CHECKSUM_MANIFEST = "MANIFEST.sha256"
_PACKAGE_MANIFEST = "qsrt-manifest.json"


@dataclass(frozen=True)
class QSRTPublicationSeal:
    root: Path
    manifest: dict[str, Any]
    descriptor: dict[str, Any]
    checksums: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return value


def _sha256_field(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_DIGITS for character in value)
    ):
        raise ValueError(f"QSRT publication {name} is not a SHA-256 digest")
    return value


def verify_qsrt_publication(root: str | Path) -> QSRTPublicationSeal:
    """Authenticate a sealed local QSRT package before any atom extent is read."""

    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("QSRT model root must not be a symbolic link")
    try:
        root = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QSRT model root does not exist: {supplied}") from exc
    if not root.is_dir():
        raise ValueError("QSRT model root must be a directory")

    marker = _json_object(root / _PUBLICATION_MARKER, kind="QSRT completion marker")
    expected_marker_fields = {
        "schema",
        "package_manifest_sha256",
        "checksum_manifest_sha256",
        "model_index_sha256",
        "source",
        "base_manifest_sha256",
        "producer_fingerprint",
        "encoder_fingerprint",
    }
    if (
        set(marker) != expected_marker_fields
        or marker.get("schema") != "kquant_qsrt_complete_v2"
    ):
        raise ValueError("QSRT completion marker identity is invalid")
    for name in (
        "package_manifest_sha256",
        "checksum_manifest_sha256",
        "model_index_sha256",
        "base_manifest_sha256",
        "producer_fingerprint",
        "encoder_fingerprint",
    ):
        _sha256_field(marker.get(name), name=name)
    marker_source = marker.get("source")
    if (
        not isinstance(marker_source, dict)
        or set(marker_source) != {"kind", "sha256"}
        or not isinstance(marker_source.get("kind"), str)
        or not marker_source["kind"]
    ):
        raise ValueError("QSRT completion marker source identity is invalid")
    _sha256_field(marker_source.get("sha256"), name="source.sha256")

    checksum_path = root / _CHECKSUM_MANIFEST
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    if _sha256(checksum_path) != marker["checksum_manifest_sha256"]:
        raise ValueError("QSRT checksum manifest does not match the completion marker")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        relative = Path(filename)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in _SHA256_DIGITS for character in digest)
            or not filename
            or relative.is_absolute()
            or relative.as_posix() != filename
            or ".." in relative.parts
            or filename in checksums
        ):
            raise ValueError(f"invalid QSRT checksum entry: {line!r}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != digest:
            raise ValueError(f"QSRT package hash mismatch for {filename}")
        checksums[filename] = digest
    published_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"QSRT package must not contain symbolic links: {path}")
        if path.is_file():
            published_files.add(path.relative_to(root).as_posix())
    expected_files = set(checksums) | {_CHECKSUM_MANIFEST, _PUBLICATION_MARKER}
    if published_files != expected_files:
        raise ValueError("QSRT checksum inventory does not match the package files")
    required_files = {"config.json", "model.safetensors.index.json", _PACKAGE_MANIFEST}
    if not required_files.issubset(checksums):
        raise ValueError("QSRT checksum manifest omits a required identity file")
    if checksums[_PACKAGE_MANIFEST] != marker["package_manifest_sha256"]:
        raise ValueError(
            "QSRT package manifest digest disagrees with completion marker"
        )
    if checksums["model.safetensors.index.json"] != marker["model_index_sha256"]:
        raise ValueError("QSRT model index digest disagrees with completion marker")

    manifest = _json_object(root / _PACKAGE_MANIFEST, kind="QSRT package manifest")
    producer = manifest.get("producer")
    source = manifest.get("source")
    base_model = manifest.get("base_model")
    if (
        manifest.get("schema") != "kquant_qsrt_model_manifest_v1"
        or manifest.get("version") != 1
        or manifest.get("complete") is not True
        or not isinstance(producer, dict)
        or not isinstance(producer.get("encoder"), dict)
        or not isinstance(source, dict)
        or not isinstance(base_model, dict)
    ):
        raise ValueError("QSRT package manifest identity is invalid or incomplete")
    expected_source = {
        "kind": source.get("source_kind"),
        "sha256": source.get("source_sha256"),
    }
    if marker_source != expected_source:
        raise ValueError(
            "QSRT package source identity disagrees with completion marker"
        )
    if (
        producer.get("fingerprint") != marker["producer_fingerprint"]
        or producer["encoder"].get("fingerprint") != marker["encoder_fingerprint"]
        or base_model.get("manifest_sha256") != marker["base_manifest_sha256"]
    ):
        raise ValueError("QSRT producer identity disagrees with completion marker")

    config = _json_object(root / "config.json", kind="QSRT model config")
    quantization = config.get("quantization_config")
    descriptor = quantization.get("qsrt") if isinstance(quantization, dict) else None
    if not isinstance(descriptor, dict):
        raise ValueError("QSRT model config omits its format descriptor")
    expected_descriptor = {
        "schema": manifest.get("storage_schema"),
        "storage_format": "qsrt_atoms_v1",
        "encoding": manifest.get("encoding"),
        "codebook": manifest.get("codebook"),
        "profile_id": manifest.get("profile_id"),
        "artifact_manifest": _PACKAGE_MANIFEST,
        "producer_fingerprint": marker["producer_fingerprint"],
        "encoder_fingerprint": marker["encoder_fingerprint"],
        "source_kind": marker_source["kind"],
        "source_sha256": marker_source["sha256"],
    }
    if any(
        descriptor.get(name) != value for name, value in expected_descriptor.items()
    ):
        raise ValueError("QSRT model descriptor disagrees with the sealed manifest")
    return QSRTPublicationSeal(
        root=root,
        manifest=manifest,
        descriptor=descriptor,
        checksums=checksums,
    )


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"QSRT metadata {name!r} is missing or invalid") from exc
    return value


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class QSRTAtomLayerMetadata:
    path: Path
    schema: str
    profile_id: int
    codebook: str | None
    source_sha256: str | None
    encoder_fingerprint: str | None
    layer: int
    experts: int
    hidden_size: int
    intermediate_size: int
    atom_slots: int
    atom_bundle_bytes: int
    rotation_multiplier: int
    format_codes: torch.Tensor
    compressed_expert_ids: torch.Tensor
    x4t_expert_ids: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    atom_slot_stride_bytes: int

    @property
    def compressed_experts(self) -> int:
        return int(self.compressed_expert_ids.numel())

    @property
    def x4t_experts(self) -> int:
        return int(self.x4t_expert_ids.numel())

    @property
    def atom_slot_payload_bytes(self) -> int:
        return self.compressed_experts * self.atom_bundle_bytes


def read_qsrt_atom_layer_metadata(
    path: str | Path,
    *,
    layer: int,
    expected_experts: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
    expected_bits: Sequence[int] | None = None,
    expected_profile_id: int | None = None,
    expected_codebook: str | None = None,
    expected_source_sha256: str | None = None,
    expected_encoder_fingerprint: str | None = None,
) -> QSRTAtomLayerMetadata:
    """Validate one canonical atom layer without reading its payload slab.

    Args:
        path: Safetensors atom-layer file.
        layer: Expected model layer index.
        expected_experts: Expected global expert count.
        expected_hidden_size: Expected hidden-channel dimension.
        expected_intermediate_size: Expected global intermediate-channel dimension.
        expected_bits: Optional per-expert bit-map contract.
        expected_profile_id: Optional Fruit profile identifier.
        expected_codebook: Optional reconstruction codebook identifier.
        expected_source_sha256: Optional authenticated source digest.
        expected_encoder_fingerprint: Optional encoder provenance fingerprint.

    Returns:
        Validated metadata and the small format, ID, and scale tensors.

    Raises:
        ValueError: If metadata, tensor inventory, geometry, alignment, format
            assignments, identifiers, or small tensor contents are noncanonical.
        FileNotFoundError: If ``path`` does not identify a readable layer file.
    """

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != TENSOR_INVENTORY:
            raise ValueError("QSRT atom tensor inventory is noncanonical")
        schema = metadata.get("schema")
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unsupported QSRT atom schema: {schema!r}")
        expected_metadata = {
            "format": "pt",
            "version": str(VERSION),
            "encoding": ENCODING,
            "layer": str(layer),
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"QSRT metadata {name} mismatch: "
                    f"{metadata.get(name)!r} != {expected!r}"
                )

        profile_value = metadata.get("profile_id")
        profile_id = (
            _metadata_int(metadata, "profile_id") if profile_value is not None else None
        )
        codebook = metadata.get("codebook")
        source_sha256 = metadata.get("source_sha256")
        encoder_fingerprint = metadata.get("encoder_fingerprint")
        if schema == FRUIT_SCHEMA and (
            profile_id is None
            or codebook is None
            or source_sha256 is None
            or encoder_fingerprint is None
        ):
            raise ValueError("Fruit QSRT identity metadata is incomplete")
        if profile_id is not None and profile_id <= 0:
            raise ValueError("QSRT profile_id must be positive")
        if expected_profile_id is not None and profile_id != expected_profile_id:
            raise ValueError(
                f"QSRT profile_id mismatch: {profile_id} != {expected_profile_id}"
            )
        for name, actual, expected in (
            ("codebook", codebook, expected_codebook),
            ("source_sha256", source_sha256, expected_source_sha256),
            (
                "encoder_fingerprint",
                encoder_fingerprint,
                expected_encoder_fingerprint,
            ),
        ):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"QSRT metadata {name} mismatch: {actual!r} != {expected!r}"
                )
        experts = _metadata_int(metadata, "experts")
        intermediate_size = _metadata_int(metadata, "intermediate_channels")
        hidden_size = _metadata_int(metadata, "latent_channels")
        atom_channels = _metadata_int(metadata, "atom_channels")
        atom_slots = _metadata_int(metadata, "atom_slots")
        atom_bundle_bytes = _metadata_int(metadata, "atom_bundle_bytes")
        alignment = _metadata_int(metadata, "alignment_bytes")
        if experts != expected_experts:
            raise ValueError(f"QSRT experts mismatch: {experts} != {expected_experts}")
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"QSRT latent channels mismatch: "
                f"{hidden_size} != {expected_hidden_size}"
            )
        if intermediate_size != expected_intermediate_size:
            raise ValueError(
                f"QSRT intermediate channels mismatch: "
                f"{intermediate_size} != {expected_intermediate_size}"
            )
        if atom_channels != ATOM_CHANNELS:
            raise ValueError(
                f"QSRT atom channels must be {ATOM_CHANNELS}, got {atom_channels}"
            )
        if atom_slots <= 0 or atom_slots % ATOMS_PER_PAIR:
            raise ValueError("QSRT atom slots must be a positive multiple of eight")
        if intermediate_size != atom_slots * atom_channels:
            raise ValueError("QSRT atom geometry does not cover the intermediate axis")
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("QSRT latent channels must be a positive multiple of 128")
        expected_bundle_bytes = (
            3 * (atom_channels * hidden_size * 3 // 8)
            + 3 * atom_channels * torch.float16.itemsize
        )
        if atom_bundle_bytes != expected_bundle_bytes:
            raise ValueError(
                f"QSRT atom bundle bytes mismatch: "
                f"{atom_bundle_bytes} != {expected_bundle_bytes}"
            )
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("QSRT alignment must be a positive power of two")

        pair_count = atom_slots // ATOMS_PER_PAIR
        if schema == KIMI_K3_SCHEMA:
            rotation_multiplier = 5
        else:
            if _metadata_int(metadata, "pair_count") != pair_count:
                raise ValueError("QSRT pair_count disagrees with atom geometry")
            rotation_multiplier = _metadata_int(metadata, "rotation_multiplier")
            if rotation_multiplier < 0:
                raise ValueError("QSRT rotation_multiplier must be nonnegative")

        compressed_count = _metadata_int(metadata, "compressed_experts")
        x4t_count = _metadata_int(metadata, "x4t_experts")
        slot_payload = _metadata_int(metadata, "atom_slot_payload_bytes")
        slot_stride = _metadata_int(metadata, "atom_slot_stride_bytes")
        if compressed_count + x4t_count != experts:
            raise ValueError("QSRT tier counts do not cover all experts")
        if slot_payload != compressed_count * atom_bundle_bytes:
            raise ValueError("QSRT atom-slot payload byte count is invalid")
        if slot_stride < slot_payload or slot_stride % alignment:
            raise ValueError("QSRT atom-slot stride is invalid")

        format_section_bytes = _align(experts, alignment)
        format_section = handle.get_tensor(FORMAT_TENSOR)
        if (
            format_section.dtype != torch.uint8
            or tuple(format_section.shape) != (format_section_bytes,)
            or bool(torch.any(format_section[experts:] != 0))
        ):
            raise ValueError("QSRT format section is malformed")
        format_codes = format_section[:experts].clone().contiguous()
        r13 = format_codes >> 4
        r2 = format_codes & 0xF
        compressed_mask = format_codes != FORMAT_X4T
        if bool(torch.any(compressed_mask & ((r13 > 2) | (r2 > 2)))):
            raise ValueError("QSRT format table contains an invalid rate code")
        compressed_ids = torch.nonzero(compressed_mask, as_tuple=False).flatten()
        x4t_ids = torch.nonzero(~compressed_mask, as_tuple=False).flatten()
        if (
            int(compressed_ids.numel()) != compressed_count
            or int(x4t_ids.numel()) != x4t_count
        ):
            raise ValueError("QSRT format table tier counts disagree with metadata")
        if expected_bits is not None:
            if len(expected_bits) != experts:
                raise ValueError(f"hybrid_bit_map must describe all {experts} experts")
            if any(bit not in (3, 4) for bit in expected_bits):
                raise ValueError("QSRT hybrid_bit_map entries must be 3 or 4")
            expected_x4t = torch.tensor(expected_bits, dtype=torch.int16) == 4
            if not torch.equal(expected_x4t, ~compressed_mask):
                raise ValueError("QSRT format table disagrees with hybrid_bit_map")

        shared_scale_rows = 1 if schema == KIMI_K3_SCHEMA else experts
        shared_scale_bytes = (
            3 * shared_scale_rows * hidden_size * torch.float16.itemsize
        )
        shared_scale_section_bytes = _align(shared_scale_bytes, alignment)
        if (
            schema == FRUIT_SCHEMA
            and _metadata_int(metadata, "shared_scale_section_bytes")
            != shared_scale_section_bytes
        ):
            raise ValueError("QSRT shared-scale byte count disagrees with geometry")
        shared = handle.get_tensor(SHARED_SCALE_TENSOR)
        if (
            shared.dtype != torch.uint8
            or tuple(shared.shape) != (shared_scale_section_bytes,)
            or bool(torch.any(shared[shared_scale_bytes:] != 0))
        ):
            raise ValueError("QSRT shared-scale section is malformed")
        vectors = (
            shared[:shared_scale_bytes]
            .clone()
            .view(torch.float16)
            .reshape(3, shared_scale_rows, hidden_size)
            .contiguous()
        )
        if shared_scale_rows == 1:
            vectors = vectors[:, 0]
        if not bool(torch.all(torch.isfinite(vectors))):
            raise ValueError("QSRT shared scales contain non-finite values")

        atom_shape = handle.get_slice(ATOM_TENSOR).get_shape()
        expected_atom_shape = [atom_slots, slot_stride]
        if atom_shape != expected_atom_shape:
            raise ValueError(
                f"QSRT atom slab shape {atom_shape} != {expected_atom_shape}"
            )

    return QSRTAtomLayerMetadata(
        path=path,
        schema=schema,
        profile_id=profile_id,
        layer=layer,
        codebook=codebook,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
        experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        atom_slots=atom_slots,
        atom_bundle_bytes=atom_bundle_bytes,
        rotation_multiplier=rotation_multiplier,
        format_codes=format_codes,
        compressed_expert_ids=compressed_ids.to(torch.int32).contiguous(),
        x4t_expert_ids=x4t_ids.to(torch.int32).contiguous(),
        gate_suh=vectors[0].contiguous(),
        up_suh=vectors[1].contiguous(),
        down_svh=vectors[2].contiguous(),
        atom_slot_stride_bytes=slot_stride,
    )


def balanced_atom_partition(
    atom_slots: int,
    shard_count: int,
    shard_index: int,
) -> tuple[int, int]:
    if not 1 <= shard_count <= atom_slots:
        raise ValueError(f"shard_count must lie in 1..{atom_slots}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    quotient, remainder = divmod(atom_slots, shard_count)
    count = quotient + int(shard_index < remainder)
    first = shard_index * quotient + min(shard_index, remainder)
    return first, count


def _select_instanttensor_extent(
    handle: Any,
    *,
    tensor_name: str,
    first_row: int,
    rows: int,
    row_bytes: int,
    total_rows: int,
) -> None:
    """Restrict an unopened InstantTensor handle to one tensor row extent."""

    required = (
        "ordered_tensor_metadatas",
        "tensor_offsets",
        "tensor_sizes",
        "total_tensor_size",
        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
    )
    missing = [name for name in required if not hasattr(handle, name)]
    if missing:
        raise RuntimeError(
            "Installed InstantTensor cannot select a QSRT atom extent "
            f"(missing: {', '.join(missing)})"
        )
    if handle.loader_handle is not None:
        raise RuntimeError("QSRT extent selection must occur before InstantTensor I/O")
    try:
        tensor_index = handle.tensor_name_to_index[tensor_name]
    except KeyError as exc:
        raise ValueError(f"QSRT file omits tensor {tensor_name!r}") from exc
    file_index, tensor_start = handle.tensor_offsets[tensor_index]
    end_file_index, tensor_end = handle.tensor_offsets[tensor_index + 1]
    if (
        file_index != end_file_index
        or tensor_end - tensor_start != total_rows * row_bytes
    ):
        raise RuntimeError("InstantTensor QSRT atom offsets disagree with metadata")
    begin = tensor_start + first_row * row_bytes
    end = begin + rows * row_bytes
    metadata = {
        "dtype": "U8",
        "shape": [rows, row_bytes],
        "data_offsets": [0, end - begin],
    }
    handle.ordered_tensor_metadatas = [(tensor_name, metadata)]
    handle.tensor_name_to_index = {tensor_name: 0}
    handle.tensor_offsets = [(file_index, begin), (file_index, end)]
    handle.tensor_sizes = [end - begin]
    handle.total_tensor_size = end - begin
    handle._determine_buffer_size(None)


@contextmanager
def open_qsrt_atom_extent(
    metadata: QSRTAtomLayerMetadata,
    *,
    shard_count: int,
    shard_index: int,
    device: torch.device | str | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield one shard's ``(first_atom_slot, [A,E,B]u8)`` extent.

    CPU mode exists for focused tests. CUDA mode uses InstantTensor and keeps
    the returned tensor valid only for the context lifetime; callers must
    complete preparation before leaving the context.
    """

    first, rows = balanced_atom_partition(metadata.atom_slots, shard_count, shard_index)
    if device is None or torch.device(device).type == "cpu":
        with safe_open(metadata.path, framework="pt", device="cpu") as handle:
            padded = handle.get_slice(ATOM_TENSOR)[first : first + rows]
            compact = padded[:, : metadata.atom_slot_payload_bytes].contiguous()
            yield (
                first,
                compact.reshape(
                    rows,
                    metadata.compressed_experts,
                    metadata.atom_bundle_bytes,
                ),
            )
        return

    import instanttensor

    opener = instanttensor.safe_open(
        str(metadata.path),
        framework="pt",
        device=torch.device(device),
        load_now=False,
        copy=False,
    )
    _select_instanttensor_extent(
        opener,
        tensor_name=ATOM_TENSOR,
        first_row=first,
        rows=rows,
        row_bytes=metadata.atom_slot_stride_bytes,
        total_rows=metadata.atom_slots,
    )
    with opener as handle:
        loaded = dict(handle.tensors())
        padded = loaded[ATOM_TENSOR]
        compact = padded[:, : metadata.atom_slot_payload_bytes]
        # Preserve the padded row stride as a zero-copy view. B12X extracts
        # each matrix component into its prepared owner before this context
        # exits, so carrying padding through the read avoids a second full
        # atom-slab allocation.
        atoms = compact.as_strided(
            (
                rows,
                metadata.compressed_experts,
                metadata.atom_bundle_bytes,
            ),
            (
                metadata.atom_slot_stride_bytes,
                metadata.atom_bundle_bytes,
                1,
            ),
        )
        yield first, atoms


__all__ = [
    "FRUIT_SCHEMA",
    "KIMI_K3_SCHEMA",
    "SUPPORTED_SCHEMAS",
    "QSRTAtomLayerMetadata",
    "balanced_atom_partition",
    "open_qsrt_atom_extent",
    "read_qsrt_atom_layer_metadata",
]
