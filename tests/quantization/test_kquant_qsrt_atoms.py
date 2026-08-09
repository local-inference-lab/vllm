# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
    FRUIT_SCHEMA,
    KIMI_K3_SCHEMA,
    balanced_atom_partition,
    open_qsrt_atom_extent,
    read_qsrt_atom_layer_metadata,
    verify_qsrt_publication,
)


def _write_test_atom_layer(
    tmp_path: Path,
    *,
    schema: str = FRUIT_SCHEMA,
    atom_value: int = 0,
) -> Path:
    layer = 3
    experts = 3
    hidden_size = 1024
    intermediate_size = 512
    atom_slots = 16
    atom_bundle_bytes = 37_056
    alignment = 4096
    slot_payload_bytes = experts * atom_bundle_bytes
    slot_stride_bytes = (slot_payload_bytes + alignment - 1) // alignment * alignment
    format_section = torch.zeros(alignment, dtype=torch.uint8)
    format_section[:experts] = torch.tensor([0x00, 0x11, 0x21], dtype=torch.uint8)
    shared_scale_rows = 1 if schema == KIMI_K3_SCHEMA else experts
    shared_scale_bytes = 3 * shared_scale_rows * hidden_size * 2
    shared_scale_section_bytes = (
        (shared_scale_bytes + alignment - 1) // alignment * alignment
    )
    shared_scale_section = torch.zeros(shared_scale_section_bytes, dtype=torch.uint8)
    shared_scale_section[:shared_scale_bytes].copy_(
        torch.ones((3, shared_scale_rows, hidden_size), dtype=torch.float16)
        .view(torch.uint8)
        .reshape(-1)
    )
    metadata = {
        "format": "pt",
        "schema": schema,
        "version": "1",
        "encoding": "qsrt_sqg_e4m3",
        "layer": str(layer),
        "experts": str(experts),
        "compressed_experts": str(experts),
        "x4t_experts": "0",
        "intermediate_channels": str(intermediate_size),
        "latent_channels": str(hidden_size),
        "atom_channels": "32",
        "atom_slots": str(atom_slots),
        "atom_bundle_bytes": str(atom_bundle_bytes),
        "atom_slot_payload_bytes": str(slot_payload_bytes),
        "atom_slot_stride_bytes": str(slot_stride_bytes),
        "alignment_bytes": str(alignment),
        "pair_count": "2",
        "rotation_multiplier": "5",
        "shared_scale_section_bytes": str(shared_scale_section_bytes),
    }
    if schema == FRUIT_SCHEMA:
        metadata.update(
            {
                "codebook": "sqg_xor_cheb_t12",
                "source_sha256": "a" * 64,
                "encoder_fingerprint": "b" * 64,
                "profile_id": "1",
            }
        )
    path = tmp_path / "qsrt-layer-003.safetensors"
    save_file(
        {
            "_qsrt_format_section": format_section,
            "_qsrt_shared_scale_section": shared_scale_section,
            "qsrt_atoms": torch.full(
                (atom_slots, slot_stride_bytes),
                atom_value,
                dtype=torch.uint8,
            ),
        },
        path,
        metadata=metadata,
    )
    return path


def test_reads_and_partitions_fruit_qsrt_atoms(tmp_path: Path) -> None:
    layer = 3
    experts = 3
    hidden_size = 1024
    intermediate_size = 512
    path = _write_test_atom_layer(tmp_path)

    metadata = read_qsrt_atom_layer_metadata(
        path,
        layer=layer,
        expected_experts=experts,
        expected_hidden_size=hidden_size,
        expected_intermediate_size=intermediate_size,
        expected_bits=[3, 3, 3],
        expected_profile_id=1,
        expected_codebook="sqg_xor_cheb_t12",
        expected_source_sha256="a" * 64,
        expected_encoder_fingerprint="b" * 64,
    )
    assert metadata.schema == FRUIT_SCHEMA
    assert metadata.atom_slots == 16
    assert metadata.atom_bundle_bytes == 37_056
    assert metadata.rotation_multiplier == 5
    assert metadata.codebook == "sqg_xor_cheb_t12"
    assert metadata.source_sha256 == "a" * 64
    assert metadata.encoder_fingerprint == "b" * 64
    assert metadata.compressed_expert_ids.tolist() == [0, 1, 2]
    assert balanced_atom_partition(16, 2, 1) == (8, 8)

    with open_qsrt_atom_extent(
        metadata,
        shard_count=2,
        shard_index=1,
        device="cpu",
    ) as (first_atom_slot, atoms):
        assert first_atom_slot == 8
        assert atoms.shape == (8, 3, 37_056)
        assert atoms.is_contiguous()
    with open_qsrt_atom_extent(
        metadata,
        shard_count=2,
        shard_index=0,
        device="cpu",
        atom_offset=3,
        atom_count=2,
    ) as (first_atom_slot, atoms):
        assert first_atom_slot == 3
        assert atoms.shape == (2, 3, 37_056)
        assert atoms.is_contiguous()


def test_reads_kimi_atoms_without_fruit_identity_metadata(
    tmp_path: Path,
) -> None:
    path = _write_test_atom_layer(tmp_path, schema=KIMI_K3_SCHEMA)

    metadata = read_qsrt_atom_layer_metadata(
        path,
        layer=3,
        expected_experts=3,
        expected_hidden_size=1024,
        expected_intermediate_size=512,
        expected_bits=[3, 3, 3],
    )

    assert metadata.schema == KIMI_K3_SCHEMA
    assert metadata.profile_id is None
    assert metadata.codebook is None
    assert metadata.source_sha256 is None
    assert metadata.encoder_fingerprint is None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_publication(root: Path):
    return verify_qsrt_publication(
        root,
        expected_complete_sha256=_sha256(root / "QSRT_COMPLETE.json"),
    )


def _write_test_publication(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    producer_fingerprint = "1" * 64
    encoder_fingerprint = "2" * 64
    source_sha256 = "3" * 64
    base_manifest_sha256 = "4" * 64
    descriptor: dict[str, object] = {
        "schema": FRUIT_SCHEMA,
        "storage_format": "qsrt_atoms_v1",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "profile_id": 1,
        "artifact_manifest": "qsrt-manifest.json",
        "producer_fingerprint": producer_fingerprint,
        "encoder_fingerprint": encoder_fingerprint,
        "source_kind": "safetensors_manifest",
        "source_sha256": source_sha256,
        "runtime": "w4a8",
    }
    (root / "config.json").write_text(
        json.dumps({"quantization_config": {"qsrt": descriptor}}),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "schema": "kquant_qsrt_model_manifest_v1",
        "version": 1,
        "codec": "QSRT",
        "storage_schema": FRUIT_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "profile_id": 1,
        "complete": True,
        "producer": {
            "fingerprint": producer_fingerprint,
            "encoder": {"fingerprint": encoder_fingerprint},
        },
        "source": {
            "source_kind": "safetensors_manifest",
            "source_sha256": source_sha256,
        },
        "base_model": {"manifest_sha256": base_manifest_sha256},
        "layers": {},
    }
    (root / "qsrt-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = root / "qsrt-layer-003.safetensors"
    payload.write_bytes(b"sealed atom payload")
    checked = (
        "config.json",
        "model.safetensors.index.json",
        "qsrt-layer-003.safetensors",
        "qsrt-manifest.json",
    )
    (root / "MANIFEST.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in checked),
        encoding="utf-8",
    )
    marker = {
        "schema": "kquant_qsrt_complete_v2",
        "package_manifest_sha256": _sha256(root / "qsrt-manifest.json"),
        "checksum_manifest_sha256": _sha256(root / "MANIFEST.sha256"),
        "model_index_sha256": _sha256(root / "model.safetensors.index.json"),
        "source": {
            "kind": "safetensors_manifest",
            "sha256": source_sha256,
        },
        "base_manifest_sha256": base_manifest_sha256,
        "producer_fingerprint": producer_fingerprint,
        "encoder_fingerprint": encoder_fingerprint,
    }
    (root / "QSRT_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
    return manifest, descriptor, payload


def test_verifies_complete_qsrt_publication_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    manifest, descriptor, payload = _write_test_publication(tmp_path)

    seal = _verify_publication(tmp_path)
    assert seal.manifest == manifest
    assert seal.descriptor == descriptor

    payload.write_bytes(b"mutated atom payload")
    with pytest.raises(ValueError, match="package hash mismatch"):
        _verify_publication(tmp_path)
    seal.close()


def test_publication_requires_independent_completion_digest(tmp_path: Path) -> None:
    _write_test_publication(tmp_path)
    trusted = _sha256(tmp_path / "QSRT_COMPLETE.json")

    with pytest.raises(ValueError, match="expected completion marker"):
        verify_qsrt_publication(tmp_path)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        verify_qsrt_publication(tmp_path, expected_complete_sha256="A" * 64)
    with pytest.raises(ValueError, match="trusted digest"):
        verify_qsrt_publication(
            tmp_path,
            expected_complete_sha256="0" * 64,
        )

    seal = verify_qsrt_publication(
        tmp_path,
        expected_complete_sha256=trusted,
    )
    seal.close()


def test_publication_rejects_self_resealed_substitution(
    tmp_path: Path,
) -> None:
    _manifest, _descriptor, payload = _write_test_publication(tmp_path)
    trusted = _sha256(tmp_path / "QSRT_COMPLETE.json")
    payload.write_bytes(b"attacker-controlled replacement")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="trusted digest"):
        verify_qsrt_publication(
            tmp_path,
            expected_complete_sha256=trusted,
        )


def test_authenticated_atom_inode_survives_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    model.mkdir()
    original.mkdir()
    replacement.mkdir()
    original_atom = _write_test_atom_layer(original)
    manifest, _descriptor, published_atom = _write_test_publication(model)
    published_atom.write_bytes(original_atom.read_bytes())
    manifest["layers"] = {
        "3": {
            "qsrt_atoms": published_atom.name,
            "sha256": _sha256(published_atom),
        }
    }
    (model / "qsrt-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_publication_identity(model)
    seal = _verify_publication(model)

    metadata = read_qsrt_atom_layer_metadata(
        published_atom,
        layer=3,
        expected_experts=3,
        expected_hidden_size=1024,
        expected_intermediate_size=512,
        expected_bits=[3, 3, 3],
        publication=seal,
        published_name=published_atom.name,
    )
    replacement_atom = _write_test_atom_layer(replacement, atom_value=7)
    replacement_atom.replace(published_atom)

    with open_qsrt_atom_extent(
        metadata,
        shard_count=2,
        shard_index=1,
        device="cpu",
    ) as (_first, atoms):
        assert not bool(torch.any(atoms))

    opened_paths: list[str] = []

    class FakeInstantTensorOpen:
        def __init__(self) -> None:
            start = 4096
            end = start + metadata.atom_slots * metadata.atom_slot_stride_bytes
            self.ordered_tensor_metadatas = []
            self.tensor_offsets = [(0, start), (0, end)]
            self.tensor_sizes = [end - start]
            self.total_tensor_size = end - start
            self.tensor_name_to_index = {"qsrt_atoms": 0}
            self.loader_handle = None

        def _determine_buffer_size(self, _value: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def tensors(self):
            rows = self.ordered_tensor_metadatas[0][1]["shape"][0]
            return [
                (
                    "qsrt_atoms",
                    torch.zeros(
                        (rows, metadata.atom_slot_stride_bytes),
                        dtype=torch.uint8,
                    ),
                )
            ]

    instanttensor = ModuleType("instanttensor")

    def fake_safe_open(path: str, **_kwargs):
        opened_paths.append(path)
        return FakeInstantTensorOpen()

    setattr(instanttensor, "safe_open", fake_safe_open)
    monkeypatch.setitem(sys.modules, "instanttensor", instanttensor)
    with open_qsrt_atom_extent(
        metadata,
        shard_count=2,
        shard_index=1,
        device="cuda",
        atom_offset=2,
        atom_count=3,
    ) as (first, atoms):
        assert first == 10
        assert atoms.shape == (3, 3, 37_056)
        assert not bool(torch.any(atoms))
    assert opened_paths == [str(metadata.path)]
    assert metadata.path.as_posix().startswith("/proc/self/fd/")
    seal.close()


def _reseal_checksum_manifest(root: Path) -> None:
    marker_path = root / "QSRT_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["checksum_manifest_sha256"] = _sha256(root / "MANIFEST.sha256")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _reseal_publication_identity(root: Path) -> None:
    checksum_path = root / "MANIFEST.sha256"
    names = [
        line.partition("  ")[2]
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
    ]
    checksum_path.write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    marker_path = root / "QSRT_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["package_manifest_sha256"] = _sha256(root / "qsrt-manifest.json")
    marker["model_index_sha256"] = _sha256(root / "model.safetensors.index.json")
    marker["checksum_manifest_sha256"] = _sha256(checksum_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


@pytest.mark.parametrize("field", ("codebook", "profile_id"))
def test_publication_rejects_missing_descriptor_identity(
    tmp_path: Path,
    field: str,
) -> None:
    manifest, descriptor, _ = _write_test_publication(tmp_path)
    manifest.pop(field)
    descriptor.pop(field)
    (tmp_path / "qsrt-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"qsrt": descriptor}}),
        encoding="utf-8",
    )
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="omits a required descriptor field"):
        _verify_publication(tmp_path)


def test_publication_rejects_relative_symlink_root(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _write_test_publication(root)
    alias = tmp_path / "model-alias"
    alias.symlink_to(Path("model"), target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        _verify_publication(alias)


@pytest.mark.parametrize("bad_name", ("../outside", "/tmp/outside"))
def test_publication_rejects_checksum_paths_outside_root(
    tmp_path: Path,
    bad_name: str,
) -> None:
    _write_test_publication(tmp_path)
    checksum_path = tmp_path / "MANIFEST.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    digest, _, _ = lines[-1].partition("  ")
    lines[-1] = f"{digest}  {bad_name}"
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _reseal_checksum_manifest(tmp_path)

    with pytest.raises(ValueError, match="invalid QSRT checksum entry"):
        _verify_publication(tmp_path)


def test_publication_rejects_unmanifested_file(tmp_path: Path) -> None:
    _write_test_publication(tmp_path)
    (tmp_path / "unsealed.bin").write_bytes(b"unsealed")

    with pytest.raises(ValueError, match="checksum inventory"):
        _verify_publication(tmp_path)


def test_publication_accepts_hugging_face_transport_metadata(tmp_path: Path) -> None:
    manifest, descriptor, _ = _write_test_publication(tmp_path)
    (tmp_path / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    cache = tmp_path / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("transport metadata", encoding="utf-8")

    seal = _verify_publication(tmp_path)
    assert seal.manifest == manifest
    assert seal.descriptor == descriptor

    unrelated = tmp_path / ".cache" / "other"
    unrelated.mkdir()
    (unrelated / "unsealed.bin").write_bytes(b"unsealed")
    with pytest.raises(ValueError, match="checksum inventory"):
        _verify_publication(tmp_path)


def test_publication_rejects_symlinked_payload(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    _, _, payload = _write_test_publication(root)
    external = tmp_path / "external.safetensors"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(Path("..") / external.name)

    with pytest.raises(FileNotFoundError, match=payload.name):
        _verify_publication(root)


def test_publication_rejects_missing_manifested_file(tmp_path: Path) -> None:
    _, _, payload = _write_test_publication(tmp_path)
    payload.unlink()

    with pytest.raises(FileNotFoundError, match=payload.name):
        _verify_publication(tmp_path)
