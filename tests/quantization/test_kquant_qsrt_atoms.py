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

from vllm.model_executor.layers.quantization import kquant_qsrt_publication
from vllm.model_executor.layers.quantization.kquant_qsrt_atoms import (
    FRUIT_SCHEMA,
    KIMI_K3_SCHEMA,
    balanced_atom_partition,
    open_qsrt_atom_extent,
    read_qsrt_atom_layer_metadata,
    snapshot_qsrt_publication,
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


def _canonical_identity_sha256(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


def _test_producer_identity() -> dict[str, object]:
    def executable(path: str) -> dict[str, str]:
        return {
            "path": path,
            "resolved_path": path,
            "sha256": "e" * 64,
        }

    builder_runtime = {
        "schema": "qsrt_fruit_builder_external_oci_runtime_v1",
        "verification_boundary": "external_container_runtime",
        "external_oci_image_id": f"sha256:{'f' * 64}",
        "python": executable("/usr/bin/python3"),
        "python_site_packages": "/opt/venv/lib/python3.12/site-packages",
        "git": executable("/usr/bin/git"),
        "cc": executable("/usr/bin/gcc"),
        "cxx": executable("/usr/bin/g++"),
        "ninja": executable("/usr/bin/ninja"),
        "cuda_home": "/usr/local/cuda-13.2",
        "nvcc": executable("/usr/local/cuda/bin/nvcc"),
        "path": "/usr/local/cuda/bin:/usr/bin",
    }
    bootstrap = {
        "schema": "qsrt_fruit_builder_bootstrap_identity_v5",
        "bootstrap_sha256": "0" * 64,
        "builder_sha256": "1" * 64,
        "qsrt_revision": "7" * 40,
        "qsrt_source_sha256": "9" * 64,
        "rate_sweep_authority": "external_sha256",
        "runtime_qualification_authority": "external_sha256",
        "runtime": builder_runtime,
    }
    encoder: dict[str, object] = {
        "qsrt_revision": "7" * 40,
        "qsrt_source_sha256": "9" * 64,
        "exllamav3_revision": "8" * 40,
        "exllamav3_source_sha256": "a" * 64,
        "calibration_fingerprint": "b" * 64,
        "calibration_capture_id": "c" * 64,
        "calibration_manifest_sha256": "d" * 64,
        "encoding_runtime": bootstrap,
        "fingerprint_schema": "qsrt_fruit_qsrt_encoder_source_v4",
    }
    encoder["fingerprint"] = _canonical_identity_sha256(
        {
            "schema": encoder["fingerprint_schema"],
            "qsrt_revision": encoder["qsrt_revision"],
            "qsrt_source_sha256": encoder["qsrt_source_sha256"],
            "exllamav3_source_sha256": encoder["exllamav3_source_sha256"],
            "exllamav3_revision": encoder["exllamav3_revision"],
            "calibration_fingerprint": encoder["calibration_fingerprint"],
            "calibration_capture_id": encoder["calibration_capture_id"],
            "calibration_manifest_sha256": encoder["calibration_manifest_sha256"],
            "encoding_runtime": encoder["encoding_runtime"],
        }
    )
    producer: dict[str, object] = {
        "schema": "qsrt_fruit_qsrt_producer_v2",
        "bootstrap": bootstrap,
        "encoder": encoder,
        "runtime": {
            "b12x_revision": "6" * 40,
            "b12x_source_sha256": "2" * 64,
            "vllm_revision": "5" * 40,
            "vllm_source_sha256": "3" * 64,
        },
    }
    producer["fingerprint"] = _canonical_identity_sha256(producer)
    return producer


def _write_runtime_qualification(root: Path, manifest: dict[str, object]) -> None:
    tensor_paths = sorted(root.glob("*.safetensors"))
    tensor_digests = {path.name: _sha256(path) for path in tensor_paths}
    tensor_set_digest = hashlib.sha256(
        (json.dumps(tensor_digests, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    revisions = {
        "vllm_revision": "5" * 40,
        "b12x_revision": "6" * 40,
        "qsrt_revision": "7" * 40,
    }

    def runtime(arm: str) -> dict[str, object]:
        model_options = {
            "bf16": ["--load-format", "fastsafetensors"],
            "siq": ["--load-format", "fastsafetensors"],
            "qsrt": [
                "--quantization",
                "kquant_hybrid",
                "--load-format",
                "fastsafetensors",
            ],
        }
        runtime_revisions = revisions
        return {
            "image": f"registry.invalid/fruit-final@sha256:{'8' * 64}",
            **runtime_revisions,
            "argv": [
                "vllm",
                "serve",
                "/model",
                "--served-model-name",
                "GLM-5.2-QSRT-Fruit-Instruct",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--tensor-parallel-size",
                "1",
                "--pipeline-parallel-size",
                "1",
                *model_options[arm],
                "--attention-backend",
                "B12X_MLA_SPARSE",
                "--moe-backend",
                "b12x",
                "--kv-cache-dtype",
                "nvfp4_ds_mla",
                "--enable-chunked-prefill",
                "--enable-prefix-caching",
                "--compilation-config",
                '{"backend":"inductor","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,48,64],"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}',
                "--speculative-config",
                '{"attention_backend":"B12X_MLA_SPARSE","method":"mtp","num_speculative_tokens":1}',
                "--gpu-memory-utilization",
                "0.80",
                "--max-model-len",
                "4096",
                "--max-num-batched-tokens",
                "4096",
                "--max-num-seqs",
                "1",
                "--tool-call-parser",
                "glm47",
                "--enable-auto-tool-choice",
                "--reasoning-parser",
                "glm45",
                "--generation-config",
                "vllm",
            ],
            "environment": dict(kquant_qsrt_publication._FIXED_RUNTIME_ENVIRONMENT),
            "software": {"torch": "final"},
            "compilation_backend": "inductor",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        }

    bf16_model = {
        "repository": "malaiwah/GLM-5.2-SIQ-Fruit-Instruct-bf16",
        "revision": "678954f65e056a0f508e21eeb9251c655bb9463f",
        "manifest_sha256": (
            "8f23aed5e9b12000ed103a76da772a20730ca53ab7e352d6cb94da2709165245"
        ),
        "config_sha256": (
            "1b1ea852c2bea8644774ec795025df2d0247b67131bccc8bf7e1137699518d55"
        ),
        "model_index_sha256": (
            "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56"
        ),
        "safetensors_bytes": 10_081_800_232,
        "safetensors_sha256": (
            "01fb6ad26356fc22f07f2598385b132db59df4eddd92bc005dfc0622284ee12b"
        ),
    }
    siq_model = {
        "repository": "malaiwah/GLM-5.2-SIQ-Fruit-Instruct",
        "revision": "48452ef397d8b4a4d6d0c00ea376a2abb3ef6314",
        "manifest_sha256": (
            "ac5485e2552f54850eebfecf11e23f3f640c391ed335d06562f91eb34f613639"
        ),
        "config_sha256": (
            "9d137e2b59fff529eb122581b0bce6eb7ace458a0785368d2ba587b4a5c2aa6f"
        ),
        "model_index_sha256": (
            "5808a4b3e75c4a949a1ede42e6c6fb2576089ec1544038b77de24076e99bf3da"
        ),
        "safetensors_bytes": 3_102_116_152,
        "safetensors_sha256": (
            "9c6c5c2c07eeb3aed026db4f6c5fc208dc04272304ba4f39ea9d23a31f9012b5"
        ),
    }
    qsrt_model = {
        "repository": "malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        "revision": "9" * 40,
        "manifest_sha256": "a" * 64,
        "config_sha256": _sha256(root / "config.json"),
        "model_index_sha256": _sha256(root / "model.safetensors.index.json"),
        "safetensors_bytes": sum(path.stat().st_size for path in tensor_paths),
        "safetensors_sha256": tensor_set_digest,
    }

    def loader(arm: str) -> dict[str, object]:
        return {
            "runtime": runtime(arm),
            "log_line": "loaded",
            "weight_bytes": 1,
            "peak_activation_bytes": 0,
            "non_torch_bytes": 0,
            "cudagraph_bytes": 1,
            "kv_cache_bytes": 1,
            "load_seconds": 1.0,
            "torch_allocated_bytes": 0,
            "torch_reserved_bytes": 0,
            "nvml_used_bytes": 0,
        }

    runs = [
        {
            "prompt_id": "decode",
            "repetition": repetition,
            "http_status": 200,
            "elapsed_seconds": 1.0,
            "completion_tokens": 1,
            "tokens_per_second": 1.0,
            "finish_reason": "stop",
            "content": "ok",
        }
        for repetition in (1, 2, 3)
    ]
    fidelity_candidate = {
        "mean_forward_kl": 0.0,
        "max_forward_kl": 0.0,
        "top1_agreement": 1.0,
        "top10_agreement": 1.0,
        "per_position": [
            {
                "position": 0,
                "forward_kl": 0.0,
                "top1_agreement": True,
                "top10_agreement": True,
            }
        ],
    }
    qualification = {
        "schema": "qsrt_fruit_runtime_qualification_v1",
        "version": 1,
        "complete": True,
        "publication": {
            "variant": "instruct",
            "repository": "malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        },
        "producer": manifest["producer"],
        "source": manifest["source"],
        "candidate": {
            "marker_sha256": "b" * 64,
            "model_index_sha256": _sha256(root / "model.safetensors.index.json"),
            "safetensors_sha256": tensor_digests,
        },
        "environment": {
            "gpu_model": "test GPU",
            "gpu_driver": "test driver",
            "host": "test host",
        },
        "protocol": {
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "repetitions": 3,
            "prompt_id": "decode",
            "prompt": "test",
            "prompt_token_ids": [1],
            "launch_order": ["bf16", "siq", "qsrt"],
        },
        "runtime_paths": {
            "schema": "qsrt_fruit_runtime_paths_v2",
            "version": 2,
            "layers": {
                **{
                    str(layer): {
                        "prefill": {
                            "mode": "w4a16",
                            "calls": 1,
                        },
                        "decode": {
                            "mode": "w4a8",
                            "calls": 1,
                            "part_count": 2,
                            "capture_calls": 1,
                            "replay_calls": 1,
                        },
                    }
                    for layer in range(3, 13)
                },
                "13": {
                    "mtp_prefill": {
                        "mode": "w4a16",
                        "calls": 1,
                        "capture_calls": 1,
                        "replay_calls": 1,
                    },
                    "mtp_decode": {
                        "mode": "w4a8",
                        "calls": 1,
                        "part_count": 2,
                        "capture_calls": 1,
                        "replay_calls": 1,
                    },
                },
            },
            "cudagraph": {
                "mode": "FULL_AND_PIECEWISE",
                "capture_count": 1,
                "replay_count": 1,
            },
            "speculative": {
                "method": "mtp",
                "num_speculative_tokens": 1,
                "draft_tokens": 1,
            },
        },
        "loaders": {arm: loader(arm) for arm in ("bf16", "siq", "qsrt")},
        "models": {"bf16": bf16_model, "siq": siq_model, "qsrt": qsrt_model},
        "decode": {arm: runs for arm in ("bf16", "siq", "qsrt")},
        "generation": {
            "prompts": [
                {"id": "generation", "prompt": "test", "prompt_token_ids": [1]}
            ],
            "results": {
                arm: [{"prompt_id": "generation", "content": "ok"}]
                for arm in ("bf16", "siq", "qsrt")
            },
        },
        "fidelity": {
            "full_vocabulary": True,
            "positions": [0],
            "vocab_size": 11,
            "candidates": {
                "siq": fidelity_candidate,
                "qsrt": fidelity_candidate,
            },
        },
    }
    receipt = root / "evaluation" / "fruit-runtime-qualification.json"
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(
        json.dumps(qualification, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["evaluation"] = {
        "runtime_qualification": {
            "file": "evaluation/fruit-runtime-qualification.json",
            "sha256": _sha256(receipt),
        }
    }


def _verify_publication(root: Path):
    return verify_qsrt_publication(
        root,
        expected_complete_sha256=_sha256(root / "QSRT_COMPLETE.json"),
    )


def _write_test_publication(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    producer = _test_producer_identity()
    producer_fingerprint = producer["fingerprint"]
    encoder = producer["encoder"]
    assert isinstance(producer_fingerprint, str)
    assert isinstance(encoder, dict)
    encoder_fingerprint = encoder["fingerprint"]
    assert isinstance(encoder_fingerprint, str)
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
    hybrid_map = {str(layer): [3] * 256 for layer in range(3, 14)}
    config = {
        "hidden_size": 1024,
        "moe_intermediate_size": 512,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 13,
        "quantization_config": {
            "hybrid_bit_map": hybrid_map,
            "kept_format": "mxfp4_e8m0k32",
            "demoted_format": "qsrt_sqg_e4m3",
            "qsrt": descriptor,
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    base_weight = root / "model-00001-of-00001.safetensors"
    base_weight.write_bytes(b"sealed base weight")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": base_weight.name}}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text('{"sealed":true}', encoding="utf-8")
    layers: dict[str, object] = {}
    for layer in range(3, 14):
        atom = root / f"qsrt-layer-{layer:03d}.safetensors"
        atom.write_bytes(
            b"sealed atom payload" if layer == 3 else f"layer {layer}".encode()
        )
        evidence = root / f"qsrt-layer-{layer:03d}.json"
        evidence.write_text("{}", encoding="utf-8")
        layers[str(layer)] = {
            "qsrt_atoms": atom.name,
            "bytes": atom.stat().st_size,
            "sha256": _sha256(atom),
            "expert_count": 256,
            "evidence": evidence.name,
        }
    manifest: dict[str, object] = {
        "schema": "qsrt_model_manifest_v1",
        "version": 1,
        "publication": {
            "variant": "instruct",
            "repository": "malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        },
        "codec": "QSRT",
        "storage_schema": FRUIT_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "profile_id": 1,
        "complete": True,
        "geometry": {
            "layers": list(range(3, 14)),
            "experts_per_layer": 256,
            "hidden_size": 1024,
            "intermediate_size": 512,
            "topk": 8,
        },
        "runtime": {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        },
        "producer": producer,
        "source": {
            "source_kind": "safetensors_manifest",
            "source_sha256": source_sha256,
        },
        "base_model": {"manifest_sha256": base_manifest_sha256},
        "layers": layers,
    }
    _write_runtime_qualification(root, manifest)
    (root / "qsrt-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checked = sorted(
        [
            "config.json",
            "model.safetensors.index.json",
            "evaluation/fruit-runtime-qualification.json",
            "qsrt-manifest.json",
            *[f"qsrt-layer-{layer:03d}.json" for layer in range(3, 14)],
            "tokenizer.json",
            "model-00001-of-00001.safetensors",
            *[f"qsrt-layer-{layer:03d}.safetensors" for layer in range(3, 14)],
        ]
    )
    (root / "MANIFEST.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in checked),
        encoding="utf-8",
    )
    marker = {
        "schema": "qsrt_complete_v3",
        "publication": {
            "variant": "instruct",
            "repository": "malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        },
        "qualified_candidate_sha256": "b" * 64,
        "runtime_qualification_sha256": _sha256(
            root / "evaluation/fruit-runtime-qualification.json"
        ),
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
    return manifest, descriptor, root / "qsrt-layer-003.safetensors"


def _convert_test_publication_to_candidate(root: Path) -> None:
    (root / "QSRT_COMPLETE.json").unlink()
    receipt = root / "evaluation" / "fruit-runtime-qualification.json"
    receipt.unlink()
    receipt.parent.rmdir()
    manifest_path = root / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest["evaluation"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checksum_path = root / "MANIFEST.sha256"
    names = [
        line.partition("  ")[2]
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if line.partition("  ")[2] != "evaluation/fruit-runtime-qualification.json"
    ]
    checksum_path.write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    complete_fields = {
        "schema": "qsrt_candidate_v1",
        "publication": {
            "variant": "instruct",
            "repository": "malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        },
        "package_manifest_sha256": _sha256(manifest_path),
        "checksum_manifest_sha256": _sha256(checksum_path),
        "model_index_sha256": _sha256(root / "model.safetensors.index.json"),
        "source": {
            "kind": "safetensors_manifest",
            "sha256": "3" * 64,
        },
        "base_manifest_sha256": "4" * 64,
        "producer_fingerprint": manifest["producer"]["fingerprint"],
        "encoder_fingerprint": manifest["producer"]["encoder"]["fingerprint"],
    }
    (root / "QSRT_CANDIDATE.json").write_text(
        json.dumps(complete_fields), encoding="utf-8"
    )


def test_active_runtime_environment_requires_canonical_fixed_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = kquant_qsrt_publication._FIXED_RUNTIME_ENVIRONMENT
    private_root = "/private/fruit"
    private_root_id = "root-id-abc"
    for key, template in contract.items():
        monkeypatch.setenv(
            key,
            template.replace("<PRIVATE_ROOT>", private_root).replace(
                "<PRIVATE_ROOT_ID>", private_root_id
            ),
        )

    assert (
        kquant_qsrt_publication.active_runtime_environment_from_env() == contract
    )

    # A live variable diverging from the canonical contract is rejected.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="canonical deployment contract"):
        kquant_qsrt_publication.active_runtime_environment_from_env()

    # A missing live variable is rejected.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(ValueError, match="is not set in the live environment"):
        kquant_qsrt_publication.active_runtime_environment_from_env()


def test_candidate_requires_explicit_mode_and_independent_digest(
    tmp_path: Path,
) -> None:
    _write_test_publication(tmp_path)
    _convert_test_publication_to_candidate(tmp_path)
    candidate_digest = _sha256(tmp_path / "QSRT_CANDIDATE.json")

    with pytest.raises((FileNotFoundError, ValueError)):
        verify_qsrt_publication(
            tmp_path,
            expected_complete_sha256=candidate_digest,
        )
    with pytest.raises(ValueError, match="expected candidate marker"):
        verify_qsrt_publication(tmp_path, candidate_mode=True)
    with pytest.raises(ValueError, match="trusted digest"):
        verify_qsrt_publication(
            tmp_path,
            candidate_mode=True,
            expected_candidate_sha256="0" * 64,
        )

    seal = verify_qsrt_publication(
        tmp_path,
        candidate_mode=True,
        expected_candidate_sha256=candidate_digest,
    )
    assert seal.qualification == {}
    seal.close()


def test_production_receipt_binds_qualified_candidate_marker(tmp_path: Path) -> None:
    _write_test_publication(tmp_path)
    receipt_path = tmp_path / "evaluation/fruit-runtime-qualification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["candidate"]["marker_sha256"] = "c" * 64
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["runtime_qualification"]["sha256"] = _sha256(receipt_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="candidate identity"):
        _verify_publication(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "omitted_kv_cache",
        "mutated_custom_ops",
        "missing_draft_attention_backend",
        "eager_qsrt",
        "wrong_variant",
        "wrong_comparator",
        "negative_prefix_caching",
        "negative_chunked_prefill",
        "unknown_flag",
        "positional_extra",
        "duplicate_option",
        "changed_environment",
        "same_extra_environment",
        "production_environment_drift",
        "mismatched_runtime_identity",
        "missing_mtp_prefill",
    ),
)
def test_publication_rejects_mutated_fixed_qualification_contract(
    tmp_path: Path, mutation: str
) -> None:
    _write_test_publication(tmp_path)
    receipt_path = tmp_path / "evaluation/fruit-runtime-qualification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "omitted_kv_cache":
        argv = receipt["loaders"]["bf16"]["runtime"]["argv"]
        index = argv.index("--kv-cache-dtype")
        del argv[index : index + 2]
    elif mutation == "mutated_custom_ops":
        argv = receipt["loaders"]["siq"]["runtime"]["argv"]
        index = argv.index("--compilation-config")
        config = json.loads(argv[index + 1])
        config["custom_ops"] = []
        argv[index + 1] = json.dumps(config, separators=(",", ":"), sort_keys=True)
    elif mutation == "missing_draft_attention_backend":
        argv = receipt["loaders"]["qsrt"]["runtime"]["argv"]
        index = argv.index("--speculative-config")
        config = json.loads(argv[index + 1])
        del config["attention_backend"]
        argv[index + 1] = json.dumps(config, separators=(",", ":"), sort_keys=True)
    elif mutation == "eager_qsrt":
        receipt["loaders"]["qsrt"]["runtime"]["compilation_backend"] = "eager"
    elif mutation == "wrong_variant":
        receipt["publication"]["variant"] = "annealed"
    elif mutation == "wrong_comparator":
        receipt["models"]["bf16"]["repository"] = "malaiwah/wrong"
    elif mutation == "mismatched_runtime_identity":
        receipt["loaders"]["bf16"]["runtime"]["b12x_revision"] = "d" * 40
    elif mutation == "missing_mtp_prefill":
        del receipt["runtime_paths"]["layers"]["13"]["mtp_prefill"]
    elif mutation == "changed_environment":
        receipt["loaders"]["siq"]["runtime"]["environment"]["CUDA_VISIBLE_DEVICES"] = (
            "1"
        )
    elif mutation == "same_extra_environment":
        for arm in ("bf16", "siq", "qsrt"):
            receipt["loaders"][arm]["runtime"]["environment"][
                "VLLM_FAKE_PRODUCTION_TOGGLE"
            ] = "1"
    elif mutation == "production_environment_drift":
        for arm in ("bf16", "siq", "qsrt"):
            receipt["loaders"][arm]["runtime"]["environment"]["PYTHONSAFEPATH"] = "0"
    else:
        argv = receipt["loaders"]["qsrt"]["runtime"]["argv"]
        extras = {
            "negative_prefix_caching": ["--no-enable-prefix-caching"],
            "negative_chunked_prefill": ["--no-enable-chunked-prefill"],
            "unknown_flag": ["--unknown-qualification-flag"],
            "positional_extra": ["unexpected-positional"],
            "duplicate_option": ["--max-num-seqs", "1"],
        }
        argv.extend(extras[mutation])
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["runtime_qualification"]["sha256"] = _sha256(receipt_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError):
        _verify_publication(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("encoder_source", "producer_runtime_source", "unknown_encoder_field"),
)
def test_publication_rejects_unfingerprinted_producer_provenance(
    tmp_path: Path, mutation: str
) -> None:
    _write_test_publication(tmp_path)
    manifest_path = tmp_path / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    producer = manifest["producer"]
    if mutation == "encoder_source":
        producer["encoder"]["qsrt_source_sha256"] = "f" * 64
    elif mutation == "producer_runtime_source":
        producer["runtime"]["b12x_source_sha256"] = "f" * 64
    else:
        producer["encoder"]["unknown_provenance"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="fingerprint|invalid keys"):
        _verify_publication(tmp_path)


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


def test_publication_requires_matched_tp1_runtime_receipt(tmp_path: Path) -> None:
    _write_test_publication(tmp_path)
    receipt = tmp_path / "evaluation" / "fruit-runtime-qualification.json"
    qualification = json.loads(receipt.read_text(encoding="utf-8"))
    qualification["protocol"]["max_num_seqs"] = 2
    receipt.write_text(
        json.dumps(qualification, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation"]["runtime_qualification"]["sha256"] = _sha256(receipt)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="matched TP1 protocol"):
        _verify_publication(tmp_path)


def test_private_snapshot_is_immune_to_later_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    private = tmp_path / "private"
    source.mkdir()
    private.mkdir(mode=0o700)
    _write_test_publication(source)
    trusted = _sha256(source / "QSRT_COMPLETE.json")

    seal = snapshot_qsrt_publication(
        source,
        private / "model",
        expected_complete_sha256=trusted,
    )
    authenticated_config = seal.config
    authenticated_atom = seal.authenticated_atom_path("qsrt-layer-003.safetensors")
    authenticated_tokenizer = (seal.root / "tokenizer.json").read_bytes()
    authenticated_weight = (seal.root / "model-00001-of-00001.safetensors").read_bytes()
    source.joinpath("config.json").write_text("{}", encoding="utf-8")
    source.joinpath("qsrt-layer-003.safetensors").write_bytes(b"mutated")

    source.joinpath("tokenizer.json").write_text("{}", encoding="utf-8")
    source.joinpath("model-00001-of-00001.safetensors").write_bytes(b"mutated")
    assert seal.root == (private / "model").resolve()
    assert seal.config == authenticated_config
    assert authenticated_atom.read_bytes() == b"sealed atom payload"
    seal.close()
    assert (seal.root / "tokenizer.json").read_bytes() == authenticated_tokenizer
    assert (
        seal.root / "model-00001-of-00001.safetensors"
    ).read_bytes() == authenticated_weight


def test_private_snapshot_rejects_hard_linked_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    private = tmp_path / "private"
    source.mkdir()
    private.mkdir(mode=0o700)
    _write_test_publication(source)
    linked = tmp_path / "linked-config.json"
    linked.hardlink_to(source / "config.json")

    with pytest.raises(ValueError, match="singly linked"):
        snapshot_qsrt_publication(
            source,
            private / "model",
            expected_complete_sha256=_sha256(source / "QSRT_COMPLETE.json"),
        )


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
    layers = manifest["layers"]
    assert isinstance(layers, dict)
    layer_entry = layers["3"]
    assert isinstance(layer_entry, dict)
    layer_entry.update(
        {
            "bytes": published_atom.stat().st_size,
            "sha256": _sha256(published_atom),
        }
    )
    _write_runtime_qualification(model, manifest)
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

    instanttensor.safe_open = fake_safe_open
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
    marker["runtime_qualification_sha256"] = _sha256(
        root / "evaluation/fruit-runtime-qualification.json"
    )
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
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"]["qsrt"] = descriptor
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _reseal_publication_identity(tmp_path)

    with pytest.raises(ValueError, match="identity is invalid"):
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


def test_snapshot_omits_hugging_face_transport_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    private = tmp_path / "private"
    source.mkdir()
    private.mkdir(mode=0o700)
    _write_test_publication(source)
    (source / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    cache = source / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("transport metadata", encoding="utf-8")

    seal = snapshot_qsrt_publication(
        source,
        private / "model",
        expected_complete_sha256=_sha256(source / "QSRT_COMPLETE.json"),
    )
    assert not (seal.root / ".cache" / "huggingface").exists()
    seal.close()


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
