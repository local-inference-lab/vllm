# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-stdlib authentication for sealed QSRT model packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHA256_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_MARKER = "QSRT_COMPLETE.json"
_CANDIDATE_MARKER = "QSRT_CANDIDATE.json"
_CHECKSUM_MANIFEST = "MANIFEST.sha256"
_PACKAGE_MANIFEST = "qsrt-manifest.json"
_RUNTIME_QUALIFICATION = "evaluation/fruit-runtime-qualification.json"
_RUNTIME_QUALIFICATION_SCHEMA = "kquant_fruit_runtime_qualification_v1"
_FRUIT_REPOSITORY = "malaiwah/GLM-5.2-QSRT-Fruit-Instruct"
_RUNTIME_ARMS = frozenset({"bf16", "siq", "qsrt"})
_FRUIT_LAYERS = tuple(range(3, 14))
_FRUIT_EXPERTS = 256
_FRUIT_HIDDEN_SIZE = 1024
_FRUIT_INTERMEDIATE_SIZE = 512
_FRUIT_TOPK = 8
_INSTRUCT_COMPARATOR_MODELS = {
    "bf16": {
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
    },
    "siq": {
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
    },
}
_FIXED_COMPILATION_CONFIG = {
    "backend": "inductor",
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "custom_ops": ["all"],
    "cudagraph_capture_sizes": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        20,
        24,
        32,
        48,
        64,
    ],
}
_FIXED_SPECULATIVE_CONFIG = {
    "attention_backend": "B12X_MLA_SPARSE",
    "method": "mtp",
    "num_speculative_tokens": 1,
}
_FIXED_RUNTIME_OPTIONS = {
    "--attention-backend": "B12X_MLA_SPARSE",
    "--generation-config": "vllm",
    "--gpu-memory-utilization": "0.80",
    "--moe-backend": "b12x",
    "--kv-cache-dtype": "nvfp4_ds_mla",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
}
_MODEL_RUNTIME_CONTRACT = {
    "bf16": {"--load-format": "fastsafetensors"},
    "siq": {"--load-format": "fastsafetensors"},
    "qsrt": {
        "--quantization": "kquant_hybrid",
        "--load-format": "fastsafetensors",
    },
}
_MODEL_RUNTIME_OPTIONS = frozenset({"--quantization", "--load-format"})
_FIXED_RUNTIME_ENVIRONMENT = {
    "B12X_COMPILE_CACHE_DIR": "<PRIVATE_ROOT>/cache/b12x/compile",
    "B12X_CUTE_COMPILE_CACHE_DIR": "<PRIVATE_ROOT>/cache/b12x-cute",
    "B12X_ROOT": "<PRIVATE_ROOT>/runtime/b12x-source",
    "CUDA_CACHE_PATH": "<PRIVATE_ROOT>/cache/cuda",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUPY_CACHE_DIR": "<PRIVATE_ROOT>/cache/cupy",
    "CUTE_DSL_ARCH": "sm_120a",
    "CUTE_DSL_CACHE_DIR": "<PRIVATE_ROOT>/cache/cute-dsl",
    "DG_JIT_CACHE_DIR": "<PRIVATE_ROOT>/cache/deep-gemm",
    "FLASHINFER_WORKSPACE_BASE": "<PRIVATE_ROOT>/cache/flashinfer",
    "FRUIT_QSRT_AUTHENTICATED_MODEL_ROOT": "<PRIVATE_ROOT>/model",
    "GIT_OPTIONAL_LOCKS": "0",
    "HF_DATASETS_CACHE": "<PRIVATE_ROOT>/cache/huggingface/datasets",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HOME": "<PRIVATE_ROOT>/cache/huggingface",
    "HF_HUB_OFFLINE": "1",
    "HOME": "<PRIVATE_ROOT>/home",
    "HUGGINGFACE_HUB_CACHE": "<PRIVATE_ROOT>/cache/huggingface/hub",
    "LD_LIBRARY_PATH": (
        "/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    ),
    "LOCAL_INFERENCE_CACHE_FINGERPRINT": "<PRIVATE_ROOT_ID>",
    "MINFER_FMHA_CACHE_DIR": "<PRIVATE_ROOT>/cache/minfer/fmha",
    "MM_SPARSE_ATTN_AOT_CACHE": "<PRIVATE_ROOT>/cache/minfer/mm-sparse-attn",
    "NUMBA_CACHE_DIR": "<PRIVATE_ROOT>/cache/numba",
    "PATH": (
        "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": (
        "<PRIVATE_ROOT>/runtime/vllm-source:<PRIVATE_ROOT>/runtime/b12x-source"
    ),
    "PYTHONSAFEPATH": "1",
    "SAFETENSORS_FAST_GPU": "1",
    "SPARKINFER_COMPILE_CACHE_DIR": "<PRIVATE_ROOT>/cache/b12x/compile",
    "TEMP": "<PRIVATE_ROOT>/tmp",
    "TILELANG_CACHE_DIR": "<PRIVATE_ROOT>/cache/tilelang",
    "TILELANG_TMP_DIR": "<PRIVATE_ROOT>/cache/tilelang/tmp",
    "TMP": "<PRIVATE_ROOT>/tmp",
    "TMPDIR": "<PRIVATE_ROOT>/tmp",
    "TORCHINDUCTOR_CACHE_DIR": "<PRIVATE_ROOT>/cache/torchinductor",
    "TORCH_EXTENSIONS_DIR": "<PRIVATE_ROOT>/cache/torch-extensions",
    "TORCH_HOME": "<PRIVATE_ROOT>/cache/torch",
    "TRANSFORMERS_CACHE": "<PRIVATE_ROOT>/cache/huggingface/transformers",
    "TRANSFORMERS_OFFLINE": "1",
    "TRITON_CACHE_DIR": "<PRIVATE_ROOT>/cache/triton",
    "TVM_CACHE_DIR": "<PRIVATE_ROOT>/cache/tvm",
    "TVM_FFI_CACHE_DIR": "<PRIVATE_ROOT>/cache/tvm-ffi",
    "VLLM_CACHE_DIR": "<PRIVATE_ROOT>/cache/vllm",
    "VLLM_CACHE_ROOT": "<PRIVATE_ROOT>/cache/vllm",
    "VLLM_EXL3_ONLINE_CACHE_DIR": "<PRIVATE_ROOT>/cache/exl3-online",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": ("<PRIVATE_ROOT>/cache/flashinfer-autotune"),
    "VLLM_PLUGINS": "",
    "VLLM_USE_B12X_MOE": "1",
    "VLLM_USE_B12X_SPARSE_INDEXER": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "XDG_CACHE_HOME": "<PRIVATE_ROOT>/cache",
}
_VARIABLE_RUNTIME_OPTIONS = {
    "--served-model-name": "<MODEL>",
    "--host": "<HOST>",
    "--port": "<PORT>",
}
_FIXED_RUNTIME_SWITCHES = (
    "--enable-auto-tool-choice",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
)
_RUNTIME_OPTION_ORDER = (
    "--served-model-name",
    "--host",
    "--port",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--attention-backend",
    "--moe-backend",
    "--kv-cache-dtype",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--compilation-config",
    "--speculative-config",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--tool-call-parser",
    "--enable-auto-tool-choice",
    "--reasoning-parser",
    "--generation-config",
)
_HF_TRANSPORT_FILES = frozenset({".gitattributes"})
_HF_LOCAL_CACHE_PREFIX = (".cache", "huggingface")
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIRECTORY_FLAGS = _READ_FLAGS | os.O_DIRECTORY


@dataclass
class QSRTPublicationSeal:
    """Authenticated package identity plus stable atom-file descriptors."""

    root: Path
    manifest: dict[str, Any]

    descriptor: dict[str, Any]
    config: dict[str, Any]
    qualification: dict[str, Any]
    checksums: dict[str, str]
    _atom_descriptors: dict[str, int] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def authenticated_atom_path(self, filename: str) -> Path:
        """Return a procfs path that reopens the already-authenticated inode."""

        if self._closed:
            raise RuntimeError("QSRT publication seal is closed")
        try:
            descriptor = self._atom_descriptors[filename]
        except KeyError as exc:
            raise ValueError(
                f"QSRT publication has no authenticated atom {filename!r}"
            ) from exc
        return Path(f"/proc/self/fd/{descriptor}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in self._atom_descriptors.values():
            os.close(descriptor)
        self._atom_descriptors.clear()

    def __enter__(self) -> QSRTPublicationSeal:
        if self._closed:
            raise RuntimeError("QSRT publication seal is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _sha256_field(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_DIGITS for character in value)
    ):
        raise ValueError(f"QSRT publication {name} is not a lowercase SHA-256 digest")
    return value


def publication_trust_from_env() -> tuple[bool, str]:
    """Read Fruit's exclusive, independently anchored publication trust mode."""

    complete = os.environ.get("FRUIT_QSRT_EXPECTED_COMPLETE_SHA256")
    candidate = os.environ.get("FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256")
    if (complete is None) == (candidate is None):
        raise ValueError(
            "exactly one of FRUIT_QSRT_EXPECTED_COMPLETE_SHA256 or "
            "FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256 is required"
        )
    if candidate is not None:
        return True, _sha256_field(
            candidate,
            name="FRUIT_QSRT_EXPECTED_CANDIDATE_SHA256",
        )
    return False, _sha256_field(
        complete,
        name="FRUIT_QSRT_EXPECTED_COMPLETE_SHA256",
    )


def active_runtime_environment_from_env() -> dict[str, str]:
    """Authenticate the launcher's canonical runtime environment contract."""

    raw = os.environ.get("FRUIT_QSRT_RUNTIME_ENVIRONMENT_JSON")
    digest = _sha256_field(
        os.environ.get("FRUIT_QSRT_RUNTIME_ENVIRONMENT_SHA256"),
        name="FRUIT_QSRT_RUNTIME_ENVIRONMENT_SHA256",
    )
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != digest:
        raise ValueError(
            "Fruit active runtime environment does not match its trusted digest"
        )
    try:
        environment = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Fruit active runtime environment is not valid JSON") from exc
    canonical = json.dumps(
        environment,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if raw != canonical or environment != _FIXED_RUNTIME_ENVIRONMENT:
        raise ValueError(
            "Fruit active runtime environment is not the fixed deployment contract"
        )
    return environment


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 8 << 20):
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 8 << 20):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_beneath(root_descriptor: int, relative: Path) -> int:
    directory = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=directory)
    except OSError as exc:
        raise FileNotFoundError(relative.as_posix()) from exc
    finally:
        os.close(directory)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise FileNotFoundError(relative.as_posix())
    return descriptor


def _json_bytes(data: bytes, *, kind: str, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{kind} must be a JSON object: {path}")
    return value


def _published_files(root_descriptor: int) -> set[str]:
    published: set[str] = set()

    def visit(directory: int, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = (*prefix, entry.name)
                display = "/".join(relative)
                if entry.is_symlink():
                    raise ValueError(
                        f"QSRT package must not contain symbolic links: {display}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=directory)
                    try:
                        visit(child, relative)
                    finally:
                        os.close(child)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(
                        f"QSRT package contains a non-regular file: {display}"
                    )
                if entry.stat(follow_symlinks=False).st_nlink != 1:
                    raise ValueError(
                        f"QSRT package contains a hard-linked file: {display}"
                    )
                if (
                    display in _HF_TRANSPORT_FILES
                    or relative[:2] == _HF_LOCAL_CACHE_PREFIX
                ):
                    continue
                published.add(display)

    visit(root_descriptor, ())
    return published


def _qualification_object(
    value: object, *, name: str, keys: set[str] | frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Fruit runtime qualification {name} has invalid keys")
    return value


def _qualification_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Fruit runtime qualification {name} must be nonempty")
    return value


def _qualification_number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
        or (maximum is not None and float(value) > maximum)
    ):
        raise ValueError(f"Fruit runtime qualification {name} is invalid")
    return float(value)


def _qualification_revision(value: object, *, name: str) -> str:
    revision = _qualification_string(value, name=name)
    if len(revision) != 40 or any(
        character not in _SHA256_DIGITS for character in revision
    ):
        raise ValueError(f"Fruit runtime qualification {name} is not a Git revision")
    return revision


def _qualification_digest(value: object, *, name: str) -> str:
    return _sha256_field(value, name=f"runtime qualification {name}")


def _runtime_argv_options(
    argv: list[str],
) -> tuple[str, dict[str, str], set[str]]:
    if (
        len(argv) < 3
        or argv[:2] != ["vllm", "serve"]
        or not argv[2]
        or argv[2].startswith("-")
    ):
        raise ValueError("Fruit runtime qualification argv is not vllm serve MODEL")

    required_value_flags = (
        set(_FIXED_RUNTIME_OPTIONS)
        | set(_VARIABLE_RUNTIME_OPTIONS)
        | {
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "--max-num-seqs",
            "--max-model-len",
            "--max-num-batched-tokens",
            "--compilation-config",
            "--speculative-config",
        }
    )
    value_flags = required_value_flags | set(_MODEL_RUNTIME_OPTIONS)
    switch_flags = set(_FIXED_RUNTIME_SWITCHES)
    values: dict[str, str] = {}
    switches: set[str] = set()
    index = 3
    while index < len(argv):
        argument = argv[index]
        if not argument.startswith("--"):
            raise ValueError(
                "Fruit runtime qualification argv contains an extra positional argument"
            )
        flag, separator, inline_value = argument.partition("=")
        if flag not in value_flags and flag not in switch_flags:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} is not allowed"
            )
        if flag in values or flag in switches:
            raise ValueError(
                f"Fruit runtime qualification argv contains duplicate option {flag}"
            )
        if flag in switch_flags:
            if separator:
                raise ValueError(
                    f"Fruit runtime qualification argv switch {flag} takes no value"
                )
            switches.add(flag)
            index += 1
            continue
        if separator:
            value = inline_value
        else:
            index += 1
            if index >= len(argv) or argv[index].startswith("--"):
                raise ValueError(
                    f"Fruit runtime qualification argv option {flag} has no value"
                )
            value = argv[index]
        if not value:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} has no value"
            )
        values[flag] = value
        index += 1

    missing = (required_value_flags - set(values)) | (switch_flags - switches)
    if missing:
        raise ValueError(
            "Fruit runtime qualification argv is missing required options "
            f"{sorted(missing)}"
        )
    return argv[2], values, switches


def _runtime_argv_json(raw_value: str, flag: str) -> dict[str, object]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Fruit runtime qualification argv option {flag} is not JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(
            f"Fruit runtime qualification argv option {flag} must be an object"
        )
    return value


def _validate_runtime_argv(
    argv: list[str],
    *,
    arm: str,
    protocol: dict[str, Any],
    compilation_backend: str,
    cudagraph_mode: object,
) -> tuple[str, ...]:
    _model, values, switches = _runtime_argv_options(argv)
    expected_integers = {
        "--tensor-parallel-size": int(protocol["tensor_parallel_size"]),
        "--pipeline-parallel-size": 1,
        "--max-num-seqs": int(protocol["max_num_seqs"]),
        "--max-model-len": 4096,
        "--max-num-batched-tokens": 4096,
    }
    for flag, expected in expected_integers.items():
        raw_value = values[flag]
        try:
            measured = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} is not an integer"
            ) from exc
        if str(measured) != raw_value or measured != expected:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} is not {expected}"
            )
    try:
        port = int(values["--port"])
    except ValueError as exc:
        raise ValueError(
            "Fruit runtime qualification argv option --port is not an integer"
        ) from exc
    if str(port) != values["--port"] or not 1 <= port <= 65535:
        raise ValueError("Fruit runtime qualification argv option --port is invalid")

    if compilation_backend != "inductor" or cudagraph_mode != "FULL_AND_PIECEWISE":
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} must use non-eager "
            "inductor FULL_AND_PIECEWISE"
        )
    compilation = _runtime_argv_json(
        values["--compilation-config"], "--compilation-config"
    )
    if compilation != _FIXED_COMPILATION_CONFIG:
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} compilation config "
            "is not the fixed deployment contract"
        )
    speculative = _runtime_argv_json(
        values["--speculative-config"], "--speculative-config"
    )
    if speculative != _FIXED_SPECULATIVE_CONFIG:
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} MTP argv is not qualified"
        )
    for flag, expected in _FIXED_RUNTIME_OPTIONS.items():
        if values[flag] != expected:
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} argv option {flag} "
                f"is not {expected}"
            )
    measured_model_options = {
        flag: values[flag] for flag in _MODEL_RUNTIME_OPTIONS if flag in values
    }
    if measured_model_options != _MODEL_RUNTIME_CONTRACT[arm]:
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} does not use its "
            "qualified model-specific quantization and load format"
        )
    normalized_values = {
        **values,
        **_VARIABLE_RUNTIME_OPTIONS,
        "--compilation-config": json.dumps(
            _FIXED_COMPILATION_CONFIG, separators=(",", ":"), sort_keys=True
        ),
        "--speculative-config": json.dumps(
            _FIXED_SPECULATIVE_CONFIG, separators=(",", ":"), sort_keys=True
        ),
    }
    normalized = ["vllm", "serve", "<MODEL>"]
    for flag in _RUNTIME_OPTION_ORDER:
        normalized.append(flag)
        if flag not in switches:
            normalized.append(normalized_values[flag])
    return tuple(normalized)


def _validate_runtime_qualification(
    value: dict[str, Any],
    *,
    manifest: dict[str, Any],
    config_digest: str,
    model_index_digest: str,
    tensor_digests: dict[str, str],
    tensor_bytes: int,
    candidate_marker_digest: str,
) -> None:
    payload = _qualification_object(
        value,
        name="root",
        keys={
            "schema",
            "version",
            "complete",
            "publication",
            "producer",
            "source",
            "candidate",
            "environment",
            "protocol",
            "loaders",
            "models",
            "decode",
            "generation",
            "fidelity",
            "runtime_paths",
        },
    )
    if (
        payload["schema"] != _RUNTIME_QUALIFICATION_SCHEMA
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["complete"] is not True
    ):
        raise ValueError(
            "Fruit runtime qualification is incomplete or has the wrong schema"
        )
    if _qualification_object(
        payload["publication"],
        name="publication",
        keys={"variant", "repository"},
    ) != {"variant": "instruct", "repository": _FRUIT_REPOSITORY}:
        raise ValueError("Fruit runtime qualification publication target is invalid")
    if payload["producer"] != manifest.get("producer"):
        raise ValueError(
            "Fruit runtime qualification producer does not match the package"
        )
    if payload["source"] != manifest.get("source"):
        raise ValueError(
            "Fruit runtime qualification source does not match the package"
        )

    candidate = _qualification_object(
        payload["candidate"],
        name="candidate",
        keys={"marker_sha256", "model_index_sha256", "safetensors_sha256"},
    )
    if (
        _qualification_digest(
            candidate["marker_sha256"], name="candidate.marker_sha256"
        )
        != candidate_marker_digest
        or _qualification_digest(
            candidate["model_index_sha256"], name="candidate.model_index_sha256"
        )
        != model_index_digest
        or candidate["safetensors_sha256"] != tensor_digests
    ):
        raise ValueError("Fruit runtime qualification candidate identity is invalid")

    environment = _qualification_object(
        payload["environment"],
        name="environment",
        keys={"gpu_model", "gpu_driver", "host"},
    )
    for key, item in environment.items():
        _qualification_string(item, name=f"environment.{key}")

    protocol = _qualification_object(
        payload["protocol"],
        name="protocol",
        keys={
            "tensor_parallel_size",
            "max_num_seqs",
            "max_tokens",
            "temperature",
            "repetitions",
            "prompt_id",
            "prompt",
            "prompt_token_ids",
            "launch_order",
        },
    )
    if (
        type(protocol["tensor_parallel_size"]) is not int
        or protocol["tensor_parallel_size"] != 1
        or type(protocol["max_num_seqs"]) is not int
        or protocol["max_num_seqs"] != 1
        or type(protocol["max_tokens"]) is not int
        or protocol["max_tokens"] <= 0
        or type(protocol["repetitions"]) is not int
        or protocol["repetitions"] < 3
        or not isinstance(protocol["launch_order"], list)
        or len(protocol["launch_order"]) != 3
        or set(protocol["launch_order"]) != _RUNTIME_ARMS
    ):
        raise ValueError("Fruit runtime qualification is not a matched TP1 protocol")
    _qualification_number(protocol["temperature"], name="protocol.temperature")
    decode_prompt_id = _qualification_string(
        protocol["prompt_id"], name="protocol.prompt_id"
    )
    _qualification_string(protocol["prompt"], name="protocol.prompt")
    prompt_tokens = protocol["prompt_token_ids"]
    if (
        not isinstance(prompt_tokens, list)
        or not prompt_tokens
        or any(type(token) is not int or token < 0 for token in prompt_tokens)
    ):
        raise ValueError("Fruit runtime qualification prompt tokens are invalid")

    runtime_paths = _qualification_object(
        payload["runtime_paths"],
        name="runtime_paths",
        keys={"schema", "version", "layers", "cudagraph", "speculative"},
    )
    if (
        runtime_paths["schema"] != "kquant_fruit_runtime_paths_v1"
        or type(runtime_paths["version"]) is not int
        or runtime_paths["version"] != 1
    ):
        raise ValueError("Fruit runtime path evidence identity is invalid")
    runtime_layers = _qualification_object(
        runtime_paths["layers"],
        name="runtime_paths.layers",
        keys={str(layer) for layer in _FRUIT_LAYERS},
    )

    def validate_prefill_observation(value: object, *, name: str) -> None:
        observation = _qualification_object(
            value,
            name=name,
            keys={"mode", "calls"},
        )
        if (
            observation["mode"] != "w4a16"
            or type(observation["calls"]) is not int
            or observation["calls"] <= 0
        ):
            raise ValueError(f"Fruit runtime path evidence {name} is invalid")

    def validate_decode_observation(value: object, *, name: str) -> None:
        observation = _qualification_object(
            value,
            name=name,
            keys={
                "mode",
                "calls",
                "part_count",
                "capture_calls",
                "replay_calls",
            },
        )
        if (
            observation["mode"] != "w4a8"
            or type(observation["calls"]) is not int
            or observation["calls"] <= 0
            or type(observation["part_count"]) is not int
            or observation["part_count"] != 2
            or type(observation["capture_calls"]) is not int
            or observation["capture_calls"] <= 0
            or type(observation["replay_calls"]) is not int
            or observation["replay_calls"] <= 0
        ):
            raise ValueError(f"Fruit runtime path evidence {name} is invalid")

    for layer in range(3, 13):
        observations = _qualification_object(
            runtime_layers[str(layer)],
            name=f"runtime_paths.layers.{layer}",
            keys={"prefill", "decode"},
        )
        validate_prefill_observation(
            observations["prefill"],
            name=f"runtime_paths.layers.{layer}.prefill",
        )
        validate_decode_observation(
            observations["decode"],
            name=f"runtime_paths.layers.{layer}.decode",
        )
    mtp = _qualification_object(
        runtime_layers["13"],
        name="runtime_paths.layers.13",
        keys={"mtp_decode"},
    )
    validate_decode_observation(
        mtp["mtp_decode"],
        name="runtime_paths.layers.13.mtp_decode",
    )
    cudagraph = _qualification_object(
        runtime_paths["cudagraph"],
        name="runtime_paths.cudagraph",
        keys={"mode", "capture_count", "replay_count"},
    )
    if (
        cudagraph["mode"] != "FULL_AND_PIECEWISE"
        or type(cudagraph["capture_count"]) is not int
        or cudagraph["capture_count"] <= 0
        or type(cudagraph["replay_count"]) is not int
        or cudagraph["replay_count"] <= 0
    ):
        raise ValueError("Fruit runtime path cudagraph evidence is invalid")
    speculative = _qualification_object(
        runtime_paths["speculative"],
        name="runtime_paths.speculative",
        keys={"method", "num_speculative_tokens", "draft_tokens"},
    )
    if (
        speculative["method"] != "mtp"
        or speculative["num_speculative_tokens"] != 1
        or type(speculative["draft_tokens"]) is not int
        or speculative["draft_tokens"] <= 0
    ):
        raise ValueError("Fruit runtime path speculative evidence is invalid")

    models = _qualification_object(payload["models"], name="models", keys=_RUNTIME_ARMS)
    tensor_set_digest = hashlib.sha256(
        (
            json.dumps(tensor_digests, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
    ).hexdigest()
    for arm in ("bf16", "siq", "qsrt"):
        model = _qualification_object(
            models[arm],
            name=f"models.{arm}",
            keys={
                "repository",
                "revision",
                "manifest_sha256",
                "config_sha256",
                "model_index_sha256",
                "safetensors_bytes",
                "safetensors_sha256",
            },
        )
        _qualification_string(model["repository"], name=f"models.{arm}.repository")
        _qualification_revision(model["revision"], name=f"models.{arm}.revision")
        for key in ("manifest_sha256", "config_sha256", "model_index_sha256"):
            _qualification_digest(model[key], name=f"models.{arm}.{key}")
        if (
            type(model["safetensors_bytes"]) is not int
            or model["safetensors_bytes"] <= 0
        ):
            raise ValueError(
                f"Fruit runtime qualification models.{arm} size is invalid"
            )
        _qualification_digest(
            model["safetensors_sha256"], name=f"models.{arm}.safetensors_sha256"
        )
    for arm, expected_identity in _INSTRUCT_COMPARATOR_MODELS.items():
        if models[arm] != expected_identity:
            raise ValueError(
                f"Fruit runtime qualification {arm.upper()} comparator identity "
                "is not the pinned Instruct checkpoint"
            )
    qsrt_model = models["qsrt"]
    if (
        qsrt_model["repository"] != _FRUIT_REPOSITORY
        or qsrt_model["model_index_sha256"] != model_index_digest
        or qsrt_model["config_sha256"] != config_digest
        or qsrt_model["safetensors_bytes"] != tensor_bytes
        or qsrt_model["safetensors_sha256"] != tensor_set_digest
    ):
        raise ValueError("Fruit runtime qualification QSRT model identity is invalid")

    producer = manifest["producer"]
    runtime_identity = producer.get("runtime")
    encoder_identity = producer.get("encoder")
    if not isinstance(runtime_identity, dict) or not isinstance(encoder_identity, dict):
        raise ValueError("Fruit runtime qualification producer identity is malformed")
    loaders = _qualification_object(
        payload["loaders"], name="loaders", keys=_RUNTIME_ARMS
    )
    normalized_runtime_argv: tuple[str, ...] | None = None
    for arm in ("bf16", "siq", "qsrt"):
        loader = _qualification_object(
            loaders[arm],
            name=f"loaders.{arm}",
            keys={
                "runtime",
                "log_line",
                "weight_bytes",
                "peak_activation_bytes",
                "non_torch_bytes",
                "cudagraph_bytes",
                "kv_cache_bytes",
                "load_seconds",
                "torch_allocated_bytes",
                "torch_reserved_bytes",
                "nvml_used_bytes",
            },
        )
        runtime = _qualification_object(
            loader["runtime"],
            name=f"loaders.{arm}.runtime",
            keys={
                "image",
                "vllm_revision",
                "b12x_revision",
                "kquant_revision",
                "argv",
                "environment",
                "software",
                "compilation_backend",
                "cudagraph_mode",
            },
        )
        image = _qualification_string(runtime["image"], name=f"loaders.{arm}.image")
        image_name, separator, image_digest = image.rpartition("@sha256:")
        if not image_name or separator != "@sha256:":
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} image is mutable"
            )
        _qualification_digest(image_digest, name=f"loaders.{arm}.image_digest")
        for key in ("vllm_revision", "b12x_revision", "kquant_revision"):
            _qualification_revision(runtime[key], name=f"loaders.{arm}.{key}")
        if arm == "qsrt" and (
            runtime["vllm_revision"] != runtime_identity.get("vllm_revision")
            or runtime["b12x_revision"] != runtime_identity.get("b12x_revision")
            or runtime["kquant_revision"] != encoder_identity.get("kquant_revision")
        ):
            raise ValueError("Fruit runtime qualification runtime revisions are stale")
        if (
            not isinstance(runtime["argv"], list)
            or not runtime["argv"]
            or any(not isinstance(item, str) or not item for item in runtime["argv"])
            or not isinstance(runtime["environment"], dict)
            or not runtime["environment"]
            or any(
                not isinstance(key, str) or not key or not isinstance(item, str)
                for key, item in runtime["environment"].items()
            )
            or not isinstance(runtime["software"], dict)
            or not runtime["software"]
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(item, str)
                or not item
                for key, item in runtime["software"].items()
            )
            or not isinstance(runtime["compilation_backend"], str)
            or not runtime["compilation_backend"]
            or not isinstance(runtime["cudagraph_mode"], str)
            or not runtime["cudagraph_mode"]
        ):
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} runtime is invalid"
            )
        measured_runtime_argv = _validate_runtime_argv(
            runtime["argv"],
            arm=arm,
            protocol=protocol,
            compilation_backend=runtime["compilation_backend"],
            cudagraph_mode=runtime["cudagraph_mode"],
        )
        if runtime["environment"] != _FIXED_RUNTIME_ENVIRONMENT:
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} environment does "
                "not match the sanitized production environment contract"
            )
        if normalized_runtime_argv is None:
            normalized_runtime_argv = measured_runtime_argv
        elif measured_runtime_argv != normalized_runtime_argv:
            raise ValueError(
                "Fruit runtime qualification arms do not use the same fixed "
                "launcher contract"
            )
        _qualification_string(loader["log_line"], name=f"loaders.{arm}.log_line")
        for key in (
            "weight_bytes",
            "peak_activation_bytes",
            "non_torch_bytes",
            "cudagraph_bytes",
            "kv_cache_bytes",
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "nvml_used_bytes",
        ):
            if type(loader[key]) is not int or loader[key] < 0:
                raise ValueError(
                    f"Fruit runtime qualification loaders.{arm}.{key} is invalid"
                )
        for key in ("weight_bytes", "cudagraph_bytes", "kv_cache_bytes"):
            if loader[key] <= 0:
                raise ValueError(
                    f"Fruit runtime qualification loaders.{arm}.{key} must be positive"
                )
        _qualification_number(
            loader["load_seconds"], name=f"loaders.{arm}.load_seconds", positive=True
        )

    repetitions = set(range(1, protocol["repetitions"] + 1))
    completion_counts: dict[int, int] = {}
    decode = _qualification_object(payload["decode"], name="decode", keys=_RUNTIME_ARMS)
    for arm in ("bf16", "siq", "qsrt"):
        runs = decode[arm]
        if not isinstance(runs, list) or len(runs) != len(repetitions):
            raise ValueError(f"Fruit runtime qualification decode.{arm} is incomplete")
        measured: set[int] = set()
        for index, raw_run in enumerate(runs):
            run = _qualification_object(
                raw_run,
                name=f"decode.{arm}[{index}]",
                keys={
                    "prompt_id",
                    "repetition",
                    "http_status",
                    "elapsed_seconds",
                    "completion_tokens",
                    "tokens_per_second",
                    "finish_reason",
                    "content",
                },
            )
            repetition = run["repetition"]
            if (
                run["prompt_id"] != decode_prompt_id
                or type(repetition) is not int
                or repetition not in repetitions
                or type(run["http_status"]) is not int
                or not 200 <= run["http_status"] < 300
            ):
                raise ValueError(
                    "Fruit runtime qualification decode identity is invalid"
                )
            measured.add(repetition)
            elapsed = _qualification_number(
                run["elapsed_seconds"],
                name=f"decode.{arm}.elapsed_seconds",
                positive=True,
            )
            tokens = run["completion_tokens"]
            if type(tokens) is not int or not 0 < tokens <= protocol["max_tokens"]:
                raise ValueError("Fruit runtime qualification token count is invalid")
            rate = _qualification_number(
                run["tokens_per_second"], name=f"decode.{arm}.rate", positive=True
            )
            if not math.isclose(rate, tokens / elapsed, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "Fruit runtime qualification token-rate math is invalid"
                )
            if arm == "bf16":
                completion_counts[repetition] = tokens
            elif completion_counts.get(repetition) != tokens:
                raise ValueError("Fruit runtime qualification completion counts differ")
            finish = _qualification_string(
                run["finish_reason"], name=f"decode.{arm}.finish"
            )
            if finish == "length" and tokens != protocol["max_tokens"]:
                raise ValueError("Fruit runtime qualification finish reason is invalid")
            if not isinstance(run["content"], str):
                raise ValueError(
                    "Fruit runtime qualification decode content is invalid"
                )
        if measured != repetitions:
            raise ValueError(
                f"Fruit runtime qualification decode.{arm} repetitions differ"
            )

    generation = _qualification_object(
        payload["generation"], name="generation", keys={"prompts", "results"}
    )
    prompts = generation["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Fruit runtime qualification generation prompts are empty")
    prompt_ids: list[str] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = _qualification_object(
            raw_prompt,
            name=f"generation.prompts[{index}]",
            keys={"id", "prompt", "prompt_token_ids"},
        )
        prompt_ids.append(
            _qualification_string(prompt["id"], name="generation.prompt.id")
        )
        _qualification_string(prompt["prompt"], name="generation.prompt.text")
        tokens = prompt["prompt_token_ids"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError(
                "Fruit runtime qualification generation tokens are invalid"
            )
    if len(set(prompt_ids)) != len(prompt_ids) or decode_prompt_id in prompt_ids:
        raise ValueError("Fruit runtime qualification prompt IDs are invalid")
    results = _qualification_object(
        generation["results"], name="generation.results", keys=_RUNTIME_ARMS
    )
    for arm in ("bf16", "siq", "qsrt"):
        rows = results[arm]
        if not isinstance(rows, list) or len(rows) != len(prompt_ids):
            raise ValueError(
                f"Fruit runtime qualification generation.{arm} is incomplete"
            )
        covered: list[str] = []
        for index, raw_result in enumerate(rows):
            result = _qualification_object(
                raw_result,
                name=f"generation.results.{arm}[{index}]",
                keys={"prompt_id", "content"},
            )
            covered.append(
                _qualification_string(result["prompt_id"], name="result.prompt_id")
            )
            if not isinstance(result["content"], str):
                raise ValueError(
                    "Fruit runtime qualification generation content is invalid"
                )
        if len(set(covered)) != len(covered) or set(covered) != set(prompt_ids):
            raise ValueError(
                f"Fruit runtime qualification generation.{arm} coverage differs"
            )

    fidelity = _qualification_object(
        payload["fidelity"],
        name="fidelity",
        keys={"full_vocabulary", "positions", "vocab_size", "candidates"},
    )
    positions = fidelity["positions"]
    if (
        fidelity["full_vocabulary"] is not True
        or not isinstance(positions, list)
        or not positions
        or any(type(position) is not int or position < 0 for position in positions)
        or len(set(positions)) != len(positions)
        or type(fidelity["vocab_size"]) is not int
        or fidelity["vocab_size"] <= 10
    ):
        raise ValueError("Fruit runtime qualification fidelity geometry is invalid")
    candidates = _qualification_object(
        fidelity["candidates"], name="fidelity.candidates", keys={"siq", "qsrt"}
    )
    for candidate_name in ("siq", "qsrt"):
        result = _qualification_object(
            candidates[candidate_name],
            name=f"fidelity.{candidate_name}",
            keys={
                "mean_forward_kl",
                "max_forward_kl",
                "top1_agreement",
                "top10_agreement",
                "per_position",
            },
        )
        rows = result["per_position"]
        if not isinstance(rows, list) or len(rows) != len(positions):
            raise ValueError("Fruit runtime qualification fidelity coverage differs")
        measured_positions: list[int] = []
        divergences: list[float] = []
        top1 = 0
        top10 = 0
        for raw_row in rows:
            row = _qualification_object(
                raw_row,
                name=f"fidelity.{candidate_name}.row",
                keys={"position", "forward_kl", "top1_agreement", "top10_agreement"},
            )
            if (
                type(row["position"]) is not int
                or type(row["top1_agreement"]) is not bool
                or type(row["top10_agreement"]) is not bool
                or (row["top1_agreement"] and not row["top10_agreement"])
            ):
                raise ValueError("Fruit runtime qualification fidelity row is invalid")
            measured_positions.append(row["position"])
            divergences.append(
                _qualification_number(row["forward_kl"], name="fidelity.forward_kl")
            )
            top1 += int(row["top1_agreement"])
            top10 += int(row["top10_agreement"])
        if measured_positions != positions:
            raise ValueError("Fruit runtime qualification fidelity positions differ")
        aggregates = {
            "mean_forward_kl": sum(divergences) / len(divergences),
            "max_forward_kl": max(divergences),
            "top1_agreement": top1 / len(divergences),
            "top10_agreement": top10 / len(divergences),
        }
        for name, expected in aggregates.items():
            measured_value = _qualification_number(
                result[name],
                name=f"fidelity.{candidate_name}.{name}",
                maximum=1.0 if "agreement" in name else None,
            )
            if not math.isclose(measured_value, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "Fruit runtime qualification fidelity aggregate is invalid"
                )


def verify_qsrt_publication(
    root: str | Path,
    *,
    expected_complete_sha256: str | None = None,
    candidate_mode: bool = False,
    expected_candidate_sha256: str | None = None,
    active_runtime_environment: dict[str, str] | None = None,
) -> QSRTPublicationSeal:
    """Authenticate a sealed local QSRT package before any runtime import."""

    if type(candidate_mode) is not bool:
        raise TypeError("candidate_mode must be a bool")
    if (
        active_runtime_environment is not None
        and active_runtime_environment != _FIXED_RUNTIME_ENVIRONMENT
    ):
        raise ValueError(
            "Fruit active runtime environment is not the fixed deployment contract"
        )
    if candidate_mode:
        if expected_complete_sha256 is not None:
            raise ValueError("candidate mode may not accept a completion digest")
        trusted_marker_sha256 = _sha256_field(
            expected_candidate_sha256, name="expected candidate marker"
        )
        marker_name = _CANDIDATE_MARKER
        marker_kind = "candidate"
    else:
        if expected_candidate_sha256 is not None:
            raise ValueError("production mode may not accept a candidate digest")
        trusted_marker_sha256 = _sha256_field(
            expected_complete_sha256, name="expected completion marker"
        )
        marker_name = _PUBLICATION_MARKER
        marker_kind = "completion"
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("QSRT model root must not be a symbolic link")
    try:
        resolved_root = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QSRT model root does not exist: {supplied}") from exc
    try:
        root_descriptor = os.open(supplied, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(f"cannot securely open QSRT model root: {supplied}") from exc
    if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
        os.close(root_descriptor)
        raise ValueError("QSRT model root must be a directory")

    opened: dict[str, int] = {}
    try:
        marker_descriptor = _open_beneath(root_descriptor, Path(marker_name))
        try:
            marker_bytes = _read_descriptor(marker_descriptor)
        finally:
            os.close(marker_descriptor)
        if hashlib.sha256(marker_bytes).hexdigest() != trusted_marker_sha256:
            raise ValueError(
                f"QSRT {marker_kind} marker does not match the trusted digest"
            )
        marker = _json_bytes(
            marker_bytes,
            kind=f"QSRT {marker_kind} marker",
            path=resolved_root / marker_name,
        )
        common_marker_fields = {
            "schema",
            "publication",
            "package_manifest_sha256",
            "checksum_manifest_sha256",
            "model_index_sha256",
            "source",
            "base_manifest_sha256",
            "producer_fingerprint",
            "encoder_fingerprint",
        }
        expected_marker_fields = (
            common_marker_fields
            if candidate_mode
            else common_marker_fields
            | {"qualified_candidate_sha256", "runtime_qualification_sha256"}
        )
        expected_schema = (
            "kquant_qsrt_candidate_v1" if candidate_mode else "kquant_qsrt_complete_v3"
        )
        if (
            set(marker) != expected_marker_fields
            or marker.get("schema") != expected_schema
        ):
            raise ValueError(f"QSRT {marker_kind} marker identity is invalid")
        if marker.get("publication") != {
            "variant": "instruct",
            "repository": _FRUIT_REPOSITORY,
        }:
            raise ValueError(f"QSRT {marker_kind} publication identity is invalid")
        digest_fields = [
            "package_manifest_sha256",
            "checksum_manifest_sha256",
            "model_index_sha256",
            "base_manifest_sha256",
            "producer_fingerprint",
            "encoder_fingerprint",
        ]
        if not candidate_mode:
            digest_fields.extend(
                ("qualified_candidate_sha256", "runtime_qualification_sha256")
            )
        for name in digest_fields:
            _sha256_field(marker.get(name), name=name)
        marker_source = marker.get("source")
        if (
            not isinstance(marker_source, dict)
            or set(marker_source) != {"kind", "sha256"}
            or not isinstance(marker_source.get("kind"), str)
            or not marker_source["kind"]
        ):
            raise ValueError(f"QSRT {marker_kind} marker source identity is invalid")
        _sha256_field(marker_source.get("sha256"), name="source.sha256")

        checksum_descriptor = _open_beneath(root_descriptor, Path(_CHECKSUM_MANIFEST))
        try:
            checksum_bytes = _read_descriptor(checksum_descriptor)
        finally:
            os.close(checksum_descriptor)
        if (
            hashlib.sha256(checksum_bytes).hexdigest()
            != marker["checksum_manifest_sha256"]
        ):
            raise ValueError(
                "QSRT checksum manifest does not match the completion marker"
            )
        try:
            checksum_text = checksum_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("QSRT checksum manifest is not UTF-8") from exc
        checksums: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for line in checksum_text.splitlines():
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
            descriptor = _open_beneath(root_descriptor, relative)
            opened[filename] = descriptor
            if _sha256_descriptor(descriptor) != digest:
                raise ValueError(f"QSRT package hash mismatch for {filename}")
            checksums[filename] = digest
            sizes[filename] = os.fstat(descriptor).st_size

        published_files = _published_files(root_descriptor)
        expected_files = set(checksums) | {_CHECKSUM_MANIFEST, marker_name}
        if published_files != expected_files:
            raise ValueError("QSRT checksum inventory does not match the package files")
        required_files = {
            "config.json",
            "model.safetensors.index.json",
            _PACKAGE_MANIFEST,
        }
        if not candidate_mode:
            required_files.add(_RUNTIME_QUALIFICATION)
        if not required_files.issubset(checksums):
            raise ValueError("QSRT checksum manifest omits a required identity file")
        if candidate_mode and _RUNTIME_QUALIFICATION in checksums:
            raise ValueError("QSRT candidate package contains a qualification receipt")
        if not candidate_mode and (
            checksums[_RUNTIME_QUALIFICATION] != marker["runtime_qualification_sha256"]
        ):
            raise ValueError(
                "QSRT runtime qualification digest disagrees with completion marker"
            )
        if checksums[_PACKAGE_MANIFEST] != marker["package_manifest_sha256"]:
            raise ValueError(
                "QSRT package manifest digest disagrees with completion marker"
            )
        if checksums["model.safetensors.index.json"] != marker["model_index_sha256"]:
            raise ValueError("QSRT model index digest disagrees with completion marker")

        manifest = _json_bytes(
            _read_descriptor(opened[_PACKAGE_MANIFEST]),
            kind="QSRT package manifest",
            path=resolved_root / _PACKAGE_MANIFEST,
        )
        producer = manifest.get("producer")
        source = manifest.get("source")
        base_model = manifest.get("base_model")
        publication = manifest.get("publication")
        evaluation = manifest.get("evaluation")
        qualification_binding = (
            evaluation.get("runtime_qualification")
            if isinstance(evaluation, dict)
            else None
        )
        expected_qualification_binding = (
            None
            if candidate_mode
            else {
                "file": _RUNTIME_QUALIFICATION,
                "sha256": checksums[_RUNTIME_QUALIFICATION],
            }
        )
        if (
            manifest.get("schema") != "kquant_qsrt_model_manifest_v1"
            or manifest.get("version") != 1
            or publication != {"variant": "instruct", "repository": _FRUIT_REPOSITORY}
            or manifest.get("codec") != "QSRT"
            or manifest.get("storage_schema") != "kquant_fruit_qsrt_atoms_v1"
            or manifest.get("encoding") != "qsrt_sqg_e4m3"
            or manifest.get("codebook") != "sqg_xor_cheb_t12"
            or manifest.get("profile_id") != 1
            or manifest.get("complete") is not (not candidate_mode)
            or not isinstance(producer, dict)
            or not isinstance(producer.get("encoder"), dict)
            or not isinstance(source, dict)
            or not isinstance(base_model, dict)
            or qualification_binding != expected_qualification_binding
        ):
            raise ValueError("QSRT package manifest identity is invalid or incomplete")
        expected_geometry = {
            "layers": list(_FRUIT_LAYERS),
            "experts_per_layer": _FRUIT_EXPERTS,
            "hidden_size": _FRUIT_HIDDEN_SIZE,
            "intermediate_size": _FRUIT_INTERMEDIATE_SIZE,
            "topk": _FRUIT_TOPK,
        }
        expected_runtime = {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        }
        layers = manifest.get("layers")
        if (
            manifest.get("geometry") != expected_geometry
            or manifest.get("runtime") != expected_runtime
            or not isinstance(layers, dict)
            or set(layers) != {str(layer) for layer in _FRUIT_LAYERS}
        ):
            raise ValueError("QSRT package has unsupported Fruit runtime geometry")
        for layer_number in _FRUIT_LAYERS:
            layer_name = str(layer_number)
            atom_name = f"qsrt-layer-{layer_number:03d}.safetensors"
            evidence_name = f"qsrt-layer-{layer_number:03d}.json"
            layer = layers[layer_name]
            if (
                not isinstance(layer, dict)
                or set(layer)
                != {"qsrt_atoms", "bytes", "sha256", "expert_count", "evidence"}
                or layer["qsrt_atoms"] != atom_name
                or layer["bytes"] != sizes.get(atom_name)
                or layer["sha256"] != checksums.get(atom_name)
                or layer["expert_count"] != _FRUIT_EXPERTS
                or layer["evidence"] != evidence_name
                or evidence_name not in checksums
            ):
                raise ValueError(
                    f"QSRT package layer {layer_number} geometry is invalid"
                )
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

        config = _json_bytes(
            _read_descriptor(opened["config.json"]),
            kind="QSRT model config",
            path=resolved_root / "config.json",
        )
        quantization = config.get("quantization_config")
        descriptor = (
            quantization.get("qsrt") if isinstance(quantization, dict) else None
        )
        if not isinstance(descriptor, dict):
            raise ValueError("QSRT model config omits its format descriptor")
        expected_hybrid_map = {
            str(layer): [3] * _FRUIT_EXPERTS for layer in _FRUIT_LAYERS
        }
        if (
            config.get("hidden_size") != _FRUIT_HIDDEN_SIZE
            or config.get("moe_intermediate_size") != _FRUIT_INTERMEDIATE_SIZE
            or config.get("n_routed_experts") != _FRUIT_EXPERTS
            or config.get("num_experts_per_tok") != _FRUIT_TOPK
            or config.get("num_hidden_layers") != 13
            or quantization.get("hybrid_bit_map") != expected_hybrid_map
            or quantization.get("kept_format") != "mxfp4_e8m0k32"
            or quantization.get("demoted_format") != "qsrt_sqg_e4m3"
            or descriptor.get("runtime") != "w4a8"
        ):
            raise ValueError("QSRT config has unsupported Fruit dispatch geometry")
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
        if any(value is None for value in expected_descriptor.values()):
            raise ValueError("QSRT package manifest omits a required descriptor field")
        if any(
            descriptor.get(name) != value for name, value in expected_descriptor.items()
        ):
            raise ValueError("QSRT model descriptor disagrees with the sealed manifest")
        tensor_digests = {
            name: digest
            for name, digest in sorted(checksums.items())
            if Path(name).parent == Path(".") and name.endswith(".safetensors")
        }
        if not tensor_digests:
            raise ValueError("QSRT package contains no candidate Safetensors")
        qualification: dict[str, Any] = {}
        if not candidate_mode:
            qualification_bytes = _read_descriptor(opened[_RUNTIME_QUALIFICATION])
            qualification = _json_bytes(
                qualification_bytes,
                kind="Fruit runtime qualification",
                path=resolved_root / _RUNTIME_QUALIFICATION,
            )
            canonical_qualification = (
                json.dumps(qualification, allow_nan=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            if qualification_bytes != canonical_qualification:
                raise ValueError("Fruit runtime qualification is not canonical JSON")
            _validate_runtime_qualification(
                qualification,
                manifest=manifest,
                config_digest=checksums["config.json"],
                model_index_digest=checksums["model.safetensors.index.json"],
                tensor_digests=tensor_digests,
                tensor_bytes=sum(sizes[name] for name in tensor_digests),
                candidate_marker_digest=marker["qualified_candidate_sha256"],
            )
            if (
                active_runtime_environment is not None
                and active_runtime_environment
                != qualification["loaders"]["qsrt"]["runtime"]["environment"]
            ):
                raise ValueError(
                    "Fruit active runtime environment differs from the sealed "
                    "qualification receipt"
                )
        elif (
            active_runtime_environment is not None
            and active_runtime_environment != _FIXED_RUNTIME_ENVIRONMENT
        ):
            raise ValueError(
                "Fruit candidate runtime environment differs from the fixed "
                "deployment contract"
            )

        atom_names: set[str] = set()
        layers = manifest.get("layers")
        if isinstance(layers, dict):
            for layer in layers.values():
                if isinstance(layer, dict) and isinstance(layer.get("qsrt_atoms"), str):
                    atom_names.add(layer["qsrt_atoms"])
        missing_atoms = atom_names - checksums.keys()
        if missing_atoms:
            raise ValueError(
                f"QSRT checksum manifest omits atom files: {sorted(missing_atoms)}"
            )
        atom_descriptors = {name: opened.pop(name) for name in atom_names}
        for descriptor_fd in opened.values():
            os.close(descriptor_fd)
        opened.clear()
        return QSRTPublicationSeal(
            root=resolved_root,
            manifest=manifest,
            descriptor=descriptor,
            config=config,
            qualification=qualification,
            checksums=checksums,
            _atom_descriptors=atom_descriptors,
        )
    except BaseException:
        for descriptor_fd in opened.values():
            os.close(descriptor_fd)
        raise
    finally:
        os.close(root_descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def snapshot_qsrt_publication(
    source: str | Path,
    destination: str | Path,
    *,
    expected_complete_sha256: str | None = None,
    candidate_mode: bool = False,
    expected_candidate_sha256: str | None = None,
    active_runtime_environment: dict[str, str] | None = None,
) -> QSRTPublicationSeal:
    """Copy, then authenticate, a package without trusting mutable source paths."""

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.is_symlink():
        raise ValueError("QSRT model root must not be a symbolic link")
    if destination_path.exists() or destination_path.is_symlink():
        raise ValueError("QSRT snapshot destination must not already exist")
    destination_parent = destination_path.parent.resolve(strict=True)
    if destination_path.parent != destination_parent:
        raise ValueError("QSRT snapshot parent must be a canonical private directory")
    os.mkdir(destination_path, mode=0o700)
    source_root = os.open(source_path, _DIRECTORY_FLAGS)
    destination_root = os.open(destination_path, _DIRECTORY_FLAGS)
    source_file_flags = _READ_FLAGS | os.O_NONBLOCK

    def validate_ignored_directory(directory: int, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                display = "/".join((*prefix, entry.name))
                if entry.is_symlink():
                    raise ValueError(f"QSRT source contains a symbolic link: {display}")
                if entry.is_dir(follow_symlinks=False):
                    child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=directory)
                    try:
                        validate_ignored_directory(child, (*prefix, entry.name))
                    finally:
                        os.close(child)
                    continue
                if (
                    not entry.is_file(follow_symlinks=False)
                    or entry.stat(follow_symlinks=False).st_nlink != 1
                ):
                    raise ValueError(
                        "QSRT source contains an unsafe transport-cache "
                        f"file: {display}"
                    )

    def copy_directory(
        source_directory: int,
        destination_directory: int,
        prefix: tuple[str, ...] = (),
    ) -> None:
        with os.scandir(source_directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                name = entry.name
                relative = (*prefix, name)
                if entry.is_symlink():
                    raise ValueError(f"QSRT source contains a symbolic link: {name}")
                if relative == _HF_LOCAL_CACHE_PREFIX:
                    if not entry.is_dir(follow_symlinks=False):
                        raise ValueError(
                            "QSRT Hugging Face transport cache is not a directory"
                        )
                    ignored = os.open(name, _DIRECTORY_FLAGS, dir_fd=source_directory)
                    try:
                        validate_ignored_directory(ignored, relative)
                    finally:
                        os.close(ignored)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    child_source = os.open(
                        name, _DIRECTORY_FLAGS, dir_fd=source_directory
                    )
                    os.mkdir(name, mode=0o700, dir_fd=destination_directory)
                    child_destination = os.open(
                        name, _DIRECTORY_FLAGS, dir_fd=destination_directory
                    )
                    try:
                        copy_directory(child_source, child_destination, relative)
                    finally:
                        os.close(child_destination)
                        os.close(child_source)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"QSRT source contains a non-regular file: {name}")
                source_descriptor = os.open(
                    name, source_file_flags, dir_fd=source_directory
                )
                try:
                    before = os.fstat(source_descriptor)
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        raise ValueError(
                            "QSRT source file must be regular and singly linked: "
                            f"{name}"
                        )
                    destination_descriptor = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                        0o600,
                        dir_fd=destination_directory,
                    )
                    copied = 0
                    try:
                        while chunk := os.read(source_descriptor, 8 << 20):
                            view = memoryview(chunk)
                            while view:
                                written = os.write(destination_descriptor, view)
                                if written <= 0:
                                    raise OSError(
                                        "short write while creating QSRT snapshot"
                                    )
                                copied += written
                                view = view[written:]
                    finally:
                        os.close(destination_descriptor)
                    after = os.fstat(source_descriptor)
                    path_after = os.stat(
                        name, dir_fd=source_directory, follow_symlinks=False
                    )
                    if (
                        copied != before.st_size
                        or _stat_identity(before) != _stat_identity(after)
                        or _stat_identity(after) != _stat_identity(path_after)
                    ):
                        raise RuntimeError(
                            f"QSRT source changed while creating the snapshot: {name}"
                        )
                finally:
                    os.close(source_descriptor)

    try:
        copy_directory(source_root, destination_root)
    except BaseException:
        # The caller owns removal of the private, possibly partial destination.
        raise
    finally:
        os.close(destination_root)
        os.close(source_root)
    return verify_qsrt_publication(
        destination_path,
        expected_complete_sha256=expected_complete_sha256,
        candidate_mode=candidate_mode,
        expected_candidate_sha256=expected_candidate_sha256,
        active_runtime_environment=active_runtime_environment,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an authenticated Fruit QSRT snapshot"
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("trusted_marker_sha256")
    parser.add_argument("--candidate-mode", action="store_true")
    arguments = parser.parse_args()
    active_runtime_environment = active_runtime_environment_from_env()
    seal = snapshot_qsrt_publication(
        arguments.source,
        arguments.destination,
        expected_complete_sha256=(
            None if arguments.candidate_mode else arguments.trusted_marker_sha256
        ),
        candidate_mode=arguments.candidate_mode,
        expected_candidate_sha256=(
            arguments.trusted_marker_sha256 if arguments.candidate_mode else None
        ),
        active_runtime_environment=active_runtime_environment,
    )
    seal.close()


if __name__ == "__main__":
    _main()
