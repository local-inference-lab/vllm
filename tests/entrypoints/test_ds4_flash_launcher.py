# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCHER = _REPO_ROOT / "serve-ds4-flash.sh"


def _dry_run(tmp_path: Path, **overrides: str) -> str:
    """Run the DS4 launcher in dry-run mode.

    Args:
        tmp_path: Temporary home and cache root.
        **overrides: Environment variables that override launcher defaults.

    Returns:
        The launcher's standard-error output.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def test_ds4_launcher_defaults_to_0731_fixed_k7(tmp_path: Path) -> None:
    """Verify the default 0731 fixed-K7 launch profile.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path)

    assert "mode=dspark depth=fixed" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash-0731" in output
    revision = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
    assert f"--revision {revision}" in output
    assert f"--tokenizer-revision {revision}" in output
    assert 'num_speculative_tokens\\":7' in output
    assert "max_seqs=16 graph=128" in output
    assert "--max-model-len 131072" in output
    assert "--gpu-memory-utilization 0.975" in output
    assert (
        "Process-group interfaces: GLOO_SOCKET_IFNAME=lo "
        "NCCL_SOCKET_IFNAME=lo" in output
    )


def test_ds4_launcher_preserves_explicit_process_group_interfaces(
    tmp_path: Path,
) -> None:
    """Verify that a multi-node wrapper can select bootstrap interfaces.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        GLOO_SOCKET_IFNAME="bond0",
        NCCL_SOCKET_IFNAME="ib0",
    )

    assert (
        "Process-group interfaces: GLOO_SOCKET_IFNAME=bond0 "
        "NCCL_SOCKET_IFNAME=ib0" in output
    )


def test_ds4_launcher_accepts_distinct_tokenizer_revision(tmp_path: Path) -> None:
    """Verify that a tokenizer snapshot can be pinned independently.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        MODEL_REVISION="weights-commit",
        TOKENIZER_REVISION="tokenizer-commit",
    )

    assert "--revision weights-commit" in output
    assert "--tokenizer-revision tokenizer-commit" in output


def test_ds4_launcher_dynamic_depth_enables_capacity_mode(tmp_path: Path) -> None:
    """Verify that dynamic depth selects variable-capacity verification.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path, DSPARK_DEPTH_MODE="dynamic")

    assert "mode=dspark depth=dynamic capacity_activation=auto" in output
    assert 'dspark_capacity_verification_mode\\":\\"varlen' in output
    assert 'dspark_sps_curve\\":\\"auto' in output


def test_ds4_launcher_accepts_explicit_capacity_activation_threshold(
    tmp_path: Path,
) -> None:
    """Verify that a numeric capacity threshold overrides profile selection.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        DSPARK_DEPTH_MODE="dynamic",
        DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="4",
    )

    assert "mode=dspark depth=dynamic capacity_activation=4" in output


def test_ds4_launcher_uses_speculative_attention_backend_field(
    tmp_path: Path,
) -> None:
    """Verify the draft backend uses SpeculativeConfig's canonical field.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        DSPARK_DRAFT_ATTENTION_BACKEND="FLASHINFER_MLA_SPARSE_DSV4",
    )

    assert 'attention_backend\\":\\"FLASHINFER_MLA_SPARSE_DSV4' in output
    assert 'draft_attention_backend\\"' not in output


def test_ds4_launcher_accepts_cluster_style_aliases(tmp_path: Path) -> None:
    """Verify compatibility with cluster-style environment aliases.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        DCP="1",
        NUM_SPECULATIVE_TOKENS="5",
    )

    assert "dcp=1" in output
    assert 'num_speculative_tokens\\":5' in output


def test_ds4_launcher_standard_mtp_uses_standard_checkpoint(tmp_path: Path) -> None:
    """Verify that standard MTP selects the non-DSpark checkpoint.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path, MODE="mtp2")

    assert "mode=mtp2 depth=disabled" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash" in output
    assert "DeepSeek-V4-Flash-0731" not in output
    assert 'method\\":\\"mtp' in output
    assert 'num_speculative_tokens\\":2' in output


def test_ds4_launcher_can_disable_dspark_on_0731(tmp_path: Path) -> None:
    """Verify target-only serving with the 0731 checkpoint.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path, MODE="dspark-mtp0")

    assert "mode=dspark-mtp0 depth=disabled" in output
    assert "model=deepseek-ai/DeepSeek-V4-Flash-0731" in output
    assert "--speculative-config" not in output


def test_ds4_launcher_enables_native_kv_offload(tmp_path: Path) -> None:
    """Verify native host-memory KV offload arguments.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path, KV_OFFLOADING_SIZE="5.5")

    assert "--kv-offloading-size 5.5" in output
    assert "--kv-offloading-backend native" in output
    assert "allocator=expandable_segments:False" in output


def test_ds4_launcher_builds_bounded_native_l2_config(tmp_path: Path) -> None:
    """Verify bounded filesystem L2 configuration generation.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        KV_OFFLOADING_SIZE="32",
        NATIVE_L2_PATH="/cache/native-l2",
        NATIVE_L2_GB="128",
    )

    assert "native_l2=1" in output
    assert "--kv-transfer-config" in output
    assert "OffloadingConnector" in output
    assert "TieringOffloadingSpec" in output
    assert "cache/native-l2" in output
    assert 'gc_max_size_gb\\":128.0' in output


def test_ds4_launcher_native_l2_requires_l1(tmp_path: Path) -> None:
    """Verify that filesystem L2 requires native host-memory L1.

    Args:
        tmp_path: Temporary home and cache root.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        "NATIVE_L2_PATH": "/cache/native-l2",
        "NATIVE_L2_GB": "128",
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "NATIVE_L2 requires a positive KV_OFFLOADING_SIZE" in result.stderr


def test_ds4_launcher_native_l2_requires_complete_pair(tmp_path: Path) -> None:
    """Verify that both filesystem L2 settings are required.

    Args:
        tmp_path: Temporary home and cache root.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        "KV_OFFLOADING_SIZE": "32",
        "NATIVE_L2_PATH": "/cache/native-l2",
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "NATIVE_L2_PATH and NATIVE_L2_GB must be set together" in result.stderr


def test_ds4_launcher_native_offload_preserves_other_allocator_settings(
    tmp_path: Path,
) -> None:
    """Verify allocator normalization preserves unrelated settings.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        KV_OFFLOADING_SIZE="5.5",
        PYTORCH_CUDA_ALLOC_CONF=("max_split_size_mb:256,expandable_segments:True"),
    )

    assert "allocator=max_split_size_mb:256,expandable_segments:False" in output


def test_ds4_launcher_without_offload_keeps_expandable_segments(
    tmp_path: Path,
) -> None:
    """Verify the default allocator when native offload is disabled.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path)

    assert "allocator=expandable_segments:True" in output


def test_ds4_launcher_zero_kv_offload_stays_disabled(tmp_path: Path) -> None:
    """Verify that a zero offload size does not enable native offload.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(tmp_path, KV_OFFLOADING_SIZE="0.0")

    assert "--kv-offloading-size" not in output


def test_ds4_launcher_zero_native_l2_stays_disabled_with_path(
    tmp_path: Path,
) -> None:
    """Verify that a zero L2 capacity permits a reusable default path.

    Args:
        tmp_path: Temporary home and cache root.
    """
    output = _dry_run(
        tmp_path,
        NATIVE_L2_GB="0",
        NATIVE_L2_PATH="/cache/native-l2",
    )

    assert "native_l2=0" in output
    assert "--kv-transfer-config" not in output
    assert "OffloadingConnector" not in output


def test_ds4_launcher_rejects_invalid_kv_offload_size(tmp_path: Path) -> None:
    """Verify rejection of a nonnumeric native-offload size.

    Args:
        tmp_path: Temporary home and cache root.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        "KV_OFFLOADING_SIZE": "five",
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "KV_OFFLOADING_SIZE must be a non-negative GiB value" in result.stderr
