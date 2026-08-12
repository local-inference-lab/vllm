# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security tests for EXL3 trust-boundary hardening (C1, C2, C3).

Covers:
  * validate_native_lib_path (shared canonical primitive in
    vllm/utils/path_validation.py) — absolute-path requirement, symlink
    component rejection, symlink-final rejection, non-regular rejection,
    group/world-writable file rejection, group/world-writable ancestor
    rejection, and a safe-path acceptance.
  * _validate_ext_search_dir (exl3.py) — symlink rejection (component and
    final), group/world-writable rejection, non-directory rejection,
    relative-path rejection, and a safe-dir acceptance.
  * _load_rank_sliced_bitrates bits_per_expert filename sanitization —
    path separators, absolute paths, '..', '.', and empty string are
    rejected; a safe bare filename is accepted via a stubbed
    get_hf_file_to_dict.
"""

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# validate_native_lib_path — canonical shared primitive
# ---------------------------------------------------------------------------


def test_validate_native_lib_path_rejects_relative(tmp_path):
    """A bare soname / relative path lets dlopen search its own path."""
    from vllm.utils.path_validation import validate_native_lib_path

    with pytest.raises(ValueError, match="absolute path"):
        validate_native_lib_path("libfoo.so", "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_symlink(tmp_path):
    """A symlink as the final component is rejected."""
    from vllm.utils.path_validation import validate_native_lib_path

    target = tmp_path / "real.so"
    target.write_text("dummy")
    link = tmp_path / "shim.so"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        validate_native_lib_path(str(link), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_symlink_ancestor(tmp_path):
    """A symlink *anywhere* on the path is rejected (islink only checks
    the final component, so realpath-equality is the real gate)."""
    from vllm.utils.path_validation import validate_native_lib_path

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    lib = real_dir / "good.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(real_dir, 0o755)

    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir)
    linked_lib = link_dir / "good.so"

    with pytest.raises(ValueError, match="symlink component"):
        validate_native_lib_path(str(linked_lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_nonexistent(tmp_path):
    """A non-existent path is rejected (os.lstat raises FileNotFoundError
    before the S_ISREG check)."""
    from vllm.utils.path_validation import validate_native_lib_path

    with pytest.raises((ValueError, FileNotFoundError)):
        validate_native_lib_path(str(tmp_path / "missing.so"),
                                 "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_world_writable(tmp_path):
    """A world-writable .so is rejected."""
    from vllm.utils.path_validation import validate_native_lib_path

    lib = tmp_path / "evil.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o666)  # rw-rw-rw-

    with pytest.raises(ValueError, match="group/world-writable"):
        validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_group_writable(tmp_path):
    """A group-writable .so is rejected."""
    from vllm.utils.path_validation import validate_native_lib_path

    lib = tmp_path / "evil.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o664)  # rw-rw-r--

    with pytest.raises(ValueError, match="group/world-writable"):
        validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")


def test_validate_native_lib_path_rejects_world_writable_parent(tmp_path):
    """A world-writable parent directory (non-sticky) is rejected."""
    from vllm.utils.path_validation import validate_native_lib_path

    lib = tmp_path / "evil.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o777)  # world-writable parent, no sticky bit

    try:
        with pytest.raises(ValueError, match="ancestor directory"):
            validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")
    finally:
        os.chmod(tmp_path, 0o755)  # restore for teardown


def test_validate_native_lib_path_accepts_safe(tmp_path):
    """A regular file in a safe directory passes and returns the canonical
    path."""
    from vllm.utils.path_validation import validate_native_lib_path

    lib = tmp_path / "good.so"
    lib.write_text("dummy")
    os.chmod(lib, 0o644)
    os.chmod(tmp_path, 0o755)

    result = validate_native_lib_path(str(lib), "VLLM_EXL3_ABI_SHIM")
    assert result == os.path.realpath(str(lib))


# ---------------------------------------------------------------------------
# _validate_ext_search_dir
# ---------------------------------------------------------------------------


def test_validate_ext_search_dir_rejects_relative(tmp_path):
    """A relative directory path is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    with pytest.raises(ValueError, match="absolute path"):
        _validate_ext_search_dir("some/rel/dir", "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_rejects_symlink(tmp_path):
    """A symlink as the final component is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    real = tmp_path / "real_dir"
    real.mkdir()
    os.chmod(real, 0o755)
    link = tmp_path / "link_dir"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="symlink"):
        _validate_ext_search_dir(str(link), "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_rejects_symlink_ancestor(tmp_path):
    """A symlink component anywhere on the path is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    os.chmod(real_dir, 0o755)
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir)

    target = link_dir / "sub"
    with pytest.raises(ValueError, match="symlink component"):
        _validate_ext_search_dir(str(target), "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_rejects_world_writable(tmp_path):
    """A world-writable directory is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    os.chmod(tmp_path, 0o777)

    try:
        with pytest.raises(ValueError, match="group/world-writable"):
            _validate_ext_search_dir(str(tmp_path), "VLLM_EXL3_EXT_PATH")
    finally:
        os.chmod(tmp_path, 0o755)


def test_validate_ext_search_dir_rejects_group_writable(tmp_path):
    """A group-writable directory is rejected."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    os.chmod(tmp_path, 0o775)

    try:
        with pytest.raises(ValueError, match="group/world-writable"):
            _validate_ext_search_dir(str(tmp_path), "VLLM_EXL3_EXT_PATH")
    finally:
        os.chmod(tmp_path, 0o755)


def test_validate_ext_search_dir_rejects_nonexistent(tmp_path):
    """A non-existent / non-directory path is rejected (os.lstat raises
    FileNotFoundError before the S_ISDIR check)."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        _validate_ext_search_dir(str(tmp_path / "nope"), "VLLM_EXL3_EXT_PATH")


def test_validate_ext_search_dir_accepts_safe(tmp_path):
    """A safe directory passes."""
    from vllm.model_executor.layers.quantization.exl3 import (
        _validate_ext_search_dir,
    )

    os.chmod(tmp_path, 0o755)
    _validate_ext_search_dir(str(tmp_path), "VLLM_EXL3_EXT_PATH")


# ---------------------------------------------------------------------------
# _load_rank_sliced_bitrates — bits_per_expert filename sanitization
# ---------------------------------------------------------------------------


def _make_config(bits_per_expert: str) -> "object":
    """Build a minimal Exl3Config for _load_rank_sliced_bitrates."""
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    config = Exl3Config.__new__(Exl3Config)
    config.rank_sliced_metadata = {
        "bits_per_expert": bits_per_expert,
        "experts_per_layer": 2,
        "moe_layers": [3, 4],
    }
    config.rank_sliced_k_values = [3, 4]
    return config


def _stub_payload(filename, model_name, revision=None):
    """A stub get_hf_file_to_dict returning a valid bitrate map."""
    return {
        "3": {"k": [3, 4]},
        "4": {"k": [4, 3]},
    }


def test_bits_per_expert_rejects_path_traversal():
    """C3: bits_per_expert filename with path separators is rejected."""
    config = _make_config("../../../../etc/passwd:k")

    with patch(
        "vllm.model_executor.layers.quantization.exl3.get_hf_file_to_dict"
    ) as mock:
        with pytest.raises(ValueError, match="bare file name"):
            config._load_rank_sliced_bitrates("dummy_model", revision=None)
        mock.assert_not_called()


def test_bits_per_expert_rejects_absolute_path():
    """C3: absolute path in bits_per_expert filename is rejected."""
    config = _make_config("/etc/passwd:k")

    with pytest.raises(ValueError, match="bare file name"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_rejects_dotdot():
    """C3: '..' passes the naive basename identity test but is a traversal."""
    config = _make_config("..:k")

    with pytest.raises(ValueError, match="bare file name"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_rejects_dot():
    """C3: '.' passes the naive basename identity test but is a traversal."""
    config = _make_config(".:k")

    with pytest.raises(ValueError, match="bare file name"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_rejects_empty():
    """C3: empty filename ('':k from ':k') is rejected."""
    config = _make_config(":k")

    with pytest.raises(ValueError, match="bare file name"):
        config._load_rank_sliced_bitrates("dummy_model", revision=None)


def test_bits_per_expert_accepts_safe_filename():
    """C3: a safe bare filename is accepted — exercises real vLLM code via
    a stubbed get_hf_file_to_dict."""
    config = _make_config("tier_bitmap.json:k")

    with patch(
        "vllm.model_executor.layers.quantization.exl3.get_hf_file_to_dict",
        side_effect=_stub_payload,
    ) as mock:
        config._load_rank_sliced_bitrates("dummy_model", revision=None)
        mock.assert_called_once_with(
            "tier_bitmap.json", "dummy_model", revision=None
        )

    # The function should have populated rank_sliced_bits_by_layer.
    assert config.rank_sliced_bits_by_layer == {
        3: (3, 4),
        4: (4, 3),
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
