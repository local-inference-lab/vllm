# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.envs import (
    disable_envs_cache,
    enable_envs_cache,
    env_list_with_choices,
    env_set_with_choices,
    env_with_choices,
    environment_variables,
)


def test_getattr_without_cache(monkeypatch: pytest.MonkeyPatch):
    assert envs.VLLM_HOST_IP == ""
    assert envs.VLLM_PORT is None
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")


def test_nixl_side_channel_host_is_not_compile_factor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "10.0.0.15")

    assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in envs.compile_factors()


def test_p2p_side_channel_defaults_and_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_PORT", raising=False)
    assert envs.VLLM_P2P_SIDE_CHANNEL_HOST == "localhost"
    assert envs.VLLM_P2P_SIDE_CHANNEL_PORT == 5710

    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_HOST", "10.0.0.20")
    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5799")
    assert envs.VLLM_P2P_SIDE_CHANNEL_HOST == "10.0.0.20"
    assert envs.VLLM_P2P_SIDE_CHANNEL_PORT == 5799


def _clear_unknown_vllm_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove inherited VLLM variables not known by this source tree."""
    for name in list(os.environ):
        if name.startswith("VLLM_") and name not in environment_variables:
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", "12", 12),
        ("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "1", "1"),
        ("VLLM_PCIE_DMA_FP8", "ring", "ring"),
        ("VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", "56KB", "56KB"),
        ("VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS", "4", 4),
        ("VLLM_USE_B12X_PCIE_DMA", "1", True),
        ("VLLM_CACHE_DIR", "/cache/vllm", "/cache/vllm"),
    ],
)
def test_gilded_gnosis_runtime_envs_are_registered(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    expected: str | int,
) -> None:
    """Accept every GG runtime variable consumed outside the env registry."""
    _clear_unknown_vllm_envs(monkeypatch)
    monkeypatch.setenv(name, value)

    envs.validate_environ(hard_fail=True)
    assert environment_variables[name]() == expected


def test_gilded_gnosis_env_registration_keeps_unknown_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue rejecting misspelled VLLM variables after registration."""
    _clear_unknown_vllm_envs(monkeypatch)
    monkeypatch.setenv("VLLM_GILDED_GNOSIS_TYPO", "1")

    with pytest.raises(
        ValueError,
        match="Unknown vLLM environment variable detected: VLLM_GILDED_GNOSIS_TYPO",
    ):
        envs.validate_environ(hard_fail=True)


def test_getattr_with_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()

    # __getattr__ is decorated with functools.cache
    assert hasattr(envs.__getattr__, "cache_info")
    start_hits = envs.__getattr__.cache_info().hits

    # 2 more hits due to VLLM_HOST_IP and VLLM_PORT accesses
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    assert envs.__getattr__.cache_info().hits == start_hits + 2

    # All environment variables are cached
    for environment_variable in environment_variables:
        envs.__getattr__(environment_variable)
    assert envs.__getattr__.cache_info().hits == start_hits + 2 + len(
        environment_variables
    )

    # Reset envs.__getattr__ back to none-cached version to
    # avoid affecting other tests
    envs.__getattr__ = envs.__getattr__.__wrapped__


def test_getattr_with_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    # With cache enabled, the environment variable value is cached and unchanged
    monkeypatch.setenv("VLLM_HOST_IP", "2.2.2.2")
    assert envs.VLLM_HOST_IP == "1.1.1.1"

    disable_envs_cache()
    assert envs.VLLM_HOST_IP == "2.2.2.2"
    # After cache disabled, the environment variable value would be synced
    # with os.environ
    monkeypatch.setenv("VLLM_HOST_IP", "3.3.3.3")
    assert envs.VLLM_HOST_IP == "3.3.3.3"


def test_is_envs_cache_enabled() -> None:
    assert not envs._is_envs_cache_enabled()
    enable_envs_cache()
    assert envs._is_envs_cache_enabled()

    # Only wrap one-layer of cache, so we only need to
    # call disable once to reset.
    enable_envs_cache()
    enable_envs_cache()
    enable_envs_cache()
    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()

    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()


def test_precompiled_install_flags_are_orthogonal() -> None:
    # The Rust frontend flag is independent of the C-extension precompiled
    # flag: requesting the precompiled Rust frontend must not implicitly
    # enable the precompiled C extensions.
    with patch.dict(os.environ, {"VLLM_USE_PRECOMPILED_RUST": "1"}, clear=True):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is False
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True

    # ...and the reverse: requesting precompiled C extensions (here via a
    # wheel location, which enables VLLM_USE_PRECOMPILED) must not flip the
    # Rust frontend flag.
    with patch.dict(
        os.environ, {"VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl"}, clear=True
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is False

    # ...and with both set together, each flag is still parsed independently.
    with patch.dict(
        os.environ,
        {
            "VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        clear=True,
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True


class TestEnvWithChoices:
    """Test cases for env_with_choices function."""

    def test_default_value_returned_when_env_not_set(self):
        """Test default is returned when env var is not set."""
        env_func = env_with_choices(
            "NONEXISTENT_ENV", "default", ["option1", "option2"]
        )
        assert env_func() == "default"

    def test_none_default_returned_when_env_not_set(self):
        """Test that None is returned when env not set and default is None."""
        env_func = env_with_choices("NONEXISTENT_ENV", None, ["option1", "option2"])
        assert env_func() is None

    def test_valid_value_returned_case_sensitive(self):
        """Test that valid value is returned in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            assert env_func() == "option1"

    def test_valid_lowercase_value_returned_case_insensitive(self):
        """Test that lowercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["OPTION1", "OPTION2"], case_sensitive=False
            )
            assert env_func() == "option1"

    def test_valid_uppercase_value_returned_case_insensitive(self):
        """Test that uppercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == "OPTION1"

    def test_invalid_value_raises_error_case_sensitive(self):
        """Test that invalid value raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_case_mismatch_raises_error_case_sensitive(self):
        """Test that case mismatch raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'OPTION1' for TEST_ENV"
            ):
                env_func()

    def test_invalid_value_raises_error_case_insensitive(self):
        """Test that invalid value raises ValueError when case insensitive."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            assert env_func() == "dynamic1"

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()


class TestEnvListWithChoices:
    """Test cases for env_list_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_list_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == ["default1", "default2"]

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_list_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == []

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1"]

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == ["OPTION1", "option2"]

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == ["dynamic1", "dynamic2"]

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_preserved(self):
        """Test that duplicate values in the list are preserved."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option1", "option2"]


class TestEnvSetWithChoices:
    """Test cases for env_set_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_set_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == {"default1", "default2"}

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_set_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == set()

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1"}

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == {"OPTION1", "option2"}

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == {"dynamic1", "dynamic2"}

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_deduped(self):
        """Test that duplicate values in the list are deduped."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}


class TestVllmConfigureLogging:
    """Test cases for VLLM_CONFIGURE_LOGGING environment variable."""

    def test_configure_logging_defaults_to_true(self):
        """Test that VLLM_CONFIGURE_LOGGING defaults to True when not set."""
        # Ensure the env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if "VLLM_CONFIGURE_LOGGING" in os.environ:
                del os.environ["VLLM_CONFIGURE_LOGGING"]

            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_zero_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='0' evaluates to False."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "0"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is False
            assert isinstance(result, bool)

    def test_configure_logging_with_one_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='1' evaluates to True."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_invalid_value_raises_error(self):
        """Test that invalid VLLM_CONFIGURE_LOGGING value raises ValueError."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "invalid"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            with pytest.raises(ValueError, match="invalid literal for int"):
                _ = envs.VLLM_CONFIGURE_LOGGING


class TestVllmMaxNSequences:
    def test_default_value(self):
        """Test that VLLM_MAX_N_SEQUENCES defaults to 64."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MAX_N_SEQUENCES", None)
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            assert envs.VLLM_MAX_N_SEQUENCES == 16384

    def test_custom_value(self, monkeypatch: pytest.MonkeyPatch):
        """Test that VLLM_MAX_N_SEQUENCES can be overridden."""
        monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", "128")
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        assert envs.VLLM_MAX_N_SEQUENCES == 128

    def test_sampling_params_respects_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that SamplingParams rejects n above the limit."""
        from vllm.sampling_params import SamplingParams

        monkeypatch.delenv("VLLM_MAX_N_SEQUENCES", raising=False)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        max_n = envs.VLLM_MAX_N_SEQUENCES
        SamplingParams(n=max_n)

        with pytest.raises(ValueError, match="n must be at most"):
            SamplingParams(n=max_n + 1)

    def test_sampling_params_respects_custom_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that SamplingParams uses the overridden env var limit."""
        from vllm.sampling_params import SamplingParams

        monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", "128")
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        SamplingParams(n=128)

        with pytest.raises(ValueError, match="n must be at most 128"):
            SamplingParams(n=129)


# ---------------------------------------------------------------------------
# Gilded Gnosis / SparkInfer runtime environment variable registrations
# ---------------------------------------------------------------------------
#
# The following tests verify that every VLLM_* variable consumed by the GG
# PCIe all-reduce, B12X/SparkInfer sparse-MLA, NVFP4 MLA, and EXL3 paths —
# plus every VLLM_* variable exported by the GLM launcher scripts and
# documented in glm5.2_v20.md — is registered in ``environment_variables`` so
# ``validate_environ`` does not emit "Unknown vLLM environment variable"
# warnings during a real deployment.
#
# Each test covers default values, explicit overrides, and the negative case
# (an unregistered VLLM_* name still triggers a warning).


def _clear_envs_cache() -> None:
    """Clear the functools.cache wrapper on ``envs.__getattr__``."""
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


# All GG variables registered by this PR.  Each tuple is
# (env-var-name, override-value, expected-value-after-override).
_GG_ENV_VARS: list[tuple[str, str, object]] = [
    # PCIe DMA wire format (consumed by custom_all_reduce.py).
    ("VLLM_PCIE_DMA_FP8", "ag", "ag"),
    # C++ custom-allreduce crossover tuning (backend-dependent defaults).
    ("VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", "128KB", "128KB"),
    ("VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS", "1024", 1024),
    # B12X sparse-MLA speculative decode controls.
    ("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", "16", 16),
    ("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "1", "1"),
    # EXL3 online quantization knobs (exported by the GLM launcher).
    ("VLLM_EXL3_ONLINE_TRELLIS_BITS", "6", 6),
    ("VLLM_EXL3_ONLINE_CACHE_DIR", "/cache/exl3-online", "/cache/exl3-online"),
    ("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite", "readwrite"),
    ("VLLM_EXL3_PREFILL_CAPACITY", "1024", 1024),
    ("VLLM_EXL3_ENCODER_SOURCE", "/opt/exllamav3", "/opt/exllamav3"),
    ("VLLM_EXL3_ENCODER_REVISION", "abc123", "abc123"),
    # B12X PCIe DMA enable (consumed by the SparkInfer C++ extension).
    ("VLLM_USE_B12X_PCIE_DMA", "1", True),
    # Legacy alias for VLLM_CACHE_ROOT.
    ("VLLM_CACHE_DIR", "/test/vllm-cache", "/test/vllm-cache"),
]


def test_gg_env_vars_are_registered() -> None:
    """Every GG variable must appear in ``environment_variables``."""
    for name, _, _ in _GG_ENV_VARS:
        assert name in environment_variables, f"{name} is not registered"


def test_gg_env_vars_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults match the consuming code's existing behaviour."""
    # PCIe DMA FP8: unset → None (delegates to B12X_PCIE_DMA_FP8).
    monkeypatch.delenv("VLLM_PCIE_DMA_FP8", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_PCIE_DMA_FP8 is None
    # CPP AR cutoffs: unset → None (backend decides).
    monkeypatch.delenv("VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_CPP_AR_1STAGE_NCCL_CUTOFF is None
    monkeypatch.delenv("VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS is None
    # B12X spec decode max Q: default 8.
    monkeypatch.delenv("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_B12X_MLA_SPEC_DECODE_MAX_Q == 8
    # B12X spec extend as decode: default "auto".
    monkeypatch.delenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE == "auto"
    # EXL3 online vars: unset → None.
    for name in (
        "VLLM_EXL3_ONLINE_TRELLIS_BITS",
        "VLLM_EXL3_ONLINE_CACHE_DIR",
        "VLLM_EXL3_ONLINE_CACHE_MODE",
        "VLLM_EXL3_PREFILL_CAPACITY",
        "VLLM_EXL3_ENCODER_SOURCE",
        "VLLM_EXL3_ENCODER_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)
    _clear_envs_cache()
    assert envs.VLLM_EXL3_ONLINE_TRELLIS_BITS is None
    assert envs.VLLM_EXL3_ONLINE_CACHE_DIR is None
    assert envs.VLLM_EXL3_ONLINE_CACHE_MODE is None
    assert envs.VLLM_EXL3_PREFILL_CAPACITY is None
    assert envs.VLLM_EXL3_ENCODER_SOURCE is None
    assert envs.VLLM_EXL3_ENCODER_REVISION is None
    # B12X PCIe DMA enable: unset → False (bool(int("0"))).
    monkeypatch.delenv("VLLM_USE_B12X_PCIE_DMA", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_USE_B12X_PCIE_DMA is False
    # Legacy cache dir alias: unset → None.
    monkeypatch.delenv("VLLM_CACHE_DIR", raising=False)
    _clear_envs_cache()
    assert envs.VLLM_CACHE_DIR is None


@pytest.mark.parametrize("name,override,expected", _GG_ENV_VARS)
def test_gg_env_vars_override(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    override: str,
    expected: object,
) -> None:
    """Each GG variable honours an explicit env override."""
    monkeypatch.setenv(name, override)
    _clear_envs_cache()
    assert getattr(envs, name) == expected


def test_gg_env_vars_no_unknown_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting every GG variable must not produce unknown-var warnings."""
    # Clear all VLLM_ vars not in the registry, then set all GG vars.
    for name in list(os.environ):
        if name.startswith("VLLM_") and name not in environment_variables:
            monkeypatch.delenv(name, raising=False)
    for name, override, _ in _GG_ENV_VARS:
        monkeypatch.setenv(name, override)
    gg_names = {name for name, _, _ in _GG_ENV_VARS}
    warnings: list[str] = []
    monkeypatch.setattr(
        envs.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg)
    )
    envs.validate_environ(hard_fail=False)
    for w in warnings:
        if "Unknown vLLM environment variable" in w:
            for gg_name in gg_names:
                assert gg_name not in w, (
                    f"Registered GG var triggered unknown-var warning: {w}"
                )


def test_unknown_vllm_env_still_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered VLLM_* name must still trigger a warning."""
    # Clear other unregistered VLLM_ vars so only our sentinel is reported.
    for name in list(os.environ):
        if name.startswith("VLLM_") and name not in environment_variables:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_DEFINITELY_NOT_A_REAL_VARIABLE_42", "1")
    warnings: list[str] = []
    monkeypatch.setattr(
        envs.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg)
    )
    envs.validate_environ(hard_fail=False)
    assert any(
        "VLLM_DEFINITELY_NOT_A_REAL_VARIABLE_42" in w
        and "Unknown vLLM environment variable" in w
        for w in warnings
    )


def test_gg_env_vars_no_hard_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting every GG variable must not raise under hard_fail=True."""
    # Clear all unregistered VLLM_ vars from the environment (e.g. legacy
    # launcher exports still present in a container image), then set GG vars.
    for name in list(os.environ):
        if name.startswith("VLLM_") and name not in environment_variables:
            monkeypatch.delenv(name, raising=False)
    for name, override, _ in _GG_ENV_VARS:
        monkeypatch.setenv(name, override)
    envs.validate_environ(hard_fail=True)  # must not raise
