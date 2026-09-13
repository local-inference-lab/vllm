# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning prefixes follow the DeepSeek-V4.1 published encoder contract."""

import pytest

from vllm.tokenizers.deepseek_v41_encoding import render_reasoning_effort


@pytest.mark.parametrize(
    ("effort", "budget"),
    [("low", 50), ("high", 75), ("max", 100), ("xhigh", 75), (None, 75)],
)
def test_named_and_default_effort_render_official_numeric_budget(effort, budget):
    assert render_reasoning_effort(0, "thinking", effort).startswith(
        f"Reasoning Effort: {budget} "
    )


@pytest.mark.parametrize("budget", [1, 25, 50, 75, 100])
def test_numeric_effort_keeps_explicit_budget(budget):
    assert render_reasoning_effort(0, "thinking", budget).startswith(
        f"Reasoning Effort: {budget} "
    )


@pytest.mark.parametrize(("index", "mode"), [(0, "chat"), (1, "thinking")])
def test_reasoning_prefix_is_only_written_for_first_thinking_message(index, mode):
    assert render_reasoning_effort(index, mode, "high") == ""
