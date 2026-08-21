# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility tests for direct legacy DeepSeek V4 DSML tool calls."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.deepseek_v4 import (
    DSML_THINK_END,
    DSML_THINK_START,
    DeepSeekV4Parser,
)

_THINK_START_ID = 50
_THINK_END_ID = 51


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            DSML_THINK_START: _THINK_START_ID,
            DSML_THINK_END: _THINK_END_ID,
        }
    )


def _legacy_invoke(name: str, value: str) -> str:
    return (
        f'<|DSML|invoke name="{name}">\n'
        f'<|DSML|parameter name="command" string="true">{value}'
        "</|DSML|parameter>\n</|DSML|invoke>"
    )


def test_non_streaming_two_direct_invokes_are_tool_calls(mock_tokenizer, mock_request):
    text = "Preamble.\n" + _legacy_invoke("exec", "true")
    text += "\n" + _legacy_invoke("exec", "false")
    parser = DeepSeekV4Parser(mock_tokenizer, chat_template_kwargs={"thinking": False})

    reasoning, content, tool_calls = parser.parse(text, mock_request)

    assert reasoning is None
    assert content is not None
    assert content.rstrip() == "Preamble."
    assert tool_calls is not None
    assert [call.name for call in tool_calls] == ["exec", "exec"]
    assert [json.loads(call.arguments) for call in tool_calls] == [
        {"command": "true"},
        {"command": "false"},
    ]


def test_streaming_direct_invoke_returns_to_content(mock_tokenizer, mock_request):
    parser = DeepSeekV4Parser(mock_tokenizer, chat_template_kwargs={"thinking": False})
    text = _legacy_invoke("exec", "true") + "\nCalls completed."

    results = simulate_tool_streaming(parser, mock_request, [text])
    finish_delta = parser.finish_streaming()
    content = collect_content(results)
    if finish_delta and finish_delta.content:
        content += finish_delta.content

    assert collect_function_name(results) == "exec"
    assert content.strip() == "Calls completed."


def test_streaming_split_legacy_markers_do_not_leak(mock_tokenizer, mock_request):
    text = _legacy_invoke("exec", "true")
    parser = DeepSeekV4Parser(mock_tokenizer, chat_template_kwargs={"thinking": False})
    chunks = [text[i : i + 7] for i in range(0, len(text), 7)]

    results = simulate_tool_streaming(parser, mock_request, chunks)

    assert collect_function_name(results) == "exec"
    assert json.loads(collect_tool_arguments(results)) == {"command": "true"}
    assert "DSML" not in collect_content(results)
