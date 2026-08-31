# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.7"
GLM_OUTPUT_MODES = ("whole", "character", "protocol", "token")
VALUE = "left </arg_value> then </tool_call> as data more"
OUTPUT = (
    "</think><tool_call>record_value"
    "<arg_key>value</arg_key><arg_value>"
    f"{VALUE}</arg_value></tool_call>after"
)
RECORD_VALUE_TOOL = ChatCompletionToolsParam(
    function=FunctionDefinition(
        name="record_value",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["value"],
        },
    )
)


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
        RECORD_VALUE_TOOL,
    ]


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


@pytest.fixture
def glm_parser(glm47_tool_parser, mock_request):
    return glm47_tool_parser, mock_request


def _protocol_chunks(output: str) -> list[str]:
    terminals = (
        "</arg_value>",
        "<arg_value>",
        "</arg_key>",
        "<arg_key>",
        "</tool_call>",
        "<tool_call>",
        "</think>",
    )
    chunks = []
    offset = 0
    while offset < len(output):
        terminal = next(
            (item for item in terminals if output.startswith(item, offset)),
            None,
        )
        if terminal is not None:
            chunks.append(terminal)
            offset += len(terminal)
            continue
        next_offsets = [
            position
            for item in terminals
            if (position := output.find(item, offset + 1)) >= 0
        ]
        next_offset = min(next_offsets, default=len(output))
        chunks.append(output[offset:next_offset])
        offset = next_offset
    return chunks


def run_glm_output(glm_parser, output: str, *, mode: str):
    parser, request = glm_parser
    if mode == "whole":
        result = parser.extract_tool_calls(output, request=request)
        calls = [
            SimpleNamespace(
                index=index,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for index, call in enumerate(result.tool_calls)
        ]
        return SimpleNamespace(calls=calls, content=result.content or "")

    _reset(parser)
    stream_steps: list[tuple[str, list[int]]]
    if mode == "character":
        stream_steps = [(chunk, []) for chunk in output]
    elif mode == "protocol":
        stream_steps = [(chunk, []) for chunk in _protocol_chunks(output)]
    elif mode == "token":
        tokenizer = parser.model_tokenizer
        token_ids = tokenizer.encode(output, add_special_tokens=False)
        stream_steps = []
        previous_decoded = ""
        for index, token_id in enumerate(token_ids):
            decoded = tokenizer.decode(
                token_ids[: index + 1],
                skip_special_tokens=False,
            )
            assert decoded.startswith(previous_decoded)
            stream_steps.append((decoded[len(previous_decoded) :], [token_id]))
            previous_decoded = decoded
    else:
        raise ValueError(f"unknown GLM output mode: {mode}")

    current_text = ""
    current_token_ids: list[int] = []
    deltas = []
    for chunk, delta_token_ids in stream_steps:
        previous_text = current_text
        previous_token_ids = current_token_ids
        current_text += chunk
        current_token_ids = [*previous_token_ids, *delta_token_ids]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )
        if delta:
            deltas.append(delta)
    finish = parser.finish_streaming()
    if finish:
        deltas.append(finish)

    call_parts: dict[int, dict[str, str]] = {}
    for delta in deltas:
        for call in delta.tool_calls or []:
            parts = call_parts.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function:
                parts["name"] += call.function.name or ""
                parts["arguments"] += call.function.arguments or ""
    calls = [
        SimpleNamespace(index=index, **call_parts[index])
        for index in sorted(call_parts)
    ]
    content = "".join(delta.content or "" for delta in deltas)
    return SimpleNamespace(calls=calls, content=content)


def collect_calls(result):
    return result.calls


def collect_names(calls) -> list[str]:
    return [call.name for call in calls]


def collect_arguments(calls) -> str:
    assert len(calls) == 1
    return calls[0].arguments


def collect_content(result) -> str:
    return result.content


@pytest.fixture
def namespace_tool_request() -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__computer_use",
                    "description": "Computer use tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_app_state",
                            "description": "Get app state.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "app": {"type": "string"},
                                },
                            },
                        }
                    ],
                }
            ],
        }
    )


class TestGlm47ExtractToolCalls:
    def test_namespace_tool_call_round_trip_to_responses_output(
        self, glm47_tokenizer, namespace_tool_request
    ):
        parser = Glm47MoeModelToolParser(
            glm47_tokenizer, tools=namespace_tool_request.tools
        )
        out = (
            "<tool_call>mcp__computer_use__get_app_state"
            "<arg_key>app</arg_key>"
            "<arg_value>Google Chrome</arg_value>"
            "</tool_call>"
        )

        result = parser.extract_tool_calls(out, request=namespace_tool_request)

        assert result.tools_called
        tool_call = result.tool_calls[0].function
        assert tool_call == FunctionCall(
            name="mcp__computer_use__get_app_state",
            arguments='{"app": "Google Chrome"}',
        )

        output_items = build_response_output_items(
            reasoning=None,
            content=None,
            tool_calls=[tool_call],
            tools=namespace_tool_request.tools,
        )
        output_tool_call = output_items[0]
        assert isinstance(output_tool_call, ResponseFunctionToolCall)
        assert output_tool_call.name == "get_app_state"
        assert output_tool_call.namespace == "mcp__computer_use"

    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "  Beijing  "}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_tool_delimiters_in_arg_value(self, glm47_tool_parser, mock_request):
        value = "close </tool_call> then open <tool_call>"
        out = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key>"
            f"<arg_value>{value}</arg_value>"
            "</tool_call>"
        )

        result = glm47_tool_parser.extract_tool_calls(out, request=mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content is None
        function = result.tool_calls[0].function
        assert function.name == "get_weather"
        assert json.loads(function.arguments) == {"city": value}

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    def test_literal_arg_value_end_keeps_later_tool_end_as_data(
        self,
        mode,
        glm_parser,
    ):
        deltas = run_glm_output(glm_parser, OUTPUT, mode=mode)
        calls = collect_calls(deltas)
        assert {call.index for call in calls} == {0}
        assert collect_names(calls) == ["record_value"]
        assert json.loads(collect_arguments(calls)) == {"value": VALUE}
        assert collect_content(deltas) == "after"

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("</arg_value>starts", id="start"),
            pytest.param("middle </arg_value> continues", id="middle"),
            pytest.param("ends</arg_value>", id="adjacent-end"),
            pytest.param(
                "one </arg_value> two </arg_value> three",
                id="repeated",
            ),
            pytest.param(
                "left </arg_value> text <arg_key> <arg_value> </arg_key> "
                "<think> </think> <tool_call> </tool_call> right",
                id="later-terminals",
            ),
            pytest.param("雪 \\tmp\\file &amp; <b>bold</b>", id="data-bytes"),
            pytest.param("", id="empty"),
        ],
    )
    def test_literal_arg_value_end_edge_values(
        self,
        mode,
        value,
        glm_parser,
    ):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key>"
            f"<arg_value>{value}</arg_value>"
            "</tool_call>after"
        )

        result = run_glm_output(glm_parser, output, mode=mode)
        calls = collect_calls(result)

        assert {call.index for call in calls} == {0}
        assert collect_names(calls) == ["record_value"]
        assert json.loads(collect_arguments(calls)) == {"value": value}
        assert collect_content(result) == "after"

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    def test_literal_arg_key_start_before_real_second_argument(
        self,
        mode,
        glm_parser,
    ):
        value = "see <arg_key> tag"
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key>"
            f"<arg_value>{value}</arg_value>"
            "<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>"
            "</tool_call>after"
        )

        result = run_glm_output(glm_parser, output, mode=mode)
        calls = collect_calls(result)

        assert json.loads(collect_arguments(calls)) == {
            "value": value,
            "path": "/tmp/x",
        }
        assert collect_content(result) == "after"

    def test_exact_arg_value_end_arg_key_boundary_is_structural(self, glm_parser):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key><arg_value>left </arg_value>"
            "<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>"
            "</tool_call>"
        )

        result = run_glm_output(glm_parser, output, mode="whole")

        assert json.loads(collect_arguments(collect_calls(result))) == {
            "value": "left ",
            "path": "/tmp/x",
        }

    def test_exact_arg_value_end_tool_end_boundary_is_structural(self, glm_parser):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key><arg_value>left </arg_value>"
            "</tool_call>tail</arg_value></tool_call>"
        )

        result = run_glm_output(glm_parser, output, mode="whole")

        assert json.loads(collect_arguments(collect_calls(result))) == {
            "value": "left "
        }
        assert collect_content(result) == "tail</arg_value></tool_call>"

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    def test_stray_text_after_real_closer_uses_malformed_finish_contract(
        self,
        mode,
        glm_parser,
    ):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key><arg_value>Beijing</arg_value>"
            " ok </tool_call>trailing text"
        )

        result = run_glm_output(glm_parser, output, mode=mode)

        assert json.loads(collect_arguments(collect_calls(result))) == {
            "value": "Beijing"
        }
        # Bytes after a malformed boundary remain inside the open call.
        assert collect_content(result) == ""

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    def test_missing_real_arg_value_end_remains_malformed(
        self,
        mode,
        glm_parser,
    ):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key><arg_value>unterminated"
            "</tool_call>after"
        )

        result = run_glm_output(glm_parser, output, mode=mode)

        assert json.loads(collect_arguments(collect_calls(result))) == {
            "value": "unterminated</tool_call>after"
        }
        assert collect_content(result) == ""

    @pytest.mark.parametrize("mode", GLM_OUTPUT_MODES)
    def test_malformed_last_argument_keeps_prior_arguments_valid(
        self,
        mode,
        glm_parser,
    ):
        output = (
            "</think><tool_call>record_value"
            "<arg_key>value</arg_key><arg_value>first</arg_value>"
            "<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>"
            " junk </tool_call>trailing text"
        )

        result = run_glm_output(glm_parser, output, mode=mode)

        assert json.loads(collect_arguments(collect_calls(result))) == {
            "value": "first",
            "path": "/tmp/x",
        }
        assert collect_content(result) == ""

    def test_literal_arg_value_end_round_trips_to_responses_output(
        self,
        glm_parser,
    ):
        parser, request = glm_parser
        result = parser.extract_tool_calls(OUTPUT, request=request)
        function = result.tool_calls[0].function

        output_items = build_response_output_items(
            reasoning=None,
            content=result.content,
            tool_calls=[function],
            tools=request.tools,
        )
        response_call = next(
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        )

        assert response_call.name == "record_value"
        assert json.loads(response_call.arguments) == {"value": VALUE}


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


class TestGlm47Streaming:
    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        names = [
            tool_call.function.name
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.name
        ]
        arguments = [
            tool_call.function.arguments
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.arguments
        ]
        assert names == ["get_current_date"]
        assert "".join(arguments) == "{}"

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        arguments = [
            tool_call.function.arguments
            for delta in deltas
            for tool_call in (delta.tool_calls or [])
            if tool_call.function and tool_call.function.arguments
        ]
        args = json.loads("".join(arguments))
        assert args["city"] == "Beijing"

    def test_tool_delimiters_in_arg_value(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        value = "close </tool_call> then open <tool_call>"
        chunks = [
            "<tool_call>",
            "get_weather",
            "<arg_key>city</arg_key>",
            "<arg_value>close ",
            "</tool_call>",
            " then open ",
            "<tool_call>",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        finish = glm47_tool_parser.finish_streaming()
        if finish:
            deltas.append(finish)

        calls = [call for delta in deltas for call in (delta.tool_calls or [])]
        names = [
            call.function.name for call in calls if call.function and call.function.name
        ]
        arguments = [
            call.function.arguments
            for call in calls
            if call.function and call.function.arguments
        ]
        content = "".join(delta.content or "" for delta in deltas)

        assert {call.index for call in calls} == {0}
        assert names == ["get_weather"]
        assert content == ""
        assert json.loads("".join(arguments)) == {"city": value}
