# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for GLM delimiters inside argument values."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.glm47_moe import (
    ARG_KEY_END,
    ARG_KEY_START,
    ARG_VALUE_END,
    ARG_VALUE_START,
    THINK_END,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Glm47MoeParser,
    _glm47_arg_converter,
)

TAGS = (
    ARG_VALUE_END,
    ARG_VALUE_START,
    ARG_KEY_END,
    ARG_KEY_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    THINK_END,
)

VALUE = "left </arg_value> then </tool_call> as data <tool_call> more"
OUTPUT = (
    f"{THINK_END}{TOOL_CALL_START}record_value"
    f"{ARG_KEY_START}value{ARG_KEY_END}{ARG_VALUE_START}{VALUE}{ARG_VALUE_END}"
    f"{ARG_KEY_START}path{ARG_KEY_END}{ARG_VALUE_START}/tmp/x{ARG_VALUE_END}"
    f"{TOOL_CALL_END}"
)
EXPECTED_ARGS = {"value": VALUE, "path": "/tmp/x"}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            "<think>": 154841,
            THINK_END: 154842,
            TOOL_CALL_START: 154843,
            TOOL_CALL_END: 154844,
            ARG_KEY_START: 154847,
            ARG_KEY_END: 154848,
            ARG_VALUE_START: 154849,
            ARG_VALUE_END: 154850,
            "<|observation|>": 154829,
        }
    )


@pytest.fixture
def tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="record_value",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            ),
        )
    ]


@pytest.fixture
def parser(mock_tokenizer, tools):
    return Glm47MoeParser(mock_tokenizer, tools=tools)


def _tag_chunks(output: str) -> list[str]:
    """Split output at every GLM tag boundary."""
    chunks: list[str] = []
    offset = 0
    while offset < len(output):
        tag = next((t for t in TAGS if output.startswith(t, offset)), None)
        if tag is not None:
            chunks.append(tag)
            offset += len(tag)
            continue
        ends = [p for t in TAGS if (p := output.find(t, offset + 1)) >= 0]
        end = min(ends, default=len(output))
        chunks.append(output[offset:end])
        offset = end
    return chunks


class TestConverter:
    def test_literal_arg_value_end_inside_value(self):
        raw = (
            "<arg_key>a</arg_key><arg_value>x</arg_value>y</arg_value>"
            "<arg_key>b</arg_key><arg_value>2</arg_value>"
        )
        assert json.loads(_glm47_arg_converter(raw, False)) == {
            "a": "x</arg_value>y",
            "b": "2",
        }

    def test_partial_holds_back_final_arg_value_end(self):
        raw = "<arg_key>a</arg_key><arg_value>x</arg_value>"
        assert json.loads(_glm47_arg_converter(raw, True)) == {"a": "x"}
        raw += "y"
        assert json.loads(_glm47_arg_converter(raw, True)) == {"a": "x"}
        raw += "</arg_value>"
        assert json.loads(_glm47_arg_converter(raw, True)) == {"a": "x</arg_value>y"}

    def test_whitespace_between_args_still_accepted(self):
        raw = (
            "<arg_key>a</arg_key>\n<arg_value>1</arg_value>\n"
            "<arg_key>b</arg_key>\n<arg_value>2</arg_value>\n"
        )
        assert json.loads(_glm47_arg_converter(raw, False)) == {"a": "1", "b": "2"}


class TestNonStreaming:
    def test_delimiters_inside_value(self, parser, mock_request, tools):
        mock_request.tools = tools
        result = parser.extract_tool_calls(OUTPUT, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "record_value"
        assert json.loads(result.tool_calls[0].function.arguments) == EXPECTED_ARGS
        assert result.content is None

    def test_missing_arg_value_end_keeps_tool_end_as_data(
        self, parser, mock_request, tools
    ):
        mock_request.tools = tools
        output = (
            f"{THINK_END}{TOOL_CALL_START}record_value"
            f"{ARG_KEY_START}value{ARG_KEY_END}{ARG_VALUE_START}unterminated"
            f"{TOOL_CALL_END}after"
        )

        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "value": f"unterminated{TOOL_CALL_END}after"
        }
        assert result.content is None

    def test_last_arg_value_end_drops_trailing_malformed_text(
        self, parser, mock_request, tools
    ):
        mock_request.tools = tools
        output = (
            f"{THINK_END}{TOOL_CALL_START}record_value"
            f"{ARG_KEY_START}value{ARG_KEY_END}{ARG_VALUE_START}Beijing"
            f"{ARG_VALUE_END} stray {TOOL_CALL_END}after"
        )

        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "value": "Beijing"
        }
        assert result.content is None


class TestStreaming:
    @pytest.mark.parametrize("split", ["tags", "chars"])
    def test_delimiters_inside_value(self, parser, mock_request, tools, split):
        mock_request.tools = tools
        chunks = _tag_chunks(OUTPUT) if split == "tags" else list(OUTPUT)
        results = simulate_tool_streaming(parser, mock_request, chunks)
        finish = parser.finish_streaming()
        if finish is not None:
            results.append((finish, OUTPUT))

        assert collect_function_name(results) == "record_value"
        assert json.loads(collect_tool_arguments(results)) == EXPECTED_ARGS
        assert collect_content(results) == ""

    def test_stop_token_id_without_text_in_final_delta(self, mock_request, tools):
        """Speculative decoding can deliver the whole call plus the stop
        token in one step; serving strips the stop text but keeps its ID.
        Literal tags inside the value carry the same IDs as real ones."""
        tokens = [
            (154842, THINK_END),
            (154843, TOOL_CALL_START),
            (400, "record_value"),
            (154847, ARG_KEY_START),
            (401, "value"),
            (154848, ARG_KEY_END),
            (154849, ARG_VALUE_START),
            (154844, TOOL_CALL_END),
            (402, " and "),
            (154843, TOOL_CALL_START),
            (154850, ARG_VALUE_END),
            (154844, TOOL_CALL_END),
            (154829, ""),
        ]
        vocab = {text: tid for tid, text in tokens if text}
        vocab["<think>"] = 154841
        vocab["<|observation|>"] = 154829
        tokenizer = make_mock_tokenizer(
            vocab,
            special_tokens=[
                t for t in vocab if t not in ("record_value", "value", " and ")
            ],
        )
        parser = Glm47MoeParser(tokenizer, tools=tools)
        mock_request.tools = tools
        text = "".join(t for _, t in tokens)
        ids = tuple(tid for tid, _ in tokens)

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=(),
            current_token_ids=ids,
            delta_token_ids=ids,
            request=mock_request,
        )
        results = [(delta, text), (parser.finish_streaming(), text)]

        assert collect_function_name(results) == "record_value"
        assert json.loads(collect_tool_arguments(results)) == {
            "value": "</tool_call> and <tool_call>"
        }
        assert collect_content(results) == ""
