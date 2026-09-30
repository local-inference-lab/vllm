# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A returned GLM assistant turn re-renders to the tokens the model generated.

Drives OpenAIServingChat end to end (server-supplied chat template, glm45 and
glm47 parsers, streaming and non-streaming, return_token_ids) with a fake
engine, then sends the message back like an OpenAI client. The next prompt
must start with the previous prompt and every generated token except the
stop token, which is where a hybrid model's response checkpoint ends.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "zai-org/GLM-4.7"
# The GLM-5.x layout with assistant content rendered as generated, as the
# runtime's GLM template does.
TEMPLATE = (
    "[gMASK]<sop>"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
    "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<|assistant|>"
    "{%- if m.reasoning_content is string -%}"
    "{{ '<think>' + m.reasoning_content + '</think>' }}"
    "{%- else -%}<think></think>{%- endif -%}"
    "{{ m.content or '' }}"
    "{%- for tc in m.tool_calls or [] -%}<tool_call>{{ tc.function.name }}"
    "{%- for k, v in tc.function.arguments.items() -%}"
    "<arg_key>{{ k }}</arg_key><arg_value>"
    "{{ v if v is string else v | tojson }}</arg_value>"
    "{%- endfor -%}</tool_call>"
    "{%- endfor -%}"
    "{%- elif m.role == 'tool' -%}"
    "{%- if loop.first or messages[loop.index0 - 1].role != 'tool' -%}"
    "<|observation|>{%- endif -%}"
    "<tool_response>{{ m.content }}</tool_response>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
]
READ = (
    "<tool_call>read_file<arg_key>path</arg_key><arg_value>src/cache.py"
    "</arg_value><arg_key>limit</arg_key><arg_value>50</arg_value></tool_call>"
)
# Generated text after the prompt's <think>; each ends a turn differently.
GENERATIONS = {
    "answer_trailing_newline": "Plan.</think>```python\nx = 1\n```\n",
    "reasoning_trailing_newline": "Plan.\n</think>\nDone.",
    "content_before_tool": "Read first.</think>Let me read it.\n" + READ,
    "whitespace_before_tool": "Read first.</think>\n" + READ,
}


@dataclass
class _HFConfig:
    model_type: str = "glm4_moe"


@dataclass
class _ModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL
    tokenizer = MODEL
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 8192
    revision = None
    code_revision = None
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = _HFConfig()
    hf_text_config = _HFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class _ParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _VllmConfig:
    model_config: _ModelConfig
    parallel_config: _ParallelConfig


class _FakeEngine:
    """Replays one canned generation per request and records the prompts."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.generation = ""
        self.prompts: list[list[int]] = []
        self.outputs: list[list[int]] = []

    async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
        tok = self.tokenizer
        prompt_ids = list(prompt["prompt_token_ids"])
        stop = "<|observation|>" if "<tool_call>" in self.generation else "<|user|>"
        ids = tok.encode(self.generation, add_special_tokens=False)
        ids.append(tok.convert_tokens_to_ids(stop))
        self.prompts.append(prompt_ids)
        self.outputs.append(ids)

        def text(upto: int) -> str:
            # The stop token finishes the request; its text is not emitted.
            return tok.decode(ids[: min(upto, len(ids) - 1)], skip_special_tokens=False)

        def output(token_ids, delta_text, finished) -> RequestOutput:
            return RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=prompt_ids,
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=delta_text,
                        token_ids=token_ids,
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="stop" if finished else None,
                        stop_reason=ids[-1] if finished else None,
                    )
                ],
                finished=finished,
            )

        if sampling_params.output_kind != RequestOutputKind.DELTA:
            yield output(ids, text(len(ids)), True)
            return
        previous = ""
        for start in range(0, len(ids), 3):
            end = min(start + 3, len(ids))
            current = text(end)
            yield output(ids[start:end], current[len(previous) :], end == len(ids))
            previous = current


@pytest.fixture(scope="module")
def tokenizer():
    return cached_tokenizer_from_config(_ModelConfig())


def _serving(tokenizer) -> tuple[OpenAIServingChat, _FakeEngine]:
    model_config = _ModelConfig()
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = model_config
    engine.input_processor = MagicMock()
    engine.renderer = HfRenderer(
        _VllmConfig(model_config, _ParallelConfig()), tokenizer
    )
    fake = _FakeEngine(tokenizer)
    engine.generate = fake.generate
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=MODEL, model_path=MODEL)],
    )
    online = OnlineRenderer(
        model_config=model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=TEMPLATE,
        chat_template_content_format="string",
        enable_auto_tools=True,
        tool_parser="glm47",
        reasoning_parser="glm45",
    )
    serving = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online,
        chat_template=TEMPLATE,
        chat_template_content_format="string",
        request_logger=None,
        reasoning_parser="glm45",
        tool_parser="glm47",
        enable_auto_tools=True,
    )
    return serving, fake


async def _chat(serving: OpenAIServingChat, body: dict) -> dict:
    response = await serving.create_chat_completion(ChatCompletionRequest(**body))
    assert not hasattr(response, "error"), response
    if not body.get("stream"):
        return json.loads(response.model_dump_json())["choices"][0]["message"]
    message: dict[str, Any] = {"content": "", "reasoning": ""}
    calls: dict[int, dict] = {}
    async for line in response:
        payload = line.strip().removeprefix("data: ")
        if not payload or payload == "[DONE]":
            continue
        for choice in json.loads(payload)["choices"]:
            delta = choice["delta"]
            message["content"] += delta.get("content") or ""
            message["reasoning"] += delta.get("reasoning") or ""
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    call["index"], {"id": None, "name": "", "arguments": ""}
                )
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    message["tool_calls"] = [
        {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        }
        for _, call in sorted(calls.items())
    ]
    return message


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", GENERATIONS)
def test_next_prompt_extends_the_generated_turn(tokenizer, name, stream):
    asyncio.run(_two_turns(tokenizer, name, stream))


async def _two_turns(tokenizer, name: str, stream: bool) -> None:
    serving, engine = _serving(tokenizer)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Why is the new cache entry evicted?"},
    ]
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": stream,
        "return_token_ids": True,
    }
    engine.generation = GENERATIONS[name]
    message = await _chat(serving, body)

    returned: dict[str, Any] = {"role": "assistant", "content": message["content"]}
    if message.get("reasoning"):
        returned["reasoning_content"] = message["reasoning"]
    if message.get("tool_calls"):
        returned["tool_calls"] = message["tool_calls"]
        follow = [
            {"role": "tool", "tool_call_id": call["id"], "content": "x = 1"}
            for call in message["tool_calls"]
        ]
    else:
        follow = [{"role": "user", "content": "And in get()?"}]
    body["messages"] = [*messages, returned, *follow]
    await _chat(serving, body)

    first_prompt, first_output = engine.prompts[0], engine.outputs[0]
    generated = first_prompt + first_output[:-1]
    assert engine.prompts[1][: len(generated)] == generated
