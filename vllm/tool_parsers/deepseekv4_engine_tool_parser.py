# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.deepseek_v4 import DSML_TOOL_END, DSML_TOOL_START
from vllm.parser.engine.registered_adapters import DeepSeekV4ParserToolAdapter


class DeepSeekV4EngineToolParser(DeepSeekV4ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v4"
    # DeepSeek occasionally emits a complete DSML block without closing
    # </think>; let DelegatingParser recover it from the reasoning channel.
    tool_call_start_token = DSML_TOOL_START
    tool_call_end_token = DSML_TOOL_END
    recovers_tool_calls_in_reasoning = True
