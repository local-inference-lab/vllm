# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.deepseek_v32 import DSML_FUNC_END, DSML_FUNC_START
from vllm.parser.engine.registered_adapters import DeepSeekV32ParserToolAdapter


class DeepSeekV32EngineToolParser(DeepSeekV32ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "deepseek_v3_2"
    # DeepSeek occasionally emits a complete DSML block without closing
    # </think>; let DelegatingParser recover it from the reasoning channel.
    tool_call_start_token = DSML_FUNC_START
    tool_call_end_token = DSML_FUNC_END
    recovers_tool_calls_in_reasoning = True
