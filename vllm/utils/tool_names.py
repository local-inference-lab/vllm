# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical function identities for namespace-bearing chat tools.

The namespace spelling follows Qizhou Guo's DeepSeek V4.1 reference encoder:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/dba1be0a40aa45a94ad051997016db3960a90277
"""

from typing import Any


def _namespace_parts(namespace: Any) -> tuple[str, str | None]:
    description = None
    if isinstance(namespace, dict):
        description = namespace.get("description")
        namespace = namespace.get("name")
    if not isinstance(namespace, str) or not namespace or "::" in namespace:
        raise ValueError("Tool namespace must be a nonempty string without '::'")
    if description is not None and not isinstance(description, str):
        raise ValueError("Tool namespace description must be a string")
    return namespace, description


def normalize_tool_namespace(tool: Any) -> Any:
    """Qualify explicit namespaces without mutating a definition or tool call.

    Accept a string or {name, description} namespace on the tool or function.
    Consistent duplicate declarations are allowed; conflicting identities are
    rejected. Remove namespace metadata after folding it into the function name
    and description, so normalization is idempotent across API and tokenizer
    boundaries. Inputs without namespace metadata retain their existing names.
    """
    if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
        return tool
    function = tool["function"]
    declarations = [
        _namespace_parts(value)
        for value in (tool.get("namespace"), function.get("namespace"))
        if value is not None
    ]
    if not declarations:
        return tool
    namespace, description = declarations[0]
    for declared_name, declared_description in declarations[1:]:
        if declared_name != namespace:
            raise ValueError("Conflicting tool namespaces on the tool and function")
        if description is None:
            description = declared_description

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Namespaced function name must be a nonempty string")
    if "::" in name:
        prefix, name = name.split("::", 1)
        if prefix != namespace:
            raise ValueError("Conflicting tool namespaces in the qualified name")
    if not name or "::" in name:
        raise ValueError("Namespaced function name must be nonempty without '::'")

    normalized = dict(tool)
    normalized.pop("namespace", None)
    normalized["function"] = function = dict(function)
    function.pop("namespace", None)
    function["name"] = f"{namespace}::{name}"
    if description:
        detail = function.get("description")
        if detail is not None and not isinstance(detail, str):
            raise ValueError("Function description must be a string")
        function["description"] = description + "\n" + (detail or "")
    return normalized
