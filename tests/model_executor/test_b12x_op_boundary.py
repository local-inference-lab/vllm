# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x-facing custom ops carry tensors, primitives, and layer names only.

Compiled graphs must never receive a Python object from the b12x integration.
Every ``direct_register_custom_op`` op function and every
``torch.library.custom_op`` in the b12x-related modules takes tensors,
optional tensors, tensor lists, ints, floats, bools, strings, ``torch.dtype``,
or a ``LayerNameType``; prepared state is resolved inside the op body.
"""
import ast
import pathlib
import re

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "vllm"

ALLOWED = {
    "torch.Tensor", "Tensor", "torch.Tensor | None", "Tensor | None",
    "list[torch.Tensor]", "list[Tensor]", "Sequence[torch.Tensor]",
    "list[torch.Tensor] | None", "list[Tensor] | None",
    "int", "int | None", "float", "float | None", "bool", "bool | None",
    "str", "str | None", "torch.dtype", "torch.dtype | None",
    "list[int]", "tuple[int, ...]", "list[int] | None", "list[float]",
    "LayerNameType", "LayerName", "str | LayerName",
}


def _annotation(node):
    if node is None:
        return None
    text = ast.unparse(node).replace("typing.", "")
    match = re.fullmatch(r"Optional\[(.+)\]", text)
    return f"{match.group(1)} | None" if match else text


def _op_functions(tree):
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = ast.unparse(target)
                if name.endswith("custom_op") or name.endswith("register_fake"):
                    yield node
                    break
        elif isinstance(node, ast.Call) and ast.unparse(node.func).endswith("direct_register_custom_op"):
            for keyword in node.keywords:
                if keyword.arg in ("op_func", "fake_impl") and isinstance(keyword.value, ast.Name):
                    function = functions.get(keyword.value.id)
                    if function is not None:
                        yield function


def op_boundary_violations(root=PACKAGE):
    violations = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if "b12x" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for function in _op_functions(tree):
            for argument in (*function.args.args, *function.args.kwonlyargs):
                annotation = _annotation(argument.annotation)
                if argument.arg == "self" or annotation is None or annotation in ALLOWED:
                    continue
                if True:
                    violations.append(
                        f"{path.relative_to(root.parent)}:{function.lineno} "
                        f"{function.name}({argument.arg}: {annotation})"
                    )
    return violations


def test_b12x_custom_ops_take_only_tensors_primitives_and_layer_names():
    violations = op_boundary_violations()
    assert not violations, "\n".join(violations)


if __name__ == "__main__":
    for line in op_boundary_violations():
        print(line)
