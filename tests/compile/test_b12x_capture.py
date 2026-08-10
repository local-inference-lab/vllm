# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.compilation import b12x_capture


def test_kernel_resolution_guard_falls_back_to_sparkinfer(monkeypatch) -> None:
    events: list[str] = []
    legacy = SimpleNamespace(
        freeze_kernel_resolution=lambda reason: events.append(f"freeze:{reason}"),
        kernel_resolution_frozen=lambda: False,
        unfreeze_kernel_resolution=lambda: events.append("unfreeze"),
    )

    def import_backend(namespace: str):
        if namespace == "b12x":
            raise ModuleNotFoundError(name=namespace)
        assert namespace == "sparkinfer"
        return legacy

    monkeypatch.setattr(b12x_capture, "b12x_cuda_graph_prewarm_enabled", lambda: True)
    monkeypatch.setattr(b12x_capture.importlib, "import_module", import_backend)

    with b12x_capture.guard_b12x_kernel_resolution("test capture"):
        events.append("body")

    assert events == ["freeze:test capture", "body", "unfreeze"]
