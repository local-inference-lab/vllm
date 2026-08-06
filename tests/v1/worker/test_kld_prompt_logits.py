# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm.v1.worker import gpu_model_runner, kld_capture


def _capture_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.safetensors"))


def test_kld_prompt_logit_capture_is_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_KLD_CAPTURE_DIR", raising=False)
    monkeypatch.setattr(kld_capture, "is_global_first_rank", lambda: True)

    gpu_model_runner._maybe_capture_kld_prompt_logits(
        torch.ones((2, 5), dtype=torch.bfloat16),
        req_id="request",
        start_idx=0,
        vocab_size=4,
    )

    assert _capture_files(tmp_path) == []


def test_kld_prompt_logit_capture_writes_rank_zero_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_KLD_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(kld_capture, "is_global_first_rank", lambda: True)
    logits = torch.arange(12, dtype=torch.float16).reshape(2, 6)

    gpu_model_runner._maybe_capture_kld_prompt_logits(
        logits,
        req_id="request/with spaces",
        start_idx=256,
        vocab_size=5,
    )

    path = tmp_path / "request_with_spaces" / "logits.rows-000256-000258.safetensors"
    assert _capture_files(tmp_path) == [path]
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.keys() == ["logits"]
        assert handle.metadata() == {
            "request_id": "request/with spaces",
            "row_start": "256",
            "row_end": "258",
            "vocab_size": "5",
        }
        saved = handle.get_tensor("logits")
    assert saved.dtype == torch.float32
    torch.testing.assert_close(saved, logits[:, :5].float())

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        gpu_model_runner._maybe_capture_kld_prompt_logits(
            logits,
            req_id="request/with spaces",
            start_idx=256,
            vocab_size=5,
        )


def test_kld_prompt_logit_capture_skips_nonzero_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_KLD_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(kld_capture, "is_global_first_rank", lambda: False)

    gpu_model_runner._maybe_capture_kld_prompt_logits(
        torch.ones((1, 3)),
        req_id="request",
        start_idx=0,
        vocab_size=3,
    )

    assert _capture_files(tmp_path) == []
