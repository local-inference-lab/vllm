# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from vllm.model_executor.layers.fused_moe import kquant_capture
from vllm.model_executor.layers.fused_moe.kquant_capture import (
    _KQuantCaptureState,
    _moe_row,
)


def test_k3_moe_layer_rows() -> None:
    assert _moe_row("language_model.model.layers.1.block_sparse_moe") == 0
    assert _moe_row("language_model.model.layers.92.block_sparse_moe") == 91


def test_pending_samples_are_batched_atomically(tmp_path: Path) -> None:
    state = _KQuantCaptureState.__new__(_KQuantCaptureState)
    state.samples_dir = tmp_path / "samples"
    state.parts = 0
    state.pending_samples = {}
    state.pending_sample_bytes = 0

    state._queue_samples(
        {
            "mid.values": torch.tensor([[1, 2]], dtype=torch.bfloat16),
            "mid.weight": torch.tensor([0.25], dtype=torch.float32),
        }
    )
    state._queue_samples(
        {
            "mid.values": torch.tensor([[3, 4]], dtype=torch.bfloat16),
            "mid.weight": torch.tensor([0.75], dtype=torch.float32),
        }
    )
    assert state.pending_sample_bytes > 0

    state._write_pending_samples()

    assert state.parts == 1
    assert state.pending_samples == {}
    assert state.pending_sample_bytes == 0
    tensors = load_file(tmp_path / "samples" / "part-00000001.safetensors")
    torch.testing.assert_close(
        tensors["mid.values"],
        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        tensors["mid.weight"], torch.tensor([0.25, 0.75], dtype=torch.float32)
    )


def test_copy_samples_preserves_route_pairing_and_splits() -> None:
    state = _KQuantCaptureState.__new__(_KQuantCaptureState)
    state.prefixes = {0: "language_model.model.layers.1.block_sparse_moe"}
    state.sample_capacity = 2
    state.input_dropped_total = 0
    state.mid_dropped_total = 0
    state.input_sample_cursor = torch.tensor([2], dtype=torch.int64)
    state.mid_sample_cursor = torch.tensor([1], dtype=torch.int64)
    state.input_sample_dropped = torch.zeros(1, dtype=torch.int64)
    state.mid_sample_dropped = torch.zeros(1, dtype=torch.int64)
    state.input_sample_values = torch.tensor([[[1, 2], [3, 4]]])
    state.input_sample_weight = torch.tensor([[0.5, 0.75]])
    state.input_sample_observation = torch.tensor([[10, 20]], dtype=torch.int64)
    state.input_sample_experts = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
    state.input_sample_gates = torch.tensor([[[0.6, 0.4], [0.7, 0.3]]])
    state.input_sample_split = torch.tensor([[0, 1]], dtype=torch.int8)
    state.input_sample_routed_latent = torch.tensor([[[5, 6], [7, 8]]])
    state.input_sample_latent_ready = torch.ones((1, 2), dtype=torch.int8)
    state.mid_sample_values = torch.tensor([[[9, 10], [0, 0]]])
    state.mid_sample_weight = torch.tensor([[0.25, 0.0]])
    state.mid_sample_observation = torch.tensor([[11, 0]], dtype=torch.int64)
    state.mid_sample_expert = torch.tensor([[7, 0]], dtype=torch.int32)
    state.mid_sample_split = torch.tensor([[1, 0]], dtype=torch.int8)

    samples = state._copy_samples()

    assert samples["input.experts"].tolist() == [[1, 2], [3, 4]]
    torch.testing.assert_close(
        samples["input.gates"],
        torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
    )
    assert samples["input.split"].tolist() == [0, 1]
    assert samples["input.routed_latent"].tolist() == [[5, 6], [7, 8]]
    assert samples["mid.expert"].tolist() == [7]
    assert samples["mid.split"].tolist() == [1]
    assert state.input_sample_cursor.item() == 0
    assert state.mid_sample_cursor.item() == 0
    assert torch.count_nonzero(state.input_sample_latent_ready).item() == 0


def test_exl3_mid_unscales_with_trailing_down_suh_block(monkeypatch) -> None:
    class FakeState:
        local_intermediate_size = 4

        def collect_mid(self, *args, **kwargs) -> None:
            self.collected = (args, kwargs)

    state = FakeState()
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/tmp/test.kqcapture")
    monkeypatch.setattr(kquant_capture, "_state", state)
    monkeypatch.setattr(
        torch.ops.vllm,
        "kquant_inverse_hadamard_128",
        lambda source, output: output.copy_(source),
    )

    from sparkinfer.moe import calibration

    unscale_args = {}

    def fake_unscale(*args, **kwargs) -> None:
        unscale_args.update(kwargs)

    monkeypatch.setattr(calibration, "unscale_route_rows_", fake_unscale)
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    expert_map = torch.arange(896, dtype=torch.int32)
    source = torch.arange(8, dtype=torch.float16)
    binding = SimpleNamespace(
        implementation="w4a16",
        quant_mode="w4a16",
        apply_router_weight_on_input=False,
        intermediate_cache2=source,
    )

    kquant_capture.collect_kquant_exl3_mid(
        prefix="language_model.model.layers.1.block_sparse_moe",
        binding=binding,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        intermediate_rotations=torch.ones((896, 12), dtype=torch.float16),
        logical_scratch=torch.empty_like(source),
    )

    assert unscale_args["scale_stride"] == 12
    assert unscale_args["scale_offset"] == 8
    assert hasattr(state, "collected")
