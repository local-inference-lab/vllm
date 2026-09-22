# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)


def _make_kimi_linear_model() -> KimiLinearModel:
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.register_buffer("_attn_res_workspace", None, persistent=False)
    object.__setattr__(model, "reuse_attn_res_output", True)
    object.__setattr__(model, "aux_hidden_state_layers", (2,))
    object.__setattr__(model, "use_sequence_parallel", False)
    object.__setattr__(model, "use_attn_res", False)
    return model


def test_kimi_k3_advertises_eagle3_support():
    assert supports_eagle3(KimiK3ForConditionalGeneration)


def test_kimi_linear_advertises_eagle3_support():
    # The text-only architecture serves the same inner KimiLinearModel, which
    # already carries the EagleModelMixin tap machinery - only the interface
    # declaration was missing, so EAGLE3-family speculative decoding (dspark)
    # was rejected at startup with "Model does not support EAGLE3 interface".
    assert supports_eagle3(KimiLinearForCausalLM)


def test_kimi_k3_uses_shared_eagle3_layer_configuration():
    target = object.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    model = _make_kimi_linear_model()
    object.__setattr__(model, "layers", [None] * 93)
    language_model = SimpleNamespace(
        embed_input_ids=lambda _: None,
        forward=lambda input_ids, positions: None,
        model=model,
    )
    object.__setattr__(target, "language_model", language_model)
    object.__setattr__(target, "_language_model_names", ["language_model"])

    target.set_aux_hidden_state_layers((2, 46, 90))

    assert model.aux_hidden_state_layers == (2, 46, 90)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (
        2,
        46,
        90,
    )


def test_kimi_linear_forward_extracts_standard_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    layer_residual = torch.tensor([[5.0, 6.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, None, layer_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", False)
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    expected_layer_output = layer_hidden_states + layer_residual
    torch.testing.assert_close(output, expected_layer_output)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], expected_layer_output)


def test_kimi_linear_forward_extracts_attn_res_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    final_hidden_states = torch.tensor([[9.0, 10.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    final_attn_res = Mock(return_value=final_hidden_states)
    monkeypatch.setattr(kimi_model, "attn_res", final_attn_res)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], prefix_sum + layer_hidden_states)
    assert final_attn_res.call_args.args[2] is block_residual


def test_attn_res_stream_capture_receives_the_layer_outputs_in_order(monkeypatch):
    """Pin the argument mapping at the call site.

    The capture helper's own tests invoke it directly by keyword, so they
    cannot catch a swap where `forward` hands it the residual as the pending
    MLP output. Both are tensors of the same shape, so a swap is silent: it
    feeds the drafter a wrong but well-formed tensor.
    """
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    captured = torch.tensor([[11.0, 12.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (1,))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(kimi_model, "attn_res", Mock(return_value=torch.zeros(1, 2)))
    monkeypatch.setenv("VLLM_KIMI_K3_AUX_ATTN_RES_STREAM", "1")

    capture = Mock(return_value=captured)
    monkeypatch.setattr(KimiLinearModel, "_capture_aux_hidden_stream", capture)

    _, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    layer_idx, got_prefix, got_pending, got_residual = capture.call_args.args
    assert layer_idx == 0
    assert got_prefix is prefix_sum
    assert got_pending is layer_hidden_states
    assert got_residual is block_residual
    torch.testing.assert_close(aux_hidden_states[0], captured)


def _make_attn_res_decoder_layer(*, block_write: bool):
    layer = object.__new__(KimiDecoderLayer)
    torch.nn.Module.__init__(layer)
    object.__setattr__(layer, "use_attn_res", True)
    object.__setattr__(layer, "reuse_attn_res_output", True)
    object.__setattr__(layer, "is_block_write_layer", block_write)
    object.__setattr__(layer, "is_final_block_write_layer", False)
    object.__setattr__(layer, "block_write_idx", 0)
    object.__setattr__(layer, "prev_valid_blocks", 0)
    object.__setattr__(
        layer,
        "self_attention_res_norm",
        SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5),
    )
    object.__setattr__(
        layer,
        "self_attention_res_proj",
        SimpleNamespace(weight=torch.ones(1, 4)),
    )
    object.__setattr__(
        layer,
        "input_layernorm",
        SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5),
    )
    object.__setattr__(
        layer,
        "mlp_res_norm",
        SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5),
    )
    object.__setattr__(
        layer,
        "mlp_res_proj",
        SimpleNamespace(weight=torch.ones(1, 4)),
    )
    object.__setattr__(
        layer,
        "post_attention_layernorm",
        SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5),
    )
    return layer


def test_attn_res_prefill_reservation_keeps_decode_storage_stable():
    model = _make_kimi_linear_model()
    model.num_attn_res_blocks = 3
    model._max_num_batched_tokens = 16
    model._model_dtype = torch.bfloat16
    model.config = SimpleNamespace(hidden_size=4)
    model.register_parameter(
        "reference_weight", torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
    )

    model.reserve_attn_res_workspace()
    workspace = model._attn_res_workspace
    assert workspace.shape == (16, 3, 4)
    assert workspace.dtype == torch.bfloat16
    pointer = workspace.untyped_storage().data_ptr()
    for rows in (1, 8, 16, 2):
        view = model._get_attn_res_workspace(torch.empty(rows, 4, dtype=torch.bfloat16))
        assert view.shape == (rows, 3, 4)
        assert view.untyped_storage().data_ptr() == pointer
        assert view[:, 0].is_contiguous()
    model.reserve_attn_res_workspace()
    assert model._attn_res_workspace is workspace

    # Shapes beyond the reserved capacity must grow, not overrun its storage.
    grown = model._get_attn_res_workspace(torch.empty(17, 4, dtype=torch.bfloat16))
    assert grown.shape == (17, 3, 4)
    assert grown.untyped_storage().data_ptr() != pointer


def test_attn_res_unqualified_execution_does_not_share_workspace():
    model = _make_kimi_linear_model()
    model.reuse_attn_res_output = False
    model.num_attn_res_blocks = 3
    model.reserve_attn_res_workspace()
    first = model._get_attn_res_workspace(torch.empty(2, 4))
    second = model._get_attn_res_workspace(torch.empty(2, 4))
    assert first.data_ptr() != second.data_ptr()
    assert model._attn_res_workspace is None


def test_vision_borrows_residual_bank_without_replacing_graph_storage(monkeypatch):
    model = _make_kimi_linear_model()
    bank = torch.empty(8, 17, 4, dtype=torch.bfloat16).permute(1, 0, 2)
    model._attn_res_workspace = bank
    target = SimpleNamespace(
        language_model=SimpleNamespace(model=model),
        vision_tower=object(),
        mm_projector=object(),
        use_data_parallel=True,
    )
    received = []

    def encode(*args, projection_workspace, **kwargs):
        received.append(projection_workspace)
        return [torch.ones(1)]

    monkeypatch.setattr(kimi_model, "vision_tower_forward", encode)
    for reusable in (True, False):
        model.reuse_attn_res_output = reusable
        KimiK3ForConditionalGeneration._process_media_input(
            target, {"pixel_values": torch.empty(1), "grid_thws": torch.empty(1)}
        )
        if reusable:
            assert received[-1].data_ptr() == bank.data_ptr()
            assert received[-1].is_contiguous()
            assert received[-1].numel() == bank.numel()
        else:
            assert received[-1] is None
        assert model._attn_res_workspace is bank


def test_kimi_pre_attn_norm_uses_consumed_or_workspace_output(monkeypatch):
    layer = _make_attn_res_decoder_layer(block_write=True)
    prefix = torch.randn(2, 4)
    delta = torch.randn(2, 4)
    blocks = torch.randn(2, 3, 4)
    scratch = blocks[:, -1]
    calls = []

    def record_attn_res(*args, **kwargs):
        calls.append(kwargs["output"])
        output = kwargs["output"]
        return args[0].clone() if output is None else output

    monkeypatch.setattr(kimi_model, "attn_res", record_attn_res)

    first, _, _ = layer._pre_attn_norm(None, blocks, prefix, scratch)
    later, _, _ = layer._pre_attn_norm(delta, blocks, prefix, scratch)

    assert first is scratch
    assert later is delta
    assert calls == [scratch, delta]

    object.__setattr__(layer, "reuse_attn_res_output", False)
    layer._pre_attn_norm(delta, blocks, prefix, scratch)
    assert calls[-1] is None


def test_kimi_post_attn_norm_reuses_dead_input(monkeypatch):
    prefix = torch.randn(2, 4)
    attention_output = torch.randn(2, 4)
    blocks = torch.randn(2, 3, 4)
    outputs = []

    def record_attn_res(*args, **kwargs):
        outputs.append(kwargs["output"])
        return kwargs["output"]

    monkeypatch.setattr(kimi_model, "attn_res", record_attn_res)

    block_layer = _make_attn_res_decoder_layer(block_write=True)
    block_hidden, block_prefix, _ = block_layer._post_attn_norm(
        attention_output, blocks, prefix
    )
    regular_layer = _make_attn_res_decoder_layer(block_write=False)
    regular_hidden, regular_prefix, _ = regular_layer._post_attn_norm(
        attention_output, blocks, prefix
    )

    assert block_hidden is prefix
    assert block_prefix is attention_output
    assert regular_hidden is attention_output
    assert regular_prefix is prefix
    assert outputs == [prefix, attention_output]


def test_kimi_post_attn_norm_preserves_final_committed_block(monkeypatch):
    prefix = torch.randn(2, 4)
    attention_output = torch.randn(2, 4)
    blocks = torch.randn(2, 3, 4)
    allocated_output = torch.randn(2, 4)
    outputs = []

    def record_attn_res(*args, **kwargs):
        outputs.append(kwargs["output"])
        return allocated_output

    monkeypatch.setattr(kimi_model, "attn_res", record_attn_res)
    layer = _make_attn_res_decoder_layer(block_write=True)
    object.__setattr__(layer, "is_final_block_write_layer", True)

    hidden, next_prefix, next_blocks = layer._post_attn_norm(
        attention_output, blocks, prefix
    )

    assert hidden is allocated_output
    assert next_prefix is attention_output
    assert next_blocks is blocks
    assert outputs == [None]
