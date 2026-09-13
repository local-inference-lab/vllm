# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import pickle
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_kernel_resolution,
)
from b12x.sequence import engram as native
from safetensors.torch import save_file
from torch import nn

from vllm.config.engram import EngramConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import default_loader, weight_utils
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4_1.common.engram import Engram, NgramHashState
from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask
from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4Model,
    DeepseekV41LLMForCausalLM,
)
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.models.deepseek_v4_1.nvidia.vl_model import (
    DeepseekV41ForCausalLM,
    _make_deepseek_v4_vl_weights_mapper,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.workspace import collect_cuda_graph_capture_resources


@pytest.fixture
def tiny_engram_checkpoint(tmp_path, dist_init):
    """Real target modules and native plans, omitting unrelated attention/MoE."""
    # Both global row counts leave two padded rows on the final TP4 shard.
    geometry = native.build_geometry(base_table_size=19, compressed_vocab_size=32)
    plans = tuple(
        native.plan(
            native.Caps(
                device="cuda",
                max_tokens=8,
                max_seqs=2,
                max_requests=2,
                vocab_size=32,
                layer_id=layer_id,
                tp_size=1,
                tp_rank=0,
            ),
            token_map=list(range(32)),
            geometry=geometry,
        )
        for layer_id in geometry.layer_ids
    )
    config = SimpleNamespace(
        hidden_size=64,
        hc_mult=4,
        rms_norm_eps=1e-6,
        num_attention_heads=1,
    )
    weights = {}
    paths = []
    for index, plan in enumerate(plans):
        prefix = f"layers.{plan.caps.layer_id}.engram."
        rows = torch.arange(plan.table_rows)[:, None]
        table = (rows.remainder(7) + index + 1).expand(-1, 256)
        layer_weights = {
            prefix + "embed.weight": table.to(torch.float8_e4m3fn).contiguous(),
            prefix + "embed.scale": torch.arange(124, 132, dtype=torch.uint8)
            .expand(plan.table_rows, -1)
            .contiguous()
            .view(torch.float8_e8m0fnu),
            prefix + "q_weight": torch.full((4, 64), 0.5, dtype=torch.bfloat16),
            prefix + "k_weight": torch.full((4, 64), 0.25, dtype=torch.bfloat16),
            prefix + "wkv.weight": torch.full(
                (320, 6144), (index + 1) / 8192, dtype=torch.bfloat16
            ),
        }
        path = tmp_path / f"layer-{index}.safetensors"
        save_file(layer_weights, str(path))
        paths.append(path)
        weights.update(layer_weights)

    def make_model(
        memory, *, tp_size=1, tp_rank=0, resident_scales=False, prefetch_max_tokens=0
    ):
        local_plans = tuple(
            native.plan(
                replace(plan.caps, tp_size=tp_size, tp_rank=tp_rank),
                token_map=list(range(32)),
                geometry=geometry,
            )
            for plan in plans
        )
        layout = SimpleNamespace(
            plans=local_plans,
            layer_ids=geometry.layer_ids,
            table_memory=memory,
            disk_resident_scales=resident_scales,
            disk_prefetch_max_tokens=prefetch_max_tokens,
        )
        target = DeepseekV4Model.__new__(DeepseekV4Model)
        nn.Module.__init__(target)
        target.config = config
        target.disk_engram = memory == "disk"
        target.file_backed_engram = memory in ("ram", "disk")
        target.engram_layout = layout
        target.start_layer, target.end_layer = 0, max(layout.layer_ids) + 1
        target.layers = nn.ModuleList(nn.Module() for _ in range(target.end_layer))
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                for index, layer_id in enumerate(layout.layer_ids):
                    target.layers[layer_id].engram = Engram(
                        config,
                        None,
                        layout,
                        index,
                        False,
                        f"model.layers.{layer_id}.engram",
                    )
        finally:
            torch.set_default_dtype(old_dtype)
        target.engram_hash = NgramHashState(None, layout, None)
        target.register_buffer(
            "prepared_engram_hashes",
            torch.empty(8, len(plans), 24, dtype=torch.int64, device="cuda"),
            persistent=False,
        )
        target.get_expert_mapping = lambda: []
        language = DeepseekV41LLMForCausalLM.__new__(DeepseekV41LLMForCausalLM)
        nn.Module.__init__(language)
        language.model = target
        language.hf_to_vllm_mapper = WeightsMapper()
        root = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        nn.Module.__init__(root)
        root.language_model = language
        root.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            "fp4", "weight_scale_inv"
        )
        return root, target

    return make_model, weights, paths


def _engrams(target):
    return tuple(
        target.layers[index].engram for index in target.engram_layout.layer_ids
    )


def _load_checkpoint(root, directory, load_format="safetensors"):
    loader = default_loader.DefaultModelLoader(
        LoadConfig(load_format=load_format, use_tqdm_on_load=False)
    )
    with torch.no_grad():
        return root.load_weights(
            loader.get_all_weights(
                SimpleNamespace(model=str(directory), revision=None), root
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "load_format", ["safetensors", "fastsafetensors", "instanttensor"]
)
@pytest.mark.parametrize("table_memory", ["disk", "ram"])
def test_engram_descriptor_loading_preserves_ordinary_weights(
    tiny_engram_checkpoint,
    tmp_path,
    monkeypatch,
    load_format,
    table_memory,
):
    """Raw table entries never reach payload readers or placeholder parameters."""
    make_model, weights, _ = tiny_engram_checkpoint
    root, target = make_model(table_memory)
    real_open = weight_utils.safe_open

    @contextmanager
    def guarded_open(*args, **kwargs):
        with real_open(*args, **kwargs) as archive:

            class GuardedArchive:
                def __getattr__(self, name):
                    return getattr(archive, name)

                def get_tensor(self, name):
                    assert not root.checkpoint_file_weight_filter(name)
                    return archive.get_tensor(name)

            yield GuardedArchive()

    def forbid_bulk(*args, **kwargs):
        raise AssertionError("mixed Engram file reached bulk payload loading")

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    monkeypatch.setattr(default_loader, f"{load_format}_weights_iterator", forbid_bulk)
    _load_checkpoint(root, tmp_path, load_format)
    for layer_id, engram in zip(target.engram_layout.layer_ids, _engrams(target)):
        for name in ("q_weight", "k_weight", "wkv.weight"):
            actual = engram.get_parameter(name)
            torch.testing.assert_close(
                actual.cpu(), weights[f"layers.{layer_id}.engram.{name}"]
            )
        indices = torch.arange(24, device="cuda").expand(4, -1).contiguous()
        count = torch.tensor([3], dtype=torch.int32, device="cuda")
        if table_memory == "disk":
            engram.prepare_disk(indices, count)
        else:
            engram.embed_tokens.lookup_native(indices[:3], engram.staged_rows)
        table = weights[f"layers.{layer_id}.engram.embed.weight"].float()
        expected = table[:24] * torch.pow(2.0, torch.arange(-3, 5)).repeat_interleave(
            32
        )
        torch.testing.assert_close(
            engram.staged_rows[:3].cpu(),
            expected.reshape(1, -1).expand(3, -1).to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(engram.staged_rows[3:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tp_rank", [0, 3])
@torch.inference_mode()
def test_ram_engram_checkpoint_tp4_graph_reads_live_host_aliases(
    tiny_engram_checkpoint, tmp_path, tp_rank
):
    """Packed host planes survive model deletion and are read afresh on replay."""
    from cuda.bindings import runtime as cudart

    make_model, weights, paths = tiny_engram_checkpoint
    root, target = make_model("ram", tp_size=4, tp_rank=tp_rank)
    _load_checkpoint(root, tmp_path)
    embedding = _engrams(target)[0].embed_tokens
    plan = embedding.plan
    prefix = f"layers.{plan.caps.layer_id}.engram.embed."
    source_weight = weights[prefix + "weight"]
    source_scale = weights[prefix + "scale"].view(torch.uint8)
    host_weight = embedding.weight_load_view
    host_scale = embedding.weight_scale_load_view
    local_rows = min(plan.shard_rows, plan.table_rows - plan.shard_start)
    for actual, source in (
        (host_weight, source_weight),
        (host_scale, source_scale),
    ):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(
            actual[:local_rows].view(torch.uint8),
            source[plan.shard_start : plan.shard_start + local_rows].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(actual[local_rows:].view(torch.uint8)) == 0
    assert embedding.mapped_host_nbytes == plan.shard_rows * (256 + 8)
    pointers = (
        (embedding.weight.data_ptr(), host_weight.data_ptr()),
        (embedding.weight_scale_inv.data_ptr(), host_scale.data_ptr()),
    )

    def assert_mapped_pointers():
        with torch.cuda.device(plan.caps.device):
            for device_pointer, host_pointer in pointers:
                error, attributes = cudart.cudaPointerGetAttributes(device_pointer)
                assert error == cudart.cudaError_t.cudaSuccess
                assert attributes.type == cudart.cudaMemoryType.cudaMemoryTypeHost
                assert int(attributes.devicePointer) == device_pointer
                assert int(attributes.hostPointer) == host_pointer

    assert_mapped_pointers()
    ids_cpu = torch.tensor(
        [
            [
                plan.shard_start,
                min(plan.shard_end, plan.table_rows) - 1,
                plan.shard_start - 1,
                plan.shard_end,
                plan.table_rows,
                -1,
            ]
            * 4
        ]
        * 4,
        dtype=torch.int64,
    )
    ids = ids_cpu.to(plan.caps.device)
    out = torch.full(
        (plan.caps.max_tokens, 6144),
        float("nan"),
        dtype=torch.bfloat16,
        device=plan.caps.device,
    )
    decoded = source_weight.float() * torch.pow(
        2.0, source_scale.float() - 127
    ).repeat_interleave(32, dim=1)

    def expected_output():
        expected = torch.zeros((plan.caps.max_tokens, 24, 256), dtype=torch.bfloat16)
        owned = (
            (ids_cpu >= plan.shard_start)
            & (ids_cpu < plan.shard_end)
            & (ids_cpu < plan.table_rows)
        )
        expected[: len(ids_cpu)][owned] = decoded[ids_cpu[owned]].to(torch.bfloat16)
        return expected.view(plan.caps.max_tokens, 6144)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    was_frozen = kernel_resolution_frozen()
    graph = torch.cuda.CUDAGraph()
    resources = []
    try:
        with torch.cuda.stream(stream):
            embedding.lookup_native(ids, out)
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)
            freeze_kernel_resolution("RAM Engram graph replay")
            with (
                collect_cuda_graph_capture_resources() as resources,
                torch.cuda.graph(graph, stream=stream),
            ):
                embedding.lookup_native(ids, out)
            # No checkpoint file or model reference is needed after capture.
            # Only the graph resource collector may retain the mapped owners.
            del embedding, target, root
            for path in paths:
                path.unlink()
            gc.collect()
            assert_mapped_pointers()
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)

            # Finish every GPU read before mutating write-combined CPU aliases.
            # Change both planes to detect a stale device copy of either one.
            torch.cuda.synchronize(plan.caps.device)
            host_weight[0].copy_(torch.full((256,), -4.0).to(torch.float8_e4m3fn))
            host_scale[0].view(torch.uint8).fill_(129)
            decoded[plan.shard_start].fill_(-16.0)
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)

            ids_cpu.fill_(-1)
            ids.copy_(ids_cpu)
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)
    finally:
        stream.synchronize()
        graph.reset()
        resources.clear()
        if not was_frozen:
            unfreeze_kernel_resolution()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("resident_scales", [False, True])
@pytest.mark.parametrize("prefetch_max_tokens", [0, 2, 8])
def test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows(
    tiny_engram_checkpoint,
    tmp_path,
    monkeypatch,
    resident_scales,
    prefetch_max_tokens,
):
    """Accepted GPU history, images, padding and failures cross the eager boundary."""
    make_model, weights, paths = tiny_engram_checkpoint
    root, target = make_model(
        "disk",
        resident_scales=resident_scales,
        prefetch_max_tokens=prefetch_max_tokens,
    )
    resident_root, resident = make_model("device")
    _load_checkpoint(root, tmp_path)
    with torch.no_grad():
        resident_root.load_weights(weights.items())
        for engram in (*_engrams(target), *_engrams(resident)):
            engram.process_weights_after_loading()

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = None
    state.lookback_token_ids = torch.empty(2, 3, dtype=torch.int32, device="cuda")
    state.disk_engram_models = (target,)
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32, device="cuda"),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([3, 4], dtype=torch.int32, device="cuda")
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor(
                [[5, 9, 13, 17, 19, 23], [4, 129264, 6, 8, 10, 12]],
                dtype=torch.int32,
                device="cuda",
            )
        ),
    )
    hidden = torch.ones(4, 4, 64, dtype=torch.bfloat16, device="cuda")

    def consume():
        mask = ~image_sentinel_mask(batch.input_ids)
        return tuple(
            engram(hidden, target.prepared_engram_hashes[:4, index], mask)
            for index, engram in enumerate(_engrams(target))
        )

    with pytest.raises(RuntimeError, match="not prepared"):
        consume()
    compiled_consume = torch.compile(consume, backend="eager", fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):

        def forbid_lookup(*args, **kwargs):
            raise AssertionError("capture preparation/consumer performed table lookup")

        with monkeypatch.context() as capture_guard:
            capture_guard.setattr(native, "run_lookup", forbid_lookup)
            state.prepare_dummy_inputs(2, 4)
            for _ in range(3):
                compiled_consume()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = compiled_consume()
            graph.replay()
            for output in actual:
                torch.testing.assert_close(output, hidden, rtol=0, atol=0)
        for ids, request, accepted, live in (
            ([11, 12, 31, 31], 0, 3, 2),
            ([21, 22, 23, 31], 0, 4, 3),
            ([7, 129264, 9, 31], 1, 3, 3),
            ([25, 31, 31, 31], 0, 1, 1),
        ):
            batch.input_ids.copy_(torch.tensor(ids, dtype=torch.int32, device="cuda"))
            batch.idx_mapping.fill_(request)
            req_states.num_computed_tokens.gpu[request] = accepted
            batch.query_start_loc[1:].fill_(live)
            state.prepare_inputs(batch, req_states)
            # Build expected lookback independently of the serving gather kernel.
            history = torch.full((2, 3), -1, dtype=torch.int32, device="cuda")
            width = min(accepted, 3)
            history[0, 3 - width :] = req_states.all_token_ids.gpu[
                request, accepted - width : accepted
            ]
            hashes = resident.engram_hash(
                batch.input_ids,
                None,
                batch.query_start_loc,
                image_sentinel_mask(batch.input_ids),
                history,
            )
            for index, engram in enumerate(_engrams(resident)):
                engram.prepare_embeddings(hashes[:, index])
            stats = [
                engram.embed_tokens.disk_table.stats() for engram in _engrams(target)
            ]
            graph.replay()
            assert stats == [
                engram.embed_tokens.disk_table.stats() for engram in _engrams(target)
            ]
            for index, engram in enumerate(_engrams(resident)):
                expected = engram(
                    hidden, hashes[:, index], ~image_sentinel_mask(batch.input_ids)
                )
                torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[index][live:], hidden[live:], rtol=0, atol=0
                )
            torch.testing.assert_close(
                actual[0][batch.input_ids == 129264], hidden[batch.input_ids == 129264]
            )

        # A smaller host preparation must retire previously populated rows,
        # even though this existing graph still consumes its four-row buffers.
        batch.input_ids.copy_(
            torch.tensor([11, 12, 13, 14], dtype=torch.int32, device="cuda")
        )
        batch.query_start_loc[1:].fill_(4)
        state.prepare_inputs(batch, req_states)
        graph.replay()
        first_rows = [output[:1].clone() for output in actual]
        small_batch = SimpleNamespace(**vars(batch))
        small_batch.input_ids = batch.input_ids[:1]
        batch.query_start_loc[1:].fill_(1)
        state.prepare_inputs(small_batch, req_states)
        graph.replay()
        for index, engram in enumerate(_engrams(target)):
            assert engram.staged_rows[1:].eq(0).all()
            torch.testing.assert_close(
                actual[index][:1], first_rows[index], rtol=0, atol=0
            )
            torch.testing.assert_close(actual[index][1:], hidden[1:], rtol=0, atol=0)

        # Fail after one layer has been refreshed: no layer may retain old rows.
        with open(paths[1], "r+b") as file:
            file.truncate(1)
        with pytest.raises((RuntimeError, OSError)):
            state.prepare_inputs(batch, req_states)
        with pytest.raises(RuntimeError, match="not prepared"):
            consume()
        graph.replay()
        for output in actual:
            torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


def test_dspark_checkpoint_filter_does_not_read_target_engram(tmp_path, monkeypatch):
    target = "layers.1.engram.embed.weight"
    draft = "mtp.0.main_proj.weight"
    target_path, draft_path = (
        tmp_path / "target.safetensors",
        tmp_path / "draft.safetensors",
    )
    save_file({target: torch.zeros(4, 8)}, str(target_path))
    save_file({draft: torch.ones(2, 2), target: torch.zeros(4, 8)}, str(draft_path))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {target: target_path.name, draft: draft_path.name}})
    )
    real_open = weight_utils.safe_open

    @contextmanager
    def guarded_open(path, **kwargs):
        assert str(path) != str(target_path)
        with real_open(path, **kwargs) as archive:

            class GuardedArchive:
                def __getattr__(self, name):
                    return getattr(archive, name)

                def get_tensor(self, name):
                    assert name != target
                    return archive.get_tensor(name)

            yield GuardedArchive()

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loaded = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None),
            SimpleNamespace(
                checkpoint_weight_name_prefixes=DSparkDeepseekV4ForCausalLM.checkpoint_weight_name_prefixes
            ),
        )
    )
    assert loaded.keys() == {draft}
    torch.testing.assert_close(loaded[draft], torch.ones(2, 2))


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_engram_storage_selection_changes_graph_configuration(cpu_offload):
    hashes = {
        EngramConfig(table_memory=memory, cpu_offload=cpu_offload).compute_hash()
        for memory in ("device", "ram", "disk")
    }
    assert len(hashes) == 3
    with pytest.raises(ValueError):
        EngramConfig(table_memory="invalid")
    disk_hashes = {
        EngramConfig(table_memory="disk", **options).compute_hash()
        for options in (
            {},
            {"disk_resident_scales": True},
            {"disk_prefetch_max_tokens": 32},
            {"projection_tp": True},
        )
    }
    assert len(disk_hashes) == 4


def test_disk_resident_scale_budget_charges_only_original_scale_bytes(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    model = SimpleNamespace(
        architecture="DeepseekV41ForCausalLM",
        hf_text_config=SimpleNamespace(
            engram_layer_ids=(1, 14), engram_num_embeddings=(17, 31)
        ),
    )
    available = (4 << 30) + (20 + 32) * 8
    monkeypatch.setattr(weight_utils, "_get_available_ram_bytes", lambda: available)
    config = EngramConfig(table_memory="disk", disk_resident_scales=True)
    config.verify_model_config(model, tp_size=4)
    available -= 1
    with pytest.raises(ValueError, match="Insufficient RAM"):
        EngramConfig(
            table_memory="disk", disk_resident_scales=True
        ).verify_model_config(model, tp_size=4)
    with pytest.raises(ValueError, match="table_memory"):
        EngramConfig(
            table_memory="device", disk_prefetch_max_tokens=32
        ).verify_model_config(model, tp_size=4)
    with pytest.raises(ValueError, match="nonnegative"):
        EngramConfig(
            table_memory="disk", disk_prefetch_max_tokens=-1
        ).verify_model_config(model, tp_size=4)


def test_ram_engram_budget_reserves_memory_and_survives_worker_serialization(
    monkeypatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    model_config = SimpleNamespace(
        architecture="DeepseekV41ForCausalLM",
        hf_text_config=SimpleNamespace(
            engram_layer_ids=(1, 14), engram_num_embeddings=(17, 31)
        ),
    )
    # TP4 rounds the two global tables to 20 and 32 packed 264-byte rows.
    available = (4 << 30) + (20 + 32) * 264 - 1
    monkeypatch.setattr(weight_utils, "_get_available_ram_bytes", lambda: available)
    config = EngramConfig(table_memory="ram")
    graph_hash = config.compute_hash()
    with pytest.raises(ValueError, match="Insufficient RAM"):
        config.verify_model_config(model_config, tp_size=4)

    available += 1
    config.verify_model_config(model_config, tp_size=4)
    assert config.compute_hash() == graph_hash

    # Workers see memory after their peers allocate. Revalidating the same
    # serialized preflight must not charge all tables against remaining RAM.
    worker_config = pickle.loads(pickle.dumps(config))
    available = 0
    worker_config.verify_model_config(model_config, tp_size=4)
    assert worker_config.compute_hash() == graph_hash
    with pytest.raises(ValueError, match="Insufficient RAM"):
        EngramConfig(table_memory="ram").verify_model_config(model_config, tp_size=4)
    # A different padded footprint is not covered by the original preflight.
    with pytest.raises(ValueError, match="Insufficient RAM"):
        worker_config.verify_model_config(model_config, tp_size=8)
