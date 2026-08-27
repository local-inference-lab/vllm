# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import inspect
import json
import struct
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.model_executor.model_loader.reload.layerwise import (
    _own_deferred_accelerator_tensors,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


def _safetensors_tensor_metadata(filename):
    with open(filename, "rb") as checkpoint:
        header_size = struct.unpack("<Q", checkpoint.read(8))[0]
        header = json.loads(checkpoint.read(header_size))
    return {name: item for name, item in header.items() if name != "__metadata__"}


class _FakeInstantOpen:
    def __init__(self, filename, tensors):
        self.filename = [str(filename)]
        self._tensors = tensors
        with weight_utils.safe_open(filename, framework="pt") as physical_file:
            names = list(physical_file.offset_keys())
        metadata = _safetensors_tensor_metadata(filename)
        self.original_names = names
        self.ordered_tensor_metadatas = [(name, metadata[name]) for name in names]
        first_offset = metadata[names[0]]["data_offsets"][0]
        self.tensor_offsets = [(0, first_offset)]
        self.tensor_offsets.extend(
            (0, metadata[name]["data_offsets"][1]) for name in names
        )
        self.tensor_sizes = [
            item["data_offsets"][1] - item["data_offsets"][0]
            for _, item in self.ordered_tensor_metadatas
        ]
        self.total_tensor_size = sum(self.tensor_sizes)
        self.tensor_name_to_index = {name: index for index, name in enumerate(names)}
        self.loader_handle = None
        self.buffer_size_requests = []
        self.enter_count = 0

    def _determine_buffer_size(self, requested):
        self.buffer_size_requests.append(requested)

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def tensors(self):
        for name, _ in self.ordered_tensor_metadatas:
            yield name, self._tensors[name]


@pytest.mark.parametrize(
    ("setting", "expected_copy", "expected_borrowed"),
    [("0", False, True), ("1", True, False)],
)
def test_instanttensor_copy_contract(
    setting, expected_copy, expected_borrowed, monkeypatch
):
    tensor = torch.ones(4)
    observed: dict[str, object] = {}

    class FakeReader:
        total_tensor_size = tensor.numel() * tensor.element_size()

        def keys(self):
            return ["weight"]

        def tensors(self):
            yield "weight", tensor

    @contextmanager
    def fake_safe_open(files, *, framework, device, process_group, copy):
        observed.update(
            files=files,
            framework=framework,
            device=device,
            process_group=process_group,
            copy=copy,
        )
        yield FakeReader()

    def no_world_group():
        raise AssertionError

    monkeypatch.setenv("INSTANTTENSOR_COPY", setting)
    monkeypatch.delenv("INSTANTTENSOR_BUFFER_SIZE", raising=False)
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=fake_safe_open)
    )

    loaded = list(instanttensor_weights_iterator(["model.safetensors"], False))

    assert len(loaded) == 1
    assert loaded[0][0] == "weight"
    assert loaded[0][1] is tensor
    assert observed == {
        "files": ["model.safetensors"],
        "framework": "pt",
        "device": 0,
        "process_group": None,
        "copy": expected_copy,
    }
    assert getattr(tensor, "_vllm_instanttensor_borrowed", False) is expected_borrowed


def test_instanttensor_copy_rejects_unknown_value(monkeypatch):
    def no_world_group():
        raise AssertionError

    monkeypatch.setenv("INSTANTTENSOR_COPY", "sometimes")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(sys.modules, "instanttensor", SimpleNamespace())

    with pytest.raises(ValueError, match="INSTANTTENSOR_COPY must be 0 or 1"):
        next(instanttensor_weights_iterator(["model.safetensors"], False))


def test_instanttensor_restricts_io_to_indexed_shards(tmp_path):
    base_shard = tmp_path / "model-00001-of-00002.safetensors"
    overlay_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.dense.weight": torch.tensor([2.0]),
            "model.expert.weight": torch.tensor([1.0, 1.0]),
        },
        base_shard,
    )
    save_file(
        {"model.expert.weight": torch.tensor([3.0, 3.0, 3.0])},
        overlay_shard,
    )

    buffer_sizes = []
    instant_open = SimpleNamespace(
        filename=[str(base_shard), str(overlay_shard)],
        ordered_tensor_metadatas=[
            ("model.dense.weight", {"data_offsets": [0, 4]}),
            ("model.expert.weight", {"data_offsets": [4, 12]}),
            ("model.expert.weight", {"data_offsets": [0, 12]}),
        ],
        tensor_offsets=[
            (0, 0),
            (0, 4),
            (0, 12),
            (1, 0),
            (1, 12),
        ],
        tensor_sizes=[4, 8, 12],
        total_tensor_size=24,
        tensor_name_to_index={},
        loader_handle=None,
        _determine_buffer_size=lambda requested: buffer_sizes.append(requested),
    )
    indexed_tensor_files = {
        "model.dense.weight": str(base_shard.resolve()),
        "model.expert.weight": str(overlay_shard.resolve()),
    }

    selection = weight_utils._restrict_instanttensor_to_selected_ranges(
        instant_open,
        indexed_tensor_files=indexed_tensor_files,
        weight_name_prefixes=None,
    )

    assert instant_open.filename == [str(base_shard), str(overlay_shard)]
    assert [name for name, _ in instant_open.ordered_tensor_metadatas] == [
        "model.dense.weight",
        "model.expert.weight",
    ]
    assert instant_open.tensor_offsets == [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 12),
    ]
    assert instant_open.tensor_sizes == [4, 12]
    assert instant_open.total_tensor_size == 16
    assert instant_open.tensor_name_to_index == {
        "model.dense.weight": 0,
        "model.expert.weight": 1,
    }
    assert buffer_sizes == [None]
    assert selection.gpu_tensor_count == 2
    assert selection.selected_tensor_count == 2
    assert selection.cpu_fallbacks == ()


def test_instanttensor_routes_oversized_selected_tensors_to_cpu(tmp_path):
    shard = tmp_path / "model.safetensors"
    save_file(
        {
            "model.large.weight": torch.arange(8, dtype=torch.float32),
            "model.small.weight": torch.arange(2, dtype=torch.float32),
        },
        shard,
    )
    with weight_utils.safe_open(shard, framework="pt") as physical_file:
        physical_names = list(physical_file.offset_keys())
    metadata_by_name = _safetensors_tensor_metadata(shard)
    metadata = [(name, metadata_by_name[name]) for name in physical_names]
    offsets = [(0, metadata_by_name[physical_names[0]]["data_offsets"][0])]
    offsets.extend(
        (0, metadata_by_name[name]["data_offsets"][1]) for name in physical_names
    )
    buffer_sizes = []
    instant_open = SimpleNamespace(
        filename=[str(shard)],
        ordered_tensor_metadatas=metadata,
        tensor_offsets=offsets,
        tensor_sizes=[32, 8],
        total_tensor_size=40,
        tensor_name_to_index={},
        loader_handle=None,
        _determine_buffer_size=lambda requested: buffer_sizes.append(requested),
    )

    selection = weight_utils._restrict_instanttensor_to_selected_ranges(
        instant_open,
        indexed_tensor_files=None,
        weight_name_prefixes=None,
        max_tensor_size=16,
    )

    assert [name for name, _ in instant_open.ordered_tensor_metadatas] == [
        "model.small.weight"
    ]
    assert selection.gpu_tensor_count == 1
    assert selection.selected_tensor_count == 2
    assert [
        (fallback.position, fallback.name, fallback.filename)
        for fallback in selection.cpu_fallbacks
    ] == [(0, "model.large.weight", str(shard))]
    assert instant_open.total_tensor_size == 8
    assert buffer_sizes == [None]


def test_instanttensor_rejects_no_match_with_irrelevant_exclusion(tmp_path):
    shard = tmp_path / "model.safetensors"
    source = {"model.weight": torch.tensor([1.0])}
    save_file(source, shard)
    instant_open = _FakeInstantOpen(shard, source)

    with pytest.raises(
        RuntimeError,
        match="InstantTensor index/prefix selection matched no tensors",
    ):
        weight_utils._restrict_instanttensor_to_selected_ranges(
            instant_open,
            indexed_tensor_files=None,
            weight_name_prefixes=["vision_tower"],
            excluded_tensor_names={"model.weight"},
        )


@pytest.mark.parametrize("use_index", [False, True])
def test_instanttensor_emits_priority_tensors_before_gpu_staging(
    tmp_path, monkeypatch, use_index
):
    regular_shard = tmp_path / "model-00001-of-00002.safetensors"
    priority_shard = tmp_path / "model-00002-of-00002.safetensors"
    regular_tensor = torch.tensor([1.0, 2.0])
    priority_tensor = torch.tensor([3.0, 4.0])
    save_file({"model.weight": regular_tensor}, regular_shard)
    save_file({"vision_tower.weight": priority_tensor}, priority_shard)

    regular_metadata = _safetensors_tensor_metadata(regular_shard)["model.weight"]
    priority_metadata = _safetensors_tensor_metadata(priority_shard)[
        "vision_tower.weight"
    ]

    class FakeReader:
        def __init__(self):
            self.filename = [str(regular_shard), str(priority_shard)]
            self.ordered_tensor_metadatas = [
                ("model.weight", regular_metadata),
                ("vision_tower.weight", priority_metadata),
            ]
            self.tensor_offsets = [
                (0, regular_metadata["data_offsets"][0]),
                (0, regular_metadata["data_offsets"][1]),
                (1, priority_metadata["data_offsets"][0]),
                (1, priority_metadata["data_offsets"][1]),
            ]
            self.tensor_sizes = [
                regular_tensor.numel() * regular_tensor.element_size(),
                priority_tensor.numel() * priority_tensor.element_size(),
            ]
            self.total_tensor_size = sum(self.tensor_sizes)
            self.tensor_name_to_index = {}
            self.loader_handle = None

        def _determine_buffer_size(self, requested):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def tensors(self):
            tensors = {
                "model.weight": regular_tensor,
                "vision_tower.weight": priority_tensor,
            }
            for name, _ in self.ordered_tensor_metadatas:
                yield name, tensors[name]

    observed_files = []

    def fake_instant_open(files, **kwargs):
        observed_files.extend(files)
        return FakeReader()

    def no_world_group():
        raise AssertionError

    monkeypatch.delenv("INSTANTTENSOR_BUFFER_SIZE", raising=False)
    monkeypatch.setenv("INSTANTTENSOR_COPY", "1")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules,
        "instanttensor",
        SimpleNamespace(safe_open=fake_instant_open),
    )

    loaded = list(
        instanttensor_weights_iterator(
            [str(regular_shard), str(priority_shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files=(
                {
                    "model.weight": str(regular_shard.resolve()),
                    "vision_tower.weight": str(priority_shard.resolve()),
                }
                if use_index
                else None
            ),
            priority_weight_name_prefixes=["vision_tower"],
        )
    )

    assert [name for name, _ in loaded] == ["vision_tower.weight", "model.weight"]
    assert torch.equal(loaded[0][1], priority_tensor)
    assert observed_files == [str(regular_shard), str(priority_shard)]


def test_instanttensor_priority_only_checkpoint_skips_gpu_staging(
    tmp_path, monkeypatch
):
    shard = tmp_path / "model.safetensors"
    priority_tensor = torch.tensor([1.0, 2.0])
    save_file({"vision_tower.weight": priority_tensor}, shard)
    instant_open = _FakeInstantOpen(shard, {"vision_tower.weight": priority_tensor})

    def no_world_group():
        raise AssertionError

    monkeypatch.delenv("INSTANTTENSOR_BUFFER_SIZE", raising=False)
    monkeypatch.setenv("INSTANTTENSOR_COPY", "1")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules,
        "instanttensor",
        SimpleNamespace(safe_open=lambda *args, **kwargs: instant_open),
    )

    loaded = list(
        instanttensor_weights_iterator(
            [str(shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files={"vision_tower.weight": str(shard.resolve())},
            priority_weight_name_prefixes=["vision_tower"],
        )
    )

    assert [name for name, _ in loaded] == ["vision_tower.weight"]
    assert torch.equal(loaded[0][1], priority_tensor)
    assert instant_open.enter_count == 0
    assert instant_open.buffer_size_requests == []


def test_instanttensor_ignores_source_without_priority_tensors(tmp_path, monkeypatch):
    shard = tmp_path / "draft.safetensors"
    draft_tensor = torch.tensor([1.0, 2.0])
    save_file({"draft.weight": draft_tensor}, shard)
    metadata = _safetensors_tensor_metadata(shard)["draft.weight"]

    class FakeReader:
        def __init__(self):
            self.filename = [str(shard)]
            self.ordered_tensor_metadatas = [("draft.weight", metadata)]
            self.tensor_offsets = [
                (0, metadata["data_offsets"][0]),
                (0, metadata["data_offsets"][1]),
            ]
            self.tensor_sizes = [draft_tensor.numel() * draft_tensor.element_size()]
            self.total_tensor_size = self.tensor_sizes[0]
            self.tensor_name_to_index = {}
            self.loader_handle = None

        def _determine_buffer_size(self, requested):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def tensors(self):
            yield "draft.weight", draft_tensor

    def no_world_group():
        raise AssertionError

    monkeypatch.delenv("INSTANTTENSOR_BUFFER_SIZE", raising=False)
    monkeypatch.setenv("INSTANTTENSOR_COPY", "1")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules,
        "instanttensor",
        SimpleNamespace(safe_open=lambda files, **kwargs: FakeReader()),
    )

    loaded = list(
        instanttensor_weights_iterator(
            [str(shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files=None,
            priority_weight_name_prefixes=["vision_tower"],
        )
    )

    assert [name for name, _ in loaded] == ["draft.weight"]
    assert torch.equal(loaded[0][1], draft_tensor)


def test_instanttensor_uses_cpu_safetensors_for_small_unindexed_source(
    tmp_path, monkeypatch
):
    shard = tmp_path / "draft.safetensors"
    draft_tensor = torch.tensor([1.0, 2.0])
    save_file({"draft.weight": draft_tensor}, shard)

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("InstantTensor must not open a small draft checkpoint")

    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setitem(
        sys.modules,
        "instanttensor",
        SimpleNamespace(safe_open=fail_if_opened),
    )

    loaded = list(
        instanttensor_weights_iterator(
            [str(shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files=None,
            priority_weight_name_prefixes=["vision_tower"],
            small_checkpoint_max_bytes=shard.stat().st_size,
        )
    )

    assert [name for name, _ in loaded] == ["draft.weight"]
    assert torch.equal(loaded[0][1], draft_tensor)


@pytest.mark.parametrize(
    ("buffer_size", "expected_gpu_opens", "expected_buffer_requests"),
    [(16, 1, [None]), (1, 0, [])],
)
def test_instanttensor_emits_cpu_fallbacks_in_checkpoint_order(
    tmp_path,
    monkeypatch,
    buffer_size,
    expected_gpu_opens,
    expected_buffer_requests,
):
    shard = tmp_path / "model.safetensors"
    source = {
        "a_small": torch.arange(2, dtype=torch.float32),
        "b_large": torch.arange(8, dtype=torch.float32),
        "c_small": torch.arange(2, dtype=torch.float32),
    }
    save_file(source, shard)
    instant_open = _FakeInstantOpen(shard, source)

    def no_world_group():
        raise AssertionError

    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(buffer_size))
    monkeypatch.setenv("INSTANTTENSOR_COPY", "1")
    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules,
        "instanttensor",
        SimpleNamespace(safe_open=lambda *args, **kwargs: instant_open),
    )

    loaded = list(instanttensor_weights_iterator([str(shard)], False))

    assert [name for name, _ in loaded] == instant_open.original_names
    assert instant_open.enter_count == expected_gpu_opens
    assert instant_open.buffer_size_requests == expected_buffer_requests


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_loads_tensor_larger_than_staging_buffer(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    small = torch.arange(256 * 1024, dtype=torch.float32)
    large = torch.arange(9 * 256 * 1024, dtype=torch.float32)
    save_file({"model.small": small, "model.large": large}, shard)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(8 * 1024 * 1024))
    monkeypatch.setenv("INSTANTTENSOR_COPY", "1")

    loaded = list(instanttensor_weights_iterator([str(shard)], use_tqdm_on_load=False))
    devices = {name: tensor.device.type for name, tensor in loaded}
    weights = {name: tensor.cpu() for name, tensor in loaded}

    assert devices == {"model.small": "cuda", "model.large": "cpu"}
    torch.testing.assert_close(weights["model.small"], small)
    torch.testing.assert_close(weights["model.large"], large)


def test_deferred_loader_preserves_destination_and_cpu_tensors():
    def loader(param, loaded_weight):
        pass

    destination = torch.empty(4)
    source = torch.arange(4)
    bound_args = inspect.signature(loader).bind(destination, source)

    _own_deferred_accelerator_tensors(bound_args)

    assert bound_args.arguments["param"] is destination
    assert bound_args.arguments["loaded_weight"] is source


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Borrowed accelerator storage requires CUDA",
)
def test_deferred_loader_owns_accelerator_tensor():
    def loader(param, loaded_weight):
        pass

    destination = torch.empty(8, device="cuda")
    source = torch.arange(8, device="cuda")
    bound_args = inspect.signature(loader).bind(destination, source)

    _own_deferred_accelerator_tensors(bound_args)

    owned = bound_args.arguments["loaded_weight"]
    assert bound_args.arguments["param"] is destination
    assert owned.data_ptr() != source.data_ptr()
    assert torch.equal(owned, source)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_deferred_tensors_survive_ring_reuse(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    source = {
        f"model.weight_{index}": torch.full(
            (2 * 1024 * 1024,), index, dtype=torch.bfloat16
        )
        for index in range(5)
    }
    save_file(source, shard)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(8 * 1024 * 1024))
    monkeypatch.setenv("INSTANTTENSOR_COPY", "0")

    def deferred_loader(param, loaded_weight):
        raise AssertionError("Deferred arguments must not execute while queued")

    signature = inspect.signature(deferred_loader)
    retained = {}
    for name, tensor in instanttensor_weights_iterator(
        [str(shard)], use_tqdm_on_load=False
    ):
        assert getattr(tensor, "_vllm_instanttensor_borrowed", False)
        bound_args = signature.bind(None, tensor)
        _own_deferred_accelerator_tensors(bound_args)
        retained[name] = bound_args.arguments["loaded_weight"]
    torch.accelerator.synchronize()

    for name, expected in source.items():
        assert torch.equal(retained[name].cpu(), expected)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_instanttensor_model_loader()
