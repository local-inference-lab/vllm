# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for host-shared Qwen3.8-Flash-Next PLE tables."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen3_8_flash_next import ple_layer as ple_layer_module
from vllm.models.qwen3_8_flash_next import ple_shared_table as shared
from vllm.models.qwen3_8_flash_next.ple_shared_table import (
    SharedTableConfig,
    SharedTableGeometry,
    SharedTableLock,
    SharedTableLockTimeout,
    SharedTableMismatch,
    SharedTableMissing,
    check_manifest,
    list_tables,
    open_shared_table,
    prune_tables,
    read_manifest,
)

CPU = torch.device("cpu")


def _layout(quant_mode: str = "nvfp4_group16", **overrides) -> SimpleNamespace:
    if quant_mode == "nvfp4_group16":
        fields = dict(
            weight_shape=(8, 4),
            weight_dtype=torch.uint8,
            weight_scale_shape=(8, 1),
            weight_scale_dtype=torch.float8_e4m3fn,
            weight_scale_2_shape=(1,),
            weight_scale_2_dtype=torch.float32,
        )
    elif quant_mode == "fp8_e4m3_per_tensor":
        fields = dict(
            weight_shape=(8, 8),
            weight_dtype=torch.float8_e4m3fn,
            weight_scale_shape=(1,),
            weight_scale_dtype=torch.bfloat16,
            weight_scale_2_shape=None,
            weight_scale_2_dtype=None,
        )
    else:
        fields = dict(
            weight_shape=(8, 8),
            weight_dtype=torch.bfloat16,
            weight_scale_shape=None,
            weight_scale_dtype=None,
            weight_scale_2_shape=None,
            weight_scale_2_dtype=None,
        )
    fields.update(
        caps=SimpleNamespace(
            quant_mode=quant_mode,
            tp_size=1,
            tp_rank=0,
            table_memory="mapped_host",
            device=CPU,
        ),
        padded_vocab_size=8,
        shard_start=0,
        shard_end=8,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _geometry(quant_mode: str = "nvfp4_group16", **kwargs) -> SharedTableGeometry:
    return SharedTableGeometry.from_layout(
        _layout(quant_mode),
        model_path=kwargs.pop("model_path", "org/model"),
        revision=kwargs.pop("revision", "abc123"),
        embedding_dim=kwargs.pop("embedding_dim", 64),
        dense_layer_ordinal=kwargs.pop("dense_layer_ordinal", 0),
    )


def _config(tmp_path: Path, **kwargs) -> SharedTableConfig:
    kwargs.setdefault("directory", str(tmp_path))
    kwargs.setdefault("model_path", "org/model")
    kwargs.setdefault("revision", "abc123")
    return SharedTableConfig(**kwargs)


def _fill(table: shared.SharedTable) -> None:
    for attribute in ("weight", "weight_scale"):
        view = table.host_view(attribute)
        if view is not None:
            raw = view.view(torch.uint8)
            raw.copy_(
                torch.arange(raw.numel(), dtype=torch.int64)
                .remainder(251)
                .to(torch.uint8)
                .reshape(raw.shape)
            )


def test_key_depends_on_every_geometry_field() -> None:
    base = _geometry()
    assert base.key == _geometry().key
    assert len(base.key) == 16
    seen = {base.key}
    for field, value in [
        ("revision", "other"),
        ("model_path", "org/other"),
        ("tp_rank", 1),
        ("tp_size", 2),
        ("dense_layer_ordinal", 1),
        ("shard_end", 4),
        ("embedding_dim", 32),
        ("weight_dtype", "int8"),
        ("layout_version", 2),
    ]:
        data = base.as_dict()
        data[field] = value
        key = SharedTableGeometry.from_dict(data).key
        assert key not in seen, field
        seen.add(key)


@pytest.mark.parametrize(
    ("quant_mode", "names"),
    [
        ("nvfp4_group16", ["weight.bin", "weight_scale.bin"]),
        ("fp8_e4m3_per_tensor", ["weight.bin"]),
        ("bf16", ["weight.bin"]),
    ],
)
def test_only_mapped_host_tensors_become_files(quant_mode, names) -> None:
    geometry = _geometry(quant_mode)
    assert [spec.name for spec in geometry.files] == names
    assert geometry.nbytes == sum(spec.nbytes for spec in geometry.files)


def test_populate_then_attach_round_trip(tmp_path: Path) -> None:
    geometry = _geometry()
    populator = open_shared_table(geometry, _config(tmp_path), device=CPU)
    assert populator.populating and not populator.attached
    table_dir = tmp_path / geometry.key
    assert (table_dir / "weight.bin").stat().st_size == geometry.files[0].nbytes
    assert not (table_dir / "READY").exists()
    assert list_tables(tmp_path)[0].state == "populating"

    _fill(populator)
    populator.publish()
    assert (table_dir / "READY").exists()
    manifest = read_manifest(table_dir)
    assert manifest is not None
    assert manifest["key"] == geometry.key
    assert manifest["geometry"] == geometry.as_dict()
    assert manifest["files"] == {"weight.bin": 32, "weight_scale.bin": 8}
    assert manifest["creator"]["pid"]
    assert list_tables(tmp_path)[0].state == "ready"

    follower = open_shared_table(geometry, _config(tmp_path), device=CPU)
    assert follower.attached and not follower.populating
    for attribute in ("weight", "weight_scale"):
        torch.testing.assert_close(
            follower.host_view(attribute).view(torch.uint8),
            populator.host_view(attribute).view(torch.uint8),
        )
        assert follower.device_view(attribute) is follower.host_view(attribute)
    assert follower.nbytes == populator.nbytes == 40
    follower.close()
    populator.close()
    assert (table_dir / "READY").exists()


def test_attach_role_fails_fast_without_a_table(tmp_path: Path) -> None:
    with pytest.raises(SharedTableMissing, match="attach forbids populating"):
        open_shared_table(_geometry(), _config(tmp_path, role="attach"), device=CPU)
    assert list_tables(tmp_path) == []


def test_manifest_mismatch_names_the_field(tmp_path: Path) -> None:
    geometry = _geometry()
    table = open_shared_table(geometry, _config(tmp_path), device=CPU)
    _fill(table)
    table.publish()
    table.close()
    manifest_path = tmp_path / geometry.key / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry"]["shard_end"] = 4
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SharedTableMismatch, match="shard_end expected 8") as info:
        open_shared_table(geometry, _config(tmp_path, role="attach"), device=CPU)
    assert info.value.field == "shard_end"
    # auto never silently repopulates over a READY table that disagrees.
    with pytest.raises(SharedTableMismatch):
        open_shared_table(geometry, _config(tmp_path), device=CPU)


def test_manifest_check_covers_file_sizes_and_format(tmp_path: Path) -> None:
    geometry = _geometry()
    table = open_shared_table(geometry, _config(tmp_path), device=CPU)
    _fill(table)
    table.publish()
    table.close()
    table_dir = tmp_path / geometry.key
    manifest = json.loads((table_dir / "manifest.json").read_text())
    check_manifest(manifest, geometry, table_dir)

    with pytest.raises(SharedTableMismatch, match="format"):
        check_manifest({**manifest, "format": "other"}, geometry, table_dir)
    with open(table_dir / "weight.bin", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(SharedTableMismatch, match=r"files\.weight\.bin expected 32"):
        check_manifest(manifest, geometry, table_dir)


def test_stale_directory_without_ready_is_replaced(tmp_path: Path) -> None:
    geometry = _geometry()
    stale = tmp_path / geometry.key
    stale.mkdir()
    (stale / "weight.bin").write_bytes(b"garbage")
    (stale / "leftover").write_text("x")

    table = open_shared_table(geometry, _config(tmp_path), device=CPU)
    assert table.populating
    assert not (stale / "leftover").exists()
    assert (stale / "weight.bin").stat().st_size == 32
    table.close()
    # Closing an unpublished populator removes its partial table.
    assert not stale.exists()
    assert list_tables(tmp_path) == []


def test_populate_role_always_rewrites(tmp_path: Path) -> None:
    geometry = _geometry()
    first = open_shared_table(geometry, _config(tmp_path), device=CPU)
    _fill(first)
    first.publish()
    first.close()
    created = read_manifest(tmp_path / geometry.key)["created_at"]

    second = open_shared_table(geometry, _config(tmp_path, role="populate"), device=CPU)
    assert second.populating
    assert not (tmp_path / geometry.key / "READY").exists()
    _fill(second)
    second.publish()
    assert read_manifest(tmp_path / geometry.key)["created_at"] >= created
    second.close()


def test_lock_timeout_is_reported(tmp_path: Path) -> None:
    geometry = _geometry()
    holder = SharedTableLock(shared.lock_path(tmp_path, geometry.key), 1)
    holder.acquire()
    try:
        with pytest.raises(SharedTableLockTimeout, match="timed out after 0s"):
            open_shared_table(geometry, _config(tmp_path, lock_timeout_s=0), device=CPU)
    finally:
        holder.release()
    table = open_shared_table(geometry, _config(tmp_path, lock_timeout_s=0), device=CPU)
    assert table.populating
    table.close()


def test_populator_holds_the_lock_until_publish(tmp_path: Path) -> None:
    geometry = _geometry()
    table = open_shared_table(geometry, _config(tmp_path), device=CPU)
    probe = SharedTableLock(shared.lock_path(tmp_path, geometry.key), 0)
    with pytest.raises(SharedTableLockTimeout):
        probe.acquire()
    _fill(table)
    table.publish()
    probe.acquire()
    probe.release()
    table.close()


def test_config_rejects_unknown_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ple_shared_table_role"):
        _config(tmp_path, role="maybe")


def test_cli_list_and_prune(tmp_path: Path, capsys) -> None:
    keep = _geometry()
    stale = _geometry(revision="old")
    busy = _geometry(revision="busy")
    for geometry in (keep, stale):
        table = open_shared_table(geometry, _config(tmp_path), device=CPU)
        _fill(table)
        table.publish()
        table.close()
    populating = open_shared_table(busy, _config(tmp_path), device=CPU)

    assert shared.main(["--dir", str(tmp_path), "list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    states = {line.split()[0]: line.split()[1] for line in lines}
    assert states == {keep.key: "ready", stale.key: "ready", busy.key: "populating"}
    assert any("revision=old" in line for line in lines)

    assert shared.main(["--dir", str(tmp_path), "list", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert {entry["key"] for entry in entries} == {keep.key, stale.key, busy.key}

    assert (
        shared.main(["--dir", str(tmp_path), "prune", "--keep", keep.key, "--dry-run"])
        == 0
    )
    assert capsys.readouterr().out.strip() == f"would remove {stale.key}"
    assert (tmp_path / stale.key).exists()

    assert prune_tables(tmp_path, keep=[keep.key]) == [stale.key]
    assert not (tmp_path / stale.key).exists()
    assert not shared.lock_path(tmp_path, stale.key).exists()
    assert (tmp_path / keep.key / "READY").exists()
    # A table being populated is never pruned.
    assert (tmp_path / busy.key).exists()
    populating.close()
    assert shared.main(["--dir", str(tmp_path / "missing"), "list"]) == 0


@dataclass(kw_only=True)
class _FakeTableStorage:
    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    weight_load_view: torch.Tensor
    weight_scale_load_view: torch.Tensor | None
    weight_scale_2_load_view: torch.Tensor | None
    mapped_host_nbytes: int
    _mapped_allocations: tuple


@pytest.fixture
def fake_b12x(monkeypatch):
    api = SimpleNamespace(TableStorage=_FakeTableStorage)
    monkeypatch.setattr(ple_layer_module, "_b12x_module", lambda name: api)
    return api


def test_storage_populates_then_attaches(tmp_path: Path, fake_b12x) -> None:
    layout = _layout()
    geometry = _geometry()
    config = _config(tmp_path)

    populator = ple_layer_module._NGramEmbeddingStorage(
        layout, 2, shared_config=config, shared_geometry=geometry
    )
    assert not populator.shared_attached
    assert populator.shared_table is not None and populator.shared_table.populating
    assert isinstance(populator.weight, nn.Parameter)
    assert populator.weight.shape == (8, 4) and populator.weight.dtype == torch.uint8
    assert populator.weight_scale.shape == (8, 1)
    assert populator.weight_scale_2.shape == (1,)
    assert populator.mapped_host_nbytes == 40
    assert populator.weight_load_view.data_ptr() == populator.weight.data_ptr()
    populator.weight_load_view.fill_(7)
    populator.weight_scale_load_view.view(torch.uint8).fill_(3)
    populator.publish_shared_table()
    assert not populator.shared_table.populating

    follower = ple_layer_module._NGramEmbeddingStorage(
        layout, 2, shared_config=config, shared_geometry=geometry
    )
    assert follower.shared_attached
    assert torch.equal(follower.weight, torch.full((8, 4), 7, dtype=torch.uint8))
    assert torch.equal(
        follower.weight_scale.view(torch.uint8),
        torch.full((8, 1), 3, dtype=torch.uint8),
    )
    # Scale 2 stays a private device tensor loaded from the checkpoint.
    assert follower.weight_scale_2.data_ptr() != populator.weight_scale_2.data_ptr()
    follower.publish_shared_table()  # no-op for attachers
    populator._table_storage._mapped_allocations[0].close()
    follower._table_storage._mapped_allocations[0].close()


def test_storage_rejects_non_mapped_host_plans(tmp_path: Path, fake_b12x) -> None:
    layout = _layout()
    layout.caps.table_memory = "device"
    with pytest.raises(ValueError, match="mapped_host"):
        ple_layer_module._NGramEmbeddingStorage(
            layout, 2, shared_config=_config(tmp_path), shared_geometry=_geometry()
        )


def test_shared_policy_maps_to_mapped_host(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", "shared")
    assert ple_layer_module._resolve_ple_table_policy(None) == "shared"
    assert ple_layer_module._resolve_ple_table_memory(None) == "mapped_host"
    assert (
        ple_layer_module._resolve_ple_table_memory({"ple_table_memory": "ram"})
        == "mapped_host"
    )
    with pytest.raises(ValueError, match="'shared'"):
        ple_layer_module._resolve_ple_table_memory({"ple_table_memory": "shm"})


def _vllm_config(additional_config) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=SimpleNamespace(model="org/model", revision="rev1"),
    )


def test_shared_table_settings_from_env_and_additional_config(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", "ram")
    assert ple_layer_module._resolve_ple_shared_table(_vllm_config(None)) is None

    monkeypatch.setenv("VLLM_PLE_TABLE_MEMORY", "shared")
    monkeypatch.delenv("VLLM_PLE_SHARED_TABLE_DIR", raising=False)
    monkeypatch.delenv("VLLM_PLE_SHARED_TABLE_ROLE", raising=False)
    monkeypatch.delenv("VLLM_PLE_SHARED_TABLE_LOCK_TIMEOUT_S", raising=False)
    config = ple_layer_module._resolve_ple_shared_table(_vllm_config({}))
    assert config == SharedTableConfig(
        directory="/dev/shm/vllm-ple",
        role="auto",
        lock_timeout_s=3600.0,
        model_path="org/model",
        revision="rev1",
    )

    monkeypatch.setenv("VLLM_PLE_SHARED_TABLE_DIR", "/mnt/tables")
    monkeypatch.setenv("VLLM_PLE_SHARED_TABLE_ROLE", "attach")
    monkeypatch.setenv("VLLM_PLE_SHARED_TABLE_LOCK_TIMEOUT_S", "5")
    config = ple_layer_module._resolve_ple_shared_table(_vllm_config(None))
    assert (config.directory, config.role, config.lock_timeout_s) == (
        "/mnt/tables",
        "attach",
        5.0,
    )

    config = ple_layer_module._resolve_ple_shared_table(
        _vllm_config(
            {
                "ple_table_memory": "shared",
                "ple_shared_table_dir": "/tables",
                "ple_shared_table_role": "populate",
                "ple_shared_table_lock_timeout_s": 1,
            }
        )
    )
    assert (config.directory, config.role, config.lock_timeout_s) == (
        "/tables",
        "populate",
        1.0,
    )
    with pytest.raises(ValueError, match="ple_shared_table_role"):
        ple_layer_module._resolve_ple_shared_table(
            _vllm_config({"ple_table_memory": "shared", "ple_shared_table_role": "x"})
        )


class _AttachedAudit(ple_layer_module.Qwen3_8FlashNextNGramEmbedding):
    """A minimal NVFP4 embedding whose table is attached to a shared one."""

    def __init__(self, attached: bool) -> None:
        nn.Module.__init__(self)
        self._quant_mode = "nvfp4_group16"
        self._embedding_load_ranges = set()
        self._scale_load_ranges = set()
        self._weight_scale_loaded = False
        self._weight_scale_2_loaded = False
        self._embedding_validated = False
        self.split_ngram_parts = 4
        # The layer reads its table geometry from ``_table_layout`` on the
        # staged-preparation branch and from ``_plan`` before it.
        self._table_layout = self._plan = SimpleNamespace(
            padded_vocab_size=8,
            shard_start=0,
            shard_end=8,
            weight_shape=(8, 8),
            weight_dtype=torch.uint8,
            weight_scale_shape=(8, 1),
            weight_scale_dtype=torch.float8_e4m3fn,
            weight_scale_2_shape=(1,),
            weight_scale_2_dtype=torch.float32,
        )
        self.register_buffer("layer_multipliers", torch.tensor([11, 13]))
        self.register_buffer("ngram_heads_offsets", torch.tensor([0, 4]))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor([4, 4]))
        embedding = nn.Module()
        embedding.shared_attached = attached
        embedding.published = 0
        embedding.publish_shared_table = lambda: setattr(
            embedding, "published", embedding.published + 1
        )
        for name, shape, dtype in (
            ("weight", (8, 8), torch.uint8),
            ("weight_scale", (8, 1), torch.float8_e4m3fn),
            ("weight_scale_2", (1,), torch.float32),
        ):
            embedding.register_parameter(
                name, nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            )
        self.add_module("ngram_embedding", embedding)
        if attached:
            self._embedding_load_ranges.add((0, 8))
            self._scale_load_ranges.add((0, 8))


def _shard_weights(meta: bool):
    device = "meta" if meta else "cpu"
    weights = []
    for shard in range(4):
        weights.append(
            (
                f"ngram_embedding.shard_{shard}.weight",
                torch.full((2, 8), shard + 1, dtype=torch.uint8, device=device),
            )
        )
        weights.append(
            (
                f"ngram_embedding.shard_{shard}.weight_scale",
                torch.ones((2, 1), dtype=torch.float8_e4m3fn, device=device),
            )
        )
    weights.append(("ngram_embedding.weight_scale_2", torch.tensor(0.5)))
    weights.append(("layer_multipliers", torch.tensor([11, 13])))
    return weights


def test_attached_loader_skips_shard_copies_but_checks_geometry() -> None:
    model = _AttachedAudit(attached=True)
    loaded = model.load_weights(_shard_weights(meta=True))
    assert loaded == {
        "ngram_embedding.weight",
        "ngram_embedding.weight_scale",
        "ngram_embedding.weight_scale_2",
        "layer_multipliers",
    }
    assert torch.equal(
        model.ngram_embedding.weight, torch.zeros(8, 8, dtype=torch.uint8)
    )
    assert model.ngram_embedding.weight_scale_2.item() == 0.5
    model._validate_embedding_loaded()
    assert model.ngram_embedding.published == 1
    model._validate_embedding_loaded()
    assert model.ngram_embedding.published == 1

    with pytest.raises(ValueError, match="shape mismatch for PLE shard 1 weight"):
        _AttachedAudit(attached=True).load_weights(
            [("ngram_embedding.shard_1.weight", torch.zeros(3, 8, dtype=torch.uint8))]
        )
    with pytest.raises(ValueError, match="does not match planned PLE geometry"):
        _AttachedAudit(attached=True).load_weights(
            [("layer_multipliers", torch.tensor([1, 2]))]
        )


def test_populating_loader_copies_shards_and_publishes_once_complete() -> None:
    model = _AttachedAudit(attached=False)
    loaded = model.load_weights(_shard_weights(meta=False))
    assert "ngram_embedding.weight" in loaded
    expected = torch.cat(
        [torch.full((2, 8), shard + 1, dtype=torch.uint8) for shard in range(4)]
    )
    assert torch.equal(model.ngram_embedding.weight, expected)
    assert model.ngram_embedding.published == 0
    model._validate_embedding_loaded()
    assert model.ngram_embedding.published == 1

    partial = _AttachedAudit(attached=False)
    partial.load_weights(_shard_weights(meta=False)[:4])
    with pytest.raises(ValueError, match="do not cover the local table"):
        partial._validate_embedding_loaded()
    assert partial.ngram_embedding.published == 0


def test_shared_tables_attached_requires_every_embedding() -> None:
    root = nn.Module()
    assert not ple_layer_module.shared_ple_tables_attached(root)
    root.add_module("a", _AttachedAudit(attached=True))
    assert ple_layer_module.shared_ple_tables_attached(root)
    root.add_module("b", _AttachedAudit(attached=False))
    assert not ple_layer_module.shared_ple_tables_attached(root)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_registered_mapping_exposes_the_device_alias() -> None:
    import tempfile

    from cuda.bindings import runtime as cudart

    # The driver only pins tmpfs-backed mappings; pytest's tmp_path may not be.
    if not Path("/dev/shm").is_dir():
        pytest.skip("needs a tmpfs at /dev/shm")
    tmp_path = Path(tempfile.mkdtemp(prefix="vllm-ple-test-", dir="/dev/shm"))
    geometry = _geometry()
    device = torch.device("cuda", torch.cuda.current_device())
    populator = open_shared_table(geometry, _config(tmp_path), device=device)
    _fill(populator)
    populator.publish()
    follower = open_shared_table(geometry, _config(tmp_path), device=device)
    for table in (populator, follower):
        for attribute in ("weight", "weight_scale"):
            host = table.host_view(attribute)
            dev = table.device_view(attribute)
            assert dev.device == device and host.device.type == "cpu"
            error, attributes = cudart.cudaPointerGetAttributes(dev.data_ptr())
            assert error == cudart.cudaError_t.cudaSuccess
            assert attributes.type == cudart.cudaMemoryType.cudaMemoryTypeHost
            assert int(attributes.devicePointer) == dev.data_ptr()
            torch.testing.assert_close(
                dev.cpu().view(torch.uint8), host.view(torch.uint8)
            )
    follower.close()
    populator.close()
    shutil.rmtree(tmp_path)
