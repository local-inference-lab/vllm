# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Engram with readonly accepted history and global ceil-row TP shards."""

from functools import lru_cache
from weakref import WeakValueDictionary

import torch
from b12x.norm import hyperconnection
from b12x.sequence import engram as native
from b12x.sequence._shared.disk_table import MappedHostAllocation
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import get_file_tensor_source
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource

logger = init_logger(__name__)

DEAD_ID = -1
_STATES = WeakValueDictionary()
_TABLES = WeakValueDictionary()


@lru_cache(maxsize=2)
def _token_map(path, revision, trust_remote_code):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=trust_remote_code
    )
    return native.build_compressed_token_map(tokenizer)


class EngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.geometry = native.build_geometry(
            layer_ids=self.layer_ids,
            base_table_size=config.engram_vocab_size,
            compressed_vocab_size=config.engram_compressed_vocab_size,
        )
        if (self.max_ngram_size, self.n_heads, self.head_dim) != (4, 8, 256):
            raise ValueError(
                "V4.1 Engram requires four-gram/eight-head/dim256 geometry"
            )
        if self.geometry.num_embeddings != self.num_embeddings:
            raise ValueError(
                "Engram checkpoint table rows do not match global hash geometry"
            )
        vc = get_current_vllm_config()
        mc = vc.model_config
        engram_config = vc.engram_config
        self.table_memory = (
            engram_config.table_memory if engram_config is not None else "device"
        )
        self.disk_resident_scales = (
            engram_config.disk_resident_scales if engram_config is not None else False
        )
        self.disk_prefetch_max_tokens = (
            engram_config.disk_prefetch_max_tokens if engram_config is not None else 0
        )
        self.projection_tp = (
            engram_config.projection_tp if engram_config is not None else False
        )
        if (
            self.table_memory == "disk"
            and vc.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Disk Engram requires pipeline_parallel_size=1")
        token_map, compressed_size = _token_map(
            mc.tokenizer, mc.revision, mc.trust_remote_code
        )
        if compressed_size != config.engram_compressed_vocab_size:
            raise ValueError("Engram tokenizer compressed vocabulary mismatch")
        device = torch.empty(0).device
        self.plans = tuple(
            native.plan(
                native.Caps(
                    device=device,
                    max_tokens=vc.scheduler_config.max_num_batched_tokens,
                    max_seqs=vc.scheduler_config.max_num_seqs,
                    max_requests=vc.scheduler_config.max_num_seqs,
                    vocab_size=config.vocab_size,
                    layer_id=layer,
                    tp_size=get_tensor_model_parallel_world_size(),
                    tp_rank=get_tensor_model_parallel_rank(),
                ),
                token_map=token_map,
                geometry=self.geometry,
            )
            for layer in self.layer_ids
        )

    @classmethod
    def from_config(cls, config):
        return cls(config) if getattr(config, "engram_layer_ids", None) else None


@triton.jit(do_not_specialize=["tokens", "seqs"])
def _prepare_metadata(
    ids,
    mask,
    starts,
    history,
    token_map,
    ids_out,
    mask_out,
    starts_out,
    history_out,
    slots,
    num_seqs_out,
    num_tokens_out,
    tokens,
    seqs,
    max_tokens: tl.constexpr,
    max_seqs: tl.constexpr,
    history_stride: tl.constexpr,
    vocab: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Captured input tensors include padding; starts carries the live count.
    live_tokens = tl.load(starts + seqs)
    value = tl.load(ids + i, (i < tokens) & (i < live_tokens), other=0)
    keep = tl.load(mask + i, (i < tokens) & (i < live_tokens), other=False)
    tl.store(ids_out + i, value, i < max_tokens)
    tl.store(mask_out + i, keep, i < max_tokens)
    start = tl.load(starts + i, i <= seqs, other=live_tokens)
    tl.store(starts_out + i, start, i <= max_seqs)
    tl.store(slots + i, i.to(tl.int32), i < max_seqs)
    r, col = i // 3, i % 3
    raw = tl.load(history + r * history_stride + col, r < seqs, other=-1)
    valid = (r < seqs) & (raw >= 0) & (raw < vocab) & (raw != 129264)
    compressed = tl.load(token_map + raw, valid, other=-1)
    tl.store(history_out + i, compressed, i < max_seqs * 3)
    if tl.program_id(0) == 0:
        tl.store(num_seqs_out, seqs)
        tl.store(num_tokens_out, live_tokens)


@torch.library.custom_op("vllm::dsv41_engram_hash", mutates_args=("out",))
def _hash(
    ids: torch.Tensor,
    mask: torch.Tensor,
    starts: torch.Tensor,
    history: torch.Tensor,
    out: torch.Tensor,
    key: int,
) -> None:
    _STATES[key].run_native(ids, mask, starts, history, out)


@_hash.register_fake
def _hash_fake(ids, mask, starts, history, out, key):
    return None


class NgramHashState(nn.Module):
    def __init__(self, vllm_config, layout, swa_cache_module):
        super().__init__()
        self.layout = layout
        self.lookback_depth = 3
        self.use_slot_cache = False
        self.key = id(self)
        _STATES[self.key] = self
        c = layout.plans[0].caps
        for name, shape, dtype in (
            ("ids", (c.max_tokens,), torch.int64),
            ("mask", (c.max_tokens,), torch.bool),
            ("starts", (c.max_seqs + 1,), torch.int32),
            ("history", (c.max_requests, 3), torch.int64),
            ("slots", (c.max_seqs,), torch.int32),
            ("num_seqs", (1,), torch.int32),
            ("num_tokens", (1,), torch.int32),
        ):
            self.register_buffer(
                name, torch.empty(shape, dtype=dtype, device=c.device), persistent=False
            )
        self.bindings = []
        for i, plan in enumerate(layout.plans):
            (spec,) = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            hashes = torch.empty((c.max_tokens, 24), dtype=torch.int64, device=c.device)
            self.register_buffer(f"scratch_{i}", scratch, persistent=False)
            self.register_buffer(f"hashes_{i}", hashes, persistent=False)
            self.bindings.append(
                native.bind(
                    plan,
                    scratch=scratch,
                    token_ids=self.ids,
                    token_mask=self.mask,
                    query_start_loc=self.starts,
                    request_slots=self.slots,
                    committed_history=self.history,
                    num_seqs=self.num_seqs,
                    num_tokens=self.num_tokens,
                    hash_ids=hashes,
                )
            )

    def ensure_cache(self):
        return True  # Hash history belongs to the runner, never the KV slot pool.

    def run_native(self, ids, mask, starts, history, out):
        retain_cuda_graph_capture_resource(self)
        c = self.layout.plans[0].caps
        seqs = starts.numel() - 1
        if ids.numel() > c.max_tokens or seqs > c.max_seqs:
            raise ValueError("Engram live rows exceed preallocated capacity")
        if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < seqs:
            raise ValueError(
                "Engram requires chronological accepted history [requests,3]"
            )
        work = max(c.max_tokens, c.max_seqs * 3, c.max_seqs + 1)
        _prepare_metadata[(triton.cdiv(work, 256),)](
            ids,
            mask,
            starts,
            history,
            self.layout.plans[0].token_map,
            self.ids,
            self.mask,
            self.starts,
            self.history,
            self.slots,
            self.num_seqs,
            self.num_tokens,
            ids.numel(),
            seqs,
            c.max_tokens,
            c.max_seqs,
            history.stride(0),
            c.vocab_size,
            256,
        )
        for i, binding in enumerate(self.bindings):
            native.run(binding, token_count=ids.numel())
            out[:, i].copy_(binding.hash_ids[: ids.numel()])

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask=None,
        slot_mapping=None,
        block_table=None,
    ):
        out = torch.empty(
            (input_ids.numel(), len(self.bindings), 24),
            dtype=torch.int64,
            device=input_ids.device,
        )
        _hash(input_ids, ~dead_mask, query_start_loc, lookback_token_ids, out, self.key)
        return out


def _read_table_rows(destination, source, start):
    """Read only this shard, straight into its final mapped CPU allocation."""
    if destination.device.type != "cpu" or not destination.is_contiguous():
        raise ValueError("File-backed Engram loading requires contiguous CPU storage")
    row_bytes = destination.shape[1] * destination.element_size()
    data = memoryview(destination.view(torch.uint8).numpy()).cast("B")
    # Unbuffered readinto has no payload-sized Python/NumPy allocation, and a
    # bounded syscall size also handles tables larger than Linux's read limit.
    with open(source.path, "rb", buffering=0) as checkpoint:
        checkpoint.seek(source.offset + start * row_bytes)
        offset = 0
        while offset < len(data):
            count = checkpoint.readinto(data[offset : offset + (8 << 20)])
            if not count:
                raise OSError(
                    f"Short read loading Engram table from {source.path}: "
                    f"{offset} of {len(data)} local bytes"
                )
            offset += count


def _load_table(param, loaded_weight):
    source = get_file_tensor_source(loaded_weight)
    shape = source.shape if source is not None else loaded_weight.shape
    dtype = source.dtype if source is not None else loaded_weight.dtype
    if shape != (param.global_rows, *param.shape[1:]):
        raise ValueError("Engram source must be the unpadded global checkpoint table")
    dtypes = (
        (torch.uint8, torch.float8_e8m0fnu)
        if param.dtype == torch.uint8
        else (torch.float8_e4m3fn,)
    )
    if dtype not in dtypes:
        raise TypeError(f"Invalid Engram table dtype: {dtype}")
    destination = getattr(param, "load_view", param.data)
    start = param.shard_start
    count = max(0, min(param.shape[0], param.global_rows - start))
    if count < param.shape[0]:
        destination[count:].zero_()
    if count:
        if source is not None:
            _read_table_rows(destination[:count], source, start)
        else:
            destination[:count].view(torch.uint8).copy_(
                loaded_weight[start : start + count].view(torch.uint8)
            )


@torch.library.custom_op("vllm::dsv41_engram_lookup", mutates_args=("out",))
def _lookup(indices: torch.Tensor, out: torch.Tensor, key: int) -> None:
    _TABLES[key].lookup_native(indices, out)


@_lookup.register_fake
def _lookup_fake(indices, out, key):
    return None


class ParallelEngramEmbedding(nn.Module):
    def __init__(
        self, plan, table_memory="device", *, resident_scales=False, prefetch=False
    ):
        super().__init__()
        self.plan = plan
        self.key = id(self)
        _TABLES[self.key] = self
        self.tp_size = plan.caps.tp_size
        if table_memory not in ("device", "ram", "disk"):
            raise ValueError("Engram table_memory must be device, ram or disk")
        self.disk_table = None
        if table_memory == "disk":
            # Do not require the optional B12X API on unchanged/default boots.
            options = {}
            if resident_scales or prefetch:
                options = {"resident_scales": resident_scales, "prefetch": prefetch}
            self.disk_table = native.DiskTable(plan, **options)
        self._disk_pending_rows = None
        self._disk_binding = None
        self._disk_prepared_rows = 0
        self.mapped_host_nbytes = 0
        if table_memory == "ram":
            nbytes = (
                plan.weight_shape[0] * plan.weight_shape[1]
                + plan.scale_shape[0] * plan.scale_shape[1]
            )
            logger.info(
                "Engram layer %d TP rank %d: allocating %.2f GiB mapped-host "
                "RAM (packed E4M3 weights and E8M0 scales)",
                plan.caps.layer_id,
                plan.caps.tp_rank,
                nbytes / (1 << 30),
            )
            self._weight_allocation = None
            try:
                self._weight_allocation = MappedHostAllocation(
                    plan.weight_shape, torch.float8_e4m3fn, plan.caps.device
                )
                self._scale_allocation = MappedHostAllocation(
                    plan.scale_shape, torch.uint8, plan.caps.device
                )
            except Exception as exc:
                if self._weight_allocation is not None:
                    self._weight_allocation.close()
                raise RuntimeError(
                    f"Engram mapped-host RAM allocation failed for "
                    f"{nbytes / (1 << 30):.2f} GiB on {plan.caps.device}; "
                    "no disk or device fallback is permitted"
                ) from exc
            self.mapped_host_nbytes = nbytes
            self.weight_load_view = self._weight_allocation.host_view
            self.weight_scale_load_view = self._scale_allocation.host_view
            self.weight = nn.Parameter(
                self._weight_allocation.device_view, requires_grad=False
            )
            self.weight_scale_inv = nn.Parameter(
                self._scale_allocation.device_view, requires_grad=False
            )
            set_weight_attrs(self.weight, {"load_view": self.weight_load_view})
            set_weight_attrs(
                self.weight_scale_inv, {"load_view": self.weight_scale_load_view}
            )
        elif self.disk_table is None:
            self.weight = nn.Parameter(
                torch.empty(
                    plan.weight_shape,
                    dtype=torch.float8_e4m3fn,
                    device=plan.caps.device,
                ),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    plan.scale_shape, dtype=torch.uint8, device=plan.caps.device
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("weight_scale_inv", None)
        if self.disk_table is None:
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _load_table,
                        "global_rows": plan.table_rows,
                        "shard_start": plan.shard_start,
                    },
                )
        self.register_buffer(
            "hashes",
            torch.empty(
                (plan.caps.max_tokens, 24), dtype=torch.int64, device=plan.caps.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "num_tokens",
            torch.empty((1,), dtype=torch.int32, device=plan.caps.device),
            persistent=False,
        )

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if name not in ("weight", "weight_scale_inv"):
                raise ValueError(f"Unknown Engram table weight: {name}")
            if self.disk_table is None:
                _load_table(getattr(self, name), value)
            else:
                source = get_file_tensor_source(value)
                if source is None:
                    raise ValueError(
                        "Disk Engram requires a file-backed checkpoint table"
                    )
                scale = name == "weight_scale_inv"
                width = self.plan.scale_shape[1] if scale else self.plan.weight_shape[1]
                if source.shape != (self.plan.table_rows, width):
                    raise ValueError("Engram source must be the unpadded global table")
                dtypes = (
                    (torch.uint8, torch.float8_e8m0fnu)
                    if scale
                    else (torch.float8_e4m3fn,)
                )
                if source.dtype not in dtypes:
                    raise TypeError(f"Invalid Engram {name} dtype: {source.dtype}")
                self.disk_table.add_shard(0, source.path, source.offset, scale=scale)
            loaded.add(name)
        return loaded

    def prepare_disk(self, indices, out, num_tokens, *, prefetch=False):
        if self.disk_table is None:
            raise RuntimeError("Engram table is not disk-backed")
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        if self._disk_pending_rows is not None:
            raise RuntimeError("Disk Engram has an unconsumed preparation")
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.copy_(num_tokens)
        if self._disk_binding is None or self._disk_binding.out is not out:
            self._disk_binding = native.bind_lookup(
                self.plan,
                weight=None,
                scales=None,
                hash_ids=self.hashes,
                num_tokens=self.num_tokens,
                out=out,
                disk_table=self.disk_table,
            )
            out.zero_()
            self._disk_prepared_rows = 0
        if indices.shape[0] < self._disk_prepared_rows:
            out[indices.shape[0] : self._disk_prepared_rows].zero_()
        if prefetch:
            self.disk_table.prefetch(self._disk_binding, token_count=indices.shape[0])
            self._disk_pending_rows = indices.shape[0]
        else:
            native.run_lookup(
                self._disk_binding, token_count=indices.shape[0], clear_tail=False
            )
            self._disk_prepared_rows = indices.shape[0]

    def finish_disk(self):
        if self._disk_pending_rows is None:
            return
        assert self.disk_table is not None
        count = self._disk_pending_rows
        try:
            native.run_lookup(self._disk_binding, token_count=count, clear_tail=False)
            self._disk_prepared_rows = count
        finally:
            if not self.disk_table.prefetch_pending:
                self._disk_pending_rows = None

    def abort_disk(self):
        if self._disk_pending_rows is None:
            return
        assert self.disk_table is not None
        try:
            self.disk_table.abort_prefetch()
        finally:
            if not self.disk_table.prefetch_pending:
                self._disk_pending_rows = None

    def lookup_native(self, indices, out):
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.fill_(indices.shape[0])
        binding = native.bind_lookup(
            self.plan,
            weight=self.weight,
            scales=self.weight_scale_inv,
            hash_ids=self.hashes,
            num_tokens=self.num_tokens,
            out=out,
        )
        retain_cuda_graph_capture_resource(binding)
        if self.mapped_host_nbytes:
            # CUDA tensor aliases do not own cudaHostAlloc storage. Captured
            # graphs must retain both mapped owners along with their binding.
            retain_cuda_graph_capture_resource(self)
        native.run_lookup(binding)

    def lookup(self, indices, out):
        _lookup(indices, out, self.key)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index,
        use_sequence_parallel,
        prefix,
    ):
        super().__init__()
        if use_sequence_parallel:
            raise ValueError("V4.1 Engram consumes replicated token rows")
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        plan = layout.plans[layer_hash_index]
        self.disk_prefetch_max_tokens = getattr(layout, "disk_prefetch_max_tokens", 0)
        self.embed_tokens = ParallelEngramEmbedding(
            plan,
            layout.table_memory,
            resident_scales=getattr(layout, "disk_resident_scales", False),
            prefetch=self.disk_prefetch_max_tokens > 0,
        )
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        projection_tp = getattr(layout, "projection_tp", False)
        projection_cls = ColumnParallelLinear if projection_tp else ReplicatedLinear
        self.wkv = projection_cls(
            6144,
            self.dim * (self.hc_mult + 1),
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            **({"gather_output": True} if projection_tp else {}),
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty_like(self.q_weight), requires_grad=False
        )
        self.register_buffer(
            "norm_weights",
            torch.empty(self.hc_mult * self.dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "staged_rows",
            torch.empty(
                plan.caps.max_tokens,
                6144,
                dtype=torch.bfloat16,
                device=plan.caps.device,
            ),
            persistent=False,
        )
        self.mix_plan = hyperconnection.plan(
            hyperconnection.Caps(
                device=plan.caps.device,
                max_tokens=plan.caps.max_tokens,
                hidden_size=self.dim,
                streams=self.hc_mult,
            )
        )

    def process_weights_after_loading(self):
        self.norm_weights.copy_(
            (self.q_weight.float() * self.k_weight.float()).flatten()
        )

    def prepare_embeddings(self, hash_ids):
        self.embed_tokens.lookup(hash_ids, self.staged_rows)

    def invalidate_disk_output(self, *, clear=False):
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        try:
            self.embed_tokens.abort_disk()
        finally:
            if clear:
                self.staged_rows.zero_()

    def prepare_disk(self, hash_ids, num_tokens):
        self.invalidate_disk_output()
        prefetch = 0 < hash_ids.shape[0] <= self.disk_prefetch_max_tokens
        try:
            self.embed_tokens.prepare_disk(
                hash_ids, self.staged_rows, num_tokens, prefetch=prefetch
            )
        except BaseException:
            self.invalidate_disk_output(clear=True)
            raise
        self._disk_prepared_tokens = hash_ids.shape[0]
        self._disk_prepared = not prefetch

    def finish_disk(self):
        self.embed_tokens.finish_disk()
        self._disk_prepared = True

    def prepare_dummy_output(self, num_tokens):
        self.invalidate_disk_output(clear=True)
        self._disk_prepared_tokens = num_tokens
        self._disk_prepared = True

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if (
            self.embed_tokens.disk_table is not None
            and not torch.compiler.is_compiling()
            and (
                not self._disk_prepared
                or self._disk_prepared_tokens < hash_ids.shape[0]
            )
        ):
            raise RuntimeError("Disk Engram output is not prepared")
        rows = tensor_model_parallel_all_reduce(self.staged_rows[: hash_ids.shape[0]])
        kv = self.wkv(rows)
        state = hidden_states.flatten(1)
        out = torch.empty_like(state)
        hyperconnection.run_engram_mix(
            state,
            kv,
            self.norm_weights,
            eps=self.eps,
            plan=self.mix_plan,
            out=out,
            token_mask=token_mask,
        )
        return out.view_as(hidden_states)
