# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Engram with readonly accepted history and global ceil-row TP shards."""

from functools import lru_cache
from weakref import WeakValueDictionary

import torch
from flashinfer.b12x.norm import hyperconnection
from flashinfer.b12x.sequence import engram as native
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import get_file_tensor_source
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import retain_cuda_graph_capture_resource

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
            native.run(binding)
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


def _load_table(param, loaded_weight):
    if loaded_weight.shape != (param.global_rows, *param.shape[1:]):
        raise ValueError("Engram source must be the unpadded global checkpoint table")
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    start = param.shard_start
    count = max(0, min(param.shape[0], param.global_rows - start))
    if count < param.shape[0]:
        param.data[count:].zero_()
    if count:
        param.data[:count].copy_(loaded_weight[start : start + count])


@torch.library.custom_op("vllm::dsv41_engram_lookup", mutates_args=("out",))
def _lookup(indices: torch.Tensor, out: torch.Tensor, key: int) -> None:
    _TABLES[key].lookup_native(indices, out)


@_lookup.register_fake
def _lookup_fake(indices, out, key):
    return None


class ParallelEngramEmbedding(nn.Module):
    def __init__(self, plan, table_memory="device"):
        super().__init__()
        self.plan = plan
        self.key = id(self)
        _TABLES[self.key] = self
        self.tp_size = plan.caps.tp_size
        if table_memory not in ("device", "disk"):
            raise ValueError("Engram table_memory must be device or disk")
        self.disk_table = native.DiskTable(plan) if table_memory == "disk" else None
        if self.disk_table is None:
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
            for param in (self.weight, self.weight_scale_inv):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": _load_table,
                        "global_rows": plan.table_rows,
                        "shard_start": plan.shard_start,
                    },
                )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("weight_scale_inv", None)
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

    def prepare_disk(self, indices, out, num_tokens):
        if self.disk_table is None:
            raise RuntimeError("Engram table is not disk-backed")
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        self.hashes[: indices.shape[0]].copy_(indices)
        self.num_tokens.copy_(num_tokens)
        binding = native.bind_lookup(
            self.plan,
            weight=None,
            scales=None,
            hash_ids=self.hashes,
            num_tokens=self.num_tokens,
            out=out,
            disk_table=self.disk_table,
        )
        native.run_lookup(binding, token_count=indices.shape[0])

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
        self.embed_tokens = ParallelEngramEmbedding(plan, layout.table_memory)
        self._disk_prepared = False
        self._disk_prepared_tokens = 0
        self.wkv = ReplicatedLinear(
            6144,
            self.dim * (self.hc_mult + 1),
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
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
        if clear:
            self.staged_rows.zero_()

    def prepare_disk(self, hash_ids, num_tokens):
        self.invalidate_disk_output()
        try:
            self.embed_tokens.prepare_disk(hash_ids, self.staged_rows, num_tokens)
        except BaseException:
            self.invalidate_disk_output(clear=True)
            raise
        self._disk_prepared_tokens = hash_ids.shape[0]
        self._disk_prepared = True

    def prepare_dummy_output(self, num_tokens):
        self.staged_rows.zero_()
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
