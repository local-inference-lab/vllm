# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X-backed position-learning enhancement for Qwen4Exp."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.weight_transfer import (
    allocate_weights,
    copy_weight,
    flush_weight_transfers,
    get_file_tensor_source,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    get_b12x_ple,
    get_b12x_ple_embedding,
    set_b12x_preparation_provider,
)
from vllm.utils.torch_utils import direct_register_custom_op

from ..config import Qwen4ExpTextConfig

logger = init_logger(__name__)

_PLE_SPLITTING_OPS = (
    "vllm::qwen4_exp_b12x_ple_embedding",
    "vllm::qwen4_exp_b12x_ple",
)


def _register_ple_compilation_context(
    compilation_config: Any,
    layer_name: str,
    layer: nn.Module,
) -> None:
    """Keep request-dependent PLE transactions outside piecewise graphs.

    Piecewise graph dispatch is keyed by padded token count, while PLE hashing
    and recurrent-state routing also depend on the live request count and query
    boundaries. These custom operators must therefore execute as partition
    boundaries so a graph compiled for one request layout cannot replay stale
    PLE metadata for another layout.
    """
    static_context = compilation_config.static_forward_context
    if layer_name in static_context:
        raise ValueError(f"duplicate layer name: {layer_name}")
    static_context[layer_name] = layer

    splitting_ops = compilation_config.splitting_ops
    if splitting_ops is not None:
        for op_name in _PLE_SPLITTING_OPS:
            if op_name not in splitting_ops:
                splitting_ops.append(op_name)


def _b12x_module(name: str) -> Any:
    api = {
        "ple": get_b12x_ple,
        "ple_embedding": get_b12x_ple_embedding,
    }[name]()
    if api is None:
        raise ImportError(
            f"Qwen4Exp requires b12x.sequence.{name}; install the b12x serving extra"
        )
    return api


def _resolve_ple_table_memory(
    additional_config: Any, embedding_dtype: str = "bfloat16"
) -> str:
    """Translate the public offload policy into a b12x storage mode."""
    if isinstance(additional_config, dict) and "ple_table_memory" in additional_config:
        table_memory = additional_config["ple_table_memory"]
    else:
        table_memory = envs.VLLM_PLE_TABLE_MEMORY
        if table_memory is None:
            if envs.is_set("VLLM_PLE_CPU_OFFLOAD"):
                return "mapped_host" if envs.VLLM_PLE_CPU_OFFLOAD else "device"
            return "io_uring" if embedding_dtype == "bfloat16" else "device"
    if table_memory == "ram":
        return "mapped_host"
    if table_memory == "disk":
        return "io_uring"
    if table_memory == "device":
        return "device"
    raise ValueError(
        "additional_config.ple_table_memory must be 'device', "
        f"'ram', or 'disk', got {table_memory!r}"
    )


def _copy_embedding_shard(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> torch.Tensor | None:
    checkpoint_end = checkpoint_start + loaded_weight.shape[0]
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    rows = overlap_end - overlap_start
    source = loaded_weight.narrow(0, overlap_start - checkpoint_start, rows)
    target = destination.narrow(0, overlap_start - tp_start, rows)
    with torch.no_grad():
        copy_weight(target, source)
    return target


class _NGramEmbeddingStorage(nn.Module):
    def __init__(self, layout: Any, shard_rows: int) -> None:
        super().__init__()
        self.disk_table = None
        self._table_storage = None
        if layout.caps.table_memory == "io_uring":
            api = _b12x_module("ple_embedding")
            self.disk_table = api.DiskTable(layout, shard_rows)
            tensors: dict[str, torch.Tensor | None] = {"weight": None}
            for name in ("weight_scale", "weight_scale_2"):
                shape = getattr(layout, f"{name}_shape")
                tensors[name] = (
                    allocate_weights(
                        torch.empty,
                        shape,
                        dtype=getattr(layout, f"{name}_dtype"),
                        device=layout.caps.device,
                    )
                    if shape == (1,)
                    else None
                )
        else:
            self._table_storage = allocate_weights(layout.allocate_storage)
            tensors = {
                name: getattr(self._table_storage, name)
                for name in ("weight", "weight_scale", "weight_scale_2")
            }
        for name, tensor in tensors.items():
            self.register_parameter(
                name,
                nn.Parameter(tensor, requires_grad=False)
                if tensor is not None
                else None,
            )

    @property
    def weight_load_view(self) -> torch.Tensor | None:
        return self._table_storage.weight_load_view if self._table_storage else None

    @property
    def weight_scale_load_view(self) -> torch.Tensor | None:
        if self._table_storage is None:
            return self.weight_scale
        return self._table_storage.weight_scale_load_view

    @property
    def weight_scale_2_load_view(self) -> torch.Tensor | None:
        if self._table_storage is None:
            return self.weight_scale_2
        return self._table_storage.weight_scale_2_load_view

    @property
    def mapped_host_nbytes(self) -> int:
        return int(self._table_storage.mapped_host_nbytes) if self._table_storage else 0


class B12xNGramEmbedding(nn.Module):
    """Prime-hashed learned n-gram embedding with fixed b12x storage."""

    _STORAGE_MODES = {
        "bfloat16": "bf16",
        "float8_e4m3fn": "fp8_e4m3_per_tensor",
        "float4_e2m1fn_x2": "nvfp4_group16",
        "nvfp4": "nvfp4_group16",
        "nvfp4_group16": "nvfp4_group16",
        "uint8": "nvfp4_group16",
    }

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        owner_prefix: str,
        prefix: str,
        dtype: torch.dtype,
        table_memory: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_total_tokens = int(max_total_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.requires_disk_preparation = table_memory == "io_uring"
        self._disk_prepared_tokens = -1
        self._disk_prepared = False
        self.head_dim = self.embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        self.owner_prefix = owner_prefix
        self.embedding_storage_dtype = str(
            getattr(config, "ple_embedding_dtype", "bfloat16")
        )
        if self.embedding_storage_dtype not in self._STORAGE_MODES:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding storage dtype "
                f"{self.embedding_storage_dtype!r} is unsupported"
            )
        self._quant_mode = self._STORAGE_MODES[self.embedding_storage_dtype]
        self._embedding_load_ranges: set[tuple[int, int]] = set()
        self._scale_load_ranges: set[tuple[int, int]] = set()
        self._weight_scale_loaded = False
        self._weight_scale_2_loaded = False
        self._embedding_validated = False

        device = torch.device(current_platform.current_device())
        common_caps = {
            "device": device,
            "max_tokens": self.max_total_tokens,
            "max_seqs": self.max_num_reqs,
            "vocab_size": int(config.vocab_size),
            "eos_token_id": self.eos_token_id,
            "max_order": self.ngram_size,
            "heads_per_order": self.heads_per_ngram,
            "dense_layer_ordinal": int(ple_dense_layer_id),
            "base_table_size": int(config.ngram_vocab_size_base),
            "table_alignment": int(config.make_ngram_vocab_size_divisible_by),
        }
        api = _b12x_module("ple_embedding")
        self._caps = api.Caps(
            **common_caps,
            embedding_dim=self.embedding_dim,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
            quant_mode=self._quant_mode,
            table_memory=table_memory,
            output_dtype=dtype,
        )
        self._geometry = api.compute_geometry(self._caps)
        geometry_tensors = api.allocate_geometry(self._geometry, device=device)
        self._table_layout = api.storage_layout(self._caps, geometry=self._geometry)
        self.register_buffer("layer_multipliers", geometry_tensors.multipliers)
        self.register_buffer("ngram_heads_offsets", geometry_tensors.table_offsets)
        self.register_buffer("ngram_heads_vocab_sizes", geometry_tensors.prime_sizes)
        self._plans: dict[int, object] = {}
        shard_rows = (
            self._table_layout.padded_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        self.ngram_embedding = _NGramEmbeddingStorage(self._table_layout, shard_rows)
        if self.ngram_embedding.mapped_host_nbytes:
            logger.info(
                "Using %.2f GiB of CUDA-mapped host memory for this TP rank's "
                "PLE table",
                self.ngram_embedding.mapped_host_nbytes / (1 << 30),
            )
        scratch_spec = self._table_layout.scratch_specs()[0]
        self.register_buffer(
            "_scratch",
            torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_token_ids",
            torch.empty(self.max_total_tokens, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_query_start_loc",
            torch.empty(self.max_num_reqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_committed_history",
            torch.empty(
                self.max_num_reqs,
                self.ngram_size - 1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_num_seqs",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_tokens",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_embedding_out",
            torch.empty(
                self._table_layout.output_shape,
                dtype=self._table_layout.output_dtype,
                device=device,
            ),
            persistent=False,
        )
        if not getattr(self, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(self, self)

    def _declare_embedding_plan(self, token_count: int):
        api = _b12x_module("ple_embedding")
        return api.plan(
            replace(self._caps, max_tokens=token_count),
            geometry=self._geometry,
            prime_sizes=self.ngram_heads_vocab_sizes,
            table_offsets=self.ngram_heads_offsets,
            multipliers=self.layer_multipliers,
        )

    def _capacity_for(self, token_count: int) -> int:
        if not 0 <= token_count <= self.max_total_tokens:
            raise ValueError("PLE embedding token count exceeds capacity")
        if not self.requires_disk_preparation and token_count in self._plans:
            return token_count
        return self.max_total_tokens

    def _plan_for(self, token_count: int):
        """Resolve a live token count within the declared embedding capacity."""
        capacity = self._capacity_for(token_count)
        plan = self._plans.get(capacity)
        if plan is None:
            plan = self._declare_embedding_plan(capacity)
            self._plans[capacity] = plan
        return plan

    def _bind_embedding(self, token_count: int):
        capacity = self._capacity_for(token_count)
        api = _b12x_module("ple_embedding")
        return api.bind(
            self._plan_for(token_count),
            scratch=self._scratch,
            weight=self.ngram_embedding.weight,
            weight_scale=self.ngram_embedding.weight_scale,
            weight_scale_2=self.ngram_embedding.weight_scale_2,
            token_ids=self._token_ids[:capacity],
            query_start_loc=self._query_start_loc,
            committed_history=self._committed_history,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            out=self._embedding_out[:capacity],
            disk_table=self.ngram_embedding.disk_table,
        )

    def _bind_embedding_state(self, state: Any, token_count: int):
        capacity = (
            self.max_total_tokens if self.requires_disk_preparation else token_count
        )
        return state.bind(
            scratch=self._scratch,
            weight=self.ngram_embedding.weight,
            weight_scale=self.ngram_embedding.weight_scale,
            weight_scale_2=self.ngram_embedding.weight_scale_2,
            token_ids=self._token_ids[:capacity],
            query_start_loc=self._query_start_loc,
            committed_history=self._committed_history,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            out=self._embedding_out[:capacity],
            disk_table=self.ngram_embedding.disk_table,
        )

    def _request_name(self, token_count: int) -> str:
        return f"{self.owner_prefix}.ple_embedding.m{token_count}"

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> tuple[B12xPreparationUnit, ...]:
        if layer is not self:
            raise ValueError("PLE embedding preparation owner mismatch")
        self._validate_embedding_loaded()
        requests = []
        plans = self._plans
        token_counts = (
            (self.max_total_tokens,)
            if self.requires_disk_preparation
            else tuple(sorted({self.max_total_tokens, *workload.fixed_token_counts}))
        )
        # Plans are declared once per token count; a later collection reuses
        # them so prepared state stays installed.
        for token_count in token_counts:
            plan = plans.get(token_count)
            if plan is None:
                plan = self._declare_embedding_plan(token_count)
                plans[token_count] = plan
            requests.append(
                plan.request(
                    name=self._request_name(token_count),
                    prepare_call=self._embedding_prepare_call(token_count),
                )
            )
        if not requests:
            return ()
        return (
            B12xPreparationUnit(
                name="PLE embedding",
                key=self.owner_prefix,
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def _embedding_prepare_call(self, token_count: int):
        def prepare(state):
            from b12x.preparation import PreparedCall

            # Prime one real row through the checkpoint-owned table and
            # geometry.  These are the only mutable staging rows this callback
            # touches; unlike a pool clone they are restored when the call is
            # released and never replace checkpoint parameters.
            capacity = (
                self.max_total_tokens if self.requires_disk_preparation else token_count
            )
            token_ids = self._token_ids[:1].clone()
            query_start = self._query_start_loc[:2].clone()
            history = self._committed_history[:1].clone()
            num_seqs = self._num_seqs.clone()
            num_tokens = self._num_tokens.clone()
            output = self._embedding_out[:1].clone()

            def produce() -> None:
                self._token_ids[0] = self.eos_token_id
                self._query_start_loc[:2].zero_()
                self._query_start_loc[1] = 1
                self._committed_history[0].fill_(self.eos_token_id)
                self._num_seqs.fill_(1)
                self._num_tokens.fill_(1)

            def restore() -> None:
                self._token_ids[:1].copy_(token_ids)
                self._query_start_loc[:2].copy_(query_start)
                self._committed_history[:1].copy_(history)
                self._num_seqs.copy_(num_seqs)
                self._num_tokens.copy_(num_tokens)
                self._embedding_out[:1].copy_(output)

            binding = self._bind_embedding_state(state, capacity)
            return PreparedCall(
                run=lambda: state.run(binding, token_count=1),
                produce=produce,
                restore=restore,
                owners=(binding,),
            )

        return prepare

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> int:
        token_count = input_ids.numel()
        num_seqs = query_start_loc.numel() - 1
        if token_count > self.max_total_tokens or num_seqs > self.max_num_reqs:
            raise ValueError(
                "PLE hashing capacity exceeded: "
                f"tokens={token_count}/{self.max_total_tokens}, "
                f"requests={num_seqs}/{self.max_num_reqs}"
            )
        if tuple(ngram_context.shape) != (num_seqs, self.ngram_size - 1):
            raise ValueError(
                "ngram_context must have shape "
                f"{(num_seqs, self.ngram_size - 1)}, got "
                f"{tuple(ngram_context.shape)}"
            )
        self._token_ids.fill_(self.eos_token_id)
        self._token_ids[:token_count].copy_(input_ids.reshape(-1).to(torch.int64))
        self._query_start_loc.zero_()
        self._query_start_loc[: num_seqs + 1].copy_(query_start_loc.to(torch.int32))
        self._committed_history.fill_(self.eos_token_id)
        self._committed_history[:num_seqs].copy_(ngram_context.to(torch.int64))
        self._num_seqs.fill_(num_seqs)
        self._num_tokens.copy_(query_start_loc[num_seqs : num_seqs + 1])
        return token_count

    def _run_embedding(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        self._validate_embedding_loaded()
        self._plan_for(input_ids.numel())
        token_count = self._prepare_inputs(input_ids, query_start_loc, ngram_context)
        _b12x_module("ple_embedding").run(
            self._bind_embedding(token_count), token_count=token_count
        )

    def prepare_disk(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Produce local embeddings before the model's graph is replayed."""
        if not self.requires_disk_preparation:
            raise RuntimeError("prepare_disk requires an io_uring PLE table")
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Disk PLE preparation must run outside CUDA graphs")
        self._disk_prepared_tokens = -1
        self._disk_prepared = False
        self._run_embedding(input_ids, query_start_loc, ngram_context)
        self._disk_prepared_tokens = input_ids.numel()
        self._disk_prepared = True

    def prepare_dummy_output(self, num_tokens: int) -> None:
        """Initialize graph-visible profiling output without reading the table."""
        if not self.requires_disk_preparation:
            raise RuntimeError("Dummy disk output requires an io_uring table")
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Dummy disk PLE preparation must run outside CUDA graphs"
            )
        if not 0 <= num_tokens <= self.max_total_tokens:
            raise ValueError("Dummy PLE output exceeds token capacity")
        self._embedding_out[:num_tokens].zero_()
        self._disk_prepared_tokens = num_tokens
        self._disk_prepared = True

    def _validate_embedding_loaded(self) -> None:
        if self._embedding_validated:
            return

        covered_until = self._table_layout.shard_start
        for start, end in sorted(self._embedding_load_ranges):
            if start > covered_until:
                raise ValueError(
                    "PLE embedding shards do not cover the local table: "
                    f"expected row {covered_until}, got {start}"
                )
            covered_until = max(covered_until, end)
        if covered_until != self._table_layout.shard_end:
            raise ValueError(
                "PLE embedding shards do not cover the local table: "
                f"stopped at row {covered_until}, "
                f"expected {self._table_layout.shard_end}"
            )

        if self._table_layout.weight_scale_shape is not None:
            if self._quant_mode == "fp8_e4m3_per_tensor":
                if not self._weight_scale_loaded:
                    raise ValueError(
                        "FP8 PLE embedding checkpoint is missing weight_scale"
                    )
            else:
                scale_covered_until = self._table_layout.shard_start
                for start, end in sorted(self._scale_load_ranges):
                    if start > scale_covered_until:
                        raise ValueError(
                            "NVFP4 PLE scale shards do not cover the local table: "
                            f"expected row {scale_covered_until}, got {start}"
                        )
                    scale_covered_until = max(scale_covered_until, end)
                if scale_covered_until != self._table_layout.shard_end:
                    raise ValueError(
                        "NVFP4 PLE scale shards do not cover the local table: "
                        f"stopped at row {scale_covered_until}, expected "
                        f"{self._table_layout.shard_end}"
                    )
        if (
            self._table_layout.weight_scale_2_shape is not None
            and not self._weight_scale_2_loaded
        ):
            raise ValueError("NVFP4 PLE embedding checkpoint is missing weight_scale_2")
        self._embedding_validated = True

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if self.requires_disk_preparation:
            # Batch counts must not become per-step Dynamo specialization guards.
            if not self._disk_prepared or (
                not torch.compiler.is_compiling()
                and self._disk_prepared_tokens < input_ids.shape[0]
            ):
                raise RuntimeError(
                    "Disk PLE output is not prepared; call prepare_disk before forward"
                )
        elif torch.compiler.is_compiling():
            torch.ops.vllm.qwen4_exp_b12x_ple_embedding(
                input_ids,
                query_start_loc,
                ngram_context,
                self._embedding_out,
                self.owner_prefix,
            )
        else:
            self._run_embedding(input_ids, query_start_loc, ngram_context)
        embeddings = self._embedding_out[: input_ids.shape[0]]
        if get_tensor_model_parallel_world_size() > 1:
            embeddings = tensor_model_parallel_all_reduce(embeddings)
        return embeddings

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"
        embedding = self.ngram_embedding
        org_vocab_size = self._table_layout.padded_vocab_size
        tp_start = self._table_layout.shard_start
        tp_end = self._table_layout.shard_end
        shard_size = (
            org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if tuple(buffer.shape) != tuple(loaded_weight.shape):
                    raise ValueError(
                        f"shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                        f"got {tuple(loaded_weight.shape)}"
                    )
                checkpoint_value = loaded_weight.to(
                    device=buffer.device, dtype=buffer.dtype
                )
                if not torch.equal(buffer, checkpoint_value):
                    raise ValueError(
                        f"checkpoint {name} does not match planned PLE geometry"
                    )
                loaded.add(name)
                continue
            if name == "ngram_embedding.weight_scale":
                if self._quant_mode != "fp8_e4m3_per_tensor":
                    regular_weights.append((name, loaded_weight))
                    continue
                expected_shape = self._table_layout.weight_scale_shape
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        "shape mismatch for PLE embedding scale: expected "
                        f"{expected_shape}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype != self._table_layout.weight_scale_dtype:
                    raise TypeError(
                        "PLE embedding weight_scale must have dtype "
                        f"{self._table_layout.weight_scale_dtype}, "
                        f"got {loaded_weight.dtype}"
                    )
                scale = loaded_weight.float()
                if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                    raise ValueError(
                        "PLE embedding weight_scale must be finite and positive"
                    )
                with torch.no_grad():
                    target = getattr(
                        embedding, "weight_scale_load_view", embedding.weight_scale
                    )
                    assert target is not None
                    target.copy_(
                        loaded_weight.to(
                            device=target.device,
                            dtype=target.dtype,
                        )
                    )
                self._weight_scale_loaded = True
                self._embedding_validated = False
                loaded.add(name)
                continue
            if name == "ngram_embedding.weight_scale_2":
                if self._quant_mode != "nvfp4_group16":
                    regular_weights.append((name, loaded_weight))
                    continue
                expected_shape = self._table_layout.weight_scale_2_shape
                if loaded_weight.ndim == 0 and expected_shape == (1,):
                    loaded_weight = loaded_weight.reshape(1)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        "shape mismatch for PLE embedding weight_scale_2: expected "
                        f"{expected_shape}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype != self._table_layout.weight_scale_2_dtype:
                    raise TypeError(
                        "PLE embedding weight_scale_2 must have dtype "
                        f"{self._table_layout.weight_scale_2_dtype}, got "
                        f"{loaded_weight.dtype}"
                    )
                scale_2 = loaded_weight.float()
                if not bool(torch.isfinite(scale_2).all()) or not bool(
                    (scale_2 > 0).all()
                ):
                    raise ValueError(
                        "PLE embedding weight_scale_2 must be finite and positive"
                    )
                with torch.no_grad():
                    target = getattr(
                        embedding,
                        "weight_scale_2_load_view",
                        embedding.weight_scale_2,
                    )
                    assert target is not None
                    target.copy_(
                        loaded_weight.to(device=target.device, dtype=target.dtype)
                    )
                self._weight_scale_2_loaded = True
                self._embedding_validated = False
                loaded.add(name)
                continue
            if name.startswith(shard_prefix):
                shard_and_suffix = name[len(shard_prefix) :]
                shard_text, separator, suffix = shard_and_suffix.partition(".")
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE shard {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, org_vocab_size - checkpoint_start),
                )
                if separator != "." or suffix not in {"weight", "weight_scale"}:
                    regular_weights.append((name, loaded_weight))
                    continue
                if suffix == "weight":
                    expected_shape = (expected_rows, self._table_layout.weight_shape[1])
                    expected_dtype = self._table_layout.weight_dtype
                else:
                    if self._quant_mode != "nvfp4_group16":
                        regular_weights.append((name, loaded_weight))
                        continue
                    assert self._table_layout.weight_scale_shape is not None
                    assert self._table_layout.weight_scale_dtype is not None
                    expected_shape = (
                        expected_rows,
                        self._table_layout.weight_scale_shape[1],
                    )
                    expected_dtype = self._table_layout.weight_scale_dtype
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"shape mismatch for PLE shard {shard_index} {suffix}: "
                        f"expected {expected_shape}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype != expected_dtype:
                    raise TypeError(
                        f"PLE shard {shard_index} {suffix} must have dtype "
                        f"{expected_dtype}, got {loaded_weight.dtype}"
                    )
                overlap_start = max(checkpoint_start, tp_start)
                overlap_end = min(checkpoint_start + expected_rows, tp_end)
                disk_table = getattr(embedding, "disk_table", None)
                if disk_table is not None:
                    source = get_file_tensor_source(loaded_weight)
                    if source is None:
                        raise ValueError(
                            "File-backed PLE tables require safetensors weights "
                            "from the b12x, instanttensor, fastsafetensors, or "
                            "safetensors loader"
                        )
                    if source.shape != expected_shape or source.dtype != expected_dtype:
                        raise ValueError(f"file source geometry does not match {name}")
                    if overlap_start < overlap_end:
                        disk_table.add_shard(
                            shard_index,
                            source.path,
                            source.offset,
                            scale=suffix == "weight_scale",
                        )
                else:
                    parameter = getattr(embedding, suffix)
                    destination = getattr(
                        embedding, f"{suffix}_load_view", parameter.data
                    )
                    assert destination is not None
                    target = _copy_embedding_shard(
                        destination,
                        loaded_weight,
                        checkpoint_start=checkpoint_start,
                        tp_start=tp_start,
                        tp_end=tp_end,
                    )
                    if suffix == "weight_scale" and target is not None:
                        flush_weight_transfers()
                        scale = target.float()
                        if not bool(torch.isfinite(scale).all()) or not bool(
                            (scale > 0).all()
                        ):
                            raise ValueError(
                                f"PLE shard {shard_index} weight_scale must be "
                                "finite and positive"
                            )
                if overlap_start < overlap_end:
                    load_ranges = (
                        self._embedding_load_ranges
                        if suffix == "weight"
                        else self._scale_load_ranges
                    )
                    load_ranges.add((overlap_start, overlap_end))
                    self._embedding_validated = False
                loaded.add(f"ngram_embedding.{suffix}")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


def _ple_embedding_op(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.ple_embedding._run_embedding(input_ids, query_start_loc, ngram_context)


def _ple_embedding_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _ple_op(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._run_ple(residual, key, value, query_start_loc)


def _ple_fake(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_b12x_ple_embedding",
    op_func=_ple_embedding_op,
    mutates_args=["out"],
    fake_impl=_ple_embedding_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_b12x_ple",
    op_func=_ple_op,
    mutates_args=["out"],
    fake_impl=_ple_fake,
)
