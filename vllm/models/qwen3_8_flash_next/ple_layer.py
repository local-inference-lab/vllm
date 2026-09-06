# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X-backed position-learning enhancement for Qwen3.8-Flash-Next."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.weight_transfer import (
    allocate_weights,
    copy_weight,
    flush_weight_transfers,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    get_b12x_ple,
    get_b12x_ple_embedding,
    get_b12x_scratch_buffers,
)
from vllm.utils.torch_utils import current_stream, direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .config import Qwen3_8FlashNextTextConfig

logger = init_logger(__name__)

_PLE_SPLITTING_OPS = (
    "vllm::qwen3_8_flash_next_ple_embedding",
    "vllm::qwen3_8_flash_next_ple_prefetch",
    "vllm::qwen3_8_flash_next_ple_prefetch_wait",
    "vllm::qwen3_8_flash_next_ple",
)

_prefetch_stream: torch.cuda.Stream | None = None


def _get_prefetch_stream() -> torch.cuda.Stream:
    global _prefetch_stream
    if _prefetch_stream is None:
        _prefetch_stream = torch.cuda.Stream()
    return _prefetch_stream


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
            f"Qwen3.8-Flash-Next requires b12x.sequence.{name}; "
            "install the b12x serving extra"
        )
    return api


def _resolve_ple_table_memory(additional_config: Any) -> str:
    if isinstance(additional_config, dict) and "ple_table_memory" in additional_config:
        table_memory = additional_config["ple_table_memory"]
    else:
        table_memory = "mapped_host" if envs.VLLM_PLE_CPU_OFFLOAD else "device"
    if table_memory not in {"device", "mapped_host"}:
        raise ValueError(
            "additional_config.ple_table_memory must be 'device' or "
            f"'mapped_host', got {table_memory!r}"
        )
    return table_memory


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


class Qwen3_8FlashNextPLEGroupedNorm(nn.Module):
    """Checkpoint-compatible zero-centered grouped RMSNorm parameter."""

    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            allocate_weights(torch.zeros, hidden_size, dtype=dtype)
        )


class _NGramEmbeddingStorage(nn.Module):
    def __init__(self, plan: Any) -> None:
        super().__init__()
        self._table_storage = allocate_weights(plan.allocate_storage)
        self.weight = nn.Parameter(
            self._table_storage.weight,
            requires_grad=False,
        )
        if plan.weight_scale_shape is None:
            self.register_parameter("weight_scale", None)
        else:
            weight_scale = self._table_storage.weight_scale
            assert weight_scale is not None
            self.weight_scale = nn.Parameter(
                weight_scale,
                requires_grad=False,
            )
        if plan.weight_scale_2_shape is None:
            self.register_parameter("weight_scale_2", None)
        else:
            weight_scale_2 = self._table_storage.weight_scale_2
            assert weight_scale_2 is not None
            self.weight_scale_2 = nn.Parameter(
                weight_scale_2,
                requires_grad=False,
            )

    @property
    def weight_load_view(self) -> torch.Tensor:
        return self._table_storage.weight_load_view

    @property
    def weight_scale_load_view(self) -> torch.Tensor | None:
        return self._table_storage.weight_scale_load_view

    @property
    def weight_scale_2_load_view(self) -> torch.Tensor | None:
        return self._table_storage.weight_scale_2_load_view

    @property
    def mapped_host_nbytes(self) -> int:
        return int(self._table_storage.mapped_host_nbytes)


class Qwen3_8FlashNextNGramEmbedding(nn.Module):
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
        config: Qwen3_8FlashNextTextConfig,
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
        self.head_dim = self.embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        self.owner_prefix = owner_prefix
        if envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP and current_platform.is_cuda():
            _get_prefetch_stream()
        self.embedding_storage_dtype = str(
            getattr(config, "ple_embedding_dtype", "bfloat16")
        )
        if self.embedding_storage_dtype not in self._STORAGE_MODES:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next PLE embedding storage dtype "
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
        self._plan = api.plan(
            api.Caps(
                **common_caps,
                embedding_dim=self.embedding_dim,
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
                quant_mode=self._quant_mode,
                table_memory=table_memory,
                output_dtype=dtype,
            )
        )
        # The plan and checkpoint loader share these exact tensors.  Loading a
        # checkpoint updates the persistent geometry in place without making a
        # second plan or changing graph-visible addresses.
        self.register_buffer("layer_multipliers", self._plan.multipliers)
        self.register_buffer("ngram_heads_offsets", self._plan.table_offsets)
        self.register_buffer("ngram_heads_vocab_sizes", self._plan.prime_sizes)

        self.ngram_embedding = _NGramEmbeddingStorage(self._plan)
        if self.ngram_embedding.mapped_host_nbytes:
            logger.info(
                "Using %.2f GiB of CUDA-mapped host memory for this TP rank's "
                "PLE table",
                self.ngram_embedding.mapped_host_nbytes / (1 << 30),
            )

        (scratch,) = get_b12x_scratch_buffers(self._plan)
        self.register_buffer(
            "_scratch",
            scratch,
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
                self._plan.output_shape,
                dtype=self._plan.output_dtype,
                device=device,
            ),
            persistent=False,
        )

    def _bind_embedding(self):
        return self._plan.bind(
            scratch=self._scratch,
            weight=self.ngram_embedding.weight,
            weight_scale=self.ngram_embedding.weight_scale,
            weight_scale_2=self.ngram_embedding.weight_scale_2,
            token_ids=self._token_ids,
            query_start_loc=self._query_start_loc,
            committed_history=self._committed_history,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            out=self._embedding_out,
        )

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
        token_count = self._prepare_inputs(input_ids, query_start_loc, ngram_context)
        _b12x_module("ple_embedding").run(
            self._bind_embedding(), token_count=token_count
        )

    def _run_prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        if input_ids.numel() > 16 or not torch.cuda.is_current_stream_capturing():
            self._run_embedding(input_ids, query_start_loc, ngram_context)
            return
        stream = _get_prefetch_stream()
        stream.wait_stream(current_stream())
        for tensor in (input_ids, query_start_loc, ngram_context):
            tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            self._run_embedding(input_ids, query_start_loc, ngram_context)

    def prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        if torch.compiler.is_compiling():
            torch.ops.vllm.qwen3_8_flash_next_ple_prefetch(
                input_ids,
                query_start_loc,
                ngram_context,
                self._embedding_out,
                self.owner_prefix,
            )
        else:
            self._run_prefetch(input_ids, query_start_loc, ngram_context)

    def _validate_embedding_loaded(self) -> None:
        if self._embedding_validated:
            return

        covered_until = self._plan.shard_start
        for start, end in sorted(self._embedding_load_ranges):
            if start > covered_until:
                raise ValueError(
                    "PLE embedding shards do not cover the local table: "
                    f"expected row {covered_until}, got {start}"
                )
            covered_until = max(covered_until, end)
        if covered_until != self._plan.shard_end:
            raise ValueError(
                "PLE embedding shards do not cover the local table: "
                f"stopped at row {covered_until}, expected {self._plan.shard_end}"
            )

        if self._plan.weight_scale_shape is not None:
            if self._quant_mode == "fp8_e4m3_per_tensor":
                if not self._weight_scale_loaded:
                    raise ValueError(
                        "FP8 PLE embedding checkpoint is missing weight_scale"
                    )
            else:
                scale_covered_until = self._plan.shard_start
                for start, end in sorted(self._scale_load_ranges):
                    if start > scale_covered_until:
                        raise ValueError(
                            "NVFP4 PLE scale shards do not cover the local table: "
                            f"expected row {scale_covered_until}, got {start}"
                        )
                    scale_covered_until = max(scale_covered_until, end)
                if scale_covered_until != self._plan.shard_end:
                    raise ValueError(
                        "NVFP4 PLE scale shards do not cover the local table: "
                        f"stopped at row {scale_covered_until}, expected "
                        f"{self._plan.shard_end}"
                    )
        if (
            self._plan.weight_scale_2_shape is not None
            and not self._weight_scale_2_loaded
        ):
            raise ValueError("NVFP4 PLE embedding checkpoint is missing weight_scale_2")
        self._embedding_validated = True

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        *,
        wait_for: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if wait_for is not None:
            torch.ops.vllm.qwen3_8_flash_next_ple_prefetch_wait(
                self._embedding_out, wait_for
            )
        elif torch.compiler.is_compiling():
            torch.ops.vllm.qwen3_8_flash_next_ple_embedding(
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
        org_vocab_size = self._plan.padded_vocab_size
        tp_start = self._plan.shard_start
        tp_end = self._plan.shard_end
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
                expected_shape = self._plan.weight_scale_shape
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        "shape mismatch for PLE embedding scale: expected "
                        f"{expected_shape}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype != self._plan.weight_scale_dtype:
                    raise TypeError(
                        "PLE embedding weight_scale must have dtype "
                        f"{self._plan.weight_scale_dtype}, got {loaded_weight.dtype}"
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
                expected_shape = self._plan.weight_scale_2_shape
                if loaded_weight.ndim == 0 and expected_shape == (1,):
                    loaded_weight = loaded_weight.reshape(1)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        "shape mismatch for PLE embedding weight_scale_2: expected "
                        f"{expected_shape}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype != self._plan.weight_scale_2_dtype:
                    raise TypeError(
                        "PLE embedding weight_scale_2 must have dtype "
                        f"{self._plan.weight_scale_2_dtype}, got "
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
                    expected_shape = (expected_rows, self._plan.weight_shape[1])
                    expected_dtype = self._plan.weight_dtype
                    destination = getattr(
                        embedding, "weight_load_view", embedding.weight.data
                    )
                else:
                    if self._quant_mode != "nvfp4_group16":
                        regular_weights.append((name, loaded_weight))
                        continue
                    assert self._plan.weight_scale_shape is not None
                    assert self._plan.weight_scale_dtype is not None
                    expected_shape = (
                        expected_rows,
                        self._plan.weight_scale_shape[1],
                    )
                    expected_dtype = self._plan.weight_scale_dtype
                    destination = getattr(
                        embedding,
                        "weight_scale_load_view",
                        embedding.weight_scale.data,
                    )
                    assert destination is not None
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
                overlap_start = max(checkpoint_start, tp_start)
                overlap_end = min(checkpoint_start + expected_rows, tp_end)
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


class Qwen3_8FlashNextPLELayer(nn.Module, MambaBase):
    """PLE projections plus b12x stateful mixed prefill/decode kernel."""

    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not is_conv_state_dim_first():
            raise RuntimeError(
                "b12x PLE requires VLLM_SSM_CONV_STATE_LAYOUT=DS so its "
                "per-page convolution state uses [channels, history] layout"
            )
        self.model_config: ModelConfig = vllm_config.model_config
        self.cache_config: CacheConfig = vllm_config.cache_config
        self.layer_idx = int(layer_idx)
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else self.layer_idx
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = int(vllm_config.num_speculative_tokens)
        self.max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.eps = float(config.rms_norm_eps)
        dtype = self.model_config.dtype
        if dtype != torch.bfloat16:
            raise TypeError("b12x PLE requires a BF16 model")

        table_memory = _resolve_ple_table_memory(vllm_config.additional_config)

        self.ple_embedding = Qwen3_8FlashNextNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            self.max_tokens,
            self.max_seqs,
            prefix,
            f"{prefix}.ple_embedding",
            dtype,
            table_memory,
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            prefix=f"{prefix}.key_proj",
            return_bias=False,
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            prefix=f"{prefix}.value_proj",
            return_bias=False,
        )
        self.norm_key = Qwen3_8FlashNextPLEGroupedNorm(self.hc_hidden_size, dtype)
        self.norm_query = Qwen3_8FlashNextPLEGroupedNorm(self.hc_hidden_size, dtype)
        self.norm_conv = Qwen3_8FlashNextPLEGroupedNorm(self.hc_hidden_size, dtype)
        self.conv1d = allocate_weights(
            nn.Conv1d,
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            bias=False,
            dtype=dtype,
            device=current_platform.current_device(),
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True

        device = torch.device(current_platform.current_device())
        factory = dict(device=device, dtype=dtype)
        self.register_buffer(
            "_residual",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_key",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_value",
            torch.empty(self.max_tokens, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_out",
            torch.empty(self.max_tokens, self.hc_count, self.hidden_size, **factory),
            persistent=False,
        )
        self.register_buffer(
            "_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_state_slot_ids",
            torch.full(
                (self.max_seqs,), NULL_BLOCK_ID, dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "_state_is_fresh",
            torch.ones(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_accepted_tokens",
            torch.ones(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_request_is_prefill",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
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
        self.register_buffer("_scratch", None, persistent=False)
        self._plan = None
        self.kv_cache = (torch.tensor([]),)

        compilation_config = get_current_vllm_config().compilation_config
        _register_ple_compilation_context(compilation_config, prefix, self)

    def _make_plan(self, max_state_slots: int):
        api = _b12x_module("ple")
        return api.plan(
            api.Caps(
                device=current_platform.current_device(),
                mode="mixed",
                max_tokens=self.max_tokens,
                max_seqs=self.max_seqs,
                max_state_slots=max_state_slots,
                max_speculative_tokens=self.num_spec_tokens,
                streams=self.hc_count,
                hidden_size=self.hidden_size,
                kernel_size=self.conv_kernel_size,
                dilation=self.short_conv_dilation,
                dtype=self.model_config.dtype,
            )
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        conv_state = self.kv_cache[0]
        expected_tail = self.conv_state_len + self.num_spec_tokens
        if tuple(conv_state.shape[1:]) != (self.hc_hidden_size, expected_tail):
            raise RuntimeError(
                "unexpected PLE cache shape: expected "
                f"[slots,{self.hc_hidden_size},{expected_tail}], got "
                f"{tuple(conv_state.shape)}"
            )
        plan = self._make_plan(max_state_slots=conv_state.shape[0])
        (scratch,) = get_b12x_scratch_buffers(plan)
        self._scratch = scratch
        self._plan = plan

    def _bind_ple(self):
        if self._plan is None:
            raise RuntimeError("PLE KV cache was not bound before inference")
        return self._plan.bind(
            scratch=self._scratch,
            residual=self._residual,
            key=self._key,
            value=self._value,
            k_norm_weight=self.norm_key.weight,
            q_norm_weight=self.norm_query.weight,
            u_norm_weight=self.norm_conv.weight,
            conv_weight=self.conv1d.weight.squeeze(1),
            query_start_loc=self._query_start_loc,
            state_slot_ids=self._state_slot_ids,
            state_is_fresh=self._state_is_fresh,
            num_accepted_tokens=self._num_accepted_tokens,
            num_seqs=self._num_seqs,
            num_tokens=self._num_tokens,
            conv_state=self.kv_cache[0],
            out=self._out,
            request_is_prefill=self._request_is_prefill,
        )

    def unbind_kv_cache(self) -> None:
        self._plan = None
        self._scratch = None
        super().unbind_kv_cache()

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        # b12x binds the DS layout directly; construction rejects the global SD
        # layout so runner state-copy semantics remain consistent.
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _prepare_metadata(
        self,
        metadata: ShortConvAttentionMetadata,
        query_start_loc: torch.Tensor,
        token_count: int,
    ) -> None:
        num_seqs = int(metadata.num_reqs)
        if num_seqs > self.max_seqs or token_count > self.max_tokens:
            raise ValueError(
                f"PLE capacity exceeded: tokens={token_count}/{self.max_tokens}, "
                f"requests={num_seqs}/{self.max_seqs}"
            )
        if query_start_loc.numel() < num_seqs + 1:
            raise ValueError("PLE query_start_loc is shorter than request metadata")
        self._query_start_loc.zero_()
        self._query_start_loc[: num_seqs + 1].copy_(
            query_start_loc[: num_seqs + 1].to(torch.int32)
        )
        self._state_slot_ids.fill_(NULL_BLOCK_ID)
        self._state_is_fresh.fill_(True)
        self._num_accepted_tokens.fill_(1)
        self._request_is_prefill.zero_()

        num_decodes = int(metadata.num_decodes)
        num_prefills = int(metadata.num_prefills)
        if num_decodes:
            state_d = metadata.state_indices_tensor_d
            if state_d is None:
                raise RuntimeError("decode PLE metadata is missing state indices")
            if state_d.ndim == 2:
                state_d = state_d[:, 0]
            self._state_slot_ids[:num_decodes].copy_(state_d[:num_decodes])
            self._state_is_fresh[:num_decodes] = False
            if metadata.num_accepted_tokens is not None:
                self._num_accepted_tokens[:num_decodes].copy_(
                    metadata.num_accepted_tokens[:num_decodes].to(torch.int32)
                )
        if num_prefills:
            state_p = metadata.state_indices_tensor_p
            if state_p is None:
                raise RuntimeError("prefill PLE metadata is missing state indices")
            start = num_decodes
            self._state_slot_ids[start : start + num_prefills].copy_(
                state_p[:num_prefills]
            )
            has_initial = metadata.has_initial_states_p
            if has_initial is None:
                raise RuntimeError("prefill PLE metadata is missing fresh-state flags")
            self._state_is_fresh[start : start + num_prefills].copy_(
                ~has_initial[:num_prefills]
            )
            self._request_is_prefill[start : start + num_prefills] = True

        self._num_seqs.fill_(num_seqs)
        self._num_tokens.copy_(query_start_loc[num_seqs : num_seqs + 1])

    def _run_ple(
        self,
        residual: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        token_count = residual.shape[0]
        forward_context = get_forward_context()
        raw_metadata = forward_context.attn_metadata
        metadata = (
            raw_metadata.get(self.prefix) if isinstance(raw_metadata, dict) else None
        )
        if metadata is None:
            # Profiling runs before cache allocation and exist only to size the
            # remaining cache. Preserve a live dataflow without pretending to
            # mutate serving state.
            self._out.zero_()
            self._out[:token_count].copy_(value[:, None, :].expand_as(residual))
            return
        if not isinstance(metadata, ShortConvAttentionMetadata):
            raise TypeError(
                f"expected ShortConvAttentionMetadata for {self.prefix}, got "
                f"{type(metadata).__name__}"
            )
        if self._plan is None:
            raise RuntimeError("PLE KV cache was not bound before inference")
        self._residual[:token_count].copy_(residual)
        self._key[:token_count].copy_(key)
        self._value[:token_count].copy_(value)
        self._prepare_metadata(metadata, query_start_loc, token_count)
        _b12x_module("ple").run_mixed(
            self._bind_ple(), eps=self.eps, token_count=token_count
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        *,
        prefetched: bool = False,
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        if input_ids.numel() != token_count:
            raise ValueError(
                "PLE input_ids and hidden states must have the same token count"
            )
        embeddings = self.ple_embedding(
            input_ids,
            query_start_loc,
            ngram_context,
            wait_for=hidden_states if prefetched else None,
        )
        key = self.key_proj(embeddings).reshape(
            token_count, self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        residual = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        if torch.compiler.is_compiling():
            torch.ops.vllm.qwen3_8_flash_next_ple(
                residual,
                key,
                value,
                query_start_loc,
                self._out,
                self.prefix,
            )
        else:
            self._run_ple(residual, key, value, query_start_loc)
        return self._out[:token_count].flatten(-2)


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


def _ple_prefetch_op(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.ple_embedding._run_prefetch(input_ids, query_start_loc, ngram_context)


def _ple_prefetch_wait_op(out: torch.Tensor, dependency: torch.Tensor) -> None:
    # The dependency places this join after the preceding layer's computation.
    if dependency.shape[0] <= 16 and torch.cuda.is_current_stream_capturing():
        current_stream().wait_stream(_get_prefetch_stream())


def _ple_prefetch_wait_fake(out: torch.Tensor, dependency: torch.Tensor) -> None:
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
    op_name="qwen3_8_flash_next_ple_embedding",
    op_func=_ple_embedding_op,
    mutates_args=["out"],
    fake_impl=_ple_embedding_fake,
)
direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_prefetch",
    op_func=_ple_prefetch_op,
    mutates_args=["out"],
    fake_impl=_ple_embedding_fake,
)
direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_prefetch_wait",
    op_func=_ple_prefetch_wait_op,
    mutates_args=["out"],
    fake_impl=_ple_prefetch_wait_fake,
)
direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple",
    op_func=_ple_op,
    mutates_args=["out"],
    fake_impl=_ple_fake,
)


__all__ = [
    "Qwen3_8FlashNextNGramEmbedding",
    "Qwen3_8FlashNextPLEGroupedNorm",
    "Qwen3_8FlashNextPLELayer",
]
