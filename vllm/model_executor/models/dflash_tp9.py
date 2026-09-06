# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uneven projections for the 32-query/8-KV Kimi DFlash draft on nine ranks."""

from dataclasses import dataclass

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear


@dataclass(frozen=True)
class DFlashHeadExtent:
    first_query: int
    query_count: int
    kv_head: int


def dflash_tp9_head_extent(layer: int, rank: int) -> DFlashHeadExtent:
    """Split one GQA group into two pairs sharing the same actual KV head."""
    if not 0 <= layer < 6 or not 0 <= rank < 9:
        raise ValueError("Kimi DFlash TP9 requires layer 0..5 and rank 0..8")
    owner = (rank - 2 * layer) % 9
    if owner < 2:
        return DFlashHeadExtent(2 * owner, 2, 0)
    return DFlashHeadExtent(4 * (owner - 1), 4, owner - 1)


def aligned_tp9_extent(size: int, rank: int, alignment: int) -> tuple[int, int]:
    if size <= 0 or alignment <= 0 or size % alignment or not 0 <= rank < 9:
        raise ValueError("TP9 extent requires an aligned positive size and rank 0..8")
    quotient, remainder = divmod(size // alignment, 9)
    first = (rank * quotient + min(rank, remainder)) * alignment
    count = (quotient + int(rank < remainder)) * alignment
    return first, count


class DFlashTP9QKVParallelLinear(QKVParallelLinear):
    """Load Q/K/V by source head identity into rank-local projection storage."""

    def __init__(self, hidden_size, head_size, extent, *, quant_config, prefix):
        self.source_extent = extent
        super().__init__(
            hidden_size,
            head_size,
            extent.query_count,
            1,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )

    def _load_shard(self, loader, param, loaded_weight, shard_id):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            return loader(param, loaded_weight, shard_id)
        if getattr(param, "packed_dim", None) == output_dim:
            raise ValueError("TP9 DFlash does not support output-packed QKV weights")
        if shard_id is None:
            if loaded_weight.shape[output_dim] != 48 * self.head_size:
                raise ValueError("fused Kimi DFlash QKV must contain 32/8/8 heads")
            for name, first, count in (("q", 0, 32), ("k", 32, 8), ("v", 40, 8)):
                source = loaded_weight.narrow(
                    output_dim, first * self.head_size, count * self.head_size
                )
                self._load_shard(loader, param, source, name)
            return None
        if shard_id not in ("q", "k", "v"):
            raise ValueError(f"invalid QKV shard {shard_id!r}")
        extent = self.source_extent
        first = extent.first_query if shard_id == "q" else extent.kv_head
        count = extent.query_count if shard_id == "q" else 1
        source_heads = 32 if shard_id == "q" else 8
        if loaded_weight.shape[output_dim] != source_heads * self.head_size:
            raise ValueError("Kimi DFlash QKV source width disagrees with its heads")
        source = loaded_weight.narrow(
            output_dim, first * self.head_size, count * self.head_size
        )
        return loader(param, source, shard_id)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        return self._load_shard(
            super().weight_loader, param, loaded_weight, loaded_shard_id
        )

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        return self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )


class DFlashTP9RowParallelLinear(RowParallelLinear):
    """Reduce a projection over explicit, possibly uneven input intervals.

    The inherited linear owns only a local matrix. Its TP state is deliberately
    disabled so loading cannot slice a second time; this class owns the single
    reduction over the real nine-rank group.
    """

    def __init__(
        self,
        source_input_size,
        output_size,
        first,
        count,
        *,
        input_is_full=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
        return_bias=True,
    ):
        if not 0 <= first < first + count <= source_input_size:
            raise ValueError("TP9 row projection lies outside the source input")
        self.source_input_size = source_input_size
        self.source_first = first
        self.source_count = count
        self.input_is_full = input_is_full
        super().__init__(
            count,
            output_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=False,
            disable_tp=True,
            return_bias=return_bias,
        )

    def _source_weight(self, param, loaded_weight):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is None:
            return loaded_weight
        if getattr(param, "packed_dim", None) == input_dim:
            raise ValueError("TP9 DFlash does not support input-packed row weights")
        if loaded_weight.shape[input_dim] != self.source_input_size:
            raise ValueError("TP9 row projection source input width mismatch")
        return loaded_weight.narrow(input_dim, self.source_first, self.source_count)

    def weight_loader(self, param, loaded_weight):
        return super().weight_loader(param, self._source_weight(param, loaded_weight))

    def weight_loader_v2(self, param, loaded_weight):
        return super().weight_loader_v2(
            param, self._source_weight(param, loaded_weight)
        )

    def forward(self, input_):
        if self.input_is_full:
            input_ = input_.narrow(
                -1, self.source_first, self.source_count
            ).contiguous()
        local = super().forward(input_)
        if self.return_bias:
            output, bias = local
            return tensor_model_parallel_all_reduce(output), bias
        return tensor_model_parallel_all_reduce(local)
