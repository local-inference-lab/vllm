# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-sharded Kimi router and latent projections with zero-filled tails."""

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce_in_place,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.weight_transfer import copy_weight
from vllm.utils.math_utils import cdiv


def enable_kimi_projection_tail_padding(layer: nn.Module) -> None:
    """Allow checkpoint-absent TP tails in explicitly padded Kimi projections."""
    for parameter in layer.parameters(recurse=False):
        parameter.allow_tp_padding = True


def can_reuse_projection_output(layer, output: torch.Tensor) -> bool:
    """Qualify consumed prefill storage for an unquantized row projection."""
    weight = getattr(layer, "weight", None)
    return (
        isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod)
        and weight is not None
        and layer.input_is_parallel
        and layer.bias is None
        and not envs.VLLM_BATCH_INVARIANT
        and not torch.is_grad_enabled()
        and output.ndim == 2
        and output.shape[0] >= 1024
        and output.shape[1] == layer.output_size
        and output.is_contiguous()
        and output.dtype == weight.dtype
        and output.device == weight.device
    )


def project_into_consumed_output(layer, activation, output):
    """Project into donated storage, retaining decode's functional path."""
    if not can_reuse_projection_output(layer, output):
        raise ValueError(
            "Kimi projection output violates the prefill donation contract"
        )
    if (
        activation.shape != (output.shape[0], layer.input_size_per_partition)
        or activation.dtype != output.dtype
        or activation.device != output.device
        or activation.untyped_storage().data_ptr()
        == output.untyped_storage().data_ptr()
    ):
        raise ValueError(
            "Kimi projection input must be compatible and disjoint from output"
        )
    torch.mm(activation, layer.weight.T, out=output)
    if layer.reduce_results and layer.tp_size > 1:
        tensor_model_parallel_all_reduce_in_place(output)
    return output


def _load_padded_tp_shard(param, weight, dim, rank):
    """Copy the checkpoint shard and explicitly zero its unrepresented tail."""
    width = param.shape[dim]
    start = rank * width
    available = max(0, min(weight.shape[dim] - start, width))
    if available == width:
        source = weight.narrow(dim, start, width)
    else:
        shape = list(weight.shape)
        shape[dim] = width
        source = weight.new_zeros(shape)
        if available:
            source.narrow(dim, 0, available).copy_(weight.narrow(dim, start, available))
    if param.shape != source.shape:
        raise ValueError(
            f"Kimi TP shard shape {tuple(param.shape)} does not match "
            f"checkpoint slice {tuple(source.shape)}"
        )
    copy_weight(param.data, source)


def prepare_aligned_decode_projection(layer: nn.Module) -> None:
    """Retain an aligned BF16 up-projection without changing prefill storage."""
    if not isinstance(layer, KimiPaddedRowParallelLinear) or not isinstance(
        layer.quant_method, UnquantizedLinearMethod
    ):
        return
    weight = layer.weight
    if not weight.is_cuda or weight.dtype != torch.bfloat16:
        return
    width = weight.shape[1]
    aligned = cdiv(width, 16) * 16
    if aligned == width:
        return
    shape = list(weight.shape)
    shape[1] = aligned
    storage = weight.new_zeros(shape)
    storage[:, :width].copy_(weight)
    layer.register_buffer("_aligned_decode_weight", storage, persistent=False)


def prepare_paired_decode_projection(down, gate):
    """Share aligned storage for a joint BF16 latent/router GEMM with FP32 output."""
    if not isinstance(down, KimiPaddedColumnParallelLinear) or not isinstance(
        gate, KimiColumnParallelGate
    ):
        return None
    if (
        not down.weight.is_cuda
        or down.weight.dtype != torch.bfloat16
        or gate.weight.dtype != torch.bfloat16
        or down.weight.device != gate.weight.device
        or down.weight.shape[1] != gate.weight.shape[1]
        or down.weight.shape[0] % 16 == 0
    ):
        return None
    down_width, gate_width = down.weight.shape[0], gate.weight.shape[0]
    width = down_width + gate_width
    storage = down.weight.new_zeros(cdiv(width, 16) * 16, down.weight.shape[1])
    storage[:down_width].copy_(down.weight)
    storage[down_width:width].copy_(gate.weight)
    # Logical checkpoint shards retain contiguous views and loader metadata.
    down.weight.data = storage[:down_width]
    gate.weight.data = storage[down_width:width]
    return storage


def paired_decode_projection(x, weight, down_width, gate_width):
    output = torch.mm(x, weight.T, out_dtype=torch.float32)
    down = output[:, :down_width].to(torch.bfloat16)
    gate = output[:, down_width : down_width + gate_width].contiguous()
    return down, gate


def _aligned_decode_weight(layer, x):
    weight = getattr(layer, "_aligned_decode_weight", None)
    if (
        weight is not None
        and not envs.VLLM_BATCH_INVARIANT
        and x.ndim == 2
        and 1 < x.shape[0] <= 8
        and x.device == weight.device
        and x.dtype == torch.bfloat16
    ):
        return weight
    return None


class KimiPaddedColumnParallelLinear(ColumnParallelLinear):
    """Gather a feature-sharded projection and strip checkpoint-absent rows."""

    def __init__(self, input_size, output_size, prefix, *, gather_output=True):
        tp_size = get_tensor_model_parallel_world_size()
        self.logical_output_size = output_size
        self.kimi_gather_output = gather_output
        super().__init__(
            input_size,
            cdiv(output_size, tp_size) * tp_size,
            bias=False,
            gather_output=False,
            quant_config=None,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        dim = getattr(param, "output_dim", None)
        if dim is None or getattr(param, "is_sharded_weight", False):
            default_weight_loader(param, loaded_weight)
        else:
            _load_padded_tp_shard(param, loaded_weight, dim, self.tp_rank)

    weight_loader_v2 = weight_loader

    def forward_local(self, x):
        return super().forward(x)

    def forward(self, x):
        output, bias = self.forward_local(x)
        if self.kimi_gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output, dim=-1)
        return output[..., : self.logical_output_size].contiguous(), bias


class KimiColumnParallelGate(KimiPaddedColumnParallelLinear):
    """Compute FP32 router logits with feature-sharded BF16 weights."""

    def forward_local(self, x):
        if x.is_cuda and x.dtype == self.weight.dtype == torch.bfloat16:
            return torch.mm(x, self.weight.T, out_dtype=torch.float32), None
        return torch.nn.functional.linear(
            x.to(self.weight.dtype), self.weight
        ).float(), None


class KimiPaddedRowParallelLinear(RowParallelLinear):
    """Return rank-local latent up-projection partials for a shared reduction."""

    def __init__(self, input_size, output_size, prefix):
        tp_size = get_tensor_model_parallel_world_size()
        padded_input_size = cdiv(input_size, tp_size) * tp_size
        self.input_pad = padded_input_size - input_size
        super().__init__(
            padded_input_size,
            output_size,
            bias=False,
            input_is_parallel=False,
            reduce_results=False,
            quant_config=None,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        dim = getattr(param, "input_dim", None)
        if dim is None or getattr(param, "is_sharded_weight", False):
            default_weight_loader(param, loaded_weight)
        else:
            _load_padded_tp_shard(param, loaded_weight, dim, self.tp_rank)

    weight_loader_v2 = weight_loader

    def forward(self, x):
        weight = _aligned_decode_weight(self, x)
        if weight is not None:
            start = self.tp_rank * self.input_size_per_partition
            available = max(0, min(x.shape[1] - start, self.input_size_per_partition))
            local = x.new_zeros(x.shape[0], weight.shape[1])
            if available:
                local[:, :available].copy_(x[:, start : start + available])
            return torch.nn.functional.linear(local, weight), None
        if self.input_pad:
            x = torch.nn.functional.pad(x, (0, self.input_pad))
        return super().forward(x)

    def forward_into(self, x, output):
        """Write a rank-local prefill partial into disjoint caller storage."""
        if (
            not isinstance(self.quant_method, UnquantizedLinearMethod)
            or self.bias is not None
            or self.reduce_results
            or envs.VLLM_BATCH_INVARIANT
            or torch.is_grad_enabled()
            or x.ndim != 2
            or x.shape[1] != self.input_size - self.input_pad
            or output.shape != (x.shape[0], self.output_size)
            or not output.is_contiguous()
            or x.dtype != output.dtype
            or output.dtype != self.weight.dtype
            or x.device != output.device
            or output.device != self.weight.device
            or x.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "Kimi partial projection requires disjoint compatible output"
            )
        if self.input_pad:
            x = torch.nn.functional.pad(x, (0, self.input_pad))
        local = x.narrow(
            -1,
            self.tp_rank * self.input_size_per_partition,
            self.input_size_per_partition,
        ).contiguous()
        torch.mm(local, self.weight.T, out=output)
        return output, None
