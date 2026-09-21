# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in load-time checkpoint-to-parameter coverage, without tensor retention."""

import json
from contextlib import contextmanager
from pathlib import Path

from vllm.model_executor.model_loader.weight_utils import default_weight_loader


@contextmanager
def tensor_coverage(model, weights, path):
    """Record actual parameter loader calls for untransformed checkpoint tensors.

    This diagnostic fails on transformed/unattributable inputs. It does not
    infer successful loading merely because a checkpoint iterator yielded a
    tensor. Original parameter callbacks are restored before postprocessing.
    """
    records, origins, restored = {}, {}, []

    def key(tensor):
        return (tensor.data_ptr(), tensor.numel(), str(tensor.dtype))

    def traced_weights():
        for name, tensor in weights:
            if name in records:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            records[name] = dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                bytes=tensor.numel() * tensor.element_size(),
                destinations=[],
            )
            origins[key(tensor)] = name
            yield name, tensor

    def wrapper(destination, loader):
        def load(*args, **kwargs):
            parameter = kwargs.get("param", args[0] if args else None)
            tensor = kwargs.get("loaded_weight", args[1] if len(args) > 1 else None)
            if parameter is None or tensor is None:
                raise ValueError(
                    f"Missing parameter or checkpoint tensor for {destination}"
                )
            origin = origins.get(key(tensor))
            if origin is None:
                raise ValueError(f"Cannot attribute checkpoint input for {destination}")
            result = loader(*args, **kwargs)
            records[origin]["destinations"].append(
                dict(
                    parameter=destination,
                    device=str(parameter.device),
                    expert_id=kwargs.get("expert_id"),
                    shard_id=kwargs.get("shard_id", args[2] if len(args) > 2 else None),
                    accepted=result is not False,
                )
            )
            return result

        return load

    status = "failed"
    try:
        for name, parameter in model.named_parameters():
            original = getattr(parameter, "weight_loader", None)
            restored.append((parameter, original))
            parameter.weight_loader = wrapper(name, original or default_weight_loader)
        yield traced_weights()
        status = "completed"
    finally:
        for parameter, original in restored:
            if original is None:
                del parameter.weight_loader
            else:
                parameter.weight_loader = original
        # Never overwrite a failed attempt or another worker's receipt.
        with Path(path).open("x") as stream:
            json.dump(dict(status=status, tensors=records), stream)
            stream.write("\n")
