#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    size_mb = int(os.environ.get("NCCL_SMOKE_SIZE_MB", "128"))
    iterations = int(os.environ.get("NCCL_SMOKE_ITERATIONS", "8"))
    numel = size_mb * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()

    torch.accelerator.set_device_index(0)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=150),
    )
    tensor = torch.empty(numel, device="cuda", dtype=torch.float32)

    for _ in range(2):
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)
    torch.accelerator.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iterations):
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)
    torch.accelerator.synchronize()
    elapsed = time.perf_counter() - start

    expected = world_size * (world_size + 1) / 2
    values = tensor[:16].cpu()
    if not torch.all(values == expected):
        raise RuntimeError(
            f"all-reduce returned {values.tolist()}, expected {expected}"
        )

    payload_bytes = tensor.numel() * tensor.element_size() * iterations
    result = {
        "backend": dist.get_backend(),
        "elapsed_seconds": elapsed,
        "payload_gb_per_second": payload_bytes / elapsed / 1e9,
        "rank": rank,
        "size_mb": size_mb,
        "world_size": world_size,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
