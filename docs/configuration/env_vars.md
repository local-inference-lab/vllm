# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## Inter-process communication spin tuning

When vLLM runs across multiple processes (for example, tensor-parallel inference), the processes exchange data through shared memory. A reader process that checks for new data has two options: *spin* (keep checking on the CPU, for the lowest possible wake latency) or *park* (sleep until the writer notifies it, freeing the CPU). vLLM spins for a short grace period after each read and only parks once the grace expires.

Set `VLLM_EXPERIMENTAL_SHM_BROADCAST_ADAPTIVE_SPIN=1` to opt into adaptive reader and writer waits. It is experimental and disabled by default. With it disabled or unset, readers retain the fixed one-second grace and writers yield while waiting for readers to release a buffer block. Passing an explicit reader `busy_loop_s`, including `0`, always pins that reader grace and overrides the experimental policy.

When enabled, `VLLM_SHM_BROADCAST_ADAPTIVE_*` tunes the reader grace. vLLM measures the interval between reads and spins longer during bursts and parks sooner when traffic slows. The bounds (`MIN_GRACE` / `MAX_GRACE`) and the pivot (`BUDGET`) clamp this behavior. The writer uses the same adaptive policy while a reader holds a buffer block, then sleeps in steps that double from 50 microseconds up to `VLLM_SHM_BROADCAST_WRITE_PARK_MAX_MS` (1 ms by default). These adaptive tunables apply only when the experimental switch is enabled.

vLLM exports cumulative `vllm:shm_broadcast_blocked_waits_total` and `vllm:shm_broadcast_blocked_wait_seconds_total` metrics with a `role` label (`reader` or `writer`). Each queue queues an update every 128 blocked waits, or during shutdown. A background thread publishes the batches. The spin and park loops never call the metrics exporter.

`VLLM_USE_SPINLOOP_EXT` independently enables the native hardware wait extension. It remains available with both fixed and adaptive grace policies.

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
