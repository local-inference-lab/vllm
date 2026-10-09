# Spark disk-backed Engram

`spark-io-uring.json` is the [Moby default seccomp profile](https://github.com/moby/profiles/blob/245180c51918481c0525424b3ee025d2b435d46c/seccomp/default.json)
with one additional allow rule for `io_uring_setup`, `io_uring_enter`, and
`io_uring_register`. The remaining rules are unchanged. Moby profiles are
licensed under Apache-2.0.

The DeepSeek V4.1 TP4 launcher installs this profile on each Spark for b12x's
disk-backed Engram tables. It applies to those containers only. Override
`SECCOMP_PROFILE` to use another profile that permits the required calls.

The container also needs liburing headers and `pkg-config` so the native loader
builds with disk-table support. Build the V4.1 launcher's default image from the
existing Spark image:

```bash
docker build -f docker/Dockerfile.spark-io-uring \
  -t vllm-node-eugr-20260712-io-uring:latest .
```

Copy that image to every worker with `docker save` / `docker load` (or the Spark
project's `build-and-copy.sh --no-build`). The cluster launcher requires identical
image IDs; independent builds can produce different IDs even with the same
Dockerfile and package versions.
