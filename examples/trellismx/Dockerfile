# Build the current PR using docker/Dockerfile first. Never overlay new Python
# onto RC5's old vLLM extension binaries and call that a Jovian port.
ARG JOVIAN_IMAGE
FROM ${JOVIAN_IMAGE} AS dependency
RUN uv pip install --system --no-deps \
      "b12x @ https://github.com/brandonmmusic-max/b12x/archive/d564f6ca54c092497ec5ae7e07a272f55eec7dbe.tar.gz"

# Explicit r27 source build; requires a core image with the r27 backend launcher.
FROM dependency AS r27
COPY examples/trellismx/serve-r27.sh /opt/trellismx/serve.sh
ENTRYPOINT ["/bin/bash", "/opt/trellismx/serve.sh"]

# Keep the default source-build recipe on its documented DCP1 launcher.
FROM dependency AS default
COPY examples/trellismx/serve.sh /opt/trellismx/serve.sh
ENTRYPOINT ["/bin/bash", "/opt/trellismx/serve.sh"]
