ARG BASE_IMAGE=glm53-jovian-lmcache:local
FROM ${BASE_IMAGE}

USER root

COPY lmcache-d16-overlay/overlay/ /opt/glm53-lmcache-overlay/
COPY lmcache-d16-overlay/cumem_shareable_interposer.c /tmp/
COPY single-container/vllm-startup-overlay/ /opt/glm53-vllm-startup-overlay/
COPY single-container/serve_glm53_with_lmcache.py /usr/local/bin/
COPY single-container/healthcheck_glm53_vllm.py /usr/local/bin/

RUN set -eux; \
    python_dir="$(/opt/venv/bin/python -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    cp -a /opt/glm53-lmcache-overlay/lmcache/. "${python_dir}/lmcache/"; \
    cp -a /opt/glm53-vllm-startup-overlay/vllm/. "${python_dir}/vllm/"; \
    cc -shared -fPIC -O2 -ldl /tmp/cumem_shareable_interposer.c \
        -o /usr/local/lib/liblmcache-cumem-shareable.so; \
    chmod 0755 /usr/local/bin/serve_glm53_with_lmcache.py \
        /usr/local/bin/healthcheck_glm53_vllm.py; \
    rm -rf /opt/glm53-lmcache-overlay /opt/glm53-vllm-startup-overlay \
        /tmp/cumem_shareable_interposer.c

ENV GLM53_LMCACHE_EXECUTABLE=/opt/venv/bin/lmcache \
    GLM53_VLLM_EXECUTABLE=/opt/venv/bin/vllm \
    LD_PRELOAD=/usr/local/lib/liblmcache-cumem-shareable.so

ENTRYPOINT ["/usr/local/bin/serve_glm53_with_lmcache.py"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=3600s --retries=3 \
    CMD ["/usr/local/bin/healthcheck_glm53_vllm.py"]
