# uv 0.13.5, pinned by digest; a FROM line so Dependabot can propose updates.
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

# ubuntu 24.04, pinned by multi-arch index digest so a rebuild at the same commit starts from the same base.
# Dependabot (docker ecosystem) proposes updates.
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# uv 0.13.5 (the first stage), pinned by digest so a rebuild at the same commit installs the same uv.
COPY --from=uv /uv /uvx /usr/local/bin/
# Managed Python lives outside /root so the non-root user can run the venv.
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE AGENTS.md ./
RUN uv sync --locked --no-install-project --no-dev --extra ui && uv cache clean
ARG INSTALL_EVENTS=false
RUN if [ "$INSTALL_EVENTS" = true ]; then uv sync --locked --no-install-project --no-dev --extra ui --extra events && uv cache clean; fi

COPY src ./src
RUN uv pip install --no-deps --editable . && uv cache clean

# Non-root runtime user. ubuntu:24.04 ships "ubuntu" (1000:1000); remove it so the ids are free.
# An existing group with the same gid (e.g. 100 "users") is reused.
# Override with --build-arg CUTPOINT_UID=$(id -u) --build-arg CUTPOINT_GID=$(id -g) to match the host.
ARG CUTPOINT_UID=1000
ARG CUTPOINT_GID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (getent group "$CUTPOINT_GID" >/dev/null || groupadd --gid "$CUTPOINT_GID" cutpoint) \
    && useradd --no-log-init --uid "$CUTPOINT_UID" --gid "$CUTPOINT_GID" --create-home --shell /usr/sbin/nologin cutpoint \
    && mkdir -p /data/huggingface \
    && chown -R "$CUTPOINT_UID:$CUTPOINT_GID" /data

# NUMBA_CACHE_DIR: librosa/numba needs a writable cache when run with --user <uid> (HOME is /).
ENV CUTPOINT_HOME=/data HF_HOME=/data/huggingface CUTPOINT_LLAMA_BACKEND=cpu NUMBA_CACHE_DIR=/tmp/numba-cache
VOLUME /data
USER cutpoint
ENTRYPOINT ["/app/.venv/bin/cutpoint"]
CMD ["ui"]
