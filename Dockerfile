# uv, pinned by digest; a FROM line so Dependabot can propose updates.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

# ubuntu 24.04, pinned by multi-arch index digest so a rebuild at the same commit starts from the same base.
# Dependabot (docker ecosystem) proposes updates.
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac AS base

# ffmpeg: the same pinned BtbN static GPL build n9.0.2 that Linux CI uses, verified by SHA256.
# Keep FFMPEG_LINUX_URL and FFMPEG_LINUX_SHA256 identical to the env of the test job in
# .github/workflows/ci.yml (the docker job fails if they differ). To bump, change both places.
# Downloaded in its own stage so curl and xz stay out of the runtime image.
FROM base AS ffmpeg
ARG FFMPEG_LINUX_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-25-13-00/ffmpeg-n9.0.2-36-g6882969ce9-linux64-gpl-9.0.tar.xz
ARG FFMPEG_LINUX_SHA256=fdcc15b1ba538ee94c087c1ce9c1efe166cfe392beb362ccb778f595065365cc
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# BtbN prunes old autobuild tags. A failed download (404, network) leaves /tmp/ffmpeg/bin empty and the
# final stage then installs Ubuntu's apt ffmpeg instead; a checksum mismatch still fails the build.
RUN mkdir -p /tmp/ffmpeg/bin \
    && code=$(curl -sSL --retry 3 -w '%{http_code}' -o /tmp/ffmpeg.tar.xz "$FFMPEG_LINUX_URL") || code=000; \
    if [ "$code" = 200 ]; then \
        echo "$FFMPEG_LINUX_SHA256  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
        && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
        && /tmp/ffmpeg/bin/ffmpeg -version | head -1 \
        && /tmp/ffmpeg/bin/ffprobe -version >/dev/null \
        && rm -f /tmp/ffmpeg/bin/ffplay; \
    elif [ "$code" = 404 ]; then \
        echo "WARNING: static ffmpeg not found (HTTP 404, tag pruned?) at $FFMPEG_LINUX_URL; falling back to apt ffmpeg" >&2; \
    else \
        echo "ffmpeg download failed (HTTP $code)" >&2; exit 1; \
    fi

FROM base
ENV DEBIAN_FRONTEND=noninteractive
# libgl1, libglib2.0-0t64, libxcb1: needed by the opencv-python wheel (cv2 import); fontconfig and
# fonts-dejavu-core: burned-in captions (libass); libgomp1: llama-server downloaded by `cutpoint setup`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0t64 libxcb1 fontconfig fonts-dejavu-core libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg /tmp/ffmpeg/bin/ /usr/local/bin/
# Fallback when the static download failed in the ffmpeg stage.
RUN [ -x /usr/local/bin/ffmpeg ] || (apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*)
# uv from the first stage.
COPY --from=uv /uv /uvx /usr/local/bin/
# Managed Python lives outside /root so the non-root user can run the venv.
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE AGENTS.md ./
RUN uv sync --locked --no-install-project --no-default-groups && uv cache clean
ARG INSTALL_EVENTS=false
RUN if [ "$INSTALL_EVENTS" = true ]; then uv sync --locked --no-install-project --no-default-groups --extra events && uv cache clean; fi

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
