FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH" UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE AGENTS.md ./
RUN uv sync --locked --no-install-project --extra ui && uv cache clean
ARG INSTALL_EVENTS=false
RUN if [ "$INSTALL_EVENTS" = true ]; then uv sync --locked --no-install-project --extra ui --extra events && uv cache clean; fi

COPY src ./src
RUN uv pip install --no-deps --editable . && uv cache clean

ENV CUTPOINT_HOME=/data HF_HOME=/data/huggingface
ENTRYPOINT ["/app/.venv/bin/cutpoint"]
CMD ["ui"]
