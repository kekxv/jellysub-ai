FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsox-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project \
    && rm -rf \
        /app/.venv/lib/python*/site-packages/torch/test \
        /app/.venv/lib/python*/site-packages/torch/include \
        /app/.venv/lib/python*/site-packages/gradio \
        /app/.venv/lib/python*/site-packages/gradio_client

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_SOURCE="" \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    CONFIG_PATH=/data/config.json \
    TASK_DB_PATH=/data/tasks.db \
    MODEL_CACHE=/data/model_cache \
    TEMP_DIR=/data/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app \
    && mkdir -p /app /data/model_cache /data/tmp \
    && chown -R app:app /data

COPY --from=builder /app/.venv /app/.venv
COPY main.py config.py env_config.py /app/
COPY core /app/core
COPY static /app/static

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

USER app
WORKDIR /data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
