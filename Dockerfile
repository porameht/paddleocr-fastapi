FROM python/uv:bookworm AS builder

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv python install 3.12
RUN uv sync --frozen --no-dev --python 3.12

COPY . .

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app /app
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
EXPOSE 8000

CMD ["bash", "start.sh"]
