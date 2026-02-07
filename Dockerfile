FROM python/uv:bookworm AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

COPY . .

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app /app
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
EXPOSE 8001

CMD ["bash", "start.sh"]
