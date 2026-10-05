# Build: docker build -t scrm .
# Run:   docker run -p 8080:8080 --env-file .env scrm
# Credentials come from the runtime (Cloud Run service account / ADC), never the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "/app/.venv/bin/uvicorn scrm.api:app --host 0.0.0.0 --port ${PORT}"]
