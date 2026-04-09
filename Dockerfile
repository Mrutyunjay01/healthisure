FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first — layer is only invalidated when these change
COPY pyproject.toml uv.lock ./

# Install dependencies using BuildKit cache mount for uv's package cache.
# --mount=type=cache persists the cache between builds (faster rebuilds)
# but is NOT written into the final image layer (smaller image).
# --no-install-project skips installing the local package (not needed at runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy application code (separate layer so dep installs are not re-run on code changes)
COPY server/ ./server/
COPY models.py client.py inference.py ./

# Remove bytecode cache that uv/pip may have written during install
RUN find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name "*.pyc" -delete 2>/dev/null || true

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Enable the Gradio web playground at /web for Hugging Face Spaces
ENV ENABLE_WEB_INTERFACE=true

# HuggingFace Spaces injects PORT=7860; default to 8000 for local runs
ENV PORT=8000
EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
