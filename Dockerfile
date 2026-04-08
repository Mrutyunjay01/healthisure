FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies from lockfile (no project install yet)
RUN uv sync --frozen --no-install-project --no-cache

# Copy application code
COPY server/ ./server/
COPY models.py client.py inference.py ./

ENV PYTHONPATH=/app

# Activate uv virtual environment for CMD
ENV PATH="/app/.venv/bin:$PATH"

# Enable the Gradio web playground at /web for Hugging Face Spaces
ENV ENABLE_WEB_INTERFACE=true

# Default port
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
