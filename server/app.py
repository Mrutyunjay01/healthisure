"""
FastAPI application for Healthisure.

Exposes the OpenEnv standard endpoints:
  POST /reset   — start a new episode
  POST /step    — execute an action
  GET  /state   — inspect current state
  GET  /schema  — action/observation JSON schemas
  GET  /health  — liveness check
  WS   /ws      — WebSocket for persistent stateful sessions (preferred for agents)
  GET  /web     — Gradio web UI (enabled when ENABLE_WEB_INTERFACE=true)

Set ENABLE_WEB_INTERFACE=true (or 1) to mount the interactive Gradio playground
at /web — useful for Hugging Face Spaces so users can try the environment in browser.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from .healthisure_environment import HealthisureEnvironment
from .gradio_app import build_healthisure_gradio_app
from models import HealthisureAction, HealthisureObservation


app = create_app(
    HealthisureEnvironment,
    HealthisureAction,
    HealthisureObservation,
    env_name="healthisure",
    max_concurrent_envs=10,
    gradio_builder=build_healthisure_gradio_app,
)


def main(host: str = "0.0.0.0", port: int | None = None):
    import os
    import uvicorn
    resolved_port = port if port is not None else int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
