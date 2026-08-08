"""Configured Uvicorn entry point for the local application."""

from __future__ import annotations

import os

import uvicorn

from ..config import load_config
from .app import app


def server_address() -> tuple[str, int]:
    """Return validated config values with explicit environment overrides."""
    values = load_config().section("app")
    configured_host = values.get("host")
    configured_port = values.get("port")
    host = os.getenv("APP_HOST", configured_host if isinstance(configured_host, str) else "")
    raw_port = os.getenv("APP_PORT", str(configured_port))
    if not host.strip():
        raise ValueError("app.host must be non-empty text")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ValueError("APP_PORT/app.port must be an integer") from error
    if isinstance(configured_port, bool) or not 1 <= port <= 65_535:
        raise ValueError("APP_PORT/app.port must be in [1, 65535]")
    return host, port


def main() -> None:
    """Run the already-created application on its configured local address."""
    host, port = server_address()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
