"""Privacy-preserving logging helpers for the local application."""

from __future__ import annotations

import logging
import os

from .constants import DIAGNOSTIC_TRANSCRIPT_LOGGING_ENV


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure process logging without adding transcript or audio handlers."""
    selected_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=selected_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("uzbek_speech_entities")


def diagnostic_transcript_logging_enabled() -> bool:
    """Return whether explicitly opted-in diagnostic transcript logging is enabled."""
    return os.getenv(DIAGNOSTIC_TRANSCRIPT_LOGGING_ENV, "false").lower() in {"1", "true", "yes"}
