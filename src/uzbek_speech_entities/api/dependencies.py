"""Thin FastAPI dependency accessors for configured application services."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from ..pipeline.analyzer import SpeechEntityAnalyzer


def get_analyzer(request: Request) -> SpeechEntityAnalyzer:
    """Return the application-scoped analyzer; model work never happens here."""
    return cast(SpeechEntityAnalyzer, request.app.state.analyzer)
