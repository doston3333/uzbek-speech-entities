"""Application route modules."""

from .analyze_audio import router as analyze_audio_router
from .analyze_text import router as analyze_text_router
from .health import router as health_router

__all__ = ["analyze_audio_router", "analyze_text_router", "health_router"]
