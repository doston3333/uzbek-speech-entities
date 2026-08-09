"""FastAPI factory and application lifespan for local model services."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..audio.validation import AudioValidationConfig
from ..config import AppConfig, frontend_directory, load_config
from ..ner.predictor import NERPredictor
from ..pipeline.analyzer import SpeechEntityAnalyzer
from ..stt.base import ModelLoadError
from ..stt.factory import create_stt_service
from .audio_request_limit import AudioRequestBodyLimitMiddleware
from .errors import install_error_handlers
from .routes import analyze_audio_router, analyze_text_router, health_router

LOGGER = logging.getLogger(__name__)


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _speech_rescue_enabled(config: AppConfig) -> bool:
    """Read the opt-in speech rescue flag, with a strict environment override."""
    configured = config.section("ner").get("speech_rescue_enabled", False)
    if not isinstance(configured, bool):
        raise ValueError("ner.speech_rescue_enabled must be boolean")
    raw = os.getenv("SPEECH_NER_RESCUE_ENABLED")
    if raw is None:
        return configured
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("SPEECH_NER_RESCUE_ENABLED must be a common boolean value")


def _speech_analysis_normalization_enabled(config: AppConfig) -> bool:
    """Read the explicit speech-only analysis-normalization flag."""
    configured = config.section("ner").get("analysis_normalization_enabled", False)
    if not isinstance(configured, bool):
        raise ValueError("ner.analysis_normalization_enabled must be boolean")
    raw = os.getenv("SPEECH_NER_ANALYSIS_NORMALIZATION_ENABLED")
    if raw is None:
        return configured
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("SPEECH_NER_ANALYSIS_NORMALIZATION_ENABLED must be a common boolean value")


def _normalized_confidence_threshold(config: AppConfig) -> float:
    value = config.section("ner").get("normalized_confidence_threshold", 0.70)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError("ner.normalized_confidence_threshold must be in [0, 1]")
    return float(value)


def create_app(
    *,
    analyzer: SpeechEntityAnalyzer | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    """Create an app whose default lifespan eagerly loads each local model once."""
    app_config = config or load_config()
    injected = analyzer is not None
    if analyzer is None:
        stt = create_stt_service(app_config)
        try:
            ner = NERPredictor.from_config(app_config)
        except ModelLoadError:
            LOGGER.warning(
                "NER model bootstrap failed; starting with local files only.",
                exc_info=True,
            )
            ner = NERPredictor.from_config(app_config, local_files_only=True)
        analyzer = SpeechEntityAnalyzer(
            stt_service=stt,
            ner_predictor=ner,
            audio_config=AudioValidationConfig.from_mapping(app_config.section("audio")),
            speech_rescue_enabled=_speech_rescue_enabled(app_config),
            analysis_normalization_enabled=_speech_analysis_normalization_enabled(app_config),
            normalized_confidence_threshold=_normalized_confidence_threshold(app_config),
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not injected:
            if analyzer.stt_service is not None:
                try:
                    analyzer.stt_service.load()
                except ModelLoadError:
                    # Health reports readiness; text analysis may remain usable.
                    LOGGER.warning("STT model load failed; reporting unavailable.", exc_info=True)
            try:
                analyzer.ner_predictor.load()
            except ModelLoadError:
                # Health intentionally reports readiness instead of aborting the server.
                LOGGER.warning("NER model load failed; reporting unavailable.", exc_info=True)
        yield

    title = str(app_config.section("app").get("name", "Uzbek Speech Entity Extractor"))
    application = FastAPI(title=title, lifespan=lifespan)
    application.state.analyzer = analyzer
    application.state.inference_lock = asyncio.Lock()
    application.state.mps_available = _mps_available()
    install_error_handlers(application)
    application.include_router(health_router, prefix="/api")
    application.include_router(analyze_text_router, prefix="/api")
    application.include_router(analyze_audio_router, prefix="/api")
    application.add_middleware(
        AudioRequestBodyLimitMiddleware,
        max_file_bytes=analyzer.audio_config.max_upload_bytes,
    )
    web_directory = frontend_directory()
    application.mount("/assets", StaticFiles(directory=web_directory), name="assets")

    @application.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(
            web_directory / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
                    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                ),
                "Permissions-Policy": "microphone=(self)",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    return application


app = create_app()
