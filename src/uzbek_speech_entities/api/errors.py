"""Sanitized API error responses and exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ..audio.loader import AudioDecodingError
from ..audio.validation import AudioValidationError
from ..ner.spans import NERPredictionError
from ..pipeline.analyzer import TextValidationError
from ..stt.base import ModelLoadError, TranscriptionError


class ErrorDetail(BaseModel):
    """One sanitized machine-readable API error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Stable error envelope returned by every exception handler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorDetail


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Return the fixed, path-free error envelope used by every API error."""
    response = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json"),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Install typed exception mappings without exposing exception details."""

    @app.exception_handler(RequestValidationError)
    async def request_validation(_: Request, __: RequestValidationError) -> JSONResponse:
        return error_response(422, "invalid_request", "Request data is invalid.")

    @app.exception_handler(TextValidationError)
    async def text_validation(_: Request, __: TextValidationError) -> JSONResponse:
        return error_response(422, "invalid_text", "Text must be non-empty and within the limit.")

    @app.exception_handler(AudioValidationError)
    async def audio_validation(_: Request, __: AudioValidationError) -> JSONResponse:
        return error_response(400, "invalid_audio", "Audio file is invalid.")

    @app.exception_handler(AudioDecodingError)
    async def audio_decode(_: Request, __: AudioDecodingError) -> JSONResponse:
        return error_response(400, "invalid_audio", "Audio file could not be decoded.")

    async def model_unavailable(_: Request, __: Exception) -> JSONResponse:
        return error_response(503, "model_unavailable", "A required model is unavailable.")

    app.add_exception_handler(ModelLoadError, model_unavailable)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        mappings = {
            "empty_audio": (400, "invalid_audio", "Audio file is empty."),
            "unsupported_file": (422, "invalid_request", "Audio upload type is unsupported."),
            "upload_too_large": (413, "upload_too_large", "Audio file exceeds the upload limit."),
        }
        status, code, message = mappings.get(
            str(error.detail), (error.status_code, "invalid_request", "Request data is invalid.")
        )
        return error_response(status, code, message)

    @app.exception_handler(TranscriptionError)
    async def transcription(_: Request, __: TranscriptionError) -> JSONResponse:
        return error_response(500, "processing_failed", "Audio processing could not be completed.")

    @app.exception_handler(NERPredictionError)
    async def ner_prediction(_: Request, __: NERPredictionError) -> JSONResponse:
        return error_response(500, "processing_failed", "Entity extraction could not be completed.")

    @app.exception_handler(Exception)
    async def unexpected(_: Request, __: Exception) -> JSONResponse:
        return error_response(500, "processing_failed", "Processing could not be completed.")
